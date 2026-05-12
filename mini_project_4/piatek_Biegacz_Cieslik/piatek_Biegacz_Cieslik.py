import os
import time
import itertools
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable, Dict, Tuple
from torch.utils.data import DataLoader, random_split
from scipy import linalg


# ---------------------------------------------------------------------------
# Ścieżki
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "train", "trafic_32"))
OUTPUT_PATH     = os.path.join(SCRIPT_DIR, "piatek_Biegacz_Cieslik.pt")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
@dataclass
class NetConfig:
    # architektura
    in_channels:    int   = 3
    out_channels:   int   = 3
    channel_sizes:  list  = field(default_factory=lambda: [32, 64, 128])
    kernel_size:    int   = 5
    activation:     str   = "silu"
    use_batchnorm:  bool  = False
    t_emb_dim:      int   = 1

    # trening
    epochs:         int   = 60
    lr:             float = 3e-4
    optimizer_name: str   = "adamw"    # "adam" | "adamw" | "sgd"
    loss_name:      str   = "mse"      # "mse"  | "mae"   | "huber"
    scheduler_name: str   = "cosine"   # "none" | "cosine"| "step"
    grad_clip:      float = 1.0        # 0.0 = wyłączony

    # dane
    batch_size:     int   = 64
    train_split:    float = 0.8
    num_workers:    int   = 2
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std:  Tuple[float, float, float] = (0.5, 0.5, 0.5)
    train_data_path: str  = TRAIN_DATA_PATH

    # zaszumianie
    corrupt_fn: Optional[Callable] = None   # None = wbudowana default_corrupt

    # FID
    eval_fid:            bool = True
    eval_every:          int  = 5
    fid_n_generated:     int  = 512
    fid_steps:           int  = 40
    fid_evaluator_epochs: int = 20
    save_best_by:        str  = "fid"   # "fid" | "loss"

    # wyjście
    save_path:  str  = OUTPUT_PATH
    log_every:  int  = 1
    print_images: bool = False


# ---------------------------------------------------------------------------
# Funkcje aktywacji
# ---------------------------------------------------------------------------
ACTIVATIONS: Dict[str, type] = {
    "silu":  nn.SiLU,
    "relu":  nn.ReLU,
    "gelu":  nn.GELU,
    "mish":  nn.Mish,
    "tanh":  nn.Tanh,
    "leaky": nn.LeakyReLU,
    "elu":   nn.ELU,
}


# ---------------------------------------------------------------------------
# Blok konwolucyjny
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch:        int,
        out_ch:       int,
        kernel_size:  int  = 5,
        activation:   str  = "silu",
        use_batchnorm: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        act_cls = ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Nieznana aktywacja '{activation}'. Dostępne: {list(ACTIVATIONS)}")
        layers.append(act_cls())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Model: ParametricUNet (Flow Matching / denoising)
# ---------------------------------------------------------------------------
class ParametricUNet(nn.Module):
    """
    Parametryczny U-Net z warunkowym embeddingiem czasu t.
    Encoder: conv + maxpool (skip connections).
    Decoder: upsample + concat skip + conv.
    """

    def __init__(self, config: NetConfig):
        super().__init__()
        self.config = config

        n_levels     = len(config.channel_sizes)
        self.channel_sizes = config.channel_sizes
        self.t_emb_dim     = config.t_emb_dim

        # --- Encoder (ścieżka w dół) ---
        down_blocks: List[nn.Module] = []
        prev_ch = config.in_channels
        for ch in config.channel_sizes:
            down_blocks.append(
                ConvBlock(prev_ch, ch, config.kernel_size, config.activation, config.use_batchnorm)
            )
            prev_ch = ch
        self.down_layers = nn.ModuleList(down_blocks)

        # --- Decoder (ścieżka w górę) ---
        reversed_ch  = list(reversed(config.channel_sizes))
        up_blocks: List[nn.Module] = []

        for i in range(n_levels):
            if i == 0:
                in_ch = reversed_ch[0] + config.t_emb_dim  # bottleneck + t_emb
            else:
                in_ch = reversed_ch[i - 1] + reversed_ch[i]  # poprzedni + skip

            out_ch = reversed_ch[i] if i < n_levels - 1 else config.out_channels

            if i == n_levels - 1:
                # ostatnia warstwa — zwykła konwolucja (bez aktivacji)
                up_blocks.append(
                    nn.Conv2d(in_ch, out_ch,
                              kernel_size=config.kernel_size,
                              padding=config.kernel_size // 2)
                )
            else:
                up_blocks.append(
                    ConvBlock(in_ch, out_ch, config.kernel_size, config.activation, config.use_batchnorm)
                )

        self.up_layers = nn.ModuleList(up_blocks)
        self.downscale = nn.MaxPool2d(2)
        self.upscale   = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)  — zaszumiony obraz
        t : (B, 1)         — czas (embedding czasu)
        """
        n_levels = len(self.channel_sizes)
        skip_connections: List[torch.Tensor] = []

        # Encoder
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            if i < n_levels - 1:
                skip_connections.append(x)
                x = self.downscale(x)

        # Wstrzykniecie t jako mapa cech (broadcast po H, W)
        t_map = t.view(t.size(0), self.t_emb_dim, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, t_map], dim=1)

        # Decoder
        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)
            x = layer(x)

        return x


# ---------------------------------------------------------------------------
# Ewaluator (prosty klasyfikator do FID)
# ---------------------------------------------------------------------------
class Evaluator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc_1   = nn.Linear(input_dim, hidden_dim)
        self.fc_2   = nn.Linear(hidden_dim, 50)
        self.fc_out = nn.Linear(50, 43)
        self.act    = nn.LeakyReLU(0.2)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.act(self.fc_1(x))
        x = self.act(self.fc_2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.get_features(x))


# ---------------------------------------------------------------------------
# Helpers: fabryki i funkcje pomocnicze
# ---------------------------------------------------------------------------
def build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Nieznany optimizer: '{name}'. Dostępne: adam, adamw, sgd")


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Nieznana funkcja straty: '{name}'. Dostępne: mse, mae, huber")


def build_scheduler(name: str, optimizer, epochs: int):
    name = name.lower()
    if name == "none":
        return None
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
    raise ValueError(f"Nieznany scheduler: '{name}'. Dostępne: none, cosine, step")


def default_corrupt(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Interpolacja między oryginalem a szumem równomiernym wg t."""
    noise  = torch.rand_like(x)
    amount = t.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount


def create_transforms(config: NetConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def read_dataset(config: NetConfig):
    """Wczytuje dataset z ImageFolder + oblicza mean/std."""
    temp_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    full_dataset = torchvision.datasets.ImageFolder(
        root=config.train_data_path,
        transform=temp_transform,
    )
    return full_dataset


# ---------------------------------------------------------------------------
# Helpers: FID
# ---------------------------------------------------------------------------
def _frechet_distance(dist1: np.ndarray, dist2: np.ndarray, eps: float = 1e-6) -> float:
    mu1, sigma1 = np.mean(dist1, axis=0), np.cov(dist1, rowvar=False)
    mu2, sigma2 = np.mean(dist2, axis=0), np.cov(dist2, rowvar=False)
    mu1, mu2    = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1      = np.atleast_2d(sigma1)
    sigma2      = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset  = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


@torch.no_grad()
def _extract_features(evaluator: Evaluator, loader, device: str) -> np.ndarray:
    evaluator.eval()
    features = []
    for x, _ in loader:
        features.append(evaluator.get_features(x.to(device)).cpu().numpy())
    return np.concatenate(features, axis=0)


def _train_evaluator(
    evaluator:   Evaluator,
    train_loader: DataLoader,
    epochs:      int,
    device:      str,
) -> Evaluator:
    evaluator.to(device).train()
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=1e-3)
    loss_fn   = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss  = loss_fn(evaluator(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  Evaluator [{epoch+1}/{epochs}] loss={total/len(train_loader):.4f}")
    return evaluator.eval()


@torch.no_grad()
def sample_images(
    net:       nn.Module,
    n_images:  int,
    img_shape: Tuple[int, int, int],
    steps:     int = 40,
    device:    str = "cpu",
) -> torch.Tensor:
    """Flow-matching sampling: interpolacja od szumu do obrazu."""
    net.eval()
    C, H, W  = img_shape
    x        = torch.rand(n_images, C, H, W, device=device)
    schedule = torch.linspace(1.0, 0.0, steps, device=device)
    for i, t_val in enumerate(schedule):
        t    = torch.full((n_images,), t_val, device=device)
        pred = net(x, t.unsqueeze(1)).clamp(0, 1)
        if i < steps - 1:
            t_next = schedule[i + 1]
            x      = pred * (1 - t_next) + torch.rand_like(x) * t_next
        else:
            x = pred
    return x


def _evaluate_fid(
    net:         nn.Module,
    evaluator:   Evaluator,
    test_loader: DataLoader,
    config:      NetConfig,
    device:      str,
) -> float:
    img_shape = (config.in_channels, 32, 32)
    real_feat = _extract_features(evaluator, test_loader, device)
    generated = sample_images(net, config.fid_n_generated, img_shape,
                               steps=config.fid_steps, device=device)
    gen_ds     = torch.utils.data.TensorDataset(generated, torch.zeros(config.fid_n_generated))
    gen_loader = DataLoader(gen_ds, batch_size=64)
    gen_feat   = _extract_features(evaluator, gen_loader, device)
    return _frechet_distance(real_feat, gen_feat)


# ---------------------------------------------------------------------------
# Klasa eksperymentu
# ---------------------------------------------------------------------------
class DiffusionExperiment:
    def __init__(self, config: NetConfig):
        self.config    = config
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = ParametricUNet(config).to(self.device)
        self.optimizer = build_optimizer(config.optimizer_name, self.model.parameters(), config.lr)
        self.scheduler = build_scheduler(config.scheduler_name, self.optimizer, config.epochs)
        self.loss_fn   = build_loss(config.loss_name)
        self.corrupt_fn = config.corrupt_fn if config.corrupt_fn is not None else default_corrupt

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Urządzenie: {self.device} | Parametry modelu: {total_params:,}")

    # ------------------------------------------------------------------
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(self.device)
            t = torch.rand(x.size(0), device=self.device)

            noisy_x = self.corrupt_fn(x, t)
            pred    = self.model(noisy_x, t.unsqueeze(1))
            loss    = self.loss_fn(pred, x)

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(self.device)
            t = torch.rand(x.size(0), device=self.device)
            noisy_x = self.corrupt_fn(x, t)
            pred    = self.model(noisy_x, t.unsqueeze(1))
            total_loss += self.loss_fn(pred, x).item()
        return total_loss / len(loader)

    # ------------------------------------------------------------------
    def fit(self, full_dataset) -> List[Dict]:
        """Trenuje model, opcjonalnie liczy FID. Zwraca historię treningu."""
        cfg    = self.config
        device = str(self.device)

        train_size = int(cfg.train_split * len(full_dataset))
        test_size  = len(full_dataset) - train_size
        train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True,  num_workers=cfg.num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=cfg.num_workers)
        print(f"Dane: {len(train_ds)} treningowe | {len(test_ds)} testowe")

        # Opcjonalny ewaluator do FID
        evaluator = None
        if cfg.eval_fid:
            if cfg.save_best_by == "fid" and cfg.eval_every > cfg.epochs:
                raise ValueError("eval_every > epochs — FID nigdy nie zostanie policzony")
            print(f"
[FID] Trening Evaluatora ({cfg.fid_evaluator_epochs} epok)...")
            input_dim = cfg.in_channels * 32 * 32
            evaluator = Evaluator(input_dim=input_dim).to(self.device)
            evaluator = _train_evaluator(evaluator, train_loader,
                                         cfg.fid_evaluator_epochs, device)
            print("[FID] Evaluator gotowy.
")

        best_loss = float("inf")
        best_fid  = float("inf")
        history: List[Dict] = []

        for epoch in range(cfg.epochs):
            t0         = time.time()
            train_loss = self._train_epoch(train_loader)
            test_loss  = self._eval_epoch(test_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            # --- FID ---
            fid = None
            should_eval_fid = (
                cfg.eval_fid
                and evaluator is not None
                and ((epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1)
            )
            if should_eval_fid:
                print(f"[FID] Obliczanie FID (epoka {epoch+1})...")
                fid = _evaluate_fid(self.model, evaluator, test_loader, cfg, device)

            # --- Zapis najlepszego modelu ---
            if cfg.save_best_by == "fid":
                if fid is not None and fid < best_fid:
                    best_fid = fid
                    torch.save(self.model.state_dict(), cfg.save_path)
            else:
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(self.model.state_dict(), cfg.save_path)

            row = {"epoch": epoch + 1, "train_loss": train_loss,
                   "test_loss": test_loss, "fid": fid}
            history.append(row)

            if (epoch + 1) % cfg.log_every == 0:
                elapsed = time.time() - t0
                lr_now  = self.optimizer.param_groups[0]["lr"]
                fid_str = f"  fid={fid:.2f}" if fid is not None else ""
                print(
                    f"Epoch [{epoch+1:>3}/{cfg.epochs}]  "
                    f"train={train_loss:.4f}  test={test_loss:.4f}"
                    f"{fid_str}  lr={lr_now:.2e}  ({elapsed:.1f}s)"
                )

            if cfg.print_images and (epoch + 1) % 10 == 0:
                self._preview(epoch + 1, (cfg.in_channels, 32, 32))

        return history

    # ------------------------------------------------------------------
    def _preview(self, epoch: int, img_shape: Tuple[int, int, int]):
        import matplotlib.pyplot as plt
        imgs = sample_images(self.model, 16, img_shape, steps=40, device=str(self.device))
        grid = torchvision.utils.make_grid(imgs.cpu(), nrow=4, normalize=True)
        plt.figure(figsize=(5, 5))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.title(f"Podgląd — epoka {epoch}")
        plt.axis("off")
        plt.show()

    # ------------------------------------------------------------------
    def generate_samples(self, n_samples: int, steps: int = 40) -> torch.Tensor:
        """Generuje n_samples obrazów metodą flow-matching."""
        return sample_images(
            self.model, n_samples,
            (self.config.in_channels, 32, 32),
            steps=steps, device=str(self.device),
        )


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
GRID = {
    "channel_sizes":  [[32, 64], [32, 64, 128], [64, 128, 256]],
    "activation":     ["silu", "gelu"],
    "use_batchnorm":  [False, True],
    "lr":             [1e-3, 3e-4],
    "optimizer_name": ["adam", "adamw"],
}

BASE_CONFIG = dict(
    in_channels=3, out_channels=3, kernel_size=5,
    loss_name="mse", scheduler_name="cosine", grad_clip=1.0,
    epochs=5, batch_size=64, train_split=0.8, num_workers=2,
    eval_fid=True, eval_every=5, fid_n_generated=64,
    fid_steps=10, fid_evaluator_epochs=20,
    save_best_by="fid", log_every=5,
)

RESULTS_DIR = Path("grid_search_results")


def run_grid_search(full_dataset) -> List[Dict]:
    RESULTS_DIR.mkdir(exist_ok=True)
    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    print(f"Grid search: {len(combos)} kombinacji | parametry: {keys}\n")

    results = []
    for idx, combo in enumerate(combos, start=1):
        params   = dict(zip(keys, combo))
        run_name = _make_run_name(idx, params)
        print(f"\n[{idx}/{len(combos)}] {run_name}  |  {params}")

        cfg = NetConfig(**{**BASE_CONFIG, **params,
                           "save_path": str(RESULTS_DIR / f"{run_name}.pt")})
        t0 = time.time()
        try:
            experiment = DiffusionExperiment(cfg)
            history    = experiment.fit(full_dataset)
            elapsed    = time.time() - t0

            fid_vals  = [r["fid"] for r in history if r["fid"] is not None]
            best_fid  = min(fid_vals) if fid_vals else float("inf")
            best_loss = min(r["test_loss"] for r in history)
            results.append({
                "run_name":       run_name,
                "params":         params,
                "best_fid":       best_fid,
                "best_fid_epoch": next((r["epoch"] for r in history if r["fid"] == best_fid), None),
                "best_loss":      best_loss,
                "elapsed_min":    elapsed / 60,
                "checkpoint":     cfg.save_path,
                "history":        history,
            })
            print(f"  FID={best_fid:.4f}  loss={best_loss:.4f}  ({elapsed/60:.1f} min)")
        except Exception as e:
            results.append({"run_name": run_name, "error": str(e)})
            print(f"  BŁĄD: {e}")

    results.sort(key=lambda r: r.get("best_fid", float("inf")))
    _save_results(results, RESULTS_DIR / "results.json")
    _print_ranking(results)
    return results


def load_best_model(results: List[Dict]) -> nn.Module:
    best = results[0]
    print(f"Najlepszy model: {best['run_name']}  FID={best['best_fid']}")
    cfg = NetConfig(**{**BASE_CONFIG, **best["params"]})
    net = ParametricUNet(cfg)
    net.load_state_dict(torch.load(best["checkpoint"], map_location="cpu"))
    return net.eval()


def _make_run_name(idx: int, params: dict) -> str:
    short = {"channel_sizes": "ch", "activation": "act",
             "use_batchnorm": "bn", "lr": "lr", "optimizer_name": "opt"}
    parts = [f"run{idx:03d}"] + [
        f"{short.get(k, k)}={str(v).replace(' ', '').replace('[', '').replace(']', '').replace(',', '-')}"
        for k, v in params.items()
    ]
    return "_".join(parts)


def _save_results(results: List[Dict], path: Path):
    serializable = [{k: v for k, v in r.items() if k != "history"} for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def _print_ranking(results: List[Dict]):
    print("\n" + "─" * 70)
    print(f"  RANKING GRID SEARCH")
    print("─" * 70)
    for i, r in enumerate(results[:10], start=1):
        if "error" in r:
            print(f"  {i:>2}. {r['run_name']}  BŁĄD: {r['error']}")
        else:
            print(f"  {i:>2}. FID={r['best_fid']:>8.4f}  loss={r['best_loss']:.4f}  {r['run_name']}")
    print("─" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    full_dataset = read_dataset(NetConfig())

    # --- Grid search ---
    results    = run_grid_search(full_dataset)
    best_model = load_best_model(results)

    # --- Pełny trening na zwycięskiej konfiguracji ---
    final_config = NetConfig(
        channel_sizes=[32, 64, 128],
        activation="gelu",
        use_batchnorm=False,
        epochs=100,
        lr=3e-4,
        optimizer_name="adamw",
        loss_name="mse",
        scheduler_name="cosine",
        grad_clip=1.0,
        eval_fid=True,
        eval_every=5,
        fid_n_generated=512,
        fid_steps=40,
        fid_evaluator_epochs=20,
        save_best_by="fid",
        save_path=OUTPUT_PATH,
        log_every=1,
        print_images=True,
    )

    print("\n--- Trening finalny (100 epok) ---")
    experiment = DiffusionExperiment(final_config)
    history    = experiment.fit(full_dataset)
    print("--- Zakończono! ---")