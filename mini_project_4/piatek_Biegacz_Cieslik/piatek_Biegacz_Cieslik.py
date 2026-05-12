import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "train", "trafic_32"))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "piatek_Biegacz_Cieslik.pt")


@dataclass
class VAEConfig:
    image_size: int = 32
    input_channels: int = 3
    latent_dimension: int = 1024
    num_classes: int = 43          
    class_embedding_dim: int = 64  
    encoder_channels: List[int] = field(default_factory=lambda: [64, 64, 128])
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 60
    kl_weight: float = 0.001

    optimizer_type: str = "Adam"
    weight_decay: float = 0.0

    train_data_path: str = TRAIN_DATA_PATH
    output_path: str = OUTPUT_PATH

    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    num_workers: int = 4

    def feature_map_size(self) -> int:
        """Keeps linear layers valid even when image size or encoder depth changes."""
        return self.image_size // (2 ** len(self.encoder_channels))

    def flattened_size(self) -> int:
        """Derives the latent projection size from the config, so architecture changes stay consistent."""
        return self.encoder_channels[-1] * self.feature_map_size() * self.feature_map_size()


class VariationalAutoencoder(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        self.class_embedding = nn.Embedding(config.num_classes, config.class_embedding_dim)

        encoder_layers = []
        current_channels = config.input_channels
        for output_channels in config.encoder_channels:
            encoder_layers.extend([
                nn.Conv2d(current_channels, output_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            current_channels = output_channels
        self.encoder = nn.Sequential(*encoder_layers)

        encoder_out_size = config.flattened_size() + config.class_embedding_dim
        self.to_mean = nn.Linear(encoder_out_size, config.latent_dimension)
        self.to_log_variance = nn.Linear(encoder_out_size, config.latent_dimension)

        decoder_in_size = config.latent_dimension + config.class_embedding_dim
        self.from_latent = nn.Linear(decoder_in_size, config.flattened_size())

        decoder_channels = list(reversed(config.encoder_channels[:-1])) + [config.input_channels]
        decoder_layers = []
        current_channels = config.encoder_channels[-1]

        for output_channels in decoder_channels[:-1]:
            decoder_layers.extend([
                nn.ConvTranspose2d(current_channels, output_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = output_channels

        decoder_layers.extend([
            nn.ConvTranspose2d(current_channels, decoder_channels[-1], kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, images: torch.Tensor, class_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        features = features.view(images.size(0), -1)
        class_emb = self.class_embedding(class_labels)
        features = torch.cat([features, class_emb], dim=1)
        return self.to_mean(features), self.to_log_variance(features)

    def sample_latent(self, mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        """Bez zmian — reparametryzacja."""
        std = torch.exp(0.5 * log_variance)
        noise = torch.randn_like(std)
        return mean + noise * std

    def decode(self, latent_vectors: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        class_emb = self.class_embedding(class_labels)
        z = torch.cat([latent_vectors, class_emb], dim=1)
        features = self.from_latent(z)
        features = features.view(
            latent_vectors.size(0),
            self.config.encoder_channels[-1],
            self.config.feature_map_size(),
            self.config.feature_map_size(),
        )
        return self.decoder(features)

    def forward(self, images: torch.Tensor, class_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(images, class_labels)
        latent_vectors = self.sample_latent(mean, log_variance)
        reconstructed_images = self.decode(latent_vectors, class_labels)
        return reconstructed_images, mean, log_variance


class VAEExperiment:
    def __init__(self, config: VAEConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VariationalAutoencoder(config).to(self.device)
        self.optimizer = self.create_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-5
        )

    def create_optimizer(self):
        """Keeps optimizer choice configurable without changing training code."""
        if self.config.optimizer_type == "AdamW":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def compute_loss(
        self,
        original_images: torch.Tensor,
        reconstructed_images: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        current_epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1_loss = F.l1_loss(reconstructed_images, original_images)
        mse_loss = F.mse_loss(reconstructed_images, original_images)
        reconstruction_loss = 0.8 * l1_loss + 0.2 * mse_loss

        kl_divergence = -0.5 * torch.mean(
            1 + log_variance - mean.pow(2) - log_variance.exp()
        )

        warmup_epochs = self.config.epochs // 2
        annealed_kl_weight = min(1.0, current_epoch / max(1, warmup_epochs)) * self.config.kl_weight

        total_loss = reconstruction_loss + annealed_kl_weight * kl_divergence

        return total_loss, {
            "total_loss": float(total_loss.detach().cpu()),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "kl_divergence": float(kl_divergence.detach().cpu()),
        }

    def train_epoch(self, data_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics_sum = {"total_loss": 0.0, "reconstruction_loss": 0.0, "kl_divergence": 0.0}

        for images, class_labels in data_loader:           # class_labels już są w DataLoader
            images = images.to(self.device)
            class_labels = class_labels.to(self.device)    # NOWE — przenieś na GPU
            self.optimizer.zero_grad()

            reconstructed_images, mean, log_variance = self.model(images, class_labels)  # NOWE — przekaż etykiety
            loss, metrics = self.compute_loss(images, reconstructed_images, mean, log_variance, epoch)

            loss.backward()
            self.optimizer.step()

            for key in metrics_sum:
                metrics_sum[key] += metrics[key]

        batch_count = len(data_loader)
        return {key: value / batch_count for key, value in metrics_sum.items()}

    def fit(self, data_loader: DataLoader, print_progress: bool = True):
        """Returns the full learning history because trends matter during model selection."""
        history = []

        for epoch in range(self.config.epochs):
            epoch_metrics = self.train_epoch(data_loader, epoch)  # <-- epoch przekazany
            self.scheduler.step()
            history.append(epoch_metrics)

            if print_progress:
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} - "
                    f"loss: {epoch_metrics['total_loss']:.4f}, "
                    f"recon: {epoch_metrics['reconstruction_loss']:.4f}, "
                    f"kl: {epoch_metrics['kl_divergence']:.4f}"
                )

        return history

    def generate_samples(self, number_of_samples: int) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            latent_vectors = torch.randn(number_of_samples, self.config.latent_dimension, device=self.device)
            class_labels = torch.arange(number_of_samples, device=self.device) % self.config.num_classes
            generated_images = self.model.decode(latent_vectors, class_labels)
        return generated_images


def create_transforms(config: VAEConfig):
    """Keeps preprocessing in one place so denormalization later uses exactly the same statistics."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # NOWE
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])


def read_train_dataset(config: VAEConfig):
    """Uses ImageFolder to stay compatible with the dataset structure described in the task."""
    dataset = torchvision.datasets.ImageFolder(
        root=config.train_data_path,
        transform=create_transforms(config),
    )
    return dataset.class_to_idx, dataset


def create_data_loader(dataset, config: VAEConfig, shuffle: bool = True):
    """Collects loading settings in one helper so experiments only change the config."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def denormalize_images(images: torch.Tensor, config: VAEConfig) -> torch.Tensor:
    """Restores pixel values to the natural image range expected for saved generated samples."""
    mean = torch.tensor(config.mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(config.std, device=images.device).view(1, 3, 1, 1)
    return torch.clamp(images * std + mean, 0.0, 1.0)


def save_generated_samples(images: torch.Tensor, output_path: str):
    """Saves samples exactly as a detached CPU tensor, which matches the required submission format."""
    torch.save(images.cpu().detach(), output_path)


def main():
    config = VAEConfig()

    if torch.cuda.is_available():
        print(f"--- Using GPU: {torch.cuda.get_device_name(0)} ---")
    else:
        print("--- Using CPU ---")

    print("Step 1/4 Loading training dataset...")
    _, train_dataset = read_train_dataset(config)
    train_loader = create_data_loader(train_dataset, config, shuffle=True)

    print("Step 2/4 Initializing and training VAE...")
    experiment = VAEExperiment(config)
    experiment.fit(train_loader, print_progress=True)

    print("Step 3/4 Generating 1000 samples...")
    generated_images = experiment.generate_samples(1000)
    generated_images = denormalize_images(generated_images, config)

    print("Step 4/4 Saving generated samples...")
    save_generated_samples(generated_images, config.output_path)

    print(f"--- Finished. Samples saved to: {config.output_path} ---")


if __name__ == "__main__":
    main()