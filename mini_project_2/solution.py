import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

CHEAP_THRESHOLD   = 100_000
AVERAGE_THRESHOLD = 350_000

CAT_COLS = ["HallwayType", "HeatingType", "AptManageType",
            "TimeToBusStop", "TimeToSubway", "SubwayStation"]

_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return strip_whitespace(df)

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    return df

def assign_house_labels(df: pd.DataFrame) -> np.ndarray:
    ''''Depending on SalePrice assigns label: cheap (0), average (1) or expensive (2)'''
    return np.select(
        [df["SalePrice"] <= CHEAP_THRESHOLD, df["SalePrice"] <= AVERAGE_THRESHOLD],
        [0, 1], default=2
    )

def prepare_train_data(path: str):
    df = load_data(path)
    df[CAT_COLS] = _enc.fit_transform(df[CAT_COLS])
    y = assign_house_labels(df)
    return df.drop(columns=["SalePrice"]), y

def prepare_test_data(path: str) -> pd.DataFrame:
    df = load_data(path)
    df[CAT_COLS] = _enc.transform(df[CAT_COLS])
    return df

@dataclass
class NetConfig:
    layers:       List[int]      = field(default_factory=lambda: [128, 64])
    dropout:      float          = 0.0
    batch_norm:   bool           = False
    sampler:      Optional[type] = None
    class_weight: bool           = False
    epochs:       int            = 100

class NeuralNetwork:
    def __init__(self, config: NetConfig):
        self.config = config

    def fit(self, X, y):
        if self.config.sampler:
            X, y = self.config.sampler(random_state=42).fit_resample(X, y)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.net = self._build_net(X.shape[1])
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self.net.parameters())
        loss_fn = nn.CrossEntropyLoss(weight=self._class_weights(y))
        for _ in range(self.config.epochs):
            opt.zero_grad()
            loss_fn(self.net(Xt), yt).backward()
            opt.step()

    def predict(self, X) -> np.ndarray:
        self.net.eval()
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return self.net(Xt).argmax(1).numpy()

    def _build_net(self, input_dim: int) -> nn.Sequential:
        def block(in_f, out_f):
            return [l for l in [
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f) if self.config.batch_norm else None,
                nn.ReLU(),
                nn.Dropout(self.config.dropout) if self.config.dropout > 0 else None,
            ] if l is not None]
        dims = [input_dim] + self.config.layers
        layers = [l for in_f, out_f in zip(dims, dims[1:]) for l in block(in_f, out_f)]
        layers.append(nn.Linear(self.config.layers[-1], 3))
        return nn.Sequential(*layers)

    def _class_weights(self, y) -> Optional[torch.Tensor]:
        if not self.config.class_weight:
            return None
        return torch.tensor(1.0 / np.bincount(y), dtype=torch.float32)

