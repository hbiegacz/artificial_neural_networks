"""
Load data, inspect the dataset, and prepare training utilities for the model.
"""

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch # pyright: ignore[reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from torch import Tensor # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler # pyright: ignore[reportMissingImports]
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence # pyright: ignore[reportMissingImports]


TRAINING_DATA_PATH = "../train.pkl"
TESTING_DATA_PATH = "../test_no_target.pkl"
PREDICTION_OUTPUT_PATH = "pred.csv"
COMPOSER_LABELS = {
    0: "bach",
    1: "beethoven",
    2: "debussy",
    3: "scarlatti",
    4: "victoria",
}
NUM_CLASSES = len(COMPOSER_LABELS)


def load_raw_data(pickle_path: str) -> list[tuple[torch.Tensor, int]]:
    """Load training data from a pickle file."""
    with open(pickle_path, "rb") as pickle_file:
        raw_data = pickle.load(pickle_file)
    return raw_data


def inspect_dataset(data: list[tuple[torch.Tensor, int]]) -> None:
    """Print basic dataset statistics."""
    first_sequence, first_label = data[0]
    input_size = first_sequence.shape[-1] if first_sequence.ndim == 2 else 1

    sequence_lengths = [seq.shape[0] for seq, _ in data]
    labels = [label for _, label in data]
    class_counts = Counter(labels)

    print(f"Samples             : {len(data)}")
    print(f"Input size          : {input_size}  (features per time step)")
    print(f"Min sequence length : {min(sequence_lengths)}")
    print(f"Max sequence length : {max(sequence_lengths)}")
    print(
        f"First sequence      : \n\tshape - {tuple(first_sequence.shape)} \n\tlabel nr - {first_label} \n\tcomposer - {COMPOSER_LABELS[first_label]}"
    )
    print("Composer class counts:")
    for class_id, count in sorted(class_counts.items()):
        print(f"    {class_id} ({COMPOSER_LABELS[class_id]}): {count} samples")


class ChordDataset(Dataset):
    """Make raw examples available for PyTorch training.
    This class stores data so the DataLoader can load each example."""

    def __init__(self, data: list[tuple[Tensor, int]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        data_item = self.data[index]
        if isinstance(data_item, tuple) and len(data_item) == 2:
            sequence, label = data_item
        else:
            sequence, label = data_item, -1
        tensor_sequence = torch.as_tensor(sequence, dtype=torch.float32)
        if tensor_sequence.ndim == 1:
            tensor_sequence = tensor_sequence.unsqueeze(-1)
        return tensor_sequence, torch.tensor(label, dtype=torch.long)


def pad_batch(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, list[int]]:
    """Pad all sequences to the same length with zeros.
    Returns padded_sequences, labels, original_lengths"""
    sequences, labels = zip(*batch)
    original_lengths = [seq.shape[0] for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences, torch.stack(labels), original_lengths


@dataclass
class NetConfig:
    """Model and training settings in one place.

    Fields:
        input_size     — number of features per time step.
        hidden_size    — hidden state size for LSTM.
        num_layers     — number of LSTM layers.
        dropout        — dropout between LSTM layers.
        bidirectional  — use bidirectional LSTM.
        lr             — learning rate for Adam.
        epochs         — number of training epochs.
        batch_size     — training batch size.
        use_class_weights — if True, compute class weights automatically.
        balance_strategy — None, "oversample" or "undersample" sampling.
        use_batch_norm — if True, apply batch normalization before classifier.
        attention      — if True, apply simple attention over LSTM outputs.
        scheduler_type — None, "step", "cosine" or "plateau" learning rate scheduler.
        scheduler_step_size — scheduler period or T_max.
        scheduler_gamma — scheduler decay factor.
        scheduler_patience — patience for plateau scheduler.
    """

    input_size: int = 0
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    use_class_weights: bool = False
    balance_strategy: str | None = None
    use_batch_norm: bool = False
    attention: bool = False
    scheduler_type: str | None = None
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 5
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    early_stopping_min_epochs: int = 1


class LSTM(nn.Module):
    """The class responsible for training the model and making composer predictions."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_directions = 2 if config.bidirectional else 1
        self._built = False

        self.lstm: nn.LSTM | None = None
        self.dropout: nn.Dropout | None = None
        self.attention: nn.Sequential | None = None
        self.batch_norm: nn.BatchNorm1d | None = None
        self.classifier: nn.Linear | None = None

        if config.input_size > 0:
            self._build_network()

    def _build_network(self) -> None:
        self._num_directions = 2 if self.config.bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.attention = (
            nn.Sequential(
                nn.Linear(self.config.hidden_size * self._num_directions, self.config.hidden_size * self._num_directions),
                nn.Tanh(),
                nn.Linear(self.config.hidden_size * self._num_directions, 1),
            )
            if self.config.attention
            else None
        )
        self.batch_norm = (
            nn.BatchNorm1d(self.config.hidden_size * self._num_directions)
            if self.config.use_batch_norm
            else None
        )
        self.classifier = nn.Linear(
            self.config.hidden_size * self._num_directions, NUM_CLASSES
        )
        self._built = True
        self.to(self.device)

    def forward(self, padded_sequences: Tensor, original_lengths: list[int]) -> Tensor:
        if not self._built:
            raise RuntimeError("LSTM network is not built. Call fit() after setting input_size.")

        packed_input = pack_padded_sequence(
            padded_sequences,
            original_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden_state, _) = self.lstm(packed_input)  # type: ignore[attr-defined]

        if self.attention is not None:
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            lengths = torch.tensor(original_lengths, device=output.device)
            attention_scores = self.attention(output).squeeze(-1)
            mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
            attention_weights = torch.softmax(attention_scores, dim=1)
            last_hidden = torch.sum(output * attention_weights.unsqueeze(-1), dim=1)
        else:
            if self._num_directions == 2:
                last_hidden = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
            else:
                last_hidden = hidden_state[-1]

        if self.batch_norm is not None:
            last_hidden = self.batch_norm(last_hidden)

        return self.classifier(self.dropout(last_hidden))  # type: ignore[return-value]

    def _compute_class_counts(self, training_data: list) -> Counter:
        return Counter(label for _, label in training_data)

    def _compute_class_weights(self, training_data: list) -> torch.Tensor:
        class_counts = self._compute_class_counts(training_data)
        total = len(training_data)
        weights = [total / (class_counts[class_id] * NUM_CLASSES) for class_id in range(NUM_CLASSES)]
        return torch.tensor(weights, dtype=torch.float32)

    def _build_sampler(self, training_data: list) -> WeightedRandomSampler | None:
        if self.config.balance_strategy is None:
            return None

        class_counts = self._compute_class_counts(training_data)
        sample_weight = [1.0 / class_counts[label] for _, label in training_data]

        if self.config.balance_strategy == "oversample":
            return WeightedRandomSampler(sample_weight, num_samples=len(training_data), replacement=True)

        if self.config.balance_strategy == "undersample":
            min_count = min(class_counts.values())
            num_samples = min_count * len(class_counts)
            return WeightedRandomSampler(sample_weight, num_samples=num_samples, replacement=False)

        raise ValueError(f"Unknown balance strategy: {self.config.balance_strategy}")

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for padded_sequences, labels, original_lengths in loader:
                padded_sequences, labels = padded_sequences.to(self.device), labels.to(self.device)
                loss = criterion(self(padded_sequences, original_lengths), labels)
                total_loss += float(loss)
                batch_count += 1
        self.train()
        return total_loss / batch_count if batch_count else total_loss

    def fit(
        self,
        training_data: list,
        class_weights: torch.Tensor | None = None,
        validation_data: list | None = None,
        early_stopping: bool | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_delta: float | None = None,
        early_stopping_min_epochs: int | None = None,
    ) -> "LSTM":
        first_sequence = training_data[0][0]
        self.config.input_size = first_sequence.shape[-1] if first_sequence.ndim == 2 else 1
        if not self._built:
            self._build_network()
        else:
            self.to(self.device)

        sampler = self._build_sampler(training_data)
        loader = DataLoader(
            ChordDataset(training_data),
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=pad_batch,
        )

        validation_loader = None
        if validation_data is not None:
            validation_loader = DataLoader(
                ChordDataset(validation_data),
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=pad_batch,
            )

        effective_weights = class_weights
        if effective_weights is None and self.config.use_class_weights:
            effective_weights = self._compute_class_weights(training_data)

        criterion = nn.CrossEntropyLoss(
            weight=effective_weights.to(self.device) if effective_weights is not None else None
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        scheduler = None
        if self.config.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler_step_size,
            )
        elif self.config.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_gamma,
            )

        use_early_stopping = self.config.early_stopping if early_stopping is None else early_stopping
        patience = self.config.early_stopping_patience if early_stopping_patience is None else early_stopping_patience
        min_delta = self.config.early_stopping_delta if early_stopping_delta is None else early_stopping_delta
        min_epochs = self.config.early_stopping_min_epochs if early_stopping_min_epochs is None else early_stopping_min_epochs

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        self.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            batch_count = 0

            for padded_sequences, labels, original_lengths in loader:
                padded_sequences, labels = padded_sequences.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self(padded_sequences, original_lengths), labels)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss)
                batch_count += 1

            average_loss = epoch_loss / batch_count if batch_count else epoch_loss
            monitor_loss = average_loss
            if validation_loader is not None:
                validation_loss = self._evaluate_loss(validation_loader, criterion)
                monitor_loss = validation_loss

            if scheduler is not None:
                if self.config.scheduler_type == "plateau":
                    scheduler.step(monitor_loss)
                else:
                    scheduler.step()

            if use_early_stopping:
                if monitor_loss + min_delta < best_loss:
                    best_loss = monitor_loss
                    best_state = {name: param.detach().cpu().clone() for name, param in self.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epoch + 1 >= min_epochs and epochs_without_improvement >= patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best loss {best_loss:.6f}, "
                        f"no improvement for {epochs_without_improvement} epochs."
                    )
                    break

        if use_early_stopping and best_state is not None:
            self.load_state_dict(best_state)

        return self

    def predict(self, data: list) -> np.ndarray:
        loader = DataLoader(
            ChordDataset(data),
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=pad_batch,
        )

        all_predictions: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for padded_sequences, _, original_lengths in loader:
                logits = self(padded_sequences.to(self.device), original_lengths)
                all_predictions.append(logits.argmax(dim=1).cpu())

        return torch.cat(all_predictions).numpy()

LSTMTrainer = LSTM #necesary to make sure that the notebook code works, since it uses the old class LSTMTrainer


def get_script_dir() -> Path:
    return Path(__file__).resolve().parent


def save_predictions(predictions: np.ndarray, output_path: Path) -> None:
    np.savetxt(output_path, predictions, fmt="%d", delimiter="\n")



if __name__ == "__main__":
    base_dir = get_script_dir()
    training_path = base_dir / TRAINING_DATA_PATH
    testing_path = base_dir / TESTING_DATA_PATH
    output_path = base_dir / PREDICTION_OUTPUT_PATH

    training_data = load_raw_data(str(training_path))
    inspect_dataset(training_data)

    model = LSTM(NetConfig( hidden_size=512, num_layers=2, dropout=0.3, epochs=100, bidirectional=False, batch_size=32, balance_strategy="oversample", 
        attention=False, scheduler_type=None))
    print("Start")
    model.fit(training_data)

    test_data = load_raw_data(str(testing_path))
    save_predictions(model.predict(test_data), output_path)

    print(f"Saved predictions to {output_path}")
