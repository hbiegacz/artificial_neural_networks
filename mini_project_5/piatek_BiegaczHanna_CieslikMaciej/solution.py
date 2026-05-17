"""
Load data, inspect the dataset, and prepare training utilities for the model.
"""

import pickle
from collections import Counter
import torch # pyright: ignore[reportMissingImports]
import torch.nn as nn # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
from torch import Tensor # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler # pyright: ignore[reportMissingImports]
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence # pyright: ignore[reportMissingImports]
from dataclasses import dataclass


TRAINING_DATA_PATH = "../train.pkl"
TESTING_DATA_PATH = "../test.pkl"
PREDICTION_OUTPUT_PATH = "../pred.csv"
COMPOSER_LABELS = {
    0: "bach",
    1: "beethoven",
    2: "debussy",
    3: "scarlatti",
    4: "victoria",
}
NUM_CLASSES = len(COMPOSER_LABELS)


def load_raw_data(pickle_path: str) -> list[tuple[torch.Tensor, int]]:
    """Load training data from a pickle file.
    Returns a list of (sequence_tensor, class_label) tuples.
    Each sequence_tensor has shape (sequence_length, input_features)."""
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
        sequence, label = self.data[index]
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
    bidirectional: bool = False
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


class LSTMModel(nn.Module):
    """Defines the neural network architecture of the LSTM model."""

    def __init__(self, config: NetConfig) -> None:
        super().__init__()

        self._num_directions = 2 if config.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.attention = (
            nn.Sequential(
                nn.Linear(config.hidden_size * self._num_directions, config.hidden_size * self._num_directions),
                nn.Tanh(),
                nn.Linear(config.hidden_size * self._num_directions, 1),
            )
            if config.attention
            else None
        )
        self.batch_norm = (
            nn.BatchNorm1d(config.hidden_size * self._num_directions)
            if config.use_batch_norm
            else None
        )
        self.classifier = nn.Linear(
            config.hidden_size * self._num_directions, NUM_CLASSES
        )

    def forward(self, padded_sequences: Tensor, original_lengths: list[int]) -> Tensor:
        """Compute the model output for a batch of sequences. This method runs the input
        through the LSTM and turns the result into score values for each composer."""

        packed_input = pack_padded_sequence(
            padded_sequences,
            original_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden_state, _) = self.lstm(packed_input)

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

        return self.classifier(self.dropout(last_hidden))


class LSTMTrainer:
    """Manages the training and prediction process of the LSTM model.
    This class initializes the model, runs the training loop, and makes predictions."""

    def __init__(self, config: NetConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LSTMModel | None = None

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

    def fit(
        self, training_data: list, class_weights: torch.Tensor | None = None
    ) -> "LSTMTrainer":
        first_sequence = training_data[0][0]
        self.config.input_size = first_sequence.shape[-1] if first_sequence.ndim == 2 else 1
        self.model = LSTMModel(self.config).to(self.device)
        assert self.model is not None

        sampler = self._build_sampler(training_data)
        loader = DataLoader(
            ChordDataset(training_data),
            batch_size=self.config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=pad_batch,
        )

        effective_weights = class_weights
        if effective_weights is None and self.config.use_class_weights:
            effective_weights = self._compute_class_weights(training_data)

        criterion = nn.CrossEntropyLoss(
            weight=effective_weights.to(self.device) if effective_weights is not None else None
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

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

        self.model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            batch_count = 0

            for padded_sequences, labels, original_lengths in loader:
                padded_sequences, labels = padded_sequences.to(self.device), labels.to(
                    self.device
                )
                optimizer.zero_grad()
                loss = criterion(
                    self.model(padded_sequences, original_lengths), labels
                )
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss)
                batch_count += 1

            if scheduler is not None:
                if self.config.scheduler_type == "plateau":
                    average_loss = epoch_loss / batch_count if batch_count else epoch_loss
                    scheduler.step(average_loss)
                else:
                    scheduler.step()

        return self

    def predict(self, data: list) -> np.ndarray:
        assert self.model is not None, "Call fit() first."
        loader = DataLoader(
            ChordDataset(data),
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=pad_batch,
        )

        all_predictions: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for padded_sequences, _, original_lengths in loader:
                logits = self.model(padded_sequences.to(self.device), original_lengths)
                all_predictions.append(logits.argmax(dim=1).cpu())

        return torch.cat(all_predictions).numpy()


if __name__ == "__main__":

    training_data = load_raw_data(TRAINING_DATA_PATH)
    inspect_dataset(training_data)

    dataset = ChordDataset(training_data)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=pad_batch)
