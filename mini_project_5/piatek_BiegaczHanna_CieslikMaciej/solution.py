"""
Load data, inspect the dataset, and prepare training utilities for the model.
"""

import pickle
from collections import Counter
import torch 
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
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
    Each sequence_tensor has shape (sequence_length, input_features). """
    with open(pickle_path, "rb") as pickle_file:
        raw_data = pickle.load(pickle_file)
    return raw_data


def inspect_dataset(data: list[tuple[torch.Tensor, int]]) -> None:
    """Print basic dataset statistics."""
    first_sequence, first_label = data[0]
    input_size = first_sequence.shape[-1]

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
    This class stores data so the DataLoader can load each example. """

    def __init__(self, data: list[tuple[Tensor, int]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sequence, label = self.data[index]
        tensor_sequence = torch.as_tensor(sequence, dtype=torch.float32)
        return tensor_sequence, torch.tensor(label, dtype=torch.long)


def pad_batch(
    batch: list[tuple[Tensor, Tensor]]
) -> tuple[Tensor, Tensor, list[int]]:
    """Pad all sequences to the same length with zeros.
    Returns padded_sequences, labels, original_lengths """
    sequences, labels = zip(*batch)
    original_lengths = [seq.shape[0] for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_sequences, torch.stack(labels), original_lengths

if __name__ == "__main__":

    training_data = load_raw_data(TRAINING_DATA_PATH)
    inspect_dataset(training_data)

    dataset = ChordDataset(training_data)
    loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=pad_batch
    )
