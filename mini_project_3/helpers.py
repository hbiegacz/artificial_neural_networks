import numpy as np


def calc_accuracy(predictions: np.ndarray, targets: np.ndarray, n_classes=50):
    assert len(predictions) == len(targets)
    accuracies = []
    for i in range(n_classes):
        accuracies.append((predictions[targets == i] == i).sum() / (targets == i).sum())
    return np.mean(accuracies)
