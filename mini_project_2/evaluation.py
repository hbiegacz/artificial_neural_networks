import numpy as np
import pandas as pd


def calc_accuracy(pred_targets, targets):
    accuracies = []
    for i in range(3):
        class_correct = (pred_targets == targets.values)[targets == i].sum()
        accuracies.append(class_correct / (targets == i).sum())
    return np.mean(accuracies)


predictions_student = pd.read_csv("pred.csv", header=None).iloc[:, 0]
labels = pd.read_csv("UNKNOWN.csv", header=None).iloc[:, 0]

print(calc_accuracy(predictions_student, labels))
