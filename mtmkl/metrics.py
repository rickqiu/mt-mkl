"""Metrics module."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy_multiple(y_true, y_pred, sample_weight=None):
    ba = [balanced_accuracy_score(yt, yp) for yt, yp in zip(y_true, y_pred)]
    return np.mean(ba)
