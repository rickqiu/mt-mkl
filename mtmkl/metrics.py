"""Add from remote."""
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy_multiple(y_true, y_pred, sample_weight=None):
    ba = [balanced_accuracy_score(yt, yp) for yt, yp in zip(y_true, y_pred)]
    #for p, b in zip(PATIENTS, ba):
    #    logging.info("Balanced accuracy for %s: %.4f" % (p, b))
    return np.mean(ba)
