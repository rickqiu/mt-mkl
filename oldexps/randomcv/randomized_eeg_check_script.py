import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import make_scorer
from multikernel.model_selection import cross_validate
import palladio.metrics
from scipy.stats import uniform

import logging
from multikernel import model_selection, linear_model, multi_logistic;
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from os.path import join, isdir
from os import listdir
import pickle



def flatten(lst):
    """Flatten a list."""
    return [y for l in lst for y in flatten(l)] \
        if isinstance(lst, (list, np.ndarray)) else [lst]


def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    """Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.
    Returns
    -------
    balanced_accuracy : float
    See also
    --------
    recall_score, roc_auc_score
    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.
    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

PATIENTS = # list of patients

def balanced_accuracy_multiple(y_true, y_pred, sample_weight=None):
    ba = [palladio.metrics.balanced_accuracy_score(yt, yp) for yt, yp in zip(y_true, y_pred)]
    #for p, b in zip(PATIENTS, ba):
    #    logging.info("Balanced accuracy for %s: %.4f" % (p, b))
    return np.mean(ba)

path = 'similarity_measures_C_PLV/'

patient_dirs = sorted([join(path, p) for p in PATIENTS])

kernels = [sorted([os.path.join(p, k) for k in os.listdir(p) if not k.startswith(p.split('/')[-1])])
                for p in patient_dirs]

y = [join(p, p.split('/')[-1] + '.csv') for p in patient_dirs]

X_list, y_list = [], []
for kk, yy in zip(kernels, y):

    X_patient, y_patient = [], []

    for f in kk:
        labels = pd.read_csv(yy, header=None, index_col=0)
        labels.columns = ['labels']

        scale = pd.read_csv(f, header=0, index_col=0)
        scale = scale.sort_index().loc[:, sorted(scale.columns)] # order
        merging = scale.merge(labels, left_index=True, right_index=True)
        labels = merging['labels']
        merging = merging[merging.index]

        X_patient.append(merging.values)
        y_patient = labels.values

    X_list.append(np.array(X_patient))
    y_list.append(y_patient)


def generate_index(X_list, y_list, cv):
    X_list_transpose = [X.transpose(1, 2, 0) for X in X_list]
    split = [cv.split(X, y) for X, y in zip(X_list_transpose, y_list)]
    n_splits = min(cv.get_n_splits(X, y, None) for X, y in zip(X_list_transpose, y_list))

    for i in range(n_splits):
        yield zip(*[next(s) for s in split])

scores = []


klc = model_selection.MultipleKernelRandomizedSearchCV(
        multi_logistic.MultipleLogisticRegressionMultipleKernel(
        gamma=0.1,
        # the gradient descent step is fixed
        verbose=0, max_iter=200, deep=False),
        # param_grid={'lamda': np.logspace(-2,1,8), 'beta': np.logspace(-2,1,8),'l1_ratio_beta': np.linspace(0.1, 0.95, 4),'l1_ratio_lamda': np.linspace(0.1, 0.95, 4)},
        param_distributions={'lamda': uniform(loc=0.01, scale=1),
                         'beta': uniform(loc=0.01, scale=1),
                         'l1_ratio_beta': uniform(loc=0.2, scale=0.75),
                         'l1_ratio_lamda': uniform(loc=0.2, scale=0.75)},
        scoring=make_scorer(balanced_accuracy_multiple),
        n_jobs=-1, verbose=1,
        cv=3,
        error_score=-1,
        n_iter=120
        )

cv_results = cross_validate(klc, X_list, y_list,
                            cv=StratifiedShuffleSplit(test_size=0.5, n_splits=50))

pickle.dump(cv_results, open("random_search_50split.pkl", "wb"))
