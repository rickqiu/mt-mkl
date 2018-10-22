import logging
import os
import pickle
from os import listdir
from os.path import isdir, join

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.utils.metaestimators import _safe_split

from mtmkl.multikernel import linear_model, model_selection, multi_logistic
from mtmkl.multikernel.model_selection import cross_validate
from mtmkl.utils import flatten, generate_index

PATIENTS = [] # list of patients

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
