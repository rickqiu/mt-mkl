# Here we perform the classification of the time series - the extraction of meaningful features using multiple task multiple kernel learning

import os
import numpy as np
import pandas as pd
from os.path import join
from mtmkl.load_kernel import load
from mtmkl.multikernel import model_selection, multi_logistic, cross_validate


def main():

    X_list, y_list = load("/home/vanessa/DATA_SEEG/PKL_FILE/")
    print(y_list)

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

    pickle.dump(cv_results, open("results_learning.pkl", "wb"))



if __name__ == '__main__':
    main()
