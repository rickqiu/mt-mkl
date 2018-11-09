import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from scipy.stats import uniform
from mtmkl.multikernel.model_selection import _safe_split_multi
from mtmkl.utils import generate_index
from sklearn.metrics import balanced_accuracy_score
from mtmkl.metrics import balanced_accuracy_multiple
from mtmkl.multikernel import multi_logistic
import itertools as it


def generate_params_comb(param_dict, gridsearch=True, ext=None):
    """
    Generation of the hyperparameters grid. Given a method for the parameters selection,
    which can be a grid of fixed values, if gridsearch is true, or ext tuples of random
    values extracted from uniform distribution.
    ---------------
    Parameters:
        param_dict: dictionary containing as keys the hyperparameters, as values
            if gridseach: to each hyperparameter there is a corresponding list of values
            else: to each key there MUST BE a correspondent list of two elements, which
                specify the lower and upper bounds of the distribution
        gridsearch: if True the tuple is created using all the possible combination of
            hyperparameters
        ext: if gridsearch is False, this number specifies the number of tuple we generate
            using random search
    ---------------
    Returns:
        param_combinations: all the possible tuples of hyperparameters
    """

    if gridsearch:
        # list of hyperparameters combinations - grid generation
        param_combinations = list(it.product(*(param_dict[param] for param in sorted(param_dict))))

    else:
        # list of hyperparameters extracted from a uniform distribution
        if ext is None:
            raise ValueError("Extraction value not valid")
        param_combinations = np.array([uniform(lb, ub, ext)
                                        for _, (lb, ub) in sorted(param_dict.items())]).T

    return param_combinations



def learning_procedure(X, y, repetitions, test_size, cvfolds, param_dict, gridsearch=True, ext=None, permutation_test=True):
    """
    This function performs the entire learning procedure.
    The function performs grid search of the parameters
            THE ENTIRE DATASET MUST BE PASSED
    Given the data (X, y), which are lists of length #tasks
    we repeat the learning and testing phase for #repetitions times.
    -------------------
    Parameters:
        X: list of length #tasks which contains all kerneks. Each element
            of the list is a tensor of dimensions #kernels, #samples for task, #samples for task
        y: list of length #tasks which contains the labels for each tasks [labels_task_1, ..., labels_task_#task]
        repetitions: number of repetitions where we evaluate the performances of the method
        test_size: test size dimensions, this is a percentage between (0,1)
        cvfolds: number of folds in the cross validation procedure
        param_dict: dictionary of hyperparameters of the method, to each hyperparameter corresponds a list
        gridsearch: bool, default value True, otherwise random search
        permutation_test: bool, default True, to assess the distance from a model learned from shuffled labels
    -------------------
    Returns:
        results, a dictionary
            keys: beta, l1_ratio_beta, l1_ratio_lamda, lamda, score, coef_, alpha_, (randomize score)

    """

    param_combinations = generate_params_comb(param_dict, gridsearch, ext)

    cv_ts_lr = StratifiedShuffleSplit(test_size=test_size, n_splits=repetitions)
    cv_tr_vl = StratifiedKFold(cvfolds)

    results = dict()
    results["beta"] = []
    results["l1_ratio_beta"] = []
    results["l1_ratio_lamda"] = []
    results["lamda"] = []
    results["estimator"] = []
    results["learn index"] = []
    results["test index"] = []
    results["single score"] = []
    results["multi score"] = []

    if permutation_test:
        results["random single score"] = []
        results["random multi score"] = []

    # this definition is needed because safesplit needs the attribute __pairwise
    # of MultipleLogisticRegressionMultipleKernel() to perform the split for kernels
    multikernel = multi_logistic.MultipleLogisticRegressionMultipleKernel()

    # split between learning and test set - number of repetitions is specified in cv_ts_lr
    for learn, test in generate_index(X, y, cv=cv_ts_lr):
        X_learn, y_learn, X_test, y_test = _safe_split_multi(multikernel, X, y,
            learn, test)

        # for all the possible combinations of hyperparameters we save the mean balanced accuracy
        balanced_accuracy_multiple_hyperparams = []

        # pick a tuple
        for beta_, l1_ratio_beta_, l1_ratio_lamda_, lamda_ in param_combinations:

            # multikernel with fixed parameters
            multikernel = multi_logistic.MultipleLogisticRegressionMultipleKernel(
                gamma=0.1, # the gradient descent step is fixed
                l1_ratio_beta=l1_ratio_beta_,
                l1_ratio_lamda=l1_ratio_lamda_,
                beta=beta_, lamda=lamda_,
                verbose=0, max_iter=200, deep=False)

            balanced_accuracy_multiple_KFold = []  # KFold, with fixed tuple

            for train, val in generate_index(X_learn, y_learn, cv=cv_tr_vl):  # split for K times
                X_train, y_train, X_val, y_val = _safe_split_multi(multikernel, X_learn, y_learn,
                train, val)  # save split

                multikernel.fit(X_train, y_train)  # we fit the model
                # we evaluate the score
                score = balanced_accuracy_multiple(y_val, multikernel.predict(X_val))
                balanced_accuracy_multiple_KFold.append(score)  # we append the score for each fold

            balanced_accuracy_multiple_hyperparams.append(  # we compute the average value across the Kfold
                np.mean(np.array(balanced_accuracy_multiple_KFold)))

        idx_best_perf = np.argmax(balanced_accuracy_multiple_hyperparams)
        beta, l1_ratio_beta, l1_ratio_lamda, lamda = param_combinations[idx_best_perf]

        results["beta"].append(beta)
        results["l1_ratio_beta"].append(l1_ratio_beta)
        results["l1_ratio_lamda"].append(l1_ratio_lamda)
        results["lamda"].append(lamda)

        # REFIT
        multikernel = multi_logistic.MultipleLogisticRegressionMultipleKernel(
                gamma=0.1, # the gradient descent step is fixed
                beta=beta,
                l1_ratio_beta=l1_ratio_beta,
                l1_ratio_lamda=l1_ratio_lamda,
                lamda=lamda,
                verbose=0, max_iter=200, deep=False)
        multikernel.fit(X_learn, y_learn)

        results["estimator"].append(multikernel)
        results["learn index"].append(learn)
        results["test index"].append(test)

        results["multi score"].append(balanced_accuracy_multiple(y_test, multikernel.predict(X_test)))
        results["single score"].append([balanced_accuracy_score(y_ts_, y_pr_)
                                        for y_ts_, y_pr_ in zip(y_test, multikernel.predict(X_test))])

        if permutation_test:
            perm_y_learn = [np.random.permutation(yy) for yy in y_learn]
            multikernel.fit(X_learn, perm_y_learn)
            results["random multi score"].append(balanced_accuracy_multiple(y_test, multikernel.predict(X_test)))
            results["random single score"].append([balanced_accuracy_score(y_ts_, y_pr_)
                                                  for y_ts_, y_pr_ in zip(y_test, multikernel.predict(X_test))])

    return results


