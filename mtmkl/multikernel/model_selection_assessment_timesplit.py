import numpy as np
from sklearn.metrics import balanced_accuracy_score
from mtmkl.multikernel import model_selection_assessment, multi_logistic
from mtmkl.metrics import balanced_accuracy_multiple



def learning_procedure(X_list, y_list, params_dict, gridsearch=True, ext=None, permutation_test=True):

    param_combination = model_selection_assessment.generate_params_comb(params_dict)

    results = dict()
    results["beta"] = []
    results["l1_ratio_beta"] = []
    results["l1_ratio_lamda"] = []
    results["lamda"] = []
    results["estimator"] = []
    results["single score"] = []
    results["multi score"] = []

    if permutation_test:
        results["random beta"] = []
        results["random l1_ratio_beta"] = []
        results["random l1_ratio_lamda"] = []
        results["random lamda"] = []
        results["random estimator"] = []
        results["random single score"] = []
        results["random multi score"] = []

    X_tr, X_vl, X_ts = X_list
    y_tr, y_vl, y_ts = y_list

    # for all the possible combinations of hyperparameters we save the mean balanced accuracy
    balanced_accuracy_multiple_hyperparams = []
    if permutation_test:
        permuted_balanced_accuracy_multiple_hyperparams = []

    # pick a tuple
    for beta_, l1_ratio_beta_, l1_ratio_lamda_, lamda_ in param_combinations:
        cvmkl = multi_logistic.MultipleLogisticRegressionMultipleKernel(
            gamma=0.1, # the gradient descent step is fixed
            l1_ratio_beta=l1_ratio_beta_,
            l1_ratio_lamda=l1_ratio_lamda_,
            beta=beta_, lamda=lamda_,
            verbose=0, max_iter=400, deep=False)

        cvmkl.fit(X_tr, y_tr)
        score = balanced_accuracy_multiple(y_vl, cvmkl.predict(X_vl))
        balanced_accuracy_multiple_hyperparams.append(score)

        if permutation_test:
            perm_y_tr = [np.random.permutation(yy) for yy in y_tr]
            # perm_y_vl = [np.random.permutation(yy) for yy in y_vl]
            cvmkl.fit(X_tr, perm_y_tr)
            perm_score = balanced_accuracy_multiple(perm_y_tr, cvmkl.predict(X_vl))
            permuted_balanced_accuracy_multiple_hyperparams.append(perm_score)

    idx_best_perf = np.argmax(balanced_accuracy_multiple_hyperparams)
    beta, l1_ratio_beta, l1_ratio_lamda, lamda = param_combinations[idx_best_perf]
    results["beta"].append(beta)
    results["l1_ratio_beta"].append(l1_ratio_beta)
    results["l1_ratio_lamda"].append(l1_ratio_lamda)
    results["lamda"].append(lamda)

    # REFIT
    bestmkl = multi_logistic.MultipleLogisticRegressionMultipleKernel(
            gamma=0.1, # the gradient descent step is fixed
            beta=beta,
            l1_ratio_beta=l1_ratio_beta,
            l1_ratio_lamda=l1_ratio_lamda,
            lamda=lamda,
            verbose=0, max_iter=400, deep=False)
    bestmkl.fit(X_tr, y_tr)

    print(bestmkl.alpha_)
    print(bestmkl.intercept_)
    print(bestmkl.coef_)

    ############ estimate over test set ############
    results["estimator"].append(bestmkl)
    results["multi score"].append(balanced_accuracy_multiple(y_ts, bestmkl.predict(X_ts)))
    print(balanced_accuracy_multiple(y_ts, bestmkl.predict(X_ts)))
    results["single score"].append([balanced_accuracy_score(y_ts_, y_pr_)
                                    for y_ts_, y_pr_ in zip(y_ts, bestmkl.predict(X_ts))])
    print([balanced_accuracy_score(y_ts_, y_pr_)
                                    for y_ts_, y_pr_ in zip(y_ts, bestmkl.predict(X_ts))])

    if permutation_test:
        idx_best_perf = np.argmax(permuted_balanced_accuracy_multiple_hyperparams)
        beta, l1_ratio_beta, l1_ratio_lamda, lamda = param_combinations[idx_best_perf]

        results["random beta"] = beta
        results["random l1_ratio_beta"] = l1_ratio_beta
        results["random l1_ratio_lamda"] = l1_ratio_lamda
        results["random lamda"] = lamda
        # REFIT
        r_bestmkl = multi_logistic.MultipleLogisticRegressionMultipleKernel(
                gamma=0.1, # the gradient descent step is fixed
                beta=beta,
                l1_ratio_beta=l1_ratio_beta,
                l1_ratio_lamda=l1_ratio_lamda,
                lamda=lamda,
                verbose=0, max_iter=400, deep=False)
        r_bestmkl.fit(X_tr, perm_y_tr)

        ############ estimate over test set ############
        results["random estimator"].append(r_bestmkl)
        results["random multi score"].append(balanced_accuracy_multiple(perm_y_tr, r_bestmkl.predict(X_ts)))
        print(balanced_accuracy_multiple(perm_y_tr, r_bestmkl.predict(X_ts)))
        results["random single score"].append([balanced_accuracy_score(y_ts_, y_pr_)
                                        for y_ts_, y_pr_ in zip(perm_y_tr, r_bestmkl.predict(X_ts))])
        print([balanced_accuracy_score(y_ts_, y_pr_)
                                        for y_ts_, y_pr_ in zip(perm_y_tr, r_bestmkl.predict(X_ts))])
    return results
