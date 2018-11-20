import sys
import numpy as np
import pickle
from mtmkl.load_kernel import load
from mtmkl.multikernel import model_selection_assessment


def main():

    X_list, y_list, kernel_list = load("/home/compbio/networkEEG/dataset_corr_cross_plv/", return_kernel_name=True)
    pickle.dump(kernel_list, open("kernel_list.pkl", "wb"))

    param_grid={'beta': [0.1, 0.4, 0.9],
                'l1_ratio_beta': [0.1, 0.4, 0.9],
                'l1_ratio_lamda': [0.1, 0.4, 0.9],
                'lamda': [0.1, 0.4, 0.9]}

    repetitions = 50
    test_size = 0.5
    kfold = 3
    split_per_file = 5

    for r in range(4, 4 + repetitions / split_per_file):
        results = model_selection_assessment.learning_procedure(X_list, y_list, split_per_file, test_size, kfold, param_grid)
        print(results)
        with open('/home/compbio/networkEEG/mtmklMLHC_allPatients/experiment_03/split' + str(r) + '.pkl', 'wb') as f:
            pickle.dump(results, f)
        #sys.exit(0)

def dummy_test():

    X_list = []
    y_list = []
    y_el = np.ones(19)
    y_el[::2] = -1

    for i in range(4):
        dim = 15+i
        X_list.append(np.array(np.random.randn(3,dim,dim)))
        y_list.append(y_el[:dim])

    param_grid={'beta': [0.1, 0.4, 0.9],
                'l1_ratio_beta': [0.1, 0.4, 0.9],
                'l1_ratio_lamda': [0.1, 0.4, 0.9],
                'lamda': [0.1, 0.4, 0.9]}
    results = model_selection_assessment.learning_procedure(X_list, y_list, 2, 0.5, 3, param_grid)

    with open('results.pkl', 'wb') as f:
       pickle.dump(results, f)


if __name__ == '__main__':
    main()
