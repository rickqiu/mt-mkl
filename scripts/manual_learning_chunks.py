from __future__ import division

import os
import pickle
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mtmkl.multikernel import model_selection_assessment_timesplit



""" This script is done to use the data from the split in three chuncks. We just infer the best model """


def load_kernel(path, return_kernel_name=False):
    """ Here we give the path which contains the kernel for each patient. This folder contains three directories - corr, plv, cross
    First we load the dataset (X, y). The folder structure is such that:
    path
        [patient_ID]
            [kernel_split]
                [test, train, valid]
                    [cross, corr, plv]  # in kernel
                        *.csv  # for each folder, scale_xyz.csv
    --------------
    Parameters:
        path, string which refers to the folder with kernels for each patient
    --------------
    Returns:
        X_list, y_list
    """
    if return_kernel_name:
        kernel_name_tr = []
        kernel_name_vl = []
        kernel_name_ts = []

    id_list = sorted([join(path,f,'kernel_split') for f in os.listdir(path) if (os.path.isdir(join(path,f)) and 'kernel_split' in os.listdir(join(path,f)))])
    # print(id_list)   # list of paths

    y = []
    kernels_tr = []
    kernels_vl = []
    kernels_ts = []
    # list_folder = ['valid', 'test']

    for id in id_list:
        tmp_out_folder = id.split('/kernel_split')[0]
        kernel_list_str_train = sorted([join(id,'train',k,s) for k in os.listdir(join(id,'train')) for s in os.listdir(join(id,'train',k))])

        kernel_list_str_valid = sorted([join(id,'valid',k,s) for k in os.listdir(join(id,'valid')) for s in os.listdir(join(id,'valid',k))])

        kernel_list_str_test = sorted([join(id,'test',k,s) for k in os.listdir(join(id,'test')) for s in os.listdir(join(id,'test',k))])

        # print(kernel_list_str_train)
        # print(kernel_list_str_valid)

        kernels_tr.append(kernel_list_str_train)  # list of files for each patient
        kernels_vl.append(kernel_list_str_valid)  # list of files for each patient
        kernels_ts.append(kernel_list_str_test)  # list of files for each patient

        # print(tmp_out_folder + "/" + tmp_out_folder.split("/")[-1] + ".csv")

        y.append(tmp_out_folder + "/" + tmp_out_folder.split("/")[-1] + ".csv")   # file that contains Y label

    X_list_tr = []
    X_list_vl = []
    X_list_ts = []

    y_tr = []
    y_vl = []
    y_ts = []

    for idx, (kk_tr, kk_vl, kk_ts, yy) in enumerate(zip(kernels_tr, kernels_vl, kernels_ts, y)):  # for each patient
        labels = pd.read_csv(yy, index_col=0, header=None)

        X_patient_tr, y_patient_tr = [], []
        X_patient_vl, y_patient_vl = [], []
        X_patient_ts, y_patient_ts = [], []

        for f_tr, f_vl, f_ts in zip(kk_tr, kk_vl, kk_ts):
            if idx == 0 and return_kernel_name:
                kernel_name_tr.append(join(f_tr.split("/")[-2], f_tr.split("/")[-1].split(".")[0]))
                kernel_name_vl.append(join(f_vl.split("/")[-2], f_vl.split("/")[-1].split(".")[0]))
                kernel_name_ts.append(join(f_ts.split("/")[-2], f_ts.split("/")[-1].split(".")[0]))

            # load the file - fixed patient - kernel - scale
            scale_tr = pd.read_csv(f_tr, header=0, index_col=0)
            scale_vl = pd.read_csv(f_vl, header=0, index_col=0)
            scale_ts = pd.read_csv(f_ts, header=0, index_col=0)

            # sort by channels name
            scale_tr = scale_tr.sort_index().loc[:, sorted(scale_tr.columns)]
            scale_vl = scale_vl.sort_index().loc[:, sorted(scale_vl.columns)]
            scale_ts = scale_ts.sort_index().loc[:, sorted(scale_ts.columns)]

            # merge all the contributions with the labels column
            merging_tr = scale_tr.merge(labels, left_index=True, right_index=True)
            merging_vl = scale_vl.merge(labels, left_index=True, right_index=True)
            merging_ts = scale_ts.merge(labels, left_index=True, right_index=True)

            labels_tr_ = merging_tr[1]  # sorted labels
            labels_vl_ = merging_vl[1]  # sorted labels
            labels_ts_ = merging_ts[1]  # sorted labels

            merging_tr = merging_tr[merging_tr.index]
            merging_vl = merging_vl[merging_vl.index]
            merging_ts = merging_ts[merging_ts.index]

            X_patient_tr.append(merging_tr.values)  # [channels, channels, 300]
            y_patient_tr = labels_tr_.values  # labels
            X_patient_vl.append(merging_vl.values)  # [channels, channels, 300]
            y_patient_vl = labels_vl_.values  # labels
            X_patient_ts.append(merging_ts.values)  # [channels, channels, 300]
            y_patient_ts = labels_ts_.values  # labels

        X_list_tr.append(np.array(X_patient_tr))
        X_list_vl.append(np.array(X_patient_vl))
        X_list_ts.append(np.array(X_patient_ts))

        y_tr.append(y_patient_tr)
        y_vl.append(y_patient_vl)
        y_ts.append(y_patient_ts)


    if return_kernel_name:
        return [X_list_tr, X_list_vl, X_list_ts], [y_tr, y_vl, y_ts], [kernel_list_str_train, kernel_list_str_valid, kernel_list_str_test]

    else:
        return [X_list_tr, X_list_vl, X_list_ts], [y_tr, y_vl, y_ts]



def main():

    # path = '/home/vanessa/DATA_SEEG/PKL_FILE'
    path = '/home/compbio/networkEEG/dataset_corr_cross_plv'

    XX_list, yy_list, kk_list = load_kernel(path, return_kernel_name=True)
    pickle.dump(kk_list, open('kernel_list_ts.pkl', 'wb'))

    param_grid={'beta': [0.1, 0.4, 0.9],
                'l1_ratio_beta': [0.1, 0.4, 0.9],
                'l1_ratio_lamda': [0.1, 0.4, 0.9],
                'lamda': [0.1, 0.4, 0.9]}

    results = model_selection_assessment_timesplit.learning_procedure(XX_list, yy_list, param_grid)

    with open('results_ts.pkl', 'wb') as f:
       pickle.dump(results, f)



if __name__ == '__main__':
    main()
