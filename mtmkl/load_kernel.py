# In this split we want to use multiple task multiple kernel learning to save the results.
# The computation of the kernels is done a priori, since it is expensive in time. We then proceed by splitting the kernels. This operation is perform by the split function
# We then train the algorithm, based on the split. We also evaluate the performances on the test set

import os
import numpy as np
import pandas as pd
from os.path import join

from mtmkl.utils import generate_index


def load(path, return_kernel_name=False):  # , prc_tr, prc_val, prc_ts):
    """ Here we give the path which contains the kernel for each patient. This folder contains three directories - corr, plv, cross
    First we load the dataset (X, y). The folder structure is such that:
    path
        [patient_ID]
            [kernel, data.pkl]
                [cross, corr, plv]  # in kernel
                    *.csv  # for each folder, scale_xyz.csv
    --------------
    Parameters:
        path, string which refers to the folder with kernels for each patient
    --------------
    Returns:
        X_list, y_list
    """
    ##### path example for kernels

    # /path_to_data/patient_ID/kernel/cross/scale*.csv
    # /path_to_data/patient_ID/kernel/corr/scale*.csv
    # /path_to_data/patient_ID/kernel/plv/scale*.csv

    ##### path example for labels

    # /path_to_data/patient_ID/data.pickle

    if return_kernel_name:
        kernel_name = []

    id_list = sorted([path + f for f in os.listdir(path) if (os.path.isdir(path + f) and "kernel" in os.listdir(path + f))])

    # here we print the path to the folder for each ID
    # each of these paths contains the Y.csv and the folder of kernels

    y = []  # path of to the labels
    kernels = []

    for id in id_list:

        # path for the kernels
        # correlation - 0, 1, 10, 11, .. 19, 2, 20, 21, .. , 82
        # cross correlation
        # plv
        kernel_list_str = sorted([join(path, id, "kernel", k, s) for k in os.listdir(join(path, id, "kernel")) for s in os.listdir(join(path, id, "kernel", k))])

        kernels.append(kernel_list_str)  # list of files for each patient
        y.append(join(id, "data.pkl"))   # file that contains Y label

    X_list, y_list = [], []

    for idx, (kk, yy) in enumerate(zip(kernels, y)):

        # load the dataframe with time series and labels
        labels = pd.DataFrame(pd.read_pickle(yy)["Y"])
        labels.columns = ['Y']

        # generate the list for data and labels
        X_patient, y_patient = [], []

        for f in kk:
            if idx == 0 and return_kernel_name:
                kernel_name.append(join(f.split("/")[-2], f.split("/")[-1].split(".")[0]))

            # load the file - fixed patient - kernel - scale
            scale = pd.read_csv(f, header=0, index_col=0)

            scale = scale.sort_index().loc[:, sorted(scale.columns)] # order

            merging = scale.merge(labels, left_index=True, right_index=True)
            labels_ = merging['Y']  # sorted labels
            merging = merging[merging.index]

            X_patient.append(merging.values)  # [channels, channels, 300]
            y_patient = labels_.values  # labels


        X_list.append(np.array(X_patient))
        y_list.append(y_patient)


    if return_kernel_name:
        X_list, y_list, kernel_name
    return X_list, y_list
