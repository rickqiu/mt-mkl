import os
import numpy as np
import pandas as pd
from os.path import join

from mtmkl.utils import generate_index


def load_and_reduce_correlation(path, return_kernel_name=False):
    """ Here we give the path which contains the kernel for each patient. See the documentation for the load function. We preprocess the data using a bipolar montage. This is such that, the activity acquired from one sensor is shared among two different samples. To reduce this effect we remove this contacts, in such a way that, given a sequence of channels as
    [B01-B02, B02-B03, B03-B04, B04-B05, ...] the new sample becomes
    [B01-B02, B03-B04, ...]. This has several consequences (i) reduction of correlation (ii) reduction of the training samples (iii) different balancedness between pathological and physiological channels. For this reason we will return the amount of positive y labels for each patient.
    --------------
    Parameters:
        path, string which refers to the folder with kernels for each patient
    --------------
    Returns:
        X_list, y_list, percentage of positive labels
    """

    if return_kernel_name:
        kernel_name = []

    id_list = sorted([path + f for f in os.listdir(path) if (os.path.isdir(path + f) and "kernel" in os.listdir(path + f))])
    # here we print the path to the folder for each ID
    # each of these paths contains the Y.csv and the folder of kernels

    y = []  # path of to the labels
    kernels = []

    for id in id_list:
        kernel_list_str = sorted([join(path, id, "kernel", k, s) for k in os.listdir(join(path, id, "kernel")) for s in os.listdir(join(path, id, "kernel", k))])

        kernels.append(kernel_list_str)  # list of files for each patient
        y.append(id + "/" + id.split("/")[-1] + ".csv")   # file that contains Y label

    X_list, y_list, proportion = [], [], []

    for idx, (kk, yy) in enumerate(zip(kernels, y)):

        # load the y dataframe
        labels = pd.read_csv(yy, index_col=0, header=None)
        all_channels = labels.index
        split_channels = ([ll.split("-") for ll in all_channels])

        ch = []  # channels name
        x = []   # idx position of the uncorrelated channels
        for i_, y_ in enumerate(split_channels):
            tmp = True
            for y__ in y_:
                tmp *=  (y__ not in ch)
                if tmp:
                    ch.append(y__)
            if tmp:
                x.append(i_)

        labels = labels.loc[all_channels[x]]

        prop_epileptic = np.sum([(y_ == 1) for y_ in labels.values]) / float(len(labels.values))

        # generate a list for each patient
        X_patient, y_patient = [], []

        for f in kk:
            if idx == 0 and return_kernel_name:
                kernel_name.append(join(f.split("/")[-2], f.split("/")[-1].split(".")[0]))

            # load the file - fixed patient - kernel - scale
            scale = pd.read_csv(f, header=0, index_col=0)

            scale = scale.sort_index().loc[:, sorted(scale.columns)] # order

            merging = scale.merge(labels, left_index=True, right_index=True)

            labels_ = merging[1]  # sorted labels

            merging = merging[merging.index]

            X_patient.append(merging.values)  # [channels, channels, 300]
            y_patient = labels_.values  # labels

        X_list.append(np.array(X_patient))
        y_list.append(y_patient)
        proportion.append(prop_epileptic)

    if return_kernel_name:
        return X_list, y_list, proportion, kernel_name

    else:
        return X_list, y_list, proportion
