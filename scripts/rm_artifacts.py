import os
from os.path import join
import numpy as np
import pandas as pd
from mtmkl.utils import flatten


def main():

    path = "/home/vanessa/DATA_SEEG/PKL_FILE/"
    arts_filename = "list_artifacts.csv"
    kernelpath = "kernel"

    folderlist = [f for f in os.listdir(path) if arts_filename in   os.listdir(join(path, f))]

    for foldername in folderlist:

        ch_arts_path = join(path, foldername, arts_filename)
        ch_arts = set(list(pd.read_csv(ch_arts_path, header=None, index_col=None)[0]))

        listkerneldir = os.listdir(join(path, foldername, kernelpath))

        for kerneldir in listkerneldir:

            listfile = os.listdir(join(path, foldername, kernelpath, kerneldir))

            for f in listfile:

                df = pd.read_csv(join(path, foldername, kernelpath, kerneldir, f), index_col=0, header=0)

                good_indexes = list(set(df.index) - ch_arts)
                df = df.loc[good_indexes, good_indexes]
                df.to_csv(join(path, foldername, kernelpath, kerneldir, f))


if __name__ == '__main__':
    main()
