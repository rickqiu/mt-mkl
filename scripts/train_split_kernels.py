# In this split we want to use multiple task multiple kernel learning to save the results.
# The computation of the kernels is done a priori, since it is expensive in time. We then proceed by splitting the kernels. This operation is perform by the split function
# We then train the algorithm, based on the split. We also evaluate the performances on the test set

import os
import numpy as np
import pandas as pd
from multikernel import multi_logistic, load_kernel
from os.path import join


def main():
    X_list, y_list = load_kernel("/home/vanessa/DATA_SEEG/PKL_FILE/")
    print(y_list)


if __name__ == '__main__':
    main()
