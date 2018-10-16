# Given a file, a pandas dataframe we apply the preprocessing. This is done in such a way that the argument of the function is a pandas dataframe

import numpy as np
from scipy.signal import butter, filtfilt, freqs


def remove_edges(X, npoints=10000):
    """ Given a signal X which can be one or two dimensional we remove npoints symmetrically at the edges, in order to eliminate border effects due to filtering operations
    ------------
    Parameters:
        X, electrical signal, numpy array can be one or two dimensional
    ------------
    Returns:
        X[:, npoints:len(X)-npoints]
    """
    flag_multiple = True

    if X.ndim == 2:
        n, T = X.shape

    elif X.ndim == 1:
        T = X.shape[0]
        flag_multiple = False

    else:
        raise ValueError("Wrong input dimension")

    if flag_multiple:
        return X[:, npoints:T-npoints]
    else:
        return X[npoints:T-npoints]


class PreProcessing():
    """ preProcessing is the class used to preprocess the data.
    It includes data cleaning from power line and harmonics.
    """
    def __init__(self, fs, dataframe=None):
        """ Initializer. Sampling frequency is needed for further analysis
        Parameters:
            fs, sampling frequency
        """
        self.fs = fs
        self.nyq = float(fs) / 2



    def remove_powerline(self, X, powerline=50., bandwidth=2.):
        """ Given a signal X, remove_powerline removes the powerline effects.
        This in Europe corresponds to 50 Hz, which is set as default value.
        Filter: Butterworth filter of fourth order.
        This is an IIR filter, in order to linearize the phase we apply
        filtfilt function.
        --------ff = pd.read_pickle(path + f + "/data.pkl")----
        Parameters:
            X, electrical signal, numpy array can be one or two dimensional
            powerline, value in Hz of the powerline frequency
        ------------
        Returns:
            filtered signal
        """

        flag_multiple = True  # if we have more than one time series
        if X.ndim == 2:
            n, T = X.shape    # number of time series times length
        elif X.ndim == 1:
            T = X.shape       # length of the time series
            flag_multiple = False
        else:
            raise ValueError("Wrong input dimensions")

        if powerline < 0. or powerline > self.nyq:
            raise ValueError("Wrong value of powerline frequency")

        if bandwidth < 0:
            raise ValueError("Wrong width for bandstop filter")

        forder = 4            # order of Butterworth filter
        halfband = float(bandwidth) / 2

        bandstop_freqs = powerline * np.arange(1, self.nyq / powerline)  # this array goes from powerline to nyq
        print(bandstop_freqs)

        for f in bandstop_freqs:
            b, a = butter(forder, [(f-halfband)/self.nyq, (f+halfband)/self.nyq], 'bandstop', analog=False)

            if not flag_multiple:
                X = filtfilt(b, a, X)
            else:
                X = np.array([filtfilt(b, a, x) for x in X])

        return X


    def check_no_artifacts(self, X=None, tolerance=100000):
        """ Given a time series this method verifies if the signal contains artifacts - like long periods where the signal is constant and the acquisition went wrong
        ------------
        Parameters:
            X, electrical signal, numpy array can be one or two dimensional
            powerline, value in Hz of the powerline frequency
            tolerance, number of points that are allowed to have constant value. We always cut off the first 10 seconds of acquisition. the first 10 seconds are then allowed, we add other 90 seconds
        ------------
        Returns:
            bool_X, a boolean object, same shape of X, equal to True if the signal has no artifacts, False otherwise
        """

        flag_multiple = True

        if (X is None and self.dataframe is None):
            raise ValueError("Wrong input, X must be given")

        else:
            if X is None:
                X = self.dataframe.values

            if X.ndim == 2:
                n, T = X.shape                # number of time series times length
                bm = np.ones(n, dtype=bool)   # by default all set to True value
            elif X.ndim == 1:
                T = X.shape                   # length of the time series
                flag_multiple = False
            else:
                raise ValueError("Wrong input dimensions")

            if flag_multiple:
                gradient = np.diff(X, axis=-1)             # difference between two time steps for each time series
                sum_diff = np.sum(gradient==0, axis=-1)  # no differences
                bm = np.array([True if sg < tolerance else False
                               for sg in sum_diff])

            else:
                gradient = np.diff(X)             # difference between two time steps
                bm = np.sum(gradient==0) < tolerance


        return bm
