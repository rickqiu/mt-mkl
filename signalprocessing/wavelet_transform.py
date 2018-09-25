import pywt
import numpy as np


# we want to compute wavelet transform for matrix
class CWTTransform():
    """Class for the computation of wavelet coefficients"""
    def __init__(self, fs, scales=None):
        """ Initializer for the ContinuousWaveletTransform. We must give here
        the sampling frequency of the signal for which we want to compute the
        representation. By default we use the Morlet wavelet.
        The parameters used in the representation are B = 1.5 and C = 1.
        B corresponds to the variance for the wavelet transform or also to the time decay of the wavelet.
        # TODO: include other wavelet - we need to compute the variance in the frequency domain
        ---------------
        Parameters:
            fs, float, sampling frequency
            scales, scales to use for the signal representation
        ---------------
        Attributes:
            fs,
            scales, if None, this covers all the spectrum
        """
        self.B = 1.5  # mother wavelet variance
        self.C = 1.   # central frequency - for dt=1, it is divided for a
        self.fs = fs  # sampling frequency
        self.T = 1. / float(self.fs)  # sampling period
        self.wtype = "cmor1.5-1."
        if np.isscalar(self.fs):
            self.nyq = float(self.fs) / 2
        else:
            raise ValueError("Wrong input value of sampling frequency")
        self.scales = scales

        if self.scales is None:
            # we must cover the spectrum from 0 to Nyquist
            # given that B is the std for the filter, we give 2. as width
            self.fc = np.linspace(2., self.nyq, self.nyq / self.B)[:-1]
            # we need to compute the corresponding scales
            self.scales =  np.array([self.C/(f * self.T) for f in self.fc])
        # verify if wtype is in the family of continuous wavelet


    def cwt(self, X, scale):
        """ Given a signal we return the wavelet coefficients. The input can be of more than one dimensional. Given a scale, in the list created once we initialize the class, we have the corresponding frequency
        ---------------
        Parameters:
            X, numpy array, 1 or 2 dimensional
            scale, scalar, specify the scale
        ---------------
        Returns:
            wavelet coefficient computed through complex morlet wavelet given a specific scale. We call here the cwt method from pywt. The cwt transform in pywt returns a tuple - one of coefficients, one for the constant term, we will keep the first term only. This justify the first index in the following code. The second is due to the fact that the default output for the coefficients is two dimensional, even if the output corresponds to an array of single dimension
        """
        flag_multi = True
        if X.ndim == 2:
            n, p = X.shape
        elif X.ndim == 1:
            p = X.size
            flag_multi = False
        else:
            raise ValueError("Wrong input dimension")

        if flag_multi:
            coefs = np.array([pywt.cwt(x, scale, self.wtype)[0][0] for x in X])
        else:
            coefs = pywt.cwt(X, scale, self.wtype)[0][0]

        return coefs
