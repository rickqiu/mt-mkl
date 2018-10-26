import pywt
import numpy as np


# we compute the continuous wavelet transform. We want the representation to be redundant. This implies that we want overlapped filters in the frequency domain. In order to get this, we made some calculus for finding the meaning of B and C. We have seen that sigma for a filter is equivalent to (sqrt(2)/B) * (1/a)


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
        self.B = 1.   # mother wavelet variance
        self.C = 1.   # central frequency - for dt=1, it is divided for a
        self.fs = fs  # sampling frequency
        self.T = 1. / float(self.fs)  # sampling period
        self.wtype = "cmor1.-1."
        if np.isscalar(self.fs):
            self.nyq = float(self.fs) / 2
        else:
            raise ValueError("Wrong input value of sampling frequency")
        self.scales = scales

        tmp_scale = 2.1  # max value of scale fs / max_scale = central freq

        # this parameter is such that we do not take central frequencies which are at sigma distance one from the next, but the central frequency taken in such a way that we consider only points which are percentage_height **(-1) to the value
        percentage_height = 20. / 19
        tmp_central_f = self.fs / tmp_scale

        # we cover the spectrum from 1 Hz to Nyquist    
        min_freq = 1.
        if self.scales is None:
            scales = []
            freqs = []
            filter_width = []

            while(tmp_central_f > min_freq):
                scales.append(tmp_scale)
                freqs.append(tmp_central_f)
                sigma_filter = 1 / (np.sqrt(2) * np.pi * self.T * tmp_scale)

                tol_width = sigma_filter * np.sqrt(2 * np.log(percentage_height))

                tmp_central_f -= tol_width
                tmp_scale = self.fs / tmp_central_f

                filter_width.append(sigma_filter)

        self.scales = np.array(scales)
        self.freqs = np.array(freqs)
        self.filter_width = np.array(filter_width)
        self.nscales = len(self.scales)


    def cwt(self, X, scale_idx):
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
            coefs = np.array([pywt.cwt(x, self.scales[scale_idx], self.wtype)[0][0] for x in X])
        else:
            coefs = pywt.cwt(X, self.scales[scale_idx], self.wtype)[0][0]

        return coefs
