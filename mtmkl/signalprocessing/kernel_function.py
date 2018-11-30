import numpy as np
from scipy.signal import correlate
from numpy.linalg import norm
from numpy.fft import fft


def correlation(x, y=None):
    """ Measure of correlation, between [-1, 1],
                        C_ij / (sqrt (C_ii * C_jj))
    We can pass as input variables
    (i) x, as a matrix of recordings of dimensions (# recordings, # length time series)
    (ii) x, y, two one dimensional arrays for which we compute the value

    Parameters:
        x, np.array of dimensions (# recordings, # time points)

        or

        x, np.array of dimension # time points
        y, np.array of dimension # time points
    Returns:
        Pearson product model correlation coefficients
    """

    if (x.ndim == 2 and y is None):
        return np.corrcoef(x)

    if (x.ndim == 1 and y.ndim == 1):
        return np.corrcoef(x, y)

    if (x.ndim == 2 and y.ndim == 2):
        correlation_chuncks = np.zeros((x.shape[0],
                                        y.shape[0]))
        for i, x_ in enumerate(x):
            for j, y_ in zip(range(i+1, x.shape[0]), y[i+1:, :]):
                correlation_chuncks[i, j] = np.corrcoef(x_, y_)[0, 1]
        correlation_chuncks += correlation_chuncks.T + np.identity(x.shape[0])
        return correlation_chuncks


def fourier_corr(x, y=None):
    """ Crosscorrelation computes the cross correlation between two time series. We return a normalized value for the cross correlation.
    We consider the Fourier transform of the signal for this scope
    We can pass as input variables
    (i) x, a matrix of recordings of dimensions (# recordings, # length time series)
    (ii) x, y, two dimensional arrays for which we compute the cross correlation

                        < |F(x)|, |F(y)|> / (||F(x)|| * ||F(y)||)

    Parameters:
        x, np.array of dimensions (# recordings, # time points)
        in this case we compute the cross correlation of all the signals (between all the rows of the x matrix)

        or

        x, np.array of dimension # time points
        y, np.array of dimension # time points

    Returns:
        Average of the cross correlation divided by the cross correlation for each recording
    """

    if x.ndim == 2 and y is None:
        n, _ = x.shape
        kernel = np.zeros((n, n))
        ampx = np.abs(fft(x, axis=-1))
        for idx, ax in enumerate(ampx):
            for id2, ay in enumerate(ampx[idx+1:]):
                idy = id2 + idx + 1
                kernel[idx, idy] = ax.dot(ay) / (norm(ax)*norm(ay))
        kernel += kernel.T + np.identity(n)
        return kernel

    elif x.ndim == 2 and y.ndim == 2:
        # we assume that the two dimensions are equivalent
        n, p = x.shape
        kernel = np.zeros((n, n))
        ampx = np.abs(fft(x, axis=-1))
        ampy = np.abs(fft(y, axis=-1))
        for id1_, ax in enumerate(ampx):
            for id2_, ay in zip(range(id1_+1, n), ampy[id1_+1:]):
                kernel[id1_, id2_] = ax.dot(ay) / (norm(ax) * norm(ay))
        kernel += kernel.T + np.identity(n)
        return kernel

    elif x.ndim == 1 and y.ndim == 1:
        if x.size != y.size:
            raise ValueError("Input values have different shape")
        ax = np.abs(fft(x))
        ay = np.abs(fft(y))
        return ax.dot(ay) / (norm(ax)*norm(ay))

    else:
        if y is None:
            raise ValueError("Wrong input")
        else:
            ax = np.abs(fft(x))
            ay = np.abs(fft(y))
            return ax.dot(ay) / (norm(ax)*norm(ay))


def phaselockingvalue(x, y=None):

    """ Here we compute the phase locking value between two time series. This returns the measure of synchronization between the two time series. As before we can have as input
    (i) x, a matrix of recordings of dimensions (# recordings, # length time series)
    (ii) x, y, two dimensional arrays for which we compute the cross correlation

            PLV(x,y) =  1 / T * sum_{i=1}^T exp[ j * (phi(x_i) - phi(y_i))]
    """

    if x.ndim == 2 and y is None:
        n, p = x.shape
        kernel = np.zeros((n,n))
        phase = np.arctan2(np.imag(x), np.real(x));
        return np.abs(np.exp(1j * phase).dot(np.exp(-1j * phase.T))) / p

    elif y.ndim == 2 and x.ndim == 2:
        n, p = x.shape
        phasex = np.arctan2(np.imag(x), np.real(x));
        phasey = np.arctan2(np.imag(y), np.real(y));
        return np.abs(np.dot(np.exp(1j * phasex), np.exp(-1j * phasey).T)) / p

    elif x.ndim == 1 and y.ndim == 1:
        p = x.size
        if x.size != y.size:
            raise ValueError("Input values have different shape")
        phasex = np.arctan2(np.imag(x), np.real(x))
        phasey = np.arctan2(np.imag(y), np.real(y))
        return np.abs(np.dot(np.exp(1j * phasex), np.exp(-1j * phasey).T)) / p

    else:
        raise ValueError("Input values have different shapes")
