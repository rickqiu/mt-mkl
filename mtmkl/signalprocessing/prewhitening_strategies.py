from scipy.stats import norm

def H0zscore(X):
    """
    Normalization of the wavelet coefficients based on the H_0 z score, from Roehri et al.
    'Time-frequency strategies for increasing high frequency oscillation detectability in intracerebral EEG'
    -----------
    Parameters:
        X, np.array matrix or one dimensional array of wavelet coefficients, dtype=np.complex
    -----------
    Returns:
        normalized wavelet coefficients, same shape of input X
    """
    repart = np.real(X)  # real part for wavelet coefficients
    impart = np.imag(X)  # imaginary part

    # 1st and 3rd quantile for real and imag distribution, computed on each channel
    firstre, thirdre = np.percentile(repart, [0.25, 0.75], axis=-1)
    firstim, thirdim = np.percentile(impart, [0.25, 0.75], axis=-1)

    iqrre = thirdre - firstre  # interquartile range for real part
    iqrim = thirdim - firstim
    minre = firstre - 1.5 * iqrre
    maxre = thirdre + 1.5 * iqrre
    minim = firstim - 1.5 * iqrim
    maxim = thirdim + 1.5 * iqrim

    if X.ndim == 2:
        mure_list = []  # create the list for mean value of the real coefs
        muim_list = []
        sigmare_list = []
        sigmaim_list = []

        distre = np.array([re_id[np.logical_and(re_id >= min_id, re_id <= max_id)] for re_id, min_id, max_id in zip(repart, minre, maxre)])
        distim = np.array([im_id[np.logical_and(im_id >= min_id, im_id <= max_id)] for im_id, min_id, max_id in zip(impart, minim, maxim)])

        for dre, dim in distre, distim:
            mure, sigmare = norm.fit(dre)  # fit for each channel
            muim, sigmaim = norm.fit(dim)
            mure_list.append(mure)
            muim_list.append(muim)
            sigmare_list.append(sigmare)
            sigmaim_list.append(sigmaim)

        mure = np.array(mure_list).reshape(-1, 1)  # reshape to use the ufunc
        muim = np.array(muim_list).reshape(-1, 1)
        sigmare = np.array(sigmare_list).reshape(-1, 1)
        sigmaim = np.array(sigmaim_list).reshape(-1, 1)

    else:

        distre = repart[np.logical_and(repart >= minre, repart <= maxre)]
        distim = impart[np.logical_and(impart >= minim, impart <= maxim)]

        # fit gaussian around these two values and extract the mean and stardard deviation
        mure, sigmare = norm.fit(distre)
        muim, sigmaim = norm.fit(distim)

    # standardize the coefficients with respect to the noise distribution
    repart = (repart - mure) / sigmare
    impart = (impart- muim) / sigmaim

    return repart + 1j*impart  # return the prewhitened coefficients
