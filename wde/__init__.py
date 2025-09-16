import numpy as np
import pandas as pd
import pywt

def wde(x, wavelet='db4', level=5):
    """
    Computes the wavelet density estimation of a 1D signal.

    Args:
        x (np.ndarray or pd.Series): The input signal.
        wavelet (str): The wavelet to use.
        level (int): The level of decomposition.

    Returns:
        (np.ndarray, np.ndarray): The estimated density and the x-grid.
    """
    if isinstance(x, pd.Series):
        x = x.values

    # Normalize the data
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    # Perform the wavelet decomposition
    coeffs = pywt.wavedec(x, wavelet, level=level)

    # Threshold the coefficients
    threshold = np.sqrt(2 * np.log(len(x)))
    coeffs_thresh = [pywt.threshold(c, threshold, 'soft') for c in coeffs]

    # Reconstruct the signal
    x_recon = pywt.waverec(coeffs_thresh, wavelet)

    # Estimate the density
    # This is a simplified approach. A more rigorous approach would involve
    # the scaling function at the lowest level.
    # For now, we will use a histogram-based approach on the reconstructed signal.
    hist, bin_edges = np.histogram(x_recon, bins='auto', density=True)

    return hist, bin_edges[:-1]
