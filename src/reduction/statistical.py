"""Statistical dimensionality reduction methods."""
import numpy as np
import pywt


def PAA_reduce(series: np.ndarray, w: int) -> np.ndarray:
    """
    Piecewise Aggregate Approximation (PAA).

    Divides the series into w equal-length segments and replaces each with
    its mean.
    """
    series = np.asarray(series)
    if len(series) == 0:
        return np.full(w, np.nan)
    # Ensure idx covers the full range of the original series
    idx = np.floor(np.linspace(0, len(series), w + 1, endpoint=True)).astype(int)
    return np.array([np.mean(series[idx[i]:idx[i+1]]) for i in range(w)])


def DFT_reduce(series: np.ndarray, w: int) -> np.ndarray:
    """
    Discrete Fourier Transform (DFT) reduction.

    Keeps the first k positive and k negative frequency components,
    reconstructs via IFFT, then subsamples to length w.
    """
    N = len(series)
    if N == 0:
        return np.zeros(w)
    X = np.fft.fft(series)
    k = w // 2
    X_reduced = np.zeros(N, dtype=complex)
    X_reduced[:k]  = X[:k]
    X_reduced[-k:] = X[-k:]
    approx = np.fft.ifft(X_reduced).real
    return approx[np.linspace(0, N - 1, w).astype(int)]


def DWT_reduce(series: np.ndarray, w: int, wavelet: str = "db6", level: int = 3) -> np.ndarray:
    """
    Discrete Wavelet Transform (DWT) reduction.

    Keeps the approximation coefficients at the given decomposition level,
    then subsamples to length w.
    """
    if len(series) == 0:
        return np.zeros(w)
    
    # The level of decomposition must be less than or equal to the log of the length of the signal
    max_level = pywt.dwt_max_level(len(series), wavelet)
    if max_level == 0: # If the series is too short for any decomposition
        return PAA_reduce(series, w)
        
    level = min(level, max_level)
    
    coeffs = pywt.wavedec(series, wavelet, level=level)
    cA = coeffs[0]
    
    if len(cA) == 0:
        return np.zeros(w)
        
    return cA[np.linspace(0, len(cA) - 1, w).astype(int)]


def SVD_reduce(series: np.ndarray, w: int, window: int = 10) -> np.ndarray:
    """
    SVD-based reduction via Hankel embedding.

    Embeds the series into a sliding-window matrix, applies SVD keeping the
    top w singular values, reconstructs, and subsamples to length w.
    """
    N = len(series)
    if N < window:
        # If series is shorter than window, use PAA as a fallback
        return PAA_reduce(series, w)
        
    X = np.array([series[i : i + window] for i in range(N - window + 1)])
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_reduced = np.zeros_like(S)
    S_reduced[: min(w, len(S))] = S[: min(w, len(S))]
    collapsed = np.mean((U * S_reduced) @ VT, axis=1)
    
    if len(collapsed) == 0:
        return np.zeros(w)
        
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]
