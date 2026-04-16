"""Manifold and matrix decomposition methods for dimensionality reduction."""
import warnings
import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap


def _sliding_window(series: np.ndarray, window: int) -> np.ndarray:
    """Create a sliding window (Hankel) matrix from a time series."""
    N = len(series)
    if N < window:
        raise ValueError("window must be smaller than series length.")
    return np.array([series[i : i + window] for i in range(N - window + 1)])


def PCA_reduce(series: np.ndarray, w: int, window: int = 10) -> np.ndarray:
    """PCA reduction via sliding-window (Hankel) embedding."""
    if len(series) < window:
        # Fallback for short series
        from .statistical import PAA_reduce
        return PAA_reduce(series, w)

    X = _sliding_window(series, window)
    # n_components cannot be larger than n_features
    n_components = min(w, X.shape[1])
    if n_components == 0:
        return np.zeros(w)
        
    X_pca = PCA(n_components=n_components).fit_transform(X)
    collapsed = np.mean(X_pca, axis=1)
    
    if len(collapsed) == 0:
        return np.zeros(w)
        
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]


def KPCA_reduce(
    series: np.ndarray,
    w: int,
    window: int = 10,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> np.ndarray:
    """Kernel PCA reduction via sliding-window embedding."""
    if len(series) < window:
        from .statistical import PAA_reduce
        return PAA_reduce(series, w)

    X = _sliding_window(series, window)
    n_components = min(w, X.shape[1])
    if n_components == 0:
        return np.zeros(w)

    X_kpca = KernelPCA(
        n_components=n_components, kernel=kernel, gamma=gamma
    ).fit_transform(X)
    collapsed = np.mean(X_kpca, axis=1)

    if len(collapsed) == 0:
        return np.zeros(w)

    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]


def Isomap_reduce(
    series: np.ndarray,
    w: int,
    window: int = 10,
    n_neighbors: int | None = None,
) -> np.ndarray:
    """Isomap reduction via sliding-window embedding."""
    if len(series) < window:
        from .statistical import PAA_reduce
        return PAA_reduce(series, w)

    X = _sliding_window(series, window)
    if n_neighbors is None:
        n_neighbors = max(5, int(np.sqrt(X.shape[0])))
    # n_neighbors must be smaller than n_samples
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    
    if n_neighbors == 0: # Not enough samples for Isomap
        return PCA_reduce(series, w, window)

    n_components = min(w, X.shape[1])
    if n_components == 0:
        return np.zeros(w)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                module="sklearn.manifold._isomap")
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning,
                                module="scipy.sparse")
        X_iso = Isomap(
            n_neighbors=n_neighbors, n_components=n_components
        ).fit_transform(X)

    collapsed = np.mean(X_iso, axis=1)
    
    if len(collapsed) == 0:
        return np.zeros(w)
        
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]
