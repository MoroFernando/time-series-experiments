"""
Neighborhood preservation metrics for dimensionality reduction evaluation.

Both functions operate on flattened 2-D arrays (n_samples, n_features).
"""

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def precision_at_k(X_orig: np.ndarray, X_reduced: np.ndarray, k: int = 5) -> float:
    """
    Mean k-nearest-neighbor precision between original and reduced spaces.

    For each sample, computes the fraction of its k nearest neighbors in the
    original space that are also among its k nearest neighbors in the reduced
    space, then averages over all samples.

    Parameters
    ----------
    X_orig    : (n_samples, n_features_orig)
    X_reduced : (n_samples, n_features_reduced)
    k         : neighborhood size

    Returns
    -------
    float in [0, 1]
    """
    n = X_orig.shape[0]
    k = min(k, n - 1)

    _, idx_orig = NearestNeighbors(n_neighbors=k + 1).fit(X_orig).kneighbors(X_orig)
    _, idx_red  = NearestNeighbors(n_neighbors=k + 1).fit(X_reduced).kneighbors(X_reduced)

    scores = [
        len(set(idx_orig[i, 1:k+1]) & set(idx_red[i, 1:k+1])) / k
        for i in range(n)
    ]
    return float(np.mean(scores))


def compute_trustworthiness(X_orig: np.ndarray, X_reduced: np.ndarray, k: int = 5) -> float:
    """
    Trustworthiness of the reduced representation w.r.t. the original space.

    Measures how well the local structure is preserved: penalises points that
    appear as neighbors in the reduced space but were far apart in the original.

    Parameters
    ----------
    X_orig    : (n_samples, n_features_orig)
    X_reduced : (n_samples, n_features_reduced)
    k         : neighborhood size

    Returns
    -------
    float in [0, 1]
    """
    k = min(k, X_orig.shape[0] - 1)
    return float(trustworthiness(X_orig, X_reduced, n_neighbors=k))


def compute_neighborhood_metrics(
    X_orig: np.ndarray, X_reduced: np.ndarray, k: int = 5
) -> dict:
    """
    Compute all neighborhood preservation metrics and return as a dict.

    Flattens both arrays before computing, so inputs can be
    (n_samples, n_channels, n_timepoints) or already 2-D.
    """
    n = X_orig.shape[0]
    orig_flat = X_orig.reshape(n, -1)
    red_flat  = X_reduced.reshape(n, -1)

    return {
        f"precision@k": precision_at_k(orig_flat, red_flat, k),
        "trustworthiness": compute_trustworthiness(orig_flat, red_flat, k),
    }
