"""
Dataset loading and preprocessing utilities.
"""

import numpy as np
from aeon.datasets import load_classification


def znorm(x: np.ndarray) -> np.ndarray:
    """Z-normalise a 1-D array. Returns x unchanged if std == 0."""
    std = np.std(x)
    if std == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def load_and_normalize(dataset_name: str) -> tuple:
    """
    Load an aeon classification dataset and apply per-series Z-normalisation.

    Parameters
    ----------
    dataset_name : name recognised by aeon.datasets.load_classification

    Returns
    -------
    X_train : np.ndarray of shape (n_train, n_channels, n_timepoints)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_channels, n_timepoints)
    y_test  : np.ndarray of shape (n_test,)
    """
    print(f"[dataset] Loading '{dataset_name}'...")
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    print(
        f"[dataset] Loaded — train: {X_train.shape}, test: {X_test.shape}"
    )

    print("[dataset] Applying Z-normalisation...")
    X_train = np.array([[znorm(s) for s in sample] for sample in X_train])
    X_test = np.array([[znorm(s) for s in sample] for sample in X_test])
    print("[dataset] Normalisation done.")

    return X_train, y_train, X_test, y_test
