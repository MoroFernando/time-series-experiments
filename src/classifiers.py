"""
Classifier registry for the dimensionality reduction experiments.

Each entry maps a short name to a freshly-constructed classifier instance.
Classifiers are re-instantiated per experiment run to avoid state bleed.
"""

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.deep_learning import LITETimeClassifier
from aeon.classification.dictionary_based import MUSE
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.classification.interval_based import QUANTClassifier


def get_classifiers(random_state: int = 1) -> dict:
    """
    Return a fresh dict of {name: classifier_instance}.

    All classifiers are constructed with the same random_state for
    reproducibility and n_jobs=-1 to use all available CPU cores.

    Parameters
    ----------
    random_state : seed for reproducibility (default 1)

    Classifiers
    -----------
    Rocket           Convolutional kernels + Ridge (fast, strong baseline).
    Catch22          22 canonical time-series features + classifier.
    QUANT            Quantile-based interval features.
    1NN-DTW          1-Nearest-Neighbour with Dynamic Time Warping (10 % window).
    LITE             Lightweight Inception-based deep classifier (TensorFlow).
    MUSE             WEASEL+MUSE: multivariate dictionary / bag-of-patterns with
                     SFA words and logistic regression (chi2 feature selection).
    FreshPRINCE      tsfresh features (comprehensive) + rotation forest.
    """
    return {
        "Rocket":        RocketClassifier(random_state=random_state, n_jobs=-1),
        "Catch22":       Catch22Classifier(random_state=random_state, n_jobs=-1),
        "QUANT":         QUANTClassifier(random_state=random_state),
        "1NN-DTW":       KNeighborsTimeSeriesClassifier(
                             n_neighbors=1,
                             distance="dtw",
                             distance_params={"window": 0.1},
                             n_jobs=-1,
                         ),
        "LITE":          LITETimeClassifier(random_state=random_state),
        "MUSE":          MUSE(random_state=random_state, n_jobs=-1),
        "FreshPRINCE":   FreshPRINCEClassifier(random_state=random_state, n_jobs=-1)
    }
