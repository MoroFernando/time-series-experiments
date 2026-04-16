"""
This file will be the new entry point for the reduction methods.
It will import the methods from the new modularized files.
"""
# Statistical methods
from .statistical import PAA_reduce, DFT_reduce, DWT_reduce, SVD_reduce

# Manifold and matrix decomposition methods
from .manifold import PCA_reduce, KPCA_reduce, Isomap_reduce

# Neural methods
from .neural.ae import AE_SIT_reduce, AEReducer
from .neural.cae import CAE_SIT_reduce, CAEReducer
from .neural.tcn import TCN_SIT_reduce, TCNReducer
from .neural.s2v import Series2VecReducer

__all__ = [
    # Statistical
    "PAA_reduce",
    "DFT_reduce",
    "DWT_reduce",
    "SVD_reduce",
    # Manifold
    "PCA_reduce",
    "KPCA_reduce",
    "Isomap_reduce",
    # Neural - SIT (Single Instance Training)
    "AE_SIT_reduce",
    "CAE_SIT_reduce",
    "TCN_SIT_reduce",
    # Neural - Global (Multi-Instance Training)
    "AEReducer",
    "CAEReducer",
    "TCNReducer",
    "Series2VecReducer",
]
