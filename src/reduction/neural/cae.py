"""1-D Convolutional Autoencoder (CAE) for time series dimensionality reduction."""
import torch
import torch.nn as nn
import numpy as np

from .base import GlobalReducer, train_autoencoder, sign_correct, get_device

class ConvAE(nn.Module):
    """
    1-D Convolutional Autoencoder.

    Encoder
    -------
    Three Conv1d layers with increasing filter counts and multi-scale kernels
    capture features at different temporal scales:
        Conv1d(1  → 32, k=7) ELU Dropout
        Conv1d(32 → 64, k=5) ELU Dropout
        Conv1d(64 → 64, k=3) ELU
        AdaptiveAvgPool1d(w)
        Conv1d(64 → 1, k=1)  → 1-channel latent

    Decoder
    -------
    Receives the 64-channel pooled representation (not the 1-channel latent),
    preserving richer gradient information during training — the same design
    choice made in TCN-AE:
        Upsample(T, linear)
        Conv1d(64 → 64, k=3) ELU Dropout
        Conv1d(64 → 32, k=5) ELU Dropout
        Conv1d(32 → 1,  k=7)
    """

    def __init__(self, input_dim: int, target_len: int, dropout: float = 0.1):
        super().__init__()
        self.target_len = target_len
        
        # Encoder
        self.enc_conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.enc_conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(target_len)
        self.latent_conv = nn.Conv1d(64, 1, kernel_size=1)
        
        # Decoder
        self.dec_upsample = nn.Upsample(size=input_dim, mode='linear', align_corners=False)
        self.dec_conv1 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.dec_conv3 = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass returns reconstruction and latent representation."""
        # Encoder
        h1 = self.dropout(self.elu(self.enc_conv1(x)))
        h2 = self.dropout(self.elu(self.enc_conv2(h1)))
        h3 = self.elu(self.enc_conv3(h2)) # No dropout on last enc layer
        
        pooled = self.pool(h3)
        latent = self.latent_conv(pooled)
        
        # Decoder
        upsampled = self.dec_upsample(pooled)
        d1 = self.dropout(self.elu(self.dec_conv1(upsampled)))
        d2 = self.dropout(self.elu(self.dec_conv2(d1)))
        recon = self.dec_conv3(d2)
        
        return recon, latent


def CAE_SIT_reduce(
    series: np.ndarray,
    w: int,
    epochs: int = 50,
    lr: float = 0.01,
    dropout: float = 0.1,
) -> np.ndarray:
    """
    1-D Convolutional Autoencoder reduction — Single Instance Training (SIT).

    Trains a fresh ConvAE on a single series and returns the
    AdaptiveAvgPool latent of length w.
    """
    N = len(series)
    if w >= N:
        raise ValueError("w must be smaller than series length.")

    device = get_device()
    x = torch.from_numpy(series.astype(np.float32)).view(1, 1, -1).to(device)
    model = ConvAE(N, w, dropout).to(device)
    train_autoencoder(model, x, epochs, lr)

    model.eval()
    with torch.no_grad():
        _, latent = model(x)
    return sign_correct(latent.view(-1).cpu().numpy(), series, w)


class CAEReducer(GlobalReducer):
    """
    1-D Convolutional Autoencoder — global training mode (default, no suffix).

    A single ConvAE is trained on all training-set series; the trained
    encoder is applied to both splits.
    """
    _has_channel_dim = True

    def __init__(
        self,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
        dropout: float = 0.1,
    ):
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.dropout    = dropout

    def _make_model(self, N: int, w: int) -> ConvAE:
        """Creates a ConvAE model instance."""
        return ConvAE(N, w, self.dropout)
