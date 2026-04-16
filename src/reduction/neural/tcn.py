"""Temporal Convolutional Network (TCN) Autoencoder for time series reduction."""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np

from .base import GlobalReducer, train_autoencoder, sign_correct, get_device

class TCNBlock(nn.Module):
    """
    Residual block with two acausal dilated convolutions (same-length output).

    Symmetric padding (acausal design).
    Weight normalisation.
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, n_filters, kernel_size,
                                           dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_filters, n_filters, kernel_size,
                                           dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, n_filters, 1) if in_channels != n_filters else None
        self.kernel_size = kernel_size
        self.dilation = dilation

    def _same_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Apply symmetric padding to ensure output length is same as input."""
        pad_size = (self.kernel_size - 1) * self.dilation
        return nn.functional.pad(x, (pad_size, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TCN residual block."""
        out = self._same_pad(x)
        out = self.relu(self.conv1(out))
        out = self.dropout(out)
        
        out = self._same_pad(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Stack of TCN residual blocks with exponentially increasing dilation rates:
    1, 2, 4, …, 2^(n_levels−1).

    Receptive field ≈ 2 × (kernel_size − 1) × (2^n_levels − 1).
    """

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_size: int,
        n_levels: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i in range(n_levels):
            dilation = 2 ** i
            block_in_channels = in_channels if i == 0 else n_filters
            layers.append(TCNBlock(block_in_channels, n_filters, kernel_size,
                                   dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TCN stack."""
        return self.network(x)


class TCNAE(nn.Module):
    """
    TCN Autoencoder.

    Encoder
    -------
        Input (1, T)
        → TCN (dilations 1…2^(n_levels−1), k=kernel_size, n_filters filters)
        → Conv1d 1×1 (latent_channels filters)   ← channel compression
        → AdaptiveAvgPool1d(w)                    ← temporal downsampling
        → Conv1d 1×1 (1 filter)                   ← 1-channel latent

    Decoder (receives the latent_channels-channel pooled representation)
    -------
        → Upsample(T, nearest)
        → TCN (same hyper-params, independent weights)
        → Conv1d 1×1 (1 filter)                   ← univariate reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        target_len: int,
        n_filters: int = 20,
        kernel_size: int = 20,
        n_levels: int = 5,
        latent_channels: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_tcn = TCN(1, n_filters, kernel_size, n_levels, dropout)
        self.to_latent_space = nn.Conv1d(n_filters, latent_channels, 1)
        self.pool = nn.AdaptiveAvgPool1d(target_len)
        self.to_latent_channel = nn.Conv1d(latent_channels, 1, 1)

        self.decoder_upsample = nn.Upsample(size=input_dim, mode='nearest')
        self.decoder_tcn = TCN(latent_channels, n_filters, kernel_size, n_levels, dropout)
        self.to_recon = nn.Conv1d(n_filters, 1, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass returns reconstruction and latent representation."""
        encoded = self.encoder_tcn(x)
        latent_multi_channel = self.to_latent_space(encoded)
        pooled = self.pool(latent_multi_channel)
        latent = self.to_latent_channel(pooled)

        upsampled = self.decoder_upsample(pooled)
        decoded = self.decoder_tcn(upsampled)
        recon = self.to_recon(decoded)
        
        return recon, latent


def TCN_SIT_reduce(
    series: np.ndarray,
    w: int,
    n_filters: int = 20,
    kernel_size: int = 20,
    n_levels: int = 5,
    latent_channels: int = 8,
    dropout: float = 0.2,
    epochs: int = 50,
    lr: float = 0.01,
) -> np.ndarray:
    """
    TCN Autoencoder reduction — Single Instance Training (SIT).

    Trains a fresh TCNAE on a single series and returns the temporal
    average pooling latent of length w.
    """
    N = len(series)
    if w >= N:
        raise ValueError("w must be smaller than series length.")

    device = get_device()
    x = torch.from_numpy(series.astype(np.float32)).view(1, 1, -1).to(device)
    model = TCNAE(
        N, w, n_filters, kernel_size, n_levels, latent_channels, dropout
    ).to(device)
    train_autoencoder(model, x, epochs, lr)

    model.eval()
    with torch.no_grad():
        _, latent = model(x)
    return sign_correct(latent.view(-1).cpu().numpy(), series, w)


class TCNReducer(GlobalReducer):
    """
    TCN Autoencoder — global training mode (default, no suffix).

    A single TCNAE is trained on all training-set series; the trained
    encoder is applied to both splits.
    """
    _has_channel_dim = True

    def __init__(
        self,
        n_filters: int = 20,
        kernel_size: int = 20,
        n_levels: int = 5,
        latent_channels: int = 8,
        dropout: float = 0.2,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
    ):
        self.n_filters       = n_filters
        self.kernel_size     = kernel_size
        self.n_levels        = n_levels
        self.latent_channels = latent_channels
        self.dropout         = dropout
        self.epochs          = epochs
        self.lr              = lr
        self.batch_size      = batch_size

    def _make_model(self, N: int, w: int) -> TCNAE:
        """Creates a TCNAE model instance."""
        return TCNAE(
            N, w, self.n_filters, self.kernel_size, self.n_levels,
            self.latent_channels, self.dropout
        )
