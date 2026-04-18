"""Dense Autoencoder (AE) for time series dimensionality reduction."""
import torch
import torch.nn as nn
import numpy as np

from .base import GlobalReducer, train_autoencoder, sign_correct, get_device, clamp

class DenseAE(nn.Module):
    """
    Dense (fully-connected) autoencoder.

    Two hidden layers per path with adaptive widths that scale with the
    input size and are clamped to avoid excessive parameter counts on long
    series:
        h1 = clamp(N // 2,  w + 1, 256)
        h2 = clamp(N // 4,  w + 1, 128)

    Improvements over a single-hidden-layer baseline:
      • Deeper feature hierarchy (two hidden layers)
      • ELU activation — smooth gradient for negative inputs (Clevert et al.)
      • Dropout for regularisation
      • No activation on the bottleneck — unbounded latent values
    """

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        h1 = clamp(input_dim // 2, latent_dim + 1, 256)
        h2 = clamp(input_dim // 4, latent_dim + 1, 128)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h1), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        """Forward pass returns reconstruction and latent representation."""
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


def AE_SIT_reduce(
    series: np.ndarray,
    w: int,
    epochs: int = 50,
    lr: float = 0.01,
    dropout: float = 0.1,
) -> np.ndarray:
    """
    Dense Autoencoder reduction — Single Instance Training (SIT).

    Trains a fresh DenseAE on a single series and returns its bottleneck
    representation of length w.
    """
    N = len(series)
    if w >= N:
        raise ValueError("w must be smaller than series length.")

    device = get_device()
    x = torch.from_numpy(series.astype(np.float32)).view(1, -1).to(device)
    model = DenseAE(N, w, dropout).to(device)
    train_autoencoder(model, x, epochs, lr)

    model.eval()
    with torch.no_grad():
        _, latent = model(x)
    result = sign_correct(latent.view(-1).cpu().numpy(), series, w)
    del model, x, latent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


class AEReducer(GlobalReducer):
    """
    Dense Autoencoder — global training mode (default, no suffix).

    A single DenseAE is trained on all training-set series; the trained
    encoder is applied to both splits.
    """
    _has_channel_dim = False

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

    def _make_model(self, N: int, w: int) -> DenseAE:
        """Creates a DenseAE model instance."""
        return DenseAE(N, w, self.dropout)
