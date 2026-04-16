"""
Dimensionality reduction methods for time series.

Naming convention
-----------------
Global methods (default, no suffix):
    Train one model on the full training set, then apply the encoder to both
    train and test sets. Exposes the dataset-level interface:

        reducer.fit_transform(X_train, w) -> np.ndarray
        reducer.transform(X_test,  w)     -> np.ndarray

    where X has shape (n_samples, n_channels, n_timepoints).

Single Instance Training (-SIT suffix):
    Train one model per series. Follows the per-series signature:

        reduce_SIT(series: np.ndarray, w: int, **kwargs) -> np.ndarray

`w` is the target number of timepoints to *retain* in all cases.

Neural architectures
--------------------
  AE  / AE-SIT   Dense (fully-connected) autoencoder
  CAE / CAE-SIT  1-D Convolutional autoencoder
  TCN / TCN-SIT  Temporal Convolutional Network autoencoder
  S2V            Series2Vec self-supervised encoder (global only — no SIT
                 variant, since similarity-based training requires pairs)

Architecture highlights
-----------------------
  AE   Two hidden layers with adaptive widths (scale with N, clamped to 256/128),
       ELU activations, dropout, unbounded bottleneck.

  CAE  Three Conv1d layers with increasing filter counts (32→64), multi-scale
       kernels (7,5,3), ELU activations, dropout. Decoder receives the full
       multi-channel pooled representation (64 channels) instead of the
       1-channel latent, following the TCN-AE design for richer gradients.

  TCN  Residual blocks with exponentially dilated acausal convolutions and
       weight normalisation, wrapped in an AE framework.
       Receptive field scales as O(2^n_levels).

  S2V  Disjoint-CNN encoder (2 layers, 16 filters) trained with a pairwise
       similarity-preserving loss in both time and frequency domains. Order-invariant Transformer attention over the
       mini-batch enforces consistent representations for similar series.
       No decoder — the encoder output IS the reduced series (length w).
"""
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import SparseEfficiencyWarning
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap


# ---------------------------------------------------------------------------
# Statistical
# ---------------------------------------------------------------------------

def PAA_reduce(series: np.ndarray, w: int) -> np.ndarray:
    """
    Piecewise Aggregate Approximation (PAA).

    Divides the series into w equal-length segments and replaces each with
    its mean.
    """
    series = np.asarray(series)
    idx = np.floor(np.linspace(0, w, len(series), endpoint=False)).astype(int)
    return np.array([np.mean(series[idx == i]) for i in range(w)])


def DFT_reduce(series: np.ndarray, w: int) -> np.ndarray:
    """
    Discrete Fourier Transform (DFT) reduction.

    Keeps the first k positive and k negative frequency components,
    reconstructs via IFFT, then subsamples to length w.
    """
    N = len(series)
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
    coeffs = pywt.wavedec(series, wavelet, level=level)
    cA = coeffs[0]
    return cA[np.linspace(0, len(cA) - 1, w).astype(int)]


def SVD_reduce(series: np.ndarray, w: int, window: int = 10) -> np.ndarray:
    """
    SVD-based reduction via Hankel embedding.

    Embeds the series into a sliding-window matrix, applies SVD keeping the
    top w singular values, reconstructs, and subsamples to length w.
    """
    N = len(series)
    if N < window:
        raise ValueError("window must be smaller than series length.")
    X = np.array([series[i : i + window] for i in range(N - window + 1)])
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_reduced = np.zeros_like(S)
    S_reduced[: min(w, len(S))] = S[: min(w, len(S))]
    collapsed = np.mean((U * S_reduced) @ VT, axis=1)
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]


# ---------------------------------------------------------------------------
# Matrix decomposition / manifold
# ---------------------------------------------------------------------------

def _sliding_window(series: np.ndarray, window: int) -> np.ndarray:
    N = len(series)
    if N < window:
        raise ValueError("window must be smaller than series length.")
    return np.array([series[i : i + window] for i in range(N - window + 1)])


def PCA_reduce(series: np.ndarray, w: int, window: int = 10) -> np.ndarray:
    """PCA reduction via sliding-window (Hankel) embedding."""
    X = _sliding_window(series, window)
    X_pca = PCA(n_components=min(w, X.shape[1])).fit_transform(X)
    collapsed = np.mean(X_pca, axis=1)
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]


def KPCA_reduce(
    series: np.ndarray,
    w: int,
    window: int = 10,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> np.ndarray:
    """Kernel PCA reduction via sliding-window embedding."""
    X = _sliding_window(series, window)
    X_kpca = KernelPCA(
        n_components=min(w, X.shape[1]), kernel=kernel, gamma=gamma
    ).fit_transform(X)
    collapsed = np.mean(X_kpca, axis=1)
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]


def Isomap_reduce(
    series: np.ndarray,
    w: int,
    window: int = 10,
    n_neighbors: int | None = None,
) -> np.ndarray:
    """Isomap reduction via sliding-window embedding."""
    import warnings

    X = _sliding_window(series, window)
    if n_neighbors is None:
        n_neighbors = max(5, int(np.sqrt(X.shape[0])))
    n_neighbors = min(n_neighbors, X.shape[0] - 1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                module="sklearn.manifold._isomap")
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning,
                                module="scipy.sparse")
        X_iso = Isomap(
            n_neighbors=n_neighbors, n_components=min(w, X.shape[1])
        ).fit_transform(X)

    collapsed = np.mean(X_iso, axis=1)
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]


# ---------------------------------------------------------------------------
# Neural — shared infrastructure
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clamp(value: int, lo: int, hi: int) -> int:
    """Integer clamping helper."""
    return max(lo, min(value, hi))


def _sign_correct(reduced: np.ndarray, series: np.ndarray, w: int) -> np.ndarray:
    """Flip sign if the reduced series anti-correlates with the subsampled original."""
    subsampled = series[np.linspace(0, len(series) - 1, w).astype(int)]
    if np.std(reduced) > 1e-9 and np.std(subsampled) > 1e-9:
        if np.corrcoef(reduced, subsampled)[0, 1] < 0:
            return -reduced
    return reduced


def _sign_correct_batch(reduced: np.ndarray, originals: np.ndarray, w: int) -> np.ndarray:
    """Apply per-series sign correction to a batch of reduced series."""
    idx = np.linspace(0, originals.shape[1] - 1, w).astype(int)
    result = reduced.copy()
    for i, (r, s) in enumerate(zip(reduced, originals)):
        sub = s[idx]
        if np.std(r) > 1e-9 and np.std(sub) > 1e-9:
            if np.corrcoef(r, sub)[0, 1] < 0:
                result[i] = -r
    return result


def _train_autoencoder(model: nn.Module, x: torch.Tensor, epochs: int, lr: float) -> None:
    """Train an autoencoder on a single batch (SIT mode)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, _ = model(x)
        criterion(recon, x).backward()
        optimizer.step()


def _print_train_progress(epoch: int, total: int, loss: float, bar_len: int = 30) -> None:
    """Overwrite the current line with an epoch-level training progress bar."""
    import sys
    pct   = epoch / total
    filled = int(pct * bar_len)
    bar   = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(
        f"\r  [training] [{bar}] {pct:>5.1%}  "
        f"epoch {epoch:>{len(str(total))}}/{total}  loss={loss:.6f}"
    )
    sys.stdout.flush()


def _train_autoencoder_batched(
    model: nn.Module,
    X: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Train an autoencoder on a dataset using mini-batches (global mode)."""
    from torch.utils.data import DataLoader, TensorDataset

    loader    = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        _print_train_progress(epoch + 1, epochs, epoch_loss / len(loader))
    print()  # newline after the progress bar


class _GlobalReducer:
    """
    Base class for global-training reducers.

    One model is trained on all training-set series (fit_transform), then
    the encoder is applied to any split (transform). Test series are never
    seen during training.

    Subclasses must:
      • set  _has_channel_dim : bool  (True for Conv/TCN, False for Dense)
      • set  epochs, lr, batch_size, _model, _w  in __init__
      • implement  _make_model(N, w) -> nn.Module
    """

    _has_channel_dim: bool = True

    # --- public API -------------------------------------------------------

    def fit_transform(self, X: np.ndarray, w: int) -> np.ndarray:
        n_samples, n_channels, N = X.shape
        if w >= N:
            raise ValueError("w must be smaller than series length.")

        device = _get_device()
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_tensor = self._to_tensor(X_flat, device)

        self._model = self._make_model(N, w).to(device)
        self._w = w
        _train_autoencoder_batched(
            self._model, X_tensor, self.epochs, self.lr, self.batch_size
        )
        return self._encode(X, device)

    def transform(self, X: np.ndarray, w: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit_transform before transform.")
        if w != self._w:
            raise ValueError(f"w={w} does not match fitted w={self._w}.")
        return self._encode(X, _get_device())

    # --- internals --------------------------------------------------------

    def _encode(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        n_samples, n_channels, N = X.shape
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_tensor = self._to_tensor(X_flat, device)

        self._model.eval()
        with torch.no_grad():
            _, latent = self._model(X_tensor)

        # Works for both (B, w) [dense] and (B, 1, w) [conv/tcn]
        reduced_flat = latent.cpu().numpy().reshape(-1, self._w)
        reduced_flat = _sign_correct_batch(reduced_flat, X_flat, self._w)
        return reduced_flat.reshape(n_samples, n_channels, self._w)

    def _to_tensor(self, X_flat: np.ndarray, device: torch.device) -> torch.Tensor:
        t = torch.from_numpy(X_flat).to(device)
        return t.unsqueeze(1) if self._has_channel_dim else t

    def _make_model(self, N: int, w: int) -> nn.Module:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Neural — Dense Autoencoder (AE / AE-SIT)
# ---------------------------------------------------------------------------

class _DenseAE(nn.Module):
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
        h1 = _clamp(input_dim // 2, latent_dim + 1, 256)
        h2 = _clamp(input_dim // 4, latent_dim + 1, 128)

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
        latent = self.encoder(x)
        return self.decoder(latent), latent


def AE_SIT_reduce(
    series: np.ndarray,
    w: int,
    epochs: int = 50,
    lr: float = 0.01,
    dropout: float = 0.1,
) -> np.ndarray:
    """
    Dense Autoencoder reduction — Single Instance Training (SIT).

    Trains a fresh _DenseAE on a single series and returns its bottleneck
    representation of length w.
    """
    N = len(series)
    if w >= N:
        raise ValueError("w must be smaller than series length.")

    device = _get_device()
    x = torch.FloatTensor(series).view(1, -1).to(device)
    model = _DenseAE(N, w, dropout).to(device)
    _train_autoencoder(model, x, epochs, lr)

    model.eval()
    with torch.no_grad():
        _, latent = model(x)
    return _sign_correct(latent.view(-1).cpu().numpy(), series, w)


class AEReducer(_GlobalReducer):
    """
    Dense Autoencoder — global training mode (default, no suffix).

    A single _DenseAE is trained on all training-set series; the trained
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
        self._model: _DenseAE | None = None
        self._w: int | None = None

    def _make_model(self, N: int, w: int) -> _DenseAE:
        return _DenseAE(N, w, self.dropout)


# ---------------------------------------------------------------------------
# Neural — 1-D Convolutional Autoencoder (CAE / CAE-SIT)
# ---------------------------------------------------------------------------

class _ConvAE(nn.Module):
    """
    1-D Convolutional Autoencoder.

    Encoder
    -------
    Three Conv1d layers with increasing filter counts and multi-scale kernels
    capture features at different temporal scales 
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
        C = 32  # base filter count

        # --- Encoder ---
        self.enc_conv = nn.Sequential(
            nn.Conv1d(1,     C,     kernel_size=7, padding=3), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(C,     C * 2, kernel_size=5, padding=2), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(C * 2, C * 2, kernel_size=3, padding=1), nn.ELU(),
        )
        self.pool         = nn.AdaptiveAvgPool1d(target_len)
        self.channel_mixer = nn.Conv1d(C * 2, 1, kernel_size=1)  # → 1-channel latent

        # --- Decoder (from the 64-channel pooled representation) ---
        self.dec_conv = nn.Sequential(
            nn.Upsample(size=input_dim, mode="linear", align_corners=False),
            nn.Conv1d(C * 2, C * 2, kernel_size=3, padding=1), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(C * 2, C,     kernel_size=5, padding=2), nn.ELU(), nn.Dropout(dropout),
            nn.Conv1d(C,     1,     kernel_size=7, padding=3),
        )

    def forward(self, x):
        enc        = self.enc_conv(x)           # (B, 64, T)
        enc_pooled = self.pool(enc)              # (B, 64, w)
        latent     = self.channel_mixer(enc_pooled)  # (B, 1, w)
        recon      = self.dec_conv(enc_pooled)   # (B, 1, T)
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

    Trains a fresh _ConvAE on a single series and returns the
    AdaptiveAvgPool latent of length w.
    """
    N = len(series)
    if w >= N:
        raise ValueError("w must be smaller than series length.")

    device = _get_device()
    x = torch.FloatTensor(series).view(1, 1, -1).to(device)
    model = _ConvAE(N, w, dropout).to(device)
    _train_autoencoder(model, x, epochs, lr)

    model.eval()
    with torch.no_grad():
        _, latent = model(x)
    return _sign_correct(latent.view(-1).cpu().numpy(), series, w)


class CAEReducer(_GlobalReducer):
    """
    1-D Convolutional Autoencoder — global training mode (default, no suffix).

    A single _ConvAE is trained on all training-set series; the trained
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
        self._model: _ConvAE | None = None
        self._w: int | None = None

    def _make_model(self, N: int, w: int) -> _ConvAE:
        return _ConvAE(N, w, self.dropout)


# ---------------------------------------------------------------------------
# Neural — Temporal Convolutional Network Autoencoder (TCN / TCN-SIT)
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
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
        self._total_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, n_filters, kernel_size, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(n_filters, n_filters, kernel_size, dilation=dilation)
        )
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip    = (
            nn.Conv1d(in_channels, n_filters, 1) if in_channels != n_filters else None
        )

    def _same_pad(self, x: torch.Tensor) -> torch.Tensor:
        p = self._total_pad
        return F.pad(x, (p // 2, p - p // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.conv1(self._same_pad(x))))
        out = self.dropout(self.relu(self.conv2(self._same_pad(out))))
        res = x if self.skip is None else self.skip(x)
        return self.relu(out + res)


class _TCN(nn.Module):
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
        self.net = nn.Sequential(*[
            _TCNBlock(
                in_channels if i == 0 else n_filters,
                n_filters, kernel_size, dilation=2 ** i, dropout=dropout,
            )
            for i in range(n_levels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TCNAE(nn.Module):
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
        # --- Encoder ---
        self.enc_tcn      = _TCN(1, n_filters, kernel_size, n_levels, dropout)
        self.enc_proj     = nn.Conv1d(n_filters, latent_channels, 1)
        self.pool         = nn.AdaptiveAvgPool1d(target_len)
        self.channel_mixer = nn.Conv1d(latent_channels, 1, 1)
        # --- Decoder (from latent_channels-channel pooled representation) ---
        self.upsample = nn.Upsample(size=input_dim, mode="nearest")
        self.dec_tcn  = _TCN(latent_channels, n_filters, kernel_size, n_levels, dropout)
        self.dec_proj = nn.Conv1d(n_filters, 1, 1)

    def forward(self, x: torch.Tensor):
        enc        = self.enc_proj(self.enc_tcn(x))    # (B, latent_channels, T)
        enc_pooled = self.pool(enc)                     # (B, latent_channels, w)
        latent     = self.channel_mixer(enc_pooled)    # (B, 1, w)
        up         = self.upsample(enc_pooled)          # (B, latent_channels, T)
        recon      = self.dec_proj(self.dec_tcn(up))   # (B, 1, T)
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

    Trains a fresh _TCNAE on a single series and returns the temporal
    average pooling latent of length w.
    """
    N = len(series)
    if w >= N:
        raise ValueError("w must be smaller than series length.")

    device = _get_device()
    x = torch.FloatTensor(series).view(1, 1, -1).to(device)
    model = _TCNAE(
        N, w, n_filters, kernel_size, n_levels, latent_channels, dropout
    ).to(device)
    _train_autoencoder(model, x, epochs, lr)

    model.eval()
    with torch.no_grad():
        _, latent = model(x)
    return _sign_correct(latent.view(-1).cpu().numpy(), series, w)


class TCNReducer(_GlobalReducer):
    """
    TCN Autoencoder — global training mode (default, no suffix).

    A single _TCNAE is trained on all training-set series; the trained
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
        self._model: _TCNAE | None = None
        self._w: int | None = None

    def _make_model(self, N: int, w: int) -> _TCNAE:
        return _TCNAE(
            N, w, self.n_filters, self.kernel_size,
            self.n_levels, self.latent_channels, self.dropout,
        )


# ---------------------------------------------------------------------------
# Neural — Series2Vec Encoder (S2V)
# ---------------------------------------------------------------------------

class _S2VEncoder(nn.Module):
    """
    Disjoint-CNN encoder for univariate time series (Series2Vec, Foumani et al. 2023).

    Follows the paper's DisjoinEncoder adapted for single-channel input:
      Conv1d(1 → emb_size, k=8) + BatchNorm + GELU   ← temporal features
      Conv1d(emb_size → emb_size, k=5) + BatchNorm + GELU
      AdaptiveAvgPool1d(w)                             ← temporal downsampling
      Conv1d(emb_size → 1, k=1)                       ← channel collapse

    Xavier initialisation on all Conv1d weights.

    Input  : (B, 1, N)
    Output : (B, w)
    """

    def __init__(self, latent_dim: int, emb_size: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,        emb_size, kernel_size=8, padding=4), nn.BatchNorm1d(emb_size), nn.GELU(),
            nn.Conv1d(emb_size, emb_size, kernel_size=5, padding=2), nn.BatchNorm1d(emb_size), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(latent_dim)
        self.proj = nn.Conv1d(emb_size, 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)     # (B, emb_size, N')  N' may differ by 1 due to even kernel
        h = self.pool(h)     # (B, emb_size, w)
        h = self.proj(h)     # (B, 1, w)
        return h.squeeze(1)  # (B, w)


class _S2VTrainer(nn.Module):
    """
    Full Series2Vec model used during pre-training only.

    Matches the paper's Pretrain_forward (Section 3.2 and source code):
      • time_enc / freq_enc  (_S2VEncoder): (B, 1, N) → (B, w)
      • Linear projection w → D_MODEL (64, paper default)
      • MultiheadAttention(Q=K=V) on shape (1, B, D_MODEL):
          seq_len=1, batch=B — acts as a per-sample linear transform
          with residual + LayerNorm, followed by FFN + LayerNorm
      • torch.cdist(z, z) → (B, B) pairwise Euclidean distance matrix
        returned directly (loss computed in the training loop)

    After pre-training only time_enc is used at inference.

    Fixed hyperparameters (paper Section 4.2):
      D_MODEL = 64   (transformer encoding dimension d_m)
      N_HEADS  = 8   (number of attention heads)
    """

    D_MODEL: int = 64
    N_HEADS:  int = 8

    def __init__(self, latent_dim: int, emb_size: int = 16):
        super().__init__()
        dm = self.D_MODEL
        nh = self.N_HEADS

        self.time_enc = _S2VEncoder(latent_dim, emb_size)
        self.freq_enc = _S2VEncoder(latent_dim, emb_size)

        self.time_proj = nn.Linear(latent_dim, dm)
        self.freq_proj = nn.Linear(latent_dim, dm)

        # MultiheadAttention: input (1, B, dm) → seq_len=1, batch=B
        # Replicates paper's Pretrain_forward attention_layer call
        self.t_attn  = nn.MultiheadAttention(dm, nh, dropout=0.1, batch_first=False)
        self.f_attn  = nn.MultiheadAttention(dm, nh, dropout=0.1, batch_first=False)
        self.t_norm1 = nn.LayerNorm(dm)
        self.f_norm1 = nn.LayerNorm(dm)
        self.t_ff    = nn.Sequential(nn.Linear(dm, dm * 4), nn.GELU(), nn.Linear(dm * 4, dm))
        self.f_ff    = nn.Sequential(nn.Linear(dm, dm * 4), nn.GELU(), nn.Linear(dm * 4, dm))
        self.t_norm2 = nn.LayerNorm(dm)
        self.f_norm2 = nn.LayerNorm(dm)

    def _branch(
        self,
        enc:   nn.Module,
        proj:  nn.Module,
        attn:  nn.Module,
        norm1: nn.Module,
        ff:    nn.Module,
        norm2: nn.Module,
        x:     torch.Tensor,
    ):
        """Encode → project → attention → FFN → pairwise distance matrix."""
        r = enc(x)                            # (B, w)
        h = proj(r)                           # (B, dm)
        # Reshape to (1, B, dm): seq_len=1, batch=B
        # This matches the paper's permute(2,0,1) on (B, dm, 1) GAP output
        h_seq = h.unsqueeze(0)                # (1, B, dm)
        att, _ = attn(h_seq, h_seq, h_seq)    # (1, B, dm)
        att = norm1(att + h_seq)              # residual + LayerNorm
        out = norm2(att + ff(att))            # FFN + residual + LayerNorm
        z   = out.squeeze(0)                  # (B, dm)
        d   = torch.cdist(z, z)              # (B, B) pairwise Euclidean distances
        return r, d

    def forward(
        self,
        x_t: torch.Tensor,   # (B, 1, N) — time domain
        x_f: torch.Tensor,   # (B, 1, N) — |FFT| magnitude of x_t
    ):
        r_T, d_T = self._branch(
            self.time_enc, self.time_proj, self.t_attn,
            self.t_norm1, self.t_ff, self.t_norm2, x_t,
        )
        _,   d_F = self._branch(
            self.freq_enc, self.freq_proj, self.f_attn,
            self.f_norm1, self.f_ff, self.f_norm2, x_f,
        )
        return r_T, d_T, d_F   # (B, w), (B, B), (B, B)


def _s2v_minmax(d: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalise a distance vector or matrix to [0, 1].

    Replicates Distance_normalizer() from the paper's source code.
    Returns zeros if all distances are identical (degenerate batch).
    """
    lo, hi = d.min(), d.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(d)
    return (d - lo) / (hi - lo)


def _train_series2vec(
    model: nn.Module,
    X_t: torch.Tensor,   # (M, 1, N) — time series
    X_f: torch.Tensor,   # (M, 1, N) — |FFT| magnitudes
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """
    Train the Series2Vec model with the similarity-preserving loss.

    For each mini-batch of B series:
      1. Encode both domains → pairwise distance matrices d_rep_T, d_rep_F (B, B).
      2. Compute pairwise Euclidean distances in input space via torch.cdist.
      3. Extract lower-triangular pairs (diagonal excluded; dist(x,x)=0 would
         cause NaN gradients through sqrt).
      4. Min-max normalise all four distance vectors to [0, 1].
      5. Loss = smooth_L1(d_rep_T_norm, d_in_T_norm)
               + smooth_L1(d_rep_F_norm, d_in_F_norm)
      6. Clip gradients to max_norm=4.0.
    """
    from torch.utils.data import DataLoader, TensorDataset

    batch_size = min(batch_size, len(X_t))
    loader = DataLoader(
        TensorDataset(X_t, X_f),
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(X_t) > batch_size),
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_t_batch, x_f_batch in loader:
            optimizer.zero_grad()

            _, d_rep_T, d_rep_F = model(x_t_batch, x_f_batch)  # (B,B), (B,B)

            # ── Lower-triangular mask (exclude diagonal: dist(x,x)=0) ──────
            B    = d_rep_T.shape[0]
            mask = torch.tril(torch.ones(B, B, device=d_rep_T.device), diagonal=-1).bool()

            # ── Pairwise distances in input space (torch.cdist is stable) ───
            x_t_flat = x_t_batch.squeeze(1)                      # (B, N)
            x_f_flat = x_f_batch.squeeze(1)                      # (B, N)
            d_in_T   = torch.cdist(x_t_flat, x_t_flat)           # (B, B)
            d_in_F   = torch.cdist(x_f_flat, x_f_flat)           # (B, B)

            # ── Extract lower-triangular pairs → 1-D vectors ────────────────
            d_rep_T_vec = torch.masked_select(d_rep_T, mask)     # (B*(B-1)/2,)
            d_rep_F_vec = torch.masked_select(d_rep_F, mask)
            d_in_T_vec  = torch.masked_select(d_in_T,  mask)
            d_in_F_vec  = torch.masked_select(d_in_F,  mask)

            # ── Min-max normalise to [0, 1] ──────────────────────────────────
            d_rep_T_vec = _s2v_minmax(d_rep_T_vec)
            d_rep_F_vec = _s2v_minmax(d_rep_F_vec)
            d_in_T_vec  = _s2v_minmax(d_in_T_vec)
            d_in_F_vec  = _s2v_minmax(d_in_F_vec)

            # ── Similarity-preserving loss (L_total = L_sim_T + L_sim_F) ────
            loss = (
                F.smooth_l1_loss(d_rep_T_vec, d_in_T_vec)
                + F.smooth_l1_loss(d_rep_F_vec, d_in_F_vec)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            epoch_loss += loss.item()

        _print_train_progress(epoch + 1, epochs, epoch_loss / max(len(loader), 1))
    print()


class Series2VecReducer:
    """
    Series2Vec — global self-supervised dimensionality reduction.

    Trains a dual-encoder (temporal + spectral domains) with a pairwise
    similarity-preserving loss. After
    training, only the temporal encoder is used for inference: it maps
    each univariate channel from N timepoints to w timepoints.

    Key properties
    --------------
    • No decoder, no reconstruction loss — the encoder is trained to
      *preserve distance structure*, not to reconstruct the input.
    • No data augmentation — similarity is computed directly between raw
      series pairs (Euclidean distance, simplified from Soft-DTW).
    • Order-invariant attention during training:
      each representation is refined via attention + FFN before distances
      are computed, allowing the training signal to be computed over the
      full B×B pairwise structure of the mini-batch.
    • Fixed d_model=64 and n_heads=8.
    • Only a global mode is provided: pairwise similarity requires ≥ 2
      series, so per-series (SIT) training is not applicable.
    """

    def __init__(
        self,
        emb_size:   int   = 16,     # CNN filter count (paper default: 16)
        epochs:     int   = 100,    # pre-training epochs (paper: 100)
        lr:         float = 0.001,  # Adam learning rate
        batch_size: int   = 64,     # mini-batch size (paper: 64)
    ):
        self.emb_size   = emb_size
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self._model: _S2VTrainer | None = None
        self._w: int | None = None

    # ---- public API (mirrors _GlobalReducer) ----------------------------

    def fit_transform(self, X: np.ndarray, w: int) -> np.ndarray:
        n_samples, n_channels, N = X.shape
        if w >= N:
            raise ValueError("w must be smaller than series length.")

        device = _get_device()
        X_flat = X.reshape(-1, N).astype(np.float32)                  # (M, N)
        X_t    = torch.from_numpy(X_flat).unsqueeze(1).to(device)     # (M, 1, N)
        X_f    = self._fft_tensor(X_flat, device)                     # (M, 1, N)

        self._model = _S2VTrainer(w, self.emb_size).to(device)
        self._w = w
        _train_series2vec(self._model, X_t, X_f, self.epochs, self.lr, self.batch_size)

        return self._encode(X, device)

    def transform(self, X: np.ndarray, w: int) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit_transform before transform.")
        if w != self._w:
            raise ValueError(f"w={w} does not match fitted w={self._w}.")
        return self._encode(X, _get_device())

    # ---- internals -------------------------------------------------------

    @staticmethod
    def _fft_tensor(X_flat: np.ndarray, device: torch.device) -> torch.Tensor:
        """Compute |FFT| magnitude spectrum → (M, 1, N) tensor on device."""
        X_freq = np.abs(np.fft.fft(X_flat, axis=1)).astype(np.float32)
        return torch.from_numpy(X_freq).unsqueeze(1).to(device)

    def _encode(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        n_samples, n_channels, N = X.shape
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_t    = torch.from_numpy(X_flat).unsqueeze(1).to(device)  # (M, 1, N)

        self._model.eval()
        with torch.no_grad():
            r_T = self._model.time_enc(X_t)   # (M, w) — time encoder only

        reduced_flat = r_T.cpu().numpy()       # (M, w)
        reduced_flat = _sign_correct_batch(reduced_flat, X_flat, self._w)
        return reduced_flat.reshape(n_samples, n_channels, self._w)
