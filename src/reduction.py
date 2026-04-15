"""
Dimensionality reduction methods for time series.

Per-series methods follow the signature:
    reduce(series: np.ndarray, w: int, **kwargs) -> np.ndarray

Global methods (train on the full dataset) expose:
    method.fit_transform(X: np.ndarray, w: int) -> np.ndarray
    method.transform(X: np.ndarray, w: int) -> np.ndarray
where X has shape (n_samples, n_channels, n_timepoints).

`w` is the target number of timepoints to *retain* in all cases.
"""
import numpy as np
import pywt
import torch
import torch.nn as nn
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

    Divides the series into w equal-length segments and replaces each with its mean.
    """
    series = np.asarray(series)
    idx = np.floor(np.linspace(0, w, len(series), endpoint=False)).astype(int)
    return np.array([np.mean(series[idx == i]) for i in range(w)])


def DFT_reduce(series: np.ndarray, w: int) -> np.ndarray:
    """
    Discrete Fourier Transform (DFT) reduction.

    Keeps the first k positive and k negative frequency components, reconstructs
    via IFFT, then subsamples to length w.
    """
    N = len(series)
    X = np.fft.fft(series)

    k = w // 2
    X_reduced = np.zeros(N, dtype=complex)
    X_reduced[:k] = X[:k]
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

    Embeds the series into a sliding-window matrix, applies SVD keeping the top w
    singular values, reconstructs, and subsamples to length w.
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
    """
    PCA-based reduction via sliding-window embedding.

    Projects the Hankel matrix via PCA, collapses row-wise, and subsamples to w.
    """
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
    """
    Kernel PCA reduction via sliding-window embedding.
    """
    X = _sliding_window(series, window)
    X_kpca = KernelPCA(n_components=min(w, X.shape[1]), kernel=kernel, gamma=gamma).fit_transform(X)
    collapsed = np.mean(X_kpca, axis=1)
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]


def Isomap_reduce(
    series: np.ndarray,
    w: int,
    window: int = 10,
    n_neighbors: int | None = None,
) -> np.ndarray:
    """
    Isomap reduction via sliding-window embedding.
    """
    import warnings

    X = _sliding_window(series, window)
    if n_neighbors is None:
        n_neighbors = max(5, int(np.sqrt(X.shape[0])))
    n_neighbors = min(n_neighbors, X.shape[0] - 1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.manifold._isomap")
        warnings.filterwarnings("ignore", category=SparseEfficiencyWarning, module="scipy.sparse")
        X_iso = Isomap(n_neighbors=n_neighbors, n_components=min(w, X.shape[1])).fit_transform(X)

    collapsed = np.mean(X_iso, axis=1)
    return collapsed[np.linspace(0, len(collapsed) - 1, w).astype(int)]


# ---------------------------------------------------------------------------
# Neural
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_autoencoder(model: nn.Module, x: torch.Tensor, epochs: int, lr: float) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, _ = model(x)
        criterion(recon, x).backward()
        optimizer.step()


def _train_autoencoder_batched(
    model: nn.Module,
    X: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Train an autoencoder on a dataset using mini-batches."""
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            criterion(recon, batch).backward()
            optimizer.step()


class _DenseAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent


class _ConvAE(nn.Module):
    def __init__(self, input_dim: int, target_len: int, n_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=3, padding=1), nn.LeakyReLU(0.5),
            nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=1), nn.LeakyReLU(0.5),
            nn.AdaptiveAvgPool1d(target_len),
        )
        self.channel_mixer = nn.Conv1d(n_channels, 1, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=1),
            nn.Upsample(size=input_dim, mode="linear", align_corners=False),
            nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=1), nn.LeakyReLU(0.5),
            nn.Conv1d(n_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        latent = self.channel_mixer(self.encoder(x))
        return self.decoder(latent), latent


def AE_reduce(
    series: np.ndarray,
    w: int,
    hidden_dim: int = 64,
    epochs: int = 30,
    lr: float = 0.01,
) -> np.ndarray:
    """
    Dense Autoencoder reduction (single-instance).

    Trains a fully-connected AE on the single input series and returns the
    bottleneck representation of size w.
    """
    N = len(series)
    if w >= N:
        raise ValueError("w must be smaller than series length.")

    device = _get_device()
    x = torch.FloatTensor(series).view(1, -1).to(device)
    model = _DenseAE(N, w, hidden_dim).to(device)
    _train_autoencoder(model, x, epochs, lr)

    model.eval()
    with torch.no_grad():
        _, latent = model(x)
    return latent.squeeze().cpu().numpy()


def CAE_reduce(
    series: np.ndarray,
    w: int,
    epochs: int = 50,
    lr: float = 0.01,
    n_channels: int = 8,
) -> np.ndarray:
    """
    Convolutional Autoencoder reduction (single-instance).

    Trains a 1-D CAE on the single input series. Sign is corrected by
    correlation with the subsampled original to ensure consistent orientation.
    """
    N = len(series)
    if w >= N:
        raise ValueError("w must be smaller than series length.")

    device = _get_device()
    x = torch.FloatTensor(series).view(1, 1, -1).to(device)
    model = _ConvAE(N, w, n_channels).to(device)
    _train_autoencoder(model, x, epochs, lr)

    model.eval()
    with torch.no_grad():
        _, latent = model(x)
    reduced = latent.squeeze().cpu().numpy()

    # Sign correction via correlation with subsampled original
    subsampled = series[np.linspace(0, N - 1, w).astype(int)]
    if np.std(reduced) > 1e-9 and np.std(subsampled) > 1e-9:
        if np.corrcoef(reduced, subsampled)[0, 1] < 0:
            reduced = -reduced

    return reduced


def _sign_correct_batch(reduced: np.ndarray, originals: np.ndarray, w: int) -> np.ndarray:
    """Apply per-series sign correction to a batch of reduced series."""
    N = originals.shape[1]
    idx = np.linspace(0, N - 1, w).astype(int)
    result = reduced.copy()
    for i, (r, s) in enumerate(zip(reduced, originals)):
        sub = s[idx]
        if np.std(r) > 1e-9 and np.std(sub) > 1e-9:
            if np.corrcoef(r, sub)[0, 1] < 0:
                result[i] = -r
    return result


class CAEGlobalReducer:
    """
    Convolutional Autoencoder reduction — global training mode.

    A single CAE is trained on all series in the training set (across all
    samples and channels). The trained encoder is then applied to both the
    training and test sets, so test series are never seen during training.

    Exposes the dataset-level interface used by `reduce_dataset`:
        fit_transform(X_train, w) -> np.ndarray
        transform(X_test, w)     -> np.ndarray
    where X has shape (n_samples, n_channels, n_timepoints).
    """

    def __init__(
        self,
        epochs: int = 50,
        lr: float = 0.01,
        n_channels: int = 8,
        batch_size: int = 32,
    ):
        self.epochs = epochs
        self.lr = lr
        self.n_channels = n_channels
        self.batch_size = batch_size
        self._model: _ConvAE | None = None
        self._w: int | None = None

    def fit_transform(self, X: np.ndarray, w: int) -> np.ndarray:
        """
        Train on all series in X, then return their latent representations.

        Parameters
        ----------
        X : (n_samples, n_channels, n_timepoints)
        w : target length after reduction

        Returns
        -------
        np.ndarray of shape (n_samples, n_channels, w)
        """
        n_samples, n_channels, N = X.shape
        if w >= N:
            raise ValueError("w must be smaller than series length.")

        device = _get_device()

        # Flatten all series into a single batch: (n_samples*n_channels, 1, N)
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_tensor = torch.from_numpy(X_flat).unsqueeze(1).to(device)

        self._model = _ConvAE(N, w, self.n_channels).to(device)
        self._w = w
        _train_autoencoder_batched(self._model, X_tensor, self.epochs, self.lr, self.batch_size)

        return self._encode(X, device)

    def transform(self, X: np.ndarray, w: int) -> np.ndarray:
        """
        Apply the already-trained encoder to X.

        Parameters
        ----------
        X : (n_samples, n_channels, n_timepoints)
        w : must match the w used in fit_transform

        Returns
        -------
        np.ndarray of shape (n_samples, n_channels, w)
        """
        if self._model is None:
            raise RuntimeError("Call fit_transform before transform.")
        if w != self._w:
            raise ValueError(f"w={w} does not match fitted w={self._w}.")

        device = _get_device()
        return self._encode(X, device)

    def _encode(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        n_samples, n_channels, N = X.shape
        w = self._w
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_tensor = torch.from_numpy(X_flat).unsqueeze(1).to(device)

        self._model.eval()
        with torch.no_grad():
            _, latent = self._model(X_tensor)
        # latent: (n_samples*n_channels, 1, w)
        reduced_flat = latent.squeeze(1).cpu().numpy()  # (n_samples*n_channels, w)

        reduced_flat = _sign_correct_batch(reduced_flat, X_flat, w)
        return reduced_flat.reshape(n_samples, n_channels, w)
