"""Shared utilities and base classes for neural dimensionality reduction."""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_device() -> torch.device:
    """Returns the available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clamp(value: int, lo: int, hi: int) -> int:
    """Integer clamping helper."""
    return max(lo, min(value, hi))


def sign_correct(reduced: np.ndarray, series: np.ndarray, w: int) -> np.ndarray:
    """Flip sign if the reduced series anti-correlates with the subsampled original."""
    if len(series) == 0 or len(reduced) == 0:
        return reduced
    subsampled = series[np.linspace(0, len(series) - 1, w, dtype=int)]
    if np.std(reduced) > 1e-9 and np.std(subsampled) > 1e-9:
        try:
            if np.corrcoef(reduced, subsampled)[0, 1] < 0:
                return -reduced
        except ValueError: # Can happen if one is all constant
            pass
    return reduced


def sign_correct_batch(reduced: np.ndarray, originals: np.ndarray, w: int) -> np.ndarray:
    """Apply per-series sign correction to a batch of reduced series."""
    if originals.shape[1] == 0:
        return reduced
    idx = np.linspace(0, originals.shape[1] - 1, w, dtype=int)
    result = reduced.copy()
    for i, (r, s) in enumerate(zip(reduced, originals)):
        sub = s[idx]
        if np.std(r) > 1e-9 and np.std(sub) > 1e-9:
            try:
                if np.corrcoef(r, sub)[0, 1] < 0:
                    result[i] = -r
            except ValueError:
                pass
    return result


def train_autoencoder(model: nn.Module, x: torch.Tensor, epochs: int, lr: float) -> None:
    """Train an autoencoder on a single batch (SIT mode)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon, _ = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()


def _print_train_progress(epoch: int, total: int, loss: float, bar_len: int = 30) -> None:
    """Overwrite the current line with an epoch-level training progress bar."""
    pct   = (epoch + 1) / total
    filled = int(pct * bar_len)
    bar   = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(
        f"\r  [training] [{bar}] {pct:>5.1%}  "
        f"epoch {epoch + 1:>{len(str(total))}}/{total}  loss={loss:.6f}"
    )
    sys.stdout.flush()


def train_autoencoder_batched(
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
        _print_train_progress(epoch, epochs, epoch_loss / len(loader))
    print()  # newline after the progress bar


class GlobalReducer:
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
    _model: nn.Module | None = None
    _w: int | None = None

    def fit_transform(self, X: np.ndarray, w: int) -> np.ndarray:
        """Fits the model on the training data and transforms it."""
        n_samples, n_channels, N = X.shape
        if w >= N:
            raise ValueError("Target length w must be smaller than series length N.")

        device = get_device()
        # Reshape to (n_samples * n_channels, N) for batch training
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_tensor = self._to_tensor(X_flat, device)

        self._model = self._make_model(N, w).to(device)
        self._w = w
        train_autoencoder_batched(
            self._model, X_tensor, self.epochs, self.lr, self.batch_size
        )
        return self._encode(X, device)

    def transform(self, X: np.ndarray, w: int) -> np.ndarray:
        """Transforms the data using the fitted model."""
        if self._model is None:
            raise RuntimeError("Must call fit_transform before transform.")
        if w != self._w:
            raise ValueError(f"Transform w ({w}) must match fit_transform w ({self._w}).")
        return self._encode(X, get_device())

    def _encode(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        """Encodes the data using the model's encoder."""
        n_samples, n_channels, N = X.shape
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_tensor = self._to_tensor(X_flat, device)

        self._model.eval()
        with torch.no_grad():
            _, latent = self._model(X_tensor)

        # Works for both (B, w) [dense] and (B, 1, w) [conv/tcn]
        reduced_flat = latent.cpu().numpy().reshape(-1, self._w)
        reduced_flat = sign_correct_batch(reduced_flat, X_flat, self._w)
        return reduced_flat.reshape(n_samples, n_channels, self._w)

    def _to_tensor(self, X_flat: np.ndarray, device: torch.device) -> torch.Tensor:
        """Converts numpy array to a tensor, adding a channel dimension if needed."""
        t = torch.from_numpy(X_flat).to(device)
        return t.unsqueeze(1) if self._has_channel_dim else t

    def _make_model(self, N: int, w: int) -> nn.Module:
        """Factory method for creating the neural network model."""
        raise NotImplementedError
