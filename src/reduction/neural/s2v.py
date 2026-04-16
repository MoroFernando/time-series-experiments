"""Series2Vec (S2V) for self-supervised time series dimensionality reduction."""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .base import get_device, sign_correct_batch

class S2VEncoder(nn.Module):
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
        self.features = nn.Sequential(
            nn.Conv1d(1, emb_size, kernel_size=8, padding='same'),
            nn.BatchNorm1d(emb_size),
            nn.GELU(),
            nn.Conv1d(emb_size, emb_size, kernel_size=5, padding='same'),
            nn.BatchNorm1d(emb_size),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(latent_dim)
        self.collapse = nn.Conv1d(emb_size, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the S2V encoder."""
        x = self.features(x)
        x = self.pool(x)
        x = self.collapse(x)
        return x.squeeze(1) # (B, w)


class S2VTrainer(nn.Module):
    """
    Full Series2Vec model used during pre-training only.

    Matches the paper's Pretrain_forward (Section 3.2 and source code):
      • time_enc / freq_enc  (S2VEncoder): (B, 1, N) → (B, w)
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
        self.time_enc = S2VEncoder(latent_dim, emb_size)
        self.freq_enc = S2VEncoder(latent_dim, emb_size)

        # Time branch attention
        self.proj_t = nn.Linear(latent_dim, self.D_MODEL)
        self.attn_t = nn.MultiheadAttention(self.D_MODEL, self.N_HEADS, batch_first=False)
        self.norm1_t = nn.LayerNorm(self.D_MODEL)
        self.ff_t = nn.Linear(self.D_MODEL, self.D_MODEL)
        self.norm2_t = nn.LayerNorm(self.D_MODEL)

        # Freq branch attention
        self.proj_f = nn.Linear(latent_dim, self.D_MODEL)
        self.attn_f = nn.MultiheadAttention(self.D_MODEL, self.N_HEADS, batch_first=False)
        self.norm1_f = nn.LayerNorm(self.D_MODEL)
        self.ff_f = nn.Linear(self.D_MODEL, self.D_MODEL)
        self.norm2_f = nn.LayerNorm(self.D_MODEL)

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
        """Process one branch (time or frequency)."""
        z = enc(x)
        z_proj = proj(z).unsqueeze(0) # (1, B, D_MODEL)
        attn_out, _ = attn(z_proj, z_proj, z_proj)
        z = norm1(z_proj + attn_out)
        ff_out = ff(z)
        z = norm2(z + ff_out)
        return z.squeeze(0) # (B, D_MODEL)

    def forward(
        self,
        x_t: torch.Tensor,   # (B, 1, N)
        x_f: torch.Tensor,   # (B, 1, N)
    ):
        """
        Forward pass for pre-training.
        Returns pairwise distance matrices for both domains.
        """
        z_t = self._branch(self.time_enc, self.proj_t, self.attn_t, self.norm1_t, self.ff_t, self.norm2_t, x_t)
        z_f = self._branch(self.freq_enc, self.proj_f, self.attn_f, self.norm1_f, self.ff_f, self.norm2_f, x_f)
        
        d_rep_t = torch.cdist(z_t, z_t)
        d_rep_f = torch.cdist(z_f, z_f)
        return d_rep_t, d_rep_f


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
    criterion = nn.SmoothL1Loss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_t_batch, x_f_batch in loader:
            B = x_t_batch.size(0)
            if B < 2: continue

            optimizer.zero_grad()
            
            # 1. Representation space distances
            d_rep_t, d_rep_f = model(x_t_batch, x_f_batch)

            # 2. Input space distances
            d_in_t = torch.cdist(x_t_batch.view(B, -1), x_t_batch.view(B, -1))
            d_in_f = torch.cdist(x_f_batch.view(B, -1), x_f_batch.view(B, -1))

            # 3. Get lower-triangular pairs
            tril_indices = torch.tril_indices(B, B, offset=-1)
            d_rep_t_pairs = d_rep_t[tril_indices[0], tril_indices[1]]
            d_rep_f_pairs = d_rep_f[tril_indices[0], tril_indices[1]]
            d_in_t_pairs  = d_in_t[tril_indices[0], tril_indices[1]]
            d_in_f_pairs  = d_in_f[tril_indices[0], tril_indices[1]]

            # 4. Normalise
            d_rep_t_norm = _s2v_minmax(d_rep_t_pairs)
            d_rep_f_norm = _s2v_minmax(d_rep_f_pairs)
            d_in_t_norm  = _s2v_minmax(d_in_t_pairs)
            d_in_f_norm  = _s2v_minmax(d_in_f_pairs)

            # 5. Loss
            loss = criterion(d_rep_t_norm, d_in_t_norm) + \
                   criterion(d_rep_f_norm, d_in_f_norm)
            
            loss.backward()
            # 6. Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # Progress bar
        pct = (epoch + 1) / epochs
        bar = "█" * int(pct * 30) + "-" * (30 - int(pct * 30))
        sys.stdout.write(f"\r  [training] [{bar}] {pct:>5.1%} loss={epoch_loss/len(loader):.6f}")
        sys.stdout.flush()
    print()


class Series2VecReducer:
    """
    Series2Vec — global self-supervised dimensionality reduction.

    Trains a dual-encoder (temporal + spectral domains) with a pairwise
    similarity-preserving loss. After training, only the temporal encoder
    is used for inference: it maps each univariate channel from N timepoints
    to w timepoints.

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
        emb_size:   int   = 16,     # Paper: 16
        epochs:     int   = 100,    # Paper: 100
        lr:         float = 0.001,  # Paper: 0.001
        batch_size: int   = 64,     # Paper: 64
    ):
        self.emb_size   = emb_size
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self._model: S2VTrainer | None = None
        self._w: int | None = None

    def fit_transform(self, X: np.ndarray, w: int) -> np.ndarray:
        """Fits the S2V model and transforms the data."""
        n_samples, n_channels, N = X.shape
        if w >= N:
            raise ValueError("Target length w must be smaller than series length N.")

        device = get_device()
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_t = torch.from_numpy(X_flat).unsqueeze(1).to(device)
        X_f = self._fft_magnitudes(X_flat).unsqueeze(1).to(device)

        self._model = S2VTrainer(w, self.emb_size).to(device)
        self._w = w
        _train_series2vec(self._model, X_t, X_f, self.epochs, self.lr, self.batch_size)
        
        return self._encode(X, device)

    def transform(self, X: np.ndarray, w: int) -> np.ndarray:
        """Transforms data using the fitted temporal encoder."""
        if self._model is None:
            raise RuntimeError("Must call fit_transform before transform.")
        if w != self._w:
            raise ValueError(f"Transform w ({w}) must match fit_transform w ({self._w}).")
        return self._encode(X, get_device())

    @staticmethod
    def _fft_magnitudes(series_batch: np.ndarray) -> torch.Tensor:
        """Compute FFT magnitudes for a batch of series."""
        ffts = np.fft.fft(series_batch, axis=1)
        mags = np.abs(ffts)
        return torch.from_numpy(mags.astype(np.float32))

    def _encode(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        """Encodes data using the temporal encoder."""
        n_samples, n_channels, N = X.shape
        X_flat = X.reshape(-1, N).astype(np.float32)
        X_tensor = torch.from_numpy(X_flat).unsqueeze(1).to(device)

        self._model.eval()
        with torch.no_grad():
            reduced_flat = self._model.time_enc(X_tensor).cpu().numpy()

        reduced_flat = sign_correct_batch(reduced_flat, X_flat, self._w)
        return reduced_flat.reshape(n_samples, n_channels, self._w)
