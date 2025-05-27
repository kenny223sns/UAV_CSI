# utils.py
"""Utility helpers: dataset loader, TA+AWGN augmentation, simple logger."""

from pathlib import Path
import csv
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "CSIDataset",
    "add_ta_and_awgn",
]

# -----------------------------------------------------------------------------
# Data augmentation helpers
# -----------------------------------------------------------------------------

def add_ta_and_awgn(
    H: np.ndarray,
    max_shift_bin: int = 5,
    snr_db: float = 25.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply Time‑Advance (TA) + AWGN on complex CSI tensor.

    Args:
        H: complex ndarray with shape (C_tx, H_rx, W_subc)
        max_shift_bin: maximum ± shift in delay bins (32 ns per bin)
        snr_db: average SNR in dB
        rng: optional NumPy RNG (for determinism)
    Returns:
        Complex ndarray with same shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    C, H_rx, W = H.shape
    # ---- TA  --------------------------------------------------------------
    shift = rng.integers(-max_shift_bin, max_shift_bin + 1)
    k = np.arange(W)
    phase = np.exp(-1j * 2 * np.pi * shift * k / W)  # (W,)
    H = H * phase  # broadcast to last dim

    # ---- AWGN ------------------------------------------------------------
    power = np.mean(np.abs(H) ** 2)
    sigma = np.sqrt(power / 10 ** (snr_db / 10))
    noise = sigma / np.sqrt(2) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    return H + noise

# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------

class CSIDataset(Dataset):
    """CSI fingerprint dataset reading .npy tensors + labels.csv.

    Each .npy must be shape (C, H, W) with float32 (real magnitude / phase
    already prepared). labels.csv columns: file, sector, x, y, los
    """

    def __init__(
        self,
        root: str | Path,
        csv_path: str | Path,
        supervised_only: bool = False,
        normalize: bool = True,
        aug_flip_delay: bool = False,
    ) -> None:
        self.root = Path(root)
        self.normalize = normalize
        self.aug_flip = aug_flip_delay
        self.items: List[Tuple[Path, Optional[np.ndarray]]] = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                p = self.root / f"sector{row['sector']}" / row["file"]
                if not p.exists():
                    continue
                if row["x"] != "":
                    label = np.array([float(row["x"]), float(row["y"])], dtype=np.float32)
                else:
                    label = None
                if supervised_only and label is None:
                    continue
                self.items.append((p, label))

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        x = np.load(path).astype(np.float32)  # (C,H,W)
        if self.aug_flip and random.random() < 0.5:
            x = x[..., ::-1]  # flip delay axis
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-6)
        x = torch.from_numpy(x)
        if label is None:
            return x, None
        return x, torch.from_numpy(label)
