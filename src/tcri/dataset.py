"""Patch dataset for the RI vision model.

Southern-hemisphere storms are mirrored north-south so their circulation sense
matches the northern ones -- a fixed transform, not an augmentation. Because
that fixes the rotation sense, augmentation is 90-degree rotations only; a flip
would undo it. Rotation does discard shear-relative orientation, which is
physically meaningful, and whether that costs skill is one of the ablations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Typical clear-sky ocean brightness temperature, used to fill the small number
# of missing pixels that survive the 5% per-frame rejection threshold.
FILL_K = 285.0
# Fixed normalisation, so a model trained on one basin can be applied to another
# without carrying a scaler around. Roughly centres the 190-300 K range.
BT_MEAN_K = 250.0
BT_STD_K = 30.0


class PatchDataset(Dataset):
    """Patch sequences for a set of case rows.

    Pass a path rather than an open array when using DataLoader workers: a
    memmap cannot be pickled into a spawned process, so each worker opens the
    file itself.
    """

    def __init__(self, array, index: pd.DataFrame, rows: np.ndarray,
                 augment: bool = False, n_frames: int | None = None):
        if isinstance(array, (str, Path)):
            self.path, self._array = Path(array), None
        else:
            self.path, self._array = None, array
        self.rows = np.asarray(rows)
        self.augment = augment
        self.n_frames = n_frames
        idx = index.set_index("row")
        self.y = idx.loc[self.rows, "ri"].to_numpy(dtype=np.float32)
        self.lat = idx.loc[self.rows, "LAT"].to_numpy(dtype=np.float32)

    @property
    def array(self):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def __getstate__(self):
        # Never ship the memmap to a worker process.
        s = self.__dict__.copy()
        if s.get("path") is not None:
            s["_array"] = None
        return s

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        x = np.asarray(self.array[r], dtype=np.float32)     # (frames, H, W)
        if self.n_frames is not None:
            x = x[-self.n_frames:]                          # keep the latest
        x = np.nan_to_num(x, nan=FILL_K)
        x = (x - BT_MEAN_K) / BT_STD_K

        if self.lat[i] < 0:                                 # southern hemisphere
            x = x[:, ::-1, :]
        if self.augment:
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k, axes=(1, 2))
        x = np.ascontiguousarray(x)
        return torch.from_numpy(x), torch.tensor(self.y[i]), r


def patch_path(processed_dir, tag: str) -> Path:
    return processed_dir / "patches" / f"patches_{tag}.npy"


def load_patches(processed_dir, tag: str):
    d = processed_dir / "patches"
    arr = np.load(d / f"patches_{tag}.npy", mmap_mode="r")
    idx = pd.read_csv(d / f"index_{tag}.csv", converters={"BASIN": str},
                      parse_dates=["ISO_TIME"])
    complete = idx["frames_ok"] == arr.shape[1]
    return arr, idx, idx.loc[complete, "row"].to_numpy()
