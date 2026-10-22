"""Load data/column_dataset.h5 into tensors for the FNO and the LSTM.

Grid: 20 depths (0.5 ... 10 m) x 1000 time steps (0.05 s, 0 ... 49.95 s).
Inputs (same for both models), per time step:
    acc   base acceleration (g), anti-alias filtered to the 0.05 s grid
    vel   base velocity (m/s)
    arias cumulative Arias intensity (m/s): a running measure of shaking energy
    dr    relative density (constant in time)
    logk  log10 permeability (constant in time)
Target: excess pore-pressure ratio ru(depth, t), clipped to [-1, 1.5].
"""
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from scipy import signal

from .dataset import DT_IN, H5_PATH, OUT_EVERY

G = 9.81
NT = 1000
FEATURES = ("acc", "vel", "arias", "dr", "logk")


@dataclass
class Split:
    x: torch.Tensor          # [n, NT, 5]   time series inputs
    y: torch.Tensor          # [n, NZ, NT]  ru field
    meta: list               # per-case attributes


def _features(acc_full, dr, perm):
    acc = signal.resample_poly(acc_full, 1, OUT_EVERY)[:NT] / G
    vel = (np.cumsum(acc_full) * DT_IN)[::OUT_EVERY][:NT]
    arias = (np.pi / (2 * G) * np.cumsum(acc_full ** 2) * DT_IN)[::OUT_EVERY][:NT]
    return np.stack([acc, vel, arias, np.full(NT, dr), np.full(NT, np.log10(perm))], axis=1)


def load(path=H5_PATH):
    """Returns {"train"/"val"/"test": Split}, depth array, and rejected case ids."""
    out = {k: ([], [], []) for k in ("train", "val", "test")}
    rejected = []
    with h5py.File(path, "r") as f:
        depth = f["depth"][:]
        for name in sorted(k for k in f if k.startswith("case_")):
            g = f[name]
            a = dict(g.attrs)
            ru = g["ru"][:NT] if "ru" in g else None
            if not a["ok"] or ru is None or not np.isfinite(ru).all():
                rejected.append(int(a["case_id"]))
                continue
            xs, ys, ms = out[a["split"]]
            xs.append(_features(g["acc"][:].astype(np.float64), a["dr"], a["perm"]))
            ys.append(np.clip(ru.T, -1.0, 1.5))
            ms.append(a)
    splits = {k: Split(torch.tensor(np.array(x), dtype=torch.float32),
                       torch.tensor(np.array(y), dtype=torch.float32), m)
              for k, (x, y, m) in out.items() if x}
    return splits, depth, rejected


class Normaliser:
    """Per-feature mean/std from the training set."""

    def __init__(self, x):
        flat = x.reshape(-1, x.shape[-1])
        self.mean, self.std = flat.mean(0), flat.std(0).clamp_min(1e-6)

    def __call__(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
