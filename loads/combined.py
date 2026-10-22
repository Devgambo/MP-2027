"""Combined wave + earthquake load histories: the part 4 "one function call".

    from loads import generate_loads
    cases = generate_loads(100, seed=0)                     # synthetic quakes until PEER exists
    cases = generate_loads(100, seed=0, records="peer")     # data/peer/*.AT2

Each case is a storm already in progress when the earthquake arrives:

    t = 0 ............ t_eq .................. t_eq + quake ........ t_end
    waves ramp up      earthquake starts       earthquake over     post-shaking
    (waves continue the whole time)

Sampled per case: Hs, Tp (steepness-limited), target PGA, which record, and the gap t_eq.
Parameters are drawn with Latin hypercube sampling so a small set still covers the space.
"""
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

from .earthquakes import PEER_DIR, G, load_peer, process, scale_to_pga, split_by_event, synthetic_library
from .waves import wave_loads

DEFAULT_RANGES = dict(
    hs=(1.0, 8.0),          # significant wave height, m
    tp=(6.0, 14.0),         # peak period, s
    pga_g=(0.05, 0.40),     # target peak ground acceleration, g
    t_eq=(30.0, 90.0),      # waves-only time before the quake, s
)
MAX_STEEPNESS = 1 / 16      # Hs / L_p limit for a physically possible sea state
MAX_SCALE = 4.0             # do not scale a record more than 4x (or below 1/4)


@dataclass
class LoadCase:
    t: np.ndarray
    acc: np.ndarray          # base acceleration, m/s^2
    force: np.ndarray        # wave force on the pile at mudline, kN
    moment: np.ndarray       # wave overturning moment about mudline, kN*m
    eta: np.ndarray          # sea-surface elevation, m
    params: dict = field(default_factory=dict)


def _lhs(n, ranges, seed):
    keys = list(ranges)
    u = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    lo = np.array([ranges[k][0] for k in keys])
    hi = np.array([ranges[k][1] for k in keys])
    x = qmc.scale(u, lo, hi)
    return [dict(zip(keys, row)) for row in x]


def _limit_steepness(hs, tp):
    """Raise Tp if needed so that Hs / (g Tp^2 / 2 pi) <= MAX_STEEPNESS."""
    tp_min = np.sqrt(hs * 2 * np.pi / (G * MAX_STEEPNESS))
    return max(tp, tp_min)


def _record_pool(records, dt, seed):
    if records == "peer":
        raw = load_peer(PEER_DIR)
        if not raw:
            raise FileNotFoundError(f"no .AT2 files in {PEER_DIR}; download from PEER NGA-West2")
        return [process(r, dt) for r in raw]
    if records == "synthetic":
        return [process(r, dt) for r in synthetic_library(40, seed=seed + 1, dt=dt)]
    return list(records)                     # already-processed Record objects


def generate_loads(n, seed=0, records="synthetic", split=None, dt=0.01, t_post=20.0,
                   ranges=None, depth=30.0, diameter=8.0):
    """Return n LoadCase objects.

    records: "synthetic", "peer", or a list of processed Record objects.
    split:   None (use all records) or "train"/"val"/"test" (whole events held out, fixed seed).
    """
    ranges = {**DEFAULT_RANGES, **(ranges or {})}
    pool = _record_pool(records, dt, seed=0 if split else seed)
    if split:
        pool = split_by_event(pool)[split]
    rng = np.random.default_rng(seed)
    cases = []
    for p in _lhs(n, ranges, seed):
        tp = _limit_steepness(p["hs"], p["tp"])
        # choose a record whose scale factor to this PGA stays within limits
        # (synthetic amplitudes are arbitrary, so the limit only applies to real records)
        ok = [r for r in pool if r.source != "peer" or 1 / MAX_SCALE <= p["pga_g"] * G / r.pga <= MAX_SCALE]
        rec = ok[rng.integers(len(ok))] if ok else min(pool, key=lambda r: abs(np.log(p["pga_g"] * G / r.pga)))
        rec = scale_to_pga(rec, p["pga_g"])

        n_pre = int(round(p["t_eq"] / dt))
        nt = n_pre + len(rec.acc) + int(round(t_post / dt))
        t = np.arange(nt) * dt
        acc = np.zeros(nt)
        acc[n_pre:n_pre + len(rec.acc)] = rec.acc
        w = wave_loads(t, p["hs"], tp, rng, depth=depth, diameter=diameter)
        params = dict(hs=p["hs"], tp=tp, tp_sampled=p["tp"], pga_g=p["pga_g"], t_eq=n_pre * dt,
                      record=rec.name, event=rec.event, source=rec.source,
                      scale_factor=rec.meta["scale_factor"], d5_95=rec.meta["d5_95"],
                      kc=w["kc"], depth=depth, diameter=diameter, dt=dt)
        cases.append(LoadCase(t=t, acc=acc, force=w["force"], moment=w["moment"], eta=w["eta"],
                              params=params))
    return cases


def save_h5(cases, path):
    """One group per case: datasets t, acc, force, moment, eta; parameters as attributes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["units"] = "t s, acc m/s^2, force kN, moment kN*m, eta m"
        for i, c in enumerate(cases):
            g = f.create_group(f"case_{i:05d}")
            for k in ("t", "acc", "force", "moment", "eta"):
                g.create_dataset(k, data=getattr(c, k).astype(np.float32), compression="gzip")
            for k, v in c.params.items():
                g.attrs[k] = v


def load_h5(path):
    cases = []
    with h5py.File(path, "r") as f:
        for name in sorted(f):
            g = f[name]
            cases.append(LoadCase(**{k: g[k][()].astype(float) for k in ("t", "acc", "force", "moment", "eta")},
                                  params=dict(g.attrs)))
    return cases
