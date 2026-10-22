"""Earthquake ground motions: PEER NGA-West2 records (.AT2) and a synthetic fallback.

Put downloaded PEER files (unzipped .AT2) in data/peer/. Every record is:
  read -> resampled to a common dt -> tapered and high-pass filtered (baseline drift)
  -> trimmed to its strong-shaking part (Arias 5-95% plus padding) -> scaled to a target PGA.

The synthetic generator (filtered white noise, Kanai-Tajimi + Clough-Penzien, with a
time envelope) lets the pipeline run before real records exist. It is labelled
source="synthetic" everywhere and must not be used for the final dataset.

Accelerations are in m/s^2 unless a name says _g.
"""
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

G = 9.81
PEER_DIR = Path(__file__).resolve().parents[1] / "data" / "peer"


@dataclass
class Record:
    acc: np.ndarray            # m/s^2
    dt: float
    name: str                  # file stem, e.g. RSN6_IMPVALL.I_I-ELC180
    event: str = ""            # earthquake name, used to hold out whole events
    source: str = "peer"       # "peer" or "synthetic"
    meta: dict = field(default_factory=dict)

    @property
    def pga(self):
        return float(np.abs(self.acc).max())

    @property
    def t(self):
        return np.arange(len(self.acc)) * self.dt


# ---------------------------------------------------------------- PEER files
_HDR = re.compile(r"NPTS\s*=\s*(\d+)\s*,?\s*DT\s*=\s*([0-9.Ee+-]+)", re.I)


def read_at2(path):
    """Parse a PEER NGA .AT2 file (4 header lines, then accelerations in g)."""
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines[:10]):
        m = _HDR.search(line)
        if m:
            npts, dt = int(m.group(1)), float(m.group(2))
            break
    else:
        raise ValueError(f"{path.name}: no 'NPTS=..., DT=...' header line")
    vals = np.array(" ".join(lines[i + 1:]).split(), dtype=float)[:npts]
    event = lines[1].split(",")[0].strip() if len(lines) > 1 else ""
    rsn = re.match(r"RSN(\d+)", path.stem, re.I)
    return Record(acc=vals * G, dt=dt, name=path.stem, event=event,
                  meta={"rsn": int(rsn.group(1)) if rsn else None, "header": lines[1].strip()})


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_peer_metadata(directory=PEER_DIR):
    """RSN -> {magnitude, rrup, vs30, luf, mechanism, year} from every PEER search-report CSV."""
    out = {}
    for f in sorted(Path(directory).glob("*.csv")):
        lines = f.read_text(errors="replace").splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith("Result ID")), None)
        if start is None:
            continue
        rows = csv.reader(lines[start:], skipinitialspace=True)
        cols = [c.strip() for c in next(rows)]
        for row in rows:
            if not row or not row[0].strip().isdigit():
                break
            r = dict(zip(cols, (x.strip() for x in row)))
            out[int(r["Record Sequence Number"])] = dict(
                magnitude=_num(r.get("Magnitude")), rrup=_num(r.get("Rrup (km)")),
                vs30=_num(r.get("Vs30 (m/sec)")), luf=_num(r.get("Lowest Useable Frequency (Hz)")),
                mechanism=r.get("Mechanism", ""), year=_num(r.get("Year")))
    return out


def load_peer(directory=PEER_DIR, horizontal_only=True):
    """All .AT2 files in a folder. Vertical components (UP/DWN/-V) are skipped by default.
    Magnitude, distance etc. are attached from the search-report CSVs in the same folder."""
    files = sorted(Path(directory).glob("*.AT2")) + sorted(Path(directory).glob("*.at2"))
    table = read_peer_metadata(directory)
    recs = []
    for f in dict.fromkeys(files):
        if horizontal_only and re.search(r"(UP|DWN|-V)\.AT2$", f.name, re.I):
            continue
        r = read_at2(f)
        r.meta.update(table.get(r.meta["rsn"], {}))
        recs.append(r)
    return recs


# ---------------------------------------------------------------- processing
def resample(acc, dt, dt_new):
    if np.isclose(dt, dt_new):
        return acc.copy()
    ratio = dt / dt_new
    up, down = (int(round(ratio)), 1) if ratio >= 1 else (1, int(round(1 / ratio)))
    if np.isclose(up / down, ratio, rtol=1e-6):
        return signal.resample_poly(acc, up, down)        # includes anti-alias filter
    t_old = np.arange(len(acc)) * dt
    if dt_new > dt:                                        # anti-alias before interpolation
        sos = signal.butter(8, 0.8 * 0.5 / dt_new, fs=1 / dt, output="sos")
        acc = signal.sosfiltfilt(sos, acc)
    return np.interp(np.arange(0, t_old[-1], dt_new), t_old, acc)


def baseline_correct(acc, dt, f_hp=0.1, taper_s=1.0):
    """Demean, cosine taper both ends, zero-pad, zero-phase 4th-order Butterworth high-pass.

    Zero padding of 1.5 * order / f_hp (Boore 2005) lets the acausal filter settle, so the
    velocity and displacement obtained by integration do not drift.
    """
    a = acc - acc.mean()
    n = min(int(taper_s / dt), len(a) // 4)
    if n > 0:
        w = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
        a[:n] *= w
        a[-n:] *= w[::-1]
    npad = int(1.5 * 4 / f_hp / dt / 2)
    a = np.pad(a, npad)
    sos = signal.butter(4, f_hp, btype="highpass", fs=1 / dt, output="sos")
    a = signal.sosfiltfilt(sos, a, padtype=None)[npad:-npad]
    return _polynomial_baseline(a, dt)


def _polynomial_baseline(acc, dt, order=6):
    """Least-squares fit d_fit = sum_{j=2..order} c_j t^j to the doubly integrated displacement
    and subtract d_fit'' from the acceleration. No t^0, t^1 terms, so d(0) = v(0) = 0 stay true."""
    t = np.arange(len(acc)) * dt
    d = np.cumsum(np.cumsum(acc) * dt) * dt
    ts = t / t[-1]                                     # scaled time for conditioning
    X = np.stack([ts ** j for j in range(2, order + 1)], axis=1)
    c, *_ = np.linalg.lstsq(X, d, rcond=None)
    d2 = sum(c[j - 2] * j * (j - 1) * ts ** (j - 2) for j in range(2, order + 1)) / t[-1] ** 2
    return acc - d2


def arias(acc, dt):
    """Cumulative Arias intensity (m/s)."""
    return np.pi / (2 * G) * np.cumsum(acc ** 2) * dt


def significant_window(acc, dt, lo=0.05, hi=0.95):
    ia = arias(acc, dt)
    ia /= ia[-1]
    return np.argmax(ia >= lo) * dt, np.argmax(ia >= hi) * dt


def trim(acc, dt, pad_before=2.0, pad_after=5.0, taper_s=0.5):
    """Keep the Arias 5-95% window plus padding; taper the new ends."""
    t5, t95 = significant_window(acc, dt)
    i0 = max(0, int((t5 - pad_before) / dt))
    i1 = min(len(acc), int((t95 + pad_after) / dt))
    a = acc[i0:i1].copy()
    n = min(int(taper_s / dt), len(a) // 4)
    w = 0.5 * (1 - np.cos(np.pi * np.arange(n) / n))
    a[:n] *= w
    a[-n:] *= w[::-1]
    return a


def process(rec, dt=0.01, f_hp=0.1, do_trim=True):
    a = resample(rec.acc, rec.dt, dt)
    if do_trim:                      # trim first: filtering after trimming keeps the ends clean
        a = trim(a, dt)
    a = baseline_correct(a, dt, f_hp)
    t5, t95 = significant_window(a, dt)
    meta = dict(rec.meta, pga_orig_g=rec.pga / G, d5_95=t95 - t5, arias=float(arias(a, dt)[-1]))
    return Record(acc=a, dt=dt, name=rec.name, event=rec.event, source=rec.source, meta=meta)


def scale_to_pga(rec, pga_g):
    f = pga_g * G / rec.pga
    return Record(acc=rec.acc * f, dt=rec.dt, name=rec.name, event=rec.event, source=rec.source,
                  meta=dict(rec.meta, scale_factor=f, pga_target_g=pga_g))


# ---------------------------------------------------------------- spectra
def response_spectrum(acc, dt, periods, zeta=0.05):
    """Pseudo-spectral acceleration Sa(T) [same units as acc], exact piecewise-linear
    recurrence (Nigam & Jennings 1969), vectorised over periods."""
    periods = np.atleast_1d(periods)
    w = 2 * np.pi / periods
    wd = w * np.sqrt(1 - zeta ** 2)
    e = np.exp(-zeta * w * dt)
    s, c = np.sin(wd * dt), np.cos(wd * dt)
    a11 = e * (zeta * w / wd * s + c)
    a12 = e * s / wd
    a21 = -e * w / np.sqrt(1 - zeta ** 2) * s
    a22 = e * (c - zeta * w / wd * s)
    t1 = (2 * zeta ** 2 - 1) / (w ** 2 * dt)
    t2 = 2 * zeta / (w ** 3 * dt)
    b11 = e * ((t1 + zeta / w) * s / wd + (t2 + 1 / w ** 2) * c) - t2
    b12 = -e * (t1 * s / wd + t2 * c) - 1 / w ** 2 + t2
    b21 = e * ((t1 + zeta / w) * (c - zeta * w / wd * s) - (t2 + 1 / w ** 2) * (wd * s + zeta * w * c)) + 1 / (w ** 2 * dt)
    b22 = -e * (t1 * (c - zeta * w / wd * s) - t2 * (wd * s + zeta * w * c)) - 1 / (w ** 2 * dt)
    x = np.zeros_like(w)
    v = np.zeros_like(w)
    umax = np.zeros_like(w)
    p = -np.asarray(acc)
    for i in range(len(p) - 1):
        x, v = (a11 * x + a12 * v + b11 * p[i] + b12 * p[i + 1],
                a21 * x + a22 * v + b21 * p[i] + b22 * p[i + 1])
        np.maximum(umax, np.abs(x), out=umax)
    return w ** 2 * umax


# ---------------------------------------------------------------- synthetic fallback
def synthetic_record(rng, dt=0.01, duration=None, fg=None, zeta_g=None, name=None):
    """Stochastic ground motion: white noise -> Kanai-Tajimi (site) -> Clough-Penzien
    (removes unrealistic low frequencies) -> Jennings-type envelope -> baseline correction."""
    duration = rng.uniform(10, 30) if duration is None else duration        # strong part, s
    fg = rng.uniform(1.5, 5.0) if fg is None else fg                         # site freq, Hz
    zeta_g = rng.uniform(0.3, 0.7) if zeta_g is None else zeta_g
    t_rise, t_decay = 0.15 * duration + 1.0, 0.6 * duration
    T = t_rise + duration + t_decay
    t = np.arange(0, T, dt)
    env = np.where(t < t_rise, (t / t_rise) ** 2,
                   np.where(t < t_rise + duration, 1.0,
                            np.exp(-3.0 * (t - t_rise - duration) / t_decay)))
    wg, wf = 2 * np.pi * fg, 2 * np.pi * fg / 10
    # Kanai-Tajimi and Clough-Penzien as continuous filters, discretised (bilinear)
    kt = signal.cont2discrete(([2 * zeta_g * wg, wg ** 2], [1, 2 * zeta_g * wg, wg ** 2]), dt, "bilinear")
    cp = signal.cont2discrete(([1, 0, 0], [1, 2 * 0.6 * wf, wf ** 2]), dt, "bilinear")
    a = rng.standard_normal(len(t)) * env
    a = signal.lfilter(np.squeeze(kt[0]), kt[1], a)
    a = signal.lfilter(np.squeeze(cp[0]), cp[1], a)
    a = baseline_correct(a, dt, 0.1)
    a *= 0.3 * G / np.abs(a).max()                        # nominal 0.3 g; rescaled later
    name = name or f"SYN_fg{fg:.1f}_d{duration:.0f}_{rng.integers(1e6):06d}"
    return Record(acc=a, dt=dt, name=name, event=name, source="synthetic",
                  meta=dict(fg=fg, zeta_g=zeta_g, duration=duration))


def synthetic_library(n, seed=0, dt=0.01):
    """n synthetic records, each its own 'event'."""
    rng = np.random.default_rng(seed)
    return [synthetic_record(rng, dt) for _ in range(n)]


# ---------------------------------------------------------------- train/test split by event
MAG_BINS = (6.5, 7.0)          # magnitude bands: < 6.5, 6.5-7, >= 7


def event_group(event):
    """Earthquake sequence an event belongs to: 'Mammoth Lakes-03' -> 'mammoth lakes'.
    Aftershocks of one sequence share stations and source, so they are held out together."""
    g = re.sub(r"[-\s]*\d+$", "", event.strip().lower())
    return g.replace("fruili", "friuli")          # PEER spells Friuli-03 as 'Fruili'


def split_by_event(records, test_frac=0.2, val_frac=0.1, seed=0, mag_bins=MAG_BINS, n_tries=500):
    """Hold out whole earthquake sequences (all stations of an event go to the same split).

    Sequences are stratified by magnitude band (meta['magnitude'], records without one form
    their own band), so every band with >= 3 sequences appears in train, val and test. Within
    that constraint, n_tries shuffles are drawn and the one whose record counts are closest to
    the target fractions is kept (deterministic for a given seed).
    """
    groups = {}
    for r in records:
        groups.setdefault(event_group(r.event), []).append(r)
    bands = {}
    for g, rs in groups.items():
        m = [x.meta.get("magnitude", np.nan) for x in rs]
        m = np.nanmax(m) if np.isfinite(m).any() else np.nan
        bands.setdefault(int(np.digitize(m, mag_bins)) if np.isfinite(m) else -1, []).append(g)
    target = np.array([1 - test_frac - val_frac, val_frac, test_frac]) * len(records)
    rng = np.random.default_rng(seed)
    best, best_err = None, np.inf
    for _ in range(n_tries):
        test, val = set(), set()
        for b in sorted(bands):
            gs = sorted(bands[b])
            rng.shuffle(gs)
            n = len(gs)
            n_test = max(1, int(round(test_frac * n))) if n >= 3 else int(n == 2)
            n_val = max(1, int(round(val_frac * n))) if n >= 3 else 0
            test |= set(gs[:n_test])
            val |= set(gs[n_test:n_test + n_val])
        counts = np.array([sum(len(groups[g]) for g in groups if g not in test | val),
                           sum(len(groups[g]) for g in val), sum(len(groups[g]) for g in test)])
        err = np.abs(counts - target).sum()
        if err < best_err:
            best, best_err = (test, val), err
    test, val = best
    return {"train": [r for r in records if event_group(r.event) not in test | val],
            "val": [r for r in records if event_group(r.event) in val],
            "test": [r for r in records if event_group(r.event) in test]}
