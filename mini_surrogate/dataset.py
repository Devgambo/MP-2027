"""Part 5 dataset: soil-column runs driven by real PEER records.

Each case: one PEER record (processed, scaled to a target PGA), a sand density Dr and a
permeability k, run through soil_column.run_column. Records are assigned to train/val/test
with loads.earthquakes.split_by_event, so test earthquakes are never seen in training.

Run from the project root (resumable: finished cases are skipped):
    .venv\\Scripts\\python.exe -m mini_surrogate.dataset --workers 10
"""
import argparse
import os
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import qmc

from loads.earthquakes import G, load_peer, process, split_by_event

ROOT = Path(__file__).resolve().parents[1]
H5_PATH = ROOT / "data" / "column_dataset.h5"

DT_IN = 0.01            # input / analysis time step (s)
T_SHAKE = 40.0          # records longer than this are cut (14 of 353 records)
T_TOTAL = 50.0          # every case is simulated for 50 s (>= 10 s after shaking)
OUT_EVERY = 5           # stored output step = 0.05 s
N_CASES = {"train": 230, "val": 35, "test": 65}
RANGES = {"pga_g": (0.05, 0.40), "dr": (0.35, 0.75), "log10_k": (-5.0, -3.0)}
MAX_SCALE = 4.0


def _taper_end(a, n):
    a = a.copy()
    a[-n:] *= 0.5 * (1 + np.cos(np.pi * np.arange(n) / n))
    return a


def plan_cases(seed=0):
    """List of case dicts (split, record, pga, dr, k, acc) – deterministic for a seed."""
    recs = [process(r, DT_IN) for r in load_peer()]
    splits = split_by_event(recs)
    cases = []
    for s_idx, (split, n) in enumerate(N_CASES.items()):
        pool = splits[split]
        rng = np.random.default_rng(seed + 100 * s_idx)
        lo = np.array([v[0] for v in RANGES.values()])
        hi = np.array([v[1] for v in RANGES.values()])
        u = qmc.LatinHypercube(d=3, seed=seed + s_idx).random(n)
        used = np.zeros(len(pool))
        for x in qmc.scale(u, lo, hi):
            pga, dr, logk = x
            ok = [i for i, r in enumerate(pool) if 1 / MAX_SCALE <= pga * G / r.pga <= MAX_SCALE]
            if not ok:                                         # fall back to the nearest record
                ok = [int(np.argmin([abs(np.log(pga * G / r.pga)) for r in pool]))]
            least = min(used[i] for i in ok)                   # spread usage over records
            i = rng.choice([i for i in ok if used[i] == least])
            used[i] += 1
            r = pool[i]
            acc = r.acc * (pga * G / r.pga)
            n_max = int(round(T_SHAKE / DT_IN))
            if len(acc) > n_max:
                acc = _taper_end(acc[:n_max], int(0.5 / DT_IN))
            cases.append(dict(case_id=len(cases), split=split, record=r.name, event=r.event,
                              magnitude=float(r.meta.get("magnitude", np.nan)), pga_g=float(pga),
                              scale_factor=float(pga * G / r.pga), dr=float(dr), perm=float(10 ** logk),
                              acc=acc.astype(np.float64)))
    return cases


def run_case(case):
    """Worker: one soil-column run. Returns arrays on the 0.05 s output grid."""
    import openseespy.opensees as ops
    from soil_column.column import run_column
    log = Path(tempfile.gettempdir()) / f"ops_{os.getpid()}.log"
    ops.logFile(str(log), "-noEcho")
    acc = case["acc"]
    t0 = time.time()
    try:
        out = run_column(acc, DT_IN, dr=case["dr"], perm=case["perm"],
                         t_post=T_TOTAL - len(acc) * DT_IN)
    except Exception as ex:                                     # gravity stage failure etc.
        return dict(case, ok=False, error=repr(ex), runtime=time.time() - t0)
    n_out = int(round(T_TOTAL / DT_IN)) // OUT_EVERY + 1
    sl = slice(0, None, OUT_EVERY)
    def grid(a):                                                # pad diverged runs with NaN
        a = a[sl][:n_out]
        pad = np.full((n_out - len(a),) + a.shape[1:], np.nan)
        return np.concatenate([a, pad]) if len(pad) else a
    full = np.zeros(int(round(T_TOTAL / DT_IN)) + 1)
    full[:len(acc)] = acc
    return dict(case, ok=bool(out["ok"]), error="", runtime=out["runtime"],
                acc=full[:-1].astype(np.float32),
                ru=grid(out["ru"][:, :-1]).astype(np.float32),     # drop surface node (ru = 0)
                depth=out["depth"][:-1], settle=grid(out["settle"]).astype(np.float32),
                disp_surf=grid(out["disp_surf"]).astype(np.float32),
                acc_surf=grid(out["acc_surf"]).astype(np.float32))


ARRAYS = ("acc", "ru", "settle", "disp_surf", "acc_surf")


def write_case(f, res):
    g = f.create_group(f"case_{res['case_id']:05d}")
    for k in ARRAYS:
        if k in res:
            g.create_dataset(k, data=np.asarray(res[k], np.float32), compression="gzip")
    for k, v in res.items():
        if k not in ARRAYS and k != "depth":
            g.attrs[k] = v
    if "depth" in res and "depth" not in f:
        f["depth"] = res["depth"]
    f.flush()


def build(workers=10, seed=0, path=H5_PATH, limit=None):
    cases = plan_cases(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        f.attrs.update(dt_in=DT_IN, dt_out=DT_IN * OUT_EVERY, t_total=T_TOTAL, seed=seed)
        done = {int(k.split("_")[1]) for k in f if k.startswith("case_")}
        todo = [c for c in cases if c["case_id"] not in done][:limit]
        print(f"{len(cases)} planned, {len(done)} done, {len(todo)} to run on {workers} workers", flush=True)
        todo.sort(key=lambda c: -c["pga_g"] / c["dr"])           # slow (strong, loose) cases first
        t0 = time.time()
        with Pool(workers) as pool:
            for k, res in enumerate(pool.imap_unordered(run_case, todo), 1):
                write_case(f, res)
                print(f"[{k}/{len(todo)}] case {res['case_id']} {res['split']} ok={res['ok']} "
                      f"{res['runtime']:.0f}s  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.workers, a.seed, limit=a.limit)
