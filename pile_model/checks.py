"""Part 6 checks: is one combined run physically sensible?

    .venv\\Scripts\\python.exe -m pile_model.checks          # runs (parallel) + figures
    .venv\\Scripts\\python.exe -m pile_model.checks --figures-only

Runs one storm + earthquake case on loose sand three ways (waves only, earthquake only,
combined) plus a 1D soil column with the same shaking (free-field reference).
Results -> results/part6/.
"""
import argparse
import json
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "part6"
RUNS = OUT / "runs"
DR, PERM = 0.40, 1e-4
CASE = dict(hs=7.0, tp=12.0, pga_g=0.25, t_eq=40.0)     # strong storm, strong quake


def make_load():
    """One combined load history from part 4 with a real PEER record (Kocaeli, M7.5, test set)."""
    from loads.earthquakes import G, load_peer, process, scale_to_pga
    from loads.waves import wave_loads
    recs = [r for r in load_peer() if r.name == "RSN1166_KOCAELI_IZN180"]
    rec = scale_to_pga(process(recs[0], 0.01), CASE["pga_g"])
    dt = 0.01
    n_pre, n_post = int(CASE["t_eq"] / dt), int(20 / dt)
    acc = np.zeros(n_pre + len(rec.acc) + n_post)
    acc[n_pre:n_pre + len(rec.acc)] = rec.acc
    t = np.arange(len(acc)) * dt
    w = wave_loads(t, CASE["hs"], CASE["tp"], np.random.default_rng(1), depth=30, diameter=8)
    return dict(t=t, acc=acc, force=w["force"], moment=w["moment"], dt=dt, record=rec.name,
                t_quake=(CASE["t_eq"], CASE["t_eq"] + len(rec.acc) * dt))


def _job(args):
    name, load = args
    import openseespy.opensees as ops
    ops.logFile(str(Path(tempfile.gettempdir()) / f"ops_p6_{os.getpid()}.log"), "-noEcho")
    acc, F, M, dt = load["acc"], load["force"], load["moment"], load["dt"]
    if name == "column":
        from soil_column.column import run_column
        r = run_column(acc, dt, dr=DR, H=30.0, dz=1.0, perm=PERM, t_post=0.0)
        np.savez(RUNS / "column.npz", t=r["t"], ru=r["ru"], depth=r["depth"], runtime=r["runtime"], ok=r["ok"])
        return name, r["runtime"], r["ok"]
    from pile_model.model import run
    zero = np.zeros_like(acc)
    kw = dict(waves=dict(acc=zero, force=F, moment=M), quake=dict(acc=acc, force=zero, moment=zero),
              combined=dict(acc=acc, force=F, moment=M))[name]
    r = run(kw["acc"], dt, force=kw["force"], moment=kw["moment"], dr=DR, perm=PERM)
    np.savez(RUNS / f"{name}.npz", **{k: v for k, v in r.items() if k != "params"},
             params=json.dumps(r["params"]))
    return name, r["runtime"], r["ok"]


def run_all():
    RUNS.mkdir(parents=True, exist_ok=True)
    load = make_load()
    np.savez(RUNS / "load.npz", **{k: v for k, v in load.items() if k != "t_quake"}, t_quake=load["t_quake"])
    jobs = [(n, load) for n in ("combined", "quake", "waves", "column")]
    with Pool(len(jobs)) as pool:
        for name, rt, ok in pool.imap_unordered(_job, jobs):
            print(f"{name}: ok={ok} runtime {rt:.0f} s", flush=True)


def other_checks():
    """Gravity state, fixed-base vs soil-structure frequency (pluck test)."""
    import openseespy.opensees as ops
    from pile_model.model import build, fixed_base_frequency, gravity, run
    ops.logFile(str(Path(tempfile.gettempdir()) / "ops_p6_checks.log"), "-noEcho")
    res = {"fixed_base_f1_hz": fixed_base_frequency()}
    pm = build(DR, PERM)
    gravity(pm)
    ni = len(pm.x)
    rows = []
    for i, j in [(0, 10), (5, 10), (7, 10), (7, 15), (9, 15), (9, 1)]:
        el = j * (ni - 1) + i + 1
        s = ops.eleResponse(el, "stress")
        d = pm.mesh.H - 0.5 * (pm.y[j] + pm.y[j + 1])
        rows.append(dict(x=f"{pm.x[i]:.1f}..{pm.x[i + 1]:.1f}", depth=d, sigma_v=-s[1], sigma_h=-s[0],
                         tau=s[2], free_field_sigma_v=pm.sigv0[0] / pm.mesh.H * d))
    res["gravity_stresses"] = rows
    res["cap_settlement_mm"] = -1e3 * ops.nodeDisp(pm.cap, 2)
    res["structure_weight_MN"] = pm.structure_weight / 1e3
    # pluck: smooth 30 MN m moment pulse at the cap, free vibration afterwards
    dt = 0.01
    t = np.arange(0, 40, dt)
    M = np.where(t < 2, 3e4 * np.sin(np.pi * t / 2) ** 2, 0.0)
    r = run(np.zeros_like(t), dt, moment=M, dt_waves=0.02, out_dt=0.02, pm=pm)
    sel = r["t"] > 4
    u = r["hub_u"][sel] - r["hub_u"][sel].mean()
    zc = r["t"][sel][1:][np.diff(np.sign(u)) != 0]
    res["system_f1_hz"] = float(1 / np.mean(2 * np.diff(zc)))
    # logarithmic decrement -> damping ratio of the first mode
    peaks = [np.abs(u[(r["t"][sel] >= a) & (r["t"][sel] < a + 5)]).max() for a in (6, 16, 26)]
    res["system_damping_ratio"] = float(np.log(peaks[0] / peaks[2]) / (2 * np.pi * 2 * res["system_f1_hz"] * 10))
    (OUT / "checks.json").write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------ figures
def _load(name):
    z = np.load(RUNS / f"{name}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def figures():
    L = _load("load")
    runs = {n: _load(n) for n in ("waves", "quake", "combined")}
    col = _load("column")
    t0, t1 = L["t_quake"]
    colors = dict(waves="#2e86c1", quake="#c0392b", combined="k")

    # 1. inputs and tilt / hub histories
    fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    axs[0].plot(L["t"], L["acc"] / 9.81, color="#c0392b", lw=0.6)
    axs[0].set_ylabel("Base acc. (g)")
    axs[1].plot(L["t"], L["force"] / 1e3, color="#2e86c1", lw=0.6)
    axs[1].set_ylabel("Wave force (MN)")
    for n, r in runs.items():
        axs[2].plot(r["t"], r["tilt"], color=colors[n], lw=1 if n != "combined" else 1.4, label=n)
        axs[3].plot(r["t"], r["hub_u"], color=colors[n], lw=0.8)
        axs[4].plot(r["t"], 1e3 * r["cap_v"], color=colors[n], lw=1)
    axs[2].set_ylabel("Pile tilt (deg)")
    axs[2].axhline(0.5, color="0.5", ls="--", lw=0.8)
    axs[2].axhline(-0.5, color="0.5", ls="--", lw=0.8)
    axs[2].legend(fontsize=8, ncol=3)
    axs[3].set_ylabel("Hub displacement (m)")
    axs[4].set_ylabel("Pile settlement (mm)")
    axs[-1].set_xlabel("Time (s)")
    for a in axs:
        a.axvspan(t0, t1, color="#f5b7b1", alpha=0.25)
        a.grid(alpha=0.3)
    fig.suptitle(f"Combined run: Hs {CASE['hs']} m, Tp {CASE['tp']} s, {L['record']} at {CASE['pga_g']} g, "
                 f"Dr {DR}, k {PERM} m/s (shaded = earthquake)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "combined_histories.png", dpi=150)

    # 2. ru field snapshots (combined run)
    r = runs["combined"]
    x, depth, mask = r["x"], r["depth"], r["mask"]
    snaps = [t0 + 5, t0 + 15, t1, r["t"][-1]]
    fig, axs = plt.subplots(1, len(snaps), figsize=(16, 4), sharey=True)
    for a, ts in zip(axs, snaps):
        k = int(np.argmin(np.abs(r["t"] - ts)))
        f = np.where(mask, r["ru"][k], np.nan)
        im = a.pcolormesh(x, depth, f, vmin=0, vmax=1, cmap="magma_r", shading="gouraud")
        a.fill_between([-4, 4], 0, 24, color="0.6")
        a.set_title(f"t = {r['t'][k]:.0f} s", fontsize=9)
        a.set_xlabel("x (m)")
    axs[0].invert_yaxis()
    axs[0].set_ylabel("Depth below mudline (m)")
    fig.colorbar(im, ax=axs, shrink=0.8, label="ru (1 = liquefied)")
    fig.savefig(OUT / "combined_ru_snapshots.png", dpi=150, bbox_inches="tight")

    # 3. free-field check: far-field ru of the 2D model vs the 1D column
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for a, d in zip(axs, (3.0, 8.0, 15.0)):
        j = int(np.argmin(np.abs(depth - d)))
        jc = int(np.argmin(np.abs(col["depth"] - d)))
        a.plot(col["t"], col["ru"][:, jc], color="k", lw=1.6, label="1D column (part 3 model)")
        a.plot(runs["quake"]["t"], runs["quake"]["ru"][:, j, 0], color="#c0392b", lw=1, label="2D model, x = -50 m")
        a.plot(runs["quake"]["t"], runs["quake"]["ru"][:, j, int(np.argmin(np.abs(x + 5.5)))], color="#d68910",
               lw=1, label="2D model, next to pile")
        a.set_title(f"depth {depth[j]:.0f} m", fontsize=9)
        a.set_xlabel("Time (s)")
        a.grid(alpha=0.3)
    axs[0].set_ylabel("ru (earthquake only)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "free_field_check.png", dpi=150)

    summary = {}
    for n, rr in runs.items():
        i0 = int(np.argmin(np.abs(rr["t"] - t0)))
        summary[n] = dict(ok=bool(rr["ok"]), runtime_min=float(rr["runtime"]) / 60, steps=int(rr["steps"]),
                          sim_time_s=float(rr["t"][-1]),
                          peak_tilt_deg=float(np.nanmax(np.abs(rr["tilt"]))),
                          final_tilt_deg=float(np.nanmean(rr["tilt"][-40:])),
                          tilt_before_quake_deg=float(np.nanmean(rr["tilt"][max(0, i0 - 40):i0])),
                          peak_hub_disp_m=float(np.nanmax(np.abs(rr["hub_u"]))),
                          final_settlement_mm=float(-1e3 * np.nanmean(rr["cap_v"][-40:])),
                          peak_ru_far_field=float(np.nanmax(rr["ru"][:, :, 0])),
                          peak_ru_near_pile=float(np.nanmax(rr["ru"][:, :, int(np.argmin(np.abs(x + 5.5)))])))
    j8 = int(np.argmin(np.abs(depth - 8)))
    jc = int(np.argmin(np.abs(col["depth"] - 8)))
    q = runs["quake"]
    summary["free_field_vs_column_ru_rmse_8m"] = float(np.sqrt(np.nanmean(
        (np.interp(col["t"], q["t"], q["ru"][:, j8, 0]) - col["ru"][:, jc]) ** 2)))
    summary["column_runtime_min"] = float(col["runtime"]) / 60
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--checks-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.checks_only:
        print(json.dumps(other_checks(), indent=1))
    elif not a.figures_only:
        run_all()
        figures()
    else:
        figures()
