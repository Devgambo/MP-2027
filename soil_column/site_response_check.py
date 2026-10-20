"""Benchmark C: shaken saturated sand column, checked against known liquefaction trends.

Input is a LEAP-style tapered 2 Hz sine on a rigid base (no PEER records needed yet).
Trend checks (from centrifuge tests such as VELACS Model 1 and LEAP):
  1. loose sand liquefies (ru -> 1); time to liquefaction is reported against depth
  2. surface acceleration collapses once the sand liquefies
  3. pore pressure dissipates and the surface settles after shaking
  4. denser sand builds up less pore pressure
  5. the liquefied zone agrees with a simplified-procedure (Seed-Idriss style) FS < 1 zone,
     using the element-calibrated CRR and the small-strain shear stress demand

Run from the project root:  .venv\\Scripts\\python.exe -m soil_column.site_response_check
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

from .column import _transient, build_column, run_column, tapered_sine
from .materials import G, P_ATM, RHO_W, crr_cycles, n160_from_dr, pm4sand_args, rho_sat, void_ratio

OUT = Path(__file__).resolve().parents[1] / "results" / "part3"
AMP_G, FREQ, N_CYC = 0.15, 2.0, 10
N_EQ = N_CYC + 1.0        # two cosine ramps (2 cycles each) ~ one extra full cycle


def k_sigma(dr, sigv):
    """Overburden correction (Idriss & Boulanger 2008), capped at 1.1."""
    c = min(1 / (18.9 - 2.55 * np.sqrt(n160_from_dr(dr))), 0.3)
    return np.minimum(1 - c * np.log(sigv / P_ATM), 1.1)


def elastic_tau_max(acc, dt, dr, H=10.0, dz=0.5):
    """Peak shear stress profile with PM4Sand held in its elastic stage (small-strain demand)."""
    nz = build_column(H, dz, ("PM4Sand", *pm4sand_args(dr)), void_ratio(dr), 1e-4, rho_sat(dr))
    ops.updateMaterialStage("-material", 1, "-stage", 0)
    _transient(5 / 6, 4 / 9)
    ops.analyze(20, 100.0)
    ops.setTime(0.0)
    ops.timeSeries("Path", 10, "-dt", dt, "-values", *acc)
    ops.pattern("UniformExcitation", 10, 1, "-accel", 10)
    w1, w2 = 2 * np.pi, 20 * np.pi
    ops.rayleigh(0.01 * 2 * w1 * w2 / (w1 + w2), 0.01 * 2 / (w1 + w2), 0, 0)
    _transient(0.6, 0.3025)
    tmax = np.zeros(nz)
    for _ in range(len(acc)):
        ops.analyze(1, dt)
        tmax = np.maximum(tmax, [abs(ops.eleResponse(e, "stress")[2]) for e in range(1, nz + 1)])
    return tmax


def fs_profile(acc, dt, dr, r):
    tmax = elastic_tau_max(acc, dt, dr)
    z = r["depth_el"]
    sigv = (rho_sat(dr) - RHO_W) * G * z
    csr = tmax / sigv
    crr = crr_cycles(dr, N_EQ) * k_sigma(dr, sigv)
    return z, csr, crr


def trend_checks(r):
    t, ru, acc = r["t"], r["ru"], r["acc_surf"]
    t_shake = r["shake_end"]
    nodes = slice(0, -1)                         # exclude the surface node (sigv0 = 0)
    ru_n, z_n = ru[:, nodes], r["depth"][nodes]
    liq = ru_n.max(axis=0) >= 0.95
    t_liq = np.array([t[np.argmax(ru_n[:, k] >= 0.95)] if liq[k] else np.nan for k in range(len(z_n))])
    # acceleration amplitude in cycles 2-4 of full shaking vs the last 3 cycles of full shaking
    early = (t > 1.0 + 1 / FREQ) & (t < 1.0 + 4 / FREQ)
    late = (t > t_shake - 1.0 - 3 / FREQ) & (t < t_shake - 1.0)
    acc_ratio = np.abs(acc[late]).max() / np.abs(acc[early]).max()
    ru_end_max = ru_n[-1].max()
    ru_peak_max = ru_n.max()
    checks = {
        "liquefied_depth_m": float(z_n[liq].max()) if liq.any() else 0.0,
        "ru_peak_max": float(ru_peak_max),
        "ru_end_max": float(ru_end_max),
        "t_liq_s_by_depth": {f"{z:.1f}": (None if np.isnan(tl) else round(float(tl), 2))
                             for z, tl in zip(z_n[::2], t_liq[::2])},
        "surface_acc_late_over_early": float(acc_ratio),
        "settlement_mm": float(-r["settle"][-1] * 1000),
    }
    return checks, t_liq, z_n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acc, dt = tapered_sine(AMP_G, FREQ, N_CYC, 2, 0.005)
    runs, summary = {}, {}
    for label, dr in [("loose", 0.35), ("medium", 0.55), ("dense", 0.75)]:
        r = run_column(acc, dt, dr=dr, t_post=15.0, verbose=True)
        r["shake_end"] = len(acc) * dt
        c, t_liq, z_n = trend_checks(r)
        c["runtime_s"] = r["runtime"]
        c["converged"] = r["ok"]
        z, csr, crr = fs_profile(acc, dt, dr, r)
        c["fs_lt_1_depth_m"] = float(z[csr > crr].max()) if np.any(csr > crr) else 0.0
        r.update(t_liq=t_liq, z_n=z_n, fs=(z, csr, crr))
        runs[label], summary[label] = r, c
        print(label, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()}),
              flush=True)
    (OUT / "site_response_summary.json").write_text(json.dumps(summary, indent=2))
    plot(runs, acc, dt)


def plot(runs, acc, dt):
    labels = list(runs)
    colors = dict(zip(labels, ["#c0392b", "#2e86c1", "#1e8449"]))
    r = runs["loose"]

    # Figure 1: loose column time histories
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(np.arange(len(acc)) * dt, acc / G, color="0.5", lw=0.8, label="base")
    axs[0].plot(r["t"], r["acc_surf"] / G, color=colors["loose"], lw=0.8, label="surface")
    axs[0].set_ylabel("Acceleration (g)")
    axs[0].legend(fontsize=8, loc="upper right")
    for zt, c in zip([1.0, 3.0, 5.0, 8.0], plt.cm.viridis(np.linspace(0, 0.85, 4))):
        k = int(np.argmin(np.abs(r["depth"] - zt)))
        axs[1].plot(r["t"], r["ru"][:, k], color=c, lw=1, label=f"z = {r['depth'][k]:.1f} m")
    axs[1].axhline(1.0, color="k", lw=0.6, ls="--")
    axs[1].set_ylabel("Excess pore-pressure ratio ru")
    axs[1].legend(fontsize=8, loc="upper right")
    axs[2].plot(r["t"], -r["settle"] * 1000, color="k", lw=1)
    axs[2].set_ylabel("Surface settlement (mm)")
    axs[2].set_xlabel("Time (s)")
    for a in axs:
        a.grid(alpha=0.3)
    fig.suptitle(f"Loose sand column (Dr = 35%), {AMP_G} g {FREQ:.0f} Hz tapered sine")
    fig.tight_layout()
    fig.savefig(OUT / "column_loose_histories.png", dpi=150)

    # Figure 2: ru field (depth x time) for the loose column
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(r["t"], r["depth"][:-1], r["ru"][:, :-1].T, cmap="magma", vmin=0, vmax=1,
                       shading="auto")
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (m)")
    fig.colorbar(im, ax=ax, label="ru")
    ax.set_title("Pore-pressure ratio field, Dr = 35% (this is the field the FNO will learn)")
    fig.tight_layout()
    fig.savefig(OUT / "column_loose_ru_field.png", dpi=150)

    # Figure 3: peak ru profiles vs density, and FS profiles
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    for lab in labels:
        rr = runs[lab]
        a1.plot(rr["ru"][:, :-1].max(axis=0), rr["depth"][:-1], "o-", ms=3, color=colors[lab],
                label=f"Dr = {rr['params']['dr']:.0%}")
        z, csr, crr = rr["fs"]
        a2.plot(crr / csr, z, "o-", ms=3, color=colors[lab], label=f"Dr = {rr['params']['dr']:.0%}")
    a1.axvline(0.95, color="k", ls="--", lw=0.6)
    a1.set_xlabel("Peak ru (OpenSees PM4Sand)")
    a1.set_ylabel("Depth (m)")
    a1.set_xlim(0, 1.05)
    a2.axvline(1.0, color="k", ls="--", lw=0.6)
    a2.set_xlabel("Simplified-procedure FS = CRR / CSR")
    a2.set_xscale("log")
    a1.invert_yaxis()
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle("Liquefied zone (ru >= 0.95) vs simplified-procedure FS < 1")
    fig.tight_layout()
    fig.savefig(OUT / "column_density_fs.png", dpi=150)


if __name__ == "__main__":
    main()
