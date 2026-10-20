"""Calibrate PM4Sand hpo so each Dr liquefies in 15 cycles at the Idriss & Boulanger (2008) CRR.

Then check the whole CSR-vs-N curve against the I&B target (b = 0.34).
Writes soil_column/pm4sand_calibration.json and results/part3/dss_*.png.

Run from the project root:  .venv\\Scripts\\python.exe -m soil_column.calibrate
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .element_test import cyclic_dss
from .materials import CALIB_FILE, crr_cycles, crr_m75

DRS = [0.35, 0.45, 0.55, 0.65, 0.75]
CHECK_CYCLES = [5, 10, 15, 30]          # CSRs chosen so the target N is these values
OUT = Path(__file__).resolve().parents[1] / "results" / "part3"


def calibrate_hpo(dr, lo=0.05, hi=3.0, iters=12):
    """Bisection on hpo: smallest hpo with N_liq >= 15 at CSR = CRR15."""
    csr = crr_m75(dr)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        n = cyclic_dss(dr, csr, mid)["N"]
        if np.isnan(n):
            raise RuntimeError(f"DSS failed to converge at Dr={dr}, hpo={mid:.3f}")
        if n >= 15:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calib = {}
    for dr in DRS:
        hpo = calibrate_hpo(dr)
        checks = []
        for nt in CHECK_CYCLES:
            csr = crr_cycles(dr, nt)
            checks.append({"csr": csr, "N_target": nt, "N_model": cyclic_dss(dr, csr, hpo)["N"]})
        calib[f"{dr:.2f}"] = {"hpo": hpo, "crr15": crr_m75(dr), "checks": checks}
        print(f"Dr={dr:.2f}  hpo={hpo:.3f}  " +
              "  ".join(f"N{c['N_target']}->{c['N_model']}" for c in checks), flush=True)
    CALIB_FILE.write_text(json.dumps(calib, indent=2))
    plot_crr_curves(calib)
    plot_example_loop()


def plot_crr_curves(calib):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    n = np.logspace(0, 2, 100)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(calib)))
    for c, (dr, v) in zip(colors, calib.items()):
        d = float(dr)
        ax.plot(n, crr_cycles(d, n), color=c, lw=1.5, label=f"Dr = {d:.0%}  I&B target")
        pts = [(k["N_model"], k["csr"]) for k in v["checks"] if np.isfinite(k["N_model"])]
        ax.plot(*zip(*pts), "o", color=c, mfc="white", mew=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Cycles to 3% shear strain, N")
    ax.set_ylabel("Cyclic stress ratio, CSR")
    ax.set_title("PM4Sand cyclic DSS (markers) vs Idriss & Boulanger 2008 (lines)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dss_crr_curves.png", dpi=150)


def plot_example_loop(dr=0.55):
    r = cyclic_dss(dr, crr_m75(dr), record=True)
    g, tau, sv = r["hist"].T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(g * 100, tau, lw=0.6)
    a1.set_xlabel("Shear strain (%)")
    a1.set_ylabel("Shear stress (kPa)")
    a2.plot(sv, tau, lw=0.6)
    a2.set_xlabel("Vertical effective stress (kPa)")
    a2.set_ylabel("Shear stress (kPa)")
    fig.suptitle(f"Constant-volume cyclic DSS, Dr = {dr:.0%}, CSR = {crr_m75(dr):.3f} "
                 f"(liquefied at N = {r['N']})")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dss_example_dr55.png", dpi=150)


if __name__ == "__main__":
    main()
