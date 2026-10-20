"""Benchmark A: Terzaghi 1D consolidation, exact solution vs the SSPquadUP column.

Elastic, weightless column; top drained, base impermeable; sudden surface load q.
Checks the u-p coupling, permeability units and drainage boundary conditions used
by the liquefaction column.

Run from the project root:  .venv\\Scripts\\python.exe -m soil_column.consolidation_check
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops

from .column import _nid, _transient, build_column, pore_pressure
from .materials import G, RHO_W

OUT = Path(__file__).resolve().parents[1] / "results" / "part3"
H, DZ, Q = 10.0, 0.25, 100.0
E, NU, PERM, VOID = 2.0e4, 0.3, 1e-5, 0.6
TV_OUT = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def terzaghi(z, tv, n_terms=200):
    """Excess pore pressure / q at depth z below the drained top, time factor tv."""
    u = np.zeros_like(z)
    for m in range(n_terms):
        M = np.pi * (2 * m + 1) / 2
        u += 2 / M * np.sin(M * z / H) * np.exp(-M ** 2 * tv)
    return u


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nz = build_column(H, DZ, ("ElasticIsotropic", E, NU, 0.0), VOID, PERM, 0.0, gravity=False)
    m_oed = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))
    cv = PERM * m_oed / (RHO_W * G)
    t_out = [tv * H ** 2 / cv for tv in TV_OUT]

    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for s in (0, 1):
        ops.load(_nid(nz, s), 0.0, -Q * DZ / 2, 0.0)
    _transient(0.6, 0.3025, tol=1e-8)

    left = [_nid(j, 0) for j in range(nz + 1)]
    depth = H - np.array([ops.nodeCoord(n, 2) for n in left])
    # log-spaced steps resolve the fast early dissipation and the slow tail
    times = np.unique(np.concatenate([np.geomspace(1e-3, t_out[-1], 600), t_out]))
    snaps, t_prev = {}, 0.0
    for t in times:
        if ops.analyze(1, t - t_prev) != 0:
            raise RuntimeError(f"failed at t = {t:.3f}")
        t_prev = t
        for tv, to in zip(TV_OUT, t_out):
            if np.isclose(t, to):
                snaps[tv] = np.array([pore_pressure(n) for n in left])

    print(f"cv = {cv:.4f} m^2/s")
    fig, ax = plt.subplots(figsize=(5.5, 6))
    zf = np.linspace(0, H, 200)
    errs = []
    for c, tv in zip(plt.cm.viridis(np.linspace(0, 0.9, len(TV_OUT))), TV_OUT):
        ax.plot(terzaghi(zf, tv), zf, color=c, lw=1.5, label=f"Tv = {tv}")
        ax.plot(snaps[tv] / Q, depth, "o", color=c, mfc="white", ms=4)
        err = np.max(np.abs(snaps[tv] / Q - terzaghi(depth, tv)))
        errs.append(err)
        print(f"Tv = {tv:<5} max |u/q error| = {err:.4f}")
    ax.invert_yaxis()
    ax.set_xlabel("Excess pore pressure / applied load, u/q")
    ax.set_ylabel("Depth below drained surface (m)")
    ax.set_title("Terzaghi consolidation: exact (lines) vs OpenSees (markers)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "terzaghi_check.png", dpi=150)
    return max(errs)


if __name__ == "__main__":
    main()
