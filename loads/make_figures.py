"""Part 4 figures and summary numbers -> results/part4/.

Run from the project root:  .venv\\Scripts\\python.exe -m loads.make_figures
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .combined import MAX_STEEPNESS, generate_loads, save_h5
from .earthquakes import G, process, response_spectrum, significant_window, synthetic_library
from .waves import jonswap, wave_loads

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "part4"


def fig_spectra():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    f = np.linspace(0.02, 0.5, 600)
    for (hs, tp), c in zip([(2, 7), (5, 10), (8, 13)], ["#2e86c1", "#d68910", "#c0392b"]):
        t = np.arange(0, 3600, 0.25)
        eta = wave_loads(t, hs, tp, np.random.default_rng(0), ramp=0)["eta"]
        fw, pw = signal.welch(eta, fs=4.0, nperseg=2048)
        ax.plot(fw, pw, color=c, lw=0.8, alpha=0.6)
        ax.plot(f, jonswap(f, hs, tp), color=c, lw=2,
                label=f"Hs = {hs} m, Tp = {tp} s (realised 4σ = {4 * eta.std():.2f} m)")
    ax.set_xlim(0, 0.4)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Wave energy density S(f) (m²/Hz)")
    ax.set_title("JONSWAP target (thick) vs 1-hour simulated sea (thin)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "wave_spectra.png", dpi=150)


def fig_wave_example():
    t = np.arange(0, 150, 0.01)
    w = wave_loads(t, 6.0, 11.0, np.random.default_rng(3))
    fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axs[0].plot(t, w["eta"], color="#2e86c1", lw=0.8)
    axs[0].set_ylabel("Sea surface η (m)")
    axs[1].plot(t, w["force"] / 1e3, color="k", lw=0.8, label="total")
    axs[1].plot(t, w["force_inertia"] / 1e3, color="#d68910", lw=0.8, label="inertia")
    axs[1].plot(t, w["force_drag"] / 1e3, color="#c0392b", lw=0.8, label="drag")
    axs[1].set_ylabel("Force at mudline (MN)")
    axs[1].legend(fontsize=8, loc="upper right")
    axs[2].plot(t, w["moment"] / 1e3, color="k", lw=0.8)
    axs[2].set_ylabel("Moment at mudline (MN·m)")
    axs[2].set_xlabel("Time (s)")
    for a in axs:
        a.grid(alpha=0.3)
    fig.suptitle(f"Wave load on an 8 m monopile in 30 m water, Hs = 6 m, Tp = 11 s (KC = {w['kc']:.2f})")
    fig.tight_layout()
    fig.savefig(OUT / "wave_load_example.png", dpi=150)
    return w


def fig_earthquakes():
    recs = [process(r) for r in synthetic_library(6, seed=4)]
    r = recs[0]
    v = np.cumsum(r.acc) * r.dt
    d = np.cumsum(v) * r.dt
    t5, t95 = significant_window(r.acc, r.dt)
    fig, axs = plt.subplots(2, 2, figsize=(11, 6))
    for a, y, lab in zip(axs[:, 0], [r.acc / G, d * 100], ["Acceleration (g)", "Displacement (cm)"]):
        a.plot(r.t, y, lw=0.7, color="k")
        a.axvspan(t5, t95, color="#d68910", alpha=0.15)
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    axs[0, 0].set_title(f"{r.name} (synthetic); shaded = Arias 5–95% ({t95 - t5:.1f} s)", fontsize=9)
    axs[1, 0].set_xlabel("Time (s)")
    periods = np.logspace(-1.5, 0.7, 60)
    gs = axs[0, 1].get_gridspec()
    for a in axs[:, 1]:
        a.remove()
    ax = fig.add_subplot(gs[:, 1])
    for rr in recs:
        sa = response_spectrum(rr.acc / rr.pga * 0.3 * G, rr.dt, periods)
        ax.plot(periods, sa / G, lw=1, label=f"fg = {rr.meta['fg']:.1f} Hz")
    ax.set_xscale("log")
    ax.set_xlabel("Period T (s)")
    ax.set_ylabel("Spectral acceleration Sa (g), 5% damping")
    ax.set_title("Response spectra, all scaled to PGA = 0.3 g", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "earthquake_examples.png", dpi=150)


def fig_combined(cases):
    fig, axs = plt.subplots(len(cases), 1, figsize=(9, 2.4 * len(cases)), sharex=False)
    for a, c in zip(axs, cases):
        p = c.params
        a.plot(c.t, c.acc / G, color="#c0392b", lw=0.6)
        a.set_ylabel("Base acc. (g)", color="#c0392b")
        b = a.twinx()
        b.plot(c.t, c.force / 1e3, color="#2e86c1", lw=0.6)
        b.set_ylabel("Wave force (MN)", color="#2e86c1")
        a.set_title(f"Hs = {p['hs']:.1f} m, Tp = {p['tp']:.1f} s | PGA = {p['pga_g']:.2f} g at "
                    f"t = {p['t_eq']:.0f} s | {p['record']}", fontsize=8)
        a.grid(alpha=0.3)
    axs[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "combined_examples.png", dpi=150)


def fig_coverage(cases):
    p = {k: np.array([c.params[k] for c in cases]) for k in ("hs", "tp", "tp_sampled", "pga_g", "t_eq")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    tp_line = np.linspace(5, 15, 50)
    a1.fill_between(tp_line, MAX_STEEPNESS * G * tp_line ** 2 / (2 * np.pi), 10, color="0.85",
                    label="too steep (waves would break)")
    a1.plot(p["tp"], p["hs"], "o", ms=3, color="#2e86c1", label="sampled sea states")
    a1.set_ylim(0, 8.5)
    a1.set_xlabel("Peak period Tp (s)")
    a1.set_ylabel("Significant wave height Hs (m)")
    a1.legend(fontsize=8)
    a2.plot(p["t_eq"], p["pga_g"], "o", ms=3, color="#c0392b")
    a2.set_xlabel("Waves-only time before the earthquake (s)")
    a2.set_ylabel("Target PGA (g)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.suptitle(f"Latin hypercube coverage of {len(cases)} load cases")
    fig.tight_layout()
    fig.savefig(OUT / "parameter_coverage.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_spectra()
    w = fig_wave_example()
    fig_earthquakes()
    t0 = time.time()
    cases = generate_loads(100, seed=0)
    gen_time = time.time() - t0
    fig_combined(cases[:3])
    fig_coverage(cases)
    save_h5(cases, ROOT / "data" / "loads_synthetic_demo.h5")
    f_peak = np.array([np.abs(c.force).max() for c in cases]) / 1e3
    m_peak = np.array([np.abs(c.moment).max() for c in cases]) / 1e3
    dur = np.array([c.t[-1] for c in cases])
    summary = {
        "n_cases": len(cases),
        "generation_time_s": round(gen_time, 1),
        "duration_s": [round(dur.min(), 1), round(dur.max(), 1)],
        "peak_wave_force_MN": [round(f_peak.min(), 2), round(f_peak.max(), 2)],
        "peak_wave_moment_MNm": [round(m_peak.min(), 1), round(m_peak.max(), 1)],
        "kc_range": [round(min(c.params["kc"] for c in cases), 2), round(max(c.params["kc"] for c in cases), 2)],
        "example_Hs6_Tp11": {"peak_force_MN": round(np.abs(w["force"]).max() / 1e3, 2),
                             "drag_share_of_peak": round(np.abs(w["force_drag"]).max() / np.abs(w["force"]).max(), 3),
                             "kc": round(w["kc"], 2)},
        "tp_raised_by_steepness": int(sum(c.params["tp"] > c.params["tp_sampled"] + 1e-9 for c in cases)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
