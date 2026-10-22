"""Part 5 figures and summary table -> results/part5/.

Run after mini_surrogate.train:  .venv\\Scripts\\python.exe -m mini_surrogate.make_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load
from .train import LIQ, OUT

NAMES = {"fno1d": "FNO-1D", "fno": "FNO-2D", "lstm": "LSTM", "bilstm": "BiLSTM"}
COLORS = {"fno1d": "#c0392b", "fno": "#e67e22", "lstm": "#2e86c1", "bilstm": "#7d3c98"}
KEYS = [("rmse", "RMSE of ru"), ("rel_l2", "relative L2"), ("peak_ru_mae", "peak-ru MAE"),
        ("final_ru_mae", "end-of-run ru MAE"), ("liq_accuracy", "liquefied? accuracy"),
        ("liq_onset_mae_s", "liquefaction onset MAE (s)")]


def summarise(results):
    table = {}
    for m in NAMES:
        runs = [v for k, v in results.items() if k.startswith(m + "_s") and v["model"] == m]
        if not runs:
            continue
        table[m] = {"n_seeds": len(runs), "n_params": runs[0]["n_params"],
                    "inference_ms": 1e3 * np.mean([r["inference_s"] for r in runs]),
                    "train_time_s": np.mean([r["train_time_s"] for r in runs])}
        for split in ("val", "test"):
            for k, _ in KEYS:
                vals = np.array([r[split][k] for r in runs])
                table[m][f"{split}_{k}"] = [float(vals.mean()), float(vals.std())]
    return table


def fig_dataset(data, depth, rejected, runtimes):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for split, c in zip(("train", "val", "test"), ("0.4", "#d68910", "#c0392b")):
        if split not in data:
            continue
        peak = data[split].y.numpy().max(-1).max(-1)
        axs[0].hist(peak, bins=np.linspace(0, 1.2, 25), alpha=0.6, color=c, label=f"{split} ({len(peak)})")
        m = data[split].meta
        axs[1].scatter([a["pga_g"] for a in m], [a["dr"] for a in m], s=10, color=c,
                       c=None, alpha=0.8, label=split)
    axs[0].set_xlabel("Largest ru anywhere in the column")
    axs[0].set_ylabel("Cases")
    axs[0].axvline(LIQ, color="k", ls="--", lw=0.8)
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("Target PGA (g)")
    axs[1].set_ylabel("Relative density Dr")
    axs[1].legend(fontsize=8)
    axs[2].hist(runtimes, bins=30, color="0.4")
    axs[2].set_xlabel("OpenSees run time per case (s)")
    axs[2].set_title(f"median {np.median(runtimes):.0f} s; {len(rejected)} rejected runs", fontsize=9)
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_overview.png", dpi=150)


def fig_fields(data, depth, preds, idx_list, fname):
    test = data["test"]
    t = np.arange(test.y.shape[-1]) * 0.05
    models = [m for m in NAMES if m in preds]
    fig, axs = plt.subplots(len(idx_list), 1 + len(models), figsize=(4 * (1 + len(models)), 3.4 * len(idx_list)),
                            squeeze=False)
    for row, i in enumerate(idx_list):
        a = test.meta[i]
        fields = [("OpenSees", test.y[i].numpy())] + [(NAMES[m], preds[m][i]) for m in models]
        for col, (lab, f) in enumerate(fields):
            ax = axs[row, col]
            im = ax.pcolormesh(t, depth, f, vmin=0, vmax=1, cmap="magma_r", shading="auto")
            ax.invert_yaxis()
            err = "" if col == 0 else f"  (RMSE {np.sqrt(((f - fields[0][1]) ** 2).mean()):.3f})"
            ax.set_title(f"{lab}{err}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{a['event']}, {a['pga_g']:.2f} g, Dr {a['dr']:.2f}\nDepth (m)", fontsize=8)
            if row == len(idx_list) - 1:
                ax.set_xlabel("Time (s)")
    fig.subplots_adjust(hspace=0.35)
    fig.colorbar(im, ax=axs, shrink=0.6, label="ru (1 = liquefied)")
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")


def fig_histories(data, depth, preds, i, fname):
    test = data["test"]
    t = np.arange(test.y.shape[-1]) * 0.05
    a = test.meta[i]
    fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(t, test.x[i, :, 0].numpy(), color="k", lw=0.6)
    axs[0].set_ylabel("Base acc. (g)")
    axs[0].set_title(f"Unseen earthquake: {a['record']} ({a['event']}, M {a['magnitude']:.1f}), "
                     f"PGA {a['pga_g']:.2f} g, Dr {a['dr']:.2f}, k {a['perm']:.1e} m/s", fontsize=9)
    for ax, d in zip(axs[1:], (2.0, 5.0, 8.0)):
        j = int(np.argmin(np.abs(depth - d)))
        ax.plot(t, test.y[i, j].numpy(), color="k", lw=1.6, label="OpenSees")
        for m in preds:
            ax.plot(t, preds[m][i, j], color=COLORS[m], lw=1, label=NAMES[m])
        ax.set_ylabel(f"ru at {depth[j]:.1f} m")
        ax.set_ylim(-0.1, 1.1)
    axs[1].legend(fontsize=8, ncol=4)
    axs[-1].set_xlabel("Time (s)")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)


def fig_metrics(table):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
    models = list(table)
    for ax, (k, lab) in zip(axs, [KEYS[0], KEYS[2], KEYS[4], KEYS[5]]):
        mean = [table[m][f"test_{k}"][0] for m in models]
        std = [table[m][f"test_{k}"][1] for m in models]
        ax.bar([NAMES[m] for m in models], mean, yerr=std, color=[COLORS[m] for m in models], capsize=4)
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Test set (earthquakes never seen in training), mean ± std over 3 training seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "test_metrics.png", dpi=150)


def fig_peak_scatter(data, preds):
    y = data["test"].y.numpy().max(-1).ravel()
    fig, axs = plt.subplots(1, len(preds), figsize=(4 * len(preds), 3.8), squeeze=False)
    for ax, m in zip(axs[0], preds):
        p = preds[m].max(-1).ravel()
        ax.scatter(y, p, s=4, alpha=0.4, color=COLORS[m])
        ax.plot([0, 1.2], [0, 1.2], "k--", lw=0.8)
        ax.set_xlabel("OpenSees peak ru")
        ax.set_ylabel(f"{NAMES[m]} peak ru")
        ax.set_title(f"MAE {np.abs(p - y).mean():.3f}", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "peak_ru_scatter.png", dpi=150)


def fig_training():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in NAMES:
        f = OUT / "models" / f"{m}_s0_history.npy"
        if f.exists():
            h = np.load(f)
            ax.plot(h[:, 0], color=COLORS[m], lw=0.8, ls="--")
            ax.plot(h[:, 1], color=COLORS[m], lw=1.2, label=f"{NAMES[m]} (dashed = train)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE of ru")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)


def paired_stats(data, a="fno1d", others=("lstm", "bilstm"), n_boot=5000, seed=0):
    """Per-case comparison on the test set, predictions averaged over the 3 seeds' errors.
    95% intervals from a bootstrap that resamples whole earthquakes (cases of one event are
    correlated, so resampling cases would overstate the certainty)."""
    y = data["test"].y.numpy()
    ev = np.array([m["event"] for m in data["test"].meta])
    yl, yo = y.max(-1) >= LIQ, np.argmax(y >= LIQ, -1) * 0.05

    def per_case(m):
        p = np.stack([np.load(OUT / "models" / f"{m}_s{s}_test_pred.npy") for s in range(3)])
        rmse = np.sqrt(((p - y) ** 2).mean(axis=(-1, -2))).mean(0)
        pl, po = p.max(-1) >= LIQ, np.argmax(p >= LIQ, -1) * 0.05
        with np.errstate(all="ignore"):
            onset = np.nanmean(np.where(pl & yl, np.abs(po - yo), np.nan), axis=(0, 2))
        return {"rmse": rmse, "onset_s": onset, "liq_acc": (pl == yl).mean(-1).mean(0)}

    rng = np.random.default_rng(seed)
    evs = np.unique(ev)
    idx_of = {e: np.where(ev == e)[0] for e in evs}
    A = per_case(a)
    out = {}
    for b in others:
        B = per_case(b)
        for k in A:
            diff = A[k] - B[k]
            d = diff[np.isfinite(diff)]                  # onset only exists where both liquefy
            boots = [np.nanmean(diff[np.concatenate([idx_of[e] for e in rng.choice(evs, len(evs))])])
                     for _ in range(n_boot)]
            out[f"{a}_minus_{b}_{k}"] = {"mean": float(np.nanmean(diff)),
                                         "ci95": [float(x) for x in np.nanpercentile(boots, [2.5, 97.5])],
                                         "a_better_frac": float(np.mean((d < 0) if k != "liq_acc" else (d > 0))),
                                         "n_cases": int(len(d))}
    return out


def main():
    results = json.loads((OUT / "results.json").read_text())
    data, depth, rejected = load()
    runtimes = np.array([a["runtime"] for s in data.values() for a in s.meta])
    table = summarise(results)
    best_seed = {m: min((v["test"]["rmse"], v["seed"]) for k, v in results.items()
                        if k.startswith(m + "_s") and v["model"] == m)[1] for m in table}
    # figures use seed 0 for every model (not the best test seed: no cherry-picking)
    preds = {m: np.load(OUT / "models" / f"{m}_s0_test_pred.npy") for m in table}
    fig_dataset(data, depth, rejected, runtimes)
    peak = data["test"].y.numpy().max(-1).max(-1)
    order = np.argsort(peak)
    pick = [int(order[len(order) // 4]), int(order[len(order) // 2]), int(order[-len(order) // 8])]
    fig_fields(data, depth, preds, pick, "test_fields.png")
    fig_histories(data, depth, preds, pick[-1], "test_histories.png")
    fig_metrics(table)
    fig_peak_scatter(data, preds)
    fig_training()
    summary = {"table": table, "opensees_runtime_s": {"median": float(np.median(runtimes)),
                                                      "mean": float(runtimes.mean())},
               "speedup_vs_opensees_median": {m: float(np.median(runtimes) / (table[m]["inference_ms"] / 1e3))
                                              for m in table},
               "n": {k: len(v.x) for k, v in data.items()}, "rejected": len(rejected),
               "paired_test_comparison": paired_stats(data),
               "figure_cases": [data["test"].meta[i]["record"] for i in pick], "_best_test_seed": best_seed}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
