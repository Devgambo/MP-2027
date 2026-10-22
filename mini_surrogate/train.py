"""Train and evaluate the part 5 surrogates.

    .venv\\Scripts\\python.exe -m mini_surrogate.train                 # all models, 3 seeds, 200 epochs
    .venv\\Scripts\\python.exe -m mini_surrogate.train --models fno --seeds 0

Model selection uses the validation set only (best epoch); the test set (held-out
earthquakes) is touched once, at the end. Results -> results/part5/.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import Normaliser, load
from .models import build, n_params

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "part5"
LIQ = 0.9           # ru above this counts as liquefied
DT_OUT = 0.05


def metrics(pred, y):
    """Engineering error measures on ru fields [n, nz, nt] (numpy)."""
    err = pred - y
    peak_p, peak_y = pred.max(-1), y.max(-1)
    liq_p, liq_y = peak_p >= LIQ, peak_y >= LIQ
    both = liq_p & liq_y

    def onset(a):                       # first time ru >= LIQ, per (case, depth)
        return np.argmax(a >= LIQ, axis=-1) * DT_OUT
    onset_err = np.abs(onset(pred) - onset(y))[both]
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "rel_l2": float(np.sqrt((err ** 2).sum() / (y ** 2).sum())),
        "rel_l2_per_case_median": float(np.median(np.linalg.norm(err.reshape(len(y), -1), axis=1)
                                                  / np.linalg.norm(y.reshape(len(y), -1), axis=1))),
        "peak_ru_mae": float(np.abs(peak_p - peak_y).mean()),
        "final_ru_mae": float(np.abs(pred[..., -1] - y[..., -1]).mean()),
        "liq_accuracy": float((liq_p == liq_y).mean()),
        "liq_onset_mae_s": float(onset_err.mean()) if both.any() else float("nan"),
    }


def predict(model, norm, x, device, batch=16):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(norm(x[i:i + batch].to(device))).cpu() for i in range(0, len(x), batch)])


def train_one(name, data, seed, epochs, device, lr=1e-3, batch=8, verbose=True, model_kw=None,
              weight_decay=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = data["train"], data["val"]
    norm = Normaliser(tr.x)
    model = build(name, **(model_kw or {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * -(-len(tr.x) // batch))
    best, best_model, hist = np.inf, None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr.x))
        loss_sum = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            x, y = norm(tr.x[idx].to(device)), tr.y[idx].to(device)
            loss = torch.mean((model(x) - y) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
        val_rmse = float(torch.sqrt(torch.mean((predict(model, norm, va.x, device) - va.y) ** 2)))
        hist.append((np.sqrt(loss_sum / len(tr.x)), val_rmse))
        if val_rmse < best:
            best, best_model, best_ep = val_rmse, copy.deepcopy(model), ep
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"  {name} s{seed} ep {ep:3d} train {hist[-1][0]:.4f} val {val_rmse:.4f}", flush=True)
    model = best_model                     # weights from the best validation epoch
    return model, norm, dict(best_epoch=best_ep, best_val_rmse=best, train_time_s=time.time() - t0,
                             history=hist)


def inference_time(model, norm, x, device, n=50):
    model.eval()
    xb = x[:1].to(device)
    with torch.no_grad():
        for _ in range(5):
            model(norm(xb))
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(norm(xb))
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["fno1d", "fno", "lstm", "bilstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, depth, rejected = load()
    print({k: len(v.x) for k, v in data.items()}, "rejected:", len(rejected), "device:", device)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name in a.models:
        for seed in a.seeds:
            model, norm, info = train_one(name, data, seed, a.epochs, device)
            pred = {k: predict(model, norm, data[k].x, device).numpy() for k in ("val", "test")}
            np.save(OUT / "models" / f"{name}_s{seed}_test_pred.npy", pred["test"])
            torch.save({"state": model.state_dict(), "mean": norm.mean, "std": norm.std},
                       OUT / "models" / f"{name}_s{seed}.pt")
            results[f"{name}_s{seed}"] = dict(
                model=name, seed=seed, n_params=n_params(model),
                val=metrics(pred["val"], data["val"].y.numpy()),
                test=metrics(pred["test"], data["test"].y.numpy()),
                inference_s=inference_time(model, norm, data["test"].x, device),
                **{k: v for k, v in info.items() if k != "history"})
            np.save(OUT / "models" / f"{name}_s{seed}_history.npy", np.array(info["history"]))
            r = results[f"{name}_s{seed}"]
            print(f"{name} s{seed}: params {r['n_params']:,}  val rmse {r['val']['rmse']:.4f}  "
                  f"test rmse {r['test']['rmse']:.4f}  peak-ru MAE {r['test']['peak_ru_mae']:.3f}", flush=True)
            res_path.write_text(json.dumps(results, indent=1))
    results["_data"] = dict(n={k: len(v.x) for k, v in data.items()}, rejected=rejected)
    res_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
