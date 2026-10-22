"""Small FNO design sweep, scored on the validation set only (the test set is not touched).

    .venv\\Scripts\\python.exe -m mini_surrogate.sweep
Writes results/part5/sweep.json.
"""
import json

import torch

from .data import load
from .models import n_params
from .train import OUT, train_one

CONFIGS = {
    "fno2d_m64_w32": ("fno", dict(modes=(10, 64), width=32), 1e-4),
    "fno2d_m256_w16": ("fno", dict(modes=(10, 256), width=16), 1e-4),
    "fno2d_m128_w24_wd": ("fno", dict(modes=(10, 128), width=24), 1e-2),
    "fno1d_m64_w48": ("fno1d", dict(modes=64, width=48), 1e-4),
    "fno1d_m128_w32": ("fno1d", dict(modes=128, width=32), 1e-4),
    "fno1d_m256_w24": ("fno1d", dict(modes=256, width=24), 1e-4),
    "fno1d_m128_w32_wd": ("fno1d", dict(modes=128, width=32), 1e-2),
}
# round 2 (after round 1 showed more time modes help and best epochs near the end): 200 epochs
CONFIGS_LONG = {
    "fno1d_m256_w24_e200": ("fno1d", dict(modes=256, width=24), 1e-4),
    "fno1d_m500_w16_e200": ("fno1d", dict(modes=500, width=16), 1e-4),
    "fno1d_m500_w32_e200": ("fno1d", dict(modes=500, width=32), 1e-4),
    "fno2d_m500_w12_e200": ("fno", dict(modes=(10, 500), width=12), 1e-4),
}


def main(epochs=100, long_epochs=200):
    data, _, _ = load()
    path = OUT / "sweep.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    jobs = [(t, c, epochs) for t, c in CONFIGS.items()] + [(t, c, long_epochs) for t, c in CONFIGS_LONG.items()]
    for tag, (name, kw, wd), n_ep in jobs:
        if tag in res:
            continue
        model, _, info = train_one(name, data, 0, n_ep, "cuda", verbose=False, model_kw=kw, weight_decay=wd)
        res[tag] = dict(model=name, kw={k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()},
                        weight_decay=wd, n_params=n_params(model), best_epoch=info["best_epoch"],
                        val_rmse=info["best_val_rmse"], final_train_rmse=float(info["history"][-1][0]),
                        train_time_s=info["train_time_s"])
        print(tag, {k: v for k, v in res[tag].items() if k not in ("kw",)}, flush=True)
        path.write_text(json.dumps(res, indent=1))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
