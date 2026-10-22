# Part 5: Mini-surrogate on the soil column

Roadmap part 5. "Do: run 200-300 soil-column cases, train a small FNO that maps base shaking to
pore pressure over depth and time, compare with a simple LSTM. Done when: the FNO beats the LSTM
on earthquakes it never saw."

**Result in one line:** after a validation-only design sweep, the FNO matches the LSTM on overall
field error (statistical tie) and clearly beats it on *when* liquefaction starts, while being far
more stable across seeds and 27x faster at inference. It does not beat it on every metric, and a
bidirectional LSTM has the lowest field error. So the "done when" is met only partly; see Verdict.

```
.venv\Scripts\python.exe -m mini_surrogate.dataset --workers 10   # 330 OpenSees runs (resumable)
.venv\Scripts\python.exe -m mini_surrogate.sweep                  # FNO design sweep (validation only)
.venv\Scripts\python.exe -m mini_surrogate.train                  # 4 models x 3 seeds x 200 epochs
.venv\Scripts\python.exe -m mini_surrogate.make_figures           # figures + summary.json
```

## Dataset (`mini_surrogate/dataset.py` -> `data/column_dataset.h5`, 33 MB)

| Item | Value |
| --- | --- |
| Model | Part 3 column: 10 m uniform PM4Sand, SSPquadUP, dz = 0.5 m, rigid impermeable base, drained top |
| Varied (Latin hypercube per split) | PGA 0.05-0.40 g, Dr 0.35-0.75 (calibrated range), k = 1e-5 to 1e-3 m/s (log-uniform) |
| Records | 353 PEER horizontal records; part 4 processing; scale factor in [1/4, 4]; least-used record first |
| Split | `split_by_event` (magnitude-stratified, sequences together): train 230 cases / 26 events, val 35 / 3, test 65 / 7 |
| Test events | Kocaeli (M7.5), Tabas (7.35), Northridge (6.7), San Fernando (6.6), Helena Montana-01/02, Westmorland |
| Time window | 50 s per case: record (cut at 40 s, 14 of 353 records, 0.5 s end taper) then zeros |
| Stored | input acc at 0.01 s; ru (20 depths x 1001 steps at 0.05 s), settlement, surface displacement and acceleration |
| Cost | 0 of 330 runs rejected; run time median 86 s, mean 172 s, max 1,823 s; 15.7 CPU-hours; 94.5 min on 10 workers |
| Content | Peak ru >= 0.9 somewhere in the column: train 64%, val 80%, test 54% of cases (dataset leans liquefied) |

ru is clipped to [-1, 1.5]; values outside are rare (< 1% of points, mostly the top 2 m: brief
dilation spikes in liquefied sand). The surface node (ru = 0 by the drained boundary) is dropped.

## Inputs and models

Same inputs for every model, per 0.05 s step: base acceleration (anti-alias filtered), base
velocity, cumulative Arias intensity, Dr and log10 k (constant in time). Per-feature
standardisation from the training set. Output: ru(20 depths, 1000 steps).

| Model | Layout | Parameters |
| --- | --- | --- |
| FNO-1D | `neuralop` FNO over time, depths as 20 output channels, 500 time modes (all), width 32, 4 layers, 25% time padding | 1.04 M |
| FNO-2D | FNO over (depth, time), inputs broadcast over depth, modes (10, 500), width 12, padding (25%, 12.5%) | 1.45 M |
| LSTM | 3 layers x 256, causal, linear head to 20 depths | 1.33 M |
| BiLSTM | 3 layers x 160 per direction, sees past and future like the FNO | 1.45 M |

Training: MSE on ru, AdamW (lr 1e-3, wd 1e-4), one-cycle schedule, batch 8, 200 epochs, gradient
clip 1; the epoch with the lowest validation RMSE is kept. Three seeds per model. RTX 3050 6 GB.

### FNO design sweep (`results/part5/sweep.json`, seed 0, validation RMSE only)

| Config | Params | Val RMSE | Note |
| --- | --- | --- | --- |
| FNO-2D, modes (10, 64), width 32 (first try) | 1.37 M | 0.153 | best epoch 27, then overfits |
| FNO-2D, modes (10, 256), width 16 | 1.32 M | 0.121 | |
| FNO-1D, 64 / 128 / 256 modes (100 epochs) | 0.28-0.34 M | 0.139 / 0.152 / 0.121 | weight decay 1e-2 did not help |
| FNO-1D, 256 modes, 200 epochs | 0.31 M | 0.120 | |
| FNO-1D, 500 modes, width 16 / **32** | 0.26 / 1.04 M | 0.130 / **0.107** | chosen |
| FNO-2D, modes (10, 500), width 12 | 1.45 M | 0.115 | chosen as 2D variant |

Diagnosis behind the sweep: 64 modes over a 50 s window keep only frequencies below 1.3 Hz, but
earthquake energy sits at 1-10 Hz, so the spectral layers barely saw the shaking. Keeping all modes
was the single biggest gain. The LSTM was not tuned (default size matched to the FNO), which if
anything favours the FNO.

## Results on the test set (65 cases from 7 unseen earthquakes; mean ± std over 3 seeds)

| Metric | FNO-1D | FNO-2D | LSTM | BiLSTM |
| --- | --- | --- | --- | --- |
| RMSE of ru | 0.091 ± 0.002 | 0.106 ± 0.008 | 0.089 ± 0.010 | **0.079 ± 0.005** |
| Relative L2 | 0.170 | 0.199 | 0.166 | **0.147** |
| Peak-ru MAE (per depth) | 0.066 ± 0.001 | 0.074 ± 0.006 | 0.062 ± 0.007 | **0.055 ± 0.008** |
| End-of-run ru MAE | **0.054 ± 0.005** | 0.066 | 0.064 | 0.070 |
| Liquefied (peak ru >= 0.9)? accuracy per depth | 0.914 ± 0.005 | 0.907 | 0.883 ± 0.026 | **0.917 ± 0.010** |
| Liquefaction onset time MAE (s) | **2.80 ± 0.22** | 3.49 | 3.83 ± 0.23 | 4.00 ± 0.43 |
| Training time | **84 s** | 349 s | 1,205 s | 2,270 s |
| Inference, one case (GPU) | **3.9 ms** | 4.0 ms | 104 ms | 45 ms |
| Speed-up vs OpenSees (median 86 s) | **~22,000x** | ~21,500x | ~830x | ~1,900x |

Paired comparison (same test cases, errors averaged over seeds; 95% interval from a bootstrap that
resamples whole earthquakes, because cases of one event are correlated):

| FNO-1D minus ... | RMSE | Onset error (s) | Liquefied? accuracy |
| --- | --- | --- | --- |
| LSTM | -0.003 [-0.011, +0.001], FNO better on 62% of cases | **-1.35 [-1.79, -0.86]**, better on 93% of 28 cases | +0.031 [-0.004, +0.051] |
| BiLSTM | +0.009 [-0.001, +0.016], FNO better on 46% | **-1.59 [-2.12, -0.79]**, better on 86% | -0.004 [-0.024, +0.016] |

## Verdict against "done when"

- **Versus the LSTM asked for in the roadmap:** the FNO-1D wins clearly on liquefaction timing,
  is slightly better on liquefaction detection and end-of-run ru, has 5x smaller seed-to-seed
  spread, trains 14x faster and predicts 27x faster. On overall field error it is a statistical
  tie (FNO slightly ahead on 62% of cases); on peak ru the LSTM is slightly better.
- **The bidirectional LSTM** (an extra, stronger baseline) has the lowest field and peak errors,
  but the worst onset timing.
- So: "FNO beats LSTM" holds on the engineering timing metric and on cost, not across the board.
  This is an honest, presentable Semester 1 result; the roadmap's instruction for a non-win
  ("find out why") is answered below.

## Why the FNO does not win outright

1. **Small data, big overfitting gap.** Train RMSE about 0.03 against validation about 0.11 for every
   FNO variant. With 230 cases, the global spectral filters fit individual records. This is exactly
   the data-scarcity regime the physics loss (part 10) is meant to help.
2. **Pore pressure is a running total of past shaking.** An LSTM carries a memory forward step by
   step, a natural fit. The FNO has no notion of causality; it must learn accumulation through
   global filters.
3. **High-frequency output jitter.** Keeping all 500 modes lets the FNO see the shaking, but its
   outputs then carry small high-frequency wiggles (visible in `test_histories.png`). A smoothness
   penalty or output low-pass is an easy next step.
4. **Validation set of 3 earthquakes** makes model selection noisy (validation ranking did not
   perfectly match test ranking).

## Implications for parts 8-10

- Use enough time modes to cover the shaking band (>= 10 Hz at the chosen time step), or move the
  high-frequency input into smooth cumulative features.
- Keep the LSTM and BiLSTM as baselines; report onset timing and liquefaction detection, not only
  RMSE.
- Enlarge the validation set (10+ events) when the 1,000-run dataset exists.
- Try: output smoothing, ensembles of seeds, and the balance-law loss at 100/300 samples.

## Limitations

- Same soil-model limitations as part 3 (settlement too small, slow dissipation, 1D column only).
- Dataset leans liquefied (about 60% of cases); errors on the many fully liquefied cases are low.
- 7 test earthquakes give wide intervals; differences under about 0.01 RMSE are not resolvable.
- FNO got a design sweep, the LSTMs did not (biases towards the FNO, yet it still did not win
  outright).
- Records were cut at 40 s (14 of 353); ru outliers clipped.
- The first (150-epoch) run's test scores were seen before the sweep (FNO 0.143 vs LSTM 0.073
  test RMSE). Every sweep choice was made on validation RMSE only, but the test set is not
  perfectly "untouched" in the strictest sense.

## Files

| Path | What |
| --- | --- |
| `mini_surrogate/dataset.py` | Case planning (LHS, record assignment by split), parallel OpenSees runs, HDF5 writer |
| `mini_surrogate/data.py` | Features, loading, rejection of failed runs, normaliser |
| `mini_surrogate/models.py` | FNO-1D, FNO-2D, LSTM, BiLSTM |
| `mini_surrogate/train.py` | Training, model selection on validation, metrics, inference timing |
| `mini_surrogate/sweep.py` | FNO design sweep (validation only) |
| `mini_surrogate/make_figures.py` | Figures, summary table, paired event-bootstrap statistics |
| `tests/test_mini_surrogate.py` | 8 tests: shapes and matched size, LSTM causality, features, normaliser, metrics |
| `results/part5/` | `dataset_overview.png`, `test_fields.png`, `test_histories.png`, `test_metrics.png`, `peak_ru_scatter.png`, `training_curves.png`, `results.json`, `sweep.json`, `summary.json` |
