# Part 5 explained: the mini-surrogate

This file explains roadmap part 5 in plain terms: what we simulated, what the AI models learned,
how we tested them fairly, and what the result means. The soil side is explained from scratch.
The ML side is explained more briefly, since that is your home ground, but the soil meaning of
every number is spelled out. The short technical version is in `docs/part5_mini_surrogate.md`.

Figures referred to below are in `results/part5/`.

---

## 1. Why part 5 exists

The final project wants an AI that copies a slow simulator (OpenSees) of a pile in the seabed.
Before building that big, expensive version, part 5 runs the **whole pipeline on a small, cheap
problem**: a column of sand with no pile.

1. Make lots of simulations (the training data).
2. Train an FNO to copy them.
3. Train an LSTM (the standard "time series" model) on the same data.
4. Test both on earthquakes neither has ever seen.

Roadmap "done when": *the FNO beats the LSTM on earthquakes it never saw.* Section 7 gives the
honest answer: partly.

Why bother with a mini version? If something is broken (data format, splits, training, metrics),
we find out now, when one simulation takes a minute, not later, when it takes fifteen.

---

## 2. The soil problem being copied (recap of part 3)

Picture a **10 m tall column of wet sand**, the water filling every gap between the grains. The
bottom sits on rock; the top is the seabed surface, where water can escape.

When an earthquake shakes the rock, the sand is sheared back and forth. Loose sand wants to
pack tighter when shaken, but the water in the gaps can't get out fast enough. So the water gets
**squeezed**, and its pressure rises. That extra pressure is called **excess pore-water pressure**.

We measure it with one number, **ru** (the pore-pressure ratio):

- **ru = 0**: the water carries no extra load; the grains press firmly on each other. Solid ground.
- **ru = 0.5**: the water has taken over half the weight the grains used to carry. Softer.
- **ru ≈ 1**: the water carries everything; grains are barely touching. The sand behaves like a
  thick liquid. This is **liquefaction**: buildings tilt, piles lose grip.

After the shaking stops, the water slowly drains out through the top and ru falls again. How fast
depends on the **permeability** (how easily water flows through the sand).

So the thing to predict is a **map of ru**: at every depth (20 points, every 0.5 m) and every
moment (1,000 time steps, every 0.05 s, over 50 s). In `test_fields.png`, each panel is one such
map: depth down the side, time along the bottom, dark = liquefied.

---

## 3. Making the training data

### 3.1 What changes from case to case

Every simulation (a "case") differs in four things:

| What | Range | Plain meaning |
| --- | --- | --- |
| Earthquake record | 353 real recordings (part 4) | The shape of the shaking: how long it lasts, how fast it vibrates |
| Strength (PGA) | 0.05 to 0.40 g | The biggest jolt. 0.05 g is a mild tremor, 0.4 g very strong |
| Sand density (Dr) | 35% to 75% | Loose (35%: liquefies easily) to dense (75%: hard to liquefy) |
| Permeability (k) | 0.00001 to 0.001 m/s | Tight silty sand (water trapped) to clean coarse sand (water drains fast) |

Strength, density and permeability were spread evenly with Latin hypercube sampling (part 4).

### 3.2 How much data

- **330 simulations**, run 10 at a time on your laptop's CPU. They took 95 minutes in total
  (16 CPU-hours of work).
- A typical case took 86 s; a hard one (loose sand, strong shaking) up to 30 minutes, because
  liquefying soil is numerically difficult and the solver has to take tiny steps.
- **None crashed.** The roadmap warns that liquefaction runs often fail to converge; ours didn't,
  thanks to the backup solver added in part 3.

### 3.3 The fair split

Like part 4, whole earthquakes are held out:

| Group | Cases | Earthquakes | Used for |
| --- | --- | --- | --- |
| Training | 230 | 26 | The models learn from these |
| Validation | 35 | 3 | Choosing settings and the best training epoch |
| Test | 65 | 7 (Kocaeli 7.5, Tabas 7.35, Northridge 6.7, San Fernando 6.6, ...) | Touched once, at the very end |

About 60% of cases liquefy somewhere in the column (`dataset_overview.png`, left). That's on
purpose (the project is about liquefaction) but it means the data leans that way.

---

## 4. The two kinds of model

Both models get exactly the same inputs at each time step:

- the base **acceleration** (the shaking itself),
- the base **velocity**,
- the running total of **shaking energy so far** (Arias intensity; it only ever goes up),
- the sand **density** and **permeability** (the same at every step).

Both output the full ru map. Both have about the same number of parameters (1.0 to 1.5 million),
so neither wins just by being bigger.

### 4.1 LSTM: the step-by-step reader

An LSTM reads the shaking **one moment at a time**, carrying a memory forward. At each step it
updates its memory and says "ru at all 20 depths is now ...". This matches the physics nicely:
pore pressure is a **running total** of past shaking, and the LSTM is literally a running-total
machine.

The normal LSTM is **causal**: it never sees the future. We also trained a **BiLSTM**, which reads
the record both forwards and backwards. That gives it the same "sees the whole record at once"
advantage the FNO has, which makes it a stronger and fairer extra baseline.

### 4.2 FNO: the whole-picture reader

An FNO looks at the **whole 50 s at once**. It breaks the signal into waves of different
frequencies (Fourier modes, like a music equaliser does), learns how to reshape each one, and
puts them back together. It repeats this a few times with simple nonlinear steps in between.

Two versions:

- **FNO-1D**: works along time only; the 20 depths are 20 output channels (same layout as the LSTM).
- **FNO-2D**: works on the whole depth × time picture at once.

---

## 5. The first FNO lost, and why (the "mode" lesson)

The first FNO had **64 time modes**: it only kept the 64 slowest waves in the signal. Over a 50 s
window, that means it ignored everything vibrating faster than **1.3 times per second**.

But earthquakes do most of their shaking at **1 to 10 times per second**. So the FNO was almost
blind to the shaking itself. Unsurprisingly, the LSTM beat it badly (error 0.073 vs 0.143).

We fixed this with a small **sweep**: try several FNO settings, judge them **only on the
validation set**, keep the best. The test set was not touched. The big win was keeping **all 500
modes** (validation error 0.153 → 0.107). The full table is in `docs/part5_mini_surrogate.md`.

Lesson for later parts: an FNO must be allowed to see the frequencies that matter for the
physics. That is easy to get wrong, and important to say in the report.

---

## 6. Results on the unseen earthquakes

Each model was trained three times with different random seeds (different starting weights),
to see how much luck matters. Numbers are average ± spread over the three.

### 6.1 What each number means

- **RMSE of ru**: average size of the error across the whole map. 0.09 means "typically off by
  0.09 in ru", on a scale where 1 is liquefied.
- **Peak-ru error**: at each depth, how wrong is the *highest* ru reached? (Did it nearly liquefy?)
- **Liquefied? accuracy**: at each depth, does the model correctly say "this layer liquefies"
  (ru reaches 0.9) or "it doesn't"? This is the yes/no an engineer cares about most.
- **Onset error**: for layers that liquefy, how many **seconds** off is the predicted moment it
  happens? Timing matters: if liquefaction starts while the shaking is still strong, the damage
  is worse.
- **End-of-run ru error**: how much pressure is left 50 s after the start (after the shaking)?
  This is about drainage.

### 6.2 The table

| | FNO-1D | FNO-2D | LSTM | BiLSTM |
| --- | --- | --- | --- | --- |
| RMSE of ru (lower = better) | 0.091 ± 0.002 | 0.106 | 0.089 ± 0.010 | **0.079** |
| Peak-ru error | 0.066 | 0.074 | 0.062 | **0.055** |
| Liquefied? accuracy (higher = better) | 91.4% | 90.7% | 88.3% | **91.7%** |
| Onset error (seconds) | **2.8 s** | 3.5 s | 3.8 s | 4.0 s |
| End-of-run ru error | **0.054** | 0.066 | 0.064 | 0.070 |
| Training time | **1.4 min** | 6 min | 20 min | 38 min |
| Time to predict one case | **4 ms** | 4 ms | 104 ms | 45 ms |

OpenSees takes about **86 s** for one case (median). The FNO answers in **4 milliseconds**: roughly
**22,000 times faster**.

### 6.3 Is the difference real or luck?

The test set has only 7 earthquakes, so we checked carefully: compare the models **case by case**
on the same tests, then repeatedly re-draw the 7 earthquakes at random (a "bootstrap") to see how
stable the difference is.

- **Onset timing:** the FNO is better by about 1.4 s, and better in 26 of the 28 cases where both
  predict liquefaction. This difference is solid.
- **Overall map error (RMSE), FNO vs LSTM:** basically a **tie**. The FNO is slightly better on 62% of
  cases, but the difference is within the noise.
- **Liquefied? accuracy:** the FNO is about 3 points better than the LSTM, but that is borderline
  (could be noise).
- **BiLSTM vs FNO on RMSE:** the BiLSTM is a little better, also borderline.

### 6.4 The pictures

- `test_histories.png`: one unseen Kocaeli case, ru over time at 2, 5 and 8 m. All models follow
  the rise to liquefaction well. The FNOs have small **jitter** (high-frequency wiggles); the
  LSTMs are smooth but react a bit late or early in places.
- `test_fields.png`: three test cases (mild, medium, strong) as full maps. All models get the
  overall picture right: which layers liquefy and roughly when.
- `peak_ru_scatter.png`: predicted vs true peak ru. Points near the diagonal are good.

---

## 7. So, did part 5 pass?

**Partly, and that is fine.**

- The FNO **beats the LSTM** on the engineering timing question (when liquefaction starts), on
  how much pressure is left afterwards, on reliability (much smaller spread between seeds), and by
  a mile on speed.
- It **ties** with the LSTM on overall map error and is slightly worse on peak ru.
- The **BiLSTM** is best on map error, but worst on timing.

The roadmap says: if the FNO does not clearly win, stop and find out why. Our reasons:

1. **Too little data for the FNO.** It fits the training set very well (error about 0.03) but much
   worse on new earthquakes (about 0.11): it memorises individual records. This is exactly the
   "data is scarce" situation the physics check in part 10 is designed to help. So this result
   actually *motivates* the research question.
2. **Pore pressure is a running total.** The LSTM's step-by-step memory is a natural fit; the FNO
   has to learn "adding up over time" indirectly.
3. **Jitter** from keeping all frequency modes. A simple smoothing step should help.
4. **Small validation set** (3 earthquakes), so choosing the best settings was a bit noisy.

For the Semester 1 presentation this is a complete, honest result: a working pipeline, a checked
dataset, FNO vs LSTM on unseen earthquakes, a clear speed-up, and a diagnosis of what to fix.

---

## 8. Honest limitations

- Everything inherits part 3's soil-model limits: one-dimensional column, settlement too small,
  drainage a bit slow.
- The data leans liquefied (about 60% of cases).
- 7 test earthquakes: differences smaller than about 0.01 in RMSE can't be told apart.
- The FNO got a settings sweep, the LSTMs didn't. That favours the FNO, yet it still didn't win on
  every metric, which makes the comparison conservative.
- 14 long records were cut at 40 s; rare extreme ru spikes were clipped.

---

## 9. How this connects to the rest of the project

- **Part 6 (pile model):** same simulation style, now with a pile and wave forces.
- **Part 7 (dataset factory):** `mini_surrogate/dataset.py` is the template: plan cases, run in
  parallel, resume after interruptions, reject failures, save to HDF5.
- **Parts 8-9 (FNO on the pile):** use enough Fourier modes for the shaking band; keep LSTM and
  BiLSTM as baselines; report timing and liquefaction metrics, not only RMSE.
- **Part 10 (physics loss):** the overfitting seen here is the gap a physics check might close.

---

## 10. Files

| Path | What it does |
| --- | --- |
| `mini_surrogate/dataset.py` | Plans the 330 cases and runs OpenSees in parallel → `data/column_dataset.h5` |
| `mini_surrogate/data.py` | Turns the HDF5 file into model inputs and targets |
| `mini_surrogate/models.py` | The FNO-1D, FNO-2D, LSTM and BiLSTM |
| `mini_surrogate/train.py` | Trains, picks the best epoch on validation, scores on test |
| `mini_surrogate/sweep.py` | The FNO settings sweep (validation only) |
| `mini_surrogate/make_figures.py` | Figures, summary table, the case-by-case statistics |
| `tests/test_mini_surrogate.py` | 8 checks (shapes, equal size, LSTM never peeks at the future, metrics) |

Trained models (`results/part5/models/`, 149 MB) and the dataset (33 MB) are not in git; the
scripts regenerate them.

---

## 11. Glossary

| Term | Meaning |
| --- | --- |
| Surrogate | A fast AI copy of a slow simulator |
| Soil column | A vertical strip of soil, simulated alone (no pile) |
| Pore-water pressure | Pressure of the water in the gaps between sand grains |
| Excess pore pressure | The extra pressure caused by shaking, above the normal (still-water) value |
| ru | Excess pore pressure ÷ the weight the grains carried before shaking. 0 = solid, 1 = liquefied |
| Liquefaction | Sand losing its strength because the water carries all the load |
| Onset | The moment a layer first liquefies (here: ru first reaches 0.9) |
| Dissipation | Pore pressure falling after shaking as water drains away |
| Relative density (Dr) | How tightly packed the sand is, 0% (loosest) to 100% (densest) |
| Permeability (k) | How easily water flows through the sand |
| PGA | Peak ground acceleration: the biggest jolt of the earthquake, in g |
| Arias intensity | A running total of shaking energy; only increases |
| Held-out earthquake | An earthquake whose records are kept out of training entirely |
| Fourier mode | One of the pure waves a signal is broken into; more modes = finer detail kept |
| Causal | Uses only the past, never the future |
| BiLSTM | An LSTM that reads the sequence both forwards and backwards |
| Seed | The random starting point of training; different seeds give slightly different models |
| RMSE | Root-mean-square error: the typical size of the error |
| Bootstrap | Re-drawing the test set at random many times to see how stable a result is |
| 95% interval | The range a difference falls in 95% of bootstrap re-draws; if it includes 0, the difference might be luck |
| Overfitting | Learning the training examples too specifically, so new cases are predicted worse |
| Speed-up | How many times faster the surrogate is than OpenSees |
