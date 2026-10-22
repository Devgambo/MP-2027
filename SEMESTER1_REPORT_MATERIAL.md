# Semester 1 report material: FNO-PINN Monopile Project

Shourya Ranjan · Semester 1 (Oct 7 – Nov 10, 2026)

This file records everything done in Semester 1 in enough detail to write the semester report
and slides from it. Each section says what was done, why, how it was checked, the numbers, and
the honest limitations. Deeper technical notes are in `docs/partN_*.md`, plain-language
explainers in `expln/partN_*_explained.md`, and figures in `results/partN/`. What remains is in
`NEXT_SEMESTER_REMAINING.md`.

---

## 1. Project in one paragraph

Offshore wind turbines stand on **monopiles**: steel tubes about 8 m wide driven 25–40 m into the
seabed. In seismic regions the seabed can be hit by storm waves and an earthquake at the same
time. Shaking raises the pore-water pressure in saturated sand; the sand weakens and can
**liquefy**, and the turbine tilts. Finite element (FE) simulators such as OpenSees capture this,
but one run takes minutes to hours, far too slow for risk studies that need thousands of
scenarios. The project trains a **Fourier Neural Operator (FNO)**, a neural network that maps a
whole input history to a whole output history, to copy the simulator, and then tests whether
adding a **physics loss** (balance laws checked during training) reduces the amount of training
data needed.

**Research question (Semester 2):** does a physics loss reduce FNO prediction error when only
100–300 training simulations are available?

**Novelty claim:** FNOs for earthquakes, physics-informed networks for seabeds, and ML models for
monopiles each exist separately; one operator-learning model that predicts full pore-pressure
fields and tilt histories under **combined wave and earthquake loading** does not.

**Scope decisions (from the roadmap):** 2D plane strain instead of 3D; outputs are the
pore-pressure ratio field and pile tilt; the physics loss uses only force balance and water mass
balance (the full linear Biot equations would contradict the nonlinear sand model); about 1,000
simulations.

---

## 2. Status against the roadmap

| Part | Title | Semester 1 status |
| --- | --- | --- |
| 1 | Scope and literature review | Partly done: scope and novelty fixed in the roadmap; formal 2-page scope note and literature write-up still to do |
| 2 | Tool warm-up | Done in practice: OpenSeesPy 3.8, PyTorch (CUDA 12.8) and neuraloperator 2.0 installed and used end to end in parts 3–6. The separate toy Burgers' FNO was skipped, since part 5 trained real FNOs |
| 3 | Soil-column model and benchmark | **Done**: three benchmarks pass |
| 4 | Load generator | **Done**: one call returns any number of combined wave + earthquake histories; 353 real PEER records; leakage-free split |
| 5 | Mini-surrogate (FNO vs LSTM) | **Done**: 330 simulations, 4 models × 3 seeds; FNO wins on liquefaction timing and speed, ties on field error |
| 6 | 2D pile model with waves | **Done with caveats** (section 7): model built, static and dynamic checks pass, combined run sensible, run time measured (3–3.5 h); far-field pore-pressure checkerboard still to fix |
| 7–12 | Dataset, surrogates, physics loss, evaluation, thesis | Semester 2 (`NEXT_SEMESTER_REMAINING.md`) |

The original 5-week plan put part 6 in the break; it was finished inside Semester 1.

**Environment:** Windows 11 laptop, 12 logical CPUs, NVIDIA RTX 3050 6 GB; Python 3.12 virtual
environment (`requirements.txt`). Code in Python packages `soil_column/`, `loads/`,
`mini_surrogate/`, `pile_model/`; 29 automated tests in `tests/`.

---

## 3. Background needed for the report (short glossary)

| Term | Meaning |
| --- | --- |
| Saturated sand | Sand grains with every gap (pore) filled with water |
| Effective stress (σ') | The part of the soil weight carried by grain-to-grain contact; it gives sand its strength |
| Excess pore pressure (Δu) | Extra water pressure caused by shaking, on top of the still-water pressure |
| ru = Δu / σ'v0 | Pore-pressure ratio: 0 = solid, 1 = water carries all the load = **liquefied** |
| Relative density (Dr) | How tightly packed the sand is: 35% loose (liquefies easily) to 75% dense |
| Permeability (k) | How easily water flows through the sand (m/s); controls how fast Δu drains |
| PGA | Peak ground acceleration, the strength of the shaking, in g |
| u-p element | A finite element with soil displacement (u) and pore pressure (p) unknowns: Biot's theory in its common engineering form |
| PM4Sand | A standard nonlinear sand model for liquefaction (Boulanger & Ziotopoulou 2017) |
| CRR / CSR | Cyclic resistance / stress ratio: the shear load at which sand liquefies in a given number of cycles |
| JONSWAP | Standard spectrum of storm-wave energy over frequency |
| Morison / MacCamy–Fuchs | Formulas for the wave force on a vertical cylinder (drag + inertia; MacCamy–Fuchs corrects inertia for wide piles) |
| Monopile, tilt | The foundation tube; the permanent rotation of the turbine, design limit about 0.5° |
| FNO | Neural operator that learns in the frequency domain; resolution-independent |
| LSTM / BiLSTM | Recurrent time-series networks (causal / both directions); the ML baselines |

---

## 4. Part 3: soil-column model and benchmark check

**Goal:** a saturated sand column in OpenSeesPy whose pore-pressure build-up matches a benchmark.

**Model:** 10 m column of uniform sand, 0.5 m SSPquadUP u-p elements, PM4Sand, water table at the
surface; rigid impermeable base, drained surface, left/right nodes tied (simple shear). Units kN,
m, s, kPa, Mg/m³. Staged analysis: elastic gravity (PM4Sand stage 0), then plastic (stage 1).

**Three benchmarks:**

1. **Terzaghi 1D consolidation (exact solution).** Checks the water-flow coupling, permeability
   units and drainage. Error < 0.1% of the load at every time factor (Tv 0.02–1.0).
   Figure `results/part3/terzaghi_check.png`.
2. **PM4Sand cyclic simple shear vs Idriss & Boulanger (2008).** The one free parameter (hpo) is
   calibrated per density so the sand liquefies in 15 cycles at the published CRR. Independent
   check: the 5, 10 and 30-cycle points fall within about 7% of the published curve.

   | Dr | hpo | CRR15 | Cycles to liquefy (targets 5 / 10 / 15 / 30) |
   | --- | --- | --- | --- |
   | 35% | 0.653 | 0.090 | 3.5 / 8.5 / 15 / 39.5 |
   | 45% | 0.555 | 0.113 | 4.5 / 9.0 / 15 / 37.5 |
   | 55% | 0.492 | 0.147 | 4.5 / 9.5 / 15 / 36.0 |
   | 65% | 0.471 | 0.199 | 5.0 / 9.5 / 15 / 35.0 |
   | 75% | 0.878 | 0.312 | 5.0 / 9.0 / 15 / 37.0 |

   Figures `dss_crr_curves.png`, `dss_example_dr55.png`.
3. **Shaken column vs known liquefaction trends** (0.15 g, 2 Hz, 10 cycles):

   | | Dr 35% | Dr 55% | Dr 75% |
   | --- | --- | --- | --- |
   | Liquefied zone (ru ≥ 0.95) | 2–10 m | 1–9 m | none |
   | Simplified procedure FS < 1 | whole column | whole column | none (FS ≈ 1.05) |
   | Surface acceleration late / early | 0.34 | 0.11 | 1.30 |
   | Run time | 13 s | 58 s | 14 s |

   Loose and medium sand liquefy within 1–3 cycles and the surface shaking then collapses (base
   isolation, as seen in centrifuge tests); dense sand does not. The liquefied zone agrees with the
   independent simplified (Seed–Idriss style) procedure. Figures `column_density_fs.png`,
   `column_loose_histories.png`, `column_loose_ru_field.png`.

**OpenSees pitfalls found** (useful "lessons" slide): `fix` imposes zero displacement (unloads the
model after gravity); an sp on a retained equalDOF node silently breaks the tie under the
Transformation handler; pore pressure is read from `nodeVel(node, 3)`; permeability input is
k/(ρw g); PM4Sand needs stage 0 → stage 1 plus `FirstCall`.

**Limitations:** settlement is far too small (known PM4Sand weakness; not an output we use);
dissipation after shaking is slow; benchmark B is against an empirical curve and C checks trends
(a centrifuge comparison, e.g. LEAP-UCD-2017, would be stronger).

---

## 5. Part 4: load generator

**Goal:** one function call returns any number of realistic combined wave + earthquake histories.

```python
cases = generate_loads(300, seed=0, records="peer", split="train")
```

Each case: time, base acceleration, wave force, mudline moment and surface elevation on a 0.01 s
grid, plus its parameters.

**Waves:** JONSWAP spectrum (γ = 3.3), 256 random-phase components with jittered frequencies
(no repeat period), normalised so 4√m0 = Hs exactly; linear (Airy) kinematics; inertia by
MacCamy–Fuchs per component (equals Morison Cm = 2 for thin piles, avoids over-predicting for the
8 m pile), drag by Morison (Cd = 1.0). Pile D = 8 m, water depth 30 m.

**Earthquakes:** 353 horizontal PEER NGA-West2 records from 36 earthquakes (170 stations; 46
records from M ≥ 7). Processing: resample to 0.01 s → trim to the 5–95% Arias duration (+2 s/+5 s)
→ taper and zero-padded 0.1 Hz high-pass → displacement baseline correction (median final drift
5.5 mm). Scaled linearly to a target PGA (scale factor limited to ¼–4).

**Combination:** waves throughout; the earthquake starts after a random 30–90 s; 20 s after
shaking. Latin hypercube over Hs 1–8 m, Tp 6–14 s, PGA 0.05–0.4 g, start time 30–90 s.

**Leakage-free split by earthquake.** Records from the same earthquake (and aftershock sequences,
e.g. Mammoth Lakes-01 to -06) are correlated, so whole sequences go to train / val / test. The
split is stratified by magnitude band (< 6.5, 6.5–7, ≥ 7) using metadata parsed from the PEER
CSV files.

| Split | Records | Events | Larger events |
| --- | --- | --- | --- |
| Train | 248 | 26 | Chi-Chi, Duzce, El Mayor-Cucapah, Darfield, ... |
| Validation | 35 | 3 | Hector Mine (7.1), Loma Prieta (6.9), Oroville |
| Test | 70 | 7 | Kocaeli (7.5), Tabas (7.35), Northridge (6.7), San Fernando (6.6), ... |

**Verification:** 15 automated tests (JONSWAP Hs to 1e-6; dispersion limits; MacCamy–Fuchs →
Morison; a simulated 1-hour sea gives Hs within 5%; response spectrum vs SciPy SDOF within 1%;
record parser, scaling, drift; split has no leakage and lands within 5% of 70/10/20).

**Numbers:** 100 cases in 19.5 s; peak wave force 0.44–6.5 MN, mudline moment 8–117 MN·m; drag
about 10% of peak force (inertia regime, KC 0.6–7.4). Figures in `results/part4/`.

**Limitations:** storms last hours but a case has 30–90 s of waves (cost); linear waves (no
breaking or slamming); no wave pressure on the seabed and no wind thrust; scaling by PGA only.

---

## 6. Part 5: mini-surrogate on the soil column

**Goal:** full ML pipeline on the cheap problem: 200–300 column simulations, FNO vs LSTM on
unseen earthquakes.

**Dataset:** 330 OpenSees column runs. Varied: PEER record, PGA 0.05–0.40 g, Dr 35–75%, k 1e-5 to
1e-3 m/s (log-uniform). Split by earthquake (train 230 / val 35 / test 65 cases; test events:
Kocaeli, Tabas, Northridge, San Fernando, Helena Montana-01/02, Westmorland). 50 s per case; ru
stored at 20 depths × 1,000 steps (0.05 s). **0 of 330 runs failed.** Run time median 86 s, max
30 min; 15.7 CPU-hours, 95 min on 10 workers.

**Models** (matched at about 1.0–1.45 M parameters): FNO-1D over time (500 modes, width 32),
FNO-2D over depth × time, LSTM (3 × 256, causal), BiLSTM (3 × 160 per direction). Inputs per step:
base acceleration, velocity, cumulative Arias intensity, Dr, log k. MSE loss, AdamW, one-cycle
schedule, 200 epochs, best validation epoch kept, 3 seeds each.

**Design sweep (validation only):** the first FNO kept 64 time modes, which over a 50 s window
means only frequencies below 1.3 Hz; earthquake energy is at 1–10 Hz, so it barely saw the
shaking (val RMSE 0.153). Keeping all 500 modes was the single biggest gain (0.107).

**Test results** (65 cases, 7 unseen earthquakes, mean ± std over 3 seeds):

| Metric | FNO-1D | FNO-2D | LSTM | BiLSTM |
| --- | --- | --- | --- | --- |
| RMSE of ru | 0.091 ± 0.002 | 0.106 ± 0.008 | 0.089 ± 0.010 | **0.079 ± 0.005** |
| Liquefied? accuracy (per depth) | 0.914 | 0.907 | 0.883 | **0.917** |
| Liquefaction onset time error | **2.80 s** | 3.49 s | 3.83 s | 4.00 s |
| End-of-run ru error | **0.054** | 0.066 | 0.064 | 0.070 |
| Training time | **84 s** | 349 s | 1,205 s | 2,270 s |
| Inference per case | **3.9 ms** | 4.0 ms | 104 ms | 45 ms |
| Speed-up vs OpenSees (86 s) | **~22,000×** | ~21,500× | ~830× | ~1,900× |

**Statistics:** paired comparison with a bootstrap that resamples whole earthquakes. FNO-1D
minus LSTM: onset error −1.35 s [95% CI −1.79, −0.86], FNO better on 26 of 28 cases; RMSE −0.003
[−0.011, +0.001] = statistical tie.

**Verdict:** the FNO beats the LSTM clearly on *when* liquefaction starts, on stability across
seeds (5× smaller spread), and on cost (14× faster training, 27× faster inference); it ties on
overall field error, and a BiLSTM has the lowest field error. "FNO beats LSTM" holds on the
engineering timing metric and cost, not across the board.

**Why not an outright win:** strong overfitting at 230 samples (train RMSE ≈ 0.03 vs val ≈ 0.11),
which is exactly the data-scarce regime the physics loss targets in Semester 2; pore pressure is
a running total of past shaking, natural for an LSTM; small high-frequency output jitter; a
3-earthquake validation set.

**Honesty notes:** the first run's test scores were seen before the sweep (all later choices used
validation only); the LSTMs were not tuned (favours the FNO).

Figures: `results/part5/test_fields.png`, `test_histories.png`, `test_metrics.png`,
`peak_ru_scatter.png`, `training_curves.png`, `dataset_overview.png`.

---

## 7. Part 6: 2D pile model with waves

**Goal:** add the monopile, the turbine and the wave force to the soil model; record the run
time; one combined run must give physically sensible results.

**Model** (`pile_model/model.py`):
- 2D plane strain, slice thickness = pile diameter (8 m). Soil 100 m × 30 m, graded mesh, 312
  SSPquadUP u-p elements with the part 3 PM4Sand calibration; periodic sides, rigid shaken base,
  drained mudline.
- Generic 10 MW turbine (DTU 10 MW, rounded): pile D 8 m, wall 80 mm, embedded 24 m (L/D = 3);
  water depth 30 m; hub at +119 m; tower 7.5 m / 35 mm; rotor-nacelle 674 t; total weight 20.8 MN.
- Embedded pile + soil plug as a stiff elastic block with the steel pile's bending stiffness
  (E_eq ≈ 9.9e6 kPa); duplicated wall nodes with friction-limited vertical springs (API-style
  τmax = 0.8 σ'v tan 22°); tower as P-Delta beam elements with water and added mass; rigid cap.
- Loads: part 4 wave force and mudline moment at the cap; earthquake at the base.
- Staged static analysis (soil weight with free wall slip → springs → turbine weight → plastic
  sand); Rayleigh damping by region; time step 0.05 s (waves) / 0.01 s (earthquake).

**Static and dynamic checks** (`results/part6/checks.json`, 6 automated tests):

| Check | Result | Expected |
| --- | --- | --- |
| Free-field σv at 8.5 m | 82.3 kPa | 82.9 kPa |
| K0 | 0.43 | 0.35–0.55 |
| σv next to the pile | 1.16× free field at 8.5 m, 1.6× at 3.5 m | ≥ 1 (no hanging; raised by skin friction) |
| Settlement under the weight | 14 mm | centimetres |
| Fixed-base first frequency | 0.226 Hz | 0.2–0.3 Hz; cantilever formula within 3% |
| Soil–structure first frequency | 0.192 Hz (−15%) | below fixed base |
| First-mode damping | 1.9% | 1–3% |
| Run with no loads | stays at rest | at rest |

**Combined run:** Hs 7 m, Tp 12 s storm; Kocaeli 1999 (RSN1166) at 0.25 g starting at 40 s;
loose sand Dr 40%, k 1e-4 m/s; 84.6 s simulated. Run as waves only / earthquake only / combined,
plus the 1D column as the free-field reference.

| Run | Run time | Peak tilt | Final tilt | Peak hub displacement | Settlement |
| --- | --- | --- | --- | --- | --- |
| Waves only | 5 min | 0.012° | 0.003° | 0.09 m | 1 mm |
| Earthquake only | 189 min | 3.82° | 3.81° | 11.7 m | 123 mm |
| Combined | 211 min | 3.01° | 2.96° | 9.4 m | 130 mm |

**Physically sensible:** waves alone are harmless; the earthquake liquefies the loose sand over
the full depth (as the 1D column); tilt ratchets one way during shaking, keeps growing about 10 s
after it (weight on liquefied soil), then locks in; settlement occurs during shaking; pressure
rises on the side the pile pushes into and drops (suction) on the other; the storm changes the
permanent tilt by about 20%, which is the interaction the surrogate must learn. Figures:
`combined_histories.png`, `combined_ru_snapshots.png`, `free_field_check.png`.

**Caveats (state in the report):**
1. **Checkerboard pore pressure** in the far field after liquefaction: SSPquadUP's stabilisation
   α = h²/(4G) was sized with the element height and the initial stiffness, but far-field cells are
   up to 12 m wide and liquefied sand has almost no stiffness. Far-field ru swings 0.2–2.0 instead
   of ≈ 1 (RMSE 0.38 vs the column at 8 m). Tests: sizing α by the longest side (now in the code)
   helps a little, 4× stronger halves the zig-zag, a finer mesh barely helps, 200× breaks the
   static stage. Trends are believed; values are not dataset quality until fixed.
2. **Run time 3–3.5 h** per combined run (roadmap warning level: 15 min). Tolerance 1e-3 instead of
   1e-4 halves it (tilt changes 1%).
3. **Tilt of 3–4°** is inflated by the 2D slice and constant wall friction; used for relative
   comparison, not as a real-turbine prediction.

**Bugs found:** soil hanging on the stiff pile; a time-ramped weight that vanished when each run
restarted the clock (spurious |ru| ≈ 2 with no load; now a regression test); eigen analysis
returning 0 Hz with u-p elements (replaced by a pluck test); non-convergence when loading plastic
sand (load in the elastic stage).

Details: `docs/part6_pile_model.md`, `expln/part6_pile_model_explained.md`.

---

## 8. Suggested report / slide structure

1. Introduction: offshore wind in seismic regions, liquefaction, why surrogates (section 1).
2. Literature and novelty (to be written; part 1 loose end).
3. Methodology overview: simulator → load generator → dataset → FNO (+ physics loss).
4. Soil model and benchmarks (section 4): the Terzaghi and CRR figures.
5. Load generator (section 5): wave and earthquake examples, the split table.
6. Mini-surrogate (section 6): test fields figure, results table, bootstrap.
7. 2D pile model (section 7): mesh sketch, combined-run histories, checks table, run time.
8. Plan for Semester 2 (`NEXT_SEMESTER_REMAINING.md`): dataset on HPC, research question, timeline.

## 9. References used

- Li et al. (2021), Fourier Neural Operator for Parametric PDEs. Li et al., PINO.
- Boulanger & Ziotopoulou (2017), PM4Sand Version 3.1 manual.
- Idriss & Boulanger (2008), Soil Liquefaction During Earthquakes (EERI monograph).
- McGann, Arduino & Mackenzie-Helnwein (2012), stabilised single-point u-p element (SSPquadUP).
- Terzaghi (1943), one-dimensional consolidation theory.
- Hasselmann et al. (1973), JONSWAP spectrum. Morison et al. (1950); MacCamy & Fuchs (1954).
- Boore (2005), on padding and filtering of strong-motion records.
- Ancheta et al. (2014), PEER NGA-West2 database.
- Jonkman et al. (2009), NREL 5-MW reference turbine; Bak et al. (2013), DTU 10-MW reference turbine (sizes for part 6).
