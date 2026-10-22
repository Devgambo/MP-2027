# Remaining work for Semester 2 (and the break before it)

Status at the end of Semester 1 (Nov 10, 2026): roadmap parts 3–6 are built and checked; parts 1–2
are partly done (see "Loose ends from Semester 1"). This file lists everything still to do, in
order, with what already exists to build on, decisions to make, and the risks we already know
about. The detailed record of Semester 1 is in `SEMESTER1_REPORT_MATERIAL.md`.

---

## 0. Loose ends from Semester 1 (small, do first)

| Item | Why | Effort |
| --- | --- | --- |
| Scope note (roadmap part 1): 2 pages with the exact novelty claim, approved by the guide | Formal "done when" of part 1 | 1 day |
| Literature review write-up (FNO, PINO, OpenSees u-p, offshore turbines under earthquakes) | Needed for the thesis background chapter | 3–5 days, ongoing |
| Commit everything to git (parts 4–6 are not committed yet) | Safety; the team needs one shared repo | minutes |
| Filter each record at its own reliable-frequency limit (the PEER CSVs have it) instead of a fixed 0.1 Hz | Cleaner long-period content in old records | 1 hour |
| Optional: compare the soil column with a centrifuge test (LEAP-UCD-2017 on DesignSafe) | Stronger, experimental benchmark than trend checks | 2–3 days |
| Optional: try output smoothing / seed ensembles on the part 5 FNO | It tied the LSTM on field error; a cheap fix might make it win | 1 day |

---

## 1. Part 6 follow-ups (over the break, before the big dataset)

The 2D pile model runs and passes its checks (see `docs/part6_pile_model.md`). Before generating
1,000 runs:

1. **Fix the checkerboard pore pressure (must do first).** After liquefaction the far-field
   pressure alternates node to node; far-field ru swings 0.2–2.0 instead of about 1 (RMSE 0.38 vs
   the 1D column). The surrogate would learn this artefact. Options, in order:
   - Stronger SSPquadUP stabilisation: α = h²/(4·G_stab) with h = longest element side is in the
     code (`Mesh.stab_G`, default 2e5). G_stab = 5e4 halved the zig-zag; 1e3 broke the static
     stage. Try 2e4 and 5e3 (started but stopped unfinished at the end of Semester 1; test method:
     10 s of the strongest Kocaeli shaking, compare x = -50 m ru with the 1D
     column and measure the neighbour difference). Possibly use a smaller α during gravity and a
     larger one for the dynamic stage.
   - Or switch the soil to `9_4_QuadUP` (9-node displacement / 4-node pressure, stable by
     construction, no stabilisation needed). Needs a mesh rewrite; about 3× the unknowns.
   - Done when: far-field ru matches the 1D column (RMSE < 0.1) and the ru map is smooth.
2. **Rerun the combined check** with the fix and tolerance 1e-3; update `results/part6/`, the part 6
   docs and the report numbers (tilt values may change).
3. **Pilot batch of 50–100 combined runs** (resumable, like part 5). Measures the per-run time,
   the crash rate across the parameter space, and the output size. Must run on HPC or over several
   nights.
4. **Speed.** Measured: 3–3.5 h per combined run with liquefaction (Newton, tolerance 1e-4, nearly
   every strong-shaking step substepped). Options:
   - tolerance 1e-3: measured 2× faster, tilt changes 1% (adopt);
   - shorter waves-only lead-in (small gain; waves-only takes 5 min of the total);
   - coarser mesh near the pile or earthquake step 0.02 s (test; may worsen convergence);
   - KrylovNewton throughout: measured slower, rejected.
5. **Decide the storm length.** The part 4 storm before the earthquake is 30–90 s; real storms last
   hours. Options: keep the short lead-in and state the limitation, or add a "pre-storm" pore
   pressure state as an extra input.
6. **Known model limitations to keep in mind** (from part 6): 2D slice (tilt over-predicted: 3–4°
   in the check case), pile modelled as a stiff block with its soil plug, interface friction fixed
   (does not drop when the soil liquefies), no wind thrust (parked turbine), wave force applied at
   the mudline only.
7. Optional upgrades if time allows: friction that drops with ru, a mean wind-thrust load
   (operating turbine), and a 3D check run on HPC for one case.

---

## 2. Part 7: dataset factory (Semester 2 weeks 2–6)

**Goal:** about 1,000 clean combined runs, split into train / validation / test by earthquake.

What exists:
- `loads.generate_loads(..., records="peer", split=...)`: Latin hypercube storm + earthquake cases,
  magnitude-stratified event split (353 records, 36 earthquakes).
- `mini_surrogate/dataset.py`: the template for a resumable, parallel, crash-rejecting run factory
  writing HDF5.
- `pile_model.model.run(...)`: one combined run.

To build:
1. `pile_model/dataset.py`: sample sand density, permeability, Hs, Tp, PGA, gap; assign records by
   split; run in parallel; reject non-converged runs automatically; record the rejection rate.
2. Map outputs to a regular grid with a pile mask (the mesh is graded; FNOs want a regular grid).
3. Fix the HDF5 layout (variable names, grid, units) before the big run, so the ML side can start
   on the pilot data.
4. Enlarge the validation set to 10+ earthquakes (part 5's 3-earthquake validation set was noisy).

**Compute** (measured in part 6: 3–3.5 h per strong-liquefaction run at tolerance 1e-4; about
half with 1e-3; weak-shaking or dense-sand runs will be much faster, so the average is lower):

| Case | Average per run | 1,000 runs on the laptop (10 parallel) | On HPC (200 cores) |
| --- | --- | --- | --- |
| Optimistic | 30 min | ~2 days | ~3 hours |
| Likely | 1 h | ~4 days | ~5 hours |
| Pessimistic | 1.75 h | ~7 days | ~9 hours |

The laptop is a fallback only (up to a week of non-stop running). Fallback from the roadmap: 500 runs. The pilot batch replaces these estimates
with measured numbers.

---

## 3. Part 8: baseline tilt model (weeks 5–7)

- FNO mapping load histories (base acceleration, wave force, moment) plus soil parameters to the
  tilt history only. LSTM and BiLSTM baselines on the same data.
- Done when: the FNO clearly beats the LSTM. If not, find out why first.
- Lessons from part 5 to apply:
  - Give the FNO enough time modes to see the 1–10 Hz shaking (64 modes over 50 s was blind).
  - Report onset/peak/permanent metrics, not only RMSE.
  - Use 3 seeds and an event-level bootstrap for the comparison.
  - Size the LSTM to the same parameter count.

## 4. Part 9: full-field FNO (weeks 7–10)

- Inputs: load histories, soil properties, pile mask. Outputs: ru field over time and tilt history.
- Memory: a 2D field × 2,000+ time steps is large for a 6 GB GPU. Plan for time windows, a coarser
  output step, or Colab/Kaggle GPUs.
- Done when: errors on held-out earthquakes meet a target agreed with the guide (for example,
  peak tilt within 10%).

## 5. Part 10: physics loss and the data-scarcity test (weeks 10–12)

- Add the balance-law checks (force balance, water mass balance) with finite differences on the
  grid; tune their weight.
- Train with and without physics at 100, 300 and 1,000 samples. One chart: error vs training size.
- Part 5 already showed the motivation: the FNO overfits strongly at 230 samples (train RMSE about
  0.03 vs validation about 0.11).

## 6. Part 11: evaluation and demo (weeks 12–14)

- Engineering errors: peak tilt, permanent tilt, peak ru.
- Speed-up over OpenSees (part 5 FNO: about 22,000×).
- Behaviour outside the training range.
- Small demo: pick an earthquake and a sea state, see the tilt instantly.

## 7. Part 12: write-up (weeks 13–16)

- Thesis, final presentation, and a conference paper if the results are strong.
- Write the method sections as each part finishes (parts 3–6 material is already in `docs/` and
  `expln/`).

---

## Key risks carried into Semester 2

| Risk | What we know now | Plan |
| --- | --- | --- |
| Checkerboard pore pressure in the 2D model | Far-field ru zig-zags after liquefaction; stronger stabilisation helps | Fix before part 7 (section 1, item 1) |
| Runs too slow | 3–3.5 h per strong-liquefaction run on the laptop (2× faster with tolerance 1e-3) | HPC; speed settings from the pilot; 500 runs as fallback |
| Runs crash | 0 of 330 column runs and 0 of 3 full 2D runs crashed (one test setting failed in the static stage); 2D crash rate across the parameter space unknown | Measure in the pilot; auto-reject; narrow the PGA range if more than 1 in 5 fail |
| FNO does not beat the LSTM | Part 5: tie on field error, win on timing and speed | Apply the part 5 lessons; a careful "no" is still a result |
| GPU memory | 6 GB laptop GPU | Time windows, coarser grid, Colab/Kaggle |
| Physics loss does not help | Unknown | Report honestly |
