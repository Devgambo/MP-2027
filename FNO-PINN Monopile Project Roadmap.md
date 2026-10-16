# FNO-PINN Monopile Project: Roadmap

Oct 7, 2026 · @Shourya Ranjan

## High-level overview

**In one line:** we train an AI model on many computer simulations, so it can predict in under a second how the seabed and a wind-turbine foundation respond when waves and an earthquake hit together.

### The structure

An offshore wind turbine stands on a **monopile**: a huge steel tube, often 6–10 m wide, driven 25–40 m into the seabed. The soil around the tube holds the turbine upright. If that soil weakens, the pile tilts, and a turbine can tolerate only a fraction of a degree of permanent tilt.

### The two loads

- **Waves:** slow, repeated pushes, roughly every 8–12 seconds in a storm, lasting hours. Each push is small, but there are thousands of them.
- **Earthquake:** fast, strong shaking for 20–40 seconds. It shakes the soil itself, not just the pile.

### What goes wrong in the soil

Seabed soil is sand or clay grains with water filling the gaps (the pores). When it is shaken quickly and repeatedly, the water has no time to escape, so its pressure rises. Think of squeezing a wet sponge fast: the water pushes back.

Higher water pressure means the grains press on each other less, so the soil gets weaker. In the worst case it behaves like a liquid, which is called **liquefaction**. Waves can raise this pressure before the earthquake even arrives, which is why the combination is dangerous.

### Why this is hard today

Engineers simulate this with finite element (FE) software such as OpenSees or PLAXIS. These tools are accurate, but one realistic 3D run takes hours. Checking risk properly needs thousands of scenarios (different earthquakes, sea states and soils), which is not practical.

### The idea: a fast copy of the simulator

Run the slow simulator many times, once. Then train an AI model to copy it. After training, the model answers a new scenario in milliseconds instead of hours. This copy is called a **surrogate model**.

### Ingredient 1: Fourier Neural Operator (FNO)

A normal neural network maps numbers to numbers. A **neural operator** maps a whole function to a whole function: a full load history goes in, and a full response history comes out.

The FNO works with frequencies. It splits the input signal into its frequencies, like the equaliser bars in a music app, and learns how each frequency changes the output. This suits our problem, because waves are low-frequency and earthquakes are higher-frequency.

### Ingredient 2: a physics check in training (the PINN part)

During training, the model is scored on two things: how close it is to the simulation data, and whether its answer obeys basic physics. The physics here comes from **Biot's theory**, which describes soil grains and pore water moving together.

Analogy: a student who learns from solved examples (the data) and also checks every answer against textbook laws (the physics). The hope is that a model which knows the laws needs fewer examples.

### Inputs and outputs

- **Inputs:** an earthquake record, a wave-force history, and soil properties (how dense the sand is, how easily water flows through it).
- **Outputs:** pore-water pressure in the soil over time, soil stresses, and the pile's tilt over time.

### What is new

FNOs for earthquakes, PINNs with Biot's equations for seabeds, and simple ML models for monopiles each exist separately. What is new is one operator-learning model that predicts **full pore-pressure fields and tilt histories under combined wave and earthquake loading**.

## The scoped-down version we will build

Same idea, smaller problem: a 2D model instead of 3D, two outputs instead of five, and a physics check that only enforces laws our soil actually obeys. This keeps the novelty and makes a 1,000-run dataset realistic.

| Aspect | Original idea | Our version | Why the change |
| --- | --- | --- | --- |
| Geometry | Full 3D soil and pile | 2D plane strain: one vertical slice through the pile and soil | A run takes minutes instead of hours, so about 1,000 runs fit in a few weeks |
| Outputs | 3D stress, pore pressure and tilt | Pore-pressure ratio field over time, plus pile tilt over time | Two outputs are enough to prove the method; stresses become a stretch goal |
| Physics check | Full Biot equations | Only the balance laws: force balance and water mass balance | Full Biot assumes linear soil; our soil is nonlinear, so the full equations would push the model towards wrong answers |
| Soil model | Not fixed | One nonlinear sand model in OpenSees (PDMY02 or PM4Sand) | Standard, well documented, and it captures pore-pressure build-up |
| Dataset | Hundreds to thousands of 3D runs | About 1,000 2D runs | Fits on the college HPC cluster |
| Main claim | A real-time 3D surrogate | Physics helps when data is scarce, tested at 100, 300 and 1,000 training runs | Gives one clear research question with a measurable answer |

### The research question

**Does adding a physics check to an FNO reduce prediction error when only a little training data is available?** Either answer is a valid result. A "no" is still publishable if it is measured carefully.

### One honest limitation to state up front

A 2D plane-strain slice treats the round pile like a long wall, so absolute tilt values will not match a real 3D pile. We compare the surrogate against the 2D simulator, not against real turbines. The method carries over to 3D, which becomes future work.

### What "finished" looks like

A trained model that takes a new earthquake record, a wave history and soil properties, and returns the pore-pressure field and tilt history in under a second. It is tested on earthquakes it never saw in training, against an error target your guide agrees on (for example, within 10% on peak tilt).

## Roadmap: the parts in order

Twelve parts, each with a clear "done when" check, so you always know when to move on. Parts 3, 6 and 7 are the longest and riskiest: if they slip, the ML parts get squeezed.

1. **Scope and literature review**
   - Do: read the FNO paper (Li et al., 2021), the PINO paper, the OpenSees saturated-soil (u-p) examples, and 3–4 papers on offshore turbines under earthquakes.
   - Done when: your guide approves a 2-page scope note with the exact novelty claim.
2. **Tool warm-up**
   - Do: install OpenSeesPy, PyTorch and the neuraloperator library. Run one OpenSees saturated-soil example. Train an FNO on a toy problem from the neuraloperator examples (for example, Burgers' equation).
   - Done when: both run end to end on your own machines.
3. **Soil-column model and benchmark check**
   - Do: build a column of saturated sand (no pile yet) with a nonlinear soil model. Shake its base with an earthquake. Compare the pore-pressure build-up with a published test or validated example.
   - Done when: your curves match the benchmark trend.
4. **Load generator** (can run alongside part 3)
   - Do: download earthquake records from PEER NGA-West2 and scale them. Generate wave forces from random sea states (JONSWAP spectrum plus Morison equation). Combine them with a random time gap between the two.
   - Done when: one function call returns any number of realistic combined load histories.
5. **Mini-surrogate on the soil column**
   - Do: run 200–300 soil-column cases (each takes seconds to minutes). Train a small FNO that maps base shaking to pore pressure over depth and time. Compare it with a simple LSTM.
   - Done when: the FNO beats the LSTM on earthquakes it never saw.
   - Why: you get a full ML result early, and you test the whole pipeline on a cheap problem.
6. **2D pile model with waves**
   - Do: add the monopile to the soil model, then add the wave force on the pile. Record how long one run takes.
   - Done when: one combined run gives physically sensible results.
7. **Dataset factory**
   - Do: sample soil and load parameters (sand density, permeability, earthquake strength, wave height and period) with Latin hypercube sampling. Run batches in parallel on HPC. Reject crashed runs automatically. Map results onto a regular grid with a pile mask. Keep some earthquakes out of training for testing.
   - Done when: about 1,000 clean samples are split into train, validation and test sets.
8. **Baseline tilt model**
   - Do: train an FNO that maps load histories to the tilt history only. Train an LSTM on the same data.
   - Done when: the FNO clearly beats the LSTM. If it does not, stop and find out why before going further.
9. **Full-field FNO**
   - Do: inputs are load histories, soil properties and the pile mask. Outputs are the pore-pressure field over time and the tilt history.
   - Done when: errors on held-out earthquakes meet your agreed target.
10. **Physics loss and the data-scarcity test**
    - Do: add the balance-law checks (computed with finite differences on the grid) and tune their weight. Train with and without physics on 100, 300 and 1,000 samples.
    - Done when: you have one chart of error against training size, with and without physics.
11. **Evaluation and demo**
    - Do: measure errors on engineering numbers (peak tilt, permanent tilt, peak pore-pressure ratio). Report the speed-up over OpenSees. Test on inputs outside the training range. Build a small demo: pick an earthquake, see the tilt instantly.
    - Done when: results tables, figures and the demo are ready.
12. **Write-up**
    - Do: write the method section as you go, not at the end. Finish the thesis, final presentation and, if results are strong, a conference paper.
    - Done when: the thesis is submitted.

**Stretch goals, only if time is left:** a small 3D model, uncertainty estimates from an ensemble of models, or fragility curves.

## Two-semester plan

Semester 1 builds and checks the simulator and ends with a working mini-surrogate. Semester 2 builds the full pile surrogate and runs the physics experiment. The plan assumes about 15 teaching weeks per semester, but Semester 1 ends on Nov 10, 2026, so it is compressed to 5 weeks (from Oct 7). Semester 2 is assumed to run about 16 weeks, and the pile model is built over the break.

&#91;embedded content: Two-semester timeline · 12 work items, 2 reviews\]

Semester 1 now covers only the soil column, the load generator and the mini-surrogate. The pile model moves into the break, and Semester 2 opens with the big dataset. The tables below give the exact weeks.

### Semester 1: build the simulator, prove the pipeline

| Weeks | Work (roadmap part) | Output |
| --- | --- | --- |
| 1 (Oct 7–13) | Scope note, reading, tool warm-up (parts 1, 2) | 1-page scope note; toy FNO running |
| 1–3 (Oct 7–27) | Soil-column model and benchmark check, starting from a documented OpenSees example (part 3) | Pore-pressure curves that match a published case |
| 2–3 (Oct 14–27) | Load generator (part 4) | Combined wave and earthquake histories on demand |
| 3–5 (Oct 21–Nov 7) | Mini-surrogate on the soil column, 200–300 runs (part 5) | FNO vs LSTM results on unseen earthquakes |
| 5 (Nov 4–10) | Semester 1 report and presentation | Report and slides |

### Between semesters

Build the 2D pile model with waves and run 50–100 pilot runs (part 6). This was Semester 1 work; moving it here is what makes a 5-week Semester 1 possible.

### Semester 2: full surrogate and the physics experiment

| Weeks | Work (roadmap part) | Output |
| --- | --- | --- |
| 1–2 | Finish the 2D pile model and pilot runs (part 6) | One checked combined-load run and its run time |
| 2–6 | Dataset factory: about 1,000 runs on HPC, then clean, grid, split (part 7) | About 1,000 clean samples |
| 5–7 | Baseline tilt model, starting on the first batch of runs (part 8) | FNO vs LSTM on tilt |
| 7–10 | Full-field FNO (part 9) | Pore-pressure field and tilt predictions |
| 10–12 | Physics loss and data-scarcity test (part 10) | Error vs training-size chart |
| 12–14 | Evaluation and demo (part 11) | Tables, figures, live demo |
| 13–16 | Thesis, final presentation, paper draft (part 12) | Submitted thesis |

### If you are a team

Split the work into a simulation owner and an ML owner, who meet at the dataset format. Fix that format (HDF5 file layout, grid size, variable names) by week 2 of Semester 1 (Oct 20). The ML owner can then build and test data loaders on fake data while the simulator is still being built.

## What to present

At the end of Semester 1 you present a checked simulator and a first working surrogate: a complete mini-version of the final project, not just a plan.

### End of Semester 1

1. **The problem and the gap:** the overview above, the literature you read, and your exact novelty claim.
2. **The checked soil model:** your pore-pressure curves plotted next to the published benchmark.
3. **The mini-surrogate:** FNO predictions against OpenSees on an earthquake the model never saw, the FNO vs LSTM error, and the measured speed-up.
4. **The pile model plan:** the 2D model setup you will build over the break, and the dataset you will run from it.
5. **The load generator:** a few examples of combined wave and earthquake inputs.
6. **The Semester 2 plan:** the research question and the timeline.

**Safety net:** items 1–3 are a complete result on their own, so they are the core of the Semester 1 presentation. With only 5 weeks, the pile model is planned for the break, not Semester 1.

### End of Semester 2

1. **Full-field predictions:** side-by-side pore-pressure animations (OpenSees vs FNO) and tilt overlays on unseen earthquakes.
2. **The headline chart:** error against training-set size, with and without the physics loss.
3. **Engineering error table:** peak tilt, permanent tilt and peak pore-pressure ratio.
4. **Speed and limits:** the speed-up over OpenSees, and how the model behaves on inputs outside its training range.
5. **Live demo:** pick an earthquake and a sea state, get the tilt history instantly.
6. **Limitations and future work:** the 2D simplification, then 3D and uncertainty estimates as next steps.

## Risks, fallbacks and tools

Most risk sits in the simulator, not the AI. Watch for these warning signs and switch to the fallback early rather than late.

| Risk | Warning sign | Fallback |
| --- | --- | --- |
| Soil model will not match the benchmark | Still off by Semester 1 week 3 (Oct 27) | Use the closest documented OpenSees example as it is; ask your guide or a senior for a checked input file |
| Liquefaction runs crash | More than 1 in 5 runs fail to converge | Smaller time steps, a narrower earthquake-strength range, auto-reject; report the rejection rate |
| Runs are too slow | One 2D run takes over 15 minutes | Coarser mesh, shorter records, 500 samples instead of 1,000 |
| GPU runs out of memory | Training crashes on full-length histories | Downsample in time, train on time windows, use a smaller grid |
| Physics loss does not help | No gain at 100 samples | Report it honestly; a careful "no" is still a result |
| Pile model late | Not running by Semester 2 week 2 | Cut the dataset to 500 runs and start the ML parts on the first batch |
| Mini-surrogate not ready by Nov 10 | No FNO results by Semester 1 week 4 (Nov 3) | Present the checked soil column and load generator, plus any early FNO results |

### Tools

| Area | Tool |
| --- | --- |
| Simulation | OpenSeesPy, saturated-soil (u-p) elements, PDMY02 or PM4Sand sand model |
| Earthquake records | PEER NGA-West2 database (free account) |
| Wave loads | JONSWAP spectrum and Morison equation, written in Python |
| Sampling | Latin hypercube sampling (SciPy) |
| Data storage | HDF5 files (h5py) |
| Machine learning | PyTorch and the neuraloperator library |
| Compute | College HPC for simulations; Kaggle or Colab GPUs for training |
| Plots | Matplotlib, ParaView |
| Version control | Git and GitHub, one shared repository |

### Starting reading list

- Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations* (2021).
- Li et al., *Physics-Informed Neural Operator for Learning Partial Differential Equations* (PINO).
- FNO surrogate for 3D seismic wave propagation (arXiv 2304.10242).
- Physics-informed multi-resolution neural operator (arXiv 2510.23810).
- PINN for the wave-induced response of a poroelastic seabed (ScienceDirect).
- Seismic response of offshore wind turbine monopiles and suction buckets, numerical and soft-computing study (ScienceDirect).
