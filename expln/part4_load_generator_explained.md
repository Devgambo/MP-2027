# Part 4 explained: the load generator

This file explains roadmap part 4 in plain terms: what "loads" are, how we make realistic
waves and earthquakes on a computer, how they are combined, and how we checked them. No ocean
or earthquake engineering background is assumed. The short technical version is in
`docs/part4_load_generator.md`.

Figures referred to below are in `results/part4/`.

---

## 1. Why part 4 exists

The simulator from part 3 (and the pile model in part 6) needs **inputs**: something pushing on
the soil and the pile. In this project there are two pushes:

- **the earthquake**, which shakes the ground from below, and
- **the waves**, which push on the pile sideways from the sea.

The AI model will later learn "these loads in → this soil and pile response out". To learn that
well, it must see many different, *realistic* loads. Part 4 is the factory that makes them.

Roadmap "done when": *one function call returns any number of realistic combined load
histories.* It now does:

```python
from loads import generate_loads
cases = generate_loads(100, seed=0)     # 100 different storm + earthquake scenarios
```

A **load history** just means "how big the push is at every moment in time": a list of numbers,
one every 0.01 seconds.

---

## 2. Part A: waves

### 2.1 Real sea waves are messy

On a calm day, waves can look regular. In a storm they never are: big and small waves mix,
arriving at irregular intervals. Engineers describe this mess with just two numbers:

| Number | Plain meaning | Our range |
| --- | --- | --- |
| **Hs** (significant wave height) | Roughly the average height of the biggest third of the waves, trough to crest. It's what an experienced sailor would "eyeball" as the wave height. | 1 to 8 m |
| **Tp** (peak period) | The time between waves that carry the most energy. Storm waves: about 6 to 14 seconds. | 6 to 14 s |

Together, (Hs, Tp) is called a **sea state**.

### 2.2 The wave "recipe": the JONSWAP spectrum

A messy sea can be thought of as many simple, smooth waves of different speeds added together,
like a chord made of many notes. A **spectrum** is the recipe: it says how much energy each
"note" (frequency) gets.

**JONSWAP** is the standard recipe for storm seas in the North Sea and beyond, measured in a
big 1960s–70s field campaign. Its shape is a sharp peak at the frequency 1/Tp, with energy
falling off on both sides.

How we build a sea from it:

1. Split the frequency range into 256 thin slices.
2. Give each slice a smooth wave whose height matches the energy JONSWAP assigns to it.
3. Give each wave a **random starting position** (phase). This randomness is what makes every
   storm different, even with the same Hs and Tp.
4. Add all 256 waves together.

A small trick: the slice frequencies are slightly jittered. With perfectly even slices, the
sea would repeat itself exactly after a while, which real seas don't do.

**Check:** `wave_spectra.png`. The thick lines are the JONSWAP recipe; the thin lines are
what our simulated 1-hour sea actually contains. They overlap. The measured wave heights
(2.01, 5.06, 7.97 m) match the targets (2, 5, 8 m) within 1%.

### 2.3 From the sea surface to water motion underwater

The pile feels the **water moving** past it, not the surface shape. For each of the 256 waves,
**linear wave theory** (also called Airy theory, the textbook model) tells us:

- how fast the water moves at each depth, and
- how fast it speeds up (its acceleration).

The motion is strongest near the surface and fades with depth. In 30 m of water, a long storm
wave still moves the water near the seabed; a short wave barely does.

One equation (the **dispersion relation**) links a wave's period to its length, and it depends
on the water depth. It has no direct formula, so the code solves it numerically, and a test
checks it against the known deep-water and shallow-water answers.

### 2.4 From water motion to force on the pile

Moving water pushes on the pile in two ways:

| Push | Plain meaning | Everyday analogy |
| --- | --- | --- |
| **Inertia force** | Comes from the water *speeding up and slowing down*. Accelerating water needs a force, and some of it is taken by the pile. | Swinging a full bucket of water: the hard part is starting and stopping it. |
| **Drag force** | Comes from the water *flowing past*, like wind on your hand out of a car window. | Hand out of a car window. |

The classic formula combining these two is the **Morison equation**. Engineers have used it for
offshore structures since 1950.

**For our 8 m wide pile, inertia dominates.** The pile is so fat compared with how far the water
moves back and forth that drag is small. Our example shows drag is only about 10% of the peak
force. A number called **KC** (Keulegan–Carpenter) measures this: KC = how far the water swings ÷
pile width. Ours is 0.6 to 7.4, which is firmly the "inertia-dominated" range.

**One upgrade: MacCamy–Fuchs.** Morison assumes the pile is thin compared with the wave length.
An 8 m pile is not that thin for shorter waves: it starts to *bend the waves around itself*
(diffraction), which reduces the force. **MacCamy–Fuchs** is the exact solution for a wide
circular pile. We use it for the inertia part. For a thin pile it gives exactly the same answer
as Morison, which one of our tests confirms.

**What the code returns for waves:**

- `force`: total sideways push on the pile, measured at the seabed (the **mudline**), in kN.
- `moment`: the "tipping" effect about the mudline (force × height above the seabed), in kN·m.
  This is what actually tries to tilt the pile, because waves push near the top while the soil
  holds near the bottom, like pushing the top of a fence post.
- `eta`: the sea surface height, mainly for plots.

**Check:** `wave_load_example.png`, for Hs = 6 m and Tp = 11 s. The force reaches about 3.8 MN
(about the weight of 380 tonnes), and the moment reaches about 70 MN·m. These are plausible values
for a large offshore monopile in a strong storm.

### 2.5 Waves that can't exist

Waves can only get so steep before they break. A wave with Hs = 8 m and a short Tp of 6 s would
be impossibly steep. So if the random sampler picks such a pair, the code raises Tp to the
steepest physically possible value (wave height ÷ wave length ≤ 1/16). In 100 cases, 11 were
adjusted. In `parameter_coverage.png`, the grey region is "too steep", and no case falls in it.

### 2.6 Ramp-up

If the waves started at full strength at t = 0, the model would get a sudden "kick" that no real
sea gives. So the waves are faded in smoothly over the first two wave periods.

---

## 3. Part B: earthquakes

### 3.1 What an earthquake record is

A **strong-motion record** is the measured acceleration of the ground at a seismic station
during a real earthquake, a list of numbers sampled every 0.005 to 0.02 s. It's the best input
we can use, because it contains everything real shaking has: the right mix of frequencies, the
build-up and decay, and random-looking spikes.

Useful words:

| Word | Plain meaning |
| --- | --- |
| **PGA** (peak ground acceleration) | The biggest acceleration in the record, in units of g (9.81 m/s²). 0.1 g is strong; 0.4 g is very strong. |
| **Event** | One earthquake. A single event is recorded by many stations, so it produces many records. |
| **Component** | Stations measure two horizontal directions and one vertical. We use only horizontal ones, since our model shakes sideways. |

### 3.2 Where real records come from: PEER NGA-West2

**PEER NGA-West2** is a free, curated database of thousands of records from shallow crustal
earthquakes worldwide (California, Japan, Taiwan, Turkey, Italy, and others). It needs a free
account, so you have to download the records yourself; `data/peer/README.md` has step-by-step
instructions and suggested search filters.

**Until then**, the code makes **synthetic earthquakes** (section 3.5) so that the whole pipeline
can be built and tested now. Every synthetic record is clearly labelled `source = "synthetic"`.
They are placeholders and must not be used for the final dataset.

### 3.3 Cleaning a record (processing)

Raw records need a little cleaning before they can go into a simulator. Each record goes through:

1. **Resample** to one common time step (0.01 s), so every case is on the same time grid.
   Down-sampling uses a filter first so high frequencies don't fold back as fake low ones (aliasing).
2. **Trim to the strong part.** Most records have long quiet stretches before and after the real
   shaking. Simulation time costs money, so we keep only the part that matters.
   We measure this with **Arias intensity**, a running total of shaking energy. We keep the window
   from 5% to 95% of the total energy (the "significant duration"), plus 2 s before and 5 s after.
3. **Remove drift (baseline correction).** If you integrate acceleration twice to get ground
   displacement, tiny errors in the record make the ground appear to fly metres away. Real ground
   doesn't do that. We remove the slow drift with a standard filter (a 0.1 Hz high-pass, applied with
   zero padding as recommended by Boore, 2005) and a final polynomial correction. After cleaning,
   the ground ends within about 2 cm of where it started.

### 3.4 Scaling to a target strength

To cover weak and strong shaking, each case asks for a **target PGA** (0.05 to 0.4 g), and the
record is multiplied by a constant to hit it exactly. It's like turning the volume knob without
changing the song.

Rule: a real record is never scaled more than 4× up or down. Turning a tiny earthquake up 20×
gives shaking with the wrong character, because big earthquakes are longer and richer in low
frequencies, not just louder. The code picks a record that needs at most 4× for the target.

### 3.5 Synthetic earthquakes (the placeholder)

We use a standard textbook recipe for fake-but-reasonable earthquakes:

1. Start with **white noise** (random numbers, all frequencies equally).
2. Pass it through a **Kanai–Tajimi filter**: this mimics how soft ground at a site amplifies
   shaking near a "site frequency" (we pick 1.5 to 5 Hz at random).
3. Pass it through a **Clough–Penzien filter**: removes unrealistically slow motions.
4. Multiply by a **time envelope**: quiet → build-up → strong part (10 to 30 s) → fade-out.
5. Clean it like a real record (section 3.3).

**Check:** `earthquake_examples.png`. Left, one synthetic record and its ground displacement
(stays within ±11 cm and returns near zero). Right, **response spectra**.

### 3.6 What a response spectrum is

Imagine a row of buildings of different heights standing on the shaking ground. Short stiff ones
sway quickly; tall flexible ones sway slowly. The **response spectrum** plots how hard each
building gets shaken, against its natural sway period. It's the standard "fingerprint" of an
earthquake in engineering.

Our synthetic records peak at periods of about 0.15 to 0.4 s, with peaks about 3× the PGA. That
is the typical shape of real records on firm ground. The code that computes it is checked against
an independent solver to 1% accuracy.

---

## 4. Part C: putting waves and earthquake together

### 4.1 The scenario

The dangerous case in this project is an **earthquake that strikes during a storm**: waves have
already been rocking the pile (and stressing the soil) when the ground starts to shake.

Every case follows the same timeline:

```
t = 0 ............ t_eq ................ t_eq + quake .......... end
waves fade in      earthquake starts     earthquake over       20 s more
(waves keep going the whole time)
```

- **t_eq** (the "gap") is how long the waves act alone before the earthquake: 30 to 90 s.
- Each case lasts 74 to 140 s in total.

**Check:** `combined_examples.png`. Three cases: the blue wave force goes from the start; the red
earthquake shaking appears partway through.

### 4.2 Choosing the scenarios: Latin hypercube sampling

Each case needs four random choices: Hs, Tp, PGA and t_eq. If you pick them purely at random, some
regions get crowded and others stay empty, especially with few cases. **Latin hypercube
sampling** (LHS) spreads them out evenly. It's like a sudoku rule: divide each range into 100
strips, and make sure each strip is used exactly once. So even 100 cases cover the whole space.

**Check:** `parameter_coverage.png`. The points fill both plots evenly, with nothing in the
"too steep" region.

### 4.3 Keeping the test fair: splitting by earthquake

Later, the AI is tested on earthquakes it has **never seen**. But one earthquake is recorded at
many stations, and those records are very similar. If one station's record were in training and
another station's (from the same earthquake) in testing, the test would be too easy: a hidden leak.

So `split_by_event` sends **whole earthquakes** to training (70%), validation (10%) or testing
(20%). A test confirms that no earthquake appears in two groups.

Two refinements were added once the real records arrived:

- **Aftershock families stay together.** "Mammoth Lakes-01" to "-06" are six quakes from one
  sequence, recorded by the same nearby stations. They behave like one earthquake, so the whole
  family goes to the same group (likewise Irpinia-01/02, Friuli-01/02/03, Helena Montana-01/02).
- **Every size of earthquake appears in every group.** Magnitude measures how big an earthquake
  is; a magnitude 7 releases about 30 times more energy than a 6, and shakes for longer. Picking
  earthquakes purely at random once put all the big ones in validation and left the test with only
  moderate ones, which would have made the test score look better than the model really is. Now
  earthquakes are sorted into three size bands (below 6.5, 6.5–7, 7 and above), and each band
  contributes to training, validation and testing. Magnitudes come from PEER's CSV files.

With the downloaded library this gives:

| Group | Records | Earthquakes | Biggest ones |
| --- | --- | --- | --- |
| Training | 248 | 26 | Chi-Chi (7.6), Düzce, El Mayor-Cucapah, Darfield, ... |
| Validation | 35 | 3 | Hector Mine (7.1), Loma Prieta (6.9) |
| Testing | 70 | 7 | Kocaeli (7.5), Tabas (7.35), Northridge (6.7), San Fernando (6.6) |

```python
train = generate_loads(300, seed=0, records="peer", split="train")
test  = generate_loads(60,  seed=1, records="peer", split="test")
```

### 4.4 Saving: HDF5

Results are saved in **HDF5**, a standard file format for scientific arrays (think "a folder of
NumPy arrays inside one file"). Each case is a group holding its time series (`t`, `acc`, `force`,
`moment`, `eta`) and its settings (Hs, Tp, PGA, which record, and so on). This is the first piece of
the data format the roadmap asks the team to fix early. A demo file of 100 cases is in
`data/loads_synthetic_demo.h5` (16 MB, not committed to git).

---

## 5. How we know it's right: the tests

`tests/test_loads.py` has 13 automatic checks. Each compares the code against a known answer:

| Test | What it proves |
| --- | --- |
| JONSWAP energy and peak | The wave recipe has exactly the requested Hs, peaking at 1/Tp |
| Dispersion limits | Wave lengths are right in deep and shallow water |
| MacCamy–Fuchs → Morison | For a thin pile the diffraction formula gives the classic answer |
| Thin-pile force | Our inertia force equals a direct Morison calculation to within 1% |
| 1-hour sea | The simulated waves really have the requested height (within 5%) |
| Response spectrum | Matches an independent solver (SciPy) within 1% |
| AT2 reader | Reads the PEER file format correctly (values, time step, earthquake name) |
| Resampling | Changing the time step doesn't distort the signal |
| Processing + scaling | Exact target PGA; ground displacement ends within 2 cm |
| Event split | No earthquake appears in both training and testing |
| Generator | Same seed → identical cases; quake starts after the gap; save/load round trip |
| Split generator | Training and test cases use different earthquakes |
| PEER pipeline | Fake PEER files go all the way through; vertical components skipped |

Run them any time with:

```
.venv\Scripts\python.exe -m pytest tests -q
```

---

## 6. Two bugs found by the tests

1. **Ground "flying away".** The first version filtered the record and *then* trimmed it.
   Trimming after filtering re-introduced drift: the ground ended up to 1.2 m from where it started.
   Fix: trim first, then filter with zero padding, then apply a polynomial correction. It now ends within 2 cm.
2. **Impossible scaling for synthetic records.** All synthetic records come out at about 0.3 g, so
   a 0.05 g target needed a 6× reduction, breaking the 4× rule. But the 4× rule exists to protect
   the character of *real* records; a synthetic record's amplitude is arbitrary anyway. Fix: apply
   the rule only to PEER records.

---

## 7. Honest limitations (put these in the report)

1. **Storms last hours; our cases have 30–90 s of waves before the quake.** Simulating hours with
   OpenSees is far too slow. This means the build-up of pore pressure from the storm *before* the
   earthquake is under-represented, and that build-up is the very thing that makes the combination
   dangerous. This needs a decision in part 6: either lengthen the gap or start each run from a
   "pre-stormed" soil state.
2. **Linear waves.** Real storm waves have sharper crests and can break and slam into the pile. Linear
   theory misses this, so the biggest forces are underestimated. All waves come from one direction,
   in line with the shaking, which matches our 2D model.
3. **Only the force on the pile.** Waves also press directly on the seabed and can pump up pore
   pressure there; the wind also pushes on the turbine rotor. Neither is included.
4. **3D force, 2D model.** The force is for the whole round pile. The 2D model in part 6 works "per
   metre of thickness", so the force must be converted there.
5. **Synthetic earthquakes are placeholders.** They lack features of real records, such as the
   velocity pulses near a fault. The final dataset must use PEER records. Scaling is by PGA only.

---

## 8. How this connects to the rest of the project

- **Part 5 (mini-surrogate):** uses only the earthquake part (`case.acc`) as the shaking at the base
  of the part 3 soil column. The FNO learns: base shaking → pore-pressure field.
- **Part 6 (pile model):** uses both `acc` (at the base) and `force`/`moment` (on the pile).
- **Part 7 (dataset factory):** calls `generate_loads(..., records="peer", split=...)` with the same
  Latin hypercube idea, also sampling soil density and permeability.

---

## 9. What you need to do

Download the PEER records (about 100–200, from at least 20 earthquakes), following
`data/peer/README.md`. Then:

```
.venv\Scripts\python.exe -c "from loads.earthquakes import load_peer; r = load_peer(); print(len(r), 'records from', len({x.event for x in r}), 'earthquakes')"
```

Once that prints sensible numbers, `generate_loads(..., records="peer")` works immediately.

**Done (2026-10-21):** 353 usable records from 36 earthquakes, 46 of them from magnitude 7 or
larger. Keep the PEER CSV files in `data/peer/`; they supply the magnitudes used for the split.

---

## 10. Files

| File | What it does |
| --- | --- |
| `loads/waves.py` | JONSWAP spectrum, wave theory, MacCamy–Fuchs + Morison force on the pile |
| `loads/earthquakes.py` | PEER reader, cleaning, scaling, response spectrum, synthetic quakes, event split |
| `loads/combined.py` | `generate_loads()`: the one function call; HDF5 save/load |
| `loads/make_figures.py` | Makes all figures in `results/part4/` and `summary.json` |
| `tests/test_loads.py` | The 13 checks |
| `data/peer/README.md` | How to download real earthquake records |

---

## 11. Glossary

| Term | Meaning |
| --- | --- |
| Load history | How big a push is at every moment in time |
| Sea state | The wave conditions of a storm, summarised by Hs and Tp |
| Hs | Significant wave height: average height of the biggest third of waves |
| Tp | Peak period: time between the most energetic waves |
| Spectrum | Recipe saying how much energy each wave frequency carries |
| JONSWAP | The standard spectrum for storm seas |
| Phase | Where in its up-down cycle a wave starts; random phases make each storm unique |
| Linear (Airy) wave theory | Textbook model giving water motion under a smooth wave |
| Dispersion relation | Equation linking a wave's period, length and the water depth |
| Inertia force | Push from water speeding up and slowing down around the pile |
| Drag force | Push from water flowing past the pile |
| Morison equation | Classic formula adding inertia and drag forces on a slender pile |
| MacCamy–Fuchs | Exact inertia force for a wide pile that bends waves around itself |
| Diffraction | Waves bending around an obstacle |
| KC number | How far the water swings ÷ pile width; small means inertia dominates |
| Mudline | The seabed surface where the pile enters the soil |
| Moment | Turning effect = force × lever arm; what tries to tilt the pile |
| Wave steepness | Wave height ÷ wave length; above ~1/7 a single wave breaks (we limit Hs/L to 1/16) |
| Strong-motion record | Measured ground acceleration during a real earthquake |
| PGA | Peak ground acceleration, the biggest value in a record |
| Event | One earthquake (recorded by many stations) |
| PEER NGA-West2 | Free database of real earthquake records |
| AT2 file | PEER's text file format for one record |
| Resampling | Changing the time step of a signal |
| Aliasing | Fast wiggles masquerading as slow ones after bad down-sampling |
| Arias intensity | Running total of shaking energy |
| Significant duration | Time between 5% and 95% of Arias intensity |
| Baseline correction | Removing slow drift so integrated displacement stays realistic |
| High-pass filter | Removes slow components, keeps faster ones |
| Scale factor | Number the record is multiplied by to hit the target PGA |
| Synthetic earthquake | Computer-made shaking from filtered random noise |
| Kanai–Tajimi filter | Mimics how a site's soil amplifies certain frequencies |
| Response spectrum | Peak response of simple oscillators vs their period; an earthquake's fingerprint |
| Latin hypercube sampling | Random sampling that spreads points evenly over each range |
| Data leakage | Test data secretly resembling training data, making scores look too good |
| HDF5 | File format for storing many numeric arrays with labels |
