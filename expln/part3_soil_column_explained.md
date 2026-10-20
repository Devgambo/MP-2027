# Part 3 explained: the soil column and the benchmark checks

This file explains what was built in roadmap part 3, why it was built that way, and what the
results mean. It assumes no civil engineering background. The technical summary lives in
`docs/part3_soil_column.md`; this one is the gentle version.

Figures referred to below are in `results/part3/`.

---

## 1. Why part 3 exists

The whole project is: run a slow soil simulator (OpenSees) many times, then train an AI model
(FNO) to copy it. An AI model trained on a wrong simulator learns wrong answers very confidently.
So before generating any training data, we have to show the simulator behaves like real soil.

Part 3 does that on the simplest useful problem: a vertical column of wet sand, shaken at its base,
with no pile yet. If the simulator gets this right, we can trust it when we add the pile later.

Roadmap "done when": *the curves match the benchmark trend.* They do (section 8).

---

## 2. The soil ideas you need (no equations yet)

### 2.1 Saturated sand = grains + water

Seabed sand is a pile of grains with the gaps between them (the **pores**) completely filled with
water. "Saturated" just means the pores are 100% water, no air.

Picture a jar packed with marbles and topped up with water. The marbles touch each other and
form a skeleton. The water fills everything in between.

### 2.2 Who carries the weight?

Any slice of soil has the weight of everything above it pressing down. That load is shared
between two carriers:

- **The grain skeleton**, through grain-to-grain contact. This part is called the **effective stress**.
- **The pore water**, through its pressure. This part is called the **pore-water pressure**.

```
total stress  =  effective stress  +  pore-water pressure
(weight above)   (grains pushing       (water pushing)
                  on grains)
```

This is the single most important idea in soil mechanics (Terzaghi's principle).

**Why it matters:** sand gets its strength from friction between grains, and friction depends on
how hard the grains press on each other, which is the effective stress. Water has no friction.
So more of the load on the water means a weaker soil.

### 2.3 Hydrostatic vs excess pore pressure

Even in still water, pressure grows with depth (about 9.81 kPa per metre). That is the
**hydrostatic** pressure. It is always there and is not a problem.

When the soil is shaken, the water pressure can rise *above* hydrostatic. That extra part is the
**excess pore pressure**. It is the dangerous part, and the thing we track.

### 2.4 How shaking creates excess pore pressure

Loose sand, when sheared back and forth, wants to compact (grains fall into the gaps). To compact,
water has to be squeezed out of the pores. During an earthquake the shaking is too fast for water
to escape, so the grains instead push onto the water and its pressure rises. Every cycle of
shaking adds a little more.

That is the wet-sponge analogy from the roadmap: squeeze fast and the water pushes back.

### 2.5 Liquefaction

As excess pore pressure rises, effective stress falls (the total weight is fixed, so if the water
takes more, the grains take less). If the excess pore pressure becomes equal to the original
effective stress, the grains carry almost nothing. The sand loses its friction and behaves like a
thick liquid. That is **liquefaction**. Buildings tilt, pipes float up, and monopiles can lean.

### 2.6 The number we track: ru

We measure how close the soil is to liquefying with the **excess pore-pressure ratio**:

```
ru = excess pore pressure / original effective stress
```

- ru = 0: nothing has happened.
- ru = 0.5: the water has taken half the load the grains used to carry. The soil is noticeably weaker.
- ru ≈ 1: liquefied.

We used ru ≥ 0.95 as "liquefied". This ru, as a function of depth and time, is exactly the field
the FNO will learn to predict.

### 2.7 Relative density (Dr): loose vs dense sand

The same sand can be packed loosely or tightly. **Relative density** Dr says where it sits
between its loosest possible state (0%) and its densest (100%).

- Loose sand (Dr ≈ 35%) compacts when shaken, so it builds pore pressure fast and liquefies easily.
- Dense sand (Dr ≈ 75%) tends to *expand* when sheared hard (grains must ride up over each other).
  Expanding sucks water in and lowers the pore pressure, so dense sand resists liquefaction.

Dr is one of the main inputs we will vary when generating the dataset.

### 2.8 Permeability

How easily water flows through the sand. High permeability means excess pressure drains away
quickly. Low permeability means it stays trapped. We used k = 1×10⁻⁴ m/s, typical for a fine
to medium clean sand.

---

## 3. What we modelled: the soil column

```
   surface  ── water can drain out here (pore pressure fixed at 0)
   ┌──┐
   │  │  ← 20 elements, each 0.5 m × 0.5 m
   │  │     10 m of uniform saturated sand
   │  │     water table at the surface
   │  │
   │  │     left and right edges tied together, so every slice
   │  │     moves sideways as a block (like a deck of cards sliding)
   │  │
   └──┘
   base ── rigid, no water can pass, shaken sideways
           (this is where the "earthquake" goes in)
```

Why a single column? Far from any structure, the ground during an earthquake mostly moves as
horizontal layers sliding over each other. A thin column captures this "free-field" behaviour
and runs in seconds. It is the standard first model in earthquake geotechnics.

The model is technically 2D (OpenSees needs a width), but with the left and right sides tied it
behaves as 1D: everything depends only on depth.

### The building blocks in OpenSees

| Piece | What we used | Plain meaning |
| --- | --- | --- |
| Element | `SSPquadUP` | A square cell that tracks both how the grains move (u) and the water pressure (p). "u-p" = solid displacement + pore pressure. It is the piece that lets grains and water interact. |
| Soil material | `PM4Sand` | The rulebook for how the sand skeleton responds to being squeezed and sheared, including pore-pressure build-up and liquefaction (section 4). |
| Water | Bulk modulus 2.2×10⁶ kPa | Water is almost incompressible; this number says so. |
| Units | kN, m, s, kPa, tonnes/m³ | 1 kPa = 1 kN/m². Atmospheric pressure ≈ 101 kPa. |

### How one simulation runs

1. **Gravity stage.** Switch gravity on and let the column settle under its own weight until the
   stresses and hydrostatic water pressure are in balance. During this stage PM4Sand is told to act
   elastically (like a spring), because real soil has been sitting there for thousands of years and
   is not "yielding" under its own weight.
2. **Switch on plasticity.** Turn PM4Sand's full nonlinear behaviour on.
3. **Shaking stage.** Move the base sideways with the input acceleration, step by step
   (every 0.005 s), and record at each step:
   - ru at every depth,
   - acceleration at the surface,
   - sideways movement and settlement of the surface,
   - shear stress and shear strain in every element.
4. **After shaking.** Keep running for 10–15 s to see what the pore pressure does once the
   shaking stops.

A small amount of extra damping (1%, "Rayleigh damping") is added to stop unrealistic numerical
ringing. Real soil damping comes mostly from PM4Sand itself.

If a time step fails to solve, the code retries it as 10 smaller steps with a more forgiving
solver. If that fails too, the run is flagged `ok = False` so that part 5 can discard it
automatically (the roadmap's "reject crashed runs").

### The input shaking

We don't have real earthquake records yet (those come with part 4 from the PEER database). So
we used a **tapered sine wave**: 2 shakes per second (2 Hz), peak 0.15 g (15% of gravity), 10 full
cycles, with smooth ramp-up and ramp-down. This is the same kind of input used in famous
centrifuge experiments (LEAP), because it is simple and repeatable.

---

## 4. PM4Sand and its one tuning knob

### What PM4Sand is

PM4Sand (Boulanger & Ziotopoulou) is a widely used sand model built specifically for
liquefaction. Given how the sand is sheared, it decides how stiff the sand is, how much it
wants to compact or expand, and therefore how pore pressure changes.

### Its inputs

- **Dr**: relative density (section 2.7).
- **G0**: how stiff the sand is at small shaking. We computed it from Dr with the standard
  formula from the PM4Sand manual, so it is not a free choice.
- **Density**: computed from Dr.
- **hpo**: the tuning knob. It controls how quickly pore pressure builds up under cyclic shear.
  A larger hpo means the sand resists liquefaction for more cycles.

So the only number we actually had to choose is hpo, and section 6 explains how we chose it.

---

## 5. Benchmark A: Terzaghi consolidation (testing the plumbing)

### The question

Before testing anything about earthquakes, check that the water-flow part of the model is wired
correctly: units, drainage boundaries, and the coupling between grains and water.

### The classic problem

Take a soil layer, drained at the top, sealed at the bottom. Suddenly put a load on top.

- At first the water carries all the new load (it cannot escape instantly), so the excess pore
  pressure everywhere equals the load.
- Water slowly flows out through the top. The pressure drops first near the top, later at depth.
- Eventually the water pressure is back to hydrostatic and the grains carry the whole load.

This is called **consolidation**. Terzaghi solved it exactly with a formula in 1925. Having an exact
answer makes it a perfect test: if the simulator disagrees, something is wired wrong.

The speed of the process is measured by a dimensionless "time factor" **Tv**. At Tv = 0.02, pressure
has barely started dropping; at Tv = 1, about 90% is gone.

### What we did

Used the same column code, but with a simple elastic soil, no gravity, and a 100 kPa load applied
suddenly on top. Compared the pore pressure profile at six moments with Terzaghi's formula.

### Result

Figure `terzaghi_check.png`: the OpenSees points sit on the exact curves. Largest error is
less than 0.1% of the load. **Pass.** The water physics, drainage boundaries and permeability
units are right.

(Side benefit: this test is how we confirmed that in this element type the pore pressure is read
from the "velocity" output slot, not the "displacement" one. See section 9.)

---

## 6. Benchmark B: the cyclic shear test (testing the sand)

### The real-world experiment it imitates

In a soil lab, engineers take a small sample of sand, squeeze it vertically to mimic the weight of
the soil above, then push its top back and forth sideways, a fixed amount each way, cycle after
cycle. This is a **cyclic direct simple shear (DSS)** test. They count how many cycles it takes
until the sample liquefies (defined as the sample sliding 3% of its height in one direction).

### Two key terms

- **CSR (cyclic stress ratio)** = how hard you push sideways ÷ the vertical effective stress.
  It's the "strength of the shaking" in dimensionless form. Bigger CSR means harder shaking.
- **CRR (cyclic resistance ratio)** = the CSR the sand can survive for a given number of cycles.
  It's the "strength of the sand".

If CSR > CRR, the sand liquefies.

### The benchmark: Idriss & Boulanger (2008)

From decades of field cases where sites did or did not liquefy in real earthquakes, Idriss and
Boulanger built a standard relationship: for a sand of a given density, what CRR it has.
Practitioners worldwide use it. It gives two things:

1. **CRR at 15 cycles** for each Dr (15 cycles ≈ a magnitude 7.5 earthquake), and
2. **how CRR changes with the number of cycles**: CRR ∝ N^(−0.34). Fewer cycles let the sand
   survive a harder push; more cycles mean it fails under a gentler one.

Calibrating PM4Sand so its simulated lab test follows this curve is exactly how PM4Sand's
authors recommend it be used.

### How we ran the test in OpenSees

One square element of PM4Sand:

1. Squeeze it vertically to 101.3 kPa (about the stress 10 m down in the ground), letting it
   compress as soil would.
2. **Lock its height.** If the volume can't change, the sand can't compact; instead the vertical
   stress drops. That drop is exactly equal to the pore-pressure rise you'd get with water trapped
   inside. This "constant-volume" trick is the standard way to run an undrained test with a dry model.
3. Push the top sideways until the shear stress hits +CSR × 101.3 kPa, reverse to −CSR × 101.3,
   and repeat.
4. Stop when the shear strain reaches 3% and record the number of cycles N.

### Calibrating hpo

For each density (35, 45, 55, 65, 75%), we searched for the hpo that makes the sand liquefy in
**exactly 15 cycles** when pushed at the Idriss & Boulanger CRR. The search method is bisection:
try an hpo, too few cycles means raise it, too many means lower it, halve the gap, and repeat 12 times.

| Dr | hpo found | target CRR at 15 cycles |
| --- | --- | --- |
| 35% | 0.653 | 0.090 |
| 45% | 0.555 | 0.113 |
| 55% | 0.492 | 0.147 |
| 65% | 0.471 | 0.199 |
| 75% | 0.878 | 0.312 |

These are saved in `soil_column/pm4sand_calibration.json`. Any later run with a Dr in between
uses a linearly interpolated hpo, so the part 7 dataset can sample any density from 35% to 75%.

### The real test (the part that isn't guaranteed)

Hitting 15 cycles at the 15-cycle CRR is guaranteed, because we tuned for it. The honest check is
whether the *rest* of the curve follows. So for each density we also pushed at the CSRs that
should cause liquefaction in 5, 10 and 30 cycles.

| Dr | target 5 | target 10 | target 30 |
| --- | --- | --- | --- |
| 35% | 3.5 | 8.5 | 39.5 |
| 45% | 4.5 | 9.0 | 37.5 |
| 55% | 4.5 | 9.5 | 36.0 |
| 65% | 5.0 | 9.5 | 35.0 |
| 75% | 5.0 | 9.0 | 37.0 |

Close at 5 and 10. At 30 the model takes a few more cycles than the target. On a log scale that is
only about 7% in CSR, which is normal for PM4Sand and acceptable. **Pass.**
Figure: `dss_crr_curves.png` (lines are the target, circles are OpenSees).

### Reading the example figure (`dss_example_dr55.png`)

- **Right panel (stress path):** each spike is one push. The vertical effective stress starts at
  ~101 kPa and steps leftwards cycle after cycle, so the grains carry less and less (pore pressure
  is rising). Near zero it forms a "butterfly" shape. That is **cyclic mobility**: at near-zero
  stress the sand briefly stiffens each time it is sheared hard (it tries to expand), then softens
  again when the push reverses. This shape is the textbook signature seen in real lab tests.
- **Left panel (stress vs strain):** for many cycles the strain stays tiny, then suddenly the
  loops become wide and flat (the sand slides 1%, 2%, 3% with little resistance). That is liquefaction.

---

## 7. Benchmark C: shaking the full column

### What we ran

The 10 m column with three sand densities: loose (35%), medium (55%), dense (75%). Same 0.15 g,
2 Hz shaking for all three.

### What real experiments tell us to expect

From centrifuge experiments (scaled-down soil models spun to mimic full-size gravity, such as
VELACS and LEAP) and real earthquakes:

1. Loose sand liquefies; ru climbs to about 1.
2. Once the sand liquefies, it can't pass shaking upward, so the surface acceleration drops
   sharply (the liquid layer acts like a shock absorber).
3. Denser sand builds up less pore pressure.
4. The liquefied zone should agree with the standard hand calculation engineers use (below).

### The independent hand check: the "simplified procedure"

Engineers usually check liquefaction with a quick calculation (Seed & Idriss's simplified
procedure). For each depth they compute:

```
factor of safety FS = CRR (sand strength) / CSR (shaking demand)
```

FS < 1 means liquefaction is expected. We computed CSR from an elastic run of the same column,
and CRR from our calibrated curve, corrected for depth (deeper sand under more pressure is a bit
weaker in this ratio sense; this is the "Kσ" correction). This calculation never looks at the
nonlinear OpenSees result, so agreement between them is meaningful.

### Results

| | Loose 35% | Medium 55% | Dense 75% |
| --- | --- | --- | --- |
| Where ru ≥ 0.95 (OpenSees) | 2–10 m | 1–9 m | nowhere (one brief 0.965 at 3.5 m) |
| Where FS < 1 (hand check) | whole column | whole column | nowhere (FS ≈ 1.05) |
| Surface shaking, late ÷ early | 0.34 | 0.11 | 1.30 |
| Run time | 13 s | 58 s | 14 s |

**Reading it:**

- Loose and medium sand liquefy within 1–3 cycles, through almost the whole column, which matches
  the hand check.
- Their surface acceleration falls to 11–34% of its early value: the shock-absorber effect.
  Dense sand keeps passing the shaking up (ratio above 1, because it amplifies a little).
- Dense sand has FS just above 1, and OpenSees shows high-but-not-liquefied ru. The two methods agree.

**Figures:**

- `column_loose_histories.png`: top, base vs surface acceleration (surface dies after ~1 s);
  middle, ru at four depths jumping to ~1 within the first couple of seconds; bottom, settlement.
- `column_loose_ru_field.png`: ru as a depth × time picture. **This exact kind of image is what
  the FNO will learn to produce in part 5.**
- `column_density_fs.png`: left, peak ru vs depth for the three densities; right, the hand-check FS.

### Why the top 1–2 m liquefy last (and why that's fine)

You might expect the surface to liquefy first. Two effects work against that here:

- The surface is where water drains out, so pressure there leaks away during shaking.
- With a rigid base and this input, the shaking demand ratio is roughly the same at all depths,
  while the depth correction makes deeper sand slightly easier to liquefy.

After the shaking, water flowing up from the liquefied layers below pushes the top layers to
ru ≈ 0.95 too. This "upward seepage" is a real, documented effect.

**Verdict: Benchmark C reproduces all the expected trends. Pass at trend level.**

---

## 8. So, is part 3 done?

Yes, by the roadmap's definition ("curves match the benchmark trend"):

- A: the water physics matches an exact solution.
- B: the sand matches the industry-standard liquefaction resistance curve.
- C: the full column shows every expected behaviour and agrees with the hand check.

---

## 9. Bugs we hit and fixed (useful to know for later parts)

1. **`fix` means "fix at zero", not "fix where it is now".** After squeezing the test sample,
   locking its height with `fix` snapped it back to its original height and released the squeeze.
   The sample then liquefied instantly, which was nonsense. Fix: lock it with `sp` at its current
   position.
2. **A constraint setting silently broke a link.** OpenSees's `Transformation` way of handling
   constraints dropped the "move together" link between the two top nodes once one of them was
   also locked. The sample was then being twisted instead of sheared. We found it by printing the
   strains (a sideways stretch appeared where there should be none). Fix: use `Lagrange`
   constraints in the element test, and design the column so no node is both locked and linked.
3. **Pore pressure is hidden in the "velocity" slot.** For this element type, OpenSees stores the
   water pressure in the velocity output of the third degree of freedom. The Terzaghi test
   confirmed this.
4. **Permeability units.** OpenSees wants permeability divided by the unit weight of water
   (k / 9.81 here), not the plain value from a soils textbook.

---

## 10. Honest limitations (put these in the report)

1. **Settlement is far too small** (~2 mm, real would be centimetres). PM4Sand is known to
   underestimate how much sand compacts as water drains out after liquefaction. Our outputs are
   pore pressure and tilt, so this doesn't hurt the project, but never quote settlement from this model.
2. **Pore pressure drains slowly.** 15 s after shaking stops, ru is still ~0.95. Liquefied sand is
   very soft, so draining takes minutes. When generating the part 5 dataset we must decide how long
   to keep simulating after the shaking, as a trade-off between capturing dissipation and run time.
3. **No direct comparison with a physical experiment yet.** B compares to an empirical design
   curve; C checks trends. If the guide wants it, the LEAP-UCD-2017 centrifuge data (free on
   DesignSafe) would give a test-vs-simulation comparison.
4. **Synthetic input.** Real earthquake records (part 4) have many frequencies, not one clean sine.
   The behaviour should carry over, but the first real-record runs should be eyeballed.

---

## 11. How this connects to the AI part

`soil_column/column.py` has one function, `run_column(acceleration, dt, dr=..., perm=...)`.
It takes a base shaking history plus soil properties and returns the ru field (depth × time), the
surface acceleration and so on, in about 15–60 seconds.

In part 5 we will call it 200–300 times with different earthquakes and densities. The FNO then
learns:

```
input:  base shaking history (+ Dr)   →   output: ru field over depth and time
```

and is compared against an LSTM baseline. Part 3 is what makes those 300 runs trustworthy.

---

## 12. Files and how to rerun

| File | What it does |
| --- | --- |
| `soil_column/materials.py` | Sand properties from Dr, the Idriss & Boulanger CRR formula, loading hpo |
| `soil_column/element_test.py` | The single-element cyclic shear test (benchmark B) |
| `soil_column/calibrate.py` | Finds hpo for each Dr, checks the curve, makes the DSS figures |
| `soil_column/consolidation_check.py` | Terzaghi benchmark (A) |
| `soil_column/column.py` | The soil column model, reused in part 5 |
| `soil_column/site_response_check.py` | Shaken-column benchmark (C) |
| `soil_column/pm4sand_calibration.json` | The calibrated hpo values |
| `results/part3/` | All figures plus `site_response_summary.json` |

From the project folder, in this order (calibration takes ~10 min, the others ~1–3 min):

```
.venv\Scripts\python.exe -m soil_column.consolidation_check
.venv\Scripts\python.exe -m soil_column.calibrate
.venv\Scripts\python.exe -m soil_column.site_response_check
```

---

## 13. Glossary

| Term | Meaning |
| --- | --- |
| Saturated soil | Soil whose pores are completely full of water |
| Pore-water pressure | Pressure of the water in the gaps between grains |
| Hydrostatic pressure | The normal water pressure from depth alone (9.81 kPa per metre) |
| Excess pore pressure | Water pressure above hydrostatic, caused by shaking or loading |
| Effective stress | The part of the load carried by grain-to-grain contact; gives sand its strength |
| ru | Excess pore pressure ÷ original effective stress; ~1 means liquefied |
| Liquefaction | Sand losing almost all strength because the water carries the load |
| Relative density (Dr) | How tightly sand is packed, 0% (loosest) to 100% (densest) |
| Permeability (k) | How easily water flows through the soil |
| Shear | Sliding deformation, like pushing the top of a deck of cards sideways |
| Shear strain | How far it slides ÷ its height (3% strain = slid 3 cm per metre of height) |
| CSR | Cyclic stress ratio: shaking demand ÷ effective stress |
| CRR | Cyclic resistance ratio: the CSR the sand can survive for N cycles |
| Factor of safety (FS) | CRR ÷ CSR; below 1 means liquefaction expected |
| DSS test | Direct simple shear: lab test pushing a sample's top back and forth |
| Undrained / constant volume | Water can't escape, so the sample's volume can't change |
| Consolidation | Slow squeezing-out of water after a load, transferring load to the grains |
| Time factor (Tv) | Dimensionless measure of how far consolidation has progressed |
| Cyclic mobility | The butterfly-shaped stress path of sand near liquefaction |
| u-p element | Finite element tracking grain movement (u) and water pressure (p) together |
| PM4Sand | The sand constitutive model (rulebook for sand behaviour) used here |
| hpo | PM4Sand's tuning knob controlling how fast pore pressure builds up |
| Rayleigh damping | Small artificial damping to suppress numerical ringing |
| Free field | Ground far from any structure |
| Centrifuge test | Small soil model spun fast so its stresses match a full-size site |
| kPa | Kilopascal, 1 kN per m². Atmospheric pressure ≈ 101 kPa |
| g | Acceleration of gravity, 9.81 m/s². "0.15 g" is 15% of that |
