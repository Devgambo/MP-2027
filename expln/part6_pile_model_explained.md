# Part 6 explained: the 2D pile model

This file explains roadmap part 6 in plain terms: what we built, why it is built that way, how we
checked it, what one storm + earthquake simulation showed, and what is still wrong with it. The
short technical version is in `docs/part6_pile_model.md`. Figures are in `results/part6/`.

---

## 1. Where part 6 fits

- Part 3 built and checked a **column of wet sand** (no pile).
- Part 4 built the **loads**: storm-wave forces and real earthquake records.
- Part 5 trained an AI on the sand column.
- **Part 6 puts the actual wind turbine into the sand** and hits it with waves and an earthquake.
  This is the simulator that will produce the 1,000 training examples in part 7.

Roadmap "done when": *one combined run gives physically sensible results*, and we must *record how
long one run takes*.

---

## 2. What is being modelled

Picture an offshore wind turbine in 30 m of water:

```
            rotor + nacelle (674 t) at 119 m above sea level
                 |
                 |   tower (steel tube, 7.5 m wide)
                 |
   ~~~~~~~~~~~~~~|~~~~~~~~~~~~~~  sea level        <- storm waves push here
                 |
                 |   monopile (steel tube, 8 m wide) in 30 m of water
                 |
   ==============|==============  seabed (mudline)
   wet sand     | |   wet sand
                | |   pile driven 24 m into the sand
                |_|
   ...................................  30 m of sand, then rock  <- earthquake shakes here
```

The turbine is a generic **10 MW** machine, the size being built today (rotor about 180 m across).
Its total weight is about 2,100 tonnes (20.8 MN).

**Why it can fail:** the sand holds the pile upright by pushing back on it. If an earthquake
liquefies the sand (part 3: the water pressure takes over and the grains stop pressing on each
other), the sand stops pushing back, and the heavy turbine leans over. Turbines can tolerate only
about **0.5° of permanent tilt** before they stop working properly.

---

## 3. How the model is built (and why)

### 3.1 A 2D slice, not the full 3D thing

We model a vertical **slice** 8 m thick (the pile's width) through the pile and the soil, like a
cut through a layer cake. Everything is assumed the same in front of and behind the slice.

- **Why:** a full 3D model takes many hours to days per run; we need about 1,000 runs.
- **The price:** in reality the sand can flow *around* the round pile; in a slice it can't, so the
  slice treats the pile like a long wall. **Tilt comes out too big.** This was accepted in the
  roadmap from the start: the AI copies the 2D simulator, and the method carries over to 3D later.

### 3.2 The soil

- 100 m wide, 30 m deep, divided into 312 rectangles (finite elements). They are small (1 m) near
  the pile and the seabed, where things change fast, and large (up to 12 m wide) far away.
- Each rectangle uses the **same sand model as part 3** (PM4Sand, calibrated against published
  liquefaction data), so the soil behaviour is already checked.
- The left and right edges are tied together, so the far-away soil behaves like the part 3 column.
- The rock at the bottom is where the earthquake enters; water can escape only through the seabed.

### 3.3 The pile under the seabed: a stiff block

A steel tube 8 cm thick is very thin compared with 8 m of width. In a 2D slice a thin wall would
let its inner soil (the **soil plug**) squeeze out, which can't happen in reality. So the pile and
its plug are modelled as **one solid block** 8 m wide and 24 m deep, made exactly as stiff in
bending as the steel pile. When the turbine leans, the block rotates, and that rotation is the
**tilt** we report.

### 3.4 The pile wall: sand can slide along it

Sand grips the pile wall by friction (like your hand on a rope). Two lessons from building this:

1. When the sand first settles under its own weight, it must be allowed to slide down past the
   pile. Otherwise the sand "hangs" on the stiff pile and the stresses come out wrong (we saw
   32 kPa instead of 84 kPa next to the pile before fixing this).
2. After that, friction springs connect them, but only up to a maximum grip (the standard
   offshore design formula). Beyond it, the pile slips.

### 3.5 Above the seabed: tower and turbine

The pile above the seabed and the tower are modelled as a flexible steel pole (beam elements)
with the rotor and nacelle as a heavy mass on top. The underwater part also carries the water
inside the tube and the water dragged along with it. The beam includes the **P-Delta effect**:
when the pole leans, the weight on top pushes it further over, as it does in reality.

### 3.6 The loads

- **Waves:** the part 4 wave force and the moment it causes, applied at the seabed.
- **Earthquake:** the part 4 earthquake record shakes the rock at the bottom of the model.
- **Weight:** the turbine's 20.8 MN, applied before anything else and kept on.

### 3.7 How the simulation runs in time

First, gravity: the soil and turbine settle under their weight with the sand made temporarily
elastic (so it can't fail while loading up). Then the sand is switched to its real liquefiable
behaviour. Then the clock starts: time steps of 0.05 s while only waves act (they're slow) and
0.01 s during the earthquake (fast shaking needs fine steps).

---

## 4. How we checked it (before any earthquake)

| Check | What we got | What it should be | Plain meaning |
| --- | --- | --- | --- |
| Soil pressure far from the pile | 82.3 kPa at 8.5 m depth | 82.9 kPa | The soil weighs what it should |
| Sideways vs downward pressure (K0) | 0.43 | 0.35–0.55 | Normal for settled sand |
| Soil pressure next to the pile | 1.16× far field at 8.5 m | ≥ 1× | The sand doesn't hang on the pile; it carries a bit of the turbine's weight through friction, as expected |
| Turbine weight | 20.8 MN | 18–25 MN for 10 MW | Right size of machine |
| Settlement under the weight | 14 mm | centimetres | Sensible |
| Natural frequency, pile clamped at the seabed | 0.226 Hz | 0.2–0.3 Hz for 10 MW | Matches the hand formula within 3% |
| Natural frequency with the soil | 0.192 Hz | lower than clamped | Soil is softer than a clamp, as it should be |
| Damping (how fast a sway dies out) | 1.9% | 1–3% | Sensible |
| A run with no loads at all | nothing moves | nothing moves | Catches a nasty bug we found (section 7) |

**Natural frequency** is how many times per second the turbine sways back and forth if you push
it and let go (about once every 5 s here). Designers keep it away from the rotor's and the waves'
frequencies; checking it is the standard sanity test for a turbine model.

---

## 5. The test simulation

One harsh scenario, run three ways so we can see what each load does:

- **Storm:** waves 7 m high (significant height), one every 12 s. A strong North Sea-type storm.
- **Earthquake:** Kocaeli, Turkey, 1999 (magnitude 7.5), strength scaled to 0.25 g, starting 40 s
  into the storm, about 25 s of shaking. (This record is in the AI's test set, so using it here
  costs nothing.)
- **Sand:** loose (density 40%, liquefies easily), medium drainage.
- Runs: **waves only**, **earthquake only**, **both together**, plus the part 3 sand column with the
  same earthquake as a reference for the far-away soil.

### 5.1 Results

| Run | Took | Biggest tilt | Tilt at the end | Top of tower moved | Pile sank |
| --- | --- | --- | --- | --- | --- |
| Waves only | 5 min | 0.012° | 0.003° | 9 cm | 1 mm |
| Earthquake only | 3.2 hours | 3.8° | 3.8° | 11.7 m | 123 mm |
| Both together | 3.5 hours | 3.0° | 3.0° | 9.4 m | 130 mm |

### 5.2 What it means (`combined_histories.png`)

- **Waves alone do nothing dangerous.** The turbine sways a few centimetres and comes back. Good:
  a parked turbine in a storm should be fine.
- **The earthquake liquefies the loose sand all the way down**, as the sand column does.
- **The tilt ratchets.** During the shaking the turbine rocks back and forth, but each rock goes a
  little further one way, because the soil is getting weaker. After the shaking stops, the tilt
  keeps growing for about 10 s (the heavy top leaning on soil that is still liquid), then stops.
  This ratcheting is exactly how real foundations fail in liquefaction.
- **3–4° permanent tilt** would put a real turbine out of service. The *direction* is right for
  very loose, fully liquefied sand under strong shaking, but the *size* is exaggerated by the 2D
  slice (section 3.1) and by the wall friction not weakening when the sand liquefies.
- **The storm changes the outcome by about 20%** (3.0° instead of 3.8°). In this case the waves
  happened to push against the lean; in other cases they will push with it. Capturing this
  interaction is the whole point of the project.
- **The pile sinks about 12 cm**, almost all during the shaking.

---

## 6. What is still wrong

### 6.1 The checkerboard

Look at `combined_ru_snapshots.png`: far from the pile, the pressure map has a **diamond /
checkerboard pattern**. Neighbouring points alternate high-low-high instead of being smooth. In
`free_field_check.png`, the far-away 2D soil swings between about 0.2 and 2.0, while the column
(black) sits calmly at 1.0.

This is a **numerical artefact**, not physics. The finite-element type we use has a known weak
spot: the pressure can form a zig-zag pattern that the equations can't "see". A built-in correction
term (a stabilisation setting) suppresses it, but its recommended size assumes:

- **square-ish elements:** ours far away are 12 m wide and only 1–3 m tall, and
- **normally stiff soil:** liquefied sand has almost no stiffness.

So the correction was far too weak exactly where the pattern appears. The part 3 column never
showed it because it is only one element wide, leaving no room for a zig-zag.

What we tried (10-second tests, compared with the column):

- Size the correction by each element's **longest side**: helped a little (now in the code).
- Make it **4× stronger** on top: halved the zig-zag, and the far field got closer to the column.
- A **finer mesh**: barely helped.
- **200× stronger**: the model broke before the earthquake even started.

**Is it a problem?** For this semester's checks, not really: averaged over a few points the
pressure is about 1, so the tilt and settlement trends are believable. For the AI dataset, yes:
the AI would learn to copy the zig-zag. It must be fixed before part 7, by finding the right
correction strength or switching to an element type that can't zig-zag.

### 6.2 It is slow

The roadmap warned: "if one 2D run takes over 15 minutes, act". Ours takes **3–3.5 hours** with
the earthquake. Why: liquefying sand is numerically hard, and almost every 0.01 s step during the
strong shaking had to be split into ten smaller steps to converge.

What we found:

- Loosening the solver's accuracy target (tolerance 1e-4 → 1e-3) **halves the time** and changes
  the tilt by only 1%.
- Other solver choices didn't help.

For 1,000 runs: about 1.5–2 hours each with the looser tolerance, 10 at a time on the laptop
means about a week of non-stop computing. On the college HPC (hundreds of cores) it is a day or
two. That's why the dataset waits for the holidays / HPC.

---

## 7. Bugs found on the way (worth a slide)

1. **Soil hanging on the pile.** The stiff pile held up the soil around it while it settled;
   fixed by letting the soil slide down the wall first.
2. **The disappearing turbine.** The weight was switched on gradually using a timer. When each
   simulation restarted its clock at zero, the timer went back to zero and **the 2,100 t turbine
   vanished**: the pile sprang upward and produced fake water pressures, even with no waves and
   no earthquake. Fixed, and a test now checks that a model with no loads stays perfectly still.
3. The OpenSees frequency calculator returns 0 for this kind of soil model, so we "plucked" the
   turbine instead: a push, then measure how fast it sways (like flicking a ruler on a desk).
4. Loading the full weight onto the liquefiable sand in one go failed; loading it while the sand
   is temporarily elastic works.

---

## 8. Glossary

| Word | Meaning |
| --- | --- |
| Monopile | The single large steel tube the turbine stands on |
| Mudline | The seabed surface |
| Plane strain / 2D slice | Model of one vertical slice, assuming everything is the same in front of and behind it |
| Soil plug | The sand inside the hollow pile |
| Skin friction | The grip between the sand and the pile wall |
| P-Delta | Extra leaning force from the weight on top of a leaning pole |
| Natural frequency | How often the structure sways per second after a push |
| Damping | How quickly a sway dies out |
| Tilt | The permanent lean of the pile; design limit about 0.5° |
| ru | Pore-pressure ratio: 0 = solid, 1 = liquefied (part 3) |
| Free field | Soil far from the pile, unaffected by it |
| Checkerboard / zig-zag | A false high-low-high pattern created by the numerical method |
| Stabilisation | A correction term in the element that suppresses the zig-zag |
| Substepping | Splitting a time step into smaller ones when the solver can't converge |
| Tolerance | How exact each step's solution must be before moving on |
| HPC | High-performance computing cluster: many computers working in parallel |
