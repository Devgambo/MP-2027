# Part 6: 2D pile model with waves

Roadmap part 6. "Do: add the monopile to the soil model, then add the wave force on the pile.
Record how long one run takes. Done when: one combined run gives physically sensible results."

**Result in one line:** the 2D soil–pile–turbine model is built, its static state and dynamic
properties pass every check, and a combined storm + earthquake run completes without crashing and
shows the expected behaviour (waves alone harmless, earthquake liquefies the loose sand, the pile
tilts and settles permanently). Two caveats: the pore-pressure field shows a **checkerboard
(odd–even) instability** in the far field once the sand liquefies, so pore-pressure values are not
yet dataset quality, and one combined run takes **3–3.5 hours** on the laptop, far above the
roadmap's 15-minute warning level. Both are carried into the break (see the end of this file and
`NEXT_SEMESTER_REMAINING.md`).

```
.venv\Scripts\python.exe -m pile_model.checks                 # 4 runs in parallel + figures (~3.5 h)
.venv\Scripts\python.exe -m pile_model.checks --checks-only   # gravity state, frequency, damping (~5 min)
.venv\Scripts\python.exe -m pile_model.checks --figures-only
.venv\Scripts\python.exe -m pytest tests/test_pile_model.py   # 6 tests (~2 min)
```

## Model (`pile_model/model.py`)

Units kN, m, s, kPa, t. x horizontal (0 at the pile axis), y up (0 at the model base).

| Item | Choice | Why |
| --- | --- | --- |
| Geometry | 2D plane strain, out-of-plane thickness = pile diameter D = 8 m | The slice gets the full pile bending stiffness and the full wave force; 3D flow of soil around the pile is missing, so tilt is over-predicted (conservative) |
| Soil domain | 100 m wide × 30 m deep, graded mesh (1 m cells near the mudline and around the pile, up to 12 m wide far away): 312 soil elements | Wide enough that the far field behaves like a free-field column |
| Soil | SSPquadUP u-p elements + PM4Sand calibrated in part 3; Dr = 40%, k = 1e-4 m/s for the checks | Same model as the benchmarked column |
| Boundaries | Rigid impermeable base (shaken), drained mudline, left and right edges tied (periodic) | Standard free-field conditions |
| Turbine | Generic 10 MW class (DTU 10 MW, rounded): pile D = 8 m, wall 80 mm, embedded 24 m (L/D = 3), water depth 30 m, platform +10 m, hub +119 m, tower D 7.5 m / wall 35 mm, rotor-nacelle mass 674 t | Representative modern offshore turbine |
| Pile below the mudline | The steel tube plus its soil plug as a stiff elastic block 8 m wide (68 elements), with E_eq = EI_pile / (t·w³/12) ≈ 9.9e6 kPa so its bending stiffness equals the steel pile's | A thin 2D wall would not hold its soil plug; the block reproduces the pile's rigid-body rotation |
| Pile wall | Soil nodes at the wall are duplicated, tied to the block horizontally; vertical friction springs (Steel01, slip at τmax = 0.8·σ'v·tan 22°, API-style) | Lets the soil settle under its own weight without hanging on the pile; then limits skin friction |
| Above the mudline | 19 elastic beam-columns (P-Delta) from the mudline to the hub; submerged part carries internal water + added mass (Ca = 1); rotor-nacelle as a lumped mass | Captures tower sway and the turbine's weight acting on the tilted pile |
| Cap | Stiff arms from the beam's base node to every block top node | The block's rotation is the pile tilt |
| Loads | Part 4 wave force and mudline moment at the cap; earthquake as base acceleration | Same generator that will feed the dataset |
| Damping | Rayleigh by region: soil 1% at 1 and 10 Hz; structure 1% at 0.25 and 2 Hz | As the soil column; typical structural damping |
| Time step | 0.05 s with waves only, 0.01 s from 1 s before to 2 s after the shaking; Newmark γ = 0.6, β = 0.3025; Newton, with 10 KrylovNewton substeps as the fallback | Waves are slow; shaking needs a fine step |

### Static stages (`gravity`)

1. Soil self-weight with PM4Sand elastic; the pile wall may slide vertically.
2. Friction springs are created at zero force (InitStrainMaterial removes the stage-1 offset);
   the turbine's 20.8 MN weight is ramped on (KrylovNewton: plain Newton stalls at the springs'
   slip corner).
3. PM4Sand switched to plastic (stage 1 + FirstCall).
4. The weight pattern is swapped for a constant one, so it stays on when the dynamic run restarts
   the clock at t = 0.

## Checks (`results/part6/checks.json`, `tests/test_pile_model.py`)

| Check | Result | Expected |
| --- | --- | --- |
| Free-field vertical stress at 8.5 m | 82.3 kPa | 82.9 kPa (soil weight) |
| Free-field K0 = σh/σv | 0.43 | 0.35–0.55 (normally consolidated sand) |
| σv next to the pile, 8.5 m / 3.5 m | 96 / 55 kPa (1.16× / 1.6× free field) | Not lower than free field (soil does not hang on the pile); raised by skin friction carrying part of the weight |
| Structure weight | 20.8 MN | 10 MW class: ~18–25 MN |
| Settlement under the weight | 14 mm | Centimetres at most |
| Tilt from gravity | 0 (symmetric) | 0 |
| Fixed-base first frequency (no soil) | 0.226 Hz | 10 MW monopiles: ~0.2–0.3 Hz; cantilever formula reproduced within 3% (test) |
| Soil–structure first frequency (pluck test) | 0.192 Hz (15% lower) | Lower than fixed base; typical drop 5–20% |
| First-mode damping (pluck test) | 1.9% | ~1–3% (structure + soil) |
| A run with no loads | Stays exactly at rest (ru, tilt and settlement < 1e-6) | 0; a regression test for the "weight vanishes at t = 0" bug |

## The combined run (`results/part6/`)

One strong storm and one strong earthquake on loose sand: Hs = 7 m, Tp = 12 s; Kocaeli 1999
(RSN1166, Izmit station, a test-set record) scaled to PGA 0.25 g, starting at t = 40 s and lasting
about 25 s; Dr = 40%, k = 1e-4 m/s; 20 s after the shaking (84.6 s in total). Run three ways (waves
only, earthquake only, combined) plus the part 3 column (30 m, 1 m elements) with the same shaking
as a free-field reference. All four ran in parallel.

| Run | Completed | Run time | Peak tilt | Final tilt | Peak hub displacement | Final settlement |
| --- | --- | --- | --- | --- | --- | --- |
| Waves only | yes | 5.1 min | 0.012° | 0.003° | 0.09 m | 1 mm |
| Earthquake only | yes | 189 min | 3.82° | 3.81° | 11.7 m | 123 mm |
| Combined | yes | 211 min | 3.01° | 2.96° | 9.4 m | 130 mm |
| 1D column (free field) | yes | 8.2 min | – | – | – | – |

Figures: `combined_histories.png` (inputs, tilt, hub displacement, settlement),
`combined_ru_snapshots.png` (ru field at four times), `free_field_check.png` (2D far field vs the
1D column at 3, 8 and 16 m). Numbers: `summary.json`.

### What is physically sensible

- **Waves alone are harmless:** tilt about 0.01°, recoverable, hub sway 9 cm, ru below 0.06 in the
  far field. A parked 10 MW turbine in a 7 m sea should behave like this.
- **The earthquake liquefies the loose sand over the full depth**, as the 1D column does (ru ≈ 1
  from about 5 s into the shaking).
- **Tilt ratchets:** it oscillates with the shaking and drifts one way more each cycle as the soil
  softens (0.3° at 5 s, 1.5° at 20 s into the shaking), keeps growing for about 10 s after the
  shaking stops (the 20 MN weight leaning on liquefied soil, P-Delta), then stops at a permanent
  value. This is the classic liquefaction failure pattern of piles and shallow foundations.
- **Settlement** happens almost entirely during the shaking (≈ 120 mm), as soil liquefies under the
  pile's skin friction.
- **Around the pile:** pressure is higher on the side the pile pushes into and negative (suction)
  on the side it moves away from near the mudline (ru up to about +1.9 and −1.4 after shaking).
- **Combined vs earthquake only:** the storm changes the permanent tilt by about 20% (3.0° vs
  3.8°). Here the waves happened to push against the tilt direction; in other cases they will add
  to it. This interaction is what the surrogate must learn.

### What is not yet right

1. **Checkerboard pore pressure.** Once the sand liquefies, the pressure field alternates
   node to node (0.8 / 1.1 / 0.8 ...), strongest in the wide far-field cells on one side of the
   pile. The far-field ru therefore swings between about 0.2 and 2.0 instead of levelling at 1.0
   like the 1D column (RMSE 0.38 at 8 m). Averaged over a few cells the field is about 1, so the
   tilt and settlement trends are believed, but the exact values are not dataset quality.
   - Cause: SSPquadUP stabilises the pressure field with α = h²/(4G). The original code used the
     element height and the initial soil stiffness (G = 2e5 kPa). The far-field cells are up to
     12 m wide but 1–3 m tall, and liquefied sand loses almost all its stiffness, so α was far
     too small. The 1D column never showed this because it is one element wide.
   - Tests (10 s of the strongest shaking, compared with the column):

     | Variant | Checkerboard (mean neighbour difference) | Far-field RMSE vs column (3 / 8 / 16 m) | Run time |
     | --- | --- | --- | --- |
     | h = largest side, G = 2e5 (now in the code) | 0.57 | 0.59 / 0.52 / 0.69 | 37 min |
     | h = largest side, G = 5e4 | 0.32 | 0.51 / 0.45 / 0.59 | 43 min |
     | Finer mesh (4 m cells), G = 2e5 | 0.43 | 0.70 / 0.55 / 0.62 | 44 min |
     | h = largest side, G = 1e3 | – | – | static stage fails |

     Stronger stabilisation helps; mesh refinement alone does not; too strong breaks the static
     stage. G = 2e4 and 5e3 were started but stopped unfinished at the end of Semester 1.
   - The full runs in the table above used the original α; `pile_model/model.py` now uses the
     largest element side (all 29 tests pass with it).
2. **Run time.** 3–3.5 h per combined run with liquefaction (nearly every step during strong
   shaking needed substeps). Speed tests on 10 s of strong shaking:

   | Setting | Run time | Substepped steps | Peak tilt |
   | --- | --- | --- | --- |
   | Newton, tolerance 1e-4 (used) | 57 min | 281 | 0.531° |
   | Newton, tolerance 1e-3 | **27 min** | 54 | 0.536° |
   | Bonded wall (no slip) | 55 min | 304 | 0.588° (changes the answer) |
   | KrylovNewton throughout | > 58 min (stopped) | – | – |

   Tolerance 1e-3 halves the time with a 1% change in tilt.
3. **Tilt is large.** 3–4° permanent tilt is far above the 0.5° serviceability limit. Expected
   direction for very loose (Dr 40%), fully liquefied sand under 0.25 g, but the magnitude is
   inflated by the 2D slice (no 3D flow of soil around the pile) and by skin friction that does
   not drop when the soil liquefies. Use the model for relative comparisons (what the surrogate
   must copy), not as a prediction for a real turbine.

## Known limitations (for the report)

- 2D plane strain: tilt over-predicted; absolute values not comparable to real 3D piles (stated in
  the roadmap from the start).
- Pile plus soil plug as a stiff block; no gapping between pile and soil.
- Interface friction fixed from the initial stress; does not drop when ru rises.
- No wind thrust (parked turbine), wave force applied at the mudline (cap) only, no wave pressure
  on the seabed.
- Storm lead-in of 40 s; real storms last hours (part 4 limitation).
- Checkerboard pore pressure in the far field (above), not yet fixed.
- One load case checked; behaviour across the parameter space is unknown until the pilot batch.

## Bugs found and fixed (useful for the "lessons" slide)

1. Soil hung on the stiff block during gravity (σv 32 instead of 84 kPa next to the pile): fixed by
   duplicating the wall nodes so the soil settles freely, then adding friction springs.
2. `setParameter E` on ElasticIsotropic resets stresses through the total strain: abandoned.
3. A 20 MN weight on plastic PM4Sand does not converge: all static loads in the elastic stage.
4. Fully bonded springs over-transferred friction: friction-limited Steel01 springs.
5. OpenSees eigen analysis returns 0 Hz with u-p elements (pressure dofs have no mass): pluck test.
6. **The turbine's weight disappeared at the start of every run:** it was applied with a
   time-ramped curve, and `setTime(0)` put the curve back at zero. The structure "jumped" upward
   and created spurious pore pressures (|ru| ≈ 2 with no load at all). Fixed with a constant
   pattern after gravity; guarded by `test_rest_stays_at_rest`.
7. NumPy integer node tags rejected by OpenSees: converted to int.

## Files

| Path | What |
| --- | --- |
| `pile_model/model.py` | `Turbine`, `Mesh`, `build`, `gravity`, `run`, `fixed_base_frequency` |
| `pile_model/checks.py` | Load case, parallel runs, static/frequency checks, figures, summary |
| `tests/test_pile_model.py` | 6 tests: cantilever frequency, 10 MW range, free-field stress and K0, soil not hanging on the pile, settlement and no gravity tilt, rest stays at rest |
| `results/part6/` | `checks.json`, `summary.json`, `combined_histories.png`, `combined_ru_snapshots.png`, `free_field_check.png`, `runs/*.npz` |
