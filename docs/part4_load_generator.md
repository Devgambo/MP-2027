# Part 4: Load generator

Roadmap part 4. "Done when: one function call returns any number of realistic combined load
histories."

```python
from loads import generate_loads, save_h5
cases = generate_loads(100, seed=0)                          # synthetic quakes (placeholder)
cases = generate_loads(100, seed=0, records="peer")          # real records from data/peer/
cases = generate_loads(300, seed=0, records="peer", split="train")   # whole events held out
save_h5(cases, "data/loads.h5")
```

Each `LoadCase` has `t, acc (m/s^2), force (kN), moment (kN*m), eta (m)` on a common 0.01 s
grid, plus `params` (Hs, Tp, PGA, record, event, scale factor, gap, KC, ...).

## Design

| Piece | Choice | Notes |
| --- | --- | --- |
| Sea state | JONSWAP, gamma = 3.3, 256 random-phase components, jittered frequencies (no repeat period) | Spectrum normalised so 4 sqrt(m0) = Hs exactly |
| Kinematics | Linear (Airy) wave theory, dispersion solved by Newton | Integrated to mean water level, no stretching |
| Inertia force | MacCamy-Fuchs per component (equals Morison Cm = 2 when kD -> 0) | D/L reaches ~0.3 for short components; plain Morison would overpredict |
| Drag force | Morison, Cd = 1.0, on total velocity, 40-point depth integration | ~10% of peak force; KC = 0.6-7.4 (inertia regime) |
| Pile / site | D = 8 m, water depth 30 m, rho = 1025 kg/m^3 | Arguments of `generate_loads` |
| Earthquakes | PEER NGA-West2 `.AT2`, or synthetic (Kanai-Tajimi + Clough-Penzien + envelope) | Synthetic labelled `source="synthetic"`; placeholder only |
| Processing | resample to 0.01 s -> trim to Arias 5-95% (+2 s / +5 s) -> taper, zero-padded 0.1 Hz high-pass (Boore 2005) -> polynomial displacement baseline | Final displacement within ~2 cm |
| Scaling | Linear to target PGA, PEER scale factor limited to [1/4, 4] | No spectral matching |
| Combination | Waves throughout; quake starts at t_eq after waves begin; 20 s post-shaking | Waves cosine-ramped over 2 Tp |
| Sampling | Latin hypercube over Hs 1-8 m, Tp 6-14 s, PGA 0.05-0.4 g, t_eq 30-90 s | Tp raised where Hs/L > 1/16 (11 of 100 cases) |
| Splits | `split_by_event`: whole earthquake sequences go to train / val / test, stratified by magnitude band (<6.5, 6.5-7, >=7; magnitude from the PEER CSVs); of 500 shuffles the one closest to 70/10/20 of records is kept | Prevents leakage between stations of one event or aftershocks of one sequence (e.g. Mammoth Lakes-01..06); every band appears in val and test |
| Storage | HDF5, one group per case, float32 gzip datasets, params as attributes | `save_h5` / `load_h5` |

## Verification (`tests/test_loads.py`, 15 tests, all pass)

- JONSWAP: Hs recovered to 1e-6, peak at 1/Tp.
- Dispersion: deep- and shallow-water limits, residual < 1e-10.
- MacCamy-Fuchs -> Morison Cm = 2 for small ka; thin-pile inertia force equals Morison to < 1%.
- Simulated 1-hour sea: 4 std(eta) within 5% of Hs (realised 2.01 / 5.06 / 7.97 m for 2 / 5 / 8 m).
- Response spectrum (Nigam-Jennings) equals `scipy.signal.lsim` SDOF to 1%.
- `.AT2` parser round trip; resampling preserves a sine; PGA scaling exact; drift < 2 cm.
- Event split has no leakage; stratified split keeps sequences together, puts every magnitude band
  in val and test, and lands within 5% of 70/10/20 records; generator reproducible by seed; HDF5 round trip.
- Real library (skipped if no files): every record has a CSV magnitude; val and test each hold an M >= 7 event.
- PEER pipeline end to end on fake `.AT2` files (vertical component skipped).

## Numbers (100 synthetic cases, `results/part4/summary.json`)

- Generation: 19.5 s for 100 cases. Case length 74-140 s.
- Peak wave force 0.44-6.5 MN, peak mudline moment 8-117 MN*m.
- Example Hs = 6 m, Tp = 11 s: peak force 3.8 MN, drag 10% of peak, KC 4.5.

## Limitations

1. **Storm length.** Real storms last hours; a case has 30-90 s of waves before the quake (cost).
   Wave pre-loading of pore pressure is therefore under-represented. Decide in part 6 whether to
   lengthen t_eq or start from a pre-conditioned state.
2. Linear waves: no second-order/nonlinear crests, breaking or slamming, so extreme forces are
   underestimated. Unidirectional waves, collinear with the shaking (fits the 2D model).
3. Only the pile force is generated. Wave pressure on the seabed (direct pumping of pore
   pressure) and wind thrust on the rotor are not included.
4. Force and moment are for the full 3D pile; the 2D plane-strain model needs a per-unit-thickness
   conversion (part 6).
5. Synthetic earthquakes are placeholders (no near-fault pulses, no real spectral shapes). The final
   dataset must use PEER records. Scaling is by PGA only.

## PEER library (downloaded 2026-10-20/21)

353 horizontal records, 36 events (170 stations; 46 from M >= 7, 56 from M 6.5-7). All process
without error; median drift after correction 5.5 mm (16 records > 5 cm, max 12 cm). Split:

| Split | Records | Events | Larger events |
| --- | --- | --- | --- |
| train | 248 | 26 | Chi-Chi, Duzce, El Mayor-Cucapah, Darfield, Kern County, Chuetsu-oki, Irpinia-01, ... |
| val | 35 | 3 | Hector Mine (7.1), Loma Prieta (6.9), Oroville |
| test | 70 | 7 | Kocaeli (7.5), Tabas (7.35), Northridge (6.7), San Fernando (6.6), ... |

## Next action for the user

Download PEER records (see `data/peer/README.md`), then:
`.venv\Scripts\python.exe -c "from loads.earthquakes import load_peer; print(len(load_peer()))"`
