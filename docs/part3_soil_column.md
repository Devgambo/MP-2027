# Part 3: Soil-column model and benchmark check

Roadmap part 3. Goal: a saturated sand column in OpenSeesPy whose pore-pressure build-up
matches a benchmark trend. "Done when: your curves match the benchmark trend."

## Choices

| Item | Choice | Why |
| --- | --- | --- |
| Sand model | PM4Sand (Boulanger & Ziotopoulou 2017) | Calibrated with one parameter (hpo) to a published liquefaction-resistance curve; standard in practice |
| Element | SSPquadUP (u-p, stabilised single point) | Stable for near-incompressible pore water; one element wide column |
| Column | 10 m uniform sand, 0.5 m elements, water table at surface | Small and fast; same code becomes the part 5 data generator |
| Boundaries | Rigid impermeable base, drained surface, left/right nodes tied (simple shear) | Standard 1D free-field column |
| Base input | Tapered 2 Hz sine, 0.15 g, 10 cycles (LEAP-style) | Needs no PEER records yet; real records arrive with part 4 |
| Units | kN, m, s, kPa, Mg/m^3 | |

## Three benchmarks

**A. Terzaghi 1D consolidation (exact solution).** Elastic, weightless column under a sudden
surface load. Checks the u-p coupling, permeability units and drainage boundaries.
Script: `soil_column/consolidation_check.py`.

**B. PM4Sand cyclic simple shear vs Idriss & Boulanger (2008).** Constant-volume cyclic DSS on
one element. hpo is calibrated per relative density so the sand liquefies (3% shear strain)
in 15 cycles at the I&B CRR (M7.5, 1 atm). Then the full CSR-vs-cycles curve is checked
against the I&B shape (CRR ~ N^-0.34). Script: `soil_column/calibrate.py`; writes
`soil_column/pm4sand_calibration.json`, used by every later run.

**C. Shaken column vs known liquefaction trends.** Loose, medium and dense columns.
Checks: loose sand liquefies, surface acceleration collapses after liquefaction, denser sand
builds up less pore pressure, and the liquefied zone agrees with a simplified-procedure
(Seed-Idriss style) FS < 1 zone. Script: `soil_column/site_response_check.py`.

Run all from the project root, in order:

```
.venv\Scripts\python.exe -m soil_column.consolidation_check
.venv\Scripts\python.exe -m soil_column.calibrate
.venv\Scripts\python.exe -m soil_column.site_response_check
```

Figures land in `results/part3/`.

## OpenSees pitfalls found on the way

1. `ops.fix` imposes **zero** displacement. Fixing a node after gravity or consolidation unloads the
   model. To lock a deformed position, use `ops.sp(node, dof, current_disp)` in a Constant pattern.
2. With `constraints('Transformation')`, an sp on the *retained* node of an `equalDOF` silently
   breaks the tie. Use `Lagrange` there, or never put an sp on a dof that is in an equalDOF
   (the column does the latter).
3. For SSPquadUP, the pore pressure is read from `nodeVel(node, 3)`, not `nodeDisp`.
4. Permeability input for u-p elements is k / (rho_w g), not the hydraulic conductivity.
5. PM4Sand: elastic stage (`updateMaterialStage ... -stage 0`) during gravity, then stage 1 and
   `setParameter -val 0 FirstCall` on every element.

## Results (Oct 16, 2026)

### A. Terzaghi: pass
Largest error is under 0.1% of the applied load at every time factor (Tv = 0.02 to 1.0).
Figure: `results/part3/terzaghi_check.png`.

### B. PM4Sand calibration: pass

| Dr | hpo | CRR15 (I&B) | Cycles to liquefy at the CSR whose target is 5 / 10 / 15 / 30 cycles |
| --- | --- | --- | --- |
| 35% | 0.653 | 0.090 | 3.5 / 8.5 / 15 / 39.5 |
| 45% | 0.555 | 0.113 | 4.5 / 9.0 / 15 / 37.5 |
| 55% | 0.492 | 0.147 | 4.5 / 9.5 / 15 / 36.0 |
| 65% | 0.471 | 0.199 | 5.0 / 9.5 / 15 / 35.0 |
| 75% | 0.878 | 0.312 | 5.0 / 9.0 / 15 / 37.0 |

The 15-cycle point matches by construction (that is the calibration). The 5, 10 and 30-cycle points
are the independent check, and they fall within about 7% of the I&B curve in CSR terms. The model's
curve is slightly flatter than N^-0.34, which is typical of PM4Sand. Stress paths show the textbook
cyclic-mobility shape. Figures: `dss_crr_curves.png`, `dss_example_dr55.png`.

### C. Shaken column (0.15 g, 2 Hz, 10 cycles): trends reproduced

| | Dr 35% | Dr 55% | Dr 75% |
| --- | --- | --- | --- |
| Liquefied zone (ru >= 0.95) | 2-10 m | 1-9 m | none (one transient 0.965 at 3.5 m) |
| Simplified procedure FS < 1 | whole column | whole column | none (FS ~ 1.05) |
| Surface acc., late / early shaking | 0.34 | 0.11 | 1.30 |
| Run time | 13 s | 58 s | 14 s |

- Loose and medium sand liquefy within 1-3 cycles. The surface acceleration then collapses (base
  isolation), which matches centrifuge observations. Dense sand does not liquefy and keeps transmitting motion.
- The liquefied zone agrees with the independent simplified-procedure prediction for all three densities.
- With a rigid base and 2 Hz input, CSR is roughly uniform with depth, and drainage at the top keeps the
  shallowest 1-2 m slightly below ru = 0.95 during shaking. Upward seepage pushes them to ~0.95 afterwards.

### Known limitations (to state in the report)

1. **Settlement is far too small** (~2 mm after 22 s). PM4Sand is known to underpredict
   post-liquefaction reconsolidation strain. This does not affect ru or tilt, which are our outputs, but
   do not report settlement from this model.
2. **Dissipation is slow.** ru is still ~0.95 at 15 s after shaking. With k = 1e-4 m/s and the very low
   stiffness of liquefied sand, full dissipation takes minutes. Part 5 must choose `t_post` with this in mind.
3. **Benchmark B is element-level against an empirical curve, and C checks trends only.** A direct
   comparison with a recorded centrifuge test (e.g. LEAP-UCD-2017 on DesignSafe) would make this stronger
   if the guide wants an experimental benchmark.
