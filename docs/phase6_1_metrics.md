# Phase 6.1 — scoring in the norm the claim is made in

**Status.** Done, evaluation only. No weights were trained; two existing
1600-epoch checkpoints were re-scored. The result changes what the rest of
Phase 6 has to demonstrate, so it is written up before Step 1 begins.

## Why

Every comparison in Phases 2–5 was scored in relative L2 over the whole
specimen. The specimen is mostly prismatic gauge, the state there is nearly
uniaxial, and the gauge carries most of the area — so an area-weighted L2 is
dominated by the easy part of the field. The paper's claim is not about the
average field. It is about **local stress-concentration accuracy at the
gauge–fillet transition**. A model can win the L2 and be wrong about that,
and nothing measured so far would have caught it.

Phase 6.1 adds the norms the claim is actually made in and applies them to
what already exists.

## What was added

`eval/compare_reference.py`

- `scf_ref`, `scf_pinn`, `scf_rel_err` — the stress concentration factor,
  `K_t = peak von Mises / area-weighted gauge mean`, computed the same way
  from both fields so the comparison is of the concentration rather than of
  the overall stress level.
- `x_peak_ref`, `x_peak_pinn`, `x_peak_err` — where the peak sits, in units
  of the tangency abscissa `x_g`. `1.0` is the gauge–fillet transition;
  larger is further into the fillet. An amplitude error and a localisation
  error are different faults with different fixes.
- `--convergence` — previously a declared flag that `main()` never read. It
  now runs the reference's own `K_t` mesh study and exits.

`verification/operator_study.py`, `verification/ablations.py`

- `REPORT_METRICS` and `mean_metric`, shared so the two harnesses stay
  comparable. `vm_peak`, `vm_fillet_Linf` and `scf` join the summary tables.
  Signed metrics are kept signed per geometry and aggregated by magnitude;
  averaging them signed would let one specimen's low peak cancel another's
  high peak and report a model as better than any prediction it made.
- `_REF_CACHE` is now keyed on `(geometry, mesh factor)`. It was keyed on the
  geometry alone, so asking for a finer reference in the same process
  silently returned the coarse one — a bug whose symptom is a refinement
  study that reports convergence.

`verification/rescore.py` — new. Loads a finished checkpoint and evaluates
it. Trains nothing.

## The reference has to be converged first

`K_t` is a maximum of a piecewise-constant field, so it is biased low and the
bias shrinks with `h`. That is harmless for a relative L2 and not harmless
here: `K_t` varies only ~3.4% across the geometry bank, so a reference that
is itself 1% off — and off by a geometry-dependent amount — eats a third of
the signal.

| mesh factor | worst \|K_t error\| vs finest |
|---|---|
| 2.50 (what the harnesses used) | 0.85% |
| 1.80 | 0.26% |
| 1.30 | 0.19% |
| 1.00 | 0.16% |

The study harnesses score at factor 2.5, where the reference is wrong by up
to 0.85% — about a third of the better model's total error, and biased in one
direction. Everything below is scored at **1.30**. Moving the reference from
2.5 to 1.3 raised the parameter-conditioned model's mean peak error from
1.84% to 2.37%: the coarse reference was flattering it by 29%.

## The result

Eight held-out geometries, taper 0.31–0.82, both arms at 1600 epochs, energy
form. Reference `K_t`: mean **1.226**, sd **0.0413** (3.37% of mean), range
1.158–1.301, correlation with taper only **+0.34** — so `K_t` is not a
function of taper alone, which is what makes it worth predicting.

| | u L2 | v L2 | vm L2 | vm fillet L2 | \|peak err\| | fillet L∞ | \|K_t err\| |
|---|---|---|---|---|---|---|---|
| parameter-conditioned | 2.26e-3 | 2.85e-2 | 1.32e-2 | 2.01e-2 | **2.37%** | 9.34e-2 | **2.26%** |
| point-cloud | 9.13e-3 | 1.51e-1 | 3.63e-2 | 5.86e-2 | **6.91%** | 1.74e-1 | **6.87%** |
| ratio | 4.04× | 5.31× | 2.76× | 2.91× | 2.92× | 1.86× | **3.04×** |

Peak error and `K_t` error agree to 0.16 percentage points on average, so the
gauge nominal is right in both arms and **the concentration error is the peak
error**.

### Neither arm beats a constant

Predicting the bank-mean `K_t` for every geometry gives **2.26%** mean error.

| arm | mean \|ΔK\|/K | R² on K_t | corr | sd(pred) | vs constant |
|---|---|---|---|---|---|
| parameter-conditioned | 2.26% | **−0.25** | +0.72 | 0.0615 | **1.00×** |
| point-cloud | 6.87% | **−6.28** | +0.06 | 0.1048 | **3.04×** |

The parameter-conditioned arm — the oracle, handed the exact geometry
parameters, the configuration Phase 4 established as the ceiling — predicts
the stress concentration factor exactly as well as a constant. It has the
trend (corr +0.72) but over-predicts the spread by 1.5× (sd 0.0615 against
the reference's 0.0413), and the amplitude error cancels the trend skill
outright.

The point-cloud arm has no trend at all (corr +0.06) and manufactures 2.5×
the variation that exists.

In relative L2 the parameter-conditioned arm reads as a strong result: 1.3%
von Mises on held-out geometries. In the norm the paper's claim is made in it
has **zero skill**. That is the finding.

### The point cloud does not know where the peak is

| | mean \|x\* error\| | peak placed in the gauge (x\* < 1) |
|---|---|---|
| parameter-conditioned | 0.031 x_g | 0 of 8 |
| point-cloud | 0.115 x_g | **4 of 8** |

The point-cloud operator puts the stress peak on the wrong side of the
gauge–fillet tangency on half the held-out geometries (geos 6, 11, 9, 22),
and 0.46 x_g into the fillet on another (geo 2). No relative L2 would show
this.

### Where the parameter-conditioned arm fails

Signed peak error against taper:

| geo | 2 | 15 | 4 | 6 | 3 | 11 | 9 | 22 |
|---|---|---|---|---|---|---|---|---|
| taper | 0.31 | 0.37 | 0.47 | 0.54 | 0.61 | 0.68 | 0.77 | 0.82 |
| x\*_ref | 1.106 | 1.091 | 1.075 | 1.054 | 1.090 | 1.065 | 1.024 | 1.033 |
| peak err | −1.4% | +0.3% | −0.9% | −0.8% | −0.8% | +1.3% | **+4.1%** | **+9.3%** |

Six of eight are within ±1.4%. The failure is two geometries at the
wide-gauge end, and it is a systematic overshoot, not noise: the signed error
correlates **+0.78** with taper.

**Two explanations, and this bank cannot separate them.** High taper means a
nearly prismatic specimen whose true concentration is weak and broad, which a
model trained mostly on sharper geometries would over-sharpen. But high taper
also puts the peak closer to the tangency point — x\*_ref is 1.024 and 1.033
on the two failures against 1.054–1.106 elsewhere — where the boundary's
curvature is discontinuous and the field's sharpest feature sits, which is a
resolution problem instead. In these eight geometries taper and x\*_ref
correlate at **−0.857**, so the two hypotheses are confounded and the
correlations above (+0.70 with taper, −0.685 with x\*_ref) do not choose
between them.

Separating them needs geometries that vary `R/L` independently of taper. That
is a bank change, not a model change, and it is cheap.

## What this does to the rest of Phase 6

The plan's Steps 1–5 all stated their predictions in relative L2 von Mises.
Those predictions are now the wrong ones to test. Restated:

- **Step 1 (FP64 + output scaling).** Was "von Mises L2 improves ≥1.3×".
  Should be: does the peak overshoot at high taper shrink? Float precision is
  a plausible cause of an L2 floor and an implausible cause of a
  taper-correlated bias, so the honest prediction is that Step 1 moves the L2
  and **not** the concentration factor. If it moves `K_t` skill, that was not
  predicted and needs explaining.
- **Step 3 (adaptive quadrature on the equilibrium residual).** Now has a
  sharp, per-geometry prediction: it should fix geos 9 and 22 specifically
  and do little for 2–11, which are already within ±1.4%. If it improves all
  eight uniformly, the mechanism is not the one claimed.
- **Step 4 (mixed u–P).** Makes stress a primary variable rather than a
  derivative of the displacement field, which is the most direct attack on a
  peak error. Prediction: it raises `K_t` R² above zero. Beating the constant
  baseline — R² > 0 — is now the bar for the whole phase, and nothing built
  so far clears it.

## Reproduce

```
python -m eval.compare_reference --convergence --geometries 2,6,22 \
    --out verification/results/kconv
python -m verification.rescore --h-factor 1.3 --conditioning oracle \
    --checkpoint verification/results/fixed_long/oracle-energy/last.pt \
    --tag param-cond --out verification/results/rescore/param_cond_h13.json
python -m verification.rescore --h-factor 1.3 --conditioning pointcloud \
    --checkpoint verification/results/fixed_long_pc/energy-w0/last.pt \
    --tag pointcloud --out verification/results/rescore/pointcloud_h13.json
```
