# Phase 6.3 — is the operator a function of the geometry, or of the cloud?

**Status.** Done, evaluation only. Eight independent boundary clouds per
geometry, eight held-out geometries, both 1600-epoch arms, one shared
reference each at `h_factor` 1.3.

## The question

A neural operator over shapes claims to map a geometry to a field. What it is
handed is a finite sample of the boundary, and the sample is arbitrary — a
different mesher, seed or point budget all describe the same dog-bone. If the
prediction moves when the sample changes, the operator is not well defined on
its stated input domain, and every error bar in this repository is missing a
term.

Phase 6.2 made this a live worry rather than a formality: the encoder's
centroid set is chosen by a sequence of argmaxes, a dog-bone outline has long
straight runs along which many candidates are equidistant from the chosen
set, and a **1e-7** perturbation of the input already changed the selection
on five of eight geometries. A genuine re-sample is far larger.

## The measurement

Same outline, same 320-point budget, different subsample, a fresh sample id
per draw so the encoder's index cache cannot hide the effect.

**Point-cloud arm, 1600 epochs:**

| metric | mean \|·\| | sd over draws | max sd | sd / mean |
|---|---|---|---|---|
| `u` rel L2 | 1.073e-2 | 1.84e-3 | 3.14e-3 | 0.17 |
| `v` rel L2 | 1.573e-1 | 3.98e-2 | 6.67e-2 | **0.25** |
| `vm` rel L2 | 4.065e-2 | 4.58e-3 | 8.26e-3 | 0.11 |
| `vm` fillet | 6.441e-2 | 6.78e-3 | 1.40e-2 | 0.11 |
| peak error | 2.37e-2 | 2.10e-2 | 3.42e-2 | **0.29** |
| `K_t` error | 2.26e-2 | 2.01e-2 | 4.10e-2 | **0.26** |
| peak location `x*` | 1.071 | 2.17e-2 | 6.02e-2 | 0.02 |

Per geometry, the concentration factor's spread across draws:

| geo | 2 | 15 | 4 | 6 | 3 | 11 | 9 | 22 |
|---|---|---|---|---|---|---|---|---|
| `K_t` error | +4.30% | +5.58% | −8.83% | −10.56% | −4.42% | −4.90% | −0.92% | +21.47% |
| sd over 8 draws | **4.10 pp** | 2.28 | 1.76 | 0.97 | 0.72 | 1.00 | 2.26 | **3.00 pp** |

**Null control.** The parameter-conditioned arm never reads the cloud, and
returns exactly 0.000e+00 spread on every metric and every geometry. The
harness measures the sampling and nothing else.

## What it means

The reference `K_t` varies by sd **0.0413** across the whole geometry
family — that is the entire signal the operator is asked to resolve.
Re-sampling the input moves the *predicted* `K_t` by sd **0.0244**, which is
**0.59× the signal**. On geometry 2 the sampling spread (4.10 pp) exceeds the
total geometry-to-geometry variation of `K_t` (3.37%) outright.

If the sampling noise were independent of the signal, it would cap the
achievable `R²` on `K_t` from a single cloud at

```
R² ≤ 1 − (0.0244 / 0.0413)² = 0.65
```

It also explains an internal inconsistency worth naming: the point-cloud
arm's `K_t` `R²` is −6.28 in Phase 6.1 and −8.88 here. Same weights, same
geometries, same reference — different draw. The reported figure itself moves
with the sampling.

## The obvious remedy, tested and refuted

If the problem is input noise, averaging the prediction over several boundary
samplings should remove it — no retraining, √K reduction, directly on the
paper's headline metric. The eight draws are already in hand, so this is post
-processing:

| clouds averaged | `R²` on `K_t` | corr | mean \|Δ`K_t`\|/`K_t` | sd(pred) |
|---|---|---|---|---|
| 1 | −8.88 | −0.11 | 7.97% | 0.1175 |
| 2 | −8.66 | −0.10 | 7.68% | 0.1153 |
| 4 | −8.51 | −0.10 | 7.61% | 0.1161 |
| 8 | −8.44 | −0.10 | 7.62% | 0.1156 |

Averaging buys nothing. The reason is visible in the last column: the
prediction's spread is 0.116, **2.8× the reference spread of 0.0413**, and it
does not shrink with averaging because it is not noise. The model's `K_t`
error is systematic, the sampling noise rides on top of it, and removing the
noise leaves the bias exactly where it was.

**So the sampling instability is real, bounded, and not the binding
constraint.** It caps the architecture at `R² ≤ 0.65`; the model sits at
−8.4. Both facts belong in the paper, and the second is the one to work on.

## What this does to the remaining plan

- The resampling term must be quoted alongside every point-cloud metric.
  Reporting one draw and calling it the operator's error is not defensible
  now that the spread is measured: 25% on `v`, 26% on `K_t`.
- Step 2 (multi-scale grouping radius) gains a second, sharper prediction. A
  multi-scale encoder aggregates over several radii, so a candidate that ties
  at one scale rarely ties at all of them. Prediction: **multi-scale reduces
  the resampling spread**, independently of whether it improves any field
  metric. That is a cleaner test of the radius hypothesis than the field
  metrics, because the seed noise on the field metrics (9–14%, Phase 6.2) is
  larger than the effect Phase 4 was looking for.
- Steps 3 and 4 target the systematic peak error, which this phase confirms
  is the binding term. Their predictions stand.
