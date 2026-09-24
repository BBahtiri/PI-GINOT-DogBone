# Phase 6.2–6.3 — Step 1, and what the audit found instead

**Status.** Infrastructure and evaluation-only measurements complete. The
matched-epoch training arms are running; their numbers land in the last
section.

## The plan's mechanism does not exist in this codebase

Step 1 was justified two ways. The first was precision:

> PyTorch's L-BFGS default `tolerance_change = 1e-7` sits *below* float32
> machine epsilon (1.19e-7), so the inner loop can stop on arithmetic rather
> than convergence. Xu et al. (arXiv 2505.10949) report **34–117×**
> improvements from the dtype alone.

There is no L-BFGS anywhere in this repository. The trainer is Adam/AdamW
with a plateau scheduler, and it has no tolerance-based stopping rule at all.
The cited effect is real in its own setting and the mechanism that produces
it is absent here — which is the failure this project has now recorded three
times: *a measurement is evidence for the configuration it was made in;
quoting it outside that configuration is a hypothesis, not a citation.* This
time the hypothesis was in the plan rather than in the literature review.

That does not settle whether float64 helps. It removes the reason to expect
a large effect, and it means the effect has to be measured rather than
assumed.

## Float64 does nothing to the forward pass

Evaluating the two finished 1600-epoch checkpoints in float64 — identical
weights, no retraining, `load_state_dict` casts — against the same
references:

| arm | worst relative fp32-vs-fp64 change, 9 metrics × 8 geometries |
|---|---|
| parameter-conditioned | **2.5e-4** (on the section force) |
| point-cloud | 6.1e-2, on `v`, **on one geometry**; ≤1.3e-3 on the other seven |

Float32 evaluation is bit-reproducible run to run, so the point-cloud outlier
is a real precision effect and not run noise. It is also not arithmetic — see
below. Peak error, `K_t` and peak location are unchanged on every geometry in
both arms.

So the forward path is not precision-limited. Whatever float64 can do here,
it must do through the training dynamics.

## What the outlier actually was

The encoder picks its centroids by farthest-point sampling: a sequence of
argmaxes over accumulated distances. In eval mode the start point is fixed
and the procedure is deterministic — but an argmax is discontinuous, and a
dog-bone outline has long straight runs (gauge flank, grip flank) along which
many candidate points sit at nearly equal distance from the already-chosen
set. Ties are generic for this geometry family rather than exceptional.

Measured on the eight held-out geometries, float32 against float64:

| geo | 2 | 15 | 4 | 6 | 3 | 11 | 9 | 22 |
|---|---|---|---|---|---|---|---|---|
| same FPS centroid set | no | yes | no | no | yes | **no** | no | yes |
| ball-query slots changed (of 576) | 13 | 0 | 72 | 36 | 0 | **191** | 36 | 0 |

Geometry 11 changes five times more grouping slots than any other, and is the
one whose field moves.

**The causal test.** Pinning the float32 centroid and grouping selection into
the float64 pass, through the encoder's own sample-id cache, after the model
is constructed — `PointSetEmbedding.__init__` clears that cache, so pinning
before construction silently does nothing and was the first attempt:

| geo | ball flips | Δv free | Δv pinned | Δu free | Δu pinned |
|---|---|---|---|---|---|
| 11 | 191 | 5.25e-3 | **1.46e-4** | 1.65e-4 | **5.33e-6** |
| 2 | 13 | 1.49e-4 | **6.87e-6** | 9.41e-6 | **5.07e-7** |
| 6 | 36 | 1.38e-4 | 1.38e-4 | 2.12e-6 | 2.12e-6 |
| 22 | 0 | 1.10e-4 | 1.10e-4 | 9.18e-7 | 9.18e-7 |

Pinning removes 96–97% of the difference where the selection changed enough
to matter, nothing where it did not, and nothing on the geometry whose
selection never changed. Geometry 6 flips 36 slots with no effect, so flip
count is not a sufficient statistic — but the discrete selection, not the
arithmetic, is what moves the field.

**The operator's latent is a discontinuous function of its input.** That is a
property of the encoder Phase 4 already identified as the bottleneck, and it
is the reason for the next section.

## So: is the operator a function of the geometry, or of the cloud?

A 1e-7 perturbation changing the answer is a warning; the deployment question
is what happens under a genuine re-sample. `verification/resample_sensitivity.py`
draws independent boundary clouds of the *same* outline — same mesh, same
320-point budget, different subsample, a fresh sample id each time so the
index cache cannot hide it — and scores each against one shared reference.

Three draws on geometry 11 (point-cloud arm, 1600 epochs):

| metric | mean | sd over draws | sd / mean |
|---|---|---|---|
| `v` rel L2 | 6.72e-2 | 2.73e-2 | **0.41** |
| `vm` rel L2 | 4.10e-2 | 4.42e-3 | 0.11 |
| peak error | −3.32% | 0.90 pp | 0.27 |
| `K_t` error | −5.33% | 0.80 pp | 0.15 |
| peak location `x*` | 0.916 | 0.048 | 0.05 |

Forty-one percent of the transverse error on this geometry is *which 320
points the model was handed*. Every number quoted in this repository is for
one arbitrary draw and carries no such term. The full eight-geometry,
eight-draw run is below.

## Output scaling: the constant was measured, and it is not the one the plan assumed

The plan set the per-component scale from the field norms: `‖v‖/‖u‖ ≈
0.03–0.06`, so the transverse output is "inherently the small one".

The heads do not produce the fields. The ansatz is

```
u = u_δ·ξ + ξ(1−ξ)·φ_u ,   v = η·φ_v ,   ξ = x/L_half,  η = y/H_grip
```

so what the network is asked for is `φ_u` and `φ_v`, and `η` spans the full
`[0, 1]`. Inverting the ansatz on the finite-element reference over the eight
held-out geometries:

| geo | 2 | 15 | 4 | 6 | 3 | 11 | 9 | 22 | mean |
|---|---|---|---|---|---|---|---|---|---|
| `‖v‖/‖u‖` | .054 | .051 | .042 | .060 | .052 | .062 | .047 | .054 | .053 |
| rms `φ_u` | .491 | .421 | .334 | .329 | .245 | .224 | .149 | .129 | .290 |
| rms `φ_v` | .112 | .098 | .074 | .092 | .079 | .086 | .062 | .069 | .084 |
| `φ_u/φ_v` | 4.37 | 4.29 | 4.53 | 3.57 | 3.12 | 2.62 | 2.41 | 1.87 | **3.46** |

The hard-BC layer had already absorbed most of the imbalance — its docstring
claims exactly that, and the claim is correct. The residual is **3.46×**, not
the 19× the field norms suggest; scaling from the fields would have
over-corrected sixfold. The arm uses `output_scale = [1.0, 0.289]`, `0.289 =
1/3.46`, which equalises the two heads on the bank mean.

Note the residual varies 4.53 → 1.87 with taper, so one constant cannot
equalise every geometry. A taper-dependent scale is available for free —
`x_max` and `y_max` are already decoder inputs — and is not tested here.

## Two dead levers found on the way

- `trainer.fit()` assigned `self.model.decoder.output_scale = 1.0` **on every
  epoch**, a leftover from a removed warmup. `DECODER_CONFIG["output_scale"]`
  has therefore been dead config for every run in this repository. Removed;
  the decoder owns its scale.
- `--h-factor` now exists on both study harnesses. They had the coarse 2.5
  wired in, which Phase 6.1 showed costs up to 0.85% on the reference's own
  `K_t` — a third of the better model's error, biased low. The default is now
  1.3.

## Predictions for the training arms

Restated in the Phase 6.1 norms, and stated before the run finished:

- **fp64.** The forward pass is not precision-limited and the L-BFGS
  mechanism is absent, so the honest prediction is **no effect beyond
  three-seed noise** (Phase 5 measured ±0.01 on the `v` ratio). If it moves
  `K_t` skill, that was not predicted and needs explaining rather than
  claiming.
- **outscale.** A 3.46× rescale multiplies the loss curvature along the
  transverse head's directions by ~12×, which is a real fraction of the 183×
  axial-to-transverse asymmetry Phase 3 measured. Prediction: the `v`-to-`u`
  error ratio falls, and the concentration metrics do not move, because the
  peak is an axial-field quantity.

## Results

Three arms, 500 matched epochs, bank 64, batch 4, lr 3e-4, energy form,
point-cloud conditioning, eight held-out geometries.

**First, the noise floor — because it had never been measured for these
metrics.** Phase 5's figure (±0.01 on the `v` ratio) is for a different
quantity, and carrying it over would have been this project's recorded error
for the fourth time. Three seeds of the baseline arm, same budget:

| metric | seed 0 | seed 1 | seed 2 | mean | seed CV |
|---|---|---|---|---|---|
| `u` | 1.064e-2 | 1.278e-2 | 1.075e-2 | 1.139e-2 | 10.6% |
| `v` | 1.588e-1 | 2.022e-1 | 1.699e-1 | 1.770e-1 | 12.7% |
| `vm` | 4.544e-2 | 4.613e-2 | 4.694e-2 | 4.617e-2 | **1.6%** |
| `vm` fillet | 6.872e-2 | 6.508e-2 | 7.512e-2 | 6.964e-2 | 7.3% |
| peak err | 11.32% | 10.90% | 8.61% | 10.28% | **14.2%** |
| fillet L∞ | 2.548e-1 | 2.113e-1 | 2.328e-1 | 2.330e-1 | 9.3% |
| `K_t` err | 11.74% | 10.21% | 9.36% | 10.44% | **11.6%** |
| \|N err\| | 0.76% | 1.32% | 0.92% | 1.00% | 28.7% |

**The stress-concentration metrics carry 9–14% seed noise.** Any Phase 6
lever claiming less than about 25% on them is unmeasurable at one seed. The
global von Mises L2 is the tight one at 1.6% — which is part of why the
project drifted into quoting it.

**Now the arms**, each against the three-seed baseline mean, with
`|z| = |ratio − 1| / seed CV`:

| arm | s/epoch | `u` | `v` | `vm` | fillet | peak | L∞ | `K_t` |
|---|---|---|---|---|---|---|---|---|
| baseline (3 seeds) | 2.35 | 1.139e-2 | 1.770e-1 | 4.617e-2 | 6.964e-2 | 10.28% | 2.330e-1 | 10.44% |
| outscale | 2.32 | 0.95 | 1.10 | 1.02 | 0.99 | 0.99 | **1.20** | 0.99 |
| fp64 | **4.62** | 0.99 | 0.88 | **1.07** | **1.15** | 1.20 | 1.11 | 1.14 |
| | |z| outscale | 0.45 | 0.75 | 1.51 | 0.16 | 0.05 | **2.15** | 0.12 |
| | |z| fp64 | 0.07 | 0.96 | **3.99** | **2.04** | 1.38 | 1.18 | 1.18 |

### Verdict on output scaling: no effect

Every metric is within one seed CV except the fillet L∞, which is 1.20×
**worse** at |z| = 2.15. Against seed 0 alone the arm looked like a 12%
improvement in `K_t` and a 22% degradation in `v`; against three seeds both
are 0.99 and 1.10, |z| = 0.12 and 0.75. Neither was real. The prediction — a
lower `v`-to-`u` ratio, unchanged concentration metrics — is neither
confirmed nor refuted; the lever simply does not move anything at this
budget.

The "capacity competition" story the single-seed numbers suggested (shrinking
the transverse head frees trunk capacity for the axial peak) had a clean
symmetric test available — raise the scale instead and see the opposite. It
is not worth running: the effect it would explain was noise.

### Verdict on float64: falsified, and the arm could not have been clean anyway

> **Correction, recorded in Phase 6.12.** The arm was confounded a second
> time, by a defect this phase introduced. The mechanical substitution of
> `dtype=torch.float32` in 6.2 turned `EnergyLoss.quad_points` into
> `def quad_points(..., dtype=torch.get_default_dtype())` — and a default
> argument is evaluated once, at import. `train_operator` imports the loss
> before it sets the run's precision, so under float64 the quadrature points
> stayed float32. The model promoted them internally and ran, but autograd
> returns a gradient at the dtype of its leaf, so **the displacement
> gradients — and so the strain energy the arm was trained on — came back
> float32.** Verified on the real code path: float64 parameters, float32
> points, `du/dx` float32.
>
> So this arm tested neither clean float64 arithmetic nor clean float64
> training. With the centroid-selection confound below, "falsified" is too
> strong: the accurate statement is **not demonstrated, by an arm that could
> not have demonstrated it**. What is unaffected is the evaluation result
> above — re-scoring identical weights in float64 went through `rescore`, not
> the quadrature, and moved every metric by under 2.5e-4 — so "the forward
> path is not precision-limited" stands. Only the training-dynamics claim
> falls. Fixed in 6.12: the dtype now resolves at call time.

The plan's falsification criterion was *"if FP64 changes nothing beyond
three-seed noise, then float32 was not a limit here and the literature's
effect does not transfer — say so and move on."* It fires. Nothing improves
outside noise, `vm` and the fillet L2 are measurably **worse**, and the arm
costs **1.92× the wall clock per epoch** — so at matched wall clock rather
than matched epochs it is far behind.

Two honest qualifications on the "worse":

1. The |z| = 3.99 on `vm` is measured against a 1.6% CV estimated from three
   runs. A CV that small from n = 3 is itself poorly determined, and the fp64
   arm is one seed.
2. More fundamentally, **this arm cannot be a clean precision ablation in
   this architecture.** Phase 6.2 showed that changing precision changes the
   encoder's centroid selection on five of eight geometries. The fp64 arm
   therefore trains against a different effective geometry encoding, not
   merely different arithmetic. Precision and conditioning are confounded by
   construction here.

So the reportable claim is the weaker and safer one: *float64 is not better,
and costs 1.92×.* Not *float64 harms training*.

### A note on the reference mesh

These arms were scored at `h_factor` 2.5, where the reference's own `K_t` is
biased low by up to 0.85%. That bias is shared by every arm, and it is
fourteen times smaller than the 11.6% seed noise on the same metric.
Re-scoring at 1.3 would correct a systematic error an order of magnitude
below the random one, so it was not done. The absolute `K_t` errors in Phase
6.1, where the comparison is against a constant baseline rather than against
another arm, are the ones that needed the finer mesh and have it.

## Where Step 1 leaves the plan

Both levers are spent and neither moved the quantity the paper is about. What
the step produced instead:

- a measured noise floor for the stress-concentration metrics (9–14%), which
  every remaining step now has to clear;
- two dead levers removed;
- the finding that the encoder's geometry latent is a discontinuous function
  of its input, and that re-sampling the boundary cloud moves the transverse
  error by 41% of itself on one geometry.

That last item is the one worth pursuing. Steps 3 and 4 attack the peak; the
resampling result says the input representation may be the larger term, and
it is measured on the arm the paper actually ships.
