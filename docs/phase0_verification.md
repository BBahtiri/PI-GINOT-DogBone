# Phase 0 verification report

Everything below was measured on this branch, on CPU (torch 2.14.0, numpy
2.4.4, scipy 1.17.1) against `checkpoints/best.pt` (epoch 1424) unless stated
otherwise. Reproduce with `pytest`, `python -m eval.dense_eval --build
--checkpoint checkpoints/best.pt`, and the snippets named in each section.

---

## 1. Field equivalence: the constitutive changes are numerically inert

Same checkpoint, same 5,000-point interior sample on `GEOMETRY_DEFAULT`,
evaluated on the pre-Phase-0 tree (`9f67941`) and on this branch.

| field | max abs (old) | max abs difference | max relative |
|---|---|---|---|
| u    | 9.999692e-01 | 0.000e+00 | 0.000e+00 |
| v    | 2.182045e-01 | 0.000e+00 | 0.000e+00 |
| P11  | 3.006870e+01 | 6.104e-05 | 2.030e-06 |
| P12  | 2.008277e+00 | 1.192e-07 | 5.936e-08 |
| P21  | 1.960087e+00 | 2.384e-07 | 1.216e-07 |
| P22  | 1.411496e+01 | 9.155e-05 | 6.486e-06 |
| detF | 1.020100e+00 | 0.000e+00 | 0.000e+00 |
| S11  | 3.095456e+01 | 6.104e-05 | 1.972e-06 |
| S22  | 1.377441e+01 | 9.060e-05 | 6.577e-06 |
| S12  | 1.941725e+00 | 3.576e-07 | 1.842e-07 |

Displacements are bit-identical — the decoder was not touched. Stresses agree
to ~2e-6 relative, which is float32 round-off: in float64 the old five-step
closure differs from the exact Lambert-W root by 2.6e-11 and the new one by
2.2e-16.

## 2. Cost of the new closure — smaller graph, but the closure was never the bottleneck

4,000 interior points, forward + backward through `equilibrium_residual`,
five repetitions, interleaved runs.

| | closure subgraph nodes | full residual graph nodes | fwd+bwd | peak RSS |
|---|---|---|---|---|
| old (5 unrolled steps) | 108 | 10,834 | 5356 / 5344 ms | 5137 / 5133 MB |
| new (12 detached + 2 differentiable) | 38 | 10,280 | 5130 / 4950 ms | 5097 / 5095 MB |

The closure subgraph shrinks 2.8x, but it is about 1% of the whole graph: the
six-layer cross-attention decoder dominates, so the end-to-end saving is
~5% in graph nodes, 4–7% in wall time, and nothing measurable in memory.

**This contradicts the plan document**, which called the closure's graph depth
"the dominant cost and the main conditioning risk". It is neither. The case
for the change is correctness, not cost:

- values exact rather than 2.6e-11;
- the second-derivative path is now verified (section 3);
- the silent `clamp(F33, min=0.01)` that could mask divergence is gone,
  replaced by a reported residual and a warning.

Do not budget a speedup from this. If Phase 2 needs one, it has to come from
the decoder.

## 3. The second derivative is the load-bearing detail

The plan's recipe — a `torch.autograd.Function` with a pure
implicit-function-theorem backward — is **wrong here**, and silently so.

`equilibrium_residual` differentiates P once with respect to x, and the
optimiser differentiates again with respect to the parameters. The closure's
*second* derivative is therefore on the training path. An IFT backward built
from detached saved tensors is exact to first order and reports
`d2F33/dJ2D2 = 0`.

Measured against the exact Lambert-W solution (Richardson-extrapolated finite
differences, float64):

| implementation | rel. error in dF33/dJ2D | rel. error in d2F33/dJ2D2 |
|---|---|---|
| old, 5 unrolled Newton steps | 4e-12 | ~7e-8 (reference precision) |
| pure IFT / 1 differentiable step | 4e-12 | **0.08 – 0.14** |
| 2 differentiable steps (shipped) | 4e-12 | ~7e-8 |
| 3 differentiable steps | 4e-12 | ~7e-8 |

Two differentiable Newton steps from the converged root recover it exactly;
a third changes nothing. `tests/test_physics.py::test_F33_second_derivative`
and `::test_F33_gradcheck` both fail if `n_correct` is dropped to 1.

## 4. The force anchor: the plan had the sign backwards

The removed anchor was `N = E * (u_delta/L_half) * 2*H_gauge`. Two errors act
on it in opposite directions:

1. **Linearisation** — `E*eps` overstates the true uniform-bar nominal stress
   by 2.0–3.5% over the operating range (eps = 0.029–0.050). This is the term
   the plan identified, and the correction it proposed.
2. **Constant section** — much larger, and the other way. The grip and fillet
   are far stiffer than the gauge, so the gauge must stretch by more than the
   average `u_delta/L_half` to supply the prescribed elongation.

Over the full 24-geometry validation bank, against the varying-section
finite-strain solve (`physics/uniaxial.section_resultant_series`):

| anchor | mean ratio to truth | min | max |
|---|---|---|---|
| legacy small-strain | 0.9419 | 0.8485 | 1.0016 |
| uniform-bar finite-strain (the plan's fix) | 0.9173 | 0.8247 | 0.9788 |

The legacy anchor **undershoots on 23 of 24 geometries**, by 5.8% on average
and 15.1% at worst — it does not overshoot by 2–3.5%. Applying only the
finite-strain correction moves the target further from the truth on every
geometry.

The mechanism is visible in the data: the single geometry where the legacy
anchor is (marginally) high, val[11], has the longest gauge fraction of the
bank at 73.6% of the half-length — the near-prismatic case where the
linearisation term is all that is left. The worst case, val[19], has a gauge
fraction of 46.1%.

Default is now `resultant_anchor="series"`, with `"uniform"`,
`"small_strain"` and `"none"` kept for the H4 ablation, plus the
reaction-consistency term as an independent, target-free check.

## 5. Frozen dense-eval baseline

24 validation geometries, 40,000 Sobol interior points and 2,000 points per
boundary segment each, seed 31337. Nondimensional: interior is
`||Div P|| * L0/S0`, boundary is `||P.N|| / S0` (full vector on the
traction-free segments, enforced component on the rest).

| metric | mean over the bank |
|---|---|
| `eq_l2` (RMS) | 3.119e-02 |
| `eq_linf` | 1.462e-01 |
| `trac_right_arc` | 1.785e-02 |
| `trac_gauge_top` | 1.286e-02 |
| `part_bottom` | 1.582e-03 |
| `part_right_grip` | 1.834e-05 |
| `part_left_symmetry` | 1.237e-05 |

Per-geometry rows are in the `--checkpoint` output. Worst equilibrium
residual is val[15] (`eq_l2` 6.31e-02), best val[22]/val[21] (~1.5e-02).
The two traction-free segments are three orders of magnitude worse than the
hard-BC faces, which is the expected signature of penalty-enforced Neumann
conditions and is the gap Phase 2 targets.

**These are the first out-of-sample physics numbers this project has.**
Everything previously reported came from the loss on the collocation points
the optimiser was fitting.

## 6. Two pre-existing defects found while verifying

- `history["L_resultant"]` read `batch_loss["L_resultant_log"]`, a key
  `_train_one_epoch` never returns. With `.get(..., 0.0)` this logged a
  constant zero for every run to date — on the term the force-balance
  argument rests on. Fixed, with a test that parses both methods.
- `checkpoints/best.pt` (epoch 1424) carries `w_bar = 1000.0`, direct
  confirmation that the configured barrier warmup never ran, and 6 dead
  `geom_aux_head.*` tensors.

## 7. Test suite

33 tests, ~5 s on CPU: `pytest`. Two of them (`test_F33_second_derivative`,
`test_F33_gradcheck`) were verified to fail when the defect they guard is
reintroduced.
