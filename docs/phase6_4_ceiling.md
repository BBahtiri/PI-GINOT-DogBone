# Phase 6.4 — is the peak failure architectural, or is it the objective?

## Why this comes before Steps 2, 3 and 4

Phase 6.1 found that the parameter-conditioned arm — the oracle, handed the
exact geometry, the ceiling Phase 4 established — sits at `R² = −0.25` on the
stress concentration factor. It matches a constant and does not beat it. So
the failure is **not** conditioning, and no encoder work can reach it.

Two explanations remain, and nothing measured so far separates them:

1. **Architectural.** The network outputs displacement; stress is a
   derivative of that output. A field can be fit to a fraction of a percent
   pointwise while its gradient at one small region is wrong, and the
   concentration factor lives entirely in that gradient. If so, no objective
   built on displacement can fix it, and mixed `u`–`P` (Step 4) is not an
   improvement but a necessity.
2. **Objective-side.** The architecture can represent the peak, and the total
   potential energy — a volume integral, in which a peak over a small region
   carries almost no weight — simply cannot see it. If so, adaptive
   quadrature (Step 3) and a stress-aware objective are the right attack and
   the architecture is fine.

The distinction decides which of the remaining steps is worth building, so it
is worth half a day before any of them.

## The test

Three arms, all oracle-conditioned so that conditioning is free and cannot be
the explanation, all scored on the same eight held-out geometries against the
same references at `h_factor` 1.3:

| arm | trained on | asks |
|---|---|---|
| **physics** | total potential energy | the shipped result — `R² = −0.25` |
| **disp-supervised** | per-component relative error on `u`, `v` | does a near-perfect displacement fit deliver the right concentration? |
| **stress-supervised** | von Mises at the element centroids, with an explicit peak term | can the architecture represent the peak when the objective targets it directly? |

The displacement-supervised arm already exists: `verification/results/oracle64`,
bank 64, 4000 epochs, oracle conditioning. It was scored in Phase 4 on `u`,
`v` and `vm` only, so re-scoring it costs a few minutes and no training.

## Prediction, stated before the run

**The displacement-supervised arm will have better displacement and *worse*
stress than the physics arm.** Phase 4's stored rows already point this way on
the two geometries the two val sets share — on geometry 11 it reaches `v` =
4.1e-3 against the physics arm's 1.29e-2, a 3× better transverse field, while
its von Mises is 4.99e-2 against 8.76e-3, 5.7× worse. If that survives a
matched re-scoring, then fitting the field does not control its gradient, and
`K_t` skill will be no better than the physics arm's `R² = −0.25`.

**Falsification.** If the displacement-supervised arm's `K_t` `R²` comes out
clearly above zero, then the architecture can deliver the concentration from
a displacement fit, the physics objective is what is failing, and Step 3
(adaptive quadrature) rather than Step 4 is the priority. If *both* supervised
arms stall at `R² < 0`, the limit is architectural and neither Step 3 nor
Step 4 as currently specified will clear the bar — the output representation
has to change.

Note the confounds being removed: the Phase 4 numbers were scored on a
different held-out set (`2,11,13,16,19`) at the coarse `h_factor` 2.5, so they
are indicative only. The comparison below is matched.

## Results
