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

All arms oracle-conditioned, scored on the same eight held-out geometries
against the same references at `h_factor` 1.3. "Overfit" means trained on
those same eight geometries — generalisation, conditioning and capacity are
all removed from the question, leaving only representability.

| arm | `u` | `v` | `vm` | fillet | \|peak\| | \|`K_t`\| | **`K_t` R²** | \|Δx\*\| |
|---|---|---|---|---|---|---|---|---|
| physics (energy, generalising) | 2.26e-3 | 2.85e-2 | 1.32e-2 | 2.01e-2 | 2.37% | 2.26% | **−0.25** | 0.031 |
| disp-supervised (generalising) | 2.64e-3 | 8.68e-3 | 5.48e-2 | 7.30e-2 | 5.55% | 5.21% | **−2.74** | 0.068 |
| disp-supervised (**overfit**) | 1.36e-3 | **2.19e-3** | 4.37e-2 | 5.41e-2 | 3.28% | 3.15% | **−0.55** | 0.044 |
| stress-supervised (**overfit**) | 2.39e-3 | 1.78e-1 | **7.75e-3** | **8.16e-3** | **0.37%** | **0.39%** | **+0.98** | **0.011** |
| predict the bank mean | — | — | — | — | — | 2.26% | 0.00 | — |

## The verdict: not architectural

**The architecture can represent the stress concentration to 0.39%, with the
peak located to 1.1% of `x_g` and `R² = +0.98`.** Nothing about the decoder,
the hard-BC ansatz or the capacity stands in the way. The limit is entirely
in what the objective sees.

## The sharper half: displacement accuracy does not buy stress accuracy

The displacement-supervised overfit reaches `u` = 1.36e-3 and `v` = 2.19e-3 —
a **0.2% displacement fit on the very specimens it is scored on**. There is no
generalisation gap, no conditioning gap, no capacity gap and no optimisation
gap left to blame. Its concentration factor is still **3.15%**, which is worse
than predicting a constant (2.26%), at `R² = −0.55`.

Stress is a derivative of the output. A field can be fit to a fraction of a
percent pointwise while its gradient over one small region is wrong, and the
concentration factor lives entirely in that gradient. This is the quantitative
version of that statement, and the literature review found nobody who had made
it: mDEM asserts it with field plots and no error tables; DCEM measures it but
is linear-elastic and its authors say the method does not extend to nonlinear
laws.

Note also the direction of the generalising displacement-supervised arm. It is
the **best transverse field this project has produced** — `v` = 8.68e-3, 3.3×
better than the physics objective — and simultaneously the **worst
concentration factor**, `R² = −2.74`. Improving displacement moved `K_t` the
wrong way.

## The tension, which Step 4 will have to face

The stress-supervised arm's transverse displacement is 1.78e-1 — **80× worse**
than the displacement-supervised arm's. The two objectives are not merely
different; at this budget they trade against each other. DCEM reports the same
shape of result in linear elasticity (one to two orders of magnitude better
stress, *worse* displacement than DEM). This reproduces it independently, at
finite strain, with a matched architecture and a matched point set — which is
the ablation the review found missing from the literature.

## What this does to the remaining plan

- **Step 4 (mixed `u`–`P`) is promoted from "the paper's centrepiece" to
  "the only thing that addresses the measured cause".** The cause is that a
  displacement-only objective cannot see the peak; making stress a primary
  variable is the direct answer. The ceiling it is aiming at is `R² = +0.98`,
  and the bar it has to clear is `R² = 0`.
- **Step 3 (adaptive quadrature) is now a supporting step, not a competing
  one.** It concentrates points where the equilibrium residual is large,
  which helps an objective that already sees stress. On its own it cannot
  create a signal the displacement-only objective does not contain.
- **Step 2 (multi-scale radius) is unaffected and still worth doing** for the
  reviewer hole and the resampling spread, but it cannot touch this.
- **The tension is a result in its own right** and belongs in the paper, with
  the caveat that it was measured under supervision rather than under a
  physics objective.

## Caveats, stated plainly

1. The +0.98 is an **overfit**. It bounds what a generalising model could
   reach; it is not a result a generalising model has reached. The gap that
   matters is between −0.25 and +0.98, and it is large.
2. The stress arm is handed the reference's own top-32 centroids as its peak
   set, so it is told where the peak is. The ablation below removes that term.
3. One seed per arm. Phase 6.2 measured 9–14% seed noise on these metrics at
   a 500-epoch physics budget; the differences here are 3–8× larger than that,
   but the supervised arms' own seed noise is unmeasured.
