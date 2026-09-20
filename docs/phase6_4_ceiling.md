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
against the same references at `h_factor` 1.3. **Overfit** means trained on
those same eight geometries — generalisation, conditioning and capacity are
removed from the question, leaving only representability.

| arm | `u` | `v` | `vm` | fillet | fillet L∞ | \|peak\| | \|`K_t`\| | **`K_t` R²** | \|Δx\*\| |
|---|---|---|---|---|---|---|---|---|---|
| physics (energy, generalising) | 2.26e-3 | 2.85e-2 | 1.32e-2 | 2.01e-2 | 9.34e-2 | 2.37% | 2.26% | **−0.25** | 0.031 |
| disp-supervised (generalising) | 2.64e-3 | 8.68e-3 | 5.48e-2 | 7.30e-2 | 3.86e-1 | 5.55% | 5.21% | **−2.74** | 0.068 |
| disp-supervised (overfit) | 1.36e-3 | **2.19e-3** | 4.37e-2 | 5.41e-2 | 2.62e-1 | 3.28% | 3.15% | **−0.55** | 0.044 |
| stress-supervised (overfit) | 2.39e-3 | 1.78e-1 | 7.75e-3 | 8.16e-3 | 2.78e-2 | **0.37%** | **0.39%** | **+0.98** | 0.011 |
| stress-supervised, **no peak term** | 3.44e-3 | 1.56e-1 | **5.08e-3** | **6.19e-3** | **2.14e-2** | 0.63% | 0.65% | **+0.94** | **0.008** |
| **both** (overfit) | **8.18e-4** | 4.22e-3 | 7.75e-3 | 8.14e-3 | 3.52e-2 | 0.61% | 0.70% | **+0.94** | 0.012 |
| predict the bank mean | — | — | — | — | — | — | 2.26% | 0.00 | — |

## The verdict: not architectural

**The architecture reaches `K_t` to 0.39%, `R² = +0.98`, with the peak located
to 1.1% of `x_g`.** Nothing about the decoder, the hard-BC ansatz or the
capacity stands in the way of the concentration factor. The limit is entirely
in what the objective sees.

And it is not because the probe was told where to look. Removing the peak term
entirely — plain von Mises field supervision, no oracle knowledge of the peak
location — still gives `R² = +0.94`, `K_t` to 0.65%, and a *better* peak
localisation (0.008 `x_g`) and a *better* von Mises field (5.08e-3) than the
version with the peak term, which spends weight on 32 points instead of the
field. The concentration falls out of fitting the stress field. It does not
have to be targeted.

## The sharper half: displacement accuracy does not buy stress accuracy

The displacement-supervised overfit reaches `u` = 1.36e-3 and `v` = 2.19e-3 —
a **0.2% displacement fit on the very specimens it is scored on**. No
generalisation gap, no conditioning gap, no capacity gap, no optimisation gap
is left to blame. Its concentration factor is **3.15%**, *worse* than
predicting a constant (2.26%), at `R² = −0.55`.

Stress is a derivative of the output. A field can be fit to a fraction of a
percent pointwise while its gradient over one small region is wrong, and the
concentration factor lives entirely in that gradient. This is the quantitative
version of that statement, and the literature review found nobody who had made
it: mDEM asserts it with field plots and no error tables; DCEM measures it but
is linear-elastic and its authors say the method does not extend to nonlinear
laws.

The generalising displacement-supervised arm makes the same point from the
other side. It is the **best transverse field this project has produced** —
`v` = 8.68e-3, 3.3× better than the physics objective — and simultaneously the
**worst concentration factor**, `R² = −2.74`. Improving displacement moved
`K_t` the wrong way.

## The trade-off is an artifact of single-objective supervision

Taken alone, the two objectives look irreconcilable: the stress-only arm's
transverse field is **80× worse** than the displacement-only arm's. That is
the shape DCEM reports in linear elasticity — one to two orders of magnitude
better stress, *worse* displacement than DEM — and it was tempting to read it
as a property of the problem.

**It is not.** Supervising on both gives `u` = 8.18e-4 — *better* than the
displacement-only arm — `v` = 4.22e-3, within **1.9×** of displacement-only
and **42× better** than stress-only, and a von Mises field (7.75e-3), fillet
error (8.14e-3) and `K_t` `R²` (+0.94) indistinguishable from stress-only. Its
section-force error, 0.2%, is the best of any arm in this project.

So the apparent tension was the objectives not being asked for both, not the
architecture being unable to deliver both. **A mixed formulation carrying both
a displacement and a stress residual should be able to have both**, and that
is a strictly better-specified target for Step 4 than the DCEM trade-off
suggested.

## What this does to the remaining plan

- **Step 4 (mixed `u`–`P`) is promoted from "the paper's centrepiece" to "the
  only step that addresses the measured cause".** The cause is that a
  displacement-only objective cannot see the peak; making stress a primary
  variable is the direct answer. It aims at a ceiling of `R² ≈ +0.95` from a
  current −0.25, and the bar it must clear is `R² = 0`.
- **Step 3 (adaptive quadrature) becomes supporting, not competing.**
  Concentrating points where the equilibrium residual is large helps an
  objective that already sees stress; on its own it cannot create a signal a
  displacement-only objective does not contain.
- **Step 2 (multi-scale radius) is unaffected** and still worth doing for the
  reviewer hole and the resampling spread, but it cannot touch this.
- **The ablation the literature review found missing is now half-built.** The
  missing experiment was mixed vs displacement-only on identical point sets
  and identical integration, reporting stress separately from displacement.
  That is exactly the disp/stress/both triplet above, under supervision. The
  physics-objective version is Step 4.

## Caveats, stated plainly

1. These are **overfits**. `R² = +0.98` bounds what a generalising model could
   reach; it is not a result a generalising model has reached. The gap that
   matters is −0.25 → +0.95, and it is large.
2. The oracle-peak-set concern is answered by the no-peak-term arm: `R²` falls
   only from +0.98 to +0.94, and the localisation improves.
3. **These are supervised arms.** They establish what the architecture can
   represent and what an objective that sees stress can extract. They do not
   establish that a *physics* objective built on stress will reach the same
   place — that is what Step 4 has to show, and it is the honest residual risk.
4. One seed per arm. Phase 6.2 measured 9–14% seed noise on these metrics at a
   500-epoch physics budget. The differences here are 3–8× larger than that,
   but the supervised arms' own seed noise is unmeasured.
