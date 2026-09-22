# Phase 6.6 — mixed u–P by stress potentials

## Why this, and why this construction

Phase 6.4 put the stress-concentration failure in the objective, not the
architecture: supervised on stress, the network reaches `K_t` R² **+0.98**;
supervised on displacement to 0.2% on the specimens it is scored on, R²
**−0.55**. Phase 6.5 closed the cheap alternative: adding the equilibrium
residual to the energy collapsed the field to uniform strain, predicted `K_t`
1.000015–1.000036 on every geometry, because a uniform stress is an exact null
of `Div P = 0`.

So stress has to be a primary variable, and equilibrium has to be something the
representation **cannot violate** rather than something the optimiser is asked
to learn. Two stress potentials — one per row of the unsymmetric first Piola
stress — do exactly that. Full derivation in `physics/mixed.py`.

## What the construction guarantees (verified, Level 0)

`verification/mixed_verify.py`, float64, eight geometries, stress heads
re-initialised at full scale so nothing is tested on the base term alone:

| property | worst over 8 geometries |
|---|---|
| `Div P` in the interior | 1.4e-13 |
| `P·N` on the gauge top | 0 exactly |
| `P·N` on the fillet arc | 5.8e-16 |
| `P_12` at `Y = 0`, `P_21` at `X = 0` | 0 exactly |
| `φ₁(X, H) − φ₁(X, 0) + c₁` | 0 exactly |
| section force, CV over `X` | 3.4e-15 |
| base term vs the bar stress | 0 exactly |

**Section-force constancy is exact by construction** — the quantity Phase 0
found at 6.5% CV and the resultant anchor existed to enforce.

**The Phase 6.5 collapse is impossible**: a uniform `P` that is traction-free on
the arc must vanish, since the arc's normals span both directions, and zero
stress contradicts the imposed stretch through the constitutive tie.

## The pilot — protocol

Matched to the Phase 6.4 ceiling probes so the numbers sit in the same table:
oracle-conditioned, trained on the same eight geometries it is scored on,
identical architecture in both arms (the decoder is built mixed in both; the
energy arm never reads the stress heads), identical epochs, batch, learning
rate and cosine schedule. The arms differ in the loss alone.

| arm | loss |
|---|---|
| energy | total potential energy — displacement-only physics |
| mixed | constitutive gap `‖P_φ − P(F(u))‖²`, equilibrium and tractions exact |

Both are scored from the displacement through `compare_reference`, comparable
with every earlier arm. The mixed arm is also scored from its own primary
output, the potential-derived stress pushed forward to Cauchy. That push-forward
is new code, so the pilot checks it against `compare_one` on the
constitutive-law path every run: it agrees to 8.6e-08 in `K_t`.

## Prediction, stated before the run

- **energy**: `K_t` R² ≤ 0. Every displacement-only objective measured so far
  has failed to see the peak, and this one has no generalisation or
  conditioning excuse to hide behind — so it should land near the
  displacement-supervised overfit (−0.55).
- **mixed**: `K_t` R² **> 0**, the bar for the phase. If the construction does
  what it claims, it should approach the stress-supervised ceiling (+0.94
  without the peak term).
- **Diagnostic**: the constitutive gap should fall to a small fraction of
  `σ_nom`. If it stays large, the optimiser did not converge the
  least-squares functional, and the conditioning fallback (`w_energy > 0`) is
  the next thing to try — not a verdict on the representation.

**Falsification.** If the mixed arm's R² ≤ 0 with the constitutive gap small,
then a stress-primary representation with exact equilibrium is *not* sufficient
under this objective, and the supervised ceiling is not reachable by
representation alone. That would be a real negative result and would be
recorded as one.

## Results

**Both predictions were wrong.** One seed, 1500 epochs, the protocol above.

| arm (oracle, overfit) | stress from | `u` | `v` | `vm` | \|`K_t`\| | **`K_t` R²** | \|Δx\*\| |
|---|---|---|---|---|---|---|---|
| energy | `u` | **1.04e-3** | **2.25e-2** | **9.72e-3** | **2.09%** | **+0.26** | 0.032 |
| mixed | `u` | 5.76e-3 | 4.84e-1 | 4.38e-2 | 3.62% | **−0.68** | **0.019** |
| mixed | `φ` | — | — | 4.39e-2 | 4.31% | **−1.18** | — |
| predict the bank mean | — | — | — | — | 2.26% | 0.00 | — |

### The energy form clears the bar when nothing else is in the way

Predicted R² ≤ 0; measured **+0.26**. The energy form, overfit on the eight
scored geometries, beats a constant on the concentration factor.

That reframes the ceiling table. Under the same protocol:

| objective, oracle, overfit | what it matches | `K_t` R² |
|---|---|---|
| displacement supervision | values of `u` | −0.55 |
| **total potential energy** | **∫ Ψ(∇u) — gradients, weighted by stiffness** | **+0.26** |
| stress supervision | values of `σ_vm` | +0.94 / +0.98 |

The ordering is coherent. `K_t` is a gradient quantity. Matching *values* of
`u` cannot see it; the energy matches a functional of the *gradient*, so it
sees it partly; matching the stress directly sees it fully. Phase 6.4's
reading — "the objective must see stress" — was right in direction and too
strong in degree: the energy already sees some. And the generalising energy
arm's −0.25 is therefore partly generalisation and conditioning, not purely
objective blindness. Both energy arms under-predict the peak (6 of 8 here,
mean −1.97%), which is what a volume integral that smooths a small feature
should do.

### The mixed form fails, and my falsification criterion was too naive

Measured R² −0.68 from `u`, −1.18 from `φ`, with the constitutive gap small —
the exact condition I pre-registered as falsifying: *"R² ≤ 0 with the
constitutive gap small."*

But the criterion rested on an assumption the run refutes: that a small gap
means a solution close to the true one. That holds only for a well-conditioned
functional. Here:

- the gap fell **1150×** (1.03e-1 → 8.98e-5), to 1.1–1.5% of `σ_nom` on every
  geometry — the optimiser converged the functional;
- and `v` is **33–64% wrong on every geometry**, 21.5× worse than the energy
  arm.

**A 1.3% constitutive gap coexists with a 48% transverse error.** The
least-squares functional is flat in the transverse direction — the same
near-null-space pathology Phase 3 measured in the energy (183× curvature
ratio), and here evidently worse. The optimiser found a pair (`u`, `φ`) that
is self-consistent to 1% and wrong, sitting in that valley.

So the honest reading is not "stress-primary representations don't work". It
is narrower and more useful: **the pure constitutive-gap functional is too
weakly conditioned in `v` to be a standalone objective.** The representation's
guarantees all held — equilibrium, tractions and section force are exact by
construction and the run cannot have violated them — but a representation that
cannot violate equilibrium can still be driven to the wrong equilibrated field
by an objective that does not discriminate between fields.

### What the mixed arm does better

Peak **location**: 0.019 `x_g` against the energy arm's 0.032, 0.61×. The
stress-primary field puts the concentration in the right place more reliably;
it gets the amplitude wrong because the lateral stress state that sets the
amplitude depends on the `v` it cannot resolve.

## What this changes

1. **The pre-registered result stands as recorded:** the pure mixed form did
   not clear R² > 0 and did not beat the energy form. Not softened.
2. **The criterion is corrected for the future:** a residual or gap being small
   is not evidence of proximity to the solution unless the functional's
   conditioning in every field is known. The energy form's own `v` problem
   (Phase 3) was the warning, and it applied here too.
3. **Next, and pre-registered as the conditioning fallback before this run:**
   `w_energy > 0` — the energy governs `u` and its conditioning, the gap ties
   an exactly-equilibrated `φ` to it. That is a new experiment with its own
   prediction, stated below before it runs, not a rescue of this one.

One seed per arm. The energy arm's +0.26 against the mixed arm's −0.68 is a
large gap at 8 points, but the sampling variance of R² at n = 8 is itself
large and unmeasured here.

## Arm 3 — mixed + energy (the pre-registered conditioning fallback)

`L = L_c + w_E · Π/(E|Ω|)`, `w_E = 100`, which makes the two terms comparable at
initialisation. The energy is computed from the displacement gradients the gap
already has, on the same quadrature points — so no second forward pass, and the
two terms integrate over the same stratified sample rather than two different
ones. Same protocol, budget and seed as the pilot.

**Mechanism being tested.** The gap is flat in `v`, so near the solution its
`v`-gradient is negligible and the transverse direction is governed by the
energy term alone — for any `w_E > 0`. The energy gets `v` right on its own
(2.25e-2). The gap then ties an exactly-equilibrated `φ` to `P(F(u))` and adds a
stress-space pull the energy lacks.

**Prediction, stated before the run.**

- `v` recovers to energy quality: **≲ 5e-2**, against the pure mixed arm's 0.48.
- `K_t` from `u`: **≥ +0.26**, the energy arm's value. It may sit right at it if
  the energy dominates the displacement.
- `K_t` from `φ` — the test that matters: **R²(φ) > +0.26**. `φ` is an exactly
  equilibrated field fit to `P(F(u))`; projecting a stress field onto equilibrated
  fields is a known stress-recovery improvement in finite elements, so if `u` is
  now good, the projection should beat the raw constitutive stress.

**Falsification.** If R²(φ) ≤ +0.26, the exactly-equilibrated potential field
buys nothing over the energy form alone under this protocol, and the
stress-potential representation is not the route to the ceiling here. Recorded
as that if it happens.

### Result

**Split. The falsification criterion fired on the test it was written for, and
the arm improved the concentration factor anyway — through a different field
than the one predicted.**

| arm (oracle, overfit) | stress from | `u` | `v` | `vm` | \|`K_t`\| | **`K_t` R²** | \|Δx\*\| |
|---|---|---|---|---|---|---|---|
| energy | `u` | 1.04e-3 | **2.25e-2** | 9.72e-3 | 2.09% | +0.26 | 0.032 |
| mixed | `u` | 5.76e-3 | 4.84e-1 | 4.38e-2 | 3.62% | −0.68 | 0.019 |
| **mixed + energy** | **`u`** | **7.27e-4** | 4.15e-2 | **8.56e-3** | **1.70%** | **+0.63** | **0.019** |
| mixed + energy | `φ` | — | — | 9.58e-3 | 2.49% | +0.24 | — |
| predict the bank mean | — | — | — | — | 2.26% | 0.00 | — |

Against the three predictions:

| prediction | measured | |
|---|---|---|
| `v` ≲ 5e-2 | 4.15e-2 | **met** — 11.7× better than pure mixed |
| `K_t` R² from `u` ≥ +0.26 | **+0.63** | **met, and far beyond** |
| `K_t` R² from `φ` > +0.26 | +0.24 | **failed — falsification fires** |

### What fired, stated plainly

The hypothesis was that `φ` — an exactly-equilibrated field fit to `P(F(u))` —
would be a *better stress estimate* than the constitutive stress it is fit to,
the way equilibrated stress recovery improves finite-element stresses. It is
not: +0.24 against the energy arm's +0.26. The potential field itself buys
nothing as an output. Recorded as falsified.

The likely reason is the one thing the representation does not build in.
`P_φ Fᵀ` is not symmetric — angular momentum is left to the constitutive tie —
and here its defect is **3.7–7.3% of `σ_nom`** on every geometry. Pushed forward
to Cauchy, that asymmetry corrupts the shear. Flagged as a hypothesis, not
measured as a cause.

### What improved instead

The *displacement's* stress. With the gap added to the energy, `P(F(u))` goes
from R² +0.26 to **+0.63**, better on 6 of 8 geometries, with the largest gains
exactly where the energy form was worst — geometry 9 from 4.84% to 2.64%,
geometry 6 from 4.27% to 3.11%. The axial field improves too (`u` 1.4× better),
and the peak location improves from 0.032 to 0.019 `x_g`.

So the mechanism is not the one predicted. The gap term acts on `u` as a
**stress-space regulariser**: it pulls `P(F(u))` toward something an exactly
equilibrated, traction-free field can represent, which is a constraint the
energy alone does not impose. **The potentials are more useful as a
training-time scaffold than as the output** — a non-obvious finding, and a
cleaner claim than the one predicted.

For scale: this closes about **54%** of the gap between the energy form (+0.26)
and the stress-supervised ceiling (+0.94), without any reference data.

### How much to trust it

**One seed.** The mean `|K_t|` error falls from 2.09% to 1.70%, −19%, against a
seed CV of 11.6% measured for this metric in Phase 6.2 — about 1.6 of those.
Suggestive, not established. R² at n = 8 is noisier still, dominated by the two
worst geometries. The claim "+0.26 → +0.63" needs at least three seeds per arm
before it is written anywhere a reviewer will read it.

And this is the **representability** protocol — trained on the geometries it is
scored on. The operator claim needs the generalising version.

## Replication — three seeds per arm

Seeds 0, 1, 2; within a seed the two arms share the model initialisation and
the geometry sampling order. Same protocol and budget throughout.
`verification/mixed_seeds_report.py` reproduces the table.

| metric | energy s0 | s1 | s2 | m+e s0 | s1 | s2 | **energy** | **m+e** |
|---|---|---|---|---|---|---|---|---|
| **`K_t` R², from `u`** | 0.259 | 0.131 | 0.343 | 0.628 | 0.754 | 0.769 | **+0.24 ± 0.11** | **+0.72 ± 0.08** |
| mean \|`K_t`\| error | 2.10% | 2.12% | 1.93% | 1.70% | 1.25% | 1.17% | 2.05 ± 0.10% | **1.37 ± 0.29%** |
| `vm` rel L2 | 0.010 | 0.010 | 0.010 | 0.009 | 0.009 | 0.009 | 0.010 | 0.009 |
| **`v` rel L2** | 0.023 | 0.023 | 0.024 | 0.042 | 0.050 | 0.043 | **0.023** | **0.045** |
| \|Δx\*\| | 0.032 | 0.029 | 0.008 | 0.019 | 0.024 | 0.013 | 0.023 | 0.019 |
| `K_t` R², from `φ` | — | — | — | 0.238 | 0.189 | 0.116 | — | +0.18 |

### Verdict: the improvement replicates

- **`K_t` R² rises from +0.24 ± 0.11 to +0.72 ± 0.08 on all three seeds**, and the
  ranges do not overlap: the *worst* mixed + energy seed (0.63) is well above
  the *best* energy seed (0.34). Welch's t with the seed as the unit: t = 6.21,
  one-sided **p = 0.002**.
- Mean `|K_t|` error falls **33%**, 2.05% → 1.37%, p = 0.021 at the seed level.
- It closes **68%** of the gap between the energy form and the stress-supervised
  ceiling (+0.94), with no reference data. Seed 0 alone had suggested 54%; it
  was the least favourable of the three.

A paired test at the finest grain — 24 (geometry, seed) pairs — gives 16 of 24
better and Wilcoxon p = 0.025. That test treats the 8 geometries within one
trained model as independent, which they are not, so it overstates the
evidence; the seed-level test above is the one to quote.

### The cost, stated as plainly as the gain

**The transverse displacement roughly doubles its error**, 0.023 → 0.045 on all
three seeds, p = 0.006. The stress-space pull the gap puts on `u` is paid for in
`v`. The von Mises field and the peak location improve slightly; `v` gets
worse. Any claim for this method has to carry both halves: it is a better
predictor of the stress concentration and a worse predictor of lateral
contraction.

### The potentials as an output: replicated as not-better

`φ`-derived `K_t` R² is +0.24, +0.19, +0.12 — mean **+0.18**, below the energy arm
on average and below the displacement-derived stress of its own arm on every
seed. The Phase 6.17 reading holds: the potentials are useful as a training
scaffold, not as the output.

### What this now supports, and what it does not

**Supported:** on the representability protocol (oracle conditioning, trained
on the scored geometries), adding an exactly-equilibrated stress-potential field
to the energy objective raises the stress-concentration R² from +0.24 to +0.72
across three seeds, at the cost of a roughly doubled transverse-displacement
error.

**Not yet supported:** anything about the operator. These models were scored on
the geometries they trained on. The generalising run — trained on the bank,
scored held-out — is what turns this into a claim about PI-GINOT, and it has
not been run. Also untested: any `w_energy` other than 100, and whether an
angular-momentum term makes `φ` a good output in its own right.

## Correction — adaptive sampling and the second-order optimiser are back

Both were recorded as dropped. Both drops quoted a measurement outside the
configuration it was made in, which is this project's standing failure mode,
and neither holds for the mixed + energy objective that is now winning.

**Adaptive sampling** was dropped in 6.11 because the hybrid's residual was
satisfied *exactly* by the trivial uniform field, so putting more points where
the residual is large could not help. That is true of the hybrid and false
here: the stress-potential representation cannot be uniform (proven, verified
at Level 0), so there is no trivial null. Mixed + energy is also precisely the
"objective that already sees stress" that 6.8 said adaptive sampling would
help. Both of its terms are integrated uniformly, so the peak region, about 1%
of the area, carries about 1% of the objective. That is the under-weighting
adaptive quadrature exists to fix, and Liu/Cai/Ramani's peak-stress recovery
in an energy formulation is directly on point. Honest prior: modest — PACMANN
finds most of RAD's advantage gone against resampled-uniform points, which is
what this project uses. **Target: the 32% of the gap to the ceiling that
mixed + energy has not closed.** Must stay resampled on every call (Phase 2),
either importance-weighted to keep the objective unbiased or deliberately
peak-weighted; they are different experiments.

**The second-order optimiser** was bounded at 2.1× by the Phase 4
physics-vs-supervision gap. That gap was measured on `v`, for the energy
objective, at oracle conditioning. The new objective has a different and far
worse conditioning problem: the pure constitutive gap reached 1.3% of
`σ_nom` with `v` 48% wrong. And the gap is a least-squares functional, for
which Gauss–Newton is the textbook method in a way it never was for the
energy. **Target: the doubled `v` error, which may be a conditioning cost
rather than a trade-off.** Practical constraints from the Phase 6 plan still
apply: 573k parameters rules out a dense Gauss–Newton matrix (use matrix-free
CG or K-FAC), and resampling corrupts curvature estimates, so fix the sample
within a step and resample between steps. GPU work.

**Order is unchanged at the top:** the held-out test first. Tuning a method
before knowing whether it generalises would spend effort on something that
might not survive the test.
