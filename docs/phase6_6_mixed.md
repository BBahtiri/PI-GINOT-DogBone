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
