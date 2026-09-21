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
