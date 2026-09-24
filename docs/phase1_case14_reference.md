# Phase 1 case 1.4 — stress at the fillet, against a reference

Reproduce:

```
python -m verification.fem.verify_fem                       # the reference verifies itself
python -m eval.compare_reference --checkpoint checkpoints/best.pt --n-geometries 24
```

---

## The gap this closes

Every accuracy number the project reported was either a residual — which says
a field is self-consistent, not that it is right — or a comparison against a
closed form that only exists for a prismatic bar. Neither can say how wrong
the stress is at a fillet, which is where the operator is documented to be
weakest and where the paper's contribution is claimed.

`verification/fem/` solves the same boundary-value problem by an independent
route: Galerkin finite elements on a conforming graded triangulation,
Newton–Raphson on the assembled residual, boundary conditions by elimination —
against pointwise collocation of the strong form, gradient descent on a
penalty sum, and hard substitution plus penalties.

The constitutive law is shared deliberately. It is already verified to machine
precision against the Lambert-W closed form and a sympy manufactured solution,
so re-deriving it would add risk, not confidence. What is being tested here is
the **solution**.

## The reference verifies itself first

| check | result |
|---|---|
| patch test: residual at the exact affine field | 1.5e-16 … 4.1e-17 |
| patch test: max ‖F − F̄‖ | 1.2e-14 |
| patch test: stress spread / error vs P(F̄) | 3.2e-14 / 1.7e-13 |
| patch test: recovery from a perturbed interior | 2.7e-02 → 2.5e-11, 4 iters |
| uniaxial: P₁₁ vs closed form | 27.438277 vs 27.440537 |
| uniaxial: Cauchy σ₁₁ and von Mises | 8.3e-05 / 7.1e-05 |
| dog-bone: N over a 14× element-count range | 288.25 … 288.41 (0.02%) |
| dog-bone: peak von Mises, Richardson h→0 | 36.04, finest mesh 0.16% |
| section-force CV on the 24 val geometries | 1.1e-04 |

Newton converges quadratically: 7.3e-04 → 1.0e-05 → 5.3e-09 → 2.2e-15.

Two solver details were not optional, and both are recorded because they would
bite anyone rebuilding this:

- Seeding `u = 0` with the grip displacement applied only on the loaded edge
  is a discontinuous state — elements touching the grip see F₁₁ ≈ 4 at the
  refined fillet spacing — and the first Newton step inverts elements. Seeding
  with the affine field fixes it.
- A full Newton step can still drive J₂D negative, outside the domain of
  `ln J`. The step is halved until the residual drops *and* every determinant
  stays positive.

**A meshing lesson worth carrying into Phase 4.** Sizing elements on the
specimen length alone left 3.7 elements across the half-gauge on the narrowest
geometries, and those solves were silently wrong — the section force varied by
up to 28% along x where a converged solve holds it to 1e-4. `default_h` sizes
on the *thinner* of length and gauge height. Two of 128 training-bank
geometries were affected, and the failure was only visible because the
section-force constancy was checked.

---

## Result: the operator learns the axial field and not the transverse one

24 validation geometries, `checkpoints/best.pt` (epoch 1424, trained with the
pre-Phase-0 code).

| metric | mean over the bank |
|---|---|
| u displacement, relative L2 | **2.42e-02** |
| v displacement, relative L2 | **1.68e+00** |
| von Mises relative L2, whole domain | 2.62e-01 |
| von Mises relative L2, gauge | 1.73e-01 |
| von Mises relative L2, fillet | 3.63e-01 |
| von Mises relative L2, band within 0.5R of the arc | 2.75e-01 |
| **peak von Mises error** | **+3.3%** |
| section force, mean \|error\| | 7.0% |
| section force, within-geometry CV (reference: 1.1e-04) | 6.5% |

Component RMS error, normalised by the reference von Mises peak:

| component | all | gauge | fillet | band |
|---|---|---|---|---|
| σ₁₁ | 1.08e-01 | 5.91e-02 | 1.45e-01 | 1.18e-01 |
| **σ₂₂** | **2.98e-01** | 2.79e-01 | 3.16e-01 | 3.05e-01 |
| σ₁₂ | 4.02e-02 | 1.99e-02 | 5.40e-02 | 4.38e-02 |
| von Mises | 1.96e-01 | 1.47e-01 | 2.36e-01 | 2.02e-01 |

### What the numbers say

**The axial response is learned; the transverse response is not.** The axial
displacement is accurate to 2.4%, in line with the preprint's 2.1–7.1%. The
transverse displacement has a relative L2 error of 168% — worse than
predicting zero, which would score 100%.

Direct evidence, on validation geometry 0 at the gauge top:

| x | v predicted | v reference |
|---|---|---|
| 0.99 | −0.12342 | −0.02544 |
| 8.10 | −0.12348 | −0.02544 |
| 15.21 | −0.12367 | −0.02537 |
| 18.77 | −0.12365 | −0.02275 |

Nearly 5× too much lateral contraction — an effective Poisson ratio near 1 —
and essentially **constant along x**, where the reference varies. The hard
boundary condition is `v = (y/H_grip)·φ_v`, so a constant v means the network's
transverse correction has barely learned any spatial structure at all.

σ₂₂ follows directly: it is the largest single error at 30% of the peak
stress, on a gauge whose lateral faces are traction-free and where the
reference has σ₂₂ ≈ 0.

**The reported metrics were the ones that hide it.** Peak von Mises is off by
+3.3%, comfortably inside the preprint's 0.9–13.3%, while the von Mises
*field* is 26% wrong in L2 and 36% wrong in the fillet. A single extremal
value cannot detect a field-wide error in a component that is small where the
peak is.

**The error is driven by taper, not by fillet length.** Correlation of the
von Mises L2 error with each parameter:

| | von Mises L2 | \|N error\| |
|---|---|---|
| W_gauge/W_grip | **−0.843** | **−0.708** |
| W_gauge | −0.815 | −0.790 |
| W_grip | +0.333 | +0.118 |
| L_total | +0.195 | +0.365 |
| R_fillet | −0.111 | −0.104 |
| gauge fraction x_g/L_half | −0.229 | −0.003 |

| | width ratio | von Mises L2 | \|N error\| |
|---|---|---|---|
| worst 5 | 0.27–0.43 | 33–70% | 8.6–23.9% |
| best 5 | 0.61–0.78 | 6.2–13.6% | 0.8–2.5% |

The more the specimen tapers, the worse the operator does — and the gauge
fraction, the obvious candidate, explains almost nothing. The reading is that
on a strongly tapered specimen the operator defaults towards the uniform-stretch
baseline its hard boundary conditions already encode, and the correction it
has to learn on top is exactly what it fails to learn. Bank-averaged numbers
look acceptable because the mildly tapered geometries dominate.

---

## Caveats

- This is the **shipped checkpoint**, trained before Phase 0. The force anchor
  was biased, the barrier guarded the wrong determinant, and the barrier
  warmup never ran. Retraining on the branch may change these numbers, and
  should be the next thing done.
- The reference is piecewise-constant stress on constant-strain triangles,
  first-order in stress. Its own peak carries a 0.16% discretisation error, so
  the +3.3% peak agreement is well outside the reference's uncertainty and is
  a real number, but a claim tighter than ~0.5% on a peak would not be.
- All 24 geometries are inside the training parameter ranges. Nothing here
  speaks to extrapolation.
