# The transverse field is in the objective's null space

The transverse displacement `v` is the one quantity that survived every lever
Phases 1 and 2 applied:

| lever | tried in | effect on `v` |
|---|---|---|
| loss formulation (penalty → energy) | Phase 2 | 13% **worse** at operator scale |
| traction weight `w_trac_top` (2 → 100) | Phase 1 | 1.46× |
| training length (400 → 1750 epochs) | Phase 2 | 3.58e-01 → 3.54e-01 |
| training-set size (bank 16 → 64) | Phase 2 | 3.58e-01 → 2.88e-01 |

Four independent levers, and the largest effect is 1.24×. That pattern is not
what a loss-weighting problem or a data problem looks like.

## What it is not: the hard-BC ansatz

The decoder enforces the symmetry condition with `v = (y/H_grip)·φ_v`, and its
docstring claims this "uses the full [0,1] range". That is true only at the
grip: in the gauge `y` reaches `H_gauge`, so the prefactor is at most the taper
ratio — as low as 0.28 here. Since Phase 1 measured the transverse error
correlating with taper at **−0.843**, this looked like the answer, with an
obvious fix: use the *local* half-height, `v = (y/h(x))·φ_v`, so the prefactor
is O(1) everywhere.

There was even a derivation. Equilibrium makes the axial resultant `N` constant
along `x`, so `ε ∼ N/(2Eh(x))` and `v ∼ −ν ε y`, giving

```
current ansatz:       φ_v = v·H_grip/y  ∼  −νN·H_grip/(2E·h(x))    ∝ 1/h(x)
local-height ansatz:  φ_v = v·h(x)/y    ∼  −νN/(2E)                 constant
```

The current form demands a field whose dynamic range *is* the taper ratio; the
local form should demand a constant. `verification/ansatz_study.py` measures
that on the finite-element solution directly, with no training:

| geo | taper | 1/taper | range, current | range, local |
|---|---|---|---|---|
| 0 | 0.276 | 3.63 | 2.25 | 5.49 |
| 2 | 0.317 | 3.16 | 2.08 | 4.71 |
| 11 | 0.776 | 1.29 | 1.57 | 1.95 |
| 13 | 0.426 | 2.35 | 2.02 | 3.85 |
| 16 | 0.519 | 1.93 | 2.09 | 3.53 |
| 19 | 0.270 | 3.70 | 2.15 | 5.35 |
| **mean** | 0.431 | 2.67 | **2.02** | **4.15** |

**The fix is worse than the defect**, by 2×, and it correlates with `1/taper`
more strongly than the thing it was meant to fix (+0.982 vs +0.823). The
derivation's error is `ε ∼ N/(2Eh)`: the grip is not in a uniaxial state, and
the section diagnostics show `E11` running 18–42% below the uniform-section
baseline there. The strain falls off *more slowly* than `1/h`, so multiplying
by `h(x)` over-corrects.

Worth noting what the table also says: the current ansatz demands a field with
a dynamic range of about **2**. That is mild. A decoder that fits `u` is not
being defeated by a factor of two.

## What it is: the objective barely responds to `v`

`verification/energy_sensitivity.py` perturbs the finite-element solution by a
controlled relative L2 error in one component at a time, holding the Dirichlet
conditions exactly — `v → v(1+δ)` preserves `v(x,0)=0`, and `u → u + δ‖u‖φ/‖φ‖`
with `φ = ξ(1−ξ)` preserves both axial conditions — and reports `ΔΠ/Π`. Both
perturbations are admissible, so Π can only rise; the comparison is of
curvature.

| δ | 0.05 | 0.10 | 0.20 | 0.35 |
|---|---|---|---|---|
| `ΔΠ/Π`, axial | 2.6e-02 | 1.0e-01 | 4.2e-01 | 1.3e+00 |
| `ΔΠ/Π`, transverse | 1.4e-04 | 5.7e-04 | 2.3e-03 | 7.0e-03 |
| **ratio** | **183.0** | **183.2** | **183.6** | **184.2** |

**The same relative error costs 183× more energy in the axial component than in
the transverse one**, and the ratio is flat in δ and flat across geometries
(2.6% spread over taper 0.27–0.78) — it is a property of the operator, not of
the perturbation size or the specimen.

The consequence is blunt: **a 35% error in `v` costs 0.7% of Π.** The energy
form tracks Π to within 0.2% over 2500 epochs of single-geometry training. So
a transverse field that is wrong by a third sits at, or below, the objective's
own residual. Nothing minimising Π has any reason to fix it.

That explains all four null results at once. Formulation does not change which
directions the physics makes flat. Bank size fixes generalisation, not
conditioning. Training length fights a gradient 183× weaker and would need
~183× the budget. And `w_trac_top` — the one lever that *did* move it, 1.46× —
is the only one of the four that touches the conditioning at all.

A rough consistency check: at a stationary point of a quadratic, errors scale
as `1/√k`, so a curvature ratio of 183 predicts `e_v/e_u ≈ 13.5`. Measured at
bank 32 and 64: 27× and 22×. Same order, which is as much as a quadratic model
of a finite-strain problem deserves.

## What follows

The transverse failure is a **conditioning** problem, not a formulation,
weighting, capacity or data problem. Three things follow, in order of how
directly the evidence supports them:

1. **The energy form's central claim needs a qualifier.** "Traction-free
   conditions become natural BCs, satisfied at the minimum rather than fought
   for by weights" is true — and insufficient. A natural BC is only free if the
   optimiser can reach the minimum, and in a direction with 183× less curvature
   it does not. This is the same shape of error as Phase 2's anchor
   investigation, where "free at the minimum" also failed to mean "free".
2. **Adding the gauge-top traction term back to the energy form is the
   principled fix.** It is redundant at the exact minimiser, so it cannot bias
   the converged answer, but it is precisely the term that carries transverse
   information — Phase 1 identified it as the only thing setting lateral
   contraction. It re-weights the *approach* without changing the destination.
3. **A scale-invariant optimiser would address the cause rather than the
   symptom.** A 183× curvature ratio is what preconditioning and second-order
   methods exist for, and it would not need a hand-chosen weight.

## Caveats

**One perturbation family per component.** The transverse perturbation is a
uniform scaling — getting the magnitude of lateral contraction wrong, which is
the dominant observed failure — and the axial one is a single `ξ(1−ξ)` mode.
Other error shapes, particularly spatial redistribution of `v`, will have
different curvatures. The 183× is the ratio for these modes, not a bound over
all perturbations.

**Π is not the penalty form's objective.** The measurement is made on the
energy functional, so it explains the energy form's behaviour directly. The
penalty form's ill-conditioning in `v` is separately evidenced but by a
different route: Phase 1 found the gauge-top traction term is the only one
carrying transverse information and it shipped with the lowest weight in the
config.

**Coarse reference mesh** (`default_h × 2.5`), as elsewhere in this project.
The ratio's flatness across geometries and δ makes a discretisation artifact
unlikely, but it has not been checked against a refined mesh.
