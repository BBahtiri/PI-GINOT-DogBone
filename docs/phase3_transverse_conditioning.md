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

> **Correction (Phase 5).** The −0.843 quoted below is Phase 1's correlation of
> the **von Mises** error with taper, not the transverse error — this document
> mis-attributed it, and the mis-attribution is what made the ansatz
> hypothesis look well-motivated. Measured directly across taper 0.31–0.82 on
> the fixed bank, the transverse error correlates with taper at **+0.108** —
> essentially not at all — while von Mises reproduces at −0.758. The ansatz
> hypothesis was refuted on its own terms anyway (below), but it was never as
> well-motivated as this section claims.

The decoder enforces the symmetry condition with `v = (y/H_grip)·φ_v`, and its
docstring claims this "uses the full [0,1] range". That is true only at the
grip: in the gauge `y` reaches `H_gauge`, so the prefactor is at most the taper
ratio — as low as 0.28 here. Phase 1's taper correlation of **−0.843** made this
look like the answer, with an
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

## The fix that follows from it does not work

Point 2 above — restore the gauge-top traction term to the energy form — is
the principled move: redundant at the minimiser, so it cannot bias the answer,
and it is the one term Phase 1 identified as carrying transverse information.
`EnergyLoss` gained `w_trac_top` (off by default) and the weight was swept over
**4.5 orders of magnitude**, bank 32, 400 epochs, everything else fixed:

| `energy_w_trac_top` | u | v | vm | vm fillet | \|N err\| |
|---|---|---|---|---|---|
| 0 (control) | **1.277e-02** | 3.507e-01 | **6.229e-02** | **8.501e-02** | 2.23% |
| 0.03 | 1.278e-02 | 3.459e-01 | 6.316e-02 | 8.652e-02 | 2.17% |
| 0.3 | 1.336e-02 | 3.830e-01 | 6.657e-02 | 9.290e-02 | 2.13% |
| 3 | 1.461e-02 | 3.933e-01 | 8.763e-02 | 1.290e-01 | **2.04%** |
| 10 | 3.387e-02 | **3.279e-01** | 1.639e-01 | 2.486e-01 | 4.90% |
| 100 | 5.054e-02 | 3.344e-01 | 1.973e-01 | 3.029e-01 | 8.27% |
| 1000 | 5.999e-02 | 3.562e-01 | 2.155e-01 | 3.272e-01 | 9.39% |

**`v` never leaves ±10% of the control, and not monotonically** — 0.346, 0.383,
0.393, 0.328, 0.334, 0.356 against 0.351. That spread is smaller than the
spread between geometries within a single arm. Meanwhile `u` degrades 4.7× and
von Mises 3.5× at the top of the range. The best `v` in the table, 0.328 at
w = 10, comes with `u` 2.7× worse and stress 2.6× worse: not a trade anyone
would take.

(One real if minor effect: the force error improves slightly at low weight,
2.23% → 2.04%, which is the traction term doing what it should — carrying
equilibrium information — just not about `v`.)

### Why adding the term does not add the curvature

I sized the first weights from a **value** ratio — `L_trac_top` is ~30% of the
reducible energy at w = 1 — and predicted the useful weight would be near the
curvature ratio, ~183. That was the wrong quantity: even w = 10 puts the
traction term at ~7× the reducible energy in the first epoch, so the entire
first sweep sat in the over-dominant regime and never tested the hypothesis.
The second sweep went below it, and found nothing there either.

The reason is visible in what degrades. The gauge-top traction residual is
`P·N` with `N = +y`, i.e. `P12` and `P22`, and those depend on gradients of
**both** components. It is not a transverse constraint that happens to be
weighted low — it is a mixed constraint, so raising its weight adds curvature
in `u` at least as fast as in `v`, and the axial error is what moves. Phase 1's
1.46× from the same lever under the penalty form was not a starting point to
build on; it was the whole effect available.

### What this rules out

**Loss-side reweighting cannot fix this.** That is now measured in both
formulations: `w_trac_top` under the penalty form (Phase 1, 1.46×) and the same
term restored to the energy form (here, nothing across 4.5 orders of
magnitude). A curvature ratio of 183 is a property of the physical operator,
and adding a penalty term does not precondition an operator — it replaces it
with a different, generally worse-conditioned one.

What remains is optimiser-side or supervision:

1. **A scale-invariant optimiser** — preconditioned, natural-gradient, or
   second-order. This addresses the cause rather than reweighting around it,
   and needs no hand-chosen constant. It is the direction the evidence
   actually supports.
2. **Supervise `v` directly** where a reference is available. The finite-element
   solver built in Phase 1 can generate one per bank geometry, and a hybrid
   physics+data objective would sidestep the conditioning entirely — at the
   cost of the reference-free property the whole approach was chosen for.

Both looked like Phase 4 questions at this point. The ceiling measured in the
next section shows neither is, which is why this section's conclusion is
narrower than it first appeared: what the sweep settles is that the transverse
field is not reachable by any weighting of the terms in either loss — not that
a better optimiser would reach it.

## The ceiling: conditioning is real but not the binding constraint

Both remaining routes — a scale-invariant optimiser, or supervision — are
expensive, and both assume the architecture could reach the transverse field if
only training let it. `verification/supervised_ceiling.py` tests that
assumption directly, and cheaply, by removing every obstacle at once: it trains
the same `PI_GINOT` on the finite-element `(u, v)` fields with a **per-component
relative** loss,

```
L = ||u_pred − u_ref||² / ||u_ref||²  +  ||v_pred − v_ref||² / ||v_ref||²
```

which gives the two components equal footing by construction — exactly the
preconditioning Π lacks — and hands the model the answer besides. It is not a
training recipe (it needs a solve per geometry, the very thing the approach
exists to avoid). It is an upper bound on what any amount of optimiser or loss
work could achieve.

| | train `u` | train `v` | held-out `u` | held-out `v` |
|---|---|---|---|---|
| supervised, bank 16 | 1.90e-03 | **4.18e-03** | 2.13e-02 | **3.30e-01** |
| supervised, bank 64 | 3.21e-03 | **6.62e-03** | 1.36e-02 | **3.04e-01** |
| *generalisation factor, bank 16* | | | *11.2×* | ***78.9×*** |
| *generalisation factor, bank 64* | | | *4.2×* | ***46.0×*** |
| energy form, bank 16 | | | 2.58e-02 | 3.58e-01 |
| energy form, bank 64 | | | 1.29e-02 | **2.88e-01** |

**The architecture fits `v` to 0.4% on geometries it has seen.** The ansatz, the
decoder capacity and the hard-BC construction are all entirely adequate — the
question `verification/ansatz_study.py` could only answer indirectly is now
answered directly.

**And it changes nothing about held-out `v`.** With conditioning removed by
construction and ground truth supplied, held-out `v` is 3.30e-01 against the
physics-trained 3.58e-01 — an 8% gap. At bank 64 the physics-trained model is
**better than the supervised one** (2.88e-01 vs 3.04e-01).

So the 183× curvature ratio is a true fact about Π, and it correctly explains
why none of the physics-side levers move `v`. But it is **not the binding
constraint**: repairing it perfectly is worth at most ~8%, because something
else caps the transverse field at ~0.3 regardless of how the model is trained.

That something is **geometry generalisation**, and the contrast with `u` is what
identifies it. Quadrupling the training set improves the axial generalisation
factor 11.2× → 4.2×, while the transverse factor only goes 78.9× → 46.0× and
the held-out error barely moves. Training error stays at 0.3–0.7% throughout,
so this is not capacity saturation. The encoder carries whatever determines the
axial field to an unseen shape, and does not carry whatever determines the
transverse one.

A plausible mechanism, offered as a hypothesis rather than a result: the axial
field is pinned by the prescribed elongation and the section areas — global,
low-frequency, strongly constrained — while the transverse field depends on the
local Poisson response through the fillet, which is precisely the fine
geometric detail the encoder's farthest-point-sampling and ball-query
subsampling is free to discard. It predicts that boundary-point resolution near
the fillet, not loss design, is what moves `v`.

## What follows — revised

The three recommendations above were written before the ceiling was measured,
and the ordering is now wrong. Corrected:

1. **Phase 4 belongs in the encoder**, not the loss and not the optimiser. The
   held-out transverse error is set by how the geometry is represented, and
   four independent training-side interventions — formulation, weighting,
   training length, bank size — plus supervision with ground truth all land
   within 15% of each other.
2. **A scale-invariant optimiser is worth at most ~8%** on this quantity, on
   this evidence. It remains the right answer to the conditioning ratio, and
   the conditioning ratio is no longer the interesting problem.
3. **Supervising `v` is not worth its cost.** It requires a solve per geometry,
   forfeits the reference-free property, and at bank 64 loses to physics
   training on the very quantity it was meant to rescue.

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

**The redundancy test is weak.** `test_energy_traction_terms_vanish_on_the_
exact_solution` evaluates the traction from element stresses *adjacent* to the
free surface, and constant-strain triangles give a piecewise-constant stress
whose boundary element carries a non-zero mean traction. The measured value is
therefore an upper bound on the true boundary residual and is looser than the
residual a trained model reaches. The test rules out a grossly non-redundant
term; it does not establish redundancy to the precision the argument would
like. Since the sweep found no weight worth using, this did not end up
mattering — but it would have, had a weight looked promising.

**One lever, one loss.** The sweep varies the gauge-top term only. The fillet
arc term (`w_trac_arc`) is implemented and was not swept, on the grounds that
Phase 1 identified the gauge top as the transverse carrier — which is an
inherited assumption, not something re-checked here.
