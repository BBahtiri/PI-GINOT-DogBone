# Phase 2 — the energy form

Reproduce:

```
python -m verification.transverse_study --geos 0,1,11 \
       --runs shipped,wtop=100,energy,energy-fixed --epochs 300
```

---

## The formulation

The total potential energy of a hyperelastic body is

```
Π = ∫_Ω Ψ(F) dV − ∫_Γt t̄·u dS
```

In this quarter model **every** boundary is one of: hard Dirichlet
(`right_grip` and the two symmetry planes, all baked into the decoder),
traction-free (`gauge_top`, `right_arc`), or a symmetry plane. There is no
applied traction anywhere, so the second integral vanishes identically and

```
Π = ∫_Ω Ψ(F) dV
```

is the whole loss. Minimising it subject to the hard Dirichlet conditions
gives `Div P = 0` in the interior **and** `P·N = 0` on the free boundaries as
*natural* boundary conditions — satisfied at the minimum rather than fought
for by penalty weights. One term replaces `L_eq`, `L_trac_top`, `L_trac_arc`,
`L_trac_hole` and `L_part`.

That is not a general aesthetic argument, it addresses a measured failure.
`docs/phase1_transverse_study.md` showed the gauge-top traction term is the
only thing setting lateral contraction, that it carried the lowest weight in
the configuration, and that raising it cut the transverse error by up to 2.1×.
In the energy form that condition is not weighted at all.

Three further properties, all of which turned out to matter:

- **The loss is an error bound.** Any field satisfying the Dirichlet
  conditions has `Π ≥ Π_exact`, so `Π_model − Π_reference` is one-sided —
  unlike a residual, it cannot be small for the wrong reason.
- **Only first derivatives are needed.** `Ψ` depends on `∇u` alone, so the
  loss differentiates the network once rather than twice.
- **`∂Ψ_reduced/∂F = P` exactly.** Under plane stress the energy is the
  reduced one, with F33 eliminated by the closure; by the envelope theorem the
  `dF33/dF` path is multiplied by a residual the solver drives to ~1e-16.
  Verified at 1.3e-15 (plane stress) and 3.6e-16 (plane strain), so the energy
  and stress formulations cannot silently diverge.

---

## The failure that comes with it: quadrature collapse

**A fixed quadrature rule does not work**, and it fails spectacularly rather
than subtly. Minimising a finite sum is not minimising an integral: once the
field is near the true minimiser, the only way left to reduce the sum is to
exploit the fixed points, and a network with this much capacity does exactly
that. Validation geometry 0, cached degree-2 rule on 1233 triangles:

| epoch | Π/Π_ref | u rel L2 | v rel L2 | vm rel L2 |
|---|---|---|---|---|
| 100 | 0.992 | 5.9e-03 | 0.277 | 0.085 |
| 200 | 0.657 | 5.0e-02 | 0.379 | 1.032 |
| 300 | **0.007** | 2.3e-01 | 0.992 | **2.701** |

At epoch 100 the energy form already beats every penalty configuration tried.
By epoch 300 the *sum* is near zero while the *field* is three times worse
than where it started.

`Π < Π_ref` is impossible for a field meeting the Dirichlet conditions, so the
energy is its own collapse detector. It reproduces on every geometry tested
(Π/Π_ref of 0.007, 0.932 and 0.858 at tapers 0.28, 0.50 and 0.78).

**The plan for this phase specified caching the triangulation** *"so a bank
geometry is triangulated once, not once per epoch"*. The triangulation should
indeed be cached — it is. The quadrature **points** must not be.

### The fix

Keep the triangulation, resample inside it: one or more uniformly distributed
points per triangle each call, weighted by that triangle's exact area. This is
*stratified* Monte Carlo, and it answers both of the plan's objections to
plain Monte Carlo — it is unbiased by construction (uniform within each cell,
exact cell measures), and stratifying by an element already graded to where
the integrand varies keeps the variance far below the unstratified estimator.
There is no fixed point set left to exploit.

Integration accuracy at the default budget (1318 triangles, 3 points each):

| | relative difference |
|---|---|
| vs a 4× finer mesh | 4.6e-05 |
| vs a higher-order rule on the same mesh | 2.2e-05 |
| geometric coverage (chord–arc slivers) | 5.6e-05 |
| degree 1 (centroid, the plan's suggestion) | 1.9e-04 / 3.1e-04 |

All well below the physics signal. Degree 2 is the default; the geometric
coverage error is comparable to the quadrature error at these mesh sizes,
which is worth knowing before spending effort on a higher-order rule.

---

## Result: the energy form transforms the accuracy

Same protocol as Phase 1 — single-geometry training, 300 epochs, measured
against the verified FEM reference. Three geometries spanning the taper range.

| geo | taper | run | u rel L2 | v rel L2 | vm rel L2 | σ₂₂/peak | N err | Π/Π_ref−1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.28 | shipped | 5.33e-02 | 8.93e-01 | 3.20e-01 | 1.06e-01 | +10.2% | +13.3% |
| 0 | 0.28 | wtop=100 | 5.47e-02 | 3.93e-01 | 3.09e-01 | 4.51e-02 | +13.2% | +11.9% |
| 0 | 0.28 | **energy** | **4.26e-03** | **2.43e-01** | **3.93e-02** | **2.83e-02** | **+0.5%** | **+0.4%** |
| 0 | 0.28 | energy-fixed | 2.29e-01 | 9.92e-01 | 2.70e+00 | 2.04e-01 | +22.9% | −99.3% |
| 1 | 0.50 | shipped | 5.80e-02 | 2.97e-01 | 2.18e-01 | 6.01e-02 | +7.1% | +6.7% |
| 1 | 0.50 | wtop=100 | 5.80e-02 | 2.02e-01 | 2.13e-01 | 4.44e-02 | +8.2% | +6.4% |
| 1 | 0.50 | **energy** | **2.55e-03** | **7.34e-02** | **2.43e-02** | **1.78e-02** | **−0.2%** | **+0.1%** |
| 1 | 0.50 | energy-fixed | 3.15e-02 | 2.16e-01 | 4.34e-01 | 8.10e-02 | +8.7% | −6.8% |
| 11 | 0.78 | shipped | 1.78e-02 | 1.49e-01 | 7.31e-02 | 2.49e-02 | +0.8% | +0.9% |
| 11 | 0.78 | wtop=100 | 1.78e-02 | 1.42e-01 | 7.26e-02 | 2.21e-02 | +1.0% | +0.8% |
| 11 | 0.78 | **energy** | **2.04e-03** | 1.50e-01 | **2.18e-02** | **1.76e-02** | **+0.3%** | **+0.1%** |
| 11 | 0.78 | energy-fixed | 5.03e-02 | 2.03e-01 | 7.91e-01 | 1.16e-01 | −15.2% | −14.2% |

Ratio of energy error to penalty error, mean over the three geometries:

| | vs shipped | vs wtop=100 |
|---|---|---|
| u | **0.080** (12.5×) | 0.079 |
| von Mises | **0.178** (5.6×) | 0.181 |
| σ₂₂ | 0.423 | 0.609 |
| v | 0.508 | 0.679 |

And it is **3.4–5.1× cheaper per epoch** (93–117 s per 300 epochs against
395–496 s), because the loss needs one differentiation of the network instead
of two.

### Reading it

**The axial field and the stress transform.** u improves 8.7–22.7× and von
Mises 3.3–8.9× on every geometry. The section-force error drops from
0.8–13.2% to at most 0.5%, and Π sits within 0.4% of the reference — the model
is essentially *at* the energy minimum.

**The transverse improvement is real, and at 300 epochs it looks bounded** —
v improves 3.7–4.0× on the two tapered geometries but not at all on the
mildest one (0.150 against 0.149). That reading was wrong; see the next
section.

**Raising `w_trac_top` is superseded.** It was the cheapest fix available
within the penalty formulation; the energy form is better on every metric and
cheaper. The reweighting row stays in the ablation matrix as the penalty-form
control, not as a recommendation.

---

## Correction: the transverse field is slow, not stuck

The 300-epoch table above says the transverse error plateaus, and the first
reading of it — recorded here and in the project notes — was that the residual
error is architectural, and that the hard-BC ansatz `v = (y/H_grip)·φ_v` was
the thing to attack next.

**That was wrong.** Two checks refuted it.

*First*, normalising the transverse error by a common displacement scale
rather than by ‖v_ref‖ (which is small — ‖v‖/‖u‖ is 0.03–0.06) shows the
energy form does reduce it in absolute terms on the tapered geometries, and
that it had become the *dominant* displacement error: 6.7e-03 against an axial
2.0e-03 on geometry 11.

*Second*, and decisively: the 300-epoch runs were simply not converged. The
transverse field converges roughly 5× slower than the axial one, so a budget
tuned to the axial field stops while the transverse field is still moving.
Running to 2500 epochs, on the same two geometries:

| geo | epoch | u rel L2 | **v rel L2** | vm rel L2 | σ₂₂/peak | N err | Π/Π_ref−1 |
|---|---|---|---|---|---|---|---|
| 0 (taper 0.28) | 250 | 3.26e-03 | 2.59e-01 | 3.78e-02 | 2.93e-02 | −1.4% | +0.3% |
| | 500 | 1.14e-03 | 5.96e-02 | 2.86e-02 | 1.70e-02 | +1.2% | +0.1% |
| | 1000 | 5.13e-04 | 3.01e-02 | 1.42e-02 | 8.66e-03 | +0.5% | +0.0% |
| | 2000 | 5.61e-04 | 1.46e-02 | 1.07e-02 | 5.63e-03 | +0.1% | −0.0% |
| | **2500** | **9.65e-04** | **2.70e-02** | **1.38e-02** | **5.44e-03** | **+0.5%** | **+0.2%** |
| 11 (taper 0.78) | 250 | 1.96e-03 | 1.51e-01 | 2.20e-02 | 1.77e-02 | +0.2% | +0.1% |
| | 500 | 1.25e-03 | 9.12e-02 | 1.57e-02 | 1.32e-02 | +0.1% | +0.0% |
| | 1000 | 1.16e-03 | 2.06e-02 | 8.27e-03 | 6.39e-03 | +0.0% | +0.0% |
| | **2500** | **6.10e-04** | **1.07e-02** | **5.97e-03** | **3.70e-03** | **−0.2%** | **−0.0%** |

Geometry 11, the one that appeared stuck at v = 0.150, reaches 0.011 — a
**14× improvement** that 300 epochs could not see. Against the penalty form on
the worst geometry (values from the table above, 300 epochs):

| | shipped | wtop=100 | energy @2500 | improvement |
|---|---|---|---|---|
| u | 5.33e-02 | 5.47e-02 | **9.65e-04** | **55×** |
| v | 8.93e-01 | 3.93e-01 | **2.70e-02** | **33×** (15× vs wtop) |
| von Mises | 3.20e-01 | 3.09e-01 | **1.38e-02** | **23×** |
| σ₂₂ | 1.06e-01 | 4.51e-02 | **5.44e-03** | **19×** |
| N error | +10.2% | +13.2% | **+0.5%** | |

Π/Π_ref stays within 0.2% of the reference across all 2500 epochs, so the
stratified resampling holds — no drift towards collapse over a realistic
training length.

**What this changes.** The hard-BC ansatz is not the constraint; it was never
shown to be. What the evidence supports is that the transverse component needs
either more epochs than the axial one, or a preconditioner / second-order
refinement stage to close the gap sooner. That is a scheduling question, not
an architectural one, and it should be settled on the retrained operator
before any change to the decoder is contemplated.

The general lesson is worth stating plainly: **a plateau observed at a budget
tuned to the dominant field component is not evidence of a floor.** Both
earlier conclusions — Phase 1's "raise w_trac_top" and this document's first
reading — came from stopping at 200–300 epochs.

---

## What this does not establish

- **Single-geometry training, not operator training.** Every number here comes
  from fine-tuning on one specimen. Integrating the energy loss into the
  128-geometry trainer is the next step and is where the remaining risk sits:
  the quadrature mesh is per geometry, so memory and caching behaviour under
  batching are untested.
- **A diagnostic budget on CPU.** The 2500-epoch runs settle the plateau
  question but say nothing about the production schedule on a 128-geometry
  bank.
- **Three geometries.** The taper trend is consistent with Phase 1 but the
  bank-wide numbers need a retrained operator.
- The **mixed u–P form** (plan step 2.4) is not implemented. On this evidence
  its motivation has shifted: it was proposed to fix traction accuracy, which
  the energy form already fixes. Its remaining argument is direct stress
  output and first-order `Div P̃` — worth revisiting after the energy form is
  in the operator.
