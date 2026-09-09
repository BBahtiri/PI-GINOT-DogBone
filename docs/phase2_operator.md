# The energy form as an operator

Everything in `docs/phase2_energy_form.md` was measured by fine-tuning on a
single specimen. That isolates the physics cleanly, and it is the right way to
ask *"is this formulation better?"* — but it is not the question the project
exists to answer. A neural **operator** has to carry a geometry family with
fixed capacity, and a formulation that wins with a whole network devoted to
one shape need not win when that capacity is shared across sixteen.

There is a specific reason to expect the two settings to disagree. The energy
form's advantage in Phase 2 was that traction-free conditions become natural
BCs — satisfied at the minimiser rather than fought for by weights. That
argument is per-geometry and does not weaken with a bank. But its cost profile
does change: the energy form carries a quadrature mesh per geometry, and under
batching those meshes are what dominate memory. So the single-geometry result
establishes the mechanism, not the operator-scale conclusion.

## Protocol

`verification/operator_study.py`. Three choices in the design are load-bearing:

**Train through the production trainer.** `PI_GINOT_Trainer`, not a bespoke
loop, so what is measured is the path that would actually run on a GPU —
including the batching, the collocation resampling, and the barrier schedule.
The wall-clock budget is spent in short `fit()` chunks; the optimiser and LR
scheduler are built in `__init__`, so chunking preserves Adam's moments and the
plateau state.

**Equal wall-clock, not equal epochs.** The energy form is 3–5× cheaper per
epoch. Equal epochs would understate it and equal wall-clock is the comparison
a practitioner actually faces: *given an hour, which formulation gives the
better operator?* Both the budget and the epochs each form reached are
reported, so either reading is available from the same run.

**"Held out" has to survive the trainer's own validation bank.** The trainer
draws its validation geometries from the same seed as the evaluation bank and
steps the LR scheduler on their loss. No gradient reaches them, but they have
influenced the model, so scoring on them would not be a held-out measurement.
The scoring set is taken from beyond that prefix and the study *refuses to run*
when the two overlap — a guard rather than a caveat, because this is exactly
the kind of thing that survives as a footnote and gets quoted without it. The
finite-element reference is solved once and shared between the forms, so
neither can be scored against a subtly different ground truth.

The five scoring geometries were chosen to span the taper ratio
`W_gauge/W_grip` — the parameter Phase 1 found the transverse error correlates
with (−0.843). **They do not.** They span 0.436–0.549, a narrow band around
the middle of the bank, because of a defect described under *Caveats* below:
the bank draw is not independent of the collocation settings, so the indices
selected for their taper under one setting are different specimens under
another. The result below is therefore a measurement at moderate taper, not
across the range.

## What Π can and cannot tell us during training

The reported `Π/(E|Ω|)` flattens early while the fields are still improving,
and that is expected rather than a stall: Π is dominated by the axial energy,
and Phase 2 measured the transverse component converging about 5× slower. A
flat energy is therefore consistent with a transverse field that is still
moving — which is the same trap that produced the conclusion Phase 2 had to
withdraw. The held-out field errors, not the training energy, are the signal.

## Result

Two arms, ~41 and ~43 minutes, 2 CPU cores. Held-out means over the five
geometries, against the finite-element reference:

| | energy | penalty | ratio |
|---|---|---|---|
| epochs reached | 1750 | 350 | 5.0× |
| s/epoch | 1.39 | 7.36 | 0.19× |
| u rel L2 | **1.94e-02** | 3.24e-02 | **0.60** |
| v rel L2 | 3.54e-01 | **3.13e-01** | **1.13** |
| von Mises rel L2 | **7.18e-02** | 1.57e-01 | **0.46** |
| von Mises, fillet | **1.01e-01** | 2.23e-01 | **0.45** |
| σ₂₂ / peak | **5.90e-02** | 6.14e-02 | 0.96 |
| section-force \|err\| | 3.48% | **2.47%** | **1.41** |

The energy form roughly halves the stress error, including at the fillet —
the stress concentration, which is the engineering quantity the whole model
exists to predict. That much transfers from the single-geometry study.

**Two things do not, and both were load-bearing assumptions.**

### The transverse advantage disappears

On one fine-tuned specimen the energy form beat the penalty form on `v` by
**33×**. Over a bank it is **13% worse**. And both forms sit at `v ≈ 0.31–0.35`
— about **12× worse than the 0.027** the same energy form reaches when a whole
network is devoted to one shape.

So the transverse field is not, at operator scale, a formulation problem.
Phase 1 diagnosed it as one and Phase 2 appeared to fix it; both conclusions
were drawn with the capacity of an entire network pointed at a single
geometry. What the operator sees instead is a capacity-and-convergence
problem that neither formulation touches.

One geometry dissents, and informatively. Geometry 11 reaches `v = 5.8e-02`
under the energy form — within striking distance of the single-geometry result
— while being the *worst* geometry for the penalty form (`v = 6.4e-01`, an 11×
swing in the opposite direction). The transverse field is not uniformly hard;
it is hard on most of the family and nearly solved on some of it.

### The force balance reverses

The energy form's section-force error is **worse** than the penalty form's
(3.48% vs 2.47%), and it is a **systematic bias, not scatter**: +4.00%,
+4.40%, +4.11%, +4.67% on four of the five geometries.

This is the direct consequence of the mechanism that makes the energy form
work. Traction-free conditions are *natural* BCs — satisfied **at the
minimum**. A network with capacity to spare reaches that minimum per geometry
and gets them for free, which is what Phase 2 measured (≤0.5% force error with
no anchor at all). A network sharing capacity across sixteen geometries stops
short of it, and a condition that is free at the minimum is merely approximate
anywhere else. A penalty term, by contrast, keeps pulling from wherever the
field happens to be.

Phase 2 disabled the section anchor under the energy form on the strength of
that ≤0.5%. **That justification does not transfer**, and the config comment
saying so has been corrected. So: restore the anchor under the energy form, and the force balance should
come back. **It does not.**

## The anchor does not fix it, and the reason is instructive

`verification/ablations.py`, four arms at 400 epochs each, everything but the
anchor weight held fixed:

| `w_resultant_energy` | u | v | vm | vm fillet | \|N err\| | s/epoch |
|---|---|---|---|---|---|---|
| 0 (control) | 2.58e-02 | **3.58e-01** | 8.43e-02 | 1.20e-01 | 4.24% | **1.46** |
| 1e-5 | 2.36e-02 | 3.93e-01 | 7.99e-02 | 1.12e-01 | 4.06% | 3.18 |
| 1e-4 | 2.29e-02 | 3.73e-01 | **7.83e-02** | **1.11e-01** | **3.70%** | 3.18 |
| 1e-3 | **2.17e-02** | 4.01e-01 | 9.42e-02 | 1.40e-01 | 4.53% | 3.23 |

At its best weight the anchor buys **0.5 percentage points** of force error —
4.24% to 3.70%, against the penalty form's 2.47% — for **2.2× the cost per
epoch** and a 4% loss on the transverse field. That is not a fix; the
hypothesis that the anchor would recover the force balance is wrong.

My first explanation was that **the anchor's target is itself biased the same
way as the error** — `resultant_anchor: "series"` runs ~2.5% high against the
reference (`physics/uniaxial.py`), so no weight on it could pull a +4% error
below +2.5%. Swapping in the calibrated target (0.74% mean error) at the
sweep's best weight tests that directly, and it **also does nothing**: 3.79%
against 3.70%.

### The sweep was not testing what I thought it was

Two refuted explanations in a row is a sign the model of the mechanism is
wrong, so: what does `w_resultant_energy` actually multiply?

`_compute_section_resultant_loss` returns

```
L_var + 2·L_adj  +  w_reaction·L_reaction  +  w_anchor·L_anchor
```

and only the last term knows the absolute force level; the first three are
*consistency* — that N(x) is the same at every slice and matches the grip
reaction. They do not differ by a little. The consistency terms start near
**0.4**. `L_anchor` is a squared relative error, so at a 2% miss it is
**3e-4**, carried behind `w_anchor = 0.5`. Against a reducible energy of
~1e-4:

| at `w_resultant_energy = 1e-4` | contribution | share of reducible Π |
|---|---|---|
| consistency terms | 4e-5 | ~40% |
| absolute anchor | 1.5e-8 | **~0.015%** |

The sweep varied the consistency pressure by four orders of magnitude and left
the absolute level untouched. Every row in that table is a real measurement of
the wrong lever — which is also why swapping the anchor's target changed
nothing: the term carrying the target was numerically irrelevant either way.
It even explains the non-monotonicity, since nothing in the sweep was pulling
the level in a consistent direction.

**A single scalar cannot fix this.** A weight large enough to make the level
matter (~0.1 on `L_anchor`) puts ~0.08 on the consistency terms, 800× the
reducible energy. The two need separate weights, which is now what they have:
`w_resultant_abs_energy` weights the anchor alone, and
`_compute_section_resultant_loss(split=True)` returns the halves.

### What it costs

The anchor more than doubles the energy form's per-epoch time (1.46 → 3.18 s),
which is most of its speed advantage over the penalty form (7.36 s). An anchor
that bought a real fix would still be worth it.

### Corrected sweep

`energy-abs-*` vary `w_resultant_abs_energy` over 0.03 / 0.3 / 3, chosen so the
anchor term lands at roughly 10%, 100% and 1000% of the reducible energy, and
against the calibrated target so the target's own 2.5% bias is out of the way.

*(running)*

## Caveats

**Undertrained, both of them.** 2 CPU cores and ~42 minutes; the penalty arm
reached 350 epochs against the 2500 the single-geometry study needed before
the transverse error stopped moving. Neither arm is converged, and the
transverse numbers in particular should be read as *"where each form is after
40 minutes"*, not as floors.

**Equal wall-clock is a choice, and it favours the energy form** by giving it
5× the epochs — but the advantage is not an artifact of the budget. The
anchor sweep's control arm is the same energy form at **400 epochs**, which is
within 15% of the penalty arm's 350:

| at ~350–400 epochs | energy | penalty | ratio |
|---|---|---|---|
| u rel L2 | **2.58e-02** | 3.24e-02 | 0.80 |
| v rel L2 | 3.58e-01 | **3.13e-01** | 1.14 |
| von Mises rel L2 | **8.43e-02** | 1.57e-01 | **0.54** |
| section-force \|err\| | 4.24% | **2.47%** | 1.72 |

Every sign is the same as the wall-clock comparison, and the stress advantage
is still ~1.9×. What the extra 1350 epochs buy is small: energy goes from
2.58e-02 to 1.94e-02 on `u` and 8.43e-02 to 7.18e-02 on `vm`, and **`v` does
not move at all** (3.58e-01 → 3.54e-01). The energy form is close to flat at
operator scale by 400 epochs. The penalty form's trajectory past 350 is
unmeasured, so this is not a claim that energy stays ahead at 1750 each.

**Moderate taper only** (0.436–0.549). The extremes, where Phase 1 found the
error largest, are not represented.

**A single seed, a single bank size, a single learning rate.** Nothing here
separates the formulations from their interaction with `lr = 3e-4`.

**The bank draw is not reproducible across collocation settings.** The
parameter draw, the mesh generation and the collocation sampling share one RNG
stream in `build_geometry_bank`, so changing `n_interior` changes which
specimens a seed produces. `geometry/banks.py` documents this and centralises
the draw, but centralising is not the same as fixing it: "validation geometry
11" in this document and in `docs/phase1_*.md` are **different shapes**. The
comparison *within* this run is unaffected — both arms and the reference share
one draw — but no geometry index should be carried between documents. Taper
ratios are reported alongside every index for exactly this reason.
