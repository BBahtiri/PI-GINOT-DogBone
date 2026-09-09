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
against the calibrated target — which is **0.44% from the FEM reference** on
these five geometries, so the target itself is not the problem.

| `w_resultant_abs_energy` | u | v | vm | \|N err\| |
|---|---|---|---|---|
| 0 (control) | **2.58e-02** | **3.58e-01** | **8.43e-02** | **4.25%** |
| 0.03 | 2.05e-02 | 4.24e-01 | 9.98e-02 | 4.62% |
| 0.3 | 3.91e-02 | 4.12e-01 | 1.71e-01 | 9.34% |
| 3 | 1.04e-01 | 6.53e-01 | 2.98e-01 | 5.03% |

**Weighting the anchor correctly makes the force error worse, monotonically,
up to the point where it destroys the field.** A term that directly penalises
the force error increases the force error. Something is wrong with the premise,
not the weight.

## What was actually wrong: it is a generalisation gap, not a bias

Measured against the calibrated closed-form target, on the **training**
geometries and the held-out ones:

| model | train \|err\| | held-out \|err\| |
|---|---|---|
| energy, no anchor | **0.39%** | 2.93% |
| energy + anchor 0.03 | 1.60% | 4.17% |
| energy + anchor 0.3 | 6.31% | 8.78% |
| penalty form | 3.33% | **1.75%** |

**The unanchored energy form already gets the force right to 0.39% on the
geometries it trained on.** Phase 2's ≤0.5% was not a single-geometry artifact
at all — it holds on every training geometry in the bank. The held-out 2.9% is
the operator failing to *carry* that accuracy to an unseen shape.

Which is why no anchor could ever have helped. An anchor is a training-time
penalty on a quantity that is already accurate to 0.39% at training time.
There is nothing left for it to reduce; all it can do is fight the energy
gradient, and the table above is what that looks like. Every hypothesis I
tested before this one — biased target, coarse quadrature, mismatched section
sets — was a search for a bias that does not exist.

The comparison with the penalty form is the substantive finding. The energy
form fits the training force **8.5× better** (0.39% vs 3.33%) and transfers it
**1.7× worse** (2.93% vs 1.75%). That is overfitting of the force functional,
and it has a plausible mechanism: the penalty form's residuals are *pointwise,
geometry-local* conditions that the network has to satisfy as a property of the
field, while Π is a **single scalar per geometry** that the encoder can drive
down by tuning the latent shape-by-shape. The first generalises by
construction; the second need not. This is a hypothesis, not a result — it
predicts that the gap should shrink with a larger training bank, which is the
test worth running next and is cheap relative to what has been spent here.

## Testing that hypothesis: bank size

If Π is being driven down shape-by-shape rather than as a field property, more
shapes should force the operator to find the property. Energy form, 400 epochs,
everything else identical, bank 16 → 32 → 64:

| training bank | u | v | vm | vm fillet | \|N err\| | force gap |
|---|---|---|---|---|---|---|
| 16 | 2.58e-02 | 3.58e-01 | 8.43e-02 | 1.20e-01 | 4.25% | +2.43 |
| 32 | **1.28e-02** | 3.51e-01 | 6.23e-02 | 8.50e-02 | **2.23%** | **+0.91** |
| 64 | 1.29e-02 | **2.88e-01** | **5.92e-02** | **8.22e-02** | 2.40% | +0.88 |
| *penalty @16* | *3.24e-02* | *3.13e-01* | *1.57e-01* | *2.23e-01* | *2.47%* | *−1.58* |

**The prediction holds.** Doubling the bank cuts the held-out force error from
4.25% to 2.23% and the train→held-out gap from 2.43 to 0.91 points; doubling
again changes neither. The regression against the penalty form is gone at bank
32 — and it was never a property of the formulation.

And the control says the effect belongs to the energy form specifically. The
penalty form at bank 32, same 400 epochs, is **unchanged on every metric**:

| penalty form | u | v | vm | vm fillet | \|N err\| | force gap |
|---|---|---|---|---|---|---|
| bank 16 | 3.24e-02 | 3.13e-01 | 1.57e-01 | 2.23e-01 | 2.47% | −1.58 |
| bank 32 | 3.38e-02 | 3.19e-01 | 1.60e-01 | 2.24e-01 | 2.46% | −1.29 |

Twice the training geometries buys the penalty form nothing, while the energy
form gains 2× on `u` and 1.9× on the force. So the two formulations do not
merely differ in accuracy at a fixed budget — **they differ in how they use
data.** At bank 32 the energy form is 2.6× better on von Mises, 2.6× better at
the fillet, 2.6× better on `u`, and level on the force; the only quantity where
it still loses is `v`.

The gap is also **insensitive to training length and sensitive to bank size**,
which is the signature that distinguishes the two explanations. The bank-16
energy model measured at 400 epochs and at 1750 epochs gives gaps of 2.43 and
2.54; four and a half times the training buys nothing, while twice the
geometries cuts it by two thirds. Underfitting would have shown the opposite.

Two things the bank does *not* fix. `u` and `vm` improve sharply from 16 to 32
and then stop, so bank 32 is where this bank saturates — the remaining stress
error is not a data-quantity problem. And `v` barely moves (3.58 → 3.51 →
2.88e-01): a 20% improvement at bank 64 against 2× on the force. **The
transverse field is the one failure that survives every lever tried in Phases
1 and 2** — formulation, weighting, training length, and now training-set size.

Three things follow for the plan:

1. **Do not add a force anchor to the energy form.** Not at any weight. It
   targets a quantity that is not wrong where the loss can see it.
2. **Train on at least 32 geometries.** The force regression that motivated
   the whole anchor investigation disappears there, and `u` and `vm` improve
   by 2× and 1.35× for no extra cost per epoch.
3. **The transverse field is the open problem**, and it is now isolated: it is
   the only quantity that has not responded to formulation, weighting, budget
   or bank size. That, not the force balance, is where Phase 3 should go.

## Caveats

**Undertrained, both of them.** 2 CPU cores and ~42 minutes; the penalty arm
reached 350 epochs against the 2500 the single-geometry study needed before
the transverse error stopped moving. Neither arm is converged, and the
transverse numbers in particular should be read as *"where each form is after
40 minutes"*, not as floors.

**Equal wall-clock is a choice, and it favours the energy form** by giving it
5× the epochs — but the advantage is not an artifact of the budget, as the
epoch-matched table above shows.

**Moderate taper only** (0.436–0.549). The extremes, where Phase 1 found the
error largest, are not represented.

**A single seed and a single learning rate.** Nothing here separates the
formulations from their interaction with `lr = 3e-4`.

**The bank sweep covers two sizes for the penalty form and three for the
energy form**, at one seed each. That the penalty form is flat from 16 to 32 is
one measurement, not a trend, and neither form was tested past 64.

**The train/held-out force table uses 8 training and 5 held-out geometries**
against the closed-form target rather than FEM. The target is 0.44% from FEM
on the held-out set, which is fine for a 3% effect, but the penalty form's
*negative* gap (better held-out than train) is within what samples this small
and taper mixes this different can produce. The direction and size of the
energy form's gap are not.

**The bank draw is not reproducible across collocation settings.** The
parameter draw, the mesh generation and the collocation sampling share one RNG
stream in `build_geometry_bank`, so changing `n_interior` changes which
specimens a seed produces. `geometry/banks.py` documents this and centralises
the draw, but centralising is not the same as fixing it: "validation geometry
11" in this document and in `docs/phase1_*.md` are **different shapes**. The
comparison *within* this run is unaffected — every arm and the reference share
one draw — but no geometry index should be carried between documents. Taper
ratios are reported alongside every index for exactly this reason.
