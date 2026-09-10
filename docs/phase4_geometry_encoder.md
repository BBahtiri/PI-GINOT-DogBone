# The geometry encoder is the bottleneck

Phase 3 ended with a puzzle and a hypothesis. The puzzle: held-out transverse
error sits at ~0.3 no matter what — four training-side levers and supervision
with ground truth all land within 15% of each other, while the same
architecture fits `v` to **0.4%** on geometries it has seen. The hypothesis:
the axial field is pinned by global constraints the encoder does carry, while
the transverse field depends on local fillet detail that farthest-point
sampling and ball-query subsampling are free to discard.

## The experiment

The specimen is **fully determined by four numbers** — `L_total`, `W_grip`,
`W_gauge`, `R_fillet`. So the conditioning path can be short-circuited: replace
the point-cloud encoder with an MLP over those four parameters, change nothing
else, and run the identical supervised protocol.

This is a clean discriminator because it removes any possibility of information
loss in the conditioning path while leaving the decoder, the hard-BC ansatz,
the loss, the optimiser and the budget untouched. Capacity is matched to within
2%: 573,378 trainable parameters against 584,591 (`OracleConditioned` in
`verification/supervised_ceiling.py` sizes the parameter MLP to ~207k against
the encoder's 220k).

## Result

Supervised, per-component relative loss, 4000 epochs, same held-out geometries:

| conditioning | bank | held-out `u` | held-out `v` | vm | vm fillet |
|---|---|---|---|---|---|
| point cloud | 16 | 2.13e-02 | 3.30e-01 | 8.75e-02 | 1.25e-01 |
| point cloud | 64 | 1.36e-02 | 3.04e-01 | 8.43e-02 | 1.16e-01 |
| **four parameters** | 16 | 5.55e-03 | **3.54e-02** | 7.04e-02 | 8.81e-02 |
| **four parameters** | 64 | **2.30e-03** | **1.07e-02** | **5.21e-02** | **6.63e-02** |
| *energy form (physics)* | *64* | *1.29e-02* | *2.88e-01* | *5.92e-02* | *8.22e-02* |

At bank 64 the transverse error falls **28×**, from 3.04e-01 to 1.07e-02. The
axial error falls 5.9×. And `v = 1.07e-02` is **better than the 2.7e-02 that
Phase 2 reached by fine-tuning a whole network on a single specimen** — the
operator, once it can actually see the geometry, beats per-geometry training,
which is the entire promise of the approach and the first time this project has
observed it.

The scaling behaviour is the other half of the result:

| bank 16 → 64 | `v` improvement |
|---|---|
| point-cloud encoder | 1.08× |
| four parameters | **3.3×** |

**More geometries are worth almost nothing through the point-cloud encoder and
a great deal through the parameters.** That is what a conditioning bottleneck
looks like: the data was there and the model could not use it. It also retires
the Phase 2 reading that transverse accuracy is a data-quantity problem — it is
a data-*access* problem.

## What this does and does not establish

**Does:** the decoder, the hard-BC ansatz, and the capacity are all adequate to
carry the transverse field across the geometry family. Nothing downstream of
the latent needs fixing. Phase 3's 183× conditioning ratio, and the whole
loss-side investigation, were operating downstream of the real constraint.

**Does not:** separate *"the encoder discards the information"* from *"the
encoder could represent it but does not learn to extract it"*. The boundary
point cloud fully determines the four parameters — the shape is generated from
them — so nothing is missing in principle. Both readings are encoder problems
and both are addressed by working on the encoder, but they call for different
remedies: sampling resolution and receptive field for the first, inductive bias
and optimisation for the second. Distinguishing them is the next measurement,
not a conclusion available here.

**A practical consequence either way.** For a parametric specimen family the
four parameters are *known*, so conditioning on them is not a diagnostic
convenience — it is a legitimate and much better model. The point-cloud encoder
buys generality beyond the parametric family, and this measures the price
currently paid for that generality: 28× on the transverse field, 5.9× on the
axial one.

## It survives physics training — and reveals what was underneath

Supervision is not a recipe: it needs a finite-element solve per geometry. The
version that matters is oracle conditioning under the **energy loss**, no
ground truth anywhere. Bank 64, 400 epochs, same held-out set and reference:

| bank 64, energy form | `u` | `v` | vm | vm fillet | \|N err\| |
|---|---|---|---|---|---|
| point-cloud encoder | 1.29e-02 | 2.88e-01 | 5.92e-02 | 8.22e-02 | 2.40% |
| **four parameters** | **2.71e-03** | **1.06e-01** | **2.63e-02** | **3.69e-02** | **1.40%** |
| *ratio* | *4.7×* | *2.7×* | *2.2×* | *2.2×* | *1.7×* |

Every quantity improves, from a change to the conditioning path alone. The
axial error is now **2.7e-03** — within 1.2× of what supervision achieves — and
the von Mises error is **better than the supervised model's** (2.63e-02 against
5.21e-02), which is what one should expect: the energy form targets the
stress state directly, while a displacement-fitting loss only reaches it
through differentiation.

### The layers, in order

Put the physics and supervised numbers side by side and the structure of the
problem is finally visible:

| bank 64, held-out `v` | | |
|---|---|---|
| energy, point-cloud encoder, 400 ep | 2.88e-01 | |
| energy, four parameters, 400 ep | 1.06e-01 | ← 2.7×, the encoder bottleneck |
| energy, four parameters, 1600 ep | 2.20e-02 | ← **4.8×, just budget** |
| supervised, four parameters | 1.07e-02 | ← **2.1×**, what is left for the optimiser |

**Phase 3's finding was not wrong. It was masked — but by less than the
unconverged run suggested.** The 183× curvature ratio is real, and with the
encoder bottleneck in place it was worth almost nothing, which is what Phase 3
measured when it concluded a scale-invariant optimiser buys at most ~8%. Remove
the bottleneck and it does become visible — but most of what the 400-epoch
comparison attributed to conditioning was simply training budget, and the
residual gap to supervision is **2.1×**, not 10×.

Two corrections, then: one to Phase 3, and one to this document's own first
draft.

- Phase 3's *"an optimiser is worth at most ~8%"* was correct for the
  architecture it was measured on and understates the value now: two
  constraints were stacked and the outer hid the inner. **A null result for
  intervention A, measured under bottleneck B, is not a null result for A.**
- This document's first reading — *"the objective's conditioning is worth
  10×"* — came from a run that had not converged, and inflated the case for
  exactly the thing Phase 3 had dismissed. **A gap measured at a budget one arm
  has outgrown is not a gap.** The same class of error, in the opposite
  direction, four sections apart.

## Where Phase 4 stands

1. **Condition on the geometry parameters** where the specimen family is
   parametric. It is not a diagnostic convenience; it is a better model, worth
   4.7× on `u`, 2.7× on `v` and 2.2× on stress under physics training alone.
2. **The optimiser is worth something, but not much** — the physics-vs-
   supervision gap at oracle conditioning is 2.1×, against the 4.8× that
   another 1200 epochs bought for free. Budget first; preconditioning is a
   second-order concern in both senses, and the supervised arm is not itself
   demonstrably converged, so 2.1× is an upper bound on the prize.
3. **The point-cloud encoder needs work if generality beyond the parametric
   family is wanted**, and the next measurement there is the one this document
   cannot make: whether it discards the information or fails to learn to
   extract it.

## Caveats

**The supervised table is a ceiling, not a recipe** — it needs a solve per
geometry. The physics-trained comparison above is the one that carries the
practical claim.

**One seed, 2 CPU cores.** The 1600-epoch oracle-physics arm still had `v`
falling slowly at the end, so 2.20e-02 remains an upper bound; and the
supervised arm was run at a fixed 4000 epochs with no convergence check of its
own, so the 2.1× gap between them is bounded on one side only.

**One seed, one architecture for the oracle MLP.** No attempt was made to tune
it; it is a three-layer MLP sized to match the encoder's parameter count.

**The held-out set is five geometries at taper 0.44–0.55**, as everywhere in
this project, and the bank draw still depends on collocation settings, so
geometry indices are not comparable with earlier documents.
