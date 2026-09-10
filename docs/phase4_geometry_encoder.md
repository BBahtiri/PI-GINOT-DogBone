> **Note.** Every number on this page was measured on the **legacy bank draw**,
> which depended on collocation settings and has since been replaced. The
> conclusions all survive re-measurement on the fixed bank across the full
> taper range — see `docs/phase5_hardening.md` — but for the headline figures
> quote that document: the parameter-conditioned arm reaches `v = 2.85e-02` at
> 1600 epochs there, against the 2.20e-02 quoted below, and the equal-budget
> ratio is 3.32 ± 0.01 over three seeds.

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

## Making the fillet available does not help

If the encoder does not carry the fillet, the obvious remedy is to let it see
more of the boundary. Sweeping the sampling pipeline on an *untrained* encoder
— a fair screen here, since trained and random score the same on the fillet —
shows the pipeline is not the hard limit:

| `n_point` | `radius` | Wg/Wgrip | R/L | R/Wgrip |
|---|---|---|---|---|
| 32 (default) | 0.15 | 0.896 | 0.156 | 0.039 |
| **128** | 0.15 | 0.939 | **0.451** | 0.189 |
| 128 | 0.05 | 0.906 | 0.389 | **0.260** |

Quadrupling the farthest-point centroids nearly **triples** fillet decodability
in an untrained encoder, past what the *trained* encoder manages at 32 (0.300).
The information is reachable; the default sampling just does not reach it.

So train with it. Energy form, bank 64, 400 epochs, `n_point` 32 → 128,
everything else identical:

| | `u` | `v` | vm | vm fillet | \|N err\| |
|---|---|---|---|---|---|
| `n_point` = 32 | 1.287e-02 | 2.880e-01 | 5.916e-02 | 8.221e-02 | 2.40% |
| `n_point` = 128 | 1.297e-02 | 2.887e-01 | 5.826e-02 | 7.995e-02 | 2.38% |

**Identical — every quantity within 1%**, for 1.7× the cost per epoch. A
threefold gain in how available the fillet is buys exactly nothing.

## The synthesis: it fails to use it, and Phase 3 says why

That negative is what separates the two readings this document could not
separate earlier. The fillet is *not* simply destroyed by the sampling —
`n_point` = 128 makes it substantially more decodable — and the model still
does not use it. So the answer to *"discards or fails to learn to extract?"* is
**fails to learn to extract**, and Phase 3 supplies the reason:

- the objective is **183× less sensitive** to the transverse field than to the
  axial one, so there is almost no gradient pressure to learn *any* transverse
  refinement;
- learning to read the fillet out of a point cloud is a hard, indirect
  sub-problem whose only reward arrives through that 183×-attenuated channel;
- so the encoder learns the taper ratio, which the axial field rewards richly
  (R² 0.97), and stops.

This also explains why oracle conditioning works so well without contradicting
any of it. Handing the decoder the four parameters does not add information the
point cloud lacked — it removes the need to *learn the extraction*. The weak
transverse gradient then only has to fit geometry → field, not point cloud →
geometry → field. The bottleneck was never the information; it was that
learning to extract it had to be paid for out of the thinnest gradient in the
problem.

### The concrete proposal this implies

Supervise the extraction directly, with an auxiliary head predicting the
geometry parameters from the latent. It needs **no reference solution** — the
parameters are known whenever a training geometry is generated — and it gives
the encoder a strong, direct gradient for exactly the quantity the physics loss
cannot reward.

Worth noting: `PI_GINOT.LEGACY_PREFIXES` is `("geom_aux_head.",)`, kept "so
checkpoints trained before the head was removed still load". **The architecture
had precisely this head, and it was deleted.** On this evidence that removal
looks like the wrong call, and restoring it is the cheapest remaining
experiment in the project.

## Supervising the extraction: the head, restored properly

The proposal above was the cheapest remaining experiment, so it was run.
`geometry/descriptors.py` defines the four dimensionless descriptors as one
shared definition — the probe and the training target cannot drift apart — and
`PI_GINOT(aux_head=True)` predicts them from the pooled latent, weighted by
`w_geom_aux` (0.0 by default). The targets cost nothing: they are known
whenever a training geometry is generated, so this stays reference-free.

**A correction to what this document said about the original head.** It did
exist, but `w_geom_aux` had no entry in the config, so the trainer's
`config.get("w_geom_aux", 0.0)` was always zero — the head was never trained,
which is why Phase 0 removed it as dead code. And its four shape targets were
the *absolute* parameters normalised to their sampling ranges: had it ever been
switched on, it would have been training the encoder toward `L_total` in
millimetres, which its scale-normalised input cannot determine. The same
mistake this document's first probe made. "The architecture had this and it was
deleted" was too kind to it.

Energy form, bank 64, 400 epochs:

| arm | `u` | `v` | vm | \|N err\| |
|---|---|---|---|---|
| baseline (`n_point` 32, no aux) | 1.287e-02 | 2.880e-01 | **5.916e-02** | 2.40% |
| `n_point` = 128 alone | 1.297e-02 | 2.887e-01 | 5.826e-02 | 2.38% |
| **aux, w = 1e-3** | **1.142e-02** | **2.389e-01** | 6.097e-02 | **0.87%** |
| aux, w = 1e-1 | 1.357e-02 | 2.537e-01 | 6.987e-02 | 1.97% |
| aux, w = 10 | 1.361e-02 | 3.336e-01 | 8.157e-02 | 1.91% |
| `n_point` = 128 **and** aux | 1.196e-02 | 2.477e-01 | 5.958e-02 | 1.56% |
| *oracle conditioning* | *2.71e-03* | ***1.06e-01*** | *2.63e-02* | *1.40%* |

The auxiliary loss helps, and modestly: **1.21× on `v`** at the lightest weight,
against 2.7× for replacing the encoder. Stronger weights make it worse. The two
interventions are **not complementary** — availability plus gradient (2.48e-01)
is no better than gradient alone (2.39e-01).

One real and nearly free win: at `w = 1e-3` the section-force error drops from
2.40% to **0.87%**, a 2.7× improvement. The aux head earns its keep on the
force balance even though it does little for the transverse field.

### Why it fails, and what that rules out

Probing the aux-trained encoders explains the ceiling. Fillet decodability
(`R/L`, pooled):

| encoder | R/L |
|---|---|
| untrained, `n_point` 32 | 0.156 |
| physics-trained, `n_point` 32 | 0.300 |
| aux w = 1e-3 | 0.322 |
| **aux w = 10** | **0.254** |
| untrained, `n_point` 128 | 0.451 |

**Even at `w = 10`, where the auxiliary term dominates the objective outright,
the encoder does not learn to extract the fillet ratio** — it ends up *worse*
than physics training alone. Predicting four numbers from a 64-dimensional
pooled vector is not a hard regression; if the information were reachable
through this pipeline, a dominant loss would reach it. It does not.

So the reading in the previous section — *"it fails to use it, not to see
it"* — was too generous, and the correction is this. There are two claims to
separate:

- the fillet is *linearly available in a random encoder at `n_point` 128*
  (0.451) — true, and that is what motivated the previous section;
- the fillet is *extractable by this architecture under training* — false, at
  either sampling density, under physics gradients or a dominant direct
  supervision.

The first does not imply the second. A representation that a ridge probe can
read off 450 samples is not thereby one that FPS pooling and cross-attention
can learn to produce and propagate under gradient descent.

**What this rules out is the whole class of in-place repairs.** Sampling
resolution, direct supervision, and both together each buy ≤1.21× on the
transverse field, while replacing the conditioning path buys 2.7× at the same
budget and 13× when trained out. For this specimen family the answer is not to
fix the point-cloud encoder; it is not to use it.

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
