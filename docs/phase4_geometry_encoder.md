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
| energy form, point-cloud encoder | 2.88e-01 | |
| energy form, four parameters | 1.06e-01 | ← **2.7×**, the encoder bottleneck |
| supervised, four parameters | 1.07e-02 | ← **10×**, the objective's conditioning |

**Phase 3's finding was not wrong. It was masked.** The 183× curvature ratio is
real, and with the encoder bottleneck in place it was worth almost nothing —
which is exactly what Phase 3 measured, and why it concluded a scale-invariant
optimiser would buy at most ~8%. Remove the encoder bottleneck and the same
ratio becomes the **binding** constraint, worth 10× on the transverse field.

That conclusion in `docs/phase3_transverse_conditioning.md` — *"a
scale-invariant optimiser is worth at most ~8% on this quantity"* — was correct
for the architecture it was measured on and is now wrong. Two constraints were
stacked, the outer one hid the inner one, and no experiment conducted with the
point-cloud encoder in place could have separated them. The lesson generalises
past this project: a null result for intervention A measured under bottleneck B
is not a null result for A.

## Which geometry information goes missing

`verification/encoder_probe.py` fits a ridge probe from the encoder's latent to
the four normalised parameters on 600 fresh geometries, 150 held out. The
parameters are recoverable *in principle* — the point cloud is generated from
them — so the probe measures whether they are recoverable *in practice*.

R², where 1 is exact recovery and 0 is no better than predicting the mean:

| encoder | representation | `L_total` | `W_grip` | `W_gauge` | `R_fillet` |
|---|---|---|---|---|---|
| random | pooled | −0.017 | 0.164 | 0.543 | 0.126 |
| random | tokens | −0.031 | 0.120 | 0.302 | 0.054 |
| **trained** | **pooled** | −0.002 | 0.225 | **0.717** | **0.198** |
| **trained** | **tokens** | 0.001 | 0.222 | **0.690** | **0.028** |

Both representations are probed because the decoder reads both — the pooled
vector through FiLM, the 32 tokens through cross-attention — and information
could live in the token structure while being averaged out of the pool.

**The reading turns on a detail of the interface.** `L_total` and `W_grip` reach
the decoder *outside* the latent, as the `x_max` and `y_max` scalars used for
coordinate normalisation and the hard BCs. The encoder does not need to carry
them, and the probe confirms it does not. That leaves exactly two parameters
that can only arrive through the latent:

- `W_gauge`, which the trained encoder carries well (R² 0.72 pooled, 0.69 from
  tokens);
- **`R_fillet`, which it barely carries at all** — 0.20 pooled and **0.03** from
  the tokens, against 0.13 and 0.05 for an *untrained* encoder.

So the single piece of geometry that reaches the decoder through no channel at
all is the fillet radius. That is precisely the local detail Phase 3 named as
the suspect, arrived at independently: Phase 3 inferred it from the correlation
between transverse error and taper, this measures it in the representation.

Training does not rescue it. The trained encoder is better than a random one on
`W_gauge` (0.72 vs 0.54) and no better on `R_fillet` — so on the answer to
*"does it discard the information or fail to learn to use it?"*, the fillet
radius is discarded, or at least never extracted, and more training of the same
architecture will not change that.

**What the probe does not show.** R² from a ridge probe measures *linear*
decodability. `R_fillet` could be present in a nonlinear code that this cannot
read — though a representation the decoder's FiLM and attention must consume is
one where linear availability is the relevant question, and an unregularised
MLP probe on these sample sizes was worse than useless (see the module
docstring). This is evidence of linear absence, not proof of absence.

## Where Phase 4 stands

1. **Condition on the geometry parameters** where the specimen family is
   parametric. It is not a diagnostic convenience; it is a better model, worth
   4.7× on `u`, 2.7× on `v` and 2.2× on stress under physics training alone.
2. **Then the optimiser is worth pursuing** — preconditioned, natural-gradient
   or second-order. The 10× gap between physics and supervision at oracle
   conditioning is what it is competing for, and unlike the earlier attempt
   there is now headroom for it to win.
3. **The point-cloud encoder needs work if generality beyond the parametric
   family is wanted**, and the next measurement there is the one this document
   cannot make: whether it discards the information or fails to learn to
   extract it.

## Caveats

**The supervised table is a ceiling, not a recipe** — it needs a solve per
geometry. The physics-trained comparison above is the one that carries the
practical claim.

**400 epochs, one seed, 2 CPU cores.** The oracle-physics arm is not converged;
`v` was still falling when the budget ran out, so 1.06e-01 is an upper bound on
what that configuration reaches, and the 10× gap to supervision may be smaller
than it looks.

**One seed, one architecture for the oracle MLP.** No attempt was made to tune
it; it is a three-layer MLP sized to match the encoder's parameter count.

**The held-out set is five geometries at taper 0.44–0.55**, as everywhere in
this project, and the bank draw still depends on collocation settings, so
geometry indices are not comparable with earlier documents.
