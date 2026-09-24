# Hardening: does the Phase 4 result survive a fair test?

Phase 4's headline — replacing the point-cloud encoder with the four geometry
parameters — was measured on **five held-out geometries all at taper
0.44–0.55**, a narrow band in the middle of the range. Phase 1 had found the
error largest at the extremes. So the strongest claim in the project was being
made exactly where the problem is mildest, and it was not a choice: the
scoring geometries were *selected* to span the range and turned out not to,
for a reason worth fixing first.

## The bank draw was not reproducible

`build_geometry_bank` drew parameters, meshes and collocation points from **one
shared RNG stream**. So how much randomness the mesh and collocation happened
to consume determined the *parameters* of every subsequent geometry: changing
`n_interior`, `n_boundary_per_segment`, or passing `mesh_only` produced a
different set of specimens from the same seed.

Two consequences, and the second is worse than the first:

- "Validation geometry 11" names different shapes in different documents, so no
  geometry index in this project is comparable across phases.
- Indices chosen for their taper under one setting cluster in the middle under
  another — which is exactly how Phases 2–4 came to evaluate on 0.44–0.55 while
  believing they had picked a spread from 0.27 to 0.78.

**Fixed** by spawning independent sub-streams per geometry from the seed, so a
geometry's parameters depend only on `(seed, index)` and on nothing else.
`legacy_rng=True` reproduces the old draw for results measured under it, and
`test_bank_parameters_depend_only_on_seed_and_index` pins the invariance
against four different collocation settings.

Every number in Phases 0–4 refers to the legacy draw. The numbers below are the
first on the fixed one.

## The result holds, and it was not being flattered

Both arms retrained from scratch on the fixed bank, 64 training geometries, 400
epochs, energy form, scored against the finite-element reference on **eight**
held-out geometries spanning taper **0.31 – 0.82**:

| | `u` | `v` | vm | vm fillet | \|N err\| |
|---|---|---|---|---|---|
| point-cloud encoder | 1.431e-02 | 2.395e-01 | 6.101e-02 | 9.075e-02 | 2.30% |
| **four parameters** | **4.228e-03** | **7.221e-02** | **2.655e-02** | **4.086e-02** | **0.56%** |
| **ratio** | **3.4×** | **3.3×** | **2.3×** | **2.2×** | **4.1×** |

And per geometry, rather than on the mean:

| taper | `u` | `v` | vm |
|---|---|---|---|
| 0.31 | 3.2× | 4.7× | 2.0× |
| 0.37 | 5.0× | 2.3× | 2.5× |
| 0.47 | 2.3× | 3.7× | 2.5× |
| 0.54 | 3.4× | 2.0× | 3.0× |
| 0.61 | 3.5× | 3.7× | 2.3× |
| 0.68 | 3.3× | 2.7× | 2.2× |
| 0.77 | 3.9× | 4.7× | 2.4× |
| 0.82 | 3.3× | 2.0× | 1.8× |

**The advantage holds at every single geometry**, with no sign of collapse at
either extreme, and the transverse ratio (3.3×) is slightly *better* on the
full range than the 2.7× measured on the narrow band at the same budget. The
narrow band was, if anything, the harder test for this comparison.

## A premise this project had wrong

Phase 1 reported a correlation of **−0.843** with the taper ratio. Phase 3 and
`verification/ansatz_study.py` both quoted it as the correlation of the
**transverse** error — and built the local-height ansatz hypothesis on it. It
is not. It is the correlation of the **von Mises** error with taper
(`docs/phase1_case14_reference.md`); the transverse error was never measured
against taper at all.

Measured now, directly, across 0.31–0.82:

| | corr with taper |
|---|---|
| von Mises error | **−0.758** ← reproduces Phase 1's −0.843 |
| transverse error | **+0.108** ← essentially none |

So the von Mises/taper relationship is real and replicates on a different bank
draw, a different training configuration and a different set of geometries.
The transverse/taper relationship, which motivated a whole line of Phase 3
work, **does not exist**.

The Phase 3 conclusion is unaffected — the local-height ansatz was refuted by
direct measurement of the field it demands, not by the correlation — but the
motivation was weaker than stated, and a hypothesis that looked well-supported
was resting on a misquoted number. Both places now carry the correction.

## What this changes

Nothing in the recommendations, which is the point of running it. What it
changes is the confidence attached to them: the headline is now measured across
the full taper range on a bank whose indices mean something, rather than on a
band that happened to fall where the problem is mildest.

## Seed variance: the first error bars in the project

Every number in Phases 0–4 came from a single seed. Three seeds on both arms,
400 epochs, same fixed bank and same eight held-out geometries:

| | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| point-cloud `v` | 2.395e-01 | 2.234e-01 | 2.772e-01 | 2.467e-01 | 2.3e-02 |
| **four params `v`** | 7.221e-02 | 6.709e-02 | 8.361e-02 | 7.430e-02 | 6.9e-03 |
| point-cloud `u` | 1.431e-02 | 1.583e-02 | 1.395e-02 | 1.470e-02 | 8.2e-04 |
| **four params `u`** | 4.228e-03 | 3.946e-03 | 4.168e-03 | 4.114e-03 | 1.2e-04 |

Absolute errors move about ±9% with the seed. **The ratio does not:**

| | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| `u` | 3.38 | 4.01 | 3.35 | **3.58 ± 0.31** |
| `v` | 3.32 | 3.33 | 3.32 | **3.32 ± 0.01** |
| vm | 2.30 | 2.38 | 2.06 | **2.25 ± 0.14** |
| vm fillet | 2.22 | 2.32 | 2.04 | **2.19 ± 0.12** |
| \|N err\| | 2.30 → 0.56% | 2.68 → 0.31% | 1.76 → 0.68% | 2.25% → 0.52% |

The transverse ratio is **3.32 ± 0.01** — three significant figures of agreement
across seeds whose absolute errors differ by 24%. That is not luck: both arms
see the same geometries in the same order for a given seed, so the seed noise
is largely common-mode and divides out. It means the *comparison* is far better
determined than either arm on its own, and it is the comparison the project's
conclusion rests on.

## The flagship figure, on the bank the code now produces

The 1600-epoch parameter-conditioned run, repeated on the fixed bank and scored
across taper 0.31–0.82:

| | `u` | `v` | vm | vm fillet | \|N err\| |
|---|---|---|---|---|---|
| legacy bank, narrow band | 1.55e-03 | 2.20e-02 | 1.13e-02 | 1.60e-02 | 0.27% |
| **fixed bank, full range** | **2.28e-03** | **2.85e-02** | **1.38e-02** | **2.16e-02** | **0.56%** |

About 30% worse across the board, which is what including the hard geometries
should do — and `v = 2.85e-02` on held-out shapes spanning the full taper range
is still at the level Phase 2 reached by devoting an entire network to **one**
specimen (2.7e-02). **Quote the fixed-bank row.** The legacy figures are not
reproducible from the current code.

### Budget: the gap widens, but not for the reason first given

An earlier draft of this section claimed the point-cloud arm was
**budget-saturated**, citing Phase 2's bank-16 measurement (3.58e-01 at 400
epochs, 3.54e-01 at 1750 — 1% for 4.4× the compute) and concluding the
trained-out gap was near 8×. That was a number borrowed from a different bank
size, a different bank draw and a different evaluation set. Measured properly —
the point-cloud arm re-run at 1600 epochs on the fixed bank, same eight
geometries — **it is not saturated at all**:

| 400 → 1600 epochs | `u` | `v` | vm | \|N err\| |
|---|---|---|---|---|
| point-cloud | 1.57× | **1.58×** | 1.67× | 2.74× |
| four parameters | 1.86× | **2.53×** | 1.93× | 0.99× |

Both improve; the parameter-conditioned arm improves faster. So the conclusion
survives and the mechanism does not:

| ratio, point-cloud / four params | `u` | `v` | vm | vm fillet |
|---|---|---|---|---|
| at 400 epochs | 3.38 | **3.32** | 2.30 | 2.22 |
| at 1600 epochs | 4.00 | **5.30** | 2.65 | 2.74 |

The gap does widen with budget — 3.3× → 5.3× on the transverse field — but
because one arm improves faster, not because the other has stopped. And 5.3×,
not 8×.

Both arms at 1600 epochs, for the record:

| | `u` | `v` | vm | vm fillet | \|N err\| |
|---|---|---|---|---|---|
| point-cloud | 9.12e-03 | 1.51e-01 | 3.65e-02 | 5.91e-02 | 0.84% |
| **four parameters** | **2.28e-03** | **2.85e-02** | **1.38e-02** | **2.16e-02** | **0.56%** |

## The last unrun lever, closed cheaply

`boundary_measure_consistent` — allocating boundary points by arc length rather
than equally per segment — was the one Phase 0/1 lever never measured on the
operator. Measuring it the obvious way turns out to be both expensive and
beside the point: under the energy form the traction conditions are **natural
BCs**, so the boundary collocation this lever reallocates is not read by the
loss at all. A training arm would take three hours to measure a knob the
objective does not consult.

What it *does* still change is the point cloud fed to the **encoder** — which
Phase 4 identified as the bottleneck. That question needs no training. With the
encoder init pinned so the two configurations differ only in the sampler:

| | Wg/Wgrip | R/L | R/Wgrip | L/Wgrip |
|---|---|---|---|---|
| default (equal per segment) | 0.862 | 0.203 | 0.095 | 0.177 |
| measure-consistent (by arc length) | 0.821 | 0.268 | −0.030 | 0.383 |

**No improvement in fillet representation.** The two fillet ratios move in
opposite directions — `R/L` up, `R/Wgrip` down — which is what noise looks
like at this sample size. Nothing here suggests the sampler is what stands
between the encoder and the fillet, which is consistent with Phase 4, where
quadrupling the sampling density also changed no field metric.

Pinning the init mattered: without it each invocation builds a differently
initialised encoder, and the init spread on `R/L` alone covers −0.03 to 0.16 —
wider than any effect being looked for. The first run of this comparison was
made that way and had to be discarded.

## The recurring failure mode

Three claims in this project have now failed the same way: a number measured
under one configuration, quoted as if it held under another.

- Phase 3 dismissed a scale-invariant optimiser on a null result measured while
  the encoder bottleneck hid the effect.
- Phase 3 and `ansatz_study.py` built the local-height hypothesis on a −0.843
  correlation that was von Mises, not transverse.
- This section claimed budget saturation from a bank-16 legacy-draw
  measurement that does not transfer to bank 64 on the fixed draw.

None of the three changed a final conclusion, which is luck rather than
method. The common shape is worth stating: **a measurement is evidence for the
configuration it was made in, and quoting it outside that configuration is a
hypothesis, not a citation.**

## Caveats

## Caveats

**Three seeds, not thirty**, and the standard deviations above are computed
from three samples. The `v` ratio's 0.01 spread should be read as "the seeds
tried agreed closely", not as a confidence interval.

**Both long arms are single-seed.** The 400-epoch comparison has three seeds;
the 1600-epoch one has one, so the 5.30× trained-out ratio carries no
uncertainty estimate while the 3.32× equal-budget one does.

**Eight held-out geometries** rather than five, but still eight. The
per-geometry table matters more than the mean at this sample size, which is why
it is given.

**The `boundary_measure_consistent` ablation remains unrun** — the last of the
four Phase 0/1 levers never measured on the operator.
