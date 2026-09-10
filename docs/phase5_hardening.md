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

## Caveats

**Still one seed and 400 epochs**, on 2 CPU cores. The 3.3× on `v` is the
short-budget figure; at 1600 epochs on the legacy bank the same change was
worth 13×, and that longer run has not been repeated on the fixed bank.

**Eight held-out geometries** rather than five, but still eight. The
per-geometry table matters more than the mean at this sample size, which is why
it is given.

**The `boundary_measure_consistent` ablation remains unrun** — the last of the
four Phase 0/1 levers never measured on the operator.
