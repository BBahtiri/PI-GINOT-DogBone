# What is left to improve accuracy — a researched, ordered plan

Written after three parallel literature reviews (mixed/variational formulations,
adaptive sampling and optimisers, geometry conditioning) cross-referenced
against what this project has actually measured.

## Where the error is now

Best configuration, fixed bank, 64 training geometries, 1600 epochs, eight
held-out geometries spanning taper 0.31–0.82:

| | `u` | `v` | von Mises | **vm, fillet** | \|N err\| |
|---|---|---|---|---|---|
| parameter-conditioned | 2.28e-03 | 2.85e-02 | 1.38e-02 | **2.16e-02** | 0.56% |
| point-cloud | 9.12e-03 | 1.51e-01 | 3.65e-02 | 5.91e-02 | 0.84% |

**The axial field is solved. Stress at the fillet is now the error**, at 2.16e-02
— nine times the axial displacement error and 1.6× the global von Mises error.
That is also precisely the quantity the paper wants to claim, so the target and
the weak spot coincide.

Two things this changes about priorities. First, any lever that improves
displacement and leaves stress alone is now worth little. Second, the
transverse field (`v` = 2.85e-02) is no longer catastrophic — it is at the
level Phase 2 reached with an entire network devoted to one specimen.

## Three corrections to the literature framing before anything is built

These came out of the review and they matter, because the original plan rests
on two of them.

**mDEM has no quantitative results.** arXiv 2104.09623 is the canonical citation
for "mixed formulation fixes stress concentrations" and it contains **no error
tables, no relative L² norms, no timings, no parameter counts**. Every
comparison against DEM/PINN/FEM is a field plot. The strongest claim in the text
is "mDEM is able to resolve the stresses that arise near the clamping whereas
DEM and PINN fail." Any percentage attributed to it is from elsewhere or
invented. It also changes *two* things at once relative to DEM — mixed outputs
**and** Delaunay integration on scattered points — and runs no ablation
separating them, so even the qualitative gain is not attributable.

**VINO and WINO cannot be cited for stress accuracy.** Both are displacement-only
with stress derived post hoc, and **neither reports a stress error metric
anywhere**, including VINO's hyperelasticity case. Separately: VINO replaces
`log J` with a 4th-order Taylor series purely to keep element integrals
closed-form, with no error bound given. That substitution is worst exactly where
volumetric strain is large — i.e. at concentrations — and, more seriously, a
truncated polynomial has no singularity at `J → 0`, so it **removes the barrier
against element inversion**. Do not copy it; this project's Phase 0 work exists
partly because of how easily that barrier is lost.

**RAD's headline numbers are against the wrong baseline.** Wu et al.'s 85× on
Burgers is versus *fixed* collocation points. PACMANN (CMAME 2025) re-ran it
against *resampled uniform* — which is what this project already does — and the
advantage largely evaporates (0.16% vs 0.40% on Burgers; RAD **worse** on
Allen–Cahn, 0.93% vs 0.42%), with 9–13% wall-clock overhead. RAR-D fails
catastrophically above ~3D (89% error on 5D Poisson). Residual-only sampling was
*worse than uniform* in two independent studies.

## The plan, ordered by expected gain per unit of work

Every step states a prediction and a falsification criterion **before** the run,
because this project has twice drawn a wrong conclusion from a measurement made
under the wrong configuration.

---

### Step 1 — Double precision and per-component output scaling

**Why first: it is the cheapest thing on this list and the literature's effect
size is the largest.** Everything here runs in float32. PyTorch's L-BFGS default
`tolerance_change = 1e-7` sits *below* float32 machine epsilon (1.19e-7), so the
inner loop can stop on arithmetic rather than convergence. Xu et al. (arXiv
2505.10949) report **34–117×** improvements from the dtype alone on standard
benchmarks — vanilla FP64 PINNs beating architectures designed to fix those same
problems.

Separately, hnPINN (*EAAI* 2024) and MMPINN (*JCP* 510:113112) both address the
pathology this project measured as the 183× asymmetry: output components of
wildly different magnitude occupy different-curvature directions. Here
`‖v‖/‖u‖ ≈ 0.03–0.06`, so the transverse output is inherently the small one.
Non-dimensionalising each output head to O(1) is the standard remedy and costs
one scaling constant.

**Prediction.** FP64 improves von Mises by ≥1.3× at matched epochs. Output
scaling reduces the `v`-to-`u` error ratio (currently 12.5) by ≥25%.

**Falsification.** If FP64 changes nothing beyond three-seed noise (the ratio
sd measured in Phase 5 was ±0.01 on `v`), then float32 was not a limit here and
the literature's effect does not transfer — say so and move on.

**Cost.** A dtype audit plus a config flag; two 25-minute arms. Half a day.

---

### Step 2 — Multi-scale grouping radius

**Why: this is the hole a reviewer will find in the Phase 4 conclusion.** Phase 4
varied the FPS centroid *count* (32 → 128) and concluded the encoder "fails to
use" the fillet rather than discarding it. But the entire availability
literature identifies the *radius*, not the count, as the lever:

- **PointNet++ (2017)**, whose grouping GINOT inherits, says single-scale
  grouping cannot capture fine-grained patterns, and introduces MSG/MRG
  specifically to fix it. **GINOT uses one radius anyway** — an unexamined
  regression relative to the architecture it descends from.
- **PGOT (arXiv 2512.23192)** derives the mechanism: aggregation is a low-pass
  filter with cutoff `ω_c ∝ 1/σ`, above which "high-frequency geometric details
  are irreversibly erased", and *confirms by diagnostic experiment* that errors
  concentrate at complex boundaries. Their multi-scale fix gives **81.3%**
  surface-field MSE reduction on AirfRANS.
- **GeoTransolver (NVIDIA, arXiv 2512.20399)** uses radii spanning **0.01–5.0**
  for exactly this reason.

This project's encoder runs `radius = 0.15` on coordinates normalised to
[-1,1] — roughly the fillet's own length scale. That is the worst possible
place to put a single smoothing kernel.

**Prediction.** One of two outcomes, and both are useful. Either multi-scale
grouping raises `R/L` decodability *and* improves held-out `vm_fillet` by ≥1.2×
— in which case Phase 4's conclusion needs qualifying to "fails to use it *at
this scale*" — or it moves neither, in which case the extraction argument
becomes considerably stronger than it currently is.

**Falsification.** Stated above; the experiment is symmetric by construction.

**Cost.** An encoder change plus the existing probe (10 min) and two training
arms (~50 min). One day.

---

### Step 3 — Adaptive quadrature by equilibrium-residual indicator

**Why: there is a published result in exactly this setting, and it is large.**
Liu, Cai & Ramani (*CMAME* 415:116229) use an energy formulation with adaptive
quadrature on a **plate with a circular hole** and recover peak stress from
**7.92 → 13.89** against an adaptive-FEA reference of 13.8876 — final error
**<0.03%**. The un-refined quadrature under-predicted the peak by ~43%.

Energy formulation, stress concentration, peak-stress quantity of interest:
all three of this project's conditions at once. Their indicator is

```
η_T = |T|^(1/d) · ‖ ∇·σ_T + f ‖_{0,T}
```

and they note something that makes it *cheaper* here than in adaptive FEM: the
inter-element jump terms vanish, because the network is C¹.

This project already has the machinery — `geometry/triangulation.py` builds and
caches the graded mesh, and Phase 2 established that the quadrature *points*
must keep moving. Adaptive refinement adds a per-element indicator and a
refinement pass; the stratified resampling stays.

**Two caveats from the same paper, both load-bearing.** On a smooth problem the
gain plateaued once "network approximation error became dominant", and on an
L-shaped re-entrant corner adaptive refinement bought essentially **nothing**
(10.99% → 10.71%). A fillet is a smooth concentration, not a singularity, so
this project is in the favourable case — but the distinction is the difference
between a 43% recovery and no effect. Relatedly, XDEM (*Nat. Commun.*) removed
adaptive sampling entirely in favour of Williams-series enrichment and did
better with 30×30 uniform points: **for a known singularity, enrich the basis;
for a smooth concentration, refine the quadrature.**

**Prediction.** Peak von Mises error at the fillet improves ≥1.3×; global von
Mises improves less; displacement is roughly unchanged.

**Falsification.** If peak fillet error does not move, the quadrature is not the
binding constraint and Step 4 becomes the main hope.

**Cost.** Indicator, marking (Dörfler), refinement pass, and a re-run of the
Phase 2 collapse check to confirm refinement does not reintroduce it. Two days.

---

### Step 4 — Mixed u–P, with the ablation the literature never ran

**Why: it is the plan's centrepiece, the literature's gap is real, and the
evidence is weaker than everyone assumes.** The review found **no published
method combining (a) finite-strain hyperelasticity, (b) structural rather than
penalised enforcement of equilibrium, and (c) a quantitative local stress
benchmark with a matched displacement-only ablation.** DCEM has (b) and (c) but
is linear-elastic and its authors state it does not extend to nonlinear
constitutive laws. mDEM has (a) and neither (b) nor (c). VINO/WINO have none.

What the evidence *does* support:

- **DCEM** (arXiv 2302.01538): 1–2 orders of magnitude better stress than
  DEM/PINN, 59% lower von Mises error on a plate with a hole, 9× faster — **and
  worse than DEM on displacement**. That dissociation is the cleanest proof in
  the literature that stress and displacement accuracy are separate objectives.
  Linear elasticity only.
- **Arora et al.** (arXiv 2201.08363, Appendix B): a controlled mixed-vs-derived
  ablation with a ~10³ training-loss gap, attributed to stress sensitivity to
  displacement-gradient noise and to first- vs second-order derivatives. Small
  strain, and the metric is training loss.
- **Rezaei/Harandi** (*IJNME* 2024): σ to 0.80–0.87% against FEM using **separate
  networks per field**; hard Dirichlet constraints + L-BFGS cut errors to **⅓**
  and cost from 4 h to 40 min. Small strain, and the mixed-vs-standard
  comparison is figures only.

**The specific missing experiment, which is small and is the contribution:**
mixed vs displacement-only **on identical point sets and identical integration**,
reporting relative L² of `P` (or von Mises) separately from `u`, in the fillet
region and globally. This project is unusually well placed to run it — the FEM
reference, the region masks, the frozen eval set and the seed protocol already
exist.

**Prediction.** `vm_fillet` improves ≥1.5×; `u` degrades slightly or is
unchanged (the DCEM dissociation); cost per epoch rises ~1.5–2×.

**Falsification.** If `vm_fillet` does not improve on a matched ablation, the
mixed form does not help in this regime and the paper must not claim it —
which, given mDEM's evidential state, would itself be a publishable negative.

**Cost.** Decoder head for the four `P` components, a constitutive-tie loss, a
weight sweep. Three to four days including runs.

---

### Step 5 — Second-order optimisation, revisited

**Why lower than it looks.** Phase 4 measured the physics-vs-supervision gap at
oracle conditioning at **2.1×**, which bounds what any optimiser can win here,
and Phase 5 showed 1200 more epochs bought 4.8× for free. Budget beats
preconditioning at this stage.

But two things in the review argue for revisiting *after* Step 1. The measured
bound was taken at float32, and the same literature says the L-BFGS stall that
would produce such a bound is often **arithmetic rather than convergence**.
Separately, natural-gradient methods are the principled answer to exactly the
183× curvature asymmetry, because `G_E⁺` rescales each direction by its own
curvature — though **no paper demonstrates this on a component-magnitude
mismatch**, so that reasoning is principled, not evidenced.

Practical notes if it is attempted: use **Armijo, not strong Wolfe** (the
documented L-BFGS failure is a zero step size, a silent early stop);
**SSBroyden/BFGS beat L-BFGS by 4–5 orders of magnitude** on Burgers (Kiyani et
al., CMAME 2025); and **resampling corrupts quasi-Newton curvature pairs** — so
with stratified resampling on every call, resample only at stalls and reset the
history, or the inverse-Hessian estimate is differencing noise.

**Prediction.** ≤2.1×, and less after Step 1.

**Cost.** Two days. Do it last.

---

### Step 6 — Measure what the paper claims

Not an improvement, a correctness fix in the metrics. The claimed contribution
is **local stress-concentration accuracy**, and this project reports relative
L². Those are different objectives and the literature shows they trade off:
R-PINN is 2× *worse* in L² and 3.9× *better* in L∞ than its comparator on the
same problem, purely from the choice of error indicator.

Add to `eval/compare_reference.py`: peak von Mises error (signed), L∞ over the
fillet region, and the stress-concentration factor against the FEM reference.
Half a day, and it should be done **before** Steps 3 and 4 so their predictions
are testable in the right norm.

---

## What this project has that the literature does not

Worth recording, because it shapes what the paper should claim.

The geometry review found **no published per-parameter probing analysis of a
neural-operator geometry latent**. This project's measurement — pooled R² of
0.974 for the taper ratio against 0.03–0.30 for the fillet — appears to be the
first. The availability-versus-extraction control (probing an *untrained*
encoder under centroid scaling, then showing the trained model gains <1% from
the larger budget) is, as far as three independent searches could determine,
without precedent.

The symptom is all over the literature and undiagnosed: MR-GVNO's energy-norm
errors run **3–5× its L² errors** in every case, and PGOT confirms error
concentration at complex boundaries. Nobody has localised it to a specific
geometric parameter.

The nearest precedent for the headline result is **Geom-DeepONet** (*CMAME*
2024), where parametric + SDF conditioning beat a PointNet encoder by **2.1×** —
but that comparison changes parameters, SDF and SIREN simultaneously. The
capacity-matched swap measured here is cleaner.

**Two things to do about it.** Close the multi-scale radius hole (Step 2) before
claiming extraction rather than availability. And cite Geom-DeepONet, PGOT,
PointNet++ and `Do Neural Operators Forget Geometry?` (arXiv 2605.05862) — the
claim that encoders lose fine geometric detail is *not* novel; the per-parameter
localisation and the availability/extraction control are.

## Suggested order

1. Step 6 (metrics) — half a day, and it gates the others' testability
2. Step 1 (FP64 + scaling) — half a day, largest effect per unit work
3. Step 2 (multi-scale radius) — one day, closes a reviewer hole
4. Step 3 (adaptive quadrature) — two days, best-evidenced stress lever
5. Step 4 (mixed u–P) — four days, the paper's centrepiece
6. Step 5 (optimiser) — two days, bounded at 2.1×

Steps 6, 1 and 2 are container-feasible now. Steps 3–5 want a GPU, and should
follow the confirmation run in `docs/gpu_confirmation.md` so they build on a
verified baseline rather than a 2-CPU one.
