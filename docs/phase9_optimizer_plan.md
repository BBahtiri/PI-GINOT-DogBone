# Phase 9 — Can a better optimizer buy accuracy? (plan)

Status: **plan, pre-registered.** Only 9.0a has been run, and it is
evaluation-only. Predictions are fixed here before any run, following the
project's rule.

- Reproduce 9.0a with `python -m verification.seed_spread`, which writes
  `docs/phase9_seed_spread.json`.
- An independent check reviewed this plan before it was committed. Its
  corrections are folded in (§6).

---

## 1. The question and the short answer

The question: two recent papers from the Karniadakis group report
orders-of-magnitude accuracy gains from replacing Adam with curvature-aware
optimizers. Is the optimizer worth working on here?

The short answer is **yes, in stages, starting with the cheapest.**

- Our own runs point to the optimizer as a real lever (§3). No run behind the
  current results ever lowered its learning rate. Much of the in-range error
  changes with nothing but the random initialisation.
- The papers' headline methods do not transfer as they stand (§2).
- Whether the *remaining* error is the optimizer's or the objective's is still
  open. 9.0b and 9.2 decide it before any GPU time is spent.

The previous plan (`docs/phase6_accuracy_plan.md`, Step 5) ranked the optimizer
last, "bounded at 2.1×". That bound needs revisiting for three reasons:
- it was measured on field L2 errors, in float32;
- it predates the `K_t` metrics and the new families;
- it predates §3.

---

## 2. What the two papers show, and what transfers

**Kiyani, Shukla, Urbán, Darbon, Karniadakis**, *Optimizing the Optimizer for
PINNs and KANs*, arXiv 2501.16371v6.
- **Method.** Self-scaled BFGS and self-scaled Broyden (SSBFGS, SSBroyden). Both
  are dense quasi-Newton updates that rescale the inverse-Hessian estimate each
  iteration. They use a strong Wolfe line search and a short Adam warm-up, in
  float64.
- **Burgers result** (1,341 parameters, Table 1). Relative L2:
  - L-BFGS 2.05e-3;
  - BFGS 1.50e-5;
  - SSBFGS 9.62e-8;
  - SSBroyden 7.57e-8.
- **Precision matters.** Allen–Cahn with SSBroyden: 4.73e-4 in float32
  (Table 7) against 1.15e-6 in float64 (Table 8). For plain BFGS the gap is
  only 2.6×.
- **Scale.** The largest network is 33,667 parameters (Stokes, Table 14).
- **Operator learning.** A data-driven DeepONet: SSBroyden 5.7e-3 / 6.9e-3
  against Adam's 2.1e-2 / 6.2e-2, at 5× the wall time (Table 15).
- **Mini-batch training** is left open (p. 29). A multi-batch variant is only
  mentioned, without numbers.

**Jnini, Kiyani, Shukla, Urbán, Ahmadi Daryakenari, Müller, Zeinhofer,
Karniadakis**, *Curvature-Aware Optimization for High-Accuracy PINNs*,
arXiv 2604.05230v1.
- **Methods.**
  - The self-scaled BFGS/Broyden optimizers above.
  - A natural gradient (NG). This is Gauss–Newton solved in sample space, with
    cost O(N²P) (§2.3).
- **Recommendation.** NG is the "first choice for elliptic and parabolic PDEs"
  when fast convergence matters. The self-scaled methods are for robustness and
  "very high final accuracy" (§7).
- **Stokes result** (Table 8). This is the same problem, network and
  quasi-Newton numbers as Kiyani et al.'s Table 14.
  - Adam reaches 2.70e-2 on velocity.
  - The self-scaled methods reach 3.25e-4 / 1.6e-4.
  - NG reaches that error in 150 s against 3,600 s.
  - **The pressure error stays at about 0.11 under every optimizer.** Some error
    is simply not the optimizer's.
- **Batched training.** The batched self-scaled BFGS is demonstrated on one 1D
  Poisson problem, in float32 (Fig. 24).

**Work on energy losses specifically:**
- Müller & Zeinhofer (ICML 2023, arXiv 2302.13163) introduced the *energy
  natural gradient* for PINNs **and the deep Ritz method**, which is our kind
  of loss.
- Guzmán-Cordero, Dangel, Goldshlager & Zeinhofer (arXiv 2505.12149) scale it
  with Woodbury, SPRING and Nyström to about 10⁶ parameters. That is on
  residual-form Poisson problems only; it is untested on a Ritz loss.
- For the deep energy method, Adam followed by L-BFGS (strong Wolfe) is common
  practice. Abueidda et al. (arXiv 2201.08690) use it for hyperelasticity.

| | the papers | this project | consequence |
|---|---|---|---|
| **parameters** | ≤ 33,667 | **573,282–573,378 trainable** (decoder 364,930 + parameter MLP 208,352–208,448; the encoder is unused) | A dense P×P matrix is **2.6 TB** in float64. SSBFGS and SSBroyden cannot be used as published. |
| **batching** | one PDE. Quasi-Newton: full point sets redrawn every 10–500 iterations. NG: fresh points every iteration | operator: 4 of 64 geometries per step, fresh random quadrature every step | Changing batches corrupt quasi-Newton curvature pairs. Redraw at intervals and reset the history, or use full-bank passes. NG tolerates fresh points. |
| **loss** | residual least squares | total potential energy (Ritz) + barrier | The NG matrix becomes the **tangent stiffness** (below). No self-scaled quasi-Newton or natural-gradient result exists for an energy-form operator. |
| **precision** | float64 throughout | float32 | Our float64 test (6.2) was confounded and is "not demonstrated". |
| **error scale** | 1e-3 → 1e-7 on smooth solutions | `K_t` 1.5–7% | The gains are not a forecast for us. |

**What transfers.**
1. The diagnosis that the optimizer, not the network, can cap accuracy.
2. float64 for any final polish.
3. The energy natural gradient.

For our loss, Π(u) = ∫W(F) dA plus the barrier, the energy natural gradient
uses the Gauss–Newton matrix

    G_ij = ∫ ∂_θi(∇u_θ) : A(F_θ) : ∂_θj(∇u_θ) dA,    A = ∂²W/∂F∂F (+ barrier).

The exact Hessian adds ∫P : ∂²_θθ∇u_θ dA. G is the **FEM tangent stiffness,
Galerkin-projected onto the network's tangent space**. A step
θ ← θ − α(G + λI)⁻¹∇Π is the function-space Newton step our FEM reference
takes, projected onto what the network can represent.

This matters because of Phase 3. That phase measured a 183× curvature ratio
between the axial and transverse fields, and G rescales each direction by its
own stiffness. That argument is principled but not yet evidenced: no paper
shows it on an operator.

---

## 3. Phase 9.0a — what our own runs say (done, evaluation only)

### 3.1 No run behind the current results ever lowered its learning rate

**The current results never decayed.**
- All 13 Tier-1 runs and all six 6.7 runs (energy and mixed + energy) trained at
  a constant **3e-4**, with the plateau scheduler cutting 0 times.
- The validation energy (an EMA over two validation geometries) was still
  drifting down at step 1600: by 0.05–0.26% over the last 400 steps. It was not
  monotone in 8 of 13 runs.
- The runs were therefore both unconverged and in the noisy regime that
  constant-rate Adam settles into on a stochastic objective.

**Elsewhere in the repository:**
- Of the runs with a history file, only the Phase 2 operator study decayed its
  rate (3e-4 → 1.5e-4).
- The Phase 4 ceiling loops and the Phase 6.6 pilot loop used cosine decay to 2%
  of the rate.

**One earlier comparison points the other way, and is confounded.** In Phase 6.6
the cosine pilot loop's energy arm scored `K_t` R² **+0.52** in range. The
constant-rate trainer loop scored **+0.93** on the same six geometries. The
loops differ in more than the schedule, and each figure is one seed. Annealing
is therefore a hypothesis to test with a control, not a known win.

### 3.2 Much of the in-range error changes with the initialisation alone

**All seeds see the identical data stream.** The batch order comes from
`default_rng(42)` in both loops, and the quadrature from `EnergyLoss(seed=0)`.
The seed changes only the initialisation.

The relative `K_t` error e[seed, geometry] over three seeds is decomposed into
two parts:
- **"Removed by averaging"**, the fraction of the per-seed mean squared error
  (MSE) that disappears when the three seeds' predictions are averaged. This
  equals mean_g var_ddof0 / mean e².
- **"Systematic"**, the estimated error of an infinite-seed average,
  sqrt(mean m² − mean s²/3). The term under the root is unbiased; the root is
  not.

| family | split | mean \|err\| | seed sd | systematic | removed by averaging | 3-seed average \|err\| |
|---|---|---|---|---|---|---|
| dog-bone | in range | 1.51% | 0.93% | 1.61% | 0.17 | 1.40% |
| open hole | in range | 3.36% | 3.13% | 3.39% | 0.31 | 3.35% |
| inclusion | in range | 1.75% | 1.09% | 1.67% | 0.20 | 1.66% |
| double notch | in range | 4.09% | 3.99% | 3.86% | 0.34 | 3.52% |
| **single notch** | in range | **6.79%** | **9.74%** | not resolvable | **0.83** | **3.22%** |
| dog-bone | out of range | 3.67% | 3.52% | 3.11% | 0.38 | 3.60% |
| open hole | out of range | 12.30% | 1.73% | 12.39% | 0.01 | 12.30% |
| inclusion | out of range | 1.80% | 1.17% | 1.76% | 0.20 | 1.71% |
| double notch | out of range | 17.26% | 6.93% | 22.80% | 0.06 | 15.24% |
| single notch | out of range | 14.77% | 9.35% | 14.75% | 0.19 | 13.91% |

**In range, 17–83% of the squared error disappears when three seeds are
averaged.** That is error set by the initialisation, under an identical data
stream.
- An optimizer that reached a unique minimum would remove this variance.
- Where that minimum lies, and whether it is close to the FEM, is what 9.0b and
  9.2 test.

**Three caveats:**
1. **The single notch cannot be split with three seeds.** The raw systematic²
   is negative (−18.3 %²).
   - Its seed variance is mostly per-seed offsets across all geometries: mean
     errors of +0.23 / +7.93 / −7.85%, which carry 66% of the seed sum of
     squares.
   - About two effective degrees of freedom are behind it.
   - Its halving under averaging rests on the +7.93 and −7.85 cancelling.
2. The dog-bone's seed variance is also mostly offsets (55%). For the other
   three families it is not (7–27%).
3. **Out of range, the error is largely systematic.**
   - Open hole 99%, double notch 94%, single notch 81%, dog-bone 62%, inclusion
     80%.
   - On the hole and notches an optimizer will not fix extrapolation.

### 3.3 The energy barely separates the seeds; `K_t` does

| family | spread of the final validation energy across seeds | in-range `K_t` seed sd |
|---|---|---|
| open hole | 0.005% | 3.13% |
| inclusion | 0.005% | 1.09% |
| double notch | 0.032% | 3.99% |
| single notch | 0.014% | 9.74% |

**Seeds that differ by 1–10% in `K_t` differ by parts in 10⁴ or less in the
validation energy.** That energy is on the two validation geometries, not the
scored ones. Moreover, the lowest-energy seed is the most accurate in only one
of four families: for the open hole it is the *least* accurate.

This is consistent with both hypotheses:
- **H-opt.** `K_t` is set by energy differences below what constant-rate
  stochastic training resolves, so a deeper minimisation would fix it. float32
  round-off, at about 1e-7 relative, is not the limit; gradient noise is.
- **H-obj.** The objective is nearly flat in the directions that move `K_t`, so
  even its exact minimiser need not be accurate there.

9.0b and 9.2 separate the two.

---

## 4. The plan

**Order:** 9.0b → 9.1 → 9.2 in the container, then 9.3 on a GPU if the rules
below say so, then 9.4.

### 9.0b — Energy against error on the scored geometries (container; about 30 min)

**Measurement.** For all 15 final models, compute Π on each of the 12 in-range
scored geometries.
- The rule is fixed and fine: the FEM reference mesh with a 6-point rule, never
  used in training.
- For each geometry, take the three seed pairs and ask whether the lower-energy
  seed has the smaller \|`K_t` err\|.

**Prediction (H-opt).** Pooled over the four new families (144 pairs), the
lower-energy seed is the more accurate in **>60%** of pairs. H-obj predicts
about 50%.

This is evidence for 9.2, not a gate on 9.1.

### 9.1 — Anneal against control (container; about 9 h)

**Arms.** Each run starts from one of the 15 checkpoints:
- the four new families × 3 seeds;
- the dog-bone 6.7 energy runs (`fixed_long/oracle-energy`,
  `retest/energy_s1`, `retest/energy_s2`).

Each continues for **800 steps** with the loop, bank, batch, quadrature and
float32 unchanged, and `ReduceLROnPlateau` disabled.

**Adam restart.** The Tier-1 checkpoints hold no optimizer state. The dog-bone
checkpoints do, but it is **deliberately discarded**, so every run restarts
Adam the same way, with a 50-step warm-up 3e-5 → 3e-4.

- **Anneal**: cosine decay 3e-4 → 3e-6 over the remaining 750 steps.
- **Control**: constant 3e-4. It separates "cooling" from "800 more steps plus a
  restart".
  - Checkpoints every 200 steps, each scored on the 12 in-range geometries.
  - For each run, the sd over its four checkpoints of each geometry's `K_t`
    error, averaged over geometries, is its *wander* W.

Both arms run on all five families × 3 seeds, 30 runs, with the scorers and
held-out sets unchanged.

**Predictions** (all anneal against control, at matched steps):
- **P9.1a (spread).** The anneal arm's in-range `K_t` seed sd is ≥30% below the
  control's on ≥3 of 5 families.
- **P9.1b (accuracy).** The anneal arm's in-range mean \|err\| is ≥15% below the
  control's on ≥3 of 5 families, and ≥25% below it on the single notch.
- **P9.1c (mechanism: noise floor, not separate basins).** In the control arm,
  W ≥ 0.5 × the family's seed sd on ≥3 of 5 families.
- **P9.1d (extrapolation).** Out-of-range mean \|err\| on the open hole and
  double notch: anneal within ±20% relative of control.

**If P9.1a and P9.1c both fail** (W < 0.25 × seed sd), the seeds sit in
separate basins:
- Neither annealing nor a curvature method, which also converges within one
  basin, will remove the spread.
- The remedies become ensembling, reported as a method with its cost, or
  changing the landscape.
- 9.2 still tells whether each basin's error is reachable by optimization.

**Cost.** About 34 min per run (measured Tier-1 rate: 2.4–2.8 s per step).
30 runs make about 17 CPU-h, or about 9 h on two lanes.

### 9.2 — Is the objective's minimum accurate? (container; about 4 h)

This separates *optimization* error from *objective* error on one geometry at a
time. It removes the constraint that one network must serve 64 geometries, so a
positive result is **necessary but not sufficient** for 9.3. It measures an
upper bound on what optimization can deliver for the operator.

**Families.** Open hole and double notch, which have the largest systematic
parts (3.4% and 3.9%).

**Geometry selection** (fixed now, tie-free):
- Among the 12 in-range geometries, take those whose seed-0 \|`K_t` err\| after
  the 9.1 anneal arm is **≥2%**.
- Pick the largest and the smallest of them, with ties going to the lower index.
- If fewer than two qualify, take the two largest \|err\| overall.

**Start.** The seed-0 anneal checkpoint. The whole network is fine-tuned on that
one geometry's energy. Only trainable parameters are passed to the optimizer;
the unused encoder sits in `model.parameters()`.

**Arms:**
- **(a) L-BFGS.** float64, history 50, strong Wolfe, 1,000 iterations.
  - The quadrature points are fixed within each 100-iteration window: twice the
    training count, stratified in the training triangulation.
  - They are redrawn at each window, with the history reset.
- **(b) Adam** with cosine decay 3e-4 → 3e-6, float32, training quadrature,
  2,000 steps. This is the first-order reference at similar wall-clock.

**Tracked every 50 iterations:**
- the `K_t` error;
- the peak location;
- the von Mises L2 error;
- Π_net on the independent 9.0b rule.

**Guard against exploiting the quadrature.**
- Refine the FEM mesh in a nested way (each triangle split in four), so the
  polygon is the same, and solve on both meshes.
- If the energy converges at rate ≥1, then
  δ = Π_FEM(h) − Π_FEM(h/2) ≥ Π_FEM(h/2) − Π_exact.
- The network's field is admissible on the polygon: both families are Dirichlet
  only on straight edges.
- A run with Π_net < Π_FEM(h/2) − δ is flagged and discarded.

**Hypotheses:**
- **H-opt (the prediction).** On ≥3 of 4 geometries, arm (a) cuts
  \|`K_t` err\| **≥3× and to ≤1%**.
- **H-obj.** On ≥3 of 4 geometries, \|`K_t` err\| stays at ≥2/3 of its start
  while Π_net settles (a change of <1e-6 relative over the last 200
  iterations).
- **In between.** 9.3 starts as a one-family, one-seed pilot before any
  multi-seed commitment.

**Cost.** A timing pilot comes first.
- Estimate: 2.5–4 s per float64 evaluation, from twice the points and a float64
  factor of 2–3. The 1.92× measured in 6.2 had a partly float32 path.
- Arm (a): 4 runs × 55–85 min.
- Arm (b): 4 runs × about 25 min.
- About 6–7.5 CPU-h in total, or 3–4 h on two lanes.

### 9.3 — Curvature-aware training of the operator (conditional on 9.2; GPU)

**Arms:**
- (i) the 9.1 recipe, as the baseline;
- (ii) plus an **L-BFGS polish** on the full 64-geometry bank.
  - float64, points fixed within windows, redraw and reset.
  - This is the deep-energy-method practice, adapted to an operator.
- (iii) the **energy natural gradient**.
  - The G above, matrix-free, with about 20 CG iterations of
    Jacobian-vector products per step.
  - Armijo line search, damping λ, curvature on a 4-geometry subsample, fresh
    points each step, float64.
  - Woodbury/SPRING (arXiv 2505.12149) are used if the hardware allows. They are
    untested on a Ritz loss.

**Out of scope: dense SSBFGS/SSBroyden** (2.6 TB at P ≈ 573k). These would need
a network about 20× smaller, which is a separate architecture study.

**Cost on the container CPU:**
- a full-bank float64 evaluation is 64 × 0.63 s × 2–3, about 1.5–2 min, so an
  L-BFGS iteration is 2–3 min;
- an energy-natural-gradient step is 20 products × 4 geometries × 0.63 s × 2–3
  × about 3, so 5–7 min before the line search;
- one run is therefore days.

On one modern GPU a run should drop to hours. That is an estimate to confirm
with a timing pilot.

The predictions are registered after 9.2.

### 9.4 — Final recipe from scratch

The winning recipe is trained from scratch on all five families, 3 seeds each,
with the same held-out sets and scorer. It replaces the paper's multi-geometry
table, and wall-clock times are reported.

### Decision rules

- **If 9.1 passes P9.1a**, annealing becomes the default schedule for every
  later run, including the ADF ablation and Tier 2.
- **If 9.2 supports H-opt**, 9.3 is worth the GPU time. It is also a paper
  section: a curvature-aware optimizer for an energy-form operator, scored on
  verified FEM `K_t`.
- **If 9.2 supports H-obj**, optimizer work stops after 9.1. The systematic
  error then belongs to the objective: next come adaptive quadrature at the
  peak and the Phase 7 C-series on the free edges.

---

## 5. Where this sits among the next steps

1. **9.0b + 9.1 + 9.2.** About 13 h in the container, feasible now. Every
   current number was produced at a constant learning rate. The seed spread
   also blurs every comparison still to be made.
2. **A-ADF ablation on the inclusion.** Run it with the 9.1 schedule.
3. **Tier 2**, one operator across families, on the settled recipe.
4. **Out-of-range notch-flank peaks.** An extrapolation problem, 94% systematic
   on the double notch and 81% on the single notch, so not an optimizer one.
5. **Preprint methods update** (arXiv 2607.23299 describes the residual form).

---

## 6. Corrections from the independent check (before commit)

**Corrected:**
- **Learning-rate history.** Cosine-decayed runs do exist (the Phase 4 ceilings
  and the 6.6 pilot), and the confounded 6.6 comparison points against
  annealing.
- **Optimizer state.** The dog-bone checkpoints do hold it.
- **Resampling range.** NG redraws points every iteration.
- **Prior energy-loss work.** L-BFGS on energy losses is common practice (DEM).
- **Systematic shares.** The out-of-range figures are 94% and 81% on the notches;
  the 99% was the open hole.
- **9.2 cost.** Underestimated about 4× (finer quadrature, float64).
- **Guzmán-Cordero et al.** Their tests are residual-form only.
- **Jnini §7.** The wording is "very high", not "highest".
- **The Allen–Cahn precision gap** is SSBroyden's; for BFGS it is 2.6×.

**Reworded:**
- "Removable by an optimizer" is now "set by the initialisation", with H-opt and
  H-obj made explicit.
- The single notch's systematic part is marked not resolvable.

**Added:**
- control arms on all five families;
- the wander measurement (P9.1c);
- a tie-free geometry rule;
- a relative H-opt threshold;
- an independent evaluation rule and a nested-mesh guard;
- 9.0b.
