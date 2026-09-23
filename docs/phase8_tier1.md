# Phase 8 Tier 1 — one operator per family: protocol and predictions

Committed **before any Tier-1 model is trained or scored.** The dog-bone keeps
its current geometry (no grip section, per the authors' decision). Its three
6.7 operators stand as its Tier-1 row, provided the loop-equivalence check
P-T4 holds.

## Families and sets

| family | model | parameters (training box) | out-of-range direction |
|---|---|---|---|
| open hole (F2) | quarter | `W` 16–26, `d/W` 0.2–0.5, `L_half/W` 1.5–2.5 | `d/W` 0.55–0.60 |
| rigid inclusion (F4) | quarter | as F2 | as F2 |
| double U-notch (F3) | quarter | `W` 16–26, `t/W` 0.1–0.3, `ρ/t` 0.25–1, `L_half/W` 1.5–2.5 | `ρ/t` 0.15–0.25 |
| single U-notch, clamped (F5) | half | as F3, `L_half/W` 2–3 | `t/W` 0.30–0.35 |

All sets are defined in `geometry/families.py` with frozen seeds:
- a 64-geometry training bank;
- 2 validation geometries (for the LR scheduler only);
- 12 in-range test geometries, stratified 4 per third of `d/W` (F2, F4), `ρ/t`
  (F3) or `t/W` (F5), inside a 5% margin of the box;
- 8 out-of-range test geometries.

## The stress concentration factor

`K_t = max Cauchy von Mises over the peak region / (N / A_gross)`
- `N` is the axial force: the grip reaction for the FEM, and the section
  integral of `P₁₁` at `x = 0.75 L` for the network.
- `A_gross` is the gross section per unit thickness (`H`, or `2H` for the half
  model).
- The peak region is the whole domain, except for F5, where it is `x ≤ L/2`:
  its clamped grip corners are weakly singular.
- The network and the FEM are evaluated at the same element centroids.
- At small load this is the handbook's gross-section `K_t`.

The first definition used a far-field band mean. It was replaced before any
training because clamping the grip moved the band (8.7).

## The references (verified, `verification/family_fem.py`)

- **Setup.** 60 elements per feature radius at the root, linear grading (0.2),
  the unchanged Newton core. Newton is seeded with the transfinite lift of each
  family's Dirichlet data.
- **Level-3 run 1 failed 4 of 6 checks.** Both causes were diagnosed and fixed
  before run 2 (`c6cd665`):
  - an abrupt size field under-resolved the field around small notches;
  - the band nominal was inside a clamped grip's end zone.
- **Run 2 (`docs/phase8_family_level3_run2.json`):**
  - shallow semicircular notches: −1.1% (double) and −1.05% (single) against
    3.065;
  - rigid inclusion: +1.24% against Goodier's 1.535;
  - clamping moves `K_t` by 0.005% (double notch) and 0.10% (inclusion);
  - 60 vs 85 elements per radius: within 0.30%;
  - the open hole on the new mesher: within 0.32% of Peterson.

**The inclusion reference value (derivation).** Take a Michell stress function
in plane stress, Kolosov `κ = (3 − ν)/(1 + ν) = 2.252` (ν = 0.23), and a
far-field `σ_xx = S`.
- **Axisymmetric part.** `σ_rr = A + B/r²`, `A = S/2`, and `2μ u_r = ((κ−1)/2) A r − B/r`.
  Setting `u_r(a) = 0` gives `B = (κ−1)A a²/2`, so `σ_rr(a) = S(κ+1)/4`.
- **`cos 2θ` part.** `φ = (−S/4) r² + D r⁻² + E` (times `cos 2θ`). Setting
  `u_r = u_θ = 0` at `r = a` gives `E = −S a²/(2κ)` and `D = S a⁴/(4κ)`, so
  `σ_rr(a) = S(κ+1)/(2κ) cos 2θ`.
- **Total at the pole (θ = 0):** `σ_rr = S(κ+1)(κ+2)/(4κ) = 1.535 S`.
- **Consistency check.** The same fields give `σ_θθ = ν σ_rr` there, which is
  zero hoop strain, as a bonded rigid boundary requires. The independent check
  of 8.5 derived the same 1.54.

## Training (`training/family_trainer.py`)

The Phase 6.7 trainer loop's mechanics, reproduced for any family:
- **Optimiser and schedule:** Adam, lr 3e-4, clip 1.0. One step per "epoch" on
  4 geometries drawn without replacement from the 64-bank (order
  `default_rng(42)`). ReduceLROnPlateau (0.7, patience 10) on an EMA of the
  validation energy, every 25 steps.
- **Budget:** 1600 steps.
- **Loss:** the energy form with stratified resampling on a ~1400-triangle
  quadrature mesh, 8 elements per feature radius; barrier 1e3 at J 0.05;
  float32.
- **Conditioning:** oracle, on the family's exact parameters.
- **Dirichlet conditions:** hard, through the family's distance-function spec
  (`geometry/adf.py`). The rigid inclusion is its first use on a curved
  boundary, and the clamped single notch its first use without a `v`-symmetry
  plane.
- **Seeds:** 0, 1, 2 per family. Measured speed is ~2.8 s/step on one thread,
  about 75 min per run, two runs in parallel.

## Predictions

| | prediction | holds if |
|---|---|---|
| **P-T1** | each new family has skill on `K_t` | in-range mean \|`K_t` err\| below the constant baseline (predict the in-range mean) on at least 2 of 3 seeds, for each of F2–F5 |
| **P-T2** | free-edge peaks are under-predicted, as on the dog-bone | mean signed in-range `K_t` error < 0 on every seed, for F2, F3 and F5 |
| **P-T3** | sharper concentrations are harder | F3, pooled over seeds and in-range geometries: Spearman(\|`K_t` err\|, reference `K_t`) > 0 with p < 0.05 |
| **P-T4** | the family loop is equivalent to the trainer loop | dog-bone through the family loop (seed 0): in-range mean \|`K_t` err\| in [0.5%, 2.4%] and R² ≥ 0.19 (the 6.7 range widened by one seed-sd), scored exactly as in 6.7 |

**Reported without a prediction:**
- out-of-range errors;
- displacement and von Mises L2;
- the F2-vs-F4 comparison in scale-free terms (R², |error|/`K_t`), with no
  mechanistic claim (8.5);
- the free-edge residual per family.

Expectations, stated so they can be wrong:
- P-T1 holds for all four families. `K_t`'s spread is dominated by parameters
  the oracle sees.
- P-T2 holds for F2 and F3 and is a coin flip for F5, where bending and
  clamping complicate the sign.
- P-T3 holds.
- P-T4 holds.

---

## Results

**Budget.** 13 runs, all completed (`docs/phase8_tier1_runs.json`):
- the four new families at seeds 0, 1 and 2;
- the dog-bone through the family loop at seed 0.

Each run took 65–74 min on one thread, two lanes in parallel, for about 8 h of
wall clock. Scores are in `docs/phase8_tier1_scores.json`, and
`python -m verification.family_score --report` reproduces every verdict.

### Verdicts

| | prediction | result | verdict |
|---|---|---|---|
| **P-T1** | every new family beats the constant in range (≥ 2 of 3 seeds) | open hole 3/3, double notch 3/3, single notch 3/3, **rigid inclusion 1/3** | **FAILS** (holds for 3 of 4 families) |
| **P-T2** | free-edge peaks under-predicted on every seed | open hole 3/3 · **double notch 0/3** (over-predicted every seed) · **single notch 1/3** (+0.2, +7.9, −7.8%) | **FAILS** (holds for the open hole only) |
| **P-T3** | sharper double notches have larger errors | Spearman −0.10, p = 0.56, n = 36 | **FAILS** |
| **P-T4** | the family loop is equivalent to the trainer loop | dog-bone seed 0: R² +0.83, \|err\| 0.96%; the 6.7 trainer loop's seed 0 gave +0.85, 0.97% | **HOLDS** |

Of the four expectations stated before the run, one was right (P-T4) and three
were wrong.

### Per family, three seeds (mean ± sd)

| family | in-range `K_t` R² | in-range \|err\| | constant | in-range bias | out-of-range \|err\| | `u` L2 | `v` L2 | vm L2 |
|---|---|---|---|---|---|---|---|---|
| dog-bone (6.7, trainer loop) | +0.61 ± 0.22 | 1.51 ± 0.49% | 2.43% | −1.34 ± 0.69% | 3.67 ± 3.06% | 1.2e-3 | 1.6e-2 | 9.9e-3 |
| open hole | +0.60 ± 0.24 | 3.36 ± 1.59% | 5.94% | −3.35 ± 1.61% | 12.30 ± 0.50% | 4.1e-3 | 1.9e-2 | 1.8e-2 |
| rigid inclusion | −0.18 ± 0.29 | 1.75 ± 0.31% | 1.60% | −1.66 ± 0.47% | 1.80 ± 0.49% | 1.3e-3 | 1.7e-2 | 8.8e-3 |
| double notch | +0.85 ± 0.10 | 4.09 ± 0.54% | 12.28% | +1.55 ± 1.08% | 17.26 ± 2.05% | 8.1e-3 | 1.0e-1 | 3.1e-2 |
| single notch, clamped | +0.67 ± 0.30 | 6.79 ± 2.53% | 15.86% | +0.11 ± 7.89% | 14.77 ± 7.87% | 7.2e-3 | 5.7e-2 | 2.1e-2 |

The L2 columns are the mean over seeds of the per-seed median over the 12
in-range geometries.

In-range reference `K_t`:
- dog-bone 1.17–1.30 (gauge definition);
- open hole 3.12–4.01;
- inclusion 1.41–1.50;
- double notch 3.08–4.84;
- single notch 3.30–5.51.

### What the numbers say

*Corrected after an independent check (commit 8.14). The first version
overstated points 2, 4, 5 and 6; what changed is listed at the end.*

1. **In range, three of the four new families have skill on `K_t`.**
   - The open hole, double notch and single notch beat the predict-the-mean
     baseline on every seed, and so did the dog-bone in 6.7.
   - Errors are 3.4–6.8%, against baselines of 5.9–15.9%.
   - Relative to each baseline, the sharper families are *not* worse:
     |err| / constant is 0.62 (dog-bone), 0.57 (hole), 0.33 (double notch) and
     0.43 (single notch).
2. **The rigid inclusion has no skill, and its field is wrong at the interface.**
   - Its reference `K_t` hardly varies (1.41–1.50, baseline 1.6%), and the
     operator's 1.75% error is a steady under-prediction.
   - The peak locations show more:
     - the FEM peak is on the interface at θ = 28–33°;
     - the network's maximum sits on the loading axis (θ < 5°) in 30 of 36
       in-range rows, and off the interface (r/a up to 1.30) in 25 of 36.
   - At the FEM's peak element, the network's von Mises is 6.0 / 8.4 / 6.0% low
     per seed, and down to −14% on one geometry.
   - So the small `K_t` error partly comes from a maximum in the wrong place,
     landing on a flat plateau. The stress near the bonded interface is off by
     6–8% at the true peak.
   - The distance-function layer imposes the displacement exactly (Level 0),
     but this is the family where the field next to a curved Dirichlet boundary
     is worst. That makes ablation A-ADF (R-equivalence vs plain product, and
     the corner behaviour of §3 of the plan) a live question, not an
     academic one.
3. **Across families the absolute error rises with the `K_t` level, within a
   family it does not.**
   - By mean reference `K_t`: 1.5% (dog-bone, 1.21), 1.8% (inclusion, 1.46),
     3.4% (hole, 3.44), 4.1% (double notch, 3.71), 6.8% (single notch, 4.47).
   - This ranking is post hoc, over five points that also differ in `K_t`
     definition (the dog-bone's), boundary conditions and model type.
   - The pre-registered within-family test failed (P-T3), and so does
     Spearman(|err|, ρ/t) (+0.02, p 0.91).
4. **The sign of the error depends on the family; a seed offset dominates only
   for the single notch.**
   - Under-predicted on every seed: the dog-bone (6.7), the hole, the inclusion.
   - Over-predicted on every seed: the double notch.
   - The single notch's bias swings from +7.9% to −7.8% between seeds. There,
     between-seed variance is 54% of the signed-error variance.
   - Elsewhere per-geometry error dominates. The between-seed share is 0.03
     (double notch), 0.12 (inclusion), 0.17 (hole) and 0.19 (dog-bone). On the
     double notch the per-geometry errors correlate across seeds at
     r = 0.48–0.59: the same geometries are hard for every seed.
5. **Out of range, it depends on where the direction takes `K_t`.**
   - **Hole:** out-of-range `K_t` 4.54–4.89, above the in-range maximum of
     4.01. The error is 12.3%, too low on every seed: the operator does not
     follow the rise.
   - **Double notch:** out-of-range `K_t` 5.37–6.62, above 4.84. The 17% is
     mostly *spurious peaks*.
     - On 2–3 of 8 geometries per seed, the network's maximum sits on the
       notch flank, where the FEM stress is low. Those rows are +38 to +54%.
     - Where the peaks coincide, the error is 7–8%.
   - **Single notch:** 5 of 8 out-of-range `K_t` values fall inside the
     in-range span, yet those still average about 11% error. Extrapolating in
     `t/W` fails even when `K_t` itself does not leave its range.
   - **Inclusion:** out-of-range `K_t` 1.36–1.38, *below* the in-range span.
     Error 1.8%.
   - **The dog-bone re-test (6.7):** out-of-range `K_t` inside the in-range
     span, error 3.7%.
   - So the two families without a cliff are the two whose out-of-range `K_t`
     stays inside or below the trained span. This design cannot say whether
     extrapolation fails generally, or only when `K_t` leaves its trained span.
6. **The free edge is less traction-free at holes and notches, but that does
   not explain the `K_t` errors** (`docs/phase8_tier1_residuals.json`,
   descriptive).

   | family | RMS \|P·N\|/σ_nom on the feature | at the peak, vm/\|σ_tt\| − 1 (seed means) |
   |---|---|---|
   | dog-bone (Phase 7; ≈ 4.7–5.4% in these units) | 2.2–2.5% of the gauge-mean vm | −0.4 to −0.9% |
   | open hole | 7.4–18.4% | −0.2 to −1.0% |
   | double notch (flank + root) | 21.0–26.7% | +0.3 to −4.5% |
   | single notch (flank + root) | 21.7–32.8% | −0.5 to −3.4% |

   - **The dog-bone row uses a different nominal.** Phase 7 normalised by the
     gauge-mean von Mises; converted to `N/A_gross` it is about 4.7–5.4%.
   - **On a like-for-like basis** the hole is 1.4–3.9× the dog-bone and the
     notches 4–7×. Relative to the local peak stress: about 1.9% on the
     dog-bone, 5–7.5% on the notches.
   - **Per geometry, the peak effect ranges from −11.7% to +9.5%.**
   - **The sign does not track the `K_t` bias.** On the double notch the peak
     effect is negative while the bias is positive on every seed, and
     Spearman(δ, err) is −0.14 to −0.35 (not significant).
   - **This script was not blind.** It was committed after four models had
     finished.
7. **The transverse displacement suffers most on the notches.** `v` L2 is 0.10
   on the double notch and 0.057 on the single notch, against 0.016–0.019 on
   the dog-bone, hole and inclusion.

**What P-T4 does and does not establish.**
- **What it does:** the family loop's optimiser, schedule and batching reproduce
  the dog-bone's `K_t` result.
- **What it does not test:**
  - the distance-function layer or the family quadrature mesh — the dog-bone
    path keeps the shipped Dirichlet layer and the standard energy loss. Those
    are verified separately, at Level 0 and Level 3.
  - displacement accuracy. The loop run's `u` and `v` L2 (1.8e-3, 3.0e-2) sit
    above the 6.7 seeds' (1.1–1.4e-3, 1.5–1.8e-2). With one run, loop cannot be
    told from seed.

### What this means for the paper

- **The per-family table is the multi-geometry validation the paper needed.**
  Every entry is against an independent, Level-3-verified FEM reference, on
  held-out geometries, over three seeds.
- **The claim it supports:** in-range stress-concentration prediction with skill
  on four of five families (the dog-bone, hole, double notch and single notch),
  at 1.5–6.8% error, beating the baseline by a factor of 1.6–3.
- **The claims it rules out:**
  - general extrapolation (hole and notch families);
  - a consistent sign of the error;
  - "skill degrades with concentration sharpness" — within a family it
    doesn't, and relative to the baseline it doesn't either.
- **Where the next gain is.** Neither of these is the Phase 7 free-edge story
  on its own:
  - on the rigid inclusion, the field next to the curved Dirichlet boundary
    (6–8% low at the true peak, maximum in the wrong place);
  - on the double notch, spurious flank peaks out of range.

  Ablation A-ADF and a look at the notch-flank field are the cheapest next
  measurements.

**Changed by the independent check (8.14):**
- single-notch seed-2 bias −7.8% (was printed −7.9%);
- the `v` L2 comparison;
- the extrapolation reading (point 5);
- the seed-offset reading (point 4);
- the inclusion reading (point 2, previously "the layer works; it is signal
  size");
- the free-edge comparison (point 6, previously "an order of magnitude" and
  "the most promising lever");
- P-T4's scope;
- "degrading with sharpness" withdrawn.

No verdict changed.
