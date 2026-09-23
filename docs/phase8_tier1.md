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

Each run took 65–70 min on one thread, two lanes in parallel, for about 8 h of
wall clock. Scores are in `docs/phase8_tier1_scores.json`, and
`python -m verification.family_score --report` reproduces every verdict.

### Verdicts

| | prediction | result | verdict |
|---|---|---|---|
| **P-T1** | every new family beats the constant in range (≥ 2 of 3 seeds) | open hole 3/3, double notch 3/3, single notch 3/3, **rigid inclusion 1/3** | **FAILS** (holds for 3 of 4 families) |
| **P-T2** | free-edge peaks under-predicted on every seed | open hole 3/3 · **double notch 0/3** (over-predicted every seed) · **single notch 1/3** (+0.2, +7.9, −7.9%) | **FAILS** (holds for the open hole only) |
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

In-range reference `K_t`:
- dog-bone 1.16–1.30 (gauge definition);
- open hole 3.12–4.01;
- inclusion 1.41–1.50;
- double notch 3.08–4.84;
- single notch 3.30–5.51.

### What the numbers say

1. **The method carries over to new geometry families, in range.**
   - Four of the five families beat the predict-the-mean baseline on every
     seed, the dog-bone included.
   - Errors are 1.5–7%, against baselines of 2.4–15.9%.
   - The loop check (P-T4) shows the rows are comparable: the same loop gives
     the dog-bone the same result as before.
2. **The rigid inclusion is accurate but shows no skill.**
   - Its reference `K_t` hardly varies over the range (1.41–1.50), so the
     baseline error is only 1.6%.
   - The operator's 1.75% error is a consistent under-prediction (−1.1, −2.0,
     −1.9%) rather than scatter.
   - In absolute terms it is as accurate as the dog-bone, and it holds up out
     of range (1.8%). That is the only new family where it does.
   - This is the first family with its peak on a curved Dirichlet boundary
     imposed by the distance-function layer. That layer works; the missing
     skill is about signal size, not boundary conditions.
3. **Error grows with the concentration, across families but not within one.**
   - In-range error rises roughly with the family's `K_t` level: 1.5% (dog-bone,
     ~1.2), 1.8% (inclusion, ~1.45), 3.4% (hole, ~3.5), 4.1% (double notch,
     ~4), 6.8% (single notch, ~4.5).
   - Within the double notch, sharper roots are not worse (P-T3 fails).
4. **The sign of the error is family- and seed-dependent.**
   - The dog-bone, the hole and the inclusion are under-predicted on every seed.
   - The double notch is over-predicted on every seed.
   - The single notch's bias swings from +7.9% to −7.9% between seeds.
   - A seed-level offset of the whole peak dominates the per-geometry scatter.
     This is the same seed sensitivity Phase 6 measured, larger on sharper
     families.
5. **Extrapolation fails on the hole and notch families.**
   - Out-of-range errors are 12–17% (open hole: too low on every seed).
   - The dog-bone re-test found no such cliff (3.7%), and the inclusion has
     none (1.8%).
   - Each family's out-of-range direction pushes `K_t` beyond anything in its
     training set, and the operator does not follow the steep rise.
   - Phase 6.7's "no extrapolation cliff" was a dog-bone result, not a general
     one.
6. **The free edge is much less traction-free at holes and notches**
   (`docs/phase8_tier1_residuals.json`, descriptive).

   | family | RMS \|P·N\|/σ_nom on the feature | at the peak, vm/\|σ_tt\| − 1 |
   |---|---|---|
   | dog-bone (Phase 7, arc) | ~2.3% | −0.4 to −0.9% |
   | open hole | 7.4–18.4% | −0.2 to −1.0% |
   | double notch | 21.0–26.7% | +0.3 to −4.5% |
   | single notch | 21.7–32.8% | −0.5 to −3.4% |

   - Phase 7's weakness of the energy form grows by an order of magnitude on
     sharp features.
   - At the peak point itself it moves von Mises by up to 4.5%, of the same
     order as the notch families' `K_t` errors.
7. **The transverse displacement suffers most on the double notch** (`v` L2
   0.10, against 0.016–0.019 elsewhere).

### What this means for the paper

- **The per-family table above is the multi-geometry validation the paper
  needed.** Every entry is against an independent, Level-3-verified FEM
  reference, on held-out geometries, over three seeds.
- **The claim it supports:** in-range stress-concentration prediction with skill
  on four of five families, at 1.5–7% error, degrading with concentration
  sharpness.
- **The claims it rules out:**
  - "extrapolates beyond the training range" (for holes and notches);
  - a consistent sign of the error.
- **The most promising lever is again the free edge.**
  - The traction residual at notch roots is 10× the dog-bone's.
  - Phase 7's C-series (a traction term on the arc, exact tractions through φ)
    now has much more to gain on these families than on the dog-bone.
