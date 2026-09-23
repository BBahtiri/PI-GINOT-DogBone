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
