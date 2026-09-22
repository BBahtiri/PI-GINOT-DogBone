# Phase 6.7 — the clean re-test, pre-registered

Committed before any model is scored on the sets below, and before any of the
five new models has finished training.

## Why

Phase 6.21 made four claims from a post-hoc split of eight geometries, one
seed, in the pilot loop:

1. the shipped energy configuration predicts `K_t` well **in range** (+0.93);
2. mixed + energy may or may not add to that in range;
3. the exactly-equilibrated `φ` field **extrapolates** far better (2.2% vs
   11–20%);
4. the constitutive gap **flags** failures without a reference.

Each rests on geometries that also suggested it. This test uses geometries that
suggested nothing.

## Design

**Sets** — `verification/heldout_sets.py`, seeds 300/301, never used before:

- **In-range (IR), 12 geometries.** Every raw parameter inside the training
  bank's drawn min/max; taper in [0.30, 0.70], a margin inside the bank's
  0.261–0.747; stratified four per third of the band. (A first unstratified draw
  landed 0.308–0.588 and left the upper third empty; changed before any scoring.)
- **Out-of-range (OOR), 8 geometries.** Every raw parameter inside its training
  range; taper ≥ 0.78 (drawn 0.784–0.846) — beyond the bank's 0.747, so
  extrapolation in the joint geometry and in no single coordinate.

Why the corner was missed: under the project's uniform sampler P(taper > 0.75) is
2.9%, so a 64-bank expects 1.9 such geometries; this one drew none.

**Models** — oracle-conditioned, **trainer loop** (the loop that produced the
+0.93), bank 64, 1600 epochs, batch 4, lr 3e-4, `n_interior` 1500:

| arm | seed 0 | seeds 1, 2 |
|---|---|---|
| energy | `fixed_long/oracle-energy`, reused — the +0.93 run, identical settings | new |
| mixed + energy (`w_E` 100) | new | new |

Mixed + energy has never completed a trainer-loop run. The energy arm uses the
standard decoder and the mixed arm the mixed one, so initialisations differ; with
three seeds per arm that is seed variance, not a bias.

**References** — FEM at `h_factor` 1.3, solved and cached before scoring.

## Predictions and falsification

| | claim | holds if | falsified if |
|---|---|---|---|
| **P1** | energy has strong in-range skill | mean IR `K_t` R² ≥ +0.60 **and** IR \|K_t\| error below the constant baseline on every seed | either condition fails |
| **P2** | mixed + energy adds nothing detectable in range | seed-level Welch two-sided p > 0.05 on IR R² | p < 0.05 either way — reported with its sign |
| **P3** | `φ` extrapolates better | OOR \|K_t\| error from `φ` < ½ of energy's on every paired seed, **and** < mixed + energy's own displacement-derived error on every seed | either fails on any seed |
| **P4** | the gap flags failure | mean gap OOR/IR ≥ 2 on every seed, Spearman(gap, \|K_t err\|) > 0 on every seed, pooled p < 0.01 | any condition fails |
| **P5** | the extrapolation failure replicates | energy OOR \|K_t\| error > 3× its IR error on every seed | fails on any seed |

P2 is a prediction of no difference, and says so. The reason: in this loop the
energy arm is already near the supervised ceiling in range, and the
representability gain for mixed + energy (+0.24 → +0.72) was measured in a loop
whose energy baseline was much weaker.

Verdicts are computed by `python -m verification.retest --report` exactly as
coded at this commit.

## Results
