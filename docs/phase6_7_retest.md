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

Six models, three seeds per arm, scored on the 12 in-range and 8 out-of-range
geometries. `python -m verification.retest --report` reproduces every number.

| arm | IR `K_t` R² | IR \|`K_t`\| | OOR \|`K_t`\| | IR `v` |
|---|---|---|---|---|
| **energy** | **+0.61 ± 0.22** | **1.51 ± 0.49%** | 3.67 ± 3.06% | **0.018** |
| mixed + energy, from `u` | −0.18 ± 0.87 | 2.97 ± 1.21% | 3.16 ± 1.35% | 0.036 |
| mixed + energy, from `φ` | +0.25 ± 0.41 | 2.27 ± 0.69% | **1.15 ± 0.41%** | — |
| predict the bank mean | 0.00 | 2.43% | — | — |

Per seed: energy IR R² +0.85 / +0.41 / +0.57; mixed + energy +0.56 / −1.14 / +0.06.
Energy OOR error 2.10 / 1.72 / 7.20%.

## Verdicts

| | claim | verdict |
|---|---|---|
| **P1** | energy has strong in-range skill | **HOLDS** — mean R² +0.61 (threshold +0.60), below the constant baseline on 3/3 seeds |
| **P2** | mixed + energy adds nothing detectable in range | **as predicted** (p = 0.256) — but see below |
| **P3** | `φ` extrapolates better | **FAILS** — not under half of energy's error on seed 1 (1.31% vs 1.72%) |
| **P4** | the gap flags failure | **FAILS** — seed 1 gives ratio 1.55 and Spearman −0.33 |
| **P5** | the extrapolation failure replicates | **FAILS** — ratios 2.2 / 0.9 / 4.4, needed > 3 on every seed |

**Three of the four findings Phase 6.21 drew from a post-hoc split do not
survive fresh geometries and three seeds.**

### P1 — holds, and it is the one that matters

The correction to Phase 6.1 is confirmed on geometries that had no part in
suggesting it: the shipped parameter-conditioned energy configuration predicts
the concentration factor with R² +0.61 ± 0.22 and 1.51% error against a
constant's 2.43%, on every seed. It passes its threshold by 0.01, so it is a
pass, not a comfortable one — the seed spread (+0.41 to +0.85) is wide relative
to the effect.

### P5 — the extrapolation cliff was two geometries, not a property

On fresh wide-gauge specimens the energy arm's out-of-range error is 2.10%,
1.72% and 7.20% — mean 3.67%, sd 3.06%, and on one seed *lower* than its
in-range error. Phase 6.21's 6.4% (trainer loop) and 19.6% (pilot loop) came
from two specific geometries, and both of those had every raw parameter inside
the training box, so they are not a harder class of extrapolation — they are two
hard specimens.

### P4 — the gap is not a reliable indicator

Seed 0 was convincing (ratio 2.82, Spearman +0.76); seed 1 reverses the
correlation (−0.33). Pooled over all three seeds and 60 points the correlation
is +0.38 at p = 0.0025, so *something* is there, but not the per-seed signal
the claim needed.

### P3 — the weaker version survives, the stated one does not

As written — under half of the energy arm's error on every paired seed — it
fails on seed 1. What does hold on every seed is that **`φ` beats the mixed
model's own displacement-derived stress out of range**: 0.68 vs 4.30, 1.31 vs
1.67, 1.45 vs 3.52. And on the means `φ` is 3.2× better than the energy arm out
of range (1.15% vs 3.67%), though at n = 3 with these variances that is
p = 0.14. A real effect at this sample size cannot be separated from a lucky
one.

### P2 — technically as predicted, and misleading

The Welch test says no detectable difference (p = 0.256), but only because
mixed + energy is wildly inconsistent: IR R² +0.56, −1.14, +0.06, sd 0.87
against the energy arm's 0.22. Its mean is far below. The honest statement is
**not** "no difference" but "in the trainer loop, mixed + energy is unstable,
and nothing suggests it is better". Its in-range `v` is also 2× worse
(0.036 vs 0.018, p = 0.10), consistent with the representability protocol.

**So the three-seed representability gain (+0.24 → +0.72) does not transfer to
the trainer loop on held-out geometries.** Both the loop and the protocol
changed at once, so which one is responsible is untested.

## What survives from Phase 6

- The energy form's in-range skill on the concentration factor (P1), and with
  it the correction to the Phase 6.1 headline.
- The representability-protocol results, which were measured, replicated over
  three seeds, and are not contradicted by this: they say what happens when a
  model is trained on the geometries it is scored on.
- The construction itself — equilibrium, traction-free edges and constant
  section force exact to 1e-13 — which is verified independently of any
  training outcome.

## What does not

- "Extrapolation is the open problem" — not on fresh geometries.
- "The equilibrated field extrapolates better" — as a claim at three seeds, no;
  as a within-model comparison, yes.
- "The gap flags failures" — not per seed.
- Any suggestion that mixed + energy improves the operator. On the protocol
  that matters it does not, and it is unstable.

## Lesson

Phase 6.21's four claims came from one seed, one loop and a post-hoc split of
eight geometries. Three did not survive. The cost of finding that out was about
nine hours of compute; the cost of not finding out would have been a paper
built on them.
