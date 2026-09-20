# Phase 6.5 — the hybrid collapses to uniform strain

**Status.** Run, decisive, negative. One arm at `w_eq = 100`, 500 matched
epochs, bank 64, scored against the three-seed baseline from Phase 6.2.

## What was tried and why

Phase 6.4 showed the objective — not the architecture — is what fails to
deliver the stress concentration. The architecture reaches `K_t` R² **+0.98**
under stress supervision; a 0.2% displacement fit on the specimens it is
scored on gives **−0.55**.

So the objective needs a term whose gradient is sensitive to stress. Without
changing the architecture, the physics offers exactly one: the equilibrium
residual. `loss_form` had always been `energy` XOR `penalty` — Phase 2
compared them, the energy form won, and `L_eq` left with the form that
carried it. Nobody had run the sum.

## Result

| metric | baseline (3 seeds) | hybrid, `w_eq` 100 | ratio | seed CV |
|---|---|---|---|---|
| `u` | 1.139e-2 | 4.292e-2 | **3.77×** | 10.6% |
| `v` | 1.770e-1 | 2.429e-1 | 1.37× | 12.7% |
| `vm` | 4.617e-2 | 1.788e-1 | **3.87×** | 1.6% |
| `vm` fillet | 6.964e-2 | 3.001e-1 | **4.31×** | 7.3% |
| peak error | 10.28% | 21.56% | 2.10× | 14.2% |
| fillet L∞ | 2.330e-1 | 7.799e-1 | **3.35×** | 9.3% |
| `K_t` error | 10.44% | 18.13% | 1.74× | 11.6% |
| \|N err\| | 1.00% | 7.34% | **7.34×** | 28.7% |

Worse on every metric, most of them far outside the seed noise. But the
ratios are not the finding. This is:

**Predicted `K_t`, all eight held-out geometries:**

```
1.000015  1.000024  1.000022  1.000025  1.000024  1.000025  1.000036  1.000031
```

A concentration factor of 1.000 means the peak equals the gauge mean — **the
predicted stress field is uniform to five decimal places.** And the peak's
location confirms it:

| | geo 2 | 15 | 4 | 6 | 3 | 11 | 9 | 22 |
|---|---|---|---|---|---|---|---|---|
| reference `x*/x_g` | 1.107 | 1.085 | 1.084 | 1.073 | 1.105 | 1.075 | 1.034 | 1.035 |
| hybrid `x*/x_g` | 0.069 | 0.034 | 0.019 | 0.009 | 0.015 | 0.027 | 0.011 | 0.031 |

The reference peak is in the fillet, just past tangency. The hybrid's "peak"
is at the symmetry plane — there is no peak, and `argmax` is picking up
rounding noise near `x = 0`.

## The mechanism

**A uniform stress field is an exact null of `Div P = 0`.** Adding `L_eq` at a
weight that matters hands the optimiser a shortcut: flatten the field, and the
residual goes to zero without any of the physics being right. The energy term
alone did not pull it back out within 500 epochs.

The training log shows it happening. At epoch 1 the equilibrium term was
**94%** of the objective (`loss` 1.14e-2 against `Pi/(E|Ω|)` 7.28e-4). At the
end it was **0.0007%** (`loss` 7.2463e-4 against `Pi` 7.2458e-4). `L_eq` was
driven to zero — not by finding the equilibrated solution, but by removing the
gradients that make equilibrium non-trivial.

This is also why the penalty form carries the scaffolding it does.
`config.py` sets `w_equilibrium: 100.0` with the comment *"must dominate to
break uniform-strain baseline"*, alongside a section-resultant anchor and
traction terms. Those exist to stop exactly this collapse. The hybrid replaced
them with the energy functional, and over this budget the energy lost.

## What it means for the plan

**You cannot buy stress sensitivity with a residual term, because a residual
is satisfied by the trivial solution.** That is a structural property of
`Div P = 0`, not an artifact of this weight or this budget.

Mixed `u`–`P` (Step 4) differs in kind, and this result is now a measured
argument for it rather than a stylistic preference. There, `P` is a *primary
output* and the constitutive residual `‖P − P(F)‖` ties it to the deformation.
Flattening `P` then costs something: it breaks the constitutive tie. The
degenerate direction this arm fell into is closed by construction.

## The pincer, and the arm that would close it

- At `w_eq = 100` the term is 94% of the objective at epoch 1 and the field
  collapses.
- At `w_eq = 1` the term is **0.23%** of the objective — an order of magnitude
  below the 9–14% seed noise on the concentration metrics, so it cannot be
  measured even if it helps.

**There may be no weight that both matters and does not collapse.** One arm at
`w_eq = 10` (≈2.3% of the objective) would test it. It is not run here: the
mechanism is structural, one measurement demonstrates it, and 80 minutes is
better spent on Step 4. Recorded as a conjecture with its test named, not as a
conclusion.

## Caveats

1. One seed, one weight, 500 epochs. The collapse is far outside the measured
   noise, but "the energy eventually pulls it back out at a longer budget" is
   not excluded — only shown not to happen by epoch 500.
2. The arm is scored at `h_factor` 2.5 like the rest of the Phase 6.2 study,
   so it is comparable to that baseline. With a uniform predicted field the
   reference mesh makes no difference to the conclusion.
