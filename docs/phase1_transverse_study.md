# Why the transverse field is not learned

Follow-up to `docs/phase1_case14_reference.md`, which established *that* the
operator learns the axial response and not the transverse one (v carries a
168% relative L2 error against the FEM reference; σ₂₂ is 30% of the peak
stress on faces where it should be zero).

Reproduce:

```
python -m verification.transverse_study --geo 0 --runs shipped,reweighted,fresh --epochs 300
```

---

## The three candidate explanations

They imply completely different fixes, so separating them matters more than
any single accuracy number.

**A — optimisation / allocation across geometries.** The loss *can* drive v:
the level-2 driver reached 2.5e-04 on a prismatic bar in 4000 epochs. But in
the 128-geometry operator setting each specimen is seen only about 94 times,
and the transverse correction may simply lose to the axial one.
*Fix: more training, or rebalancing across geometries.*

**B — loss weighting.** The traction-free condition on the gauge top is the
**only** term that sets the lateral contraction there: with N = (0,1),
`P·N = 0` means `P₂₂ = P₁₂ = 0`. Its weight is `w_trac_top = 2.0` — the lowest
in the config, and 50× below `w_equilibrium = 100`.
*Fix: reweight.*

**C — architecture.** The hard boundary condition is `v = (y/H_grip)·φ_v`. A
constant φ_v gives v linear in y and independent of x, which is right in the
gauge but wrong across the taper, where the true field is `v ≈ −ν·ε(x)·y`.
*Fix: change the ansatz.*

## The experiment

Fine-tuning on a **single** geometry removes the multi-geometry allocation
problem entirely. If v converges with the shipped weights, the cause is A; if
only after reweighting, B; if neither, C. A fresh initialisation on the same
geometry is the control — it distinguishes "stuck in a basin inherited from
the checkpoint" from "this is where the loss goes".

Validation geometry 0, the worst in case 1.4: taper `W_gauge/W_grip = 0.28`.
300 epochs, Adam 3e-4, 1200 interior collocation points. `PhysicsLoss` only —
the section-resultant anchor is a trainer-level term and is not in this loop,
so the loss here is equilibrium + tractions + barrier, whose unique minimiser
is the true solution.

## Result

| run | u rel L2 | **v rel L2** | vm rel L2 | **σ₂₂/peak** | N error |
|---|---|---|---|---|---|
| start (shipped checkpoint) | 3.02e-02 | 5.40 | 6.99e-01 | 8.16e-01 | −23.9% |
| shipped weights, 300 ep | 5.33e-02 | 8.93e-01 | 3.20e-01 | 1.06e-01 | +10.2% |
| **reweighted** (`w_trac_top` 2→100, `w_traction_partial` 2→20) | 5.45e-02 | **3.80e-01** | 3.07e-01 | **4.45e-02** | +13.3% |
| fresh init, shipped weights | 5.82e-02 | 9.32e-01 | 3.37e-01 | 1.11e-01 | +10.9% |

All three plateaued: the loss moved by under 3% over the final 150 epochs.

### What it says

**It is not an allocation problem (A is not the main cause).** A fresh network
trained on this one geometry alone reaches v = 0.932 — statistically the same
as fine-tuning the checkpoint to v = 0.893. Removing the other 127 geometries
and the shared encoder budget does not fix the transverse field, and the
plateau is not a basin inherited from the checkpoint.

**Loss weighting is the largest single lever found (B is a real cause).**
Raising the gauge-top traction weight from 2 to 100 cuts the transverse
displacement error **2.4×** (0.893 → 0.380) and σ₂₂ **2.4×** (0.106 → 0.044),
with no cost in axial accuracy. This is the term that sets lateral contraction,
and it was the lowest-weighted term in the configuration — deliberately, per
the config comment: *"Reduced: equilibrium + resultant should drive learning."*
That reduction is what starves the transverse field.

**Reweighting alone does not close it (C is still live).** 38% v error remains
at a plateau. Something beyond weighting limits it — most plausibly the
conditioning of the penalty formulation, which is precisely what the mixed
u–P form and adaptive loss balancing in Phase 2 exist to address.

**Axial and transverse accuracy are in competition under this loss.**
Fine-tuning improves v from 5.40 to 0.89 but degrades u from 0.030 to 0.053
and swings the section force from −23.9% to +10.2%. A formulation where these
do not trade against each other is the real goal, not a better weight.

## What this does *not* establish

- One geometry, the most strongly tapered one. The lever should be confirmed
  across the taper range before the default changes.
- Single-geometry training is not operator training. The multi-geometry error
  (168%) is worse than this plateau (89%), so allocation does contribute — it
  is just not what sets the floor.
- 300 epochs at 1200 collocation points is a diagnostic budget, not a
  production one.

## Recommendation

`w_trac_top` is **not** changed in `config.py` on this evidence — one geometry
is too narrow a basis for a hyperparameter with training-wide effects. It
belongs in the ablation matrix as its own row, alongside `resultant_anchor`
and `boundary_measure_consistent`, and it should be swept across the taper
range on a retrained model.

If the 2.4× reduction holds across geometries, this is the cheapest
improvement available to the project — a one-line configuration change that
addresses the documented weakness directly, and it should be measured before
the far larger Phase 2 rewrite is undertaken.
