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

---

## Confirmation across the taper range

The single-geometry result above is one specimen. Repeating it on six more,
spanning the full taper range of the parameter space, varying **only**
`w_trac_top` (2 → 100) so the effect is attributable to one knob. 200 epochs;
all runs plateaued.

Final errors against the reference:

| geo | taper | run | u rel L2 | v rel L2 | vm rel L2 | σ₂₂/peak | N err |
|---|---|---|---|---|---|---|---|
| 19 | 0.27 | shipped | 9.38e-02 | 7.53e-01 | 4.14e-01 | 9.10e-02 | +22.0% |
| 19 | 0.27 | wtop=100 | 9.33e-02 | 3.55e-01 | 3.98e-01 | 5.07e-02 | +24.4% |
| 2 | 0.32 | shipped | 7.29e-02 | 5.54e-01 | 3.31e-01 | 7.20e-02 | +15.2% |
| 2 | 0.32 | wtop=100 | 7.21e-02 | 3.03e-01 | 3.21e-01 | 4.58e-02 | +16.7% |
| 13 | 0.43 | shipped | 5.52e-02 | 3.37e-01 | 2.34e-01 | 5.50e-02 | +8.3% |
| 13 | 0.43 | wtop=100 | 5.51e-02 | 2.18e-01 | 2.30e-01 | 4.09e-02 | +9.3% |
| 1 | 0.50 | shipped | 5.82e-02 | 2.97e-01 | 2.18e-01 | 6.01e-02 | +7.1% |
| 1 | 0.50 | wtop=100 | 5.80e-02 | 2.02e-01 | 2.13e-01 | 4.44e-02 | +8.2% |
| 4 | 0.60 | shipped | 4.66e-02 | 2.42e-01 | 1.69e-01 | 4.97e-02 | +4.4% |
| 4 | 0.60 | wtop=100 | 4.66e-02 | 1.95e-01 | 1.67e-01 | 3.84e-02 | +5.2% |
| 11 | 0.78 | shipped | 1.78e-02 | 1.49e-01 | 7.31e-02 | 2.49e-02 | +0.8% |
| 11 | 0.78 | wtop=100 | 1.78e-02 | 1.42e-01 | 7.26e-02 | 2.21e-02 | +1.0% |

Ratio of the reweighted error to the shipped error — below 1 is better:

| geo | taper | v | σ₂₂ | von Mises | u |
|---|---|---|---|---|---|
| 19 | 0.27 | **0.471** | 0.557 | 0.962 | 0.995 |
| 2 | 0.32 | 0.547 | 0.636 | 0.969 | 0.989 |
| 13 | 0.43 | 0.648 | 0.744 | 0.983 | 0.998 |
| 1 | 0.50 | 0.681 | 0.737 | 0.979 | 0.998 |
| 4 | 0.60 | 0.806 | 0.774 | 0.984 | 1.000 |
| 11 | 0.78 | 0.952 | 0.885 | 0.994 | 0.999 |
| **mean** | | **0.684** | **0.722** | 0.979 | 0.997 |

Four things worth reading off this:

1. **It helps on every geometry, and never hurts.** v improves on all six,
   σ₂₂ on all six.
2. **The benefit scales with the failure.** The improvement is largest exactly
   where the operator is worst (0.471 at taper 0.27) and fades to nothing
   where it is already accurate (0.952 at 0.78). A lever that targets the
   failure mode rather than shifting error around.
3. **The axial field is untouched** — u ratio 0.997 on average, within noise
   on every geometry.
4. **The section force degrades by about one percentage point** consistently
   (+22.0% → +24.4%, +15.2% → +16.7%, …). Small but systematic, and it is the
   competition noted above showing up again.

Note the single-geometry run earlier changed *two* weights and reached a v
ratio of 0.426 on geometry 0 (taper 0.28); the comparable single-knob run here
reaches 0.471 at taper 0.27. So `w_trac_top` accounts for most of the effect
and `w_traction_partial` adds a little.

The mean v improvement across the range is **1.46×**, not the 2.4× the worst
single geometry suggested — because the mildly tapered specimens, where there
is little to gain, are included.

---

## What this does *not* establish

- Single-geometry training is not operator training. All of this removes the
  128-geometry allocation problem by construction; the multi-geometry dynamics
  could differ.
- Single-geometry training is not operator training. The multi-geometry error
  (168%) is worse than this plateau (89%), so allocation does contribute — it
  is just not what sets the floor.
- 300 epochs at 1200 collocation points is a diagnostic budget, not a
  production one.

## Recommendation

`w_trac_top` is **still not changed** in `config.py`. The evidence is now seven
geometries spanning the full taper range rather than one, but it is all
single-geometry training, and a training-wide hyperparameter is the kind of
decision that should be made deliberately rather than inherited from a
diagnostic. Changing it would also silently break comparability with every
number reported so far.

It is recorded next to the value in `config.py` and belongs in the ablation
matrix as its own row, alongside `resultant_anchor` and
`boundary_measure_consistent` — all three documented, ablatable, and off by
default, so that Phase 0 and Phase 1 change what is *measured* and not what is
trained.

**The recommendation itself is strong.** A one-line configuration change cuts
the transverse displacement error by 1.46× on average and 2.1× where the
operator is weakest, at no cost to axial accuracy and about one percentage
point on the section force. That is the cheapest improvement available to the
project, and it should be measured on a retrained model before the far larger
Phase 2 rewrite is undertaken.
