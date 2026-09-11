# GPU confirmation run

Everything measured in this project ran on **2 CPU cores**, at bank ≤64, ≤1600
epochs, scored on eight held-out geometries. The findings are consistent and
the equal-budget comparison has error bars, but nothing has run at the scale a
result should be stated at. This is the run that closes that, and it needs a
GPU.

## What to run

The winning configuration is now reachable from the production entry point —
it was previously only in the verification harness, which is not where a
training job should be launched from.

```bash
# the configuration under test
python main.py --mode train --conditioning parameters --loss_form energy \
    --epochs 4000 --device cuda --seed 0

# the baseline it is being compared against
python main.py --mode train --conditioning pointcloud --loss_form energy \
    --epochs 4000 --device cuda --seed 0
```

Repeat both for `--seed 1` and `--seed 2` at minimum. `config.py` already
carries `bank_train_size: 128`; the container runs used 64 because of memory,
so the GPU run is also the first at full bank.

Then score each checkpoint against the finite-element reference:

```bash
python -m eval.compare_reference --checkpoint <path>
```

### Save somewhere else

`main.py` defaults to `--save_dir checkpoints`, which is **tracked in git and
holds the original pre-Phase-0 `best.pt`** that every Phase 0/1 number was
measured against. A training run overwrites it. Pass an explicit directory:

```bash
python main.py ... --save_dir runs/params_seed0
```

This is not hypothetical — a two-epoch smoke test of the command above
overwrote `checkpoints/best.pt` and wrote ~160 files into the working tree
while this document was being written. It was restored from git.

## What the container measured, to compare against

Bank 64, energy form, eight held-out geometries spanning taper 0.31–0.82, on
the fixed bank draw.

**Equal budget, 400 epochs, three seeds** — the comparison the conclusion rests
on:

| ratio, point-cloud / parameters | mean ± sd |
|---|---|
| `u` | 3.58 ± 0.31 |
| `v` | **3.32 ± 0.01** |
| von Mises | 2.25 ± 0.14 |
| von Mises, fillet | 2.19 ± 0.12 |

**Absolute values**, both budgets, single seed at 1600:

| | `u` | `v` | vm | vm fillet | \|N err\| |
|---|---|---|---|---|---|
| point-cloud, 400 ep | 1.431e-02 | 2.395e-01 | 6.101e-02 | 9.075e-02 | 2.30% |
| point-cloud, 1600 ep | 9.12e-03 | 1.51e-01 | 3.65e-02 | 5.91e-02 | 0.84% |
| parameters, 400 ep | 4.228e-03 | 7.221e-02 | 2.655e-02 | 4.086e-02 | 0.56% |
| **parameters, 1600 ep** | **2.28e-03** | **2.85e-02** | **1.38e-02** | **2.16e-02** | **0.56%** |

## What would confirm, and what would refute

**Confirms:** the parameter-conditioned arm beats the point-cloud arm on `u`,
`v` and von Mises at equal budget, by a ratio within roughly a factor of two of
the container figures, on at least three seeds at bank 128.

**Refutes, or at least complicates:**

- The ratio collapses toward 1 at bank 128. The container measured bank 16 →
  64 improving the parameter-conditioned arm 3.3× on `v` while barely moving
  the point-cloud arm; if that asymmetry reverses with more data, the
  conclusion is bank-size-dependent and the recommendation changes.
- The point-cloud arm keeps improving past 1600 epochs while the
  parameter-conditioned one flattens. The container saw the opposite (1.58×
  against 2.53× for 4× the epochs), but both were still improving at 1600, so
  neither is converged.
- Seed spread at bank 128 swamps the ratio. At bank 64 the ratio was stable to
  ±0.01 across seeds *because the seed noise is common-mode between arms* —
  same geometries, same order. That should hold at 128, and if it does not it
  is worth understanding why before trusting either number.

## Two things to watch

**`v` was still falling** at 1600 epochs in the parameter-conditioned arm, so
4000 is a floor rather than a generous budget. Watch whether it flattens.

**Memory at bank 128 under the energy form.** Each geometry carries its own
cached quadrature mesh (`energy_n_elem: 1400` triangles, 3 stratified points
each). The container peaked around 1.5 GB at bank 64 with batch 4; bank 128
holds twice the meshes. If it is tight, `energy_n_elem` is the knob — the
integration error at 1400 triangles was 4.6e-05 against a 4× finer mesh, so
there is room.

## What this does not settle

The **parametric-vs-general fork**. Parameter conditioning is valid only for a
family whose shapes are four known numbers. Phase 4 ruled out repairing the
point-cloud encoder in place (sampling resolution, a supervised geometry head,
and both together each buy ≤1.21×), so generality beyond this family needs a
different conditioning architecture — and this run does not bear on that
question either way.
