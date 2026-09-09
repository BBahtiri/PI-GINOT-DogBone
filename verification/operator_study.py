#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the energy form work as an *operator*, not just per geometry?

Everything measured so far -- Phase 1's transverse study and Phase 2's energy
comparison -- fine-tunes on a single specimen.  That isolates the physics but
sidesteps the question the project actually exists to answer: can one network
carry the whole geometry family, and does the energy form still win when its
capacity is shared across a bank rather than devoted to one shape?

This trains through the **production trainer** (``PI_GINOT_Trainer``) rather
than a bespoke loop, so what is measured is the path that would run on a GPU,
then evaluates on validation-bank geometries the network never saw -- the
train and validation banks are drawn from different seeds, so they are
genuinely disjoint.

Equal wall-clock, not equal epochs
----------------------------------
The energy form is 3-5x cheaper per epoch, so equal epochs would understate
it and equal wall-clock is the comparison a practitioner actually faces:
*given an hour, which formulation gives the better operator?*  Both budgets
and the epochs each reached are reported, so either reading is available.

What "held out" has to mean here
-------------------------------
The trainer draws its own validation bank from the same seed and steps the
LR scheduler on its loss, so those geometries are not held out even though
no gradient reaches them.  The scoring set is therefore taken from beyond
that prefix, and the study refuses to run if the two overlap -- a guard
rather than a caveat.  The finite-element reference is solved once and
shared between the forms, so both are scored against identical ground truth.

Run with:
    python -m verification.operator_study --minutes 45 --bank 16
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import config as cfg
from config import (DECODER_CONFIG, ENCODER_CONFIG, LOADING_CONFIG,
                    MATERIAL_CONFIG, NONDIM_SCALES, TRAINING_CONFIG,
                    get_fillet_geometry)
from eval.compare_reference import compare_one
from geometry.banks import VAL_BANK_SEED, build_geometry_bank
from models.pi_ginot import PI_GINOT
from physics.losses import PhysicsLoss
from physics.weak_form import EnergyLoss
from training.manifest import set_deterministic, write_manifest
from verification.fem.mesh import build_mesh, default_h
from verification.fem.solver import solve as fem_solve

MU = MATERIAL_CONFIG["mu"]
LAM = MATERIAL_CONFIG["lam"]
STATE = MATERIAL_CONFIG["state"]


class _Budget:
    """Stop training once a wall-clock budget is spent.

    Implemented by shrinking the epoch count the trainer sees, checked from a
    callback the trainer already calls each epoch -- simpler and less
    intrusive than threading a deadline through fit().
    """

    def __init__(self, seconds: float):
        self.deadline = time.time() + seconds
        self.epochs = 0

    def expired(self) -> bool:
        return time.time() >= self.deadline


def _build_loss(form: str, c: dict):
    if form == "energy":
        return EnergyLoss(
            mu=MU, lam=LAM, stress_state=STATE,
            n_elem_target=c["energy_n_elem"], resample=True,
            w_barrier=c["w_barrier"], j_min=c["barrier_delta"],
            E_scale=NONDIM_SCALES["S0"])
    return PhysicsLoss(
        mu=MU, lam=LAM, w_equilibrium=c["w_equilibrium"],
        w_trac_top=c["w_trac_top"], w_trac_arc=c["w_trac_arc"],
        w_traction_partial=c["w_traction_partial"], w_barrier=c["w_barrier"],
        j_min=c["barrier_delta"], adaptive_beta=c["adaptive_beta"],
        L0=NONDIM_SCALES["L0"], S0=NONDIM_SCALES["S0"], stress_state=STATE)


def train_operator(form: str, minutes: float, bank: int, batch: int,
                   lr: float, device: str, seed: int, out: str,
                   n_interior: int, chunk_epochs: int = 25,
                   overrides: dict = None, colloc_overrides: dict = None,
                   tag: str = None):
    """Train one operator under a wall-clock budget, in short fit() chunks.

    ``overrides`` / ``colloc_overrides`` let an ablation change one lever and
    nothing else; both are applied *before* the loss object is built, so a
    lever the loss reads (w_trac_top, the barrier) actually takes effect.
    Note that ``colloc_overrides`` mutates the module-level config -- the
    caller is responsible for restoring it between arms.
    """
    set_deterministic(seed)
    cfg.COLLOCATION_CONFIG["n_interior"] = n_interior
    if colloc_overrides:
        cfg.COLLOCATION_CONFIG.update(colloc_overrides)
    from training.trainer import PI_GINOT_Trainer

    c = dict(TRAINING_CONFIG)
    c.update(batch_size=batch, bank_train_size=bank, bank_val_size=2,
             print_every=chunk_epochs, val_every=10 ** 6, plot_every=0,
             save_geo=False, adaptive_start_epoch=10 ** 6,
             loss_form=form, learning_rate=lr, epochs=chunk_epochs,
             barrier_warmup_epochs=0)
    if overrides:
        c.update(overrides)

    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).to(device)
    tr = PI_GINOT_Trainer(model=model, loss_fn=_build_loss(form, c),
                          config=c, device=device,
                          save_dir=os.path.join(out, tag or form))

    t0 = time.time()
    deadline = t0 + minutes * 60.0
    epochs = 0
    while time.time() < deadline:
        tr.start_epoch = 0
        tr.fit(epochs=chunk_epochs, print_every=10 ** 6)
        epochs += chunk_epochs
    wall = time.time() - t0
    print(f"  [{tag or form}] {epochs} epochs in {wall / 60:.1f} min "
          f"({wall / max(epochs, 1):.2f} s/epoch)", flush=True)
    return model, epochs, wall


_REF_CACHE: dict = {}


def _reference(bank, gi, ref_h_factor):
    """FEM reference for one validation geometry, solved once per session.

    Cached across forms deliberately: the reference does not depend on the
    model, and sharing it removes any chance that the two formulations are
    scored against subtly different ground truths.
    """
    if gi not in _REF_CACHE:
        gmesh, _ = bank[gi]
        fi = get_fillet_geometry(gmesh.params)
        mesh = build_mesh(gmesh.params, h=default_h(fi) * ref_h_factor)
        _REF_CACHE[gi] = fem_solve(mesh, LOADING_CONFIG["u_max"], MU, LAM,
                                   STATE, n_steps=1, verbose=False)
    return _REF_CACHE[gi]


def evaluate(model, val_indices, device, ref_h_factor=2.5):
    """Compare against the FEM reference on held-out geometries."""
    bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                               TRAINING_CONFIG["bank_geo_ranges"],
                               holes_on=False, seed=VAL_BANK_SEED)
    rows = []
    for gi in val_indices:
        gmesh, coll = bank[gi]
        p = gmesh.params
        sol = _reference(bank, gi, ref_h_factor)
        r = compare_one(model, p, LOADING_CONFIG["u_max"], None, device,
                        sample_id=100 + gi, coll=coll, sol=sol)
        rows.append({
            "geo": gi, "taper": p["W_gauge"] / p["W_grip"],
            "u": r["disp_u_rel_L2"], "v": r["disp_v_rel_L2"],
            "vm": r["vm_all"]["rel_L2"],
            "vm_fillet": r["vm_fillet"]["rel_L2"],
            "s22": r["s22_all"]["rms_over_scale"],
            "N_err": r["N_rel_err"],
        })
        print(f"    geo {gi:>2} (taper {rows[-1]['taper']:.2f}): "
              f"u={rows[-1]['u']:.3e} v={rows[-1]['v']:.3e} "
              f"vm={rows[-1]['vm']:.3e} N={100 * rows[-1]['N_err']:+.1f}%",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forms", default="energy,penalty")
    ap.add_argument("--minutes", type=float, default=45.0)
    ap.add_argument("--bank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-interior", type=int, default=1500)
    ap.add_argument("--val", default="2,11,13,16,19")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default="verification/results/operator")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, cfg, extra={"case": "operator_study",
                                         "args": vars(args)})
    val_indices = [int(g) for g in args.val.split(",")]

    # The trainer draws its *own* validation bank from VAL_BANK_SEED and takes
    # the first ``bank_val_size`` entries, whose loss drives the LR scheduler.
    # Those geometries are therefore not held out in any useful sense, even
    # though no gradient flows through them.  Refuse to score on them rather
    # than caveat the result later.
    n_seen = 2  # bank_val_size passed to the trainer in train_operator
    leaked = [g for g in val_indices if g < n_seen]
    if leaked:
        raise SystemExit(
            f"--val {leaked} overlaps the trainer's validation bank "
            f"(indices 0..{n_seen - 1} of the same seed drive the LR "
            f"scheduler). Pick indices >= {n_seen}.")

    print("Operator-mode study — one network over a geometry bank")
    print("=" * 82)
    print(f"  train bank {args.bank} geometries, batch {args.batch}, "
          f"lr {args.lr}, {args.n_interior} interior points")
    print(f"  held-out validation geometries {val_indices} "
          "(different bank seed, so disjoint)")
    print(f"  equal wall-clock: {args.minutes:g} min per form\n")

    results = {}
    for form in [f.strip() for f in args.forms.split(",")]:
        print(f"  --- {form} ---", flush=True)
        model, epochs, wall = train_operator(
            form, args.minutes, args.bank, args.batch, args.lr, args.device,
            args.seed, args.out, args.n_interior)
        rows = evaluate(model, val_indices, args.device)
        results[form] = {"epochs": epochs, "wall_s": wall, "rows": rows}
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(results, fh, indent=2, default=float)

    print("\n" + "=" * 82)
    print("Held-out geometries, mean over the set")
    print(f"{'form':>10}{'epochs':>9}{'s/epoch':>9}{'u':>11}{'v':>11}"
          f"{'vm':>11}{'vm fillet':>12}{'|N err|':>10}")
    print("-" * 83)
    for form, r in results.items():
        m = lambda k: np.mean([x[k] for x in r["rows"]])
        print(f"{form:>10}{r['epochs']:>9d}"
              f"{r['wall_s'] / max(r['epochs'], 1):>9.2f}"
              f"{m('u'):>11.3e}{m('v'):>11.3e}{m('vm'):>11.3e}"
              f"{m('vm_fillet'):>12.3e}"
              f"{100 * np.mean([abs(x['N_err']) for x in r['rows']]):>9.1f}%")
    if len(results) == 2:
        a, b = list(results)
        print("-" * 83)
        ma = lambda f, k: np.mean([x[k] for x in results[f]["rows"]])
        print(f"{'ratio':>10}{'':>18}" + "".join(
            f"{ma(a, k) / ma(b, k):>11.3f}" for k in ("u", "v", "vm")) +
            f"{ma(a, 'vm_fillet') / ma(b, 'vm_fillet'):>12.3f}"
            f"{np.mean([abs(x['N_err']) for x in results[a]['rows']]) / max(np.mean([abs(x['N_err']) for x in results[b]['rows']]), 1e-12):>10.3f}")
        print(f"           ({a} / {b}; below 1 means {a} is better)")
    print(f"\nWrote {os.path.join(args.out, 'result.json')}")


if __name__ == "__main__":
    main()
