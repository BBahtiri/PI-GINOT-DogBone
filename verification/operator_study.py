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
from training.manifest import (set_deterministic, set_precision,
                               write_manifest)
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
    if form == "mixed":
        # Stress potentials: equilibrium and tractions exact by construction,
        # the constitutive gap the only thing enforced.  See physics/mixed.py.
        from physics.mixed import MixedLoss
        return MixedLoss(
            mu=MU, lam=LAM, stress_state=STATE,
            n_elem_target=c["energy_n_elem"], resample=True,
            w_barrier=c["w_barrier"], j_min=c["barrier_delta"],
            E_scale=NONDIM_SCALES["S0"], S0=NONDIM_SCALES["S0"],
            E=MATERIAL_CONFIG["E"],
            w_energy=c.get("w_energy_mixed", 0.0))
    if form == "hybrid":
        # The energy functional with the equilibrium residual added back --
        # a combination that had never been run, because loss_form was
        # always one form or the other.  See physics.weak_form.HybridLoss.
        from physics.weak_form import HybridLoss
        return HybridLoss(
            mu=MU, lam=LAM, stress_state=STATE,
            n_elem_target=c["energy_n_elem"], resample=True,
            w_barrier=c["w_barrier"], j_min=c["barrier_delta"],
            E_scale=NONDIM_SCALES["S0"],
            w_trac_top=c.get("energy_w_trac_top", 0.0),
            w_trac_arc=c.get("energy_w_trac_arc", 0.0),
            S0=NONDIM_SCALES["S0"], L0=NONDIM_SCALES["L0"],
            w_eq=c.get("w_equilibrium_hybrid", 1.0))
    if form == "energy":
        return EnergyLoss(
            mu=MU, lam=LAM, stress_state=STATE,
            n_elem_target=c["energy_n_elem"], resample=True,
            w_barrier=c["w_barrier"], j_min=c["barrier_delta"],
            E_scale=NONDIM_SCALES["S0"],
            w_trac_top=c.get("energy_w_trac_top", 0.0),
            w_trac_arc=c.get("energy_w_trac_arc", 0.0),
            S0=NONDIM_SCALES["S0"])
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
                   tag: str = None, max_epochs: int = None,
                   conditioning: str = "pointcloud"):
    """Train one operator under a wall-clock budget, in short fit() chunks.

    ``max_epochs`` stops on an epoch count instead of the clock.  Equal
    wall-clock is the right budget when comparing formulations that differ in
    cost per epoch; equal epochs is the right budget when comparing a single
    lever within one formulation, where it also makes the result immune to
    whatever else the machine happens to be doing.

    ``overrides`` / ``colloc_overrides`` let an ablation change one lever and
    nothing else; both are applied *before* the loss object is built, so a
    lever the loss reads (w_trac_top, the barrier) actually takes effect.
    Note that ``colloc_overrides`` mutates the module-level config -- the
    caller is responsible for restoring it between arms.
    """
    # Reserved override keys are popped before anything is built.  Precision
    # has to come first of all: it changes torch's default dtype, and every
    # tensor created after this point -- parameters included -- takes it.
    overrides = dict(overrides) if overrides else {}
    precision = overrides.pop("_precision", "float32")
    set_precision(precision)
    set_deterministic(seed)
    cfg.COLLOCATION_CONFIG["n_interior"] = n_interior
    if colloc_overrides:
        cfg.COLLOCATION_CONFIG.update(colloc_overrides)
    from training.trainer import PI_GINOT_Trainer

    # Encoder overrides travel inside the training overrides under a
    # reserved key, so an ablation arm can change the encoder without a
    # second plumbing path through every caller.
    enc = dict(ENCODER_CONFIG)
    enc.update(overrides.pop("_encoder", {}))
    dec = dict(DECODER_CONFIG)
    dec.update(overrides.pop("_decoder", {}))

    c = dict(TRAINING_CONFIG)
    c.update(batch_size=batch, bank_train_size=bank, bank_val_size=2,
             print_every=chunk_epochs, val_every=10 ** 6, plot_every=0,
             save_geo=False, adaptive_start_epoch=10 ** 6,
             loss_form=form, learning_rate=lr, epochs=chunk_epochs,
             barrier_warmup_epochs=0)
    if overrides:
        c.update(overrides)

    if conditioning == "oracle":
        from verification.supervised_ceiling import OracleConditioned
        model = OracleConditioned(enc, dec,
                                  TRAINING_CONFIG["bank_geo_ranges"]).to(device)
    else:
        model = PI_GINOT(enc, dec,
                         aux_head=c.get("w_geom_aux", 0.0) > 0.0).to(device)
    for label, cur, base in (("encoder", enc, ENCODER_CONFIG),
                             ("decoder", dec, DECODER_CONFIG)):
        if cur != base:
            print(f"  {label} overrides: " + ", ".join(
                f"{k}={v}" for k, v in cur.items() if base.get(k) != v))
    if precision != "float32":
        print(f"  precision: {precision}")
    tr = PI_GINOT_Trainer(model=model, loss_fn=_build_loss(form, c),
                          config=c, device=device,
                          save_dir=os.path.join(out, tag or form))

    t0 = time.time()
    deadline = t0 + minutes * 60.0 if max_epochs is None else float('inf')
    epochs = 0
    while time.time() < deadline and (max_epochs is None
                                      or epochs < max_epochs):
        n = chunk_epochs if max_epochs is None else min(
            chunk_epochs, max_epochs - epochs)
        tr.start_epoch = 0
        tr.fit(epochs=n, print_every=10 ** 6)
        epochs += n
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
    # Keyed on the mesh factor as well as the geometry.  Keying on the
    # geometry alone would silently hand back the first mesh ever solved for
    # it, so a second call asking for a finer reference would be scored
    # against the coarse one and report no change -- the failure would look
    # like a converged result.
    key = (gi, round(float(ref_h_factor), 6))
    if key not in _REF_CACHE:
        gmesh, _ = bank[gi]
        fi = get_fillet_geometry(gmesh.params)
        mesh = build_mesh(gmesh.params, h=default_h(fi) * ref_h_factor)
        _REF_CACHE[key] = fem_solve(mesh, LOADING_CONFIG["u_max"], MU, LAM,
                                    STATE, n_steps=1, verbose=False)
    return _REF_CACHE[key]


# Metrics whose sign carries information: a peak 5% low and a peak 5% high
# are different failures.  The per-geometry rows keep the sign, but averaging
# signed errors across geometries lets one specimen's low peak cancel
# another's high peak and report a model as better than any single prediction
# it made, so the summary tables aggregate these by magnitude.
SIGNED = ("vm_peak", "scf", "N_err")

# Column order for the summary tables, shared with verification.ablations so
# the two harnesses stay directly comparable.
REPORT_METRICS = ("u", "v", "vm", "vm_fillet", "vm_peak",
                  "vm_fillet_Linf", "scf")


def mean_metric(rows, key):
    """Mean over geometries, magnitude-averaged for the signed metrics."""
    v = [x[key] for x in rows]
    return float(np.mean(np.abs(v) if key in SIGNED else v))


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
        if hasattr(model, "set_geometry"):
            model.set_geometry(p)
        r = compare_one(model, p, LOADING_CONFIG["u_max"], None, device,
                        sample_id=100 + gi, coll=coll, sol=sol)
        rows.append({
            "geo": gi, "taper": p["W_gauge"] / p["W_grip"],
            "u": r["disp_u_rel_L2"], "v": r["disp_v_rel_L2"],
            "vm": r["vm_all"]["rel_L2"],
            "vm_fillet": r["vm_fillet"]["rel_L2"],
            # The stress-concentration metrics.  A relative L2 is an average
            # over the specimen, and the specimen is mostly prismatic gauge
            # in which the state is nearly uniaxial and easy -- so a model
            # can win the L2 while missing the one number a fillet study is
            # about.  These three are the norms the claim is actually made
            # in: the signed error in the peak von Mises, the worst
            # pointwise error anywhere in the fillet as a fraction of the
            # reference peak, and the signed error in the concentration
            # factor itself.  Reported alongside the L2, not instead of it,
            # because the pair is what distinguishes a model that has the
            # level right and the peak wrong from one that has neither.
            "vm_peak": r["vm_all"]["peak_rel_err"],
            "vm_fillet_Linf": r["vm_fillet"]["max"],
            "scf": r["scf_rel_err"],
            "scf_ref": r["scf_ref"], "scf_pinn": r["scf_pinn"],
            "x_peak_ref": r["x_peak_ref"], "x_peak_pinn": r["x_peak_pinn"],
            "x_peak_err": r["x_peak_err"],
            "vm_nominal_ref": r["vm_nominal_ref"],
            "vm_nominal_pinn": r["vm_nominal_pinn"],
            "s22": r["s22_all"]["rms_over_scale"],
            "N_err": r["N_rel_err"],
        })
        print(f"    geo {gi:>2} (taper {rows[-1]['taper']:.2f}): "
              f"u={rows[-1]['u']:.3e} v={rows[-1]['v']:.3e} "
              f"vm={rows[-1]['vm']:.3e} "
              f"peak={100 * rows[-1]['vm_peak']:+.1f}% "
              f"SCF={rows[-1]['scf_pinn']:.3f}/{rows[-1]['scf_ref']:.3f} "
              f"x*={rows[-1]['x_peak_pinn']:.3f}/"
              f"{rows[-1]['x_peak_ref']:.3f} "
              f"N={100 * rows[-1]['N_err']:+.1f}%", flush=True)
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
    ap.add_argument("--h-factor", type=float, default=1.3,
                    help="reference element size = default_h * this. Phase "
                         "6.1 measured the reference's own K_t error at "
                         "0.85%% for 2.5 and 0.19%% for 1.3; everything "
                         "before that phase was scored at 2.5, so older "
                         "numbers are comparable to each other and not to "
                         "these")
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
        rows = evaluate(model, val_indices, args.device,
                        ref_h_factor=args.h_factor)
        results[form] = {"epochs": epochs, "wall_s": wall, "rows": rows}
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(results, fh, indent=2, default=float)

    hdr = (f"{'form':>10}{'epochs':>8}{'s/ep':>7}" +
           "".join(f"{k:>15}" for k in REPORT_METRICS) + f"{'|N err|':>9}")
    print("\n" + "=" * len(hdr))
    print("Held-out geometries, mean over the set "
          "(vm_peak / scf averaged by magnitude)")
    print(hdr)
    print("-" * len(hdr))
    for form, r in results.items():
        cells = "".join(f"{mean_metric(r['rows'], k):>15.3e}"
                        for k in REPORT_METRICS)
        print(f"{form:>10}{r['epochs']:>8d}"
              f"{r['wall_s'] / max(r['epochs'], 1):>7.2f}{cells}"
              f"{100 * mean_metric(r['rows'], 'N_err'):>8.1f}%")
    if len(results) == 2:
        a, b = list(results)
        print("-" * len(hdr))
        ra = lambda k: (mean_metric(results[a]["rows"], k)
                        / max(mean_metric(results[b]["rows"], k), 1e-30))
        print(f"{'ratio':>10}{'':>15}"
              + "".join(f"{ra(k):>15.3f}" for k in REPORT_METRICS)
              + f"{ra('N_err'):>9.3f}")
        print(f"           ({a} / {b}; below 1 means {a} is better)")
    print(f"\nWrote {os.path.join(args.out, 'result.json')}")


if __name__ == "__main__":
    main()
