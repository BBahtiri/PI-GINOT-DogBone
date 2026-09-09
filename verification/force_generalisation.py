#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the energy form's section-force error a bias, or a generalisation gap?

The operator study found the energy form's held-out force error (3.5%) worse
than the penalty form's (2.5%), and systematic rather than scattered.  The
obvious reading is a bias in the formulation, and the obvious fix is to put
the section anchor back.  Both are wrong, and this module is what shows it:
the unanchored energy form is accurate to **0.39%** on the geometries it
trained on.  There is no bias for an anchor to remove -- what fails is the
transfer to an unseen shape.

Reference
---------
Against ``series_calibrated`` rather than a finite-element solve, so that
sixteen or sixty-four training geometries can be scored in seconds instead of
an hour.  On the held-out set that target sits 0.44% from the FEM reference,
which is ample for a 3% effect, and it is the *same* target for train and
held-out geometries -- so the gap, which is what this measures, is unaffected
by the target's own error.

Run with:
    python -m verification.force_generalisation \\
        --ckpt control=verification/results/anchor_sweep/energy-w0/last.pt \\
        --ckpt penalty=verification/results/operator/penalty/last.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import config as cfg
from config import (DECODER_CONFIG, ENCODER_CONFIG, LOADING_CONFIG,
                    MATERIAL_CONFIG, TRAINING_CONFIG, get_fillet_geometry)
from eval.compare_reference import _pinn_resultant
from geometry.banks import (TRAIN_BANK_SEED, VAL_BANK_SEED,
                            build_geometry_bank)
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import generate_dogbone
from models.pi_ginot import PI_GINOT
from physics.uniaxial import SERIES_FEM_CALIBRATION, section_resultant_series

MU, LAM = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"]
XI = np.array([0.15, 0.35, 0.55, 0.75, 0.92])


def _train_entries(n: int, seed: int = 7):
    """Training-bank geometries, meshed the way the trainer meshes them.

    The trainer holds the train bank params-only and regenerates mesh and
    collocation each epoch, so there is no frozen (mesh, coll) to reuse; these
    are drawn once here with a fixed seed.
    """
    params = build_geometry_bank(n, TRAINING_CONFIG["bank_geo_ranges"],
                                 holes_on=False, seed=TRAIN_BANK_SEED,
                                 mesh_only=True)
    rng = np.random.default_rng(seed)
    n_seg = cfg.COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
    out = []
    for p in params:
        m = generate_dogbone(p, n_pts_per_segment=n_seg, rng=rng)
        out.append((m, sample_collocation_points(m, rng=rng)))
    return out


def force_errors(model, entries, sample_ids, device="cpu"):
    """Signed relative error of the mean section resultant, per geometry."""
    u = LOADING_CONFIG["u_max"]
    errs = []
    for (gmesh, coll), sid in zip(entries, sample_ids):
        fi = get_fillet_geometry(gmesh.params)
        N = np.mean([_pinn_resultant(model, gmesh.params, x * fi["L_half"], u,
                                     device, sid, coll, fi) for x in XI])
        target = section_resultant_series(fi, u, MU, LAM) \
            * SERIES_FEM_CALIBRATION
        errs.append((N - target) / target)
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", action="append", default=[],
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--n-train", type=int, default=8,
                    help="training geometries to score (the first N of the "
                         "bank; scoring all 64 is slow and adds little)")
    ap.add_argument("--val", default="2,11,13,16,19")
    ap.add_argument("--n-interior", type=int, default=1500,
                    help="must match the run being scored — it changes which "
                         "specimens the bank seed produces")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not args.ckpt:
        raise SystemExit("give at least one --ckpt NAME=PATH")

    cfg.COLLOCATION_CONFIG["n_interior"] = args.n_interior
    val_idx = [int(g) for g in args.val.split(",")]
    val_bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                                   TRAINING_CONFIG["bank_geo_ranges"],
                                   holes_on=False, seed=VAL_BANK_SEED)
    val_entries = [val_bank[g] for g in val_idx]
    train_entries = _train_entries(args.n_train)

    print(f"Section-force error against the calibrated closed-form target")
    print(f"  {args.n_train} training geometries, held-out {val_idx}")
    print("=" * 74)
    print(f"{'model':>26}{'train |err|':>14}{'held-out |err|':>16}{'gap':>10}")
    print("-" * 74)
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).to(args.device)
        ck = torch.load(path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        et = np.abs(force_errors(model, train_entries,
                                 range(args.n_train), args.device)).mean()
        ev = np.abs(force_errors(model, val_entries,
                                 [100 + g for g in val_idx],
                                 args.device)).mean()
        print(f"{name:>26}{100 * et:>13.2f}%{100 * ev:>15.2f}%"
              f"{100 * (ev - et):>9.2f}")


if __name__ == "__main__":
    main()
