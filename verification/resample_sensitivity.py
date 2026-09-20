#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the operator a function of the geometry, or of the point cloud it was handed?

A neural operator over shapes claims to map a geometry to a field.  What it
is actually given is a finite sample of that geometry's boundary, and the
sample is arbitrary: a different mesher, a different seed, a different point
budget all describe the same dog-bone.  If the prediction moves when the
sample changes, the operator is not well defined on its stated input domain,
and any error bar quoted for it is missing a term.

Phase 6.2 found the mechanism that makes this a live worry here rather than a
formality.  The encoder picks its centroids by farthest-point sampling, which
is a sequence of argmaxes, and a dog-bone outline has long straight runs
along which many candidate points sit at nearly equal distance from the
already-chosen set.  Ties are generic for this geometry family, so the
centroid set is a discontinuous function of the cloud: perturbing the input
at the 1e-7 level was already enough to change it on five of eight held-out
geometries.  A genuine re-sample is a far larger perturbation.

This module measures the resulting spread directly.  For each geometry it
draws ``--draws`` independent boundary clouds of the *same* outline -- same
mesh, same point budget, different subsample -- and scores each against one
shared finite-element reference.  The number that matters is the last column:
the spread across draws as a fraction of the model's own mean error.  At 1.0
the sampling of the input is as large a source of error as the model, and
every metric in this repository is quoted for one arbitrary draw.

Evaluation only; trains nothing.

Run with:
    python -m verification.resample_sensitivity --draws 8 \
        --checkpoint verification/results/fixed_long_pc/energy-w0/last.pt
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import models.modules.pointnet2_utils as pnet
from config import LOADING_CONFIG, TRAINING_CONFIG
from eval.compare_reference import compare_one
from geometry.banks import VAL_BANK_SEED, build_geometry_bank
from geometry.collocation import sample_collocation_points
from training.manifest import set_precision
from verification.operator_study import _reference
from verification.rescore import load

# Metric name -> (extractor, is_signed).  Signed metrics are summarised by
# their spread as well as their magnitude; an unsigned one would hide a
# sampling-driven sign flip.
METRICS = {
    "u": (lambda r: r["disp_u_rel_L2"], False),
    "v": (lambda r: r["disp_v_rel_L2"], False),
    "vm": (lambda r: r["vm_all"]["rel_L2"], False),
    "vm_fillet": (lambda r: r["vm_fillet"]["rel_L2"], False),
    "vm_peak": (lambda r: r["vm_all"]["peak_rel_err"], True),
    "scf": (lambda r: r["scf_rel_err"], True),
    "x_peak": (lambda r: r["x_peak_pinn"], True),
}


def draws_for(model, gi, bank, sol, n_draws, device, base_seed=9000):
    """Score one geometry under ``n_draws`` independent boundary clouds."""
    gmesh, _ = bank[gi]
    out = []
    for d in range(n_draws):
        coll = sample_collocation_points(
            gmesh, n_interior=1, rng=np.random.default_rng(base_seed + d))
        # A fresh sample id per draw: the encoder caches its centroid and
        # grouping indices by id, so reusing one would score a new cloud
        # against the previous cloud's sampling and hide the effect entirely.
        sid = 10_000 + 100 * gi + d
        # A parameter-conditioned model needs its geometry stashed, and is
        # the right null control for this harness: it never reads the cloud,
        # so its spread over draws must come out at exactly zero.  If it does
        # not, the harness is measuring something other than the sampling.
        if hasattr(model, "set_geometry"):
            model.set_geometry(gmesh.params)
        r = compare_one(model, gmesh.params, LOADING_CONFIG["u_max"], None,
                        device, sample_id=sid, coll=coll, sol=sol)
        out.append({k: f(r) for k, (f, _) in METRICS.items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--conditioning", default="pointcloud",
                    choices=("pointcloud", "oracle"))
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--val", default="2,15,4,6,3,11,9,22")
    ap.add_argument("--h-factor", type=float, default=1.3)
    ap.add_argument("--precision", default="float32",
                    choices=("float32", "float64"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_precision(args.precision)
    tag = args.tag or os.path.basename(os.path.dirname(args.checkpoint))
    model, ck = load(args.checkpoint, args.conditioning, args.device)
    bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                               TRAINING_CONFIG["bank_geo_ranges"],
                               holes_on=False, seed=VAL_BANK_SEED)
    val = [int(g) for g in args.val.split(",")]

    print(f"Boundary-cloud resampling sensitivity — {tag} "
          f"({args.conditioning}), {args.draws} draws per geometry")
    print("=" * 86)
    print("  same outline, same point budget, different subsample; one "
          "shared reference per geometry\n")

    per_geo = {}
    for gi in val:
        sol = _reference(bank, gi, args.h_factor)
        pnet.CACHE_SAMPLE_AND_GROUP_INDECIES.clear()
        rows = draws_for(model, gi, bank, sol, args.draws, args.device)
        per_geo[gi] = rows
        vm = np.array([r["vm"] for r in rows])
        sc = np.array([r["scf"] for r in rows])
        xp = np.array([r["x_peak"] for r in rows])
        print(f"  geo {gi:>2}: vm {vm.mean():.3e} +- {vm.std(ddof=1):.1e} "
              f"({100 * vm.std(ddof=1) / vm.mean():.1f}% of the error)   "
              f"scf {100 * sc.mean():+.2f}% +- {100 * sc.std(ddof=1):.2f}pp   "
              f"x* {xp.mean():.3f} +- {xp.std(ddof=1):.3f}", flush=True)

    print("\n" + "=" * 86)
    print("Across geometries: spread over draws, against the model's own "
          "error in the same metric")
    hdr = (f"{'metric':>12}{'mean |.|':>13}{'sd over draws':>16}"
           f"{'max sd':>11}{'sd / |mean|':>13}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for k, (_, signed) in METRICS.items():
        a = np.array([[r[k] for r in per_geo[g]] for g in val])   # [geo, draw]
        sd = a.std(axis=1, ddof=1)
        # Signed metrics are summarised by magnitude: their signed mean
        # cancels across geometries (a peak low on one, high on another) and
        # would make the ratio below meaningless.
        mean = np.abs(a).mean()
        summary[k] = {"signed_mean": float(a.mean()), "abs_mean": float(mean),
                      "sd_mean": float(sd.mean()), "sd_max": float(sd.max()),
                      "ratio": float(sd.mean() / max(mean, 1e-300))}
        print(f"{k:>12}{mean:>13.3e}{sd.mean():>16.3e}"
              f"{sd.max():>11.3e}{sd.mean() / max(mean, 1e-300):>13.3f}")
    print("-" * len(hdr))
    print("  A ratio near 1 means which points the model was handed matters "
          "as much as the model.")

    out = args.out or os.path.join(os.path.dirname(args.checkpoint),
                                   "resample.json")
    with open(out, "w") as fh:
        json.dump({"tag": tag, "checkpoint": args.checkpoint,
                   "conditioning": args.conditioning, "draws": args.draws,
                   "precision": args.precision,
                   "ref_h_factor": args.h_factor,
                   "summary": summary,
                   "per_geometry": {str(k): v for k, v in per_geo.items()}},
                  fh, indent=2, default=float)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
