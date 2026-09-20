#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the encoder discard the geometry, or fail to learn to use it?

Phase 4 showed that conditioning the decoder on the four exact geometry
parameters instead of on the boundary point cloud takes held-out transverse
error from 3.04e-01 to 1.07e-02 supervised, and from 2.88e-01 to 1.06e-01
under physics training.  The conditioning path is the bottleneck.  What that
does *not* say is which kind of bottleneck it is:

  (a) the information is destroyed -- farthest-point sampling and ball-query
      pooling throw away the fillet detail before the decoder ever sees it; or
  (b) the information survives but the model never learns to read it.

Both are encoder problems and both were consistent with every number measured
so far, but they call for opposite remedies -- sampling resolution and
receptive field for (a), inductive bias and optimisation for (b).

The probe
---------
The boundary point cloud is *generated from* the four parameters, so they are
recoverable in principle.  Fit a probe from the encoder's pooled latent to the
parameters and see whether they are recoverable in practice:

Two representations are probed, because the decoder reads both and they can
disagree:

  * the **pooled** latent (mean over tokens), which is what FiLM conditioning
    sees; and
  * the **full token set** (32 x 64, flattened), which cross-attention sees.
    Information can live in the token structure and be averaged away in the
    pool, so a low pooled score alone would not show the encoder destroys
    anything.

Both use ridge regression with the penalty chosen on a validation split, which
matters at these sample sizes: an unregularised MLP probe scored *worse than
chance* on every parameter here (mean R^2 -0.68) purely by overfitting 180
training points, and would have been read as "the information is absent".

Run on three encoders, which separate the two readings:

  * ``trained``  -- the encoder from a physics run.  A high score means the
    learned representation contains the geometry and the failure is
    downstream: reading (b).
  * ``random``   -- an untrained encoder.  This is the architecture's raw
    information-preservation, before any learning.  If a random encoder scores
    well and the trained one does not, training actively discarded it.
  * ``chance``   -- predicting the training-set mean, so the reported R^2 has
    a floor that is not zero when the parameter ranges are narrow.

Per-parameter scores matter as much as the mean: the Phase 3 hypothesis was
specifically that ``R_fillet`` -- the local fillet detail the transverse field
depends on -- is what goes missing, and that is a claim this can check
directly rather than by inference from field errors.

Run with:
    python -m verification.encoder_probe --n 240 \\
        --ckpt verification/results/bank64/energy-w0/last.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import config as cfg
from config import DECODER_CONFIG, ENCODER_CONFIG, TRAINING_CONFIG
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params
from models.pi_ginot import PI_GINOT

KEYS = ("L_total", "W_grip", "W_gauge", "R_fillet")

# The encoder is handed the boundary point cloud normalised to [-1, 1] -- by
# x_max in x and y_max in y -- so absolute millimetres are removed from its
# input before it sees anything.  Asking it for R_fillet in mm is asking for
# something its input cannot contain, and a low score would say nothing about
# the encoder.  These are the dimensionless quantities that survive the
# normalisation and that it could in principle carry; the decoder recovers
# absolute sizes by combining them with the x_max / y_max scalars it gets
# separately.
RATIOS = {
    "Wg/Wgrip": lambda p: p[:, 2] / p[:, 1],
    "R/L": lambda p: p[:, 3] / p[:, 0],
    "R/Wgrip": lambda p: p[:, 3] / p[:, 1],
    "L/Wgrip": lambda p: p[:, 0] / p[:, 1],
}


def build_dataset(n: int, seed: int, device: str, model, cache: str = None):
    """Latent tokens and normalised parameters for n fresh geometries.

    Cached, because encoding dominates the cost and the probe design went
    through several revisions on the same encodings.
    """
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if "P" in d:
            return d["Z"], d["Y"], d["P"]
    rng = np.random.default_rng(seed)
    ranges = TRAINING_CONFIG["bank_geo_ranges"]
    n_seg = cfg.COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
    Z, Y, P = [], [], []
    with torch.no_grad():
        for i in range(n):
            p = sample_geometry_params(rng, geometry_ranges=ranges,
                                       holes_enabled=False)
            mesh = generate_dogbone(p, n_pts_per_segment=n_seg, rng=rng)
            coll = sample_collocation_points(mesh, rng=rng)
            bpc = torch.tensor(coll.boundary_pc, dtype=torch.get_default_dtype(),
                               device=device).unsqueeze(0)
            xm = torch.tensor([coll.x_max], dtype=torch.get_default_dtype(), device=device)
            ym = torch.tensor([coll.y_max], dtype=torch.get_default_dtype(), device=device)
            z = model.encode(bpc, xm, ym,
                             sample_ids=torch.tensor([i], dtype=torch.long))
            # Keep the full token set; the caller pools it as needed.
            Z.append(z[0].cpu().numpy())
            Y.append([(float(p[k]) - ranges[k][0])
                      / (ranges[k][1] - ranges[k][0]) for k in KEYS])
            P.append([float(p[k]) for k in KEYS])
            if (i + 1) % 40 == 0:
                print(f"\r    encoded {i + 1}/{n}", end="", flush=True)
    print()
    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, Z=Z, Y=Y, P=P)
    return Z, Y, P


def r2(pred, true):
    """Per-column R^2 against the mean predictor."""
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-30)


def ridge(Xtr, Ytr, Xte, lam):
    """Ridge with an UNPENALISED intercept, via centering.

    Penalising the intercept is not a detail: with it in the penalty matrix,
    a large lambda shrinks predictions toward zero rather than toward the
    training mean, so R^2 goes to -3 instead of to 0 and the selection step
    then reads a well-regularised fit as catastrophic.  That is what the first
    version of this probe reported for the token representation.
    """
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    A = Xtr - xm
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return (Xte - xm) @ W + ym


def ridge_probe(Xtr, Ytr, Xte, val_frac=0.25, seed=3):
    """Ridge with the penalty selected on a validation split.

    Selection is not optional here.  The token representation is 2048
    dimensional against a few hundred training rows, so an unpenalised fit
    interpolates the training set and reports a negative held-out R^2 that
    says nothing about what the representation contains.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xtr))
    n_val = max(8, int(val_frac * len(Xtr)))
    va, tr = order[:n_val], order[n_val:]
    best, best_lam = -np.inf, 1.0
    for lam in np.logspace(-3, 6, 19):
        sc = r2(ridge(Xtr[tr], Ytr[tr], Xtr[va], lam), Ytr[va]).mean()
        if sc > best:
            best, best_lam = sc, lam
    return ridge(Xtr, Ytr, Xte, best_lam), best_lam


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--ckpt", default="verification/results/bank64/"
                                      "energy-w0/last.pt")
    ap.add_argument("--n-point", type=int, default=None,
                    help="override ENCODER_CONFIG['n_point'] (FPS centroids)")
    ap.add_argument("--radius", type=float, default=None,
                    help="override ENCODER_CONFIG['radius'] (ball query)")
    ap.add_argument("--n-sample", type=int, default=None,
                    help="override ENCODER_CONFIG['n_sample']")
    ap.add_argument("--tag", default="",
                    help="suffix for the encoding cache, so sweep arms do "
                         "not read each other's cached latents")
    ap.add_argument("--variants", default="random,trained",
                    help="which encoders to probe; a trained checkpoint "
                         "cannot be loaded under overridden encoder settings")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg.COLLOCATION_CONFIG["n_interior"] = 1500
    n_te = int(args.n * args.test_frac)

    enc = dict(ENCODER_CONFIG)
    for key, val in (("n_point", args.n_point), ("radius", args.radius),
                     ("n_sample", args.n_sample)):
        if val is not None:
            enc[key] = val
    if enc != ENCODER_CONFIG:
        print(f"  encoder overrides: "
              + ", ".join(f"{k}={enc[k]}" for k in
                          ("n_point", "radius", "n_sample")))

    wanted = [v.strip() for v in args.variants.split(",")]
    variants = {}
    if "random" in wanted:
        # Pin the init.  Without this each invocation builds a differently
        # initialised encoder, and comparing two sampling configurations across
        # invocations compares two random networks as much as two samplers --
        # the init spread on R/L alone covers -0.03 to 0.16.
        torch.manual_seed(args.seed)
        variants["random"] = PI_GINOT(enc, DECODER_CONFIG).to(
            args.device).eval()
    try:
        if "trained" not in wanted:
            raise RuntimeError("not requested")
        m_tr = PI_GINOT(enc, DECODER_CONFIG).to(args.device)
        sd = torch.load(args.ckpt, map_location=args.device,
                        weights_only=False)["model_state_dict"]
        sd, _ = PI_GINOT.strip_legacy_keys(sd)
        m_tr.load_state_dict(sd)
        variants["trained"] = m_tr.eval()
    except Exception as exc:                       # noqa: BLE001
        print(f"  (no trained encoder: {exc})")

    print("Can the four geometry parameters be read out of the latent?")
    print("=" * 82)
    print(f"  {args.n} fresh geometries, {n_te} held out for the probe\n")

    for name, model in variants.items():
        print(f"  --- {name} encoder ---")
        Ztok, Y, Praw = build_dataset(
            args.n, args.seed, args.device, model,
            cache=f"verification/results/probe_cache/{name}{args.tag}_"
                  f"{args.n}_{args.seed}.npz")
        reps = {"pooled": Ztok.mean(axis=1),
                "tokens": Ztok.reshape(len(Ztok), -1)}
        Yr = np.stack([f(Praw) for f in RATIOS.values()], axis=1)
        for label, target, cols in (("absolute (mm)", Y, KEYS),
                                    ("dimensionless", Yr, tuple(RATIOS))):
            print(f"    {label}")
            print(f"    {'probe':>8}" + "".join(f"{k:>12}" for k in cols)
                  + f"{'mean':>9}{'lambda':>10}")
            for rname, X in reps.items():
                Xtr, Xte = X[n_te:], X[:n_te]
                Ytr, Yte = target[n_te:], target[:n_te]
                pred, lam = ridge_probe(Xtr, Ytr, Xte)
                sc = r2(pred, Yte)
                print(f"    {rname:>8}" + "".join(f"{v:>12.3f}" for v in sc)
                      + f"{sc.mean():>9.3f}{lam:>10.1e}")
            print()
    print("  R^2 = 1 is exact recovery, 0 is no better than predicting the")
    print("  mean.  High scores mean the geometry survives the encoder and the")
    print("  failure is in using it; low scores mean it is destroyed.")


if __name__ == "__main__":
    main()
