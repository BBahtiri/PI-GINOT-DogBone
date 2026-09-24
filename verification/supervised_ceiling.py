#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The ceiling: what can this architecture do on `v` when conditioning is free?

Phase 3 established that the transverse field sits in the objective's
near-null space -- the same relative error costs 183x more energy in `u` than
in `v` -- and that no weighting of the terms in either loss reaches it.  Two
routes remain (a scale-invariant optimiser, or supervision), and both are
expensive to pursue.  Before spending on either, there is a cheap question
that bounds them: **can this architecture represent the transverse field at
all, across a geometry family?**

If a supervised fit reaches `v` ~0.02, the ansatz and the decoder are fine and
everything that remains is objective and optimiser -- both routes are live.  If
a supervised fit *also* stalls near 0.2, then no optimiser and no amount of
supervision will help, because the operator itself cannot carry `v` across the
family, and Phase 4 belongs in the architecture instead.

Why this is a ceiling and not a proposal
----------------------------------------
The loss here is the per-component **relative** error

    L = ||u_pred - u_ref||^2 / ||u_ref||^2  +  ||v_pred - v_ref||^2 / ||v_ref||^2

which gives the two components equal footing by construction -- exactly the
preconditioning the energy functional lacks.  Plain MSE would not: `v` is
0.03-0.06 of `u` here, so an unweighted fit would rediscover the same
imbalance from the data side and measure nothing.  This is deliberately the
most favourable setting available, which is what makes a failure here
informative and a success here only an upper bound.

It is also not a training recipe.  It needs a finite-element solve per
geometry, which is the reference-free property the whole approach exists to
avoid.  The number it produces is a bound, not a model.

Run with:
    python -m verification.supervised_ceiling --bank 16 --epochs 800
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import torch

import config as cfg
from config import (DECODER_CONFIG, ENCODER_CONFIG, LOADING_CONFIG,
                    MATERIAL_CONFIG, TRAINING_CONFIG, get_fillet_geometry)
from eval.compare_reference import compare_one
from geometry.banks import (TRAIN_BANK_SEED, VAL_BANK_SEED,
                            build_geometry_bank)
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import generate_dogbone
import torch.nn as nn

from models.pi_ginot import PI_GINOT
from physics.neo_hookean import full_stress_state, von_mises_stress
from training.manifest import set_deterministic, write_manifest
from verification.fem.mesh import build_mesh, default_h
from verification.operator_study import REPORT_METRICS, mean_metric
from verification.fem.solver import solve as fem_solve

MU, LAM = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"]
STATE = MATERIAL_CONFIG["state"]
CACHE = "verification/results/fem_cache"


class OracleConditioned(PI_GINOT):
    """The same decoder, conditioned on the four exact geometry parameters.

    This separates two architectural explanations that the supervised ceiling
    leaves entangled.  Held-out `v` is ~0.3 however the model is trained, and
    the encoder was the suspect -- farthest-point sampling and ball-query
    subsampling are free to discard the fillet detail the transverse field
    depends on.  But "the encoder loses information" and "the map from
    geometry to transverse field is what the decoder cannot generalise" both
    predict exactly what was measured.

    The geometry here is *fully* determined by four numbers, so handing the
    decoder those numbers removes any possibility of information loss in the
    conditioning path.  If held-out `v` improves markedly, the encoder is the
    bottleneck and Phase 4 is an encoder problem.  If it does not, no encoder
    can help, because the information was never missing.

    The parameter MLP is sized to ~207k weights against the point-cloud
    encoder's 220k, so the comparison is not confounded by capacity.  The
    inherited encoder is left constructed but unused and excluded from the
    optimiser; ``trainable_parameters`` is what should be optimised.
    """

    ALL_KEYS = ("L_total", "W_grip", "W_gauge", "R_fillet")

    def __init__(self, encoder_config, decoder_config, ranges, hidden=96,
                 keys=None):
        super().__init__(encoder_config, decoder_config)
        # Withholding a parameter turns this into an ablation of *which*
        # geometry information matters.  Note that L_total and W_grip already
        # reach the decoder outside the latent, as x_max and y_max, so the
        # informative ablations are the other two.
        self.PARAM_KEYS = tuple(keys) if keys else self.ALL_KEYS
        self._n_lat = encoder_config["n_point"]
        self._dim = decoder_config["embed_dim"]
        self._ranges = ranges
        self.param_mlp = nn.Sequential(
            nn.Linear(len(self.PARAM_KEYS), hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self._n_lat * self._dim),
        )
        self._geo = None

    def trainable_parameters(self):
        return list(self.decoder.parameters()) + list(
            self.param_mlp.parameters())

    def set_geometry(self, params: dict):
        """Stash the parameters the next encode() call should use.

        Stateful, deliberately: eval/compare_reference drives the model
        through ``encode(boundary_pc, ...)`` and has no way to pass geometry
        parameters, and changing that signature to accommodate a diagnostic
        would be the wrong trade.
        """
        self._geo = params

    def encode(self, boundary_pc, x_max, y_max, sample_ids=None):
        """Latent from the stashed geometry parameters.

        ``boundary_pc`` is ignored.  Deriving the parameters from it instead
        was tried and abandoned: `L_total`, `W_grip` and `W_gauge` come back
        exactly, but `R_fillet` does not.  The arc is tangent to the gauge, so
        near `x_g` it departs from `y = H_gauge` quadratically and the tangency
        point cannot be located from sampled boundary points without a fit --
        worst-case error 79-88% on both banks.  Which is its own small echo of
        Phase 4: the fillet is the hard parameter to recover even analytically.
        """
        geo = self._geo
        if geo is None:
            raise RuntimeError(
                "OracleConditioned.encode: no geometry set.  The loss calls "
                "set_geometry(params) for the training path; a diagnostic that "
                "encodes outside the loss must call it too, or be skipped for "
                "parameter-conditioned models.")
        dev = next(self.param_mlp.parameters()).device
        x = torch.tensor(
            [[(float(geo[k]) - self._ranges[k][0])
              / (self._ranges[k][1] - self._ranges[k][0])
              for k in self.PARAM_KEYS]], dtype=torch.get_default_dtype(), device=dev)
        return self.param_mlp(x).view(1, self._n_lat, self._dim)


def _solve_cached(params, tag: str, h_factor: float):
    """FEM reference for one geometry, cached on disk.

    The solves dominate the cost of this study and do not depend on anything
    that changes between runs, so they are worth keeping.
    """
    # The mesh factor belongs in the key.  Without it, asking for a finer
    # reference returns the coarse solve that happened to be cached first,
    # and the run reports whatever the earlier run measured -- the same
    # defect that was fixed in verification.operator_study._REF_CACHE.
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{tag}_h{float(h_factor):.2f}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    fi = get_fillet_geometry(params)
    mesh = build_mesh(params, h=default_h(fi) * h_factor)
    sol = fem_solve(mesh, LOADING_CONFIG["u_max"], MU, LAM, STATE,
                    n_steps=1, verbose=False)
    with open(path, "wb") as fh:
        pickle.dump(sol, fh)
    return sol


def _entries(n_train: int, seed: int = 7, bank_seed: int = TRAIN_BANK_SEED,
             pick=None):
    """Training geometries with a frozen mesh and collocation set.

    The trainer regenerates these each epoch; here they must be fixed, because
    the finite-element reference is tied to one realisation.

    ``bank_seed`` and ``pick`` exist for the representability probe, which
    trains on the validation geometries themselves.
    """
    params = build_geometry_bank(n_train, TRAINING_CONFIG["bank_geo_ranges"],
                                 holes_on=False, seed=bank_seed,
                                 mesh_only=True)
    if pick is not None:
        params = [params[i] for i in pick]
    rng = np.random.default_rng(seed)
    n_seg = cfg.COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
    out = []
    for p in params:
        m = generate_dogbone(p, n_pts_per_segment=n_seg, rng=rng)
        out.append((m, sample_collocation_points(m, rng=rng)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-points", type=int, default=800,
                    help="target nodes drawn per geometry per epoch; the fit "
                         "does not need every node and the decoder pass over "
                         "them dominates the cost")
    ap.add_argument("--val", default="2,11,13,16,19")
    ap.add_argument("--h-factor", type=float, default=1.3,
                    help="reference element size = default_h * this. 1.3 "
                         "since Phase 6.1: at 2.5 the reference's own K_t "
                         "is biased low by up to 0.85%%, a third of the "
                         "better model's error")
    ap.add_argument("--train-on", default="train",
                    choices=("train", "val"),
                    help="'val' trains on the *same* geometries it scores. "
                         "That is an overfit and is labelled one: it drops "
                         "generalisation from the question and asks only "
                         "whether the architecture can represent these "
                         "fields at all, which is the cheapest decisive "
                         "test of an architectural limit")
    ap.add_argument("--supervise", default="disp",
                    choices=("disp", "stress", "both"),
                    help="disp: per-component relative error on (u, v) -- "
                         "the original ceiling.  stress: von Mises at the "
                         "element centroids plus an explicit peak term.  "
                         "both: the sum")
    ap.add_argument("--w-peak", type=float, default=1.0,
                    help="weight on the peak-region term of the stress loss")
    ap.add_argument("--n-peak", type=int, default=32,
                    help="how many highest-reference-stress centroids form "
                         "the peak region")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--params", default=None,
                    help="comma-separated subset of L_total,W_grip,W_gauge,"
                         "R_fillet for oracle conditioning; default all four")
    ap.add_argument("--conditioning", default="pointcloud",
                    choices=["pointcloud", "oracle"],
                    help="'oracle' replaces the point-cloud encoder with an "
                         "MLP over the four exact geometry parameters")
    ap.add_argument("--out", default="verification/results/supervised")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_deterministic(args.seed)
    cfg.COLLOCATION_CONFIG["n_interior"] = 1500
    write_manifest(args.out, cfg, extra={"case": "supervised_ceiling",
                                         "args": vars(args)})

    print("Supervised ceiling — per-component relative loss, no physics")
    print("=" * 78)
    print(f"  {args.bank} training geometries, {args.epochs} epochs, "
          f"batch {args.batch}, lr {args.lr}")

    val_idx = [int(g) for g in args.val.split(",")]
    if args.train_on == "val":
        print("  TRAINING ON THE VALIDATION GEOMETRIES — this is an overfit, "
              "and measures representability, not generalisation")
        entries = _entries(TRAINING_CONFIG["bank_val_size"],
                           bank_seed=VAL_BANK_SEED, pick=val_idx)
        tags = [f"val_{VAL_BANK_SEED}_{g}" for g in val_idx]
    else:
        entries = _entries(args.bank)
        tags = [f"train_{TRAIN_BANK_SEED}_{i}" for i in range(len(entries))]
    targets = []
    t0 = time.time()
    for i, (gmesh, _) in enumerate(entries):
        sol = _solve_cached(gmesh.params, tags[i], args.h_factor)
        targets.append(sol)
        print(f"\r  reference solves: {i + 1}/{len(entries)}", end="",
              flush=True)
    print(f"  ({time.time() - t0:.0f}s)")

    if args.conditioning == "oracle":
        keys = ([k.strip() for k in args.params.split(",")]
                if args.params else None)
        model = OracleConditioned(ENCODER_CONFIG, DECODER_CONFIG,
                                  TRAINING_CONFIG["bank_geo_ranges"],
                                  keys=keys).to(args.device)
        print(f"  oracle parameters: {model.PARAM_KEYS}")
        train_params = model.trainable_parameters()
    else:
        model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).to(args.device)
        train_params = list(model.parameters())
    print(f"  conditioning: {args.conditioning} "
          f"({sum(p.numel() for p in train_params):,} trainable)")
    opt = torch.optim.Adam(train_params, lr=args.lr)
    # Cosine over the fixed budget rather than ReduceLROnPlateau: the loss
    # here is stochastic in both the geometry batch and the node subsample, so
    # a plateau rule fires on noise -- at patience 25 the rate fell from 3e-4
    # to 2.5e-5 by epoch 350 and would have been frozen long before the budget
    # was spent.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.02)

    def tens(a, req=False):
        t = torch.tensor(a, dtype=torch.get_default_dtype(),
                         device=args.device).unsqueeze(0)
        return t.requires_grad_(req) if req else t

    packed = []
    for (gmesh, coll), sol in zip(entries, targets):
        # Stress targets live at the element centroids, where the
        # piecewise-constant finite-element stress is defined.  The peak set
        # is the n_peak highest-reference-stress centroids, chosen once by
        # the reference itself -- the network never has to find the peak
        # region, only match it there.  That is deliberately the most
        # favourable setting available, which is what makes a failure here
        # informative and a success here only an upper bound.
        vm_ref = sol.von_mises()
        peak_idx = np.argsort(vm_ref)[-args.n_peak:]
        packed.append((
            gmesh.params,
            tens(coll.boundary_pc),
            tens(sol.mesh.nodes),
            tens(sol.u),
            torch.tensor([coll.x_max], dtype=torch.get_default_dtype()),
            torch.tensor([coll.y_max], dtype=torch.get_default_dtype()),
            tens(sol.centroids),
            tens(vm_ref[:, None])[..., 0],
            torch.from_numpy(peak_idx.copy()).long(),
        ))

    u_delta = torch.tensor([LOADING_CONFIG["u_max"]], dtype=torch.get_default_dtype(),
                           device=args.device)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        idx = rng.choice(len(packed), size=min(args.batch, len(packed)),
                         replace=False)
        total = torch.zeros((), device=args.device)
        for k in idx:
            (gparams, bpc, pts, ref, xm, ym,
             cent, vm_ref, peak_idx) = packed[k]
            if args.conditioning == "oracle":
                model.set_geometry(gparams)
            sid = torch.tensor([int(k)], dtype=torch.long)
            z = model.encode(bpc, xm, ym, sample_ids=sid)

            if args.supervise in ("disp", "both"):
                p_, r_ = pts, ref
                if args.n_points and p_.shape[1] > args.n_points:
                    sel = torch.from_numpy(
                        rng.choice(p_.shape[1], args.n_points, replace=False))
                    p_, r_ = p_[:, sel], r_[:, sel]
                uv = model.decode(p_, z, u_delta, xm, ym)
                # Per-component relative error: the preconditioning the
                # energy functional does not have.  Without it this measures
                # the same 183x imbalance from the data side and says
                # nothing new.
                for c in (0, 1):
                    d = uv[0, :, c] - r_[0, :, c]
                    total = total + (d.pow(2).sum()
                                     / r_[0, :, c].pow(2).sum()
                                     .clamp_min(1e-30))

            if args.supervise in ("stress", "both"):
                # A uniform subsample for the field term, plus the fixed
                # peak set.  Concatenated into one forward pass so the two
                # terms share a graph; the peak points are appended last so
                # they can be sliced back off.
                n_f = min(args.n_points or cent.shape[1], cent.shape[1])
                sel = torch.from_numpy(
                    rng.choice(cent.shape[1], n_f, replace=False))
                qi = torch.cat([sel, peak_idx])
                q = cent[:, qi].clone().requires_grad_(True)
                t = vm_ref[:, qi]
                _, a, b, c_, d_ = model.predict_with_grad_latent(
                    q, z, u_delta, xm, ym)
                S11, S22, S33, S12, _ = full_stress_state(
                    a, b, c_, d_, MU, LAM, STATE)
                vm = von_mises_stress(S11, S22, S33, S12)[0, :, 0]
                e = vm[:n_f] - t[0, :n_f]
                total = total + (e.pow(2).sum()
                                 / t[0, :n_f].pow(2).sum().clamp_min(1e-30))
                # The peak region, as a level rather than a maximum: the max
                # over a subsample is a noisy statistic with a gradient
                # reaching one point, while the mean over the reference's
                # own top-n_peak carries the same information with a
                # gradient reaching all of them.
                if args.w_peak > 0.0:
                    pk, pr = vm[n_f:].mean(), t[0, n_f:].mean()
                    total = total + args.w_peak * (
                        (pk - pr) / pr.clamp_min(1e-30)).pow(2)
        loss = total / len(idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:>5}  loss={loss.item():.4e}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"({(time.time() - t0) / ep:.2f} s/epoch)", flush=True)

    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(args.out, "last.pt"))

    val_bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                                   TRAINING_CONFIG["bank_geo_ranges"],
                                   holes_on=False, seed=VAL_BANK_SEED)
    print("\n  held-out geometries")
    rows = []
    for gi in val_idx:
        gmesh, coll = val_bank[gi]
        sol = _solve_cached(gmesh.params, f"val_{VAL_BANK_SEED}_{gi}",
                            args.h_factor)
        if args.conditioning == "oracle":
            model.set_geometry(gmesh.params)
        r = compare_one(model, gmesh.params, LOADING_CONFIG["u_max"], None,
                        args.device, sample_id=100 + gi, coll=coll, sol=sol)
        fi = get_fillet_geometry(gmesh.params)
        rows.append({"geo": gi, "taper": fi["H_gauge"] / fi["H_grip"],
                     "u": r["disp_u_rel_L2"], "v": r["disp_v_rel_L2"],
                     "vm": r["vm_all"]["rel_L2"],
                     "vm_fillet": r["vm_fillet"]["rel_L2"],
                     # Phase 6.1 metrics: this study's whole point is now
                     # whether the architecture can deliver a concentration
                     # factor, and a relative L2 cannot answer that.
                     "vm_peak": r["vm_all"]["peak_rel_err"],
                     "vm_fillet_Linf": r["vm_fillet"]["max"],
                     "scf": r["scf_rel_err"],
                     "scf_ref": r["scf_ref"], "scf_pinn": r["scf_pinn"],
                     "x_peak_ref": r["x_peak_ref"],
                     "x_peak_pinn": r["x_peak_pinn"],
                     "x_peak_err": r["x_peak_err"],
                     "N_err": r["N_rel_err"]})
        print(f"    geo {gi:>2} (taper {rows[-1]['taper']:.2f}): "
              f"u={rows[-1]['u']:.3e} v={rows[-1]['v']:.3e} "
              f"vm={rows[-1]['vm']:.3e} "
              f"peak={100 * rows[-1]['vm_peak']:+.1f}% "
              f"SCF={rows[-1]['scf_pinn']:.3f}/{rows[-1]['scf_ref']:.3f} "
              f"N={100 * rows[-1]['N_err']:+.1f}%", flush=True)

    hdr = "".join(f"{k:>15}" for k in REPORT_METRICS) + f"{'|N err|':>9}"
    print("\n" + "=" * len(hdr))
    print(f"  supervise={args.supervise}, conditioning={args.conditioning}, "
          f"mean over {len(rows)} held-out geometries")
    print(hdr)
    print("-" * len(hdr))
    print("".join(f"{mean_metric(rows, k):>15.3e}" for k in REPORT_METRICS)
          + f"{100 * mean_metric(rows, 'N_err'):>8.1f}%")
    ref = np.array([x["scf_ref"] for x in rows])
    prd = np.array([x["scf_pinn"] for x in rows])
    r2 = 1 - ((prd - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum()
    const = float(np.mean(np.abs(ref - ref.mean()) / ref))
    print(f"\n  K_t: R^2 = {r2:+.2f}  corr = {np.corrcoef(prd, ref)[0, 1]:+.3f}"
          f"  sd(pred) = {prd.std(ddof=1):.4f} vs reference "
          f"{ref.std(ddof=1):.4f}")
    print(f"       predicting the bank mean costs {100 * const:.2f}%; "
          f"this arm costs {100 * mean_metric(rows, 'scf'):.2f}%")
    print(f"  mean |x* error| = "
          f"{np.mean(np.abs([x['x_peak_err'] for x in rows])):.3f} x_g")
    import json
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump({"rows": rows, "args": vars(args),
                   "K_t_R2": float(r2)}, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
