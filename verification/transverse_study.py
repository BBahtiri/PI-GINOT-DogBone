#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why is the transverse field not learned?

Case 1.4 established *that* the operator learns the axial response and not the
transverse one: v carries a 168% relative L2 error against the finite-element
reference, and sigma_22 is 30% of the peak stress on faces where it should be
zero. This driver asks *why*, because the three candidate explanations imply
completely different fixes:

  A. Optimisation / capacity allocation across geometries.  The loss can drive
     v -- the level-2 uniaxial driver got it to 2.5e-04 on a prismatic bar in
     4000 epochs -- but in the 128-geometry operator setting each specimen is
     seen only about 94 times, and the transverse correction may simply lose
     to the axial one. Fix: more training, or rebalancing across geometries.

  B. Loss weighting.  The traction-free condition on the gauge top is the
     *only* term that sets the lateral contraction there: with N = (0,1),
     P.N = 0 means P22 = P12 = 0. Its weight is w_trac_top = 2.0 -- the lowest
     in the config, and 50x below w_equilibrium = 100. Fix: reweight.

  C. Architecture.  The hard boundary condition is v = (y/H_grip) * phi_v.
     A constant phi_v gives v linear in y and independent of x, which is right
     in the gauge but wrong across the taper, where the true field is
     v ~ -nu * eps(x) * y. If the decoder cannot express the required
     x-dependence, no amount of training or reweighting helps. Fix: change
     the ansatz.

The experiment separates them. Fine-tuning the shipped checkpoint on a *single*
geometry removes the multi-geometry allocation problem entirely:

  - if v converges with the shipped weights, the cause is A;
  - if it converges only once w_trac_top is raised, the cause is B;
  - if it does not converge either way, the cause is C.

A fresh-initialisation run on the same geometry is the control.

Usage
-----
    python -m verification.transverse_study --runs shipped,reweighted,fresh
    python -m verification.transverse_study --geo 0 --epochs 800
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from config import (DECODER_CONFIG, ENCODER_CONFIG, LOADING_CONFIG,
                    MATERIAL_CONFIG, NONDIM_SCALES, TRAINING_CONFIG,
                    get_fillet_geometry)
from eval.compare_reference import compare_one
from geometry.banks import VAL_BANK_SEED, build_geometry_bank
from models.pi_ginot import PI_GINOT
from physics.losses import PhysicsLoss
from physics.weak_form import EnergyLoss, reference_energy
from training.manifest import set_deterministic, write_manifest
from verification.fem.mesh import build_mesh, default_h
from verification.fem.solver import solve as fem_solve

MU = MATERIAL_CONFIG["mu"]
LAM = MATERIAL_CONFIG["lam"]
STATE = MATERIAL_CONFIG["state"]

# Named loss-weight sets.  'shipped' is config.py verbatim.  A run may also be
# named ``wtop=<value>``, which varies *only* w_trac_top -- one knob at a time,
# so the effect is attributable.
WEIGHT_SETS = {
    "shipped": {},
    "reweighted": {"w_trac_top": 100.0, "w_traction_partial": 20.0},
}


def weight_overrides(name: str) -> dict:
    """Loss-weight overrides for a run name."""
    if name in WEIGHT_SETS:
        return dict(WEIGHT_SETS[name])
    if name.startswith("wtop="):
        return {"w_trac_top": float(name.split("=", 1)[1])}
    if name in ("fresh", "energy", "energy-fresh", "energy-fixed"):
        return {}
    raise ValueError(f"unknown run {name!r}; expected one of "
                     f"{sorted(WEIGHT_SETS)}, 'fresh', 'wtop=<value>', "
                     "'energy', 'energy-fixed' or 'energy-fresh'")


def is_energy(name: str) -> bool:
    return name.startswith("energy")


def _tensors(coll, device, u_delta, sample_id):
    def T(a):
        return torch.tensor(a, dtype=torch.get_default_dtype(),
                            device=device).unsqueeze(0)
    return dict(
        interior=T(coll.interior_pts).requires_grad_(True),
        tf_pts=T(coll.traction_free_pts).requires_grad_(True),
        tf_norms=T(coll.traction_free_normals),
        tf_tags=torch.tensor(coll.traction_free_tags, dtype=torch.long,
                             device=device),
        pt_pts=T(coll.partial_traction_pts).requires_grad_(True),
        pt_norms=T(coll.partial_traction_normals),
        pt_dirs=torch.tensor(coll.partial_traction_dirs, dtype=torch.long,
                             device=device),
        bpc=T(coll.boundary_pc),
        u_d=torch.tensor([u_delta], dtype=torch.get_default_dtype(), device=device),
        x_m=torch.tensor([coll.x_max], dtype=torch.get_default_dtype(), device=device),
        y_m=torch.tensor([coll.y_max], dtype=torch.get_default_dtype(), device=device),
        sid=torch.tensor([sample_id], dtype=torch.long, device=device),
    )


def _make_loss(overrides, device):
    c = dict(TRAINING_CONFIG)
    c.update(overrides)
    return PhysicsLoss(
        mu=MU, lam=LAM,
        w_equilibrium=c["w_equilibrium"], w_trac_top=c["w_trac_top"],
        w_trac_arc=c["w_trac_arc"], w_traction_partial=c["w_traction_partial"],
        w_barrier=c["w_barrier"], j_min=c["barrier_delta"],
        L0=NONDIM_SCALES["L0"], S0=NONDIM_SCALES["S0"], stress_state=STATE,
    ).to(device), c


def run_one(name: str, params: dict, coll, sol, checkpoint, epochs: int,
            lr: float, device: str, sample_id: int, track_every: int,
            seed: int = 0) -> dict:
    """Fine-tune (or train) on one geometry, tracking error against the FEM."""
    set_deterministic(seed)
    overrides = weight_overrides(name)
    from_scratch = name == "fresh"

    from_scratch = from_scratch or name == "energy-fresh"
    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).to(device)
    if not from_scratch:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(
            PI_GINOT.strip_legacy_keys(ck["model_state_dict"])[0])

    cfg = dict(TRAINING_CONFIG)
    cfg.update(overrides)
    energy_mode = is_energy(name)
    if energy_mode:
        loss_fn = EnergyLoss(
            mu=MU, lam=LAM, stress_state=STATE, quad_degree=2,
            resample=(name != "energy-fixed"),
            w_barrier=cfg["w_barrier"], j_min=cfg["barrier_delta"],
            E_scale=MATERIAL_CONFIG["E"]).to(device)
    else:
        loss_fn, cfg = _make_loss(overrides, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    u_delta = LOADING_CONFIG["u_max"]
    t = _tensors(coll, device, u_delta, sample_id)
    pi_ref = reference_energy(sol, MU, LAM, STATE)

    def measure():
        return compare_one(model, params, u_delta, None, device,
                           sample_id=sample_id, coll=coll, sol=sol)

    def energy_excess():
        """(Pi_model - Pi_reference) / Pi_reference.

        Pi >= Pi_exact for any field meeting the Dirichlet conditions, so this
        is a one-sided measure of distance from the solution -- unlike a
        residual, it cannot be small for the wrong reason.
        """
        with torch.no_grad():
            z = model.encode(t["bpc"], t["x_m"], t["y_m"],
                             sample_ids=t["sid"])
        Pi = loss_fn.energy(model, params, z, t["u_d"], t["x_m"],
                            t["y_m"])[0] if energy_mode else None
        if Pi is None:
            e = EnergyLoss(mu=MU, lam=LAM, stress_state=STATE, quad_degree=2,
                           E_scale=MATERIAL_CONFIG["E"]).to(device)
            Pi = e.energy(model, params, z, t["u_d"], t["x_m"], t["y_m"])[0]
        return float(Pi.detach()) / pi_ref - 1.0

    hist = []
    m = measure()
    hist.append({"epoch": 0, "u": m["disp_u_rel_L2"], "v": m["disp_v_rel_L2"],
                 "vm": m["vm_all"]["rel_L2"],
                 "s22": m["s22_all"]["rms_over_scale"],
                 "N_err": m["N_rel_err"], "dPi": energy_excess(),
                 "loss": None})
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        if energy_mode:
            ld = loss_fn(model, params, t["bpc"], t["u_d"], t["x_m"],
                         t["y_m"], sample_ids=t["sid"])
        else:
            ld = loss_fn(model, t["interior"], t["tf_pts"], t["tf_norms"],
                         t["tf_tags"], t["pt_pts"], t["pt_norms"],
                         t["pt_dirs"], t["bpc"], t["u_d"], t["x_m"],
                         t["y_m"], sample_ids=t["sid"])
        ld["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       cfg["grad_clip_norm"])
        opt.step()
        if epoch % track_every == 0 or epoch == epochs:
            m = measure()
            hist.append({"epoch": epoch, "u": m["disp_u_rel_L2"],
                         "v": m["disp_v_rel_L2"], "vm": m["vm_all"]["rel_L2"],
                         "s22": m["s22_all"]["rms_over_scale"],
                         "N_err": m["N_rel_err"], "dPi": energy_excess(),
                         "loss": float(ld["loss"].item())})
            h = hist[-1]
            print(f"    [{name}] epoch {epoch:5d}  loss={h['loss']:.3e}  "
                  f"u={h['u']:.3e}  v={h['v']:.3e}  vm={h['vm']:.3e}  "
                  f"s22={h['s22']:.3e}  N={100 * h['N_err']:+.1f}%  "
                  f"dPi={100 * h['dPi']:+.1f}%  [{time.time() - t0:.0f}s]",
                  flush=True)
    return {"name": name, "overrides": overrides, "history": hist,
            "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geos", default="0",
                    help="comma-separated validation-bank indices")
    ap.add_argument("--runs", default="shipped,reweighted,fresh",
                    help="named sets, 'fresh', or 'wtop=<value>'")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--track-every", type=int, default=50)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--ref-h-factor", type=float, default=2.5,
                    help="reference element size relative to default_h; the "
                         "tracked errors are insensitive to this (v 5.28 vs "
                         "5.40, vm 0.6995 vs 0.6994 between 1.0 and 2.5) and "
                         "it makes tracking 17x cheaper")
    ap.add_argument("--n-interior", type=int, default=1200)
    ap.add_argument("--out", default="verification/results/transverse")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, __import__("config"),
                   extra={"case": "transverse_study", "args": vars(args)})

    bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                               TRAINING_CONFIG["bank_geo_ranges"],
                               holes_on=False, seed=VAL_BANK_SEED)
    geos = [int(g) for g in args.geos.split(",")]
    run_names = [r.strip() for r in args.runs.split(",")]

    print("Transverse-field study — one geometry at a time, "
          "allocation removed")
    print("=" * 84)
    print(f"  geometries {geos}   runs {run_names}")
    print(f"  {args.epochs} epochs, Adam {args.lr}, device {args.device}\n")

    all_results = {}
    for gi in geos:
        gmesh, coll = bank[gi]
        params = gmesh.params
        fi = get_fillet_geometry(params)
        taper = params["W_gauge"] / params["W_grip"]

        if args.n_interior and args.n_interior < len(coll.interior_pts):
            rng = np.random.default_rng(0)
            idx = rng.choice(len(coll.interior_pts), args.n_interior,
                             replace=False)
            coll.interior_pts = coll.interior_pts[idx]

        print(f"  === val[{gi}]  L={params['L_total']:.1f} "
              f"W_grip={params['W_grip']:.1f} "
              f"W_gauge={params['W_gauge']:.1f} R={params['R_fillet']:.1f}   "
              f"taper {taper:.2f} ===", flush=True)
        fmesh = build_mesh(params, h=default_h(fi) * args.ref_h_factor)
        sol = fem_solve(fmesh, LOADING_CONFIG["u_max"], MU, LAM, STATE,
                        n_steps=1, verbose=False)
        print(f"      reference: {fmesh.n_elem} elements, "
              f"{sol.n_newton} Newton iterations", flush=True)

        results = []
        for name in run_names:
            results.append(run_one(name, params, coll, sol, args.checkpoint,
                                   args.epochs, args.lr, args.device,
                                   sample_id=100 + gi,
                                   track_every=args.track_every))
        all_results[gi] = {"params": params, "taper": taper,
                           "runs": results}
        path = os.path.join(args.out, f"geo{gi}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)

    print("\n" + "=" * 84)
    print("Final errors against the finite-element reference")
    print(f"{'geo':>5}{'taper':>7}{'run':>14}{'u rel L2':>11}"
          f"{'v rel L2':>11}{'vm rel L2':>11}{'s22/peak':>11}{'N err':>9}"
          f"{'dPi':>9}")
    print("-" * 89)
    for gi, block in all_results.items():
        for r in block["runs"]:
            b = r["history"][-1]
            print(f"{gi:>5}{block['taper']:>7.2f}{r['name']:>14}"
                  f"{b['u']:>11.3e}{b['v']:>11.3e}{b['vm']:>11.3e}"
                  f"{b['s22']:>11.3e}{100 * b['N_err']:>8.1f}%"
                  f"{100 * b['dPi']:>8.1f}%")

    if len(geos) > 1 and len(run_names) == 2:
        base, alt = run_names
        print("\n" + "-" * 80)
        print(f"Effect of {alt} versus {base}: ratio of final error "
              "(below 1 is better)")
        print(f"{'geo':>5}{'taper':>7}{'v ratio':>10}{'s22 ratio':>11}"
              f"{'vm ratio':>10}{'u ratio':>10}")
        print("-" * 53)
        ratios = []
        for gi, block in all_results.items():
            a = {r["name"]: r["history"][-1] for r in block["runs"]}
            v = a[alt]["v"] / a[base]["v"]
            s2 = a[alt]["s22"] / a[base]["s22"]
            vm = a[alt]["vm"] / a[base]["vm"]
            uu = a[alt]["u"] / a[base]["u"]
            ratios.append((v, s2, vm, uu))
            print(f"{gi:>5}{block['taper']:>7.2f}{v:>10.3f}{s2:>11.3f}"
                  f"{vm:>10.3f}{uu:>10.3f}")
        m = np.array(ratios).mean(axis=0)
        print("-" * 53)
        print(f"{'mean':>12}{m[0]:>10.3f}{m[1]:>11.3f}{m[2]:>10.3f}"
              f"{m[3]:>10.3f}")

    print(f"\nWrote per-geometry JSON to {args.out}/")


if __name__ == "__main__":
    main()
