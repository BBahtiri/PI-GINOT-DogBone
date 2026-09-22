#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The pilot: does a *physics* objective built on stress reach the ceiling?

Phase 6.4 left one honest open risk.  Supervised on stress, this architecture
reaches K_t R^2 +0.98 -- but that was with the answer key.  Whether a
physics-only objective gets to the same place was not shown, and it is the
only thing that makes the ceiling worth anything.

This runs the matched pair on exactly the ceiling's protocol, so the numbers
sit in the same table:

  - oracle-conditioned, so conditioning cannot be the explanation;
  - trained on the same eight geometries it is scored on, so generalisation
    cannot be either -- this is representability *under a physics
    objective*, the one cell the ceiling study could not fill;
  - identical architecture in both arms (the decoder is built mixed in both,
    so capacity matches; the energy arm simply never reads the stress heads);
  - identical epochs, batch, learning rate and schedule.

The two arms differ in the loss alone:

  energy   total potential energy -- displacement-only physics, the form
           that has won every comparison since Phase 2
  mixed    the constitutive gap ||P_phi - P(F(u))||^2 with equilibrium and
           tractions exact by construction (physics/mixed.py)

Both are scored through eval.compare_reference, from the displacement, so
they are comparable with every earlier arm.  The mixed arm is also scored from
its own primary output, the potential-derived stress pushed to Cauchy, and
carries two diagnostics the representation makes available: how far the two
stress fields still disagree, and the angular-momentum defect it does not
build in.

Run with:
    python -m verification.mixed_pilot --forms energy,mixed --epochs 1500
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
                    MATERIAL_CONFIG, TRAINING_CONFIG, get_fillet_geometry)
from eval.compare_reference import compare_one
from geometry.banks import TRAIN_BANK_SEED, VAL_BANK_SEED
from physics.mixed import cauchy_from_P, mixed_fields
from physics.neo_hookean import von_mises_stress
from training.manifest import set_deterministic, write_manifest
from verification.operator_study import _build_loss
from verification.supervised_ceiling import (OracleConditioned, _entries,
                                             _solve_cached)

MU, LAM = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"]
E, STATE = MATERIAL_CONFIG["E"], MATERIAL_CONFIG["state"]
UD = LOADING_CONFIG["u_max"]


def train(form, entries, args):
    set_deterministic(args.seed)
    model = OracleConditioned(ENCODER_CONFIG, dict(DECODER_CONFIG, mixed=True),
                              TRAINING_CONFIG["bank_geo_ranges"])
    # "mixed+energy" is the mixed loss with the energy added back, the
    # conditioning fallback pre-registered before the pure-gap pilot ran.
    c = dict(TRAINING_CONFIG)
    if form.startswith("mixed+energy"):
        c["w_energy_mixed"] = args.w_energy
    loss_fn = _build_loss("mixed" if form.startswith("mixed") else form, c)
    params = model.trainable_parameters()
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    d = torch.get_default_dtype()
    packed = [(gm.params, torch.tensor(c.boundary_pc, dtype=d).unsqueeze(0),
               torch.tensor([c.x_max], dtype=d), torch.tensor([c.y_max], dtype=d))
              for gm, c in entries]
    u_d = torch.tensor([UD], dtype=d)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        idx = rng.choice(len(packed), size=min(args.batch, len(packed)),
                         replace=False)
        total, logs = 0.0, []
        for k in idx:
            p, bpc, xm, ym = packed[k]
            out = loss_fn(model, p, bpc, u_d, xm, ym,
                          sample_ids=torch.tensor([int(k)]))
            total = total + out["loss"]
            logs.append(out.get("L_c_log", out.get("L_energy_log", 0.0)))
        loss = total / len(idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist.append(float(np.mean(logs)))
        if ep % 100 == 0 or ep == 1:
            print(f"    [{form}] epoch {ep:>5}  loss={loss.item():.4e}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"({(time.time() - t0) / ep:.2f} s/epoch)", flush=True)
    return model, time.time() - t0, hist


def phi_scores(model, gm, coll, sol):
    """K_t and friends from the potential-derived stress, at the centroids."""
    fi = get_fillet_geometry(gm.params)
    d = torch.get_default_dtype()
    model.set_geometry(gm.params)
    bpc = torch.tensor(coll.boundary_pc, dtype=d).unsqueeze(0)
    xm, ym = torch.tensor([coll.x_max], dtype=d), torch.tensor([coll.y_max], dtype=d)
    with torch.no_grad():
        z = model.encode(bpc, xm, ym, sample_ids=torch.tensor([999]))
    q = torch.tensor(sol.centroids, dtype=d).unsqueeze(0).requires_grad_(True)
    f = mixed_fields(model, q, z, torch.tensor([UD], dtype=d), xm, ym, fi,
                     MU, LAM, STATE, E, create_graph=False)
    out = {}
    for tag, P in (("phi", f["P_phi"]), ("F", f["P_F"])):
        s11, s22, s12, asym = cauchy_from_P(P, f["grad_u"], f["J3D"])
        vm = von_mises_stress(s11, s22, torch.zeros_like(s11), s12)
        vm = vm[0, :, 0].detach().numpy()
        out[f"vm_{tag}"] = vm
        out[f"asym_{tag}"] = float(asym.detach().abs().max()
                                   / (E * UD / fi["L_half"]))
    gap = sum((a - b) ** 2 for a, b in zip(f["P_phi"], f["P_F"]))
    out["gap_rms"] = float(gap.detach().mean().sqrt() / (E * UD / fi["L_half"]))
    out["axial_force"] = float(f["axial_force"].detach()[0])
    return out, fi


def score(model, entries, ids, args, split="val"):
    """Score ``entries`` against their finite-element references.

    ``split`` picks the reference cache tag and keeps the sample ids of the
    two splits apart.
    """
    rows = []
    for (gm, coll), gi in zip(entries, ids):
        tag = (f"val_{VAL_BANK_SEED}_{gi}" if split == "val"
               else f"train_{TRAIN_BANK_SEED}_{gi}")
        sol = _solve_cached(gm.params, tag, args.h_factor)
        model.set_geometry(gm.params)
        r = compare_one(model, gm.params, UD, None, "cpu",
                        sample_id=(100 if split == "val" else 5000) + gi,
                        coll=coll, sol=sol)
        row = {"geo": gi, "taper": gm.params["W_gauge"] / gm.params["W_grip"],
               "u": r["disp_u_rel_L2"], "v": r["disp_v_rel_L2"],
               "vm": r["vm_all"]["rel_L2"], "vm_fillet": r["vm_fillet"]["rel_L2"],
               "vm_peak": r["vm_all"]["peak_rel_err"], "scf": r["scf_rel_err"],
               "scf_ref": r["scf_ref"], "scf_pinn": r["scf_pinn"],
               "x_peak_err": r["x_peak_err"], "N_err": r["N_rel_err"]}
        # The potential-derived stress: the mixed method's primary output.
        ph, fi = phi_scores(model, gm, coll, sol)
        w, c = sol.mesh.areas, sol.centroids
        gauge = c[:, 0] < fi["x_g"]
        vm_ref = sol.von_mises()
        for tag in ("phi", "F"):
            vm = ph[f"vm_{tag}"]
            K = vm.max() / np.average(vm[gauge], weights=w[gauge])
            row[f"scf_pinn_{tag}"] = float(K)
            row[f"scf_{tag}"] = float((K - r["scf_ref"]) / r["scf_ref"])
            row[f"vm_{tag}"] = float(np.sqrt((w * (vm - vm_ref) ** 2).sum()
                                             / (w * vm_ref ** 2).sum()))
            row[f"x_peak_{tag}"] = float(c[np.argmax(vm), 0] / fi["x_g"])
        row["gap_rms"] = ph["gap_rms"]
        # The Cauchy push-forward is new code; check it against the scorer
        # every earlier arm used.  From P(F) it must reproduce compare_one's
        # displacement-derived K_t, or the phi numbers mean nothing.
        row["cauchy_check"] = abs(row["scf_pinn_F"] - r["scf_pinn"])
        row["asym_phi"] = ph["asym_phi"]
        rows.append(row)
        print(f"      geo {gi:>2} (taper {row['taper']:.2f}): vm={row['vm']:.3e}"
              f"  K_t ref {row['scf_ref']:.3f} | from u {row['scf_pinn_F']:.3f}"
              f" | from phi {row['scf_pinn_phi']:.3f}  gap {row['gap_rms']:.2e}",
              flush=True)
    return rows


def r2(rows, key):
    ref = np.array([x["scf_ref"] for x in rows])
    pr = np.array([x[key] for x in rows])
    return float(1 - ((pr - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forms", default="energy,mixed")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", default="2,15,4,6,3,11,9,22")
    ap.add_argument("--h-factor", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-energy", type=float, default=100.0,
                    help="energy weight for the mixed+energy arm; 100 makes "
                         "the two terms comparable at initialisation")
    ap.add_argument("--train-on", default="val", choices=("val", "train"),
                    help="val: train on the geometries that are scored -- the "
                         "representability protocol.  train: train on the "
                         "training bank and score the held-out geometries -- "
                         "the operator protocol")
    ap.add_argument("--bank", type=int, default=64,
                    help="training-bank size for --train-on train")
    ap.add_argument("--score-train", type=int, default=8,
                    help="with --train-on train, also score this many "
                         "training geometries, so each arm's generalisation "
                         "gap is measured inside the same run")
    ap.add_argument("--out", default="verification/results/mixed_pilot")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, cfg, extra={"case": "mixed_pilot",
                                         "args": vars(args)})
    val_idx = [int(g) for g in args.val.split(",")]
    entries = _entries(TRAINING_CONFIG["bank_val_size"],
                       bank_seed=VAL_BANK_SEED, pick=val_idx)
    for (gm, _), gi in zip(entries, val_idx):
        _solve_cached(gm.params, f"val_{VAL_BANK_SEED}_{gi}", args.h_factor)
    if args.train_on == "train":
        # The physics loss needs no reference, so the training bank needs no
        # solves; only the geometries scored on the training side do.
        train_entries = _entries(args.bank)
        train_ids = list(range(min(args.score_train, len(train_entries))))
        for i in train_ids:
            _solve_cached(train_entries[i][0].params,
                          f"train_{TRAIN_BANK_SEED}_{i}", args.h_factor)
        what = (f"trained on the {args.bank}-geometry training bank, scored "
                f"on {len(val_idx)} held-out geometries")
    else:
        train_entries, train_ids = entries, []
        what = "trained on the geometries it is scored on"

    print(f"Mixed pilot — physics-only, oracle-conditioned, {what}")
    print("=" * 92)
    results = {}
    for form in [f.strip() for f in args.forms.split(",")]:
        print(f"\n  --- {form} ---", flush=True)
        model, wall, hist = train(form, train_entries, args)
        print(f"    {args.epochs} epochs in {wall / 60:.1f} min")
        print("    held-out" if args.train_on == "train" else "    scored")
        rows = score(model, entries, val_idx, args, split="val")
        rows_train = []
        if train_ids:
            print("    training geometries")
            rows_train = score(model, [train_entries[i] for i in train_ids],
                               train_ids, args, split="train")
        results[form] = {"wall_s": wall, "rows": rows,
                         "rows_train": rows_train, "loss_hist": hist,
                         "train_on": args.train_on}
        torch.save({"model_state_dict": model.state_dict()},
                   os.path.join(args.out, f"{form}.pt"))
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(results, fh, indent=2, default=float)

    ref = np.array([x["scf_ref"] for x in next(iter(results.values()))["rows"]])
    const = float(np.mean(np.abs(ref - ref.mean()) / ref))
    print("\n" + "=" * 92)
    print(f"{'arm':>10}{'stress from':>13}{'u':>10}{'v':>10}{'vm':>10}"
          f"{'|K_t|':>9}{'R2(K)':>8}{'|dx*|':>8}")
    print("-" * 78)
    m = lambda rows, k: float(np.mean([abs(x[k]) for x in rows]))
    for form, res in results.items():
        rows = res["rows"]
        print(f"{form:>10}{'u':>13}{m(rows, 'u'):>10.3e}{m(rows, 'v'):>10.3e}"
              f"{m(rows, 'vm'):>10.3e}{100 * m(rows, 'scf'):>8.2f}%"
              f"{r2(rows, 'scf_pinn'):>8.2f}{m(rows, 'x_peak_err'):>8.3f}")
        if form.startswith("mixed"):
            print(f"{'':>10}{'phi':>13}{'':>20}{m(rows, 'vm_phi'):>10.3e}"
                  f"{100 * m(rows, 'scf_phi'):>8.2f}%"
                  f"{r2(rows, 'scf_pinn_phi'):>8.2f}")
            print(f"{'':>23}Cauchy push-forward vs compare_one: max |dK| "
                  f"{max(x['cauchy_check'] for x in rows):.1e}")
            print(f"{'':>23}constitutive gap rms {m(rows, 'gap_rms'):.2e} "
                  f"sigma_nom, angular-momentum defect "
                  f"{max(x['asym_phi'] for x in rows):.2e} sigma_nom")
    print(f"{'constant':>10}{'':>43}{100 * const:>8.2f}%{0.0:>8.2f}")
    if any(res["rows_train"] for res in results.values()):
        print("\n  generalisation gap, K_t from u — training geometries vs held-out")
        print(f"{'arm':>14}{'train R2':>10}{'held-out R2':>13}"
              f"{'train |K_t|':>13}{'held-out |K_t|':>16}{'train v':>10}"
              f"{'held-out v':>12}")
        for form, res in results.items():
            rt, rh = res["rows_train"], res["rows"]
            print(f"{form:>14}{r2(rt, 'scf_pinn'):>10.2f}{r2(rh, 'scf_pinn'):>13.2f}"
                  f"{100 * m(rt, 'scf'):>12.2f}%{100 * m(rh, 'scf'):>15.2f}%"
                  f"{m(rt, 'v'):>10.3e}{m(rh, 'v'):>12.3e}")
    if args.train_on == "val":
        print("\n  ceiling (Phase 6.4, supervised on stress, same protocol): "
              "R2 +0.98 / +0.94 without the peak term")
    else:
        print("\n  context: the Phase 6.4 ceiling (+0.94) was measured on the "
              "representability protocol and does not bound this one")


if __name__ == "__main__":
    main()
