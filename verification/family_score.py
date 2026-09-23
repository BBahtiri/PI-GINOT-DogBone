#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score the Tier-1 per-family operators against their FEM references and judge
the predictions of docs/phase8_tier1.md (committed before any training).

For every family, seed and held-out geometry (12 in-range, 8 out-of-range):

  K_t      peak Cauchy vm over the peak region / (N / A_gross), network and
           FEM each from their own field (N: grip reaction for the FEM,
           section integral of P11 at x = 0.75 L for the network), both at
           the FEM's element centroids
  u, v     relative L2 at the FEM nodes
  vm       area-weighted relative L2 at the centroids
  x_peak   location of the peak element, network vs FEM

P-T4 (the dog-bone through the family loop) is scored with the 6.7 scorer
itself (verification.retest.score_set), so it is compared like for like.

    python -m verification.family_score            # score what has finished
    python -m verification.family_score --report   # verdicts from the saved file
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

from eval.compare_reference import LAM, MU, STATE
from geometry.families import FAMILIES
from physics.neo_hookean import (first_piola_kirchhoff_stress,
                                 full_stress_state, von_mises_stress)
from training.family_trainer import UD, load
from verification.family_fem import reaction, reference

ROOT = "verification/results/phase8/tier1"
OUT = os.path.join(ROOT, "scores.json")
SEEDS = (0, 1, 2)


def _done(fam, seed):
    d = os.path.join(ROOT, f"{fam}_s{seed}")
    return os.path.exists(os.path.join(d, "result.json")), os.path.join(d, "last.pt")


def _fields(model, fam, p, pts):
    dt = torch.get_default_dtype()
    model.set_geometry(p)
    L, Hs = fam.extent(p)
    x_m, y_m = torch.tensor([L], dtype=dt), torch.tensor([Hs], dtype=dt)
    with torch.no_grad():
        z = model.encode(torch.zeros(1, 1, 2, dtype=dt), x_m, y_m)
    out = {k: [] for k in ("u", "v", "vm", "P11")}
    for chunk in np.array_split(np.asarray(pts), max(1, len(pts) // 4000)):
        q = torch.tensor(chunk, dtype=dt).unsqueeze(0).requires_grad_(True)
        uv, a, b, c, d = model.predict_with_grad_latent(q, z, torch.tensor([UD], dtype=dt),
                                                        x_m, y_m)
        P11 = first_piola_kirchhoff_stress(a, b, c, d, MU, LAM, STATE)[0]
        S11, S22, S33, S12, _ = full_stress_state(a, b, c, d, MU, LAM, STATE)
        vm = von_mises_stress(S11, S22, S33, S12)
        g = lambda t: t[0, :, 0].detach().double().numpy()
        out["u"].append(uv[0, :, 0].detach().double().numpy())
        out["v"].append(uv[0, :, 1].detach().double().numpy())
        out["vm"].append(g(vm)); out["P11"].append(g(P11))
    return {k: np.concatenate(v) for k, v in out.items()}


def _section_force(model, fam, p, n=256):
    Hs = fam.gross_height(p)
    xg, wg = np.polynomial.legendre.leggauss(n)
    y = 0.5 * Hs * (xg + 1.0)
    pts = np.stack([np.full(n, fam.section_x(p)), y], -1)
    return float(0.5 * Hs * (wg * _fields(model, fam, p, pts)["P11"]).sum())


def score_geometry(model, fam, p, sol):
    c, w = sol.centroids, sol.mesh.areas
    vm_f = sol.von_mises()
    K_ref, _, k_ref = fam.kt(vm_f, c, reaction(sol), p)
    fc = _fields(model, fam, p, c)
    fn = _fields(model, fam, p, sol.mesh.nodes)
    N_p = _section_force(model, fam, p)
    K_p, _, k_p = fam.kt(fc["vm"], c, N_p, p)
    rel = lambda a, b, ww=None: float(np.sqrt(np.average((a - b) ** 2, weights=ww))
                                      / np.sqrt(np.average(b ** 2, weights=ww)))
    return {"K_ref": K_ref, "K_pinn": K_p, "err": K_p / K_ref - 1.0,
            "N_err": N_p / reaction(sol) - 1.0,
            "u": rel(fn["u"], sol.u[:, 0]), "v": rel(fn["v"], sol.u[:, 1]),
            "vm": rel(fc["vm"], vm_f, w),
            "x_peak_ref": c[k_ref].tolist(), "x_peak_pinn": c[k_p].tolist()}


def score_all():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name, fam in FAMILIES.items():
        for s in SEEDS:
            key = f"{name}|{s}"
            ok, path = _done(name, s)
            if key in res or not ok:
                continue
            print(f"scoring {key} ...", flush=True)
            model = load(name, path)
            res[key] = {}
            for st, plist in (("IR", fam.in_range()), ("OOR", fam.out_of_range())):
                res[key][st] = [score_geometry(model, fam, p, reference(fam, p, f"{st.lower()}{i}"))
                                for i, p in enumerate(plist)]
            json.dump(res, open(OUT, "w"), indent=2)
    # P-T4: dog-bone through the family loop, scored by the 6.7 scorer
    ok, path = _done("dogbone", 0)
    if ok and "dogbone|0" not in res:
        from verification.heldout_sets import IR_SEED, entries, param_sets
        from verification.retest import score_set
        ir_p, _, _ = param_sets()
        model = load("dogbone", path)
        res["dogbone|0"] = {"IR": score_set(model, "energy", entries(ir_p, IR_SEED),
                                            f"ir_{IR_SEED}")}
        json.dump(res, open(OUT, "w"), indent=2)
    return res


def _r2(ref, pred):
    ref, pred = np.asarray(ref), np.asarray(pred)
    return float(1 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def report(res):
    print("Phase 8 Tier 1 -- per-family operators against the FEM")
    print("=" * 100)
    print(f"{'family':<14}{'seed':>5}{'IR R2':>8}{'IR |err|':>10}{'const':>8}{'IR bias':>9}"
          f"{'OOR |err|':>11}{'OOR bias':>10}{'u L2':>9}{'v L2':>9}{'vm L2':>9}{'N err':>8}")
    verdict = {}
    for name in FAMILIES:
        seeds = [s for s in SEEDS if f"{name}|{s}" in res]
        beats, neg = 0, 0
        pooled = []
        for s in seeds:
            ir, oor = res[f"{name}|{s}"]["IR"], res[f"{name}|{s}"]["OOR"]
            ref = np.array([r["K_ref"] for r in ir]); pk = np.array([r["K_pinn"] for r in ir])
            e = np.array([r["err"] for r in ir]); eo = np.array([r["err"] for r in oor])
            const = float(np.mean(np.abs(ref - ref.mean()) / ref))
            beats += np.mean(np.abs(e)) < const
            neg += np.mean(e) < 0
            pooled += [(abs(r["err"]), r["K_ref"]) for r in ir]
            med = lambda k: np.median([r[k] for r in ir])
            print(f"{name:<14}{s:>5}{_r2(ref, pk):>+8.2f}{100 * np.mean(np.abs(e)):>9.2f}%"
                  f"{100 * const:>7.2f}%{100 * np.mean(e):>+8.2f}%{100 * np.mean(np.abs(eo)):>10.2f}%"
                  f"{100 * np.mean(eo):>+9.2f}%{med('u'):>9.1e}{med('v'):>9.1e}{med('vm'):>9.1e}"
                  f"{100 * np.mean([r['N_err'] for r in ir]):>+7.2f}%")
        verdict[name] = {"n": len(seeds), "beats": int(beats), "neg": int(neg),
                         "pooled": pooled}
    print("-" * 100)
    for name, v in verdict.items():
        if v["n"] == 0:
            continue
        print(f"P-T1 {name:<14} beats the constant on {v['beats']} of {v['n']} seeds"
              + (f"  -> {'HOLDS' if v['beats'] >= 2 else 'FAILS'}" if v["n"] == 3 else "  (pending seeds)"))
    for name in ("open_hole", "double_notch", "single_notch"):
        v = verdict[name]
        if v["n"]:
            print(f"P-T2 {name:<14} under-predicts on {v['neg']} of {v['n']} seeds"
                  + (f"  -> {'HOLDS' if v['neg'] == 3 else 'FAILS'}" if v["n"] == 3 else "  (pending seeds)"))
    v = verdict["double_notch"]
    if v["n"]:
        a = np.array(v["pooled"])
        sp = stats.spearmanr(a[:, 0], a[:, 1])
        print(f"P-T3 double_notch Spearman(|err|, K_ref) {sp.correlation:+.2f}, p {sp.pvalue:.3g}, "
              f"n {len(a)}" + (f"  -> {'HOLDS' if sp.correlation > 0 and sp.pvalue < 0.05 else 'FAILS'}"
                              if v["n"] == 3 else "  (pending seeds)"))
    if "dogbone|0" in res:
        ir = res["dogbone|0"]["IR"]
        ref = np.array([r["scf_ref"] for r in ir]); pk = np.array([r["scf_u"] for r in ir])
        mae = 100 * np.mean([abs(r["err_u"]) for r in ir]); r2 = _r2(ref, pk)
        ok = 0.5 <= mae <= 2.4 and r2 >= 0.19
        print(f"P-T4 dog-bone, family loop, seed 0: IR R2 {r2:+.2f}, |err| {mae:.2f}%  "
              f"(6.7 trainer loop: +0.85/+0.41/+0.57, 0.97/1.93/1.64%)  -> {'HOLDS' if ok else 'FAILS'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(json.load(open(OUT)) if a.report else score_all())
