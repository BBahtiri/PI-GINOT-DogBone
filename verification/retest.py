#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The re-test: score trained models on the fresh in-range and out-of-range sets.

Six models, oracle-conditioned, trained in the trainer loop (PI_GINOT_Trainer
via verification.ablations), bank 64, 1600 epochs, batch 4, lr 3e-4:

  energy  seed 0   verification/results/fixed_long/oracle-energy  (reused --
                   the run that produced the +0.93, identical settings)
  energy  seeds 1, 2                      verification/results/retest/energy_s*
  mixed + energy  seeds 0, 1, 2           verification/results/retest/me_s*

Each is scored on verification.heldout_sets: 12 in-range and 8 out-of-range
geometries from seeds never used for any hypothesis.  Mixed + energy models
are scored from both stress fields -- the displacement's P(F(u)) and the
potentials' P_phi -- and carry the constitutive gap.  The pre-registered
predictions these numbers are judged against are in docs/phase6_7_retest.md,
committed before any of the five new models finished training.

Run with:
    python -m verification.retest            # scores whatever exists
    python -m verification.retest --report   # verdicts from the saved scores
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

from config import (DECODER_CONFIG, ENCODER_CONFIG, LOADING_CONFIG,
                    TRAINING_CONFIG)
from eval.compare_reference import compare_one
from models.pi_ginot import PI_GINOT
from verification.heldout_sets import IR_SEED, OOR_SEED, entries, param_sets
from verification.mixed_pilot import phi_scores
from verification.supervised_ceiling import OracleConditioned, _solve_cached

UD = LOADING_CONFIG["u_max"]
H = 1.3
OUT = "verification/results/retest/scores.json"
MODELS = {("energy", 0): "verification/results/fixed_long/oracle-energy/last.pt"}
for s in (1, 2):
    MODELS[("energy", s)] = \
        f"verification/results/retest/energy_s{s}/oracle-energy/last.pt"
for s in (0, 1, 2):
    MODELS[("m+e", s)] = \
        f"verification/results/retest/me_s{s}/oracle-mixed-energy/last.pt"


def load(arm, path):
    dec = dict(DECODER_CONFIG, mixed=(arm == "m+e"))
    m = OracleConditioned(ENCODER_CONFIG, dec, TRAINING_CONFIG["bank_geo_ranges"])
    st, _ = PI_GINOT.strip_legacy_keys(
        torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
    st = {k: v for k, v in st.items() if not k.startswith("geom_aux_head.")}
    m.load_state_dict(st, strict=True)
    return m.eval()


def score_set(model, arm, ents, tag):
    rows = []
    for i, (gm, coll) in enumerate(ents):
        sol = _solve_cached(gm.params, f"{tag}_{i}", H)
        model.set_geometry(gm.params)
        r = compare_one(model, gm.params, UD, None, "cpu",
                        sample_id=20000 + i, coll=coll, sol=sol)
        row = {"i": i, "taper": gm.params["W_gauge"] / gm.params["W_grip"],
               "u": r["disp_u_rel_L2"], "v": r["disp_v_rel_L2"],
               "vm": r["vm_all"]["rel_L2"], "scf_ref": r["scf_ref"],
               "scf_u": r["scf_pinn"], "err_u": r["scf_rel_err"]}
        if arm == "m+e":
            ph, fi = phi_scores(model, gm, coll, sol)
            w, c = sol.mesh.areas, sol.centroids
            g = c[:, 0] < fi["x_g"]
            vm = ph["vm_phi"]
            K = float(vm.max() / np.average(vm[g], weights=w[g]))
            row.update(scf_phi=K, err_phi=(K - r["scf_ref"]) / r["scf_ref"],
                       gap=ph["gap_rms"])
        rows.append(row)
    return rows


def score_all():
    ir_p, oor_p, _ = param_sets()
    sets = {"IR": (entries(ir_p, IR_SEED), f"ir_{IR_SEED}"),
            "OOR": (entries(oor_p, OOR_SEED), f"oor_{OOR_SEED}")}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for (arm, s), path in MODELS.items():
        key = f"{arm}|{s}"
        # The trainer rewrites last.pt after every chunk, so its existence
        # says nothing about whether training finished -- the first version of
        # this loop started scoring a half-trained energy seed 1 (stopped
        # before it wrote anything).  The ablation harness writes result.json
        # into the run's directory only after the arm completes, so that is
        # the completion marker.
        done = os.path.join(os.path.dirname(os.path.dirname(path)), "result.json")
        if key in res or not os.path.exists(path) or not os.path.exists(done):
            continue
        print(f"scoring {key} ...", flush=True)
        m = load(arm, path)
        res[key] = {name: score_set(m, arm, e, t) for name, (e, t) in sets.items()}
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=2, default=float)
    return res


def r2(rows, k):
    ref = np.array([x["scf_ref"] for x in rows]); p = np.array([x[k] for x in rows])
    return float(1 - ((p - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def mae(rows, k):
    return float(100 * np.mean([abs(x[k]) for x in rows]))


def report(res):
    seeds = lambda arm: sorted(int(k.split("|")[1]) for k in res if k.startswith(arm + "|"))
    E, M = seeds("energy"), seeds("m+e")
    get = lambda arm, s, st: res[f"{arm}|{s}"][st]
    ir0 = get("energy", E[0], "IR")
    ref = np.array([x["scf_ref"] for x in ir0])
    const_ir = 100 * np.mean(np.abs(ref - ref.mean()) / ref)
    print(f"Re-test — energy seeds {E}, mixed + energy seeds {M}")
    print("=" * 88)
    print(f"{'':>26}{'IR R2':>9}{'IR |K|':>9}{'OOR |K|':>10}{'OOR bias':>10}"
          f"{'IR v':>9}{'IR vm':>9}")
    rows_out = {}
    for arm in ("energy", "m+e"):
        for s in (E if arm == "energy" else M):
            ir, oor = get(arm, s, "IR"), get(arm, s, "OOR")
            fields = [("u", "scf_u", "err_u")] + ([("phi", "scf_phi", "err_phi")]
                                                  if arm == "m+e" else [])
            for src, k, e in fields:
                bias = 100 * np.mean([x[e] for x in oor])
                v = np.mean([x["v"] for x in ir]) if src == "u" else float("nan")
                vm = np.mean([x["vm"] for x in ir]) if src == "u" else float("nan")
                print(f"{arm + ' s' + str(s) + ' (' + src + ')':>26}"
                      f"{r2(ir, k):>+9.2f}{mae(ir, e):>8.2f}%{mae(oor, e):>9.2f}%"
                      f"{bias:>+9.2f}%{v:>9.3f}{vm:>9.4f}")
                rows_out.setdefault((arm, src), []).append(
                    (r2(ir, k), mae(ir, e), mae(oor, e)))
    print(f"{'constant (IR)':>26}{0.0:>+9.2f}{const_ir:>8.2f}%")

    print("\nVerdicts against docs/phase6_7_retest.md")
    print("-" * 88)
    eu = np.array(rows_out[("energy", "u")]); mu = np.array(rows_out[("m+e", "u")])
    mp = np.array(rows_out.get(("m+e", "phi"), np.empty((0, 3))))
    # P1 -- in-range skill of the energy form
    ok1 = eu[:, 0].mean() >= 0.6 and all(eu[:, 1] < const_ir)
    print(f"P1 energy in-range R2 mean {eu[:, 0].mean():+.2f} (>= +0.60), "
          f"|K| below constant on {int((eu[:, 1] < const_ir).sum())}/{len(eu)} seeds"
          f"  -> {'HOLDS' if ok1 else 'FAILS'}")
    # P2 -- m+e vs energy in range
    if len(mu) >= 2 and len(eu) >= 2:
        t = stats.ttest_ind(mu[:, 0], eu[:, 0], equal_var=False)
        print(f"P2 in-range R2: m+e {mu[:, 0].mean():+.2f} vs energy {eu[:, 0].mean():+.2f}, "
              f"Welch two-sided p = {t.pvalue:.3f}  -> "
              f"{'no detectable difference (as predicted)' if t.pvalue > 0.05 else ('m+e better' if mu[:, 0].mean() > eu[:, 0].mean() else 'm+e worse')}")
    # P3 -- phi extrapolates
    if len(mp):
        n = min(len(mp), len(eu))
        half = all(mp[i, 2] < 0.5 * eu[i, 2] for i in range(n))
        own = all(mp[i, 2] < mu[i, 2] for i in range(len(mp)))
        print(f"P3 OOR |K|: phi {mp[:, 2].mean():.2f}% vs energy(u) {eu[:, 2].mean():.2f}% "
              f"vs m+e(u) {mu[:, 2].mean():.2f}%; phi < half of energy on every "
              f"paired seed: {half}; phi < own u on every seed: {own}"
              f"  -> {'HOLDS' if half and own else 'FAILS'}")
    # P4 -- the gap flags failure
    gaps, pooled_g, pooled_e = [], [], []
    for s in M:
        ir, oor = get("m+e", s, "IR"), get("m+e", s, "OOR")
        ratio = np.mean([x["gap"] for x in oor]) / np.mean([x["gap"] for x in ir])
        rho = stats.spearmanr([x["gap"] for x in ir + oor],
                              [abs(x["err_u"]) for x in ir + oor])[0]
        gaps.append((ratio, rho))
        pooled_g += [x["gap"] for x in ir + oor]
        pooled_e += [abs(x["err_u"]) for x in ir + oor]
    if gaps:
        rp = stats.spearmanr(pooled_g, pooled_e)
        ok4 = all(g[0] >= 2 for g in gaps) and all(g[1] > 0 for g in gaps) \
            and rp.pvalue < 0.01
        print(f"P4 gap OOR/IR ratio per seed {[round(g[0], 2) for g in gaps]} (>= 2), "
              f"Spearman per seed {[round(g[1], 2) for g in gaps]} (> 0), pooled "
              f"rho {rp.correlation:+.2f} p {rp.pvalue:.1e} (< 0.01)"
              f"  -> {'HOLDS' if ok4 else 'FAILS'}")
    # P5 -- the extrapolation failure replicates
    ok5 = all(eu[:, 2] > 3 * eu[:, 1])
    print(f"P5 energy OOR/IR |K| ratio per seed "
          f"{[round(a / b, 1) for a, b in zip(eu[:, 2], eu[:, 1])]} (> 3)"
          f"  -> {'HOLDS' if ok5 else 'FAILS'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    res = json.load(open(OUT)) if args.report else score_all()
    report(res)


if __name__ == "__main__":
    main()
