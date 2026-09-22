#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 7 A -- the boundary-condition audit.  Evaluation only.

Pre-registered in docs/phase7_boundary_conditions.md (commit 5f06482), before
this script existed.  Scores the three Phase 6.7 energy seeds on the 12
in-range and 8 out-of-range geometries against the cached FEM references
(h_factor 1.3):

  A1  Dirichlet exactness on x = 0, x = L_half and y = 0.
  A2  Natural-condition residuals.  Free edge (gauge top + arc): the PINN
      evaluated on the boundary itself, and -- like for like -- the PINN and
      the FEM on the arc's boundary elements, each as |P(c_e) . N_e| with c_e
      the element centroid and N_e the edge's outward normal.  Symmetry-plane
      and grip-face shears.
  A3  delta_BC = vm / |sigma_tt| - 1 at the PINN's boundary von Mises peak, in
      the deformed frame.  On a traction-free point of a plane-stress body
      vm = |sigma_tt| exactly, so delta_BC is the share of the peak that comes
      from violating the free-edge condition.
  A4  Error-density ratios in bands of width 0.1 H_gauge along each boundary.
  A5  Spearman of the seed-mean in-range |K_t err| against dx / H_grip.

Two clarifications, fixed here before the first run.  Every verdict is judged
on the in-range set (the set rule R1 names); out-of-range numbers are reported
alongside and decide nothing.  In A1 the grip face is x = L_half in float64,
where the FEM constrains it; the model's own end, x_max, which is L_half
rounded to float32, is reported too.

    python -m verification.bc_audit            # compute and save
    python -m verification.bc_audit --report   # verdicts from the saved file
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

from config import get_fillet_geometry
from eval.compare_reference import LAM, MU, STATE, _pinn_fields
from physics.neo_hookean import (first_piola_kirchhoff_stress,
                                 full_stress_state, von_mises_stress)
from verification.heldout_sets import IR_SEED, OOR_SEED, entries, param_sets
from verification.retest import MODELS, UD, H, load
from verification.retest import OUT as RETEST_SCORES
from verification.supervised_ceiling import _solve_cached

OUT = "verification/results/bc_audit/audit.json"
N_SEG, N_ARC, BAND = 400, 2000, 0.1


# --------------------------------------------------------------------------
# The operator at arbitrary points, with the stresses the audit needs
# --------------------------------------------------------------------------
def _latent(model, coll, params, sid):
    model.set_geometry(params)
    dt = torch.get_default_dtype()
    bpc = torch.tensor(coll.boundary_pc, dtype=dt).unsqueeze(0)
    x_m = torch.tensor([coll.x_max], dtype=dt)
    y_m = torch.tensor([coll.y_max], dtype=dt)
    model.eval()
    with torch.no_grad():
        z = model.encode(bpc, x_m, y_m, sample_ids=torch.tensor([sid]))
    return z, x_m, y_m


def _eval(model, lat, pts):
    """u, v, grad u, P and Cauchy stress at ``pts`` (N, 2), as float64 numpy."""
    z, x_m, y_m = lat
    dt = torch.get_default_dtype()
    q = torch.tensor(np.asarray(pts), dtype=dt).unsqueeze(0).requires_grad_(True)
    u_d = torch.tensor([UD], dtype=dt)
    uv, a, b, c, d = model.predict_with_grad_latent(q, z, u_d, x_m, y_m)
    P11, P12, P21, P22, _ = first_piola_kirchhoff_stress(a, b, c, d, MU, LAM, STATE)
    S11, S22, S33, S12, _ = full_stress_state(a, b, c, d, MU, LAM, STATE)
    vm = von_mises_stress(S11, S22, S33, S12)
    g = lambda t: t[0, :, 0].detach().double().cpu().numpy()
    out = {k: g(t) for k, t in dict(ux=a, uy=b, vx=c, vy=d, P11=P11, P12=P12,
                                     P21=P21, P22=P22, S11=S11, S22=S22,
                                     S12=S12, vm=vm).items()}
    out["u"] = uv[0, :, 0].detach().double().cpu().numpy()
    out["v"] = uv[0, :, 1].detach().double().cpu().numpy()
    return out


def _traction(e, N):
    """P . N for per-point reference normals N (n, 2)."""
    return np.stack([e["P11"] * N[:, 0] + e["P12"] * N[:, 1],
                     e["P21"] * N[:, 0] + e["P22"] * N[:, 1]], -1)


def _frame(e, N):
    """sigma_tt, sigma_nn, sigma_nt in the deformed boundary frame."""
    F = np.stack([np.stack([1 + e["ux"], e["uy"]], -1),
                  np.stack([e["vx"], 1 + e["vy"]], -1)], -2)       # (n, 2, 2)
    T = np.stack([-N[:, 1], N[:, 0]], -1)
    n = np.einsum("pji,pj->pi", np.linalg.inv(F), N)             # F^-T N
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = np.einsum("pij,pj->pi", F, T)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    S = np.stack([np.stack([e["S11"], e["S12"]], -1),
                  np.stack([e["S12"], e["S22"]], -1)], -2)
    q = lambda a, b: np.einsum("pi,pij,pj->p", a, S, b)
    return q(t, t), q(n, n), q(t, n)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _free_edge(fi):
    """Points and outward normals on the gauge top and the fillet arc."""
    xg, Hg, R = fi["x_g"], fi["H_gauge"], fi["R_fillet"]
    xc, yc = fi["arc_center"]
    top = np.stack([np.linspace(0.0, xg, N_SEG), np.full(N_SEG, Hg)], -1)
    n_top = np.tile([0.0, 1.0], (N_SEG, 1))
    th0 = np.arctan2(fi["dH"] - R, fi["dx"])          # grip corner
    th = np.linspace(th0, -0.5 * np.pi, N_ARC)          # ... to tangency
    arc = np.stack([xc + R * np.cos(th), yc + R * np.sin(th)], -1)
    n_arc = np.stack([xc - arc[:, 0], yc - arc[:, 1]], -1) / R
    return top, n_top, arc, n_arc


def _d_free(p, fi):
    xc, yc = fi["arc_center"]
    d_top = fi["H_gauge"] - p[:, 1]
    d_arc = np.hypot(p[:, 0] - xc, p[:, 1] - yc) - fi["R_fillet"]
    return np.where(p[:, 0] <= fi["x_g"], d_top, d_arc)


def _boundary_edges(mesh, seg):
    """(element, node a, node b) for every mesh boundary edge on ``seg``."""
    tri = mesh.tris
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    owner = np.tile(np.arange(len(tri)), 3)
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True,
                            return_counts=True)
    on = set(int(i) for i in mesh.boundary[seg])
    keep = (cnt[inv.ravel()] == 1) & np.array(
        [int(a) in on and int(b) in on for a, b in key])
    return owner[keep], key[keep, 0], key[keep, 1]


def _density_ratio(err2, w, mask):
    tot_e, tot_w = float((w * err2).sum()), float(w.sum())
    if not mask.any() or tot_e <= 0:
        return float("nan")
    return float(((w * err2)[mask].sum() / tot_e) / (w[mask].sum() / tot_w))


# --------------------------------------------------------------------------
# One model on one geometry
# --------------------------------------------------------------------------
def audit_one(model, gm, coll, sol, sid):
    p = gm.params
    fi = get_fillet_geometry(p)
    lat = _latent(model, coll, p, sid)
    mesh = sol.mesh
    cent, areas = sol.centroids, mesh.areas
    vm_f = sol.von_mises()
    gauge = cent[:, 0] < fi["x_g"]
    s_n = float(np.average(vm_f[gauge], weights=areas[gauge]))
    row = {"dx_over_Hgrip": fi["dx"] / fi["H_grip"],
           "taper": p["W_gauge"] / p["W_grip"], "sigma_n": s_n}

    # A1 -- Dirichlet exactness
    L, Hg, Hgr = fi["L_half"], fi["H_gauge"], fi["H_grip"]
    left = np.stack([np.zeros(N_SEG), np.linspace(0, Hg, N_SEG)], -1)
    grip = np.stack([np.full(N_SEG, L), np.linspace(0, Hgr, N_SEG)], -1)
    grip_m = np.stack([np.full(N_SEG, coll.x_max), np.linspace(0, Hgr, N_SEG)], -1)
    bot = np.stack([np.linspace(0, L, N_SEG), np.zeros(N_SEG)], -1)
    eL, eG, eGm, eB = (_eval(model, lat, q) for q in (left, grip, grip_m, bot))
    vmax = float(np.abs(sol.u[:, 1]).max())
    row.update(
        e_left=float(np.abs(eL["u"]).max() / UD),
        e_grip=float(np.abs(eG["u"] - UD).max() / UD),
        e_grip_model_end=float(np.abs(eGm["u"] - UD).max() / UD),
        e_bottom=float(np.abs(eB["v"]).max() / vmax))
    row["e_D"] = max(row["e_left"], row["e_grip"], row["e_bottom"])

    # A2c -- the natural component on each Dirichlet edge
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    row.update(r_left=rms(eL["P21"]) / s_n, r_bottom=rms(eB["P12"]) / s_n,
               r_grip=rms(eG["P21"]) / s_n)

    # A2a -- the free edge, on the boundary itself
    top, n_top, arc, n_arc = _free_edge(fi)
    eT, eA = _eval(model, lat, top), _eval(model, lat, arc)
    tT = np.linalg.norm(_traction(eT, n_top), axis=1) / s_n
    tA = np.linalg.norm(_traction(eA, n_arc), axis=1) / s_n
    row.update(r_top_rms=rms(tT), r_top_max=float(tT.max()),
               r_arc_rms=rms(tA), r_arc_max=float(tA.max()))

    # A3 -- the peak, decomposed in the deformed frame
    vm_b = np.concatenate([eT["vm"], eA["vm"]])
    k = int(np.argmax(vm_b))
    e_pk = {kk: np.concatenate([eT[kk], eA[kk]])[k:k + 1] for kk in eT}
    N_pk = np.concatenate([n_top, n_arc])[k:k + 1]
    x_pk = np.concatenate([top, arc])[k]
    s_tt, s_nn, s_nt = (float(v[0]) for v in _frame(e_pk, N_pk))
    row.update(x_peak_over_xg=float(x_pk[0] / fi["x_g"]),
               delta_BC=float(e_pk["vm"][0] / abs(s_tt) - 1.0),
               snn_over_stt=s_nn / s_tt, snt_over_stt=s_nt / s_tt,
               r_peak=float(np.linalg.norm(_traction(e_pk, N_pk)) / s_n))

    # A2b -- like for like on the arc's (and gauge top's) boundary elements
    xc, yc = fi["arc_center"]
    for seg in ("right_arc", "gauge_top"):
        el, a, b = _boundary_edges(mesh, seg)
        pa, pb = mesh.nodes[a], mesh.nodes[b]
        mid, ln = 0.5 * (pa + pb), np.linalg.norm(pb - pa, axis=1)
        if seg == "right_arc":
            N = np.stack([xc - mid[:, 0], yc - mid[:, 1]], -1)
            N /= np.linalg.norm(N, axis=1, keepdims=True)
        else:
            N = np.tile([0.0, 1.0], (len(el), 1))
        r_fem = np.linalg.norm(np.einsum("eij,ej->ei", sol.P[el], N), axis=1)
        e_c = _eval(model, lat, cent[el])
        r_pin = np.linalg.norm(_traction(e_c, N), axis=1)
        wrms = lambda r: float(np.sqrt((ln * r ** 2).sum() / ln.sum()))
        tag = "arc" if seg == "right_arc" else "top"
        row[f"fem_{tag}_elem_rms"] = wrms(r_fem) / s_n
        row[f"pinn_{tag}_elem_rms"] = wrms(r_pin) / s_n
        row[f"rho_{tag}"] = wrms(r_pin) / max(wrms(r_fem), 1e-300)
        row[f"n_edges_{tag}"] = int(len(el))

    # A4 -- where the error lives
    pin_c = _pinn_fields(model, p, cent, UD, "cpu", sid, coll)
    pin_n = _pinn_fields(model, p, mesh.nodes, UD, "cpu", sid, coll)
    A_node = np.zeros(mesh.n_node)
    np.add.at(A_node, mesh.tris.ravel(), np.repeat(areas / 3.0, 3))
    b = BAND * Hg
    for fld, pts, err2, w in (
            ("vm", cent, (pin_c["vm"] - vm_f) ** 2, areas),
            ("v", mesh.nodes, (pin_n["v"] - sol.u[:, 1]) ** 2, A_node)):
        bands = {"free": _d_free(pts, fi) < b, "grip": (L - pts[:, 0]) < b,
                 "x0": pts[:, 0] < b, "y0": pts[:, 1] < b}
        for name, m in bands.items():
            row[f"DR_{fld}_{name}"] = _density_ratio(err2, w, m)
    return row


# --------------------------------------------------------------------------
def compute():
    ir_p, oor_p, _ = param_sets()
    sets = {"IR": (entries(ir_p, IR_SEED), f"ir_{IR_SEED}"),
            "OOR": (entries(oor_p, OOR_SEED), f"oor_{OOR_SEED}")}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for (arm, s), path in MODELS.items():
        if arm != "energy":
            continue
        key = f"{arm}|{s}"
        if key in res:
            continue
        print(f"auditing {key} ...", flush=True)
        m = load(arm, path)
        res[key] = {}
        for name, (ents, tag) in sets.items():
            rows = []
            for i, (gm, coll) in enumerate(ents):
                sol = _solve_cached(gm.params, f"{tag}_{i}", H)
                r = audit_one(m, gm, coll, sol, 20000 + i)
                r["i"] = i
                rows.append(r)
            res[key][name] = rows
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(res, open(OUT, "w"), indent=2, default=float)
    return res


def report(res):
    sc = json.load(open(RETEST_SCORES))
    seeds = sorted(int(k.split("|")[1]) for k in res if k.startswith("energy|"))
    col = lambda rows, k: np.array([r[k] for r in rows], float)
    kt = lambda s, st: np.array([abs(r["err_u"]) for r in sc[f"energy|{s}"][st]])
    kts = lambda s, st: np.array([r["err_u"] for r in sc[f"energy|{s}"][st]])

    print("Phase 7 A -- boundary-condition audit, energy seeds", seeds)
    print("=" * 92)
    hdr = ("e_D", "r_top_rms", "r_arc_rms", "r_arc_max", "r_peak", "r_left",
           "r_bottom", "r_grip", "fem_arc_elem_rms", "pinn_arc_elem_rms",
           "rho_arc", "rho_top")
    for st in ("IR", "OOR"):
        print(f"\n[{st}]  medians over geometries (residuals in units of sigma_n)")
        print(f"{'seed':>5}" + "".join(f"{h:>13}" for h in hdr))
        for s in seeds:
            rows = res[f"energy|{s}"][st]
            print(f"{s:>5}" + "".join(f"{np.median(col(rows, h)):>13.3e}"
                                      for h in hdr))

    print("\n[IR]  the peak: delta_BC = vm/|s_tt| - 1, against the K_t error")
    print(f"{'seed':>5}{'mean|dBC|':>11}{'mean|Kt|':>10}{'ratio':>8}"
          f"{'snn/stt':>10}{'snt/stt':>10}{'x*/x_g':>8}{'rho(dBC,Kt)':>13}")
    r1_fire = 0
    for s in seeds:
        rows = res[f"energy|{s}"]["IR"]
        d = col(rows, "delta_BC")
        k = kt(s, "IR")
        ratio = np.mean(np.abs(d)) / np.mean(k)
        r1_fire += ratio >= 1.0 / 3.0
        rho = stats.spearmanr(d, kts(s, "IR")).correlation
        print(f"{s:>5}{100 * np.mean(np.abs(d)):>10.3f}%{100 * np.mean(k):>9.2f}%"
              f"{ratio:>8.2f}{np.median(col(rows, 'snn_over_stt')):>+10.2e}"
              f"{np.median(col(rows, 'snt_over_stt')):>+10.2e}"
              f"{np.median(col(rows, 'x_peak_over_xg')):>8.3f}{rho:>+13.2f}")

    print("\n[IR]  error-density ratios, medians over geometries")
    bands = ("free", "grip", "x0", "y0")
    print(f"{'seed':>5}" + "".join(f"{'vm_' + b:>10}" for b in bands)
          + "".join(f"{'v_' + b:>10}" for b in bands))
    r3_ok, free_top = 0, 0
    for s in seeds:
        rows = res[f"energy|{s}"]["IR"]
        med = {f"{f}_{b}": float(np.nanmedian(col(rows, f"DR_{f}_{b}")))
               for f in ("vm", "v") for b in bands}
        print(f"{s:>5}" + "".join(f"{med['vm_' + b]:>10.2f}" for b in bands)
              + "".join(f"{med['v_' + b]:>10.2f}" for b in bands))
        r3_ok += (med["vm_grip"] < 3) and (med["v_grip"] < 3)
        free_top += med["vm_free"] >= max(med[f"vm_{b}"] for b in bands)

    # A5
    ir_rows = res[f"energy|{seeds[0]}"]["IR"]
    dxh = col(ir_rows, "dx_over_Hgrip")
    kt_mean = np.mean([kt(s, "IR") for s in seeds], axis=0)
    sp = stats.spearmanr(kt_mean, dxh)

    # A1
    eD = np.concatenate([col(res[f"energy|{s}"][st], "e_D")
                         for s in seeds for st in ("IR", "OOR")])
    eGm = np.concatenate([col(res[f"energy|{s}"][st], "e_grip_model_end")
                          for s in seeds for st in ("IR", "OOR")])
    rho_med = [np.median(col(res[f"energy|{s}"]["IR"], "rho_arc")) for s in seeds]

    print("\nVerdicts against docs/phase7_boundary_conditions.md")
    print("-" * 92)
    a1 = ("HOLDS" if eD.max() <= 1e-6 else
          "FALSIFIED" if eD.max() > 1e-5 else "NEITHER (1e-6 < max <= 1e-5)")
    print(f"P-A1 max e_D over {len(eD)} pairs = {eD.max():.2e} "
          f"(model's own end x_max: {eGm.max():.2e})  -> {a1}")
    print(f"P-A2 median rho_arc per seed {[round(float(r), 2) for r in rho_med]} "
          f"(< 3 on every seed)  -> {'HOLDS' if max(rho_med) < 3 else 'FAILS'}")
    print(f"P-A3 mean|dBC| / mean|Kt err| >= 1/3 on {r1_fire} of {len(seeds)} seeds "
          f"(R1 fires on >= 2)  -> {'FAILS (R1 fires)' if r1_fire >= 2 else 'HOLDS'}")
    print(f"P-A4 grip DR < 3 for vm and v on {r3_ok} of {len(seeds)} seeds; "
          f"free band has the largest vm DR on {free_top} of {len(seeds)}  -> "
          f"{'HOLDS' if r3_ok >= 2 and free_top >= 2 else 'FAILS'}")
    print(f"P-A5 Spearman(|Kt err|, dx/H_grip) over 12 IR = {sp.correlation:+.2f}, "
          f"p = {sp.pvalue:.3f}  -> {'HOLDS' if sp.pvalue >= 0.05 else 'FAILS (R4 fires)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    res = json.load(open(OUT)) if a.report else compute()
    report(res)


if __name__ == "__main__":
    main()
