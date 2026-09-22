#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 7 B1 -- does the grip idealisation move K_t?  FEM only, no network.

Pre-registered in docs/phase7_boundary_conditions.md (commit 5f06482).  The
reference problem holds the grip face with u = u_delta and leaves v free (a
frictionless grip).  This re-solves the same 20 geometries -- same meshes,
h_factor 1.3 -- with v = 0 added on the grip face (a clamped grip), and
compares the concentration factor measured away from the grip corner:

    K_t^fillet = max vm over elements with x <= x_g + dx/2, / sigma_n

where sigma_n is the area-weighted gauge-mean von Mises of the same solution.
Restricting the search keeps a clamped-corner stress (Williams 1952) from
being read as the fillet peak; the unrestricted maximum is reported as well.

    P-B1: |dK_t / K_t| >= 0.5% on at least 4 of the 12 in-range geometries.

    python -m verification.grip_sensitivity
"""

from __future__ import annotations

import json
import os

import numpy as np

from config import get_fillet_geometry
from eval.compare_reference import LAM, MU, STATE
from verification.fem.solver import FemSolution, _dirichlet, assemble, newton
from verification.heldout_sets import IR_SEED, OOR_SEED, param_sets
from verification.retest import H, UD
from verification.supervised_ceiling import _solve_cached

OUT = "verification/results/bc_audit/grip_sensitivity.json"


def solve_clamped(mesh, u_delta, n_steps=1):
    """The reference problem with v = 0 added on the grip face."""
    con0, _ = _dirichlet(mesh, u_delta)
    extra = 2 * np.asarray(mesh.boundary["right_grip"], dtype=np.int64) + 1
    con = np.unique(np.concatenate([con0, extra]))

    def vals(frac):
        c0, v0 = _dirichlet(mesh, u_delta * frac)
        d = dict(zip(c0.tolist(), v0.tolist()))
        return np.array([d.get(int(k), 0.0) for k in con])

    u0 = np.zeros((mesh.n_node, 2))
    u0[:, 0] = u_delta * mesh.nodes[:, 0] / mesh.fillet_info["L_half"]
    u, hist, it = newton(mesh, con, vals, u0, MU, LAM, STATE, n_steps=n_steps,
                         verbose=False)
    _, _, P, J2D = assemble(mesh, u, MU, LAM, STATE, want_tangent=False)
    return FemSolution(mesh=mesh, u=u, u_delta=u_delta, P=P, J2D=J2D,
                       residual_history=hist, n_newton=it)


def measures(sol, fi):
    c, w = sol.centroids, sol.mesh.areas
    vm = sol.von_mises()
    g = c[:, 0] < fi["x_g"]
    s_n = float(np.average(vm[g], weights=w[g]))
    near = c[:, 0] <= fi["x_g"] + 0.5 * fi["dx"]
    k = int(np.argmax(np.where(near, vm, -np.inf)))
    xs = np.array([0.15, 0.35, 0.55, 0.75]) * fi["L_half"]
    return {"Kt_fillet": float(vm[k] / s_n), "Kt_global": float(vm.max() / s_n),
            "x_peak_over_xg": float(c[k, 0] / fi["x_g"]),
            "x_global_over_xg": float(c[int(np.argmax(vm)), 0] / fi["x_g"]),
            "sigma_n": s_n,
            "N": float(np.mean([sol.axial_resultant(x) for x in xs]))}


def main():
    ir_p, oor_p, _ = param_sets()
    res = {}
    for name, plist, seed in (("IR", ir_p, IR_SEED), ("OOR", oor_p, OOR_SEED)):
        rows = []
        for i, p in enumerate(plist):
            fi = get_fillet_geometry(p)
            ref = _solve_cached(p, f"{name.lower()}_{seed}_{i}", H)
            try:
                cl = solve_clamped(ref.mesh, UD, n_steps=1)
            except Exception:                       # noqa: BLE001
                cl = solve_clamped(ref.mesh, UD, n_steps=3)
            a, b = measures(ref, fi), measures(cl, fi)
            dv = np.linalg.norm(cl.u[:, 1] - ref.u[:, 1]) / np.linalg.norm(ref.u[:, 1])
            rows.append({"i": i, "dx_over_Hgrip": fi["dx"] / fi["H_grip"],
                         "corner_deg": float(np.degrees(np.arctan2(
                             fi["R_fillet"] - fi["dH"], fi["dx"]))),
                         "roller": a, "clamped": b,
                         "dKt_rel": b["Kt_fillet"] / a["Kt_fillet"] - 1.0,
                         "dN_rel": b["N"] / a["N"] - 1.0, "dv_rel": float(dv)})
        res[name] = rows
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    print("Phase 7 B1 -- clamped (v = 0) vs frictionless (v free) grip face")
    print("=" * 96)
    print(f"{'set':>4}{'i':>3}{'dx/Hgr':>8}{'corner':>8}{'Kt roller':>11}"
          f"{'Kt clamp':>10}{'dKt/Kt':>9}{'x*/x_g':>8}{'glob x/x_g':>11}"
          f"{'dN/N':>9}{'|dv|/|v|':>10}")
    for name, rows in res.items():
        for r in rows:
            print(f"{name:>4}{r['i']:>3}{r['dx_over_Hgrip']:>8.2f}"
                  f"{r['corner_deg']:>7.0f}°{r['roller']['Kt_fillet']:>11.4f}"
                  f"{r['clamped']['Kt_fillet']:>10.4f}{100 * r['dKt_rel']:>+8.2f}%"
                  f"{r['clamped']['x_peak_over_xg']:>8.3f}"
                  f"{r['clamped']['x_global_over_xg']:>11.3f}"
                  f"{100 * r['dN_rel']:>+8.2f}%{r['dv_rel']:>10.3f}")
    ir = np.array([abs(r["dKt_rel"]) for r in res["IR"]])
    n_big = int((ir >= 0.005).sum())
    print("-" * 96)
    print(f"P-B1: |dK_t/K_t| >= 0.5% on {n_big} of {len(ir)} in-range geometries "
          f"(mean {100 * ir.mean():.2f}%, max {100 * ir.max():.2f}%)  -> "
          f"{'HOLDS' if n_big >= 4 else 'FAILS'}")


if __name__ == "__main__":
    main()
