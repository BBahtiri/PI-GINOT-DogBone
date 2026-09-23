#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8.2 feasibility: does the FEM reference carry over to a new family?

The project validates every prediction against an independent FEM reference,
verified on the dog-bone (patch test, uniaxial closed form, mesh convergence).
Before a second family can be scored, its reference needs the same
treatment.  This runs the *unchanged* solver (verification.fem.solver.solve)
on the open-hole quarter plate from geometry/shapes.py and checks it against
the classical linear-elastic stress concentration.

Plate: L = 40, H = 10 (width W = 20, length 80, L/W = 2), hole radius
r = d/2 with d/W in {0.2, 0.3, 0.4, 0.5}.

  K_tn = max P11 / (F / (H - r))      net-section nominal, reference config.

F is the grip reaction per unit thickness of the quarter model.  At small
load P11 = sigma_xx = the tangential stress at the hole's crown, so K_tn is
Peterson's quantity (Pilkey, Pilkey & Bi 2020, Ch. 4, fit to Howland 1930):

  K_tn = 3.000 - 3.140 (d/W) + 3.667 (d/W)^2 - 1.527 (d/W)^3

Heywood's (1952) 2 + (1 - d/W)^3 is printed alongside.  In 2D linear
elasticity with a traction-free hole K_t does not depend on Poisson's ratio
(Michell), so plane stress with this material is directly comparable.

Predictions, fixed before the first run:

  P-8.2a  small load (u_delta = 1e-3 mm, strain 2.5e-5): the mesh-extrapolated
          K_tn is within 1.5% of Peterson's value at every d/W.
  P-8.2b  the finest mesh is within 1% of the extrapolated value -- so a
          single fine solve can serve as the reference, as h_factor 1.3 does
          for the dog-bone.
  P-8.2c  grip insensitivity by design: at d/W = 0.4, clamping the grip face
          (v = 0 added) moves K_tn by less than 0.3%.  The dog-bone moved by
          1.2-4.2% (Phase 7 B1) because its grip sits next to the fillet.
  P-8.2d  at the project's load (u_delta = 1 mm, nominal strain 2.5%) K_tn
          moves by less than 3% from its small-load value.

    python -m verification.open_hole_fem
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from eval.compare_reference import LAM, MU, STATE
from geometry.shapes import OpenHolePlate, build_shape_mesh
from verification.fem.solver import assemble, solve
from verification.grip_sensitivity import solve_clamped

OUT = "verification/results/phase8/open_hole_fem.json"
L, H = 40.0, 10.0
N_R = (30, 42, 60, 85)          # elements per hole radius at the hole


def peterson(a):
    return 3.0 - 3.14 * a + 3.667 * a * a - 1.527 * a ** 3


def heywood(a):
    return 2.0 + (1.0 - a) ** 3


def ktn(sol, r):
    mesh = sol.mesh
    f, _, P, _ = assemble(mesh, sol.u, MU, LAM, STATE, want_tangent=False)
    F = float(f[2 * mesh.boundary["right_grip"]].sum())
    return float(P[:, 0, 0].max() / (F / (H - r))), F


def mesh_for(r, n_r, h_far=0.8):
    return build_shape_mesh(OpenHolePlate(L, H, r), h=h_far,
                            refine=max(1.0, h_far * n_r / r), decay=0.6)


def main():
    res = {"small": {}, "grip": None, "finite": None}
    print("Phase 8.2 -- FEM reference on the open-hole plate (unchanged solver)")
    print("=" * 92)
    print(f"{'d/W':>5}{'elements':>28}{'K_tn per mesh':>34}{'extrap':>9}"
          f"{'Peterson':>10}{'Heywood':>9}{'err':>8}")
    for a in (0.2, 0.3, 0.4, 0.5):
        r = a * H
        ks, hs, ne = [], [], []
        for n_r in N_R:
            t = time.time()
            sol = solve(mesh_for(r, n_r), 1e-3, MU, LAM, STATE, n_steps=1,
                        verbose=False)
            k, _ = ktn(sol, r)
            ks.append(k); hs.append(r / n_r); ne.append(sol.mesh.n_elem)
        # first-order (centroid-sampled CST) extrapolation from the three finest
        A = np.stack([np.ones(3), np.array(hs[1:])], -1)
        k0 = float(np.linalg.lstsq(A, np.array(ks[1:]), rcond=None)[0][0])
        pe = peterson(a)
        res["small"][str(a)] = {"n_elem": ne, "h_hole": hs, "Ktn": ks,
                                "Ktn_extrap": k0, "peterson": pe,
                                "heywood": heywood(a)}
        print(f"{a:>5.1f}{str(ne):>28}  " + " ".join(f"{k:.4f}" for k in ks)
              + f"{k0:>9.4f}{pe:>10.4f}{heywood(a):>9.4f}{100 * (k0 / pe - 1):>+7.2f}%")

    # P-8.2c: grip sensitivity, and P-8.2d: finite strain, at d/W = 0.4
    r = 0.4 * H
    m = mesh_for(r, N_R[-2])
    ref = solve(m, 1e-3, MU, LAM, STATE, n_steps=1, verbose=False)
    cl = solve_clamped(m, 1e-3)
    k_ref, _ = ktn(ref, r)
    k_cl, _ = ktn(cl, r)
    # n_steps=1, as every reference solve: the solver seeds the full-load affine
    # field, so load steps would start from an inconsistent state (first run
    # crashed on exactly that, before any P-8.2c/d verdict was computed).
    fin = solve(m, 1.0, MU, LAM, STATE, n_steps=1, verbose=False)
    k_fin, _ = ktn(fin, r)
    res["grip"] = {"Ktn_frictionless": k_ref, "Ktn_clamped": k_cl,
                   "rel": k_cl / k_ref - 1}
    res["finite"] = {"Ktn_small": k_ref, "Ktn_1mm": k_fin, "rel": k_fin / k_ref - 1}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    print(f"\nd/W = 0.4, {m.n_elem} elements: clamped grip {k_cl:.4f} vs "
          f"frictionless {k_ref:.4f} ({100 * res['grip']['rel']:+.3f}%); "
          f"u_delta = 1 mm: {k_fin:.4f} ({100 * res['finite']['rel']:+.2f}%)")

    s = res["small"]
    a_ok = all(abs(v["Ktn_extrap"] / v["peterson"] - 1) <= 0.015 for v in s.values())
    b_ok = all(abs(v["Ktn"][-1] / v["Ktn_extrap"] - 1) <= 0.01 for v in s.values())
    print("-" * 92)
    print(f"P-8.2a extrapolated K_tn within 1.5% of Peterson at every d/W  -> "
          f"{'HOLDS' if a_ok else 'FAILS'}")
    print(f"P-8.2b finest mesh within 1% of extrapolated                  -> "
          f"{'HOLDS' if b_ok else 'FAILS'}")
    print(f"P-8.2c clamping moves K_tn by < 0.3% ({100 * abs(res['grip']['rel']):.3f}%)"
          f"          -> {'HOLDS' if abs(res['grip']['rel']) < 0.003 else 'FAILS'}")
    print(f"P-8.2d 1 mm moves K_tn by < 3% ({100 * abs(res['finite']['rel']):.2f}%)"
          f"                  -> {'HOLDS' if abs(res['finite']['rel']) < 0.03 else 'FAILS'}")


if __name__ == "__main__":
    main()
