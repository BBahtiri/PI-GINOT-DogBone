#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM references for the Phase 8 families, and their Level-3 checks.

The solver is ``verification/fem/solver.py`` unchanged -- its Newton core
takes an arbitrary Dirichlet set -- driven with each family's own Dirichlet
DOFs (geometry/families.py) on meshes from geometry/shapes.py.

Level-3 checks (small load, u_delta = 1e-3 mm), predictions fixed before the
first run:

  L3-F3  double-edge semicircular notch, t = rho = 0.01 W:     K_t within 2%
  L3-F5  single-edge semicircular notch, t = rho = 0.01 W,     of 3.065, the
         clamped:                                               semi-infinite
                                                                edge-notch value
  L3-F4  rigid inclusion, d/W = 0.05 (wide plate):  peak sigma_xx / far-field
         sigma_xx within 2% of (1 + k)(k + 2) / (4 k) = 1.535, k = 2.252
         (plane stress, nu = 0.23) -- derived in docs/phase8_tier1.md from the
         Michell solution with u = 0 on the interface; the same derivation
         gives sigma_tt = nu sigma_rr there, i.e. zero hoop strain, as a
         bonded rigid boundary must.
  G-F3/F4  clamping the grip (v = 0 added) moves K_t by < 0.3% at L/W = 1.5
  M      the sharpest in-range geometry of each family: K_t at 60 vs 85
         elements per feature radius agree within 0.5%, so 60 is the
         reference resolution.

    python -m verification.family_fem --check      # the Level-3 checks
    python -m verification.family_fem --refs       # solve and cache IR/OOR
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from config import LOADING_CONFIG
from eval.compare_reference import LAM, MU, STATE
from geometry.families import FAMILIES
from geometry.shapes import build_shape_mesh
from verification.fem.solver import FemSolution, assemble, newton

CACHE = "verification/results/fem_cache/phase8"
N_R = 60
UD = LOADING_CONFIG["u_max"]


GRADE = 0.2


def mesh_for(fam, p, n_r=N_R, h_far=0.8, seed=0):
    sh = fam.shape(p)
    return build_shape_mesh(sh, h=h_far, refine=max(1.0, h_far * n_r / sh.feature_length),
                            grade=GRADE, seed=seed)


def reaction(sol):
    """Axial force per unit thickness through the model: the grip reaction."""
    f, _, _, _ = assemble(sol.mesh, sol.u, MU, LAM, STATE, want_tangent=False)
    return float(f[2 * sol.mesh.boundary["right_grip"]].sum())


def solve_family(fam, p, u_delta, mesh=None, clamp_grip=False, n_r=N_R,
                 h_far=0.8):
    mesh = mesh if mesh is not None else mesh_for(fam, p, n_r, h_far)
    con, vals = fam.fem_dirichlet(mesh, u_delta)
    if clamp_grip:
        extra = 2 * np.asarray(mesh.boundary["right_grip"], np.int64) + 1
        d = dict(zip(con.tolist(), vals(1.0).tolist()))
        for e in extra.tolist():
            d.setdefault(int(e), 0.0)
        con = np.array(sorted(d), np.int64)
        full = np.array([d[c] for c in con])
        vals = (lambda frac, full=full: full * frac)
    # Initial guess: the transfinite lift of the family's Dirichlet data
    # (geometry/adf.py), which satisfies every Dirichlet condition exactly.
    # The affine field u_delta x / L does not on the rigid inclusion (u = 0 on
    # the arc), and Newton's first step then starts from a jump the line
    # search cannot repair (first --refs run, inclusion ir2).  For the other
    # families the lift *is* the affine field.
    import torch
    spec = fam.spec(p, u_delta)
    X = torch.tensor(mesh.nodes, dtype=torch.float64)
    u0 = np.stack([spec.lift(X, 0).numpy(), spec.lift(X, 1).numpy()], -1)
    u, hist, it = newton(mesh, con, vals, u0, MU, LAM, STATE, n_steps=1,
                         verbose=False)
    _, _, P, J2D = assemble(mesh, u, MU, LAM, STATE, want_tangent=False)
    return FemSolution(mesh=mesh, u=u, u_delta=u_delta, P=P, J2D=J2D,
                       residual_history=hist, n_newton=it)


def cauchy(sol):
    from verification.fem.solver import _F33, deformation_gradients
    F = deformation_gradients(sol.mesh, sol.u)
    J3D = sol.J2D * _F33(sol.J2D)
    s = np.einsum("eij,ekj->eik", sol.P, F) / J3D[:, None, None]
    return s[:, 0, 0], s[:, 1, 1], 0.5 * (s[:, 0, 1] + s[:, 1, 0])


def kt_of(fam, p, sol):
    return fam.kt(sol.von_mises(), sol.centroids, reaction(sol), p)[0]


def reference(fam, p, tag):
    """FEM reference at the project load, cached on disk by family and tag."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{fam.name}_{tag}_nr{N_R}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            sol, pp = pickle.load(fh)
        assert all(abs(pp[k] - p[k]) < 1e-12 for k in fam.keys), "cache/params mismatch"
        return sol
    sol = solve_family(fam, p, UD)
    with open(path, "wb") as fh:
        pickle.dump((sol, dict(p)), fh)
    return sol


# --------------------------------------------------------------------------
def checks():
    out = {}
    k = 2.252
    goodier = (1 + k) * (k + 2) / (4 * k)
    fams = FAMILIES
    print("Phase 8.2c -- Level-3 checks of the family FEM references (small load)")
    print("=" * 84)
    # L3-F3 / L3-F5: shallow semicircular notches
    for name, key in (("double_notch", "L3-F3"), ("single_notch", "L3-F5")):
        fam = fams[name]
        p = {"W": 40.0, "tW": 0.01, "rt": 1.0, "LW": 2.0 if name == "double_notch" else 2.5}
        sol = solve_family(fam, p, 1e-3, n_r=40, h_far=1.0)
        kt = kt_of(fam, p, sol)
        out[key] = {"Kt": kt, "ref": 3.065, "rel": kt / 3.065 - 1, "n_elem": sol.mesh.n_elem}
        print(f"{key}: {name:13s} t = rho = 0.01 W   K_t {kt:.4f}  vs 3.065  "
              f"({100 * (kt / 3.065 - 1):+.2f}%)  [{sol.mesh.n_elem} elements]")
    # L3-F4: wide plate with a rigid inclusion
    fam = fams["inclusion"]
    p = {"W": 80.0, "dW": 0.05, "LW": 2.0}
    sol = solve_family(fam, p, 1e-3, n_r=40, h_far=1.6)
    s11, _, _ = cauchy(sol)
    ratio = float(s11.max() / (reaction(sol) / fam.gross_height(p)))
    out["L3-F4"] = {"sxx_ratio": ratio, "ref": goodier, "rel": ratio / goodier - 1,
                    "n_elem": sol.mesh.n_elem}
    print(f"L3-F4: inclusion     d/W = 0.05        peak s_xx / far {ratio:.4f}  vs "
          f"{goodier:.4f}  ({100 * (ratio / goodier - 1):+.2f}%)  [{sol.mesh.n_elem} elements]")
    # G-F3, G-F4: grip sensitivity at the shortest in-range length
    for name, p, key in (("double_notch", {"W": 20.0, "tW": 0.2, "rt": 0.5, "LW": 1.5}, "G-F3"),
                         ("inclusion", {"W": 20.0, "dW": 0.4, "LW": 1.5}, "G-F4")):
        fam = fams[name]
        m = mesh_for(fam, p)
        a = kt_of(fam, p, solve_family(fam, p, 1e-3, mesh=m))
        b = kt_of(fam, p, solve_family(fam, p, 1e-3, mesh=m, clamp_grip=True))
        out[key] = {"frictionless": a, "clamped": b, "rel": b / a - 1}
        print(f"{key}: {name:13s} L/W = 1.5  clamped vs frictionless {100 * (b / a - 1):+.3f}%")
    # M: resolution of the sharpest in-range geometry per family
    for name, fam in fams.items():
        ir = fam.in_range()
        sharp = max(ir, key=lambda q: kt_of(fam, q, solve_family(fam, q, UD, n_r=30)))
        k60 = kt_of(fam, sharp, solve_family(fam, sharp, UD, n_r=60))
        k85 = kt_of(fam, sharp, solve_family(fam, sharp, UD, n_r=85))
        out[f"M-{name}"] = {"p": sharp, "Kt60": k60, "Kt85": k85, "rel": k60 / k85 - 1}
        print(f"M: {name:13s} sharpest in-range  K_t {k60:.4f} (60) vs {k85:.4f} (85)  "
              f"({100 * (k60 / k85 - 1):+.2f}%)")
    ok = {
        "L3-F3": abs(out["L3-F3"]["rel"]) <= 0.02, "L3-F5": abs(out["L3-F5"]["rel"]) <= 0.02,
        "L3-F4": abs(out["L3-F4"]["rel"]) <= 0.02,
        "G-F3": abs(out["G-F3"]["rel"]) < 0.003, "G-F4": abs(out["G-F4"]["rel"]) < 0.003,
        "M": all(abs(out[f"M-{n}"]["rel"]) <= 0.005 for n in fams),
    }
    print("-" * 84)
    for k_, v in ok.items():
        print(f"{k_:6s} {'HOLDS' if v else 'FAILS'}")
    os.makedirs("verification/results/phase8", exist_ok=True)
    json.dump(out, open("verification/results/phase8/family_level3.json", "w"),
              indent=2, default=float)


def refs():
    t0 = time.time()
    for name, fam in FAMILIES.items():
        for st, plist in (("ir", fam.in_range()), ("oor", fam.out_of_range())):
            for i, p in enumerate(plist):
                sol = reference(fam, p, f"{st}{i}")
        print(f"{name}: 20 references cached ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--refs", action="store_true")
    a = ap.parse_args()
    if a.check:
        checks()
    if a.refs:
        refs()
