#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level 0 of the verification hierarchy: are the physics *operators* right?

Every check here drives the production code path -- ``equilibrium_residual``,
``traction``, ``first_piola_kirchhoff_stress``, the real collocation sampler
and the real boundary normals -- with a closed-form field whose exact answer
is known.  No training, no fitting, no network: these run in seconds and
either pass to machine precision or reveal a defect.

They are the checks that gate everything above them.  A sign error in the
traction pairing or an inward-pointing fillet normal cannot be seen in a
training curve: the optimiser simply converges to a different field and every
reported stress is quietly wrong.

Cases
-----
1. Rigid translation and finite rigid rotation give exactly zero stress
   (the rotation is the objectivity check -- it has a non-zero displacement
   gradient, so it catches a law that is only accidentally right at F = I).
2. Objectivity of P: superposing a rotation Q on any F must give P -> Q P.
3. An affine field must give a spatially constant, exactly correct stress at
   every point of the irregular interior set -- the finite-strain patch test.
   (Its divergence is trivially zero and is checked against a *non-zero*
   known value in verification/mms.py instead, which is where the
   second-order AD chain is actually exercised.)
4. On the real dog-bone geometry, a laterally-free uniaxial stretch makes
   P12 = P21 = P22 = 0. Every boundary condition of the quarter model is then
   satisfied exactly *except* on the fillet arc, so the traction operator,
   the segment tagging and the normal orientation are all checked against a
   known answer on the actual curved domain.
5. The section resultant of that field equals 2*H*P11 in closed form, which
   checks the quadrature the force-balance loss depends on.

Run with:  python -m verification.operators
"""

from __future__ import annotations

import numpy as np
import torch

from config import GEOMETRY_DEFAULT, MATERIAL_CONFIG
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import (_rejection_sample_interior,
                                         generate_dogbone)
from physics.equilibrium import equilibrium_residual, traction
from physics.neo_hookean import (deformation_gradient,
                                 first_piola_kirchhoff_stress)
from verification.analytic_fields import (AffineField, AnalyticModel,
                                          HomogeneousUniaxial, RigidRotation,
                                          RigidTranslation)

MU = MATERIAL_CONFIG["mu"]
LAM = MATERIAL_CONFIG["lam"]
E = MATERIAL_CONFIG["E"]
STATE = MATERIAL_CONFIG["state"]

DTYPE = torch.float64

# Which traction component each non-free segment of the quarter model
# actually constrains (the complement of the hard Dirichlet condition).
PARTIAL_DIRS = {"bottom": 0, "left_symmetry": 1, "right_grip": 1}


def _P_of_F(F: np.ndarray, state: str = STATE):
    """1st Piola stress for a constant deformation gradient, as a 2x2 array."""
    H = np.asarray(F, dtype=float) - np.eye(2)
    g = [torch.full((1, 1, 1), H[i, j], dtype=DTYPE)
         for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    P11, P12, P21, P22, _ = first_piola_kirchhoff_stress(*g, MU, LAM, state)
    return np.array([[P11.item(), P12.item()],
                     [P21.item(), P22.item()]])


def check_zero_stress_states() -> dict:
    """Rigid translation and finite rotation must give P = 0 exactly."""
    out = {}
    for name, F in (("translation", np.eye(2)),
                    ("rotation(0.35 rad)", RigidRotation(0.35).Q)):
        P = _P_of_F(F)
        out[name] = float(np.abs(P).max() / E)
    return out


def check_objectivity(thetas=(0.1, 0.35, 1.0, 2.5)) -> dict:
    """P(QF) must equal Q P(F) for every rotation Q.

    Frame indifference of the constitutive law, checked on a deformation that
    is neither small nor symmetric.
    """
    F = np.array([[1.08, 0.05], [-0.03, 0.96]])
    P = _P_of_F(F)
    out = {}
    for th in thetas:
        c, s = np.cos(th), np.sin(th)
        Q = np.array([[c, -s], [s, c]])
        lhs = _P_of_F(Q @ F)
        rhs = Q @ P
        out[f"theta={th}"] = float(np.abs(lhs - rhs).max()
                                   / max(np.abs(rhs).max(), 1e-300))
    return out


def check_affine_patch_test(n: int = 4000, seed: int = 11) -> dict:
    """Constant F must give a spatially constant, exactly correct stress.

    The finite-strain patch test.  Displacement gradients are obtained by
    autograd on the closed-form field -- not prescribed -- so this checks
    ``deformation_gradient``, the plane-stress closure and
    ``first_piola_kirchhoff_stress`` pointwise over an irregular point set
    inside the real curved domain.

    Two numbers per case:
        spread : max_ij std(P_ij) / |P| , the plan's std/mean criterion
        error  : max_ij |P_ij(x) - P_ij(F)| / |P| against the closed form

    Div P is *not* checked here.  For an affine field the stress is a
    constant with no autograd graph, so a divergence check would be testing
    the constant-stress shortcut in equilibrium_residual rather than the
    differentiation chain.  The second-order chain is verified against a
    non-zero known divergence in verification/mms.py instead.
    """
    mesh = generate_dogbone(GEOMETRY_DEFAULT, rng=np.random.default_rng(0))
    pts = _rejection_sample_interior(mesh.fillet_info, [], n,
                                     np.random.default_rng(seed))
    out = {}
    for name, F in (
        ("uniaxial l1=1.05", np.diag([1.05, 1.0])),
        ("shear", np.array([[1.0, 0.08], [0.0, 1.0]])),
        ("general", np.array([[1.08, 0.05], [-0.03, 0.96]])),
        ("large l1=1.30", np.diag([1.30, 0.90])),
    ):
        model = AnalyticModel(AffineField(F), dtype=DTYPE)
        q = model.query(pts.astype(np.float64))
        _, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(q)
        P = first_piola_kirchhoff_stress(du_dx, du_dy, dv_dx, dv_dy,
                                         MU, LAM, STATE)[:4]
        P_ref = _P_of_F(F)
        scale = max(np.abs(P_ref).max(), 1e-300)
        spread = max(float(c.detach().std()) for c in P) / scale
        err = max(float((c.detach() - r).abs().max())
                  for c, r in zip(P, P_ref.ravel())) / scale
        out[f"{name} spread"] = spread
        out[f"{name} error"] = err
    return out


def check_dogbone_boundary_conditions(params: dict = None,
                                      n_per_segment: int = 800) -> dict:
    """A laterally-free uniaxial stretch on the real quarter-model geometry.

    For that field P12 = P21 = P22 = 0, so with the correct outward normals:
        gauge_top      (N = +e_y):  P.N = 0        both components
        bottom         (N = -e_y):  P.N = 0        both components
        left_symmetry  (N = -e_x):  (P.N)_y = 0    enforced component
        right_grip     (N = +e_x):  (P.N)_y = 0    enforced component
        right_arc                  :  non-zero, and must be, since a uniform
                                      stretch is not the dog-bone solution

    Anything non-zero in the first four rows is a defect in the normals, the
    segment tagging or the traction index pairing -- none of which a training
    curve can reveal.  Values are ||P.N||/E.
    """
    params = params or GEOMETRY_DEFAULT
    mesh = generate_dogbone(params, n_pts_per_segment=n_per_segment,
                            rng=np.random.default_rng(0))
    field = HomogeneousUniaxial(1.05, MU, LAM)
    model = AnalyticModel(field, dtype=DTYPE)
    u_d, x_m, y_m = model.eval_mode_args(1.0, 1.0, 1.0)

    out = {"_P11_exact": field.P11}
    for seg in mesh.boundary_segments:
        q = model.query(seg.points.astype(np.float64))
        nrm = torch.tensor(seg.normals.astype(np.float64),
                           dtype=DTYPE).unsqueeze(0)
        _, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(q)
        P = first_piola_kirchhoff_stress(du_dx, du_dy, dv_dx, dv_dy,
                                         MU, LAM, STATE)
        tx, ty = traction(*P[:4], nrm)
        if seg.name in PARTIAL_DIRS:
            comp = tx if PARTIAL_DIRS[seg.name] == 0 else ty
            val = comp.abs().max().item()
        else:
            val = torch.sqrt(tx ** 2 + ty ** 2).max().item()
        out[seg.name] = float(val / E)
    return out


def check_section_resultant(params: dict = None, n_y: int = 2048) -> dict:
    """The quadrature behind the force-balance loss, against a closed form.

    Under the uniaxial field the resultant of any gauge section is exactly
    2 * H_gauge * P11, independent of x.
    """
    params = params or GEOMETRY_DEFAULT
    mesh = generate_dogbone(params, rng=np.random.default_rng(0))
    fi = mesh.fillet_info
    field = HomogeneousUniaxial(1.05, MU, LAM)
    model = AnalyticModel(field, dtype=DTYPE)

    exact = 2.0 * fi["H_gauge"] * field.P11
    out = {"_N_exact": exact}
    for frac in (0.05, 0.25, 0.50, 0.75):
        x_k = frac * fi["x_g"]
        y_edges = np.linspace(0.0, fi["H_gauge"], n_y + 1)
        y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
        dy = fi["H_gauge"] / n_y
        pts = np.stack([np.full(n_y, x_k), y_mid], axis=-1)
        q = model.query(pts)
        _, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(q)
        P11 = first_piola_kirchhoff_stress(du_dx, du_dy, dv_dx, dv_dy,
                                           MU, LAM, STATE)[0]
        N = 2.0 * dy * P11[0, :, 0].sum().item()
        out[f"x/x_g={frac}"] = abs(N - exact) / exact
    return out


def run_all() -> dict:
    """Every level-0 check, as a flat dict of relative errors."""
    return {
        "zero_stress": check_zero_stress_states(),
        "objectivity": check_objectivity(),
        "patch_test": check_affine_patch_test(),
        "dogbone_bcs": check_dogbone_boundary_conditions(),
        "resultant": check_section_resultant(),
    }


def main():
    report = run_all()
    print("Level 0 — physics operator verification")
    print("=" * 62)
    print("\nAll figures are relative errors; exact answers are 0.\n")
    for group, vals in report.items():
        print(f"  {group}")
        for k, v in vals.items():
            if k.startswith("_"):
                print(f"    {k[1:]:<26} {v:.9f}   (reference value)")
            else:
                print(f"    {k:<26} {v:.3e}")
        print()
    worst = max(v for vals in report.values()
                for k, v in vals.items() if not k.startswith("_")
                and "right_arc" not in k)
    print("=" * 62)
    print(f"Worst relative error (excluding the fillet arc, which is "
          f"non-zero by construction): {worst:.3e}")


if __name__ == "__main__":
    main()
