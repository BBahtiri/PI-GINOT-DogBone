#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification of the finite-element reference itself.

A reference solution nobody has checked is not a reference. Before any
PINN-versus-FEM number is quoted, the FEM has to demonstrate that it
reproduces answers already known independently.

Three checks, in increasing weakness of the available truth:

1. **Patch test.**  Prescribe an affine displacement on *every* boundary
   node.  The exact solution is that affine field, and a correct
   displacement-based element reproduces it to round-off with a spatially
   constant stress.  This is the strongest statement available: it tests
   assembly, the constitutive evaluation, the boundary treatment and the
   Newton solve against an exact answer, on the real graded mesh with its
   real element shapes.

2. **Uniaxial extension.**  On a near-prismatic specimen the homogeneous
   laterally-free solution is exact, and the loading is the same
   displacement-controlled setup the dog-bone uses.  This is the same case
   level 2 puts the PINN through, so the two are directly comparable.

3. **Mesh convergence on the dog-bone.**  No exact answer exists, so the
   quantity of interest is extrapolated over successive refinements and the
   remaining discretisation error is reported as the reference's error bar.
   Constant-strain triangles are first-order in stress, which is exactly why
   this is not optional.

Run with:  python -m verification.fem.verify_fem
"""

from __future__ import annotations

import numpy as np

from config import GEOMETRY_DEFAULT, MATERIAL_CONFIG, get_fillet_geometry
from physics.uniaxial import uniaxial_lateral_stretch, uniaxial_P11
from verification.fem.mesh import SEGMENTS, build_mesh, summarize
from verification.fem.solver import (deformation_gradients, newton, solve)

MU = MATERIAL_CONFIG["mu"]
LAM = MATERIAL_CONFIG["lam"]
E = MATERIAL_CONFIG["E"]
STATE = MATERIAL_CONFIG["state"]

# Near-prismatic, same specimen the PINN level-2 driver uses.
PRISMATIC = dict(L_total=54.0, W_grip=20.0, W_gauge=19.99, R_fillet=8.0,
                 holes=[])


def _boundary_node_ids(mesh) -> np.ndarray:
    return np.unique(np.concatenate([mesh.boundary[s] for s in SEGMENTS]))


def run_patch_test(h: float = 1.0, perturb_frac: float = 0.05,
                   verbose: bool = False) -> dict:
    """Affine displacement on the whole boundary; interior must follow.

    Stated as two things, because the second alone would be a statement about
    the solver rather than about the element:

    (a) *The affine field is an exact solution of the discrete equations.*
        Assemble at u = (F-I)X with affine boundary data and check the
        free-DOF residual is zero. This is the patch test proper -- it tests
        assembly, the constitutive evaluation and the quadrature against an
        exact answer on the real graded mesh.

    (b) *It is a stable solution.*  Perturb the interior and confirm Newton
        returns to it. The perturbation is scaled to the smallest element
        (``perturb_frac`` of h_min) because white noise at the scale of the
        *coarse* element size inverts the refined elements at the fillet --
        which is a statement about the perturbation, not about the mesh.
    """
    out = {}
    mesh = build_mesh(GEOMETRY_DEFAULT, h=h)
    bnd = _boundary_node_ids(mesh)
    con = np.sort(np.concatenate([2 * bnd, 2 * bnd + 1]))
    free = np.setdiff1d(np.arange(2 * mesh.n_node), con)
    scale = MU * mesh.areas.sum()
    h_min = h / 6.0                      # build_mesh default refine factor
    rng = np.random.default_rng(0)

    from verification.fem.solver import assemble, constitutive

    for name, Fbar in (
        ("uniaxial", np.diag([1.05, 1.0])),
        ("shear", np.array([[1.0, 0.06], [0.0, 1.0]])),
        ("general", np.array([[1.08, 0.05], [-0.03, 0.96]])),
    ):
        u_exact = mesh.nodes @ (Fbar - np.eye(2)).T

        # (a) residual and stress at the exact affine field
        f, _, P, _ = assemble(mesh, u_exact, MU, LAM, STATE,
                              want_tangent=False)
        F = deformation_gradients(mesh, u_exact)
        P_ref, _, _ = constitutive(Fbar[None], MU, LAM, STATE,
                                   want_tangent=False)
        p_scale = max(np.abs(P_ref).max(), 1e-300)
        out[f"{name}: residual at u*"] = float(
            np.linalg.norm(f[free]) / scale)
        out[f"{name}: max |F - Fbar|"] = float(np.abs(F - Fbar).max())
        out[f"{name}: std(P)/|P|"] = float(
            P.reshape(-1, 4).std(0).max() / p_scale)
        out[f"{name}: max |P - P(Fbar)|/|P|"] = float(
            np.abs(P - P_ref[0]).max() / p_scale)

        # (b) recovery from a perturbed interior
        u0 = u_exact.copy()
        pert = np.zeros(2 * mesh.n_node)
        pert[free] = perturb_frac * h_min * rng.standard_normal(len(free))
        u0 = (u0.ravel() + pert).reshape(-1, 2)
        vals = u_exact.ravel()
        u, hist, n_it = newton(mesh, con, lambda fr, v=vals, c=con: fr * v[c],
                               u0, MU, LAM, STATE, n_steps=1, verbose=verbose)
        out[f"{name}: recovery |u-u*|/|u*|"] = float(
            np.abs(u - u_exact).max() / max(np.abs(u_exact).max(), 1e-300))
        out[f"{name}: recovery r0 -> r"] = f"{hist[0]:.2e} -> {hist[-1]:.2e}"
        out[f"{name}: recovery iters"] = n_it
    return out


def run_uniaxial(h: float = 1.0, u_delta: float = 1.0,
                 verbose: bool = False) -> dict:
    """Displacement-controlled uniaxial on a near-prismatic specimen."""
    fi = get_fillet_geometry(PRISMATIC)
    l1 = 1.0 + u_delta / fi["L_half"]
    beta = float(uniaxial_lateral_stretch(l1, MU, LAM))
    P11_ex = float(uniaxial_P11(l1, MU, LAM))

    mesh = build_mesh(PRISMATIC, h=h, refine=2.0)
    sol = solve(mesh, u_delta, MU, LAM, STATE, n_steps=1, verbose=verbose)

    u_ex = np.stack([(l1 - 1.0) * mesh.nodes[:, 0],
                     (beta - 1.0) * mesh.nodes[:, 1]], axis=-1)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))

    p11 = sol.P[:, 0, 0]
    return {
        "mesh": summarize(mesh),
        "u_rel_L2": rel(sol.u[:, 0], u_ex[:, 0]),
        "v_rel_L2": rel(sol.u[:, 1], u_ex[:, 1]),
        "P11_rel_max": float(np.abs(p11 - P11_ex).max() / P11_ex),
        "P11_std_over_mean": float(p11.std() / abs(p11.mean())),
        "P22_over_E": float(np.abs(sol.P[:, 1, 1]).max() / E),
        "P11_mean": float(p11.mean()),
        "P11_exact": P11_ex,
        "newton_iters": sol.n_newton,
    }


def run_convergence(params: dict = None, hs=(2.0, 1.4, 1.0, 0.7, 0.5),
                    u_delta: float = 1.0, verbose: bool = False) -> dict:
    """Mesh refinement on the dog-bone: the reference's own error bar.

    Tracks the section resultant (a global quantity, converges fast) and the
    peak von Mises stress at the fillet (a local extremum on piecewise
    constant stress, converges slowly and from below).
    """
    params = params or GEOMETRY_DEFAULT
    fi = get_fillet_geometry(params)
    rows = []
    for h in hs:
        mesh = build_mesh(params, h=h)
        sol = solve(mesh, u_delta, MU, LAM, STATE, n_steps=1, verbose=verbose)
        xs = np.array([0.15, 0.35, 0.55, 0.75, 0.92]) * fi["L_half"]
        N = np.array([sol.axial_resultant(x) for x in xs])
        vm = sol.von_mises()
        rows.append({
            "h": h, "n_elem": mesh.n_elem, "n_node": mesh.n_node,
            "quality_p01": mesh.quality()["p01"],
            "N_mean": float(N.mean()),
            "N_cv": float(N.std() / abs(N.mean())),
            "vm_peak": float(vm.max()),
            "vm_p99": float(np.percentile(vm, 99)),
            "newton_iters": sol.n_newton,
            "wall_s": sol.wall_time,
        })
    return {"rows": rows, "params": params}


def richardson(values, sizes, order: float = 1.0):
    """Extrapolate to h -> 0 from the two finest levels, assuming O(h^order)."""
    (h1, v1), (h2, v2) = (sizes[-2], values[-2]), (sizes[-1], values[-1])
    r = h1 / h2
    v_inf = v2 + (v2 - v1) / (r ** order - 1.0)
    return float(v_inf), float(abs(v2 - v_inf) / max(abs(v_inf), 1e-300))


def main():
    print("Verification of the finite-element reference")
    print("=" * 72)

    print("\n1. Patch test — affine displacement on the whole boundary")
    for k, v in run_patch_test().items():
        print(f"   {k:<38} {v:.3e}" if isinstance(v, float)
              else f"   {k:<38} {v}")

    print("\n2. Uniaxial extension on a near-prismatic specimen")
    r = run_uniaxial()
    print(f"   {r['mesh']}")
    for k in ("u_rel_L2", "v_rel_L2", "P11_rel_max", "P11_std_over_mean",
              "P22_over_E"):
        print(f"   {k:<36} {r[k]:.3e}")
    print(f"   {'P11 mean vs exact [MPa]':<36} "
          f"{r['P11_mean']:.6f}  vs  {r['P11_exact']:.6f}")

    print("\n3. Mesh convergence on the dog-bone (GEOMETRY_DEFAULT)")
    conv = run_convergence()
    print(f"   {'h':>6}{'elems':>9}{'q_p01':>8}{'N_mean':>11}{'N_cv':>10}"
          f"{'vm_peak':>10}{'vm_p99':>10}{'iters':>7}{'s':>7}")
    print("   " + "-" * 78)
    for r in conv["rows"]:
        print(f"   {r['h']:>6.2f}{r['n_elem']:>9d}{r['quality_p01']:>8.3f}"
              f"{r['N_mean']:>11.4f}{r['N_cv']:>10.2e}{r['vm_peak']:>10.4f}"
              f"{r['vm_p99']:>10.4f}{r['newton_iters']:>7d}"
              f"{r['wall_s']:>7.1f}")

    hs = [r["h"] for r in conv["rows"]]
    for key, order in (("N_mean", 2.0), ("vm_peak", 1.0), ("vm_p99", 1.0)):
        vals = [r[key] for r in conv["rows"]]
        v_inf, err = richardson(vals, hs, order)
        print(f"   {key:>8}  h->0 estimate {v_inf:>11.4f}   "
              f"finest-mesh error {100 * err:>6.2f}%   (assumed O(h^{order:g}))")


if __name__ == "__main__":
    main()
