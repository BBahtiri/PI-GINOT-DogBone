#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Method of manufactured solutions for the finite-strain equilibrium operator.

What this verifies that nothing else does
-----------------------------------------
The patch test in ``verification/operators.py`` checks the constitutive law
pointwise, but an affine field has a spatially constant stress whose
divergence is trivially zero.  Here the manufactured field is genuinely
inhomogeneous, so ``Div P`` is a non-trivial known function of position and
the *second-order* automatic differentiation chain -- network output to
displacement gradients to F to the plane-stress closure to P to Div P -- is
compared against an independently derived answer.

Independence is the point.  ``P*`` and ``b* = -Div P*`` are built with sympy
from the closed-form Lambert-W solution of the plane-stress closure, not from
this repository's Newton solve, so a defect in the torch implementation
cannot cancel itself out.  For plane strain the symbolic expression is
elementary.

    F33 = sqrt( (lam/2mu) * W0( (2mu/lam) * exp(2c/lam) ) ),
    c   = mu - lam*ln(J2D)

which satisfies mu*F33^2 + lam*ln(J2D*F33) - mu = 0 identically -- verified
symbolically by ``check_closure_identity`` before anything else runs.

Manufactured field (small enough to keep J well away from zero over the
whole dog-bone, large enough that every term in P is exercised):

    u(X, Y) = a * sin(pi X / Lx) * (Y / Ly)
    v(X, Y) = b * (X / Lx) * (Y / Ly)**2

Run with:  python -m verification.mms
"""

from __future__ import annotations

import numpy as np
import sympy as sp
import torch

from config import GEOMETRY_DEFAULT, MATERIAL_CONFIG
from geometry.parametric_dogbone import (_rejection_sample_interior,
                                         generate_dogbone)
from physics.equilibrium import equilibrium_residual, traction
from physics.neo_hookean import first_piola_kirchhoff_stress
from verification.analytic_fields import AnalyticModel, CallableField

MU = MATERIAL_CONFIG["mu"]
LAM = MATERIAL_CONFIG["lam"]
E = MATERIAL_CONFIG["E"]
DTYPE = torch.float64

DEFAULT_AMPLITUDE = (0.35, 0.25)     # (a, b) in mm


# ------------------------------------------------------------- symbolic ---

def _symbolic_fields(state: str, Lx: float, Ly: float,
                     amplitude=DEFAULT_AMPLITUDE):
    """Build P*, Div P* and the manufactured displacement field with sympy."""
    X, Y = sp.symbols("X Y", real=True)
    a, b = amplitude

    u = a * sp.sin(sp.pi * X / Lx) * (Y / Ly)
    v = b * (X / Lx) * (Y / Ly) ** 2

    F11 = 1 + sp.diff(u, X)
    F12 = sp.diff(u, Y)
    F21 = sp.diff(v, X)
    F22 = 1 + sp.diff(v, Y)
    J2D = F11 * F22 - F12 * F21

    mu, lam = sp.Float(MU, 30), sp.Float(LAM, 30)
    if state == "plane stress":
        c = mu - lam * sp.log(J2D)
        F33 = sp.sqrt((lam / (2 * mu))
                      * sp.LambertW((2 * mu / lam) * sp.exp(2 * c / lam)))
        J = J2D * F33
    else:
        F33 = sp.Integer(1)
        J = J2D

    coeff = lam * sp.log(J) - mu
    P11 = mu * F11 + coeff * F22 / J2D
    P12 = mu * F12 - coeff * F21 / J2D
    P21 = mu * F21 - coeff * F12 / J2D
    P22 = mu * F22 + coeff * F11 / J2D

    div_x = sp.diff(P11, X) + sp.diff(P12, Y)
    div_y = sp.diff(P21, X) + sp.diff(P22, Y)

    return dict(X=X, Y=Y, u=u, v=v, F33=F33, J2D=J2D,
                P=(P11, P12, P21, P22), div=(div_x, div_y))


def _modules():
    """sympy -> numpy, with LambertW routed to scipy's real principal branch."""
    from scipy.special import lambertw
    return [{"LambertW": lambda z: np.real(lambertw(np.asarray(z, float)))},
            "numpy"]


def _lambdify(expr, *symbols):
    """Numeric callable of the given symbols."""
    return sp.lambdify(tuple(symbols), expr, modules=_modules())


def check_closure_identity(n: int = 200) -> float:
    """The Lambert-W form really does solve the closure residual.

    Guards the reference itself: if this is not ~0 the whole MMS comparison
    is meaningless.
    """
    z = sp.symbols("z", positive=True)
    mu, lam = sp.Float(MU, 30), sp.Float(LAM, 30)
    c = mu - lam * sp.log(z)
    F33 = sp.sqrt((lam / (2 * mu))
                  * sp.LambertW((2 * mu / lam) * sp.exp(2 * c / lam)))
    resid = mu * F33 ** 2 + lam * sp.log(z * F33) - mu
    f = _lambdify(resid, z)
    J = np.linspace(0.4, 2.5, n)
    return float(np.abs(np.asarray(f(J), dtype=float)).max() / MU)


# ------------------------------------------------------------- numeric ----

def _torch_field(Lx: float, Ly: float, amplitude=DEFAULT_AMPLITUDE):
    a, b = amplitude

    def fn(X, Y):
        u = a * torch.sin(np.pi * X / Lx) * (Y / Ly)
        v = b * (X / Lx) * (Y / Ly) ** 2
        return u, v

    return CallableField(fn, name=f"mms(a={a}, b={b})")


def run_interior(state: str, params: dict = None, n: int = 4000,
                 seed: int = 23, amplitude=DEFAULT_AMPLITUDE) -> dict:
    """Compare the code's Div P against the symbolic Div P*.

    Points are drawn from inside the real dog-bone, so the fillet region --
    where the manufactured field varies fastest -- is included.
    """
    params = params or GEOMETRY_DEFAULT
    mesh = generate_dogbone(params, rng=np.random.default_rng(0))
    fi = mesh.fillet_info
    Lx, Ly = fi["L_half"], fi["H_grip"]

    pts = _rejection_sample_interior(fi, [], n,
                                     np.random.default_rng(seed)
                                     ).astype(np.float64)

    sym = _symbolic_fields(state, Lx, Ly, amplitude)
    Xs, Ys = sym["X"], sym["Y"]
    div_x = _lambdify(sym["div"][0], Xs, Ys)(pts[:, 0], pts[:, 1])
    div_y = _lambdify(sym["div"][1], Xs, Ys)(pts[:, 0], pts[:, 1])
    J2D = _lambdify(sym["J2D"], Xs, Ys)(pts[:, 0], pts[:, 1])

    model = AnalyticModel(_torch_field(Lx, Ly, amplitude), dtype=DTYPE)
    q = model.query(pts)
    u_d, x_m, y_m = model.eval_mode_args(1.0, Lx, Ly)
    f_x, f_y, _ = equilibrium_residual(model, q, None, u_d, x_m, MU, LAM,
                                       y_m, state, create_graph=False)
    fx = f_x.detach().numpy().ravel()
    fy = f_y.detach().numpy().ravel()

    ref = np.sqrt(div_x ** 2 + div_y ** 2)
    err = np.sqrt((fx - div_x) ** 2 + (fy - div_y) ** 2)
    denom = np.linalg.norm(ref)
    return {
        "state": state,
        "n_points": n,
        "min_J2D": float(np.min(J2D)),
        "max_J2D": float(np.max(J2D)),
        "div_rms_reference": float(np.sqrt(np.mean(ref ** 2))),
        "rel_L2": float(np.linalg.norm(err) / denom),
        "rel_Linf": float(err.max() / np.abs(ref).max()),
    }


def run_boundary(state: str, params: dict = None, n_per_segment: int = 400,
                 amplitude=DEFAULT_AMPLITUDE) -> dict:
    """Compare the code's P.N against the symbolic P*.N, segment by segment.

    Uses the mesh's own normals on both sides, so this checks the traction
    operator and the stress, not the normals themselves -- those are verified
    against their analytic form in verification/geometry_checks.py.
    """
    params = params or GEOMETRY_DEFAULT
    mesh = generate_dogbone(params, n_pts_per_segment=n_per_segment,
                            rng=np.random.default_rng(0))
    fi = mesh.fillet_info
    Lx, Ly = fi["L_half"], fi["H_grip"]

    sym = _symbolic_fields(state, Lx, Ly, amplitude)
    Xs, Ys = sym["X"], sym["Y"]
    Pf = [_lambdify(c, Xs, Ys) for c in sym["P"]]

    model = AnalyticModel(_torch_field(Lx, Ly, amplitude), dtype=DTYPE)
    out = {}
    for seg in mesh.boundary_segments:
        pts = seg.points.astype(np.float64)
        nrm_np = seg.normals.astype(np.float64)
        q = model.query(pts)
        nrm = torch.tensor(nrm_np, dtype=DTYPE).unsqueeze(0)
        _, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(q)
        P = first_piola_kirchhoff_stress(du_dx, du_dy, dv_dx, dv_dy,
                                         MU, LAM, state)[:4]
        tx, ty = traction(*P, nrm)
        tx = tx.detach().numpy().ravel()
        ty = ty.detach().numpy().ravel()

        Pv = [f(pts[:, 0], pts[:, 1]) for f in Pf]
        rx = Pv[0] * nrm_np[:, 0] + Pv[1] * nrm_np[:, 1]
        ry = Pv[2] * nrm_np[:, 0] + Pv[3] * nrm_np[:, 1]

        err = np.sqrt((tx - rx) ** 2 + (ty - ry) ** 2)
        scale = max(np.sqrt(rx ** 2 + ry ** 2).max(), 1e-300)
        out[seg.name] = float(err.max() / scale)
    return out


# Amplitude sweep: the default field keeps J2D within ~2% of 1, which is the
# training regime but a weak test.  The larger amplitudes drive J2D down to
# ~0.22 -- well past anything the barrier would allow -- so the operator is
# checked over the whole range where it could plausibly be asked to work.
SEVERITY_SWEEP = ((0.35, 0.25), (1.5, 1.0), (4.0, 3.0), (8.0, 6.0))


def run_all(amplitude=DEFAULT_AMPLITUDE, n: int = 4000) -> dict:
    return {
        "closure_identity": check_closure_identity(),
        "interior": {s: run_interior(s, n=n, amplitude=amplitude)
                     for s in ("plane strain", "plane stress")},
        "boundary": {s: run_boundary(s, amplitude=amplitude)
                     for s in ("plane strain", "plane stress")},
        "severity": [run_interior("plane stress", n=1500, amplitude=a)
                     | {"amplitude": a} for a in SEVERITY_SWEEP],
    }


def main():
    rep = run_all()
    print("Level 1 — method of manufactured solutions")
    print("=" * 66)
    print(f"\nSymbolic closure identity |r|/mu over J in [0.4, 2.5]: "
          f"{rep['closure_identity']:.3e}")
    print("  (the Lambert-W reference satisfies the closure it stands in for)")

    print("\nInterior: Div P from equilibrium_residual vs symbolic Div P*")
    print(f"{'state':>14}{'J2D range':>22}{'|Div P*| rms':>15}"
          f"{'rel L2':>12}{'rel Linf':>12}")
    print("-" * 75)
    for st, r in rep["interior"].items():
        rng = f"[{r['min_J2D']:.4f}, {r['max_J2D']:.4f}]"
        print(f"{st:>14}{rng:>22}{r['div_rms_reference']:>15.4e}"
              f"{r['rel_L2']:>12.3e}{r['rel_Linf']:>12.3e}")

    print("\nSeverity sweep (plane stress): the operator over the whole "
          "deformation range")
    print(f"{'(a, b) mm':>14}{'J2D range':>22}{'|Div P*| rms':>15}"
          f"{'rel L2':>12}{'rel Linf':>12}")
    print("-" * 75)
    for r in rep["severity"]:
        amp = f"({r['amplitude'][0]}, {r['amplitude'][1]})"
        rng = f"[{r['min_J2D']:.4f}, {r['max_J2D']:.4f}]"
        print(f"{amp:>14}{rng:>22}{r['div_rms_reference']:>15.4e}"
              f"{r['rel_L2']:>12.3e}{r['rel_Linf']:>12.3e}")

    print("\nBoundary: P.N vs symbolic P*.N, relative L-inf per segment")
    segs = list(rep["boundary"]["plane strain"])
    print(f"{'state':>14}" + "".join(f"{s:>17}" for s in segs))
    print("-" * (14 + 17 * len(segs)))
    for st, r in rep["boundary"].items():
        print(f"{st:>14}" + "".join(f"{r[s]:>17.3e}" for s in segs))


if __name__ == "__main__":
    main()
