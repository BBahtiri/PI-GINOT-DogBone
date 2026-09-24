#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level 2: does the *solver* find a solution we already know?

Levels 0 and 1 verify the operators.  They say nothing about whether training
the operator on those residuals actually converges to the right field.  This
driver closes that gap on the one configuration where the answer is available
in closed form.

The case
--------
Take a near-prismatic specimen (W_gauge just under W_grip, so the fillet is a
fraction of a millimetre and the domain is a rectangle to within a rounding
error).  The decoder's hard boundary conditions are then exactly the ones a
uniaxial test needs:

    u(0, Y) = 0        u(L_half, Y) = u_delta        v(X, 0) = 0

with both lateral faces traction-free and v unconstrained at the ends.  The
homogeneous field

    u = (l1 - 1) X,    v = (beta - 1) Y,    l1 = 1 + u_delta / L_half

satisfies every boundary condition and equilibrium exactly, so it *is* the
solution -- not an approximation to it.  Anything the trained model does
differently is solver error, measured against a number rather than against
another discretisation.

Metrics and their tolerances (from the Phase 1 plan)
---------------------------------------------------
    u, v relative L2                 < 0.1%
    P11 relative error               < 0.1%
    std(P11) / mean(P11)             < 1e-6    (the patch-test criterion)
    |P22|/E, |P12|/E                 ~ 0       (laterally free)
    |P33|/E                          < 1e-8    (structural: the closure
                                                enforces it, so this checks
                                                the closure is on, not the
                                                solver)

Note on the patch-test criterion: 1e-6 is a *discretisation* tolerance from
finite elements, where a correct element reproduces constant strain to
round-off.  A network trained by gradient descent will not reach it; the
useful reading is how far it gets, and whether the spread shrinks with
training.  The number is reported either way rather than quietly relaxed.

Usage
-----
    python -m verification.solve_uniaxial --epochs 2000 --device cuda
    python -m verification.solve_uniaxial --epochs 50 --n-interior 500   # smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from config import (DECODER_CONFIG, ENCODER_CONFIG, MATERIAL_CONFIG,
                    NONDIM_SCALES, TRAINING_CONFIG, get_fillet_geometry)
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import generate_dogbone
from models.pi_ginot import PI_GINOT
from physics.losses import PhysicsLoss
from physics.neo_hookean import first_piola_kirchhoff_stress
from physics.uniaxial import uniaxial_lateral_stretch, uniaxial_P11
from training.manifest import set_deterministic, write_manifest

MU = MATERIAL_CONFIG["mu"]
LAM = MATERIAL_CONFIG["lam"]
E = MATERIAL_CONFIG["E"]
STATE = MATERIAL_CONFIG["state"]

# Near-prismatic: dH = 0.005 mm, so the fillet spans 0.28 mm of a 27 mm
# half-length.  validate_geometry needs dH > 0 and R > dH, so an exact
# rectangle is not expressible; this is the closest admissible shape.
PRISMATIC = dict(L_total=54.0, W_grip=20.0, W_gauge=19.99, R_fillet=8.0,
                 holes=[])

TOLERANCES = {
    "u_rel_L2": 1e-3,
    "v_rel_L2": 1e-3,
    "P11_rel_max": 1e-3,
    "P33_over_E": 1e-8,
}


def exact_solution(u_delta: float, L_half: float) -> dict:
    """Closed-form homogeneous uniaxial state for this specimen and load."""
    l1 = 1.0 + u_delta / L_half
    beta = float(uniaxial_lateral_stretch(l1, MU, LAM))
    return {"l1": l1, "beta": beta,
            "P11": float(uniaxial_P11(l1, MU, LAM))}


def _tensors(coll, device, u_delta):
    def T(a):
        return torch.tensor(a, dtype=torch.get_default_dtype(),
                            device=device).unsqueeze(0)
    return dict(
        interior=T(coll.interior_pts).requires_grad_(True),
        tf_pts=T(coll.traction_free_pts).requires_grad_(True),
        tf_norms=T(coll.traction_free_normals),
        tf_tags=torch.tensor(coll.traction_free_tags, dtype=torch.long,
                             device=device),
        pt_pts=T(coll.partial_traction_pts).requires_grad_(True),
        pt_norms=T(coll.partial_traction_normals),
        pt_dirs=torch.tensor(coll.partial_traction_dirs, dtype=torch.long,
                             device=device),
        bpc=T(coll.boundary_pc),
        u_d=torch.tensor([u_delta], dtype=torch.get_default_dtype(), device=device),
        x_m=torch.tensor([coll.x_max], dtype=torch.get_default_dtype(), device=device),
        y_m=torch.tensor([coll.y_max], dtype=torch.get_default_dtype(), device=device),
        sid=torch.tensor([0], dtype=torch.long, device=device),
    )


@torch.enable_grad()
def evaluate(model, coll, fi, u_delta, device) -> dict:
    """Compare the trained field against the closed form."""
    ex = exact_solution(u_delta, fi["L_half"])
    t = _tensors(coll, device, u_delta)
    model.eval()
    z = model.encode(t["bpc"], t["x_m"], t["y_m"], sample_ids=t["sid"])

    pts = coll.interior_pts.astype(np.float64)
    q = torch.tensor(pts, dtype=torch.get_default_dtype(),
                     device=device).unsqueeze(0).requires_grad_(True)
    uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
        q, z, t["u_d"], t["x_m"], t["y_m"])
    P11, P12, P21, P22, _ = first_piola_kirchhoff_stress(
        du_dx, du_dy, dv_dx, dv_dy, MU, LAM, STATE)

    u = uv[0, :, 0].detach().cpu().numpy().astype(np.float64)
    v = uv[0, :, 1].detach().cpu().numpy().astype(np.float64)
    u_ex = (ex["l1"] - 1.0) * pts[:, 0]
    v_ex = (ex["beta"] - 1.0) * pts[:, 1]

    p11 = P11[0, :, 0].detach().cpu().numpy().astype(np.float64)
    p22 = P22[0, :, 0].detach().cpu().numpy().astype(np.float64)
    p12 = P12[0, :, 0].detach().cpu().numpy().astype(np.float64)

    # P33 is zero by construction under the plane-stress closure; recomputing
    # it here checks that the closure is actually active, not that the solver
    # converged.  Done in float64 on the (float32) gradients, so the number
    # reports the closure's residual rather than the storage precision --
    # float32 bottoms out near 1e-7 and would fail the 1e-8 tolerance for
    # reasons that have nothing to do with the physics.
    from physics.neo_hookean import solve_F33_plane_stress
    d64 = [t.detach().double() for t in (du_dx, du_dy, dv_dx, dv_dy)]
    J2D = (d64[0] + 1.0) * (d64[3] + 1.0) - d64[1] * d64[2]
    F33 = solve_F33_plane_stress(J2D, MU, LAM)
    P33 = MU * F33 + (LAM * torch.log(J2D * F33) - MU) / F33

    rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b),
                                                         1e-300))
    return {
        "l1_exact": ex["l1"], "beta_exact": ex["beta"], "P11_exact": ex["P11"],
        "u_rel_L2": rel(u, u_ex),
        "v_rel_L2": rel(v, v_ex),
        "P11_mean": float(p11.mean()),
        "P11_rel_max": float(np.abs(p11 - ex["P11"]).max() / ex["P11"]),
        "P11_std_over_mean": float(p11.std() / abs(p11.mean())),
        "P22_over_E": float(np.abs(p22).max() / E),
        "P12_over_E": float(np.abs(p12).max() / E),
        "P33_over_E": float(P33.abs().max().item() / E),
    }


def train(model, loss_fn, coll, u_delta, device, epochs, lr, clip,
          log_every=100, eval_every=0, fi=None) -> list:
    """Plain single-geometry loop: no banks, no plots, no diagnostics."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t = _tensors(coll, device, u_delta)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        ld = loss_fn(model, t["interior"], t["tf_pts"], t["tf_norms"],
                     t["tf_tags"], t["pt_pts"], t["pt_norms"], t["pt_dirs"],
                     t["bpc"], t["u_d"], t["x_m"], t["y_m"],
                     sample_ids=t["sid"])
        ld["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            msg = (f"  epoch {epoch:5d}  loss={ld['loss'].item():.4e}  "
                   f"L_eq={ld['L_eq_log']:.3e}  "
                   f"L_top={ld['L_trac_top_log']:.3e}  "
                   f"L_part={ld['L_part_log']:.3e}  "
                   f"[{time.time() - t0:.0f}s]")
            if eval_every and fi is not None:
                m = evaluate(model, coll, fi, u_delta, device)
                msg += (f"  u_err={m['u_rel_L2']:.2e}  "
                        f"P11_err={m['P11_rel_max']:.2e}")
                history.append({"epoch": epoch, **m})
            print(msg, flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=TRAINING_CONFIG["learning_rate"])
    ap.add_argument("--n-interior", type=int, default=4000)
    ap.add_argument("--n-boundary", type=int, default=1600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="verification/results/uniaxial")
    ap.add_argument("--track", action="store_true",
                    help="evaluate against the exact solution while training")
    args = ap.parse_args()

    set_deterministic(args.seed)
    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, __import__("config"),
                   extra={"case": "uniaxial", "args": vars(args)})

    fi = get_fillet_geometry(PRISMATIC)
    u_delta = 1.0
    ex = exact_solution(u_delta, fi["L_half"])

    print("Level 2 — homogeneous uniaxial extension")
    print("=" * 70)
    print(f"  specimen  L_half={fi['L_half']:.2f}  H_grip={fi['H_grip']:.4f}  "
          f"H_gauge={fi['H_gauge']:.4f}  fillet span={fi['dx']:.4f} mm")
    print(f"  load      u_delta={u_delta}  ->  l1={ex['l1']:.8f}")
    print(f"  exact     beta={ex['beta']:.8f}   P11={ex['P11']:.6f} MPa")
    print(f"  device    {args.device}   epochs {args.epochs}\n")

    mesh = generate_dogbone(PRISMATIC, rng=np.random.default_rng(args.seed))
    coll = sample_collocation_points(mesh, n_interior=args.n_interior,
                                     n_total_boundary=args.n_boundary,
                                     rng=np.random.default_rng(args.seed + 1))

    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).to(args.device)
    loss_fn = PhysicsLoss(
        mu=MU, lam=LAM,
        w_equilibrium=TRAINING_CONFIG["w_equilibrium"],
        w_trac_top=TRAINING_CONFIG["w_trac_top"],
        w_trac_arc=TRAINING_CONFIG["w_trac_arc"],
        w_traction_partial=TRAINING_CONFIG["w_traction_partial"],
        w_barrier=TRAINING_CONFIG["w_barrier"],
        j_min=TRAINING_CONFIG["barrier_delta"],
        L0=NONDIM_SCALES["L0"], S0=NONDIM_SCALES["S0"],
        stress_state=STATE,
    ).to(args.device)

    before = evaluate(model, coll, fi, u_delta, args.device)
    history = train(model, loss_fn, coll, u_delta, args.device, args.epochs,
                    args.lr, TRAINING_CONFIG["grad_clip_norm"],
                    eval_every=args.track, fi=fi)
    after = evaluate(model, coll, fi, u_delta, args.device)

    print("\n" + "=" * 70)
    print(f"{'metric':>22}{'untrained':>14}{'trained':>14}"
          f"{'tolerance':>13}{'':>7}")
    print("-" * 70)
    for k in ("u_rel_L2", "v_rel_L2", "P11_rel_max", "P11_std_over_mean",
              "P22_over_E", "P12_over_E", "P33_over_E"):
        tol = TOLERANCES.get(k)
        mark = "" if tol is None else ("  PASS" if after[k] < tol
                                       else "  FAIL")
        tol_s = "—" if tol is None else f"{tol:.0e}"
        print(f"{k:>22}{before[k]:>14.3e}{after[k]:>14.3e}"
              f"{tol_s:>13}{mark:>7}")
    print("-" * 70)
    print(f"{'P11 mean [MPa]':>22}{before['P11_mean']:>14.4f}"
          f"{after['P11_mean']:>14.4f}{after['P11_exact']:>13.4f}"
          f"{'  exact':>7}")

    result = {"exact": ex, "before": before, "after": after,
              "history": history, "args": vars(args)}
    path = os.path.join(args.out, "result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nWrote {path}")

    failed = [k for k, tol in TOLERANCES.items() if after[k] >= tol]
    print("\n" + ("FAIL: " + ", ".join(failed) if failed
                  else "All tolerances met."))
    print("Note: P11_std_over_mean has no pass/fail line — the plan's 1e-6 is "
          "a\nfinite-element discretisation tolerance, not reachable by "
          "gradient descent.\nIt is reported so the trend can be tracked.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
