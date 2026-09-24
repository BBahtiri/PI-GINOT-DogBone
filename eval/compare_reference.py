#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 case 1.4: predicted stress against a finite-element reference.

This is the measurement the paper's central claim rests on. Every accuracy
number the project reported before this was either a residual -- which says a
field is self-consistent, not that it is right -- or a comparison against a
closed form that only exists for a prismatic bar. Neither can say how wrong
the stress is at a fillet, which is precisely where the operator is documented
to be weakest.

Method
------
Solve the same boundary-value problem with the independent finite-element
discretisation in ``verification/fem`` (verified against the patch test and
the closed-form uniaxial solution first), then evaluate the operator at the
*element centroids*, where the reference stress is defined exactly rather
than interpolated. Displacements are compared at the mesh nodes.

Regions
-------
Reported separately, because a bulk average hides exactly the failure mode of
interest:

    gauge   x < x_g                       the prismatic part
    fillet  x >= x_g                      the filleted part
    band    within 0.5 R of the arc       the concentration itself

Relative L2 is not the quantity a stress-concentration claim rests on, and the
two can disagree sharply -- R-PINN (arXiv 2506.10243) reports being 2x worse in
L2 and 3.9x better in L-infinity than its comparator on the same problem, purely
from the choice of error indicator.  So every region also carries `max` (the
L-infinity error normalised by the reference peak), `peak_rel_err` (signed, so
under- and over-prediction are distinguishable) and the percentiles, and the
geometry carries a stress concentration factor.  Report the one the claim is
about.

Reading the numbers
-------------------
The reference is piecewise-constant stress on constant-strain triangles, so it
is first-order in stress and its own peak carries a discretisation error.
``--convergence`` reports that error bar; quoting a peak-stress agreement
tighter than the reference's own uncertainty would be meaningless.

Run with:
    python -m eval.compare_reference --checkpoint checkpoints/best.pt
    python -m eval.compare_reference --checkpoint checkpoints/best.pt --convergence
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from config import (DECODER_CONFIG, ENCODER_CONFIG, GEOMETRY_DEFAULT,
                    LOADING_CONFIG, MATERIAL_CONFIG, TRAINING_CONFIG)
from geometry.banks import val_bank_params
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import generate_dogbone
from models.pi_ginot import PI_GINOT
from physics.neo_hookean import full_stress_state, von_mises_stress
from verification.fem.mesh import build_mesh
from verification.fem.solver import solve as fem_solve

MU = MATERIAL_CONFIG["mu"]
LAM = MATERIAL_CONFIG["lam"]
E = MATERIAL_CONFIG["E"]
STATE = MATERIAL_CONFIG["state"]


def _pinn_fields(model, params, pts, u_delta, device, sample_id=700,
                 coll=None):
    """Displacement and stress from the operator at arbitrary points.

    ``coll`` should be the collocation data the trainer itself used for this
    geometry.  The encoder caches its farthest-point-sampling indices by
    sample id, so encoding a *different* boundary point cloud under the same
    id would compare the operator against an input it was never validated on.
    """
    if coll is None:
        mesh = generate_dogbone(params, rng=np.random.default_rng(0))
        coll = sample_collocation_points(mesh, n_interior=1,
                                         rng=np.random.default_rng(1))
    bpc = torch.tensor(coll.boundary_pc, dtype=torch.get_default_dtype(),
                       device=device).unsqueeze(0)
    x_m = torch.tensor([coll.x_max], dtype=torch.get_default_dtype(), device=device)
    y_m = torch.tensor([coll.y_max], dtype=torch.get_default_dtype(), device=device)
    u_d = torch.tensor([u_delta], dtype=torch.get_default_dtype(), device=device)
    sid = torch.tensor([sample_id], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        z = model.encode(bpc, x_m, y_m, sample_ids=sid)

    q = torch.tensor(pts, dtype=torch.get_default_dtype(),
                     device=device).unsqueeze(0).requires_grad_(True)
    uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
        q, z, u_d, x_m, y_m)
    S11, S22, S33, S12, _ = full_stress_state(
        du_dx, du_dy, dv_dx, dv_dy, MU, LAM, STATE)
    vm = von_mises_stress(S11, S22, S33, S12)
    g = lambda t: t[0, :, 0].detach().cpu().numpy().astype(np.float64)
    return {
        "u": uv[0, :, 0].detach().cpu().numpy().astype(np.float64),
        "v": uv[0, :, 1].detach().cpu().numpy().astype(np.float64),
        "s11": g(S11), "s22": g(S22), "s12": g(S12), "vm": g(vm),
    }


def _fem_cauchy(sol):
    """Element Cauchy stress components from the FEM solution."""
    from verification.fem.solver import _F33, deformation_gradients
    F = deformation_gradients(sol.mesh, sol.u)
    J3D = sol.J2D * _F33(sol.J2D)
    s = np.einsum("eij,ekj->eik", sol.P, F) / J3D[:, None, None]
    s12 = 0.5 * (s[:, 0, 1] + s[:, 1, 0])
    return s[:, 0, 0], s[:, 1, 1], s12


def _regions(pts, fi):
    x = pts[:, 0]
    xc, yc = fi["arc_center"]
    d_arc = np.abs(np.hypot(x - xc, pts[:, 1] - yc) - fi["R_fillet"])
    return {
        "all": np.ones(len(pts), bool),
        "gauge": x < fi["x_g"],
        "fillet": x >= fi["x_g"],
        "band": d_arc < 0.5 * fi["R_fillet"],
    }


def _errors(pred, ref, mask, weights=None, scale=None):
    """Weighted relative L2, plus error percentiles and the peak.

    ``scale`` normalises the pointwise errors by a *common* stress magnitude
    (the reference von Mises peak) rather than by each component's own norm.
    Without it, sigma_22 and sigma_12 -- which are near zero over most of the
    specimen by construction -- report relative errors in the hundreds of
    percent that say nothing about how wrong the stress state is.
    """
    p, r = pred[mask], ref[mask]
    w = np.ones_like(r) if weights is None else weights[mask]
    denom = np.sqrt((w * r ** 2).sum())
    rel_l2 = float(np.sqrt((w * (p - r) ** 2).sum()) / max(denom, 1e-300))
    sc = max(np.abs(r).max() if scale is None else scale, 1e-300)
    norm_l2 = float(np.sqrt((w * (p - r) ** 2).sum() / max(w.sum(), 1e-300))
                    / sc)
    rel_pt = np.abs(p - r) / sc
    return {
        "rel_L2": rel_l2,
        "rms_over_scale": norm_l2,
        "p50": float(np.percentile(rel_pt, 50)),
        "p95": float(np.percentile(rel_pt, 95)),
        "p99": float(np.percentile(rel_pt, 99)),
        "max": float(rel_pt.max()),
        "peak_pred": float(p.max()),
        "peak_ref": float(r.max()),
        "peak_rel_err": float((p.max() - r.max()) / max(abs(r.max()), 1e-300)),
        "n": int(mask.sum()),
    }


def compare_one(model, params: dict, u_delta: float, h: float,
                device: str = "cpu", sample_id: int = 700,
                verbose: bool = False, coll=None, sol=None) -> dict:
    """Compare the operator against the FEM reference on one geometry.

    ``sol`` lets a caller reuse a previously computed reference, which matters
    when the same geometry is evaluated repeatedly (e.g. tracking a training
    run) -- the reference does not change, so re-solving it would be pure
    waste and would risk comparing against a slightly different mesh.
    """
    if sol is None:
        mesh = build_mesh(params, h=h)
        sol = fem_solve(mesh, u_delta, MU, LAM, STATE, n_steps=1,
                        verbose=verbose)
    else:
        mesh = sol.mesh
    fi = mesh.fillet_info

    cent = sol.centroids
    s11_f, s22_f, s12_f = _fem_cauchy(sol)
    vm_f = sol.von_mises()

    pin_c = _pinn_fields(model, params, cent, u_delta, device, sample_id, coll)
    pin_n = _pinn_fields(model, params, mesh.nodes, u_delta, device,
                         sample_id, coll)

    reg = _regions(cent, fi)
    w = mesh.areas                       # area-weighted: mesh-density neutral
    out = {"params": params, "h": h, "n_elem": mesh.n_elem,
           "n_node": mesh.n_node,
           "quality_p01": mesh.quality()["p01"],
           "fem_newton_iters": sol.n_newton}

    # displacement, at the nodes
    u_ref = sol.u
    for k, i in (("u", 0), ("v", 1)):
        d = pin_n[k] - u_ref[:, i]
        out[f"disp_{k}_rel_L2"] = float(
            np.linalg.norm(d) / max(np.linalg.norm(u_ref[:, i]), 1e-300))

    # stress, at the element centroids
    vm_scale = float(vm_f.max())
    out["vm_ref_peak"] = vm_scale
    for name, ref, pred in (("vm", vm_f, pin_c["vm"]),
                            ("s11", s11_f, pin_c["s11"]),
                            ("s22", s22_f, pin_c["s22"]),
                            ("s12", s12_f, pin_c["s12"])):
        for rname, mask in reg.items():
            out[f"{name}_{rname}"] = _errors(pred, ref, mask, w,
                                             scale=vm_scale)

    # Stress concentration factor -- the engineering statement of what this
    # model is for, and the number a mechanics reader will look for first.
    # Defined the standard way, peak over nominal, with the nominal taken as
    # the mean von Mises over the prismatic gauge.  Both SCFs come from the
    # same field, so the comparison is of the concentration itself rather than
    # of the overall stress level, and a model that gets the level right and
    # the peak wrong is caught here where a relative L2 would hide it.
    gauge = reg["gauge"]
    for tag, field in (("ref", vm_f), ("pinn", pin_c["vm"])):
        nominal = float(np.average(field[gauge], weights=w[gauge])) \
            if gauge.any() else float("nan")
        out[f"scf_{tag}"] = float(field.max() / max(nominal, 1e-300))
        out[f"vm_nominal_{tag}"] = nominal
    out["scf_rel_err"] = float(
        (out["scf_pinn"] - out["scf_ref"]) / max(abs(out["scf_ref"]), 1e-300))

    # Where the peak sits, in units of the tangency abscissa, so 1.0 is the
    # gauge-fillet transition itself and larger is further into the fillet.
    # An amplitude error and a localisation error are different faults with
    # different fixes -- a peak of the right height in the wrong place is a
    # resolution problem, one in the right place at the wrong height is not
    # -- and reporting only the magnitude would not tell them apart.
    for tag, field in (("ref", vm_f), ("pinn", pin_c["vm"])):
        out[f"x_peak_{tag}"] = float(
            cent[int(np.argmax(field)), 0] / max(fi["x_g"], 1e-300))
    out["x_peak_err"] = out["x_peak_pinn"] - out["x_peak_ref"]

    # global force balance
    xs = np.array([0.15, 0.35, 0.55, 0.75, 0.92]) * fi["L_half"]
    N_fem = np.array([sol.axial_resultant(x) for x in xs])
    N_pinn = np.array([_pinn_resultant(model, params, x, u_delta, device,
                                       sample_id, coll, fi) for x in xs])
    out["N_fem"] = float(N_fem.mean())
    out["N_fem_cv"] = float(N_fem.std() / abs(N_fem.mean()))
    out["N_pinn"] = float(N_pinn.mean())
    out["N_pinn_cv"] = float(N_pinn.std() / abs(N_pinn.mean()))
    out["N_rel_err"] = float((N_pinn.mean() - N_fem.mean()) / N_fem.mean())
    return out


def _pinn_resultant(model, params, x, u_delta, device, sample_id, coll, fi,
                    n_y: int = 400):
    """N(x) = 2 * integral of P11 over the reference section, from the operator."""
    from physics.neo_hookean import first_piola_kirchhoff_stress
    from physics.uniaxial import section_half_height
    h = float(section_half_height(x, fi))
    y = np.linspace(0.0, h, n_y + 1)
    y = 0.5 * (y[:-1] + y[1:])
    pts = np.stack([np.full(n_y, x), y], -1)
    mesh = generate_dogbone(params, rng=np.random.default_rng(0))
    c = coll if coll is not None else sample_collocation_points(
        mesh, n_interior=1, rng=np.random.default_rng(1))
    bpc = torch.tensor(c.boundary_pc, dtype=torch.get_default_dtype(),
                       device=device).unsqueeze(0)
    x_m = torch.tensor([c.x_max], dtype=torch.get_default_dtype(), device=device)
    y_m = torch.tensor([c.y_max], dtype=torch.get_default_dtype(), device=device)
    u_d = torch.tensor([u_delta], dtype=torch.get_default_dtype(), device=device)
    sid = torch.tensor([sample_id], dtype=torch.long, device=device)
    with torch.no_grad():
        z = model.encode(bpc, x_m, y_m, sample_ids=sid)
    q = torch.tensor(pts, dtype=torch.get_default_dtype(),
                     device=device).unsqueeze(0).requires_grad_(True)
    _, a, b, cc, d = model.predict_with_grad_latent(q, z, u_d, x_m, y_m)
    P11 = first_piola_kirchhoff_stress(a, b, cc, d, MU, LAM, STATE)[0]
    return float(2.0 * (h / n_y) * P11[0, :, 0].detach().sum())


def _load_model(path, device):
    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    state, dropped = PI_GINOT.strip_legacy_keys(ck["model_state_dict"])
    if dropped:
        print(f"  dropped {len(dropped)} legacy checkpoint key(s)")
    model.load_state_dict(state)
    return model, ck


def _convergence(args, factors=(2.5, 1.8, 1.3, 1.0, 0.75)):
    """How much of the reference concentration factor is the mesh?

    Every SCF the model is scored against is a maximum of a piecewise-constant
    field over a finite mesh, so it is biased low and the bias shrinks with h.
    That matters here in a way it would not for a relative L2: the quantity
    being predicted varies only ~3% across the geometry bank, so a reference
    that is itself 1% off -- and off by a *geometry-dependent* amount -- eats
    a third of the signal and contaminates any R^2 computed against it.

    Reports K_t against the finest mesh for a few geometries, so the factor
    used for scoring can be chosen rather than inherited.
    """
    from config import get_fillet_geometry
    from geometry.banks import VAL_BANK_SEED, build_geometry_bank
    from verification.fem.mesh import default_h

    bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                               TRAINING_CONFIG["bank_geo_ranges"],
                               holes_on=False, seed=VAL_BANK_SEED)
    geos = [int(g) for g in args.geometries.split(",")]
    print("Reference discretisation error in the stress concentration factor")
    print("=" * 74)
    print(f"  K_t = peak von Mises / area-weighted gauge mean, "
          f"{len(factors)} meshes per geometry")
    print(f"  element size = default_h * factor; smaller factor, finer mesh\n")
    hdr = (f"{'geo':>4}{'taper':>7}{'factor':>8}{'elems':>9}{'peak':>10}"
           f"{'nominal':>10}{'K_t':>9}{'vs finest':>11}")
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for gi in geos:
        gmesh, _ = bank[gi]
        p = gmesh.params
        fi = get_fillet_geometry(p)
        rows = []
        for f in factors:
            mesh = build_mesh(p, h=default_h(fi) * f)
            sol = fem_solve(mesh, LOADING_CONFIG["u_max"], MU, LAM, STATE,
                            n_steps=1, verbose=False)
            vm, w, c = sol.von_mises(), mesh.areas, sol.centroids
            g = c[:, 0] < fi["x_g"]
            nom = float(np.average(vm[g], weights=w[g]))
            rows.append({"factor": f, "n_elem": int(mesh.n_elem),
                         "peak": float(vm.max()), "nominal": nom,
                         "K_t": float(vm.max() / nom)})
        fine = rows[-1]["K_t"]
        for r in rows:
            r["rel_to_finest"] = (r["K_t"] - fine) / fine
            print(f"{gi:>4}{p['W_gauge'] / p['W_grip']:>7.2f}"
                  f"{r['factor']:>8.2f}{r['n_elem']:>9d}{r['peak']:>10.4g}"
                  f"{r['nominal']:>10.4g}{r['K_t']:>9.4f}"
                  f"{100 * r['rel_to_finest']:>+10.2f}%")
        print()
        out[gi] = rows
    worst = {f: max(abs(out[g][i]["rel_to_finest"]) for g in geos)
             for i, f in enumerate(factors)}
    print("worst |error| across the geometries, by factor:")
    for f in factors:
        print(f"  factor {f:>4.2f}: {100 * worst[f]:>5.2f}%")
    path = os.path.join(args.out, "k_convergence.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in out.items()}, fh, indent=2)
    print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--n-geometries", type=int, default=8)
    ap.add_argument("--h-factor", type=float, default=38.0,
                    help="element size = L_half / this")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--convergence", action="store_true",
                    help="report the reference's own K_t discretisation "
                         "error and exit, without loading a model")
    ap.add_argument("--geometries", default="2,6,22",
                    help="--convergence only: which bank geometries to "
                         "refine (default spans the taper range)")
    ap.add_argument("--out", default="verification/results/compare")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.convergence:
        return _convergence(args)
    model, ck = _load_model(args.checkpoint, args.device)
    u_delta = LOADING_CONFIG["u_max"]
    from geometry.banks import VAL_BANK_SEED, build_geometry_bank
    bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                               TRAINING_CONFIG["bank_geo_ranges"],
                               holes_on=False, seed=VAL_BANK_SEED)
    bank = bank[:args.n_geometries]

    print("Phase 1 case 1.4 — operator vs finite-element reference")
    print("=" * 88)
    print(f"  checkpoint {args.checkpoint}  (epoch {ck.get('epoch')})")
    print(f"  {len(bank)} validation geometries; element size resolves both "
          "the length and the gauge\n")

    results = []
    from config import get_fillet_geometry
    from verification.fem.mesh import default_h
    hdr = (f"{'geo':>4}{'elems':>8}{'u L2':>9}{'v L2':>9}"
           f"{'vm all':>9}{'vm gauge':>10}{'vm fill':>9}{'vm band':>9}"
           f"{'peak err':>10}{'K pinn':>9}{'K ref':>8}{'K err':>8}"
           f"{'N err':>9}{'N cv':>8}")
    print(hdr)
    print("-" * len(hdr))
    for i, (gmesh, coll) in enumerate(bank):
        p = gmesh.params
        h = default_h(get_fillet_geometry(p))
        r = compare_one(model, p, u_delta, h, args.device, sample_id=100 + i,
                        coll=coll)
        results.append(r)
        print(f"{i:>4}{r['n_elem']:>8d}{r['disp_u_rel_L2']:>9.2e}"
              f"{r['disp_v_rel_L2']:>9.2e}"
              f"{r['vm_all']['rel_L2']:>9.2e}{r['vm_gauge']['rel_L2']:>10.2e}"
              f"{r['vm_fillet']['rel_L2']:>9.2e}{r['vm_band']['rel_L2']:>9.2e}"
              f"{r['vm_all']['peak_rel_err']:>10.2e}"
              f"{r['scf_pinn']:>9.3f}{r['scf_ref']:>8.3f}"
              f"{100 * r['scf_rel_err']:>7.1f}%"
              f"{100 * r['N_rel_err']:>8.1f}%{100 * r['N_pinn_cv']:>7.1f}%",
              flush=True)

    print("-" * len(hdr))
    agg = lambda f: np.mean([f(r) for r in results])
    print(f"{'mean':>4}{'':>8}{agg(lambda r: r['disp_u_rel_L2']):>9.2e}"
          f"{agg(lambda r: r['disp_v_rel_L2']):>9.2e}"
          f"{agg(lambda r: r['vm_all']['rel_L2']):>9.2e}"
          f"{agg(lambda r: r['vm_gauge']['rel_L2']):>10.2e}"
          f"{agg(lambda r: r['vm_fillet']['rel_L2']):>9.2e}"
          f"{agg(lambda r: r['vm_band']['rel_L2']):>9.2e}"
          f"{agg(lambda r: r['vm_all']['peak_rel_err']):>10.2e}"
          f"{'':>17}{100 * agg(lambda r: abs(r['scf_rel_err'])):>8.1f}%"
          f"{100 * agg(lambda r: abs(r['N_rel_err'])):>8.1f}%"
          f"{100 * agg(lambda r: r['N_pinn_cv']):>7.1f}%")
    print(f"     FEM reference section-force CV (should be ~0): "
          f"{agg(lambda r: r['N_fem_cv']):.2e}")

    print("\nvon Mises error percentiles, relative to the reference peak")
    print(f"{'region':>10}{'p50':>12}{'p95':>12}{'p99':>12}{'max':>12}")
    print("-" * 58)
    for rname in ("all", "gauge", "fillet", "band"):
        print(f"{rname:>10}"
              f"{agg(lambda r: r[f'vm_{rname}']['p50']):>12.3e}"
              f"{agg(lambda r: r[f'vm_{rname}']['p95']):>12.3e}"
              f"{agg(lambda r: r[f'vm_{rname}']['p99']):>12.3e}"
              f"{agg(lambda r: r[f'vm_{rname}']['max']):>12.3e}")

    print("\nComponent RMS error, normalised by the reference von Mises peak")
    print(f"{'component':>10}{'all':>12}{'gauge':>12}{'fillet':>12}{'band':>12}")
    print("-" * 58)
    for c in ("s11", "s22", "s12", "vm"):
        print(f"{c:>10}" + "".join(
            f"{agg(lambda r, c=c, k=k: r[f'{c}_{k}']['rms_over_scale']):>12.3e}"
            for k in ("all", "gauge", "fillet", "band")))

    path = os.path.join(args.out, "compare.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
