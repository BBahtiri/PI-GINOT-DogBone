#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level 0 for the stress-potential representation: verify it before training it.

Every property physics/mixed.py claims is by construction gets checked here,
in float64, on a network with *random* weights -- the stress head is
re-initialised at full scale, because at its tiny training initialisation the
potentials are almost pure base term and a test on them would be a test of
the base term only.  Nothing here trains.

  1. Div P_phi = 0 in the interior, both rows.
  2. P_phi . N = 0 along the whole traction-free top, gauge and arc.
  3. Shear-free symmetry planes: P_12 = 0 on Y = 0, P_21 = 0 on X = 0.
  4. The axial force through vertical sections is the same at every X and
     equals -c_1, the value the ansatz was built to carry.
  5. With the stress channels zeroed, P_phi on the gauge is the uniaxial bar
     stress: P_11 = F / H_gauge, every other component zero.

Every error is normalised by sigma_nom = E u_delta / L_half (or by
sigma_nom / L_half for the divergence), so "1e-13" means the same thing
across geometries.

Run with:
    python -m verification.mixed_verify
"""

from __future__ import annotations

import numpy as np
import torch

from training.manifest import set_precision

set_precision("float64")

from config import (DECODER_CONFIG, ENCODER_CONFIG, LOADING_CONFIG,  # noqa
                    MATERIAL_CONFIG, TRAINING_CONFIG, get_fillet_geometry)
from geometry.banks import VAL_BANK_SEED, build_geometry_bank  # noqa: E402
from models.pi_ginot import PI_GINOT  # noqa: E402
from physics.mixed import half_height, mixed_fields  # noqa: E402

MU, LAM = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"]
E, STATE = MATERIAL_CONFIG["E"], MATERIAL_CONFIG["state"]
UD = LOADING_CONFIG["u_max"]


def build(seed=0):
    torch.manual_seed(seed)
    dec = dict(DECODER_CONFIG, mixed=True)
    m = PI_GINOT(ENCODER_CONFIG, dec).eval()
    d = m.decoder
    # Full-scale random heads, so the by-construction claims are tested on
    # potentials that are nothing like the base term.
    for lin in (d.output_proj, d.stress_proj):
        torch.nn.init.normal_(lin.weight, std=0.3)
        torch.nn.init.normal_(lin.bias, std=0.3)
    torch.nn.init.normal_(d.force_head[-1].weight, std=0.3)
    return m


def fields(m, coll, fi, pts_np):
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    bpc = t(coll.boundary_pc).unsqueeze(0)
    xm, ym = t([coll.x_max]), t([coll.y_max])
    with torch.no_grad():
        z = m.encode(bpc, xm, ym, sample_ids=torch.tensor([7]))
    q = t(pts_np).unsqueeze(0).requires_grad_(True)
    f = mixed_fields(m, q, z, t([UD]), xm, ym, fi, MU, LAM, STATE, E)
    return f, q


def top_boundary(fi, n=400):
    """Points on y = H(X) with outward normals (-H', 1) / |.|."""
    X = torch.linspace(1e-6, fi["L_half"] * (1 - 1e-6), n,
                       dtype=torch.float64).requires_grad_(True)
    H = half_height(X, fi)
    Hp = torch.autograd.grad(H.sum(), X)[0]
    N = torch.stack([-Hp, torch.ones_like(Hp)], -1)
    N = N / N.norm(dim=-1, keepdim=True)
    pts = torch.stack([X.detach(), H.detach()], -1).numpy()
    return pts, N.detach().numpy(), (X.detach().numpy() <= fi["x_g"])


def check(m, gi, gmesh, coll):
    fi = get_fillet_geometry(gmesh.params)
    sig = E * UD / fi["L_half"]
    out = {}

    # 1. divergence, at the interior collocation points
    f, q = fields(m, coll, fi, coll.interior_pts[:400, :2])
    P11, P12, P21, P22 = f["P_phi"]
    g = lambda a: torch.autograd.grad(a, q, torch.ones_like(a),
                                      retain_graph=True)[0]
    div1 = g(P11)[..., 0] + g(P12)[..., 1]
    div2 = g(P21)[..., 0] + g(P22)[..., 1]
    scale_div = sig / fi["L_half"]
    out["div"] = float(torch.cat([div1, div2]).abs().max() / scale_div)
    out["P_mag"] = float(torch.stack([P11, P12, P21, P22]).detach()
                         .abs().max() / sig)

    # 2. traction on the whole traction-free top
    pts, N, on_gauge = top_boundary(fi)
    f, _ = fields(m, coll, fi, pts)
    P = [c[0, :, 0].detach().numpy() for c in f["P_phi"]]
    t1 = P[0] * N[:, 0] + P[1] * N[:, 1]
    t2 = P[2] * N[:, 0] + P[3] * N[:, 1]
    tr = np.abs(np.concatenate([t1, t2])) / sig
    out["trac_gauge"] = float(tr[np.concatenate([on_gauge, on_gauge])].max())
    out["trac_arc"] = float(tr[~np.concatenate([on_gauge, on_gauge])].max())

    # 3. shear-free symmetry planes
    xs = np.linspace(0.01, 0.99, 200) * fi["L_half"]
    bot = np.stack([xs, np.zeros_like(xs)], -1)
    f, _ = fields(m, coll, fi, bot)
    out["P12_bottom"] = float(f["P_phi"][1].detach().abs().max() / sig)
    ys = np.linspace(0.01, 0.99, 200) * fi["H_gauge"]
    left = np.stack([np.zeros_like(ys), ys], -1)
    f, _ = fields(m, coll, fi, left)
    out["P21_left"] = float(f["P_phi"][2].detach().abs().max() / sig)

    # 4. section force at several X.  Checked two ways, because the first
    #    version of this test failed at 1e-4 and the failure was the test:
    #    with full-scale random weights the stress is highly oscillatory
    #    (|P| up to 22 sigma_nom here) and 64 Gauss-Legendre points
    #    under-resolve it.  Measured on geometry 22 at X = 0.7 L_half, the
    #    integral converges spectrally -- 4.1e-3 at 32 points, 7.5e-5 at 64,
    #    2.4e-10 at 128, 2.6e-15 at 256 -- while the identity it rests on,
    #    phi_1(X, H) - phi_1(X, 0) = -c_1, holds to exactly 0.  So the
    #    identity is checked directly, and the quadrature at 256 points.
    gx, gw = np.polynomial.legendre.leggauss(256)
    forces = []
    for xf in (0.1, 0.3, 0.5, 0.7, 0.85, 0.95):
        X = xf * fi["L_half"]
        Hx = float(half_height(torch.tensor(X, dtype=torch.float64), fi))
        Y = 0.5 * Hx * (gx + 1.0)
        f, _ = fields(m, coll, fi, np.stack([np.full_like(Y, X), Y], -1))
        forces.append(float((f["P_phi"][0][0, :, 0].detach().numpy()
                             * 0.5 * Hx * gw).sum()))
    forces = np.array(forces)
    F_expected = float(f["axial_force"].detach()[0])
    ends = []
    for xf in (0.1, 0.3, 0.5, 0.7, 0.85, 0.95):
        X = xf * fi["L_half"]
        Hx = float(half_height(torch.tensor(X, dtype=torch.float64), fi))
        f, _ = fields(m, coll, fi, np.array([[X, 0.0], [X, Hx]]))
        b, t_ = [float(v) for v in f["phi1"][0, :, 0].detach()]
        ends.append(abs(t_ - b - F_expected) / abs(F_expected))
    out["F_identity"] = float(max(ends))
    out["F_cv"] = float(forces.std() / abs(forces.mean()))
    out["F_vs_c1"] = float(np.abs(forces - F_expected).max() / abs(F_expected))
    return out


def check_uniaxial(m, gmesh, coll):
    """Stress channels zeroed: the gauge should carry the bar stress."""
    fi = get_fillet_geometry(gmesh.params)
    sig = E * UD / fi["L_half"]
    d = m.decoder
    saved = (d.stress_proj.weight.data.clone(), d.stress_proj.bias.data.clone())
    d.stress_proj.weight.data.zero_(); d.stress_proj.bias.data.zero_()
    try:
        X = np.linspace(0.05, 0.95, 30) * fi["x_g"]
        Y = np.linspace(0.05, 0.95, 30) * fi["H_gauge"]
        XX, YY = np.meshgrid(X, Y)
        f, _ = fields(m, coll, fi, np.stack([XX.ravel(), YY.ravel()], -1))
        P11, P12, P21, P22 = [c[0, :, 0].detach().numpy() for c in f["P_phi"]]
        F = float(f["axial_force"].detach()[0])
        bar = F / fi["H_gauge"]
        return {"P11_vs_bar": float(np.abs(P11 - bar).max() / sig),
                "off_diag": float(max(np.abs(P12).max(), np.abs(P21).max(),
                                      np.abs(P22).max()) / sig)}
    finally:
        d.stress_proj.weight.data.copy_(saved[0])
        d.stress_proj.bias.data.copy_(saved[1])


def main():
    bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                               TRAINING_CONFIG["bank_geo_ranges"],
                               holes_on=False, seed=VAL_BANK_SEED)
    m = build()
    print("Level 0 — stress-potential representation, float64, random heads")
    print("=" * 96)
    hdr = (f"{'geo':>4}{'taper':>7}{'|P|/sig':>9}{'Div P':>11}{'P.N gauge':>11}"
           f"{'P.N arc':>11}{'P12 y=0':>10}{'P21 x=0':>10}{'F cv':>10}"
           f"{'F vs -c1':>10}{'phi jump':>10}")
    print(hdr)
    print("-" * len(hdr))
    worst = {}
    for gi in (2, 15, 4, 6, 3, 11, 9, 22):
        gmesh, coll = bank[gi]
        r = check(m, gi, gmesh, coll)
        for k, v in r.items():
            if k != "P_mag":
                worst[k] = max(worst.get(k, 0.0), v)
        p = gmesh.params
        print(f"{gi:>4}{p['W_gauge'] / p['W_grip']:>7.2f}{r['P_mag']:>9.2f}"
              f"{r['div']:>11.2e}{r['trac_gauge']:>11.2e}{r['trac_arc']:>11.2e}"
              f"{r['P12_bottom']:>10.2e}{r['P21_left']:>10.2e}"
              f"{r['F_cv']:>10.2e}{r['F_vs_c1']:>10.2e}"
              f"{r['F_identity']:>10.2e}")
    print("-" * len(hdr))
    print("worst: " + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))

    u = check_uniaxial(m, *bank[2])
    print(f"\nbase term alone (stress channels zeroed), gauge of geo 2: "
          f"|P11 - F/H_gauge| = {u['P11_vs_bar']:.2e}, "
          f"max off-diagonal = {u['off_diag']:.2e}   (units of sigma_nom)")

    ok = (max(worst.values()) < 1e-10 and u["P11_vs_bar"] < 1e-10
          and u["off_diag"] < 1e-10)
    print("\n" + ("PASS — every by-construction property holds to roundoff"
                  if ok else "FAIL — a property the ansatz claims does not hold"))


if __name__ == "__main__":
    main()
