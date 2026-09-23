#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level-0 verification of the distance-function Dirichlet layer (Phase 8.1).

Float64, no training, five quarter/half models -- the dog-bone and the four
families of the Phase 8 plan:

  dogbone     quarter dog-bone (default geometry)
  open_hole   quarter plate with a central hole          u: x=0, x=L   v: y=0
  dent        quarter plate, double-edge U-notch         u: x=0, x=L   v: y=0
  inclusion   quarter plate, bonded rigid inclusion      u: x=0, arc, x=L
                                                         v: y=0, arc
  sent        half plate, single-edge U-notch, clamped   u: x=0, x=L   v: x=L
  grip_region dog-bone quarter with a clamped grip *segment* on the top face
              (trimmed ADF; Phase 7 B2)

Checks, each an identity that must hold to rounding (thresholds fixed here,
before the first run):

  Z  phi_c = 0 on every Dirichlet piece of component c         <= 1e-12
  N  d phi_c / dn = 1 on every piece, away from its ends        |.-1| <= 1e-6
     (evaluated 1e-9 inside the domain; ends = 5% of the piece)
  P  phi_c > 0 at 20 000 random interior points (outside a 1e-6 band)
  G  the transfinite lift equals the prescribed data on each piece <= 1e-12
  E  the full ansatz, with a random smooth network, satisfies the data <= 1e-12
  R  dog-bone: the layer reproduces PhysicsDecoder._apply_hard_bc   <= 1e-13
     (relative, float64, random points and random raw outputs)

and one comparison, reported rather than thresholded:

  C  inclusion: d phi_u / dn along the arc as it approaches the corner (0, r),
     R-equivalence against the plain product of the same distances (scaled to
     1 at the arc's midpoint).  The literature (Berrone et al. 2023) predicts
     the product's normal derivative decays to 0 at the corner while
     R-equivalence holds 1; at a Dirichlet boundary that derivative *is* the
     network's handle on the strain, so this is where the choice shows up in
     the stress.

    python -m verification.adf_verify
"""

from __future__ import annotations

import math

import numpy as np
import torch

from config import GEOMETRY_DEFAULT, get_fillet_geometry
from geometry import adf
from models.physics_decoder import PhysicsDecoder

torch.set_default_dtype(torch.float64)
UD = 1.0
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# Domains: membership predicates and Dirichlet pieces with their normals
# --------------------------------------------------------------------------
def _lines(p0, p1, n=400):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) * np.asarray(p0, float) + t * np.asarray(p1, float)


def dogbone():
    fi = get_fillet_geometry(GEOMETRY_DEFAULT)
    L, Hg, Hgr, xg, (xc, yc), R = (fi["L_half"], fi["H_gauge"], fi["H_grip"],
                                   fi["x_g"], fi["arc_center"], fi["R_fillet"])
    def inside(p):
        x, y = p[:, 0], p[:, 1]
        h = np.where(x <= xg, Hg, yc - np.sqrt(np.clip(R * R - (x - xc) ** 2, 0, None)))
        return (x >= 0) & (x <= L) & (y >= 0) & (y <= h)
    spec = adf.quarter_tension(L, Hgr, UD)
    pieces = {0: [(_lines((0, 0), (0, Hg)), (1, 0), 0.0),
                  (_lines((L, 0), (L, Hgr)), (-1, 0), UD)],
              1: [(_lines((0, 0), (L, 0)), (0, 1), 0.0)]}
    return dict(spec=spec, inside=inside, pieces=pieces, box=(L, Hgr), L=L, H=Hgr)


def open_hole(L=40.0, H=10.0, r=4.0):
    def inside(p):
        x, y = p[:, 0], p[:, 1]
        return (x >= 0) & (x <= L) & (y >= 0) & (y <= H) & (x * x + y * y >= r * r)
    pieces = {0: [(_lines((0, r), (0, H)), (1, 0), 0.0),
                  (_lines((L, 0), (L, H)), (-1, 0), UD)],
              1: [(_lines((r, 0), (L, 0)), (0, 1), 0.0)]}
    return dict(spec=adf.quarter_tension(L, H, UD), inside=inside,
                pieces=pieces, box=(L, H))


def _u_notch(p, x0, top, depth, rho):
    """Points inside a U-notch cut down from y = top at x = x0."""
    x, y = p[:, 0], p[:, 1]
    yc = top - depth + rho
    slot = (np.abs(x - x0) < rho) & (y > yc)
    cap = (x - x0) ** 2 + (y - yc) ** 2 < rho * rho
    return slot | cap


def dent(L=40.0, H=10.0, depth=3.0, rho=1.5):
    def inside(p):
        x, y = p[:, 0], p[:, 1]
        box = (x >= 0) & (x <= L) & (y >= 0) & (y <= H)
        return box & ~_u_notch(p, 0.0, H, depth, rho)
    pieces = {0: [(_lines((0, 0), (0, H - depth)), (1, 0), 0.0),
                  (_lines((L, 0), (L, H)), (-1, 0), UD)],
              1: [(_lines((0, 0), (L, 0)), (0, 1), 0.0)]}
    return dict(spec=adf.quarter_tension(L, H, UD), inside=inside,
                pieces=pieces, box=(L, H))


def inclusion(L=40.0, H=10.0, r=4.0):
    base = open_hole(L, H, r)
    th = np.linspace(0, 0.5 * np.pi, 400)
    arc = np.stack([r * np.cos(th), r * np.sin(th)], -1)
    arc_n = arc / r                                   # outward from the inclusion
    pieces = {0: [(_lines((0, r), (0, H)), (1, 0), 0.0),
                  (arc, arc_n, 0.0),
                  (_lines((L, 0), (L, H)), (-1, 0), UD)],
              1: [(_lines((r, 0), (L, 0)), (0, 1), 0.0),
                  (arc, arc_n, 0.0)]}
    return dict(spec=adf.quarter_tension(L, H, UD, arc=((0.0, 0.0), r)),
                inside=base["inside"], pieces=pieces, box=(L, H), r=r)


def sent(L=40.0, H=10.0, depth=3.0, rho=1.5):
    def inside(p):
        x, y = p[:, 0], p[:, 1]
        box = (x >= 0) & (x <= L) & (y >= -H) & (y <= H)
        return box & ~_u_notch(p, 0.0, H, depth, rho)
    pieces = {0: [(_lines((0, -H), (0, H - depth)), (1, 0), 0.0),
                  (_lines((L, -H), (L, H)), (-1, 0), UD)],
              1: [(_lines((L, -H), (L, H)), (-1, 0), 0.0)]}
    return dict(spec=adf.half_clamped_tension(L, H, UD), inside=inside,
                pieces=pieces, box=(L, H), ylo=-H)


def grip_region(L=40.0, H=10.0, Hg=5.0, x_s=30.0):
    """Stepped quarter specimen: gauge up to x = 20, grip section beyond,
    clamped on the top face over x_s <= x <= L."""
    def inside(p):
        x, y = p[:, 0], p[:, 1]
        h = np.where(x < 20.0, Hg, H)
        return (x >= 0) & (x <= L) & (y >= 0) & (y <= h)
    pieces = {0: [(_lines((0, 0), (0, Hg)), (1, 0), 0.0),
                  (_lines((x_s, H), (L, H)), (0, -1), UD)],
              1: [(_lines((0, 0), (L, 0)), (0, 1), 0.0),
                  (_lines((x_s, H), (L, H)), (0, -1), 0.0)]}
    return dict(spec=adf.clamped_grip_region(L, H, x_s, UD), inside=inside,
                pieces=pieces, box=(L, H))


CASES = {"dogbone": dogbone, "open_hole": open_hole, "dent": dent,
         "inclusion": inclusion, "sent": sent, "grip_region": grip_region}


# --------------------------------------------------------------------------
class RandomField(torch.nn.Module):
    """A random smooth 'network output' N(X) -- a fixed random tanh MLP."""

    def __init__(self, scale, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.s = torch.tensor(scale)
        self.W1 = torch.randn(2, 64, generator=g)
        self.b1 = torch.randn(64, generator=g)
        self.W2 = torch.randn(64, 2, generator=g) / 8.0

    def forward(self, X):
        return torch.tanh((X / self.s) @ self.W1 + self.b1) @ self.W2


def check(name, case):
    spec, inside = case["spec"], case["inside"]
    L, H = case["box"]
    ylo = case.get("ylo", 0.0)
    net = RandomField(scale=[L, H])
    res = {"Z": 0.0, "N": 0.0, "G": 0.0, "E": 0.0, "P": np.inf}
    for c, plist in case["pieces"].items():
        for pts, n, val in plist:
            n = np.broadcast_to(np.asarray(n, float), pts.shape)
            X = torch.tensor(pts)
            res["Z"] = max(res["Z"], float(spec.phi(X, c).abs().max()))
            res["G"] = max(res["G"], float((spec.lift(X, c) - val).abs().max()))
            w = spec.apply(X, net(X))[..., c]
            res["E"] = max(res["E"], float((w - val).abs().max()))
            # normal derivative just inside, away from the piece's ends
            k = len(pts)
            core = slice(int(0.05 * k), int(0.95 * k))
            Xi = torch.tensor(pts[core] + 1e-9 * n[core], requires_grad=True)
            (gphi,) = torch.autograd.grad(spec.phi(Xi, c).sum(), Xi)
            dn = (gphi * torch.tensor(n[core])).sum(-1)
            res["N"] = max(res["N"], float((dn - 1.0).abs().max()))
    # positivity in the interior, away from the Dirichlet pieces
    P = np.stack([RNG.uniform(0, L, 200000), RNG.uniform(ylo, H, 200000)], -1)
    P = P[inside(P)][:20000]
    X = torch.tensor(P)
    for c in case["pieces"]:
        phi = spec.phi(X, c)
        dmin = np.full(len(P), np.inf)
        for pts, _, _ in case["pieces"][c]:
            d = np.min(np.linalg.norm(P[:, None, :] - pts[None, ::4, :], axis=-1), 1)
            dmin = np.minimum(dmin, d)
        far = dmin > 1e-6
        res["P"] = min(res["P"], float(phi[torch.tensor(far)].min()))
    return res


def regression():
    """The dog-bone case against the shipped decoder's hard-BC layer."""
    fi = get_fillet_geometry(GEOMETRY_DEFAULT)
    L, H = fi["L_half"], fi["H_grip"]
    g = torch.Generator().manual_seed(1)
    X = torch.rand(1, 5000, 2, generator=g) * torch.tensor([L, H])
    raw = torch.randn(1, 5000, 2, generator=g) * 3.0
    ref = PhysicsDecoder._apply_hard_bc(None, raw, X, torch.tensor([UD]),
                                        torch.tensor([L]), torch.tensor([H]))
    new = adf.quarter_tension(L, H, UD).apply(X, raw)
    return float(((new - ref).abs().max() / ref.abs().max()))


def corner_comparison():
    """d phi_u / dn on the inclusion arc approaching the corner (0, r)."""
    case = inclusion()
    spec, r = case["spec"], case["r"]
    L = case["box"][0]
    rows = []
    for combine, tag in ((adf.r_equivalence, "R-equivalence"),
                         (adf.naive_product, "plain product")):
        # scale so that d/dn = 1 at the arc's midpoint (45 deg)
        def dn_at(theta):
            p = np.array([[r * math.cos(theta), r * math.sin(theta)]])
            n = p / r
            Xi = torch.tensor(p + 1e-9 * n, requires_grad=True)
            (g,) = torch.autograd.grad(spec.phi(Xi, 0, combine).sum(), Xi)
            return float((g * torch.tensor(n)).sum())
        ref = dn_at(math.pi / 4)
        # distance from the corner along the arc: s = r (pi/2 - theta)
        vals = [dn_at(math.pi / 2 - s / r) / ref for s in (0.1 * r, 0.01 * r, 0.001 * r)]
        rows.append((tag, vals))
    return rows


def main():
    print("Phase 8.1 -- Level-0 verification of the distance-function layer (float64)")
    print("=" * 84)
    print(f"{'case':<13}{'Z: max|phi|':>13}{'N: max|dn-1|':>14}{'P: min phi':>12}"
          f"{'G: lift err':>13}{'E: ansatz err':>15}  verdict")
    ok_all = True
    for name, make in CASES.items():
        r = check(name, make())
        ok = (r["Z"] <= 1e-12 and r["N"] <= 1e-6 and r["P"] > 0
              and r["G"] <= 1e-12 and r["E"] <= 1e-12)
        ok_all &= ok
        print(f"{name:<13}{r['Z']:>13.1e}{r['N']:>14.1e}{r['P']:>12.2e}"
              f"{r['G']:>13.1e}{r['E']:>15.1e}  {'PASS' if ok else 'FAIL'}")
    reg = regression()
    ok_all &= reg <= 1e-13
    print(f"\nR  dog-bone layer vs PhysicsDecoder._apply_hard_bc: max rel diff "
          f"{reg:.1e}  {'PASS' if reg <= 1e-13 else 'FAIL'}")
    print("\nC  inclusion arc, d phi_u/dn relative to its value at 45 deg, "
          "at arc distance s from the corner (0, r):")
    print(f"   {'':<16}{'s = 0.1 r':>12}{'s = 0.01 r':>12}{'s = 0.001 r':>13}")
    for tag, vals in corner_comparison():
        print(f"   {tag:<16}" + "".join(f"{v:>12.4f}" for v in vals))
    print("\nAll Level-0 checks:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
