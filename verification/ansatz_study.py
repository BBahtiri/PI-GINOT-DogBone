#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why the transverse field is hard: the hard-BC ansatz, not the loss.

The transverse displacement is the one quantity that has not responded to
anything Phases 1 and 2 tried -- formulation (penalty vs energy), weighting
(w_trac_top), training length (400 vs 1750 epochs) or training-set size
(bank 16 -> 64).  Four independent levers, no movement.  That pattern points
at the decoder rather than the objective.

The suspect
-----------
``models/physics_decoder.py`` enforces the symmetry condition v(x, 0) = 0 with

    v(x, y) = (y / H_grip) * phi_v(x, y)

and its docstring says this "uses the full [0, 1] range".  That is true only
at the grip.  In the gauge y reaches H_gauge, so the prefactor is at most
H_gauge/H_grip -- the taper ratio, as low as 0.28 in this bank.

NOTE (Phase 5): this module was motivated by Phase 1's taper correlation of
-0.843, which was quoted here as a *transverse* correlation.  It is not -- it
is the correlation of the **von Mises** error with taper.  Measured directly
across taper 0.31-0.82, the transverse error correlates with taper at +0.108,
essentially not at all.  The conclusion below is unaffected, since it refutes
the local-height ansatz on direct measurement rather than on the correlation,
but the motivation was weaker than stated.

What the physics says phi_v has to be
-------------------------------------
Equilibrium makes the axial resultant N constant along x, so with h(x) the
reference half-height,

    P11 ~ N / (2 h(x)),   eps ~ N / (2 E h(x)),   v ~ -nu * eps * y

and therefore

    current ansatz:      phi_v = v * H_grip / y  ~  -nu N H_grip / (2 E h(x))
    local-height ansatz: phi_v = v * h(x)  / y   ~  -nu N / (2 E)

The first is proportional to 1/h(x): the network is asked to learn a field
whose dynamic range across the specimen is *exactly the taper ratio*, and the
range grows as the specimen gets harder.  The second is **constant** -- the
easiest target a network can be given -- and it is constant because the h(x)
in the ansatz cancels the 1/h(x) that equilibrium puts into the strain.

So the ansatz is not neutral: it injects the geometry's severity into the
target field.  This module measures that directly, on the finite-element
solution, with no training involved -- if the predicted dynamic range is not
there, the argument is wrong and no amount of retraining would have shown it.

Run with:
    python -m verification.ansatz_study --geos 0,2,11,13,19
"""

from __future__ import annotations

import argparse

import numpy as np

from config import (LOADING_CONFIG, MATERIAL_CONFIG, TRAINING_CONFIG,
                    get_fillet_geometry)
from geometry.banks import VAL_BANK_SEED, build_geometry_bank
from physics.uniaxial import section_half_height
from verification.fem.mesh import build_mesh, default_h
from verification.fem.solver import solve as fem_solve

MU, LAM = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"]
STATE = MATERIAL_CONFIG["state"]


def phi_v_fields(nodes, v, fi, y_frac_min: float = 0.25):
    """phi_v demanded by each ansatz, at nodes away from the symmetry plane.

    Both ansaetze divide by y, so nodes near y = 0 are 0/0 up to rounding and
    are excluded -- they carry no information about the dynamic range anyway,
    since both forms agree there by construction.
    """
    x, y = nodes[:, 0], nodes[:, 1]
    h = section_half_height(x, fi)
    keep = y > y_frac_min * h
    x, y, h, v = x[keep], y[keep], h[keep], v[keep]
    return {
        "current  (y/H_grip)": v * fi["H_grip"] / y,
        "local    (y/h(x))": v * h / y,
    }, x, h


def spread(a):
    """Dynamic range of a field, robust to the odd outlying node."""
    lo, hi = np.percentile(np.abs(a), [5, 95])
    return hi / max(lo, 1e-30)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geos", default="0,2,11,13,19")
    ap.add_argument("--h-factor", type=float, default=2.5)
    args = ap.parse_args()

    bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                              TRAINING_CONFIG["bank_geo_ranges"],
                              holes_on=False, seed=VAL_BANK_SEED)
    u_max = LOADING_CONFIG["u_max"]

    print("Dynamic range of the field the decoder must learn")
    print("=" * 78)
    print("  phi_v is what the network output has to be for the ansatz to")
    print("  reproduce the finite-element transverse displacement.")
    print()
    print(f"{'geo':>4}{'taper':>8}{'1/taper':>9}"
          f"{'range: current':>17}{'range: local':>15}{'ratio':>8}")
    print("-" * 78)
    rows = []
    for gi in [int(g) for g in args.geos.split(",")]:
        gmesh, _ = bank[gi]
        fi = get_fillet_geometry(gmesh.params)
        mesh = build_mesh(gmesh.params, h=default_h(fi) * args.h_factor)
        sol = fem_solve(mesh, u_max, MU, LAM, STATE, n_steps=1, verbose=False)
        fields, _, _ = phi_v_fields(mesh.nodes, sol.u[:, 1], fi)
        r_cur = spread(fields["current  (y/H_grip)"])
        r_loc = spread(fields["local    (y/h(x))"])
        taper = fi["H_gauge"] / fi["H_grip"]
        rows.append((taper, r_cur, r_loc))
        print(f"{gi:>4}{taper:>8.3f}{1 / taper:>9.2f}"
              f"{r_cur:>17.2f}{r_loc:>15.2f}{r_cur / r_loc:>8.2f}")

    t = np.array([r[0] for r in rows])
    c = np.array([r[1] for r in rows])
    l = np.array([r[2] for r in rows])
    print("-" * 78)
    print(f"{'mean':>4}{t.mean():>8.3f}{np.mean(1 / t):>9.2f}"
          f"{c.mean():>17.2f}{l.mean():>15.2f}{np.mean(c / l):>8.2f}")
    if len(rows) > 2:
        print(f"\n  corr(dynamic range, 1/taper): current "
              f"{np.corrcoef(c, 1 / t)[0, 1]:+.3f}, "
              f"local {np.corrcoef(l, 1 / t)[0, 1]:+.3f}")
        print("  The prediction is that the current ansatz tracks 1/taper and")
        print("  the local-height one does not.")


if __name__ == "__main__":
    main()
