#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How much does the objective care about the transverse field?

The transverse displacement has not responded to formulation, weighting,
training length or training-set size, and ``verification/ansatz_study.py``
rules out the hard-BC prefactor as the cause -- the field it demands has a
dynamic range of only ~2, and the obvious "fix" (a local half-height in the
prefactor) makes that range worse, not better.

That leaves a possibility none of the levers would have touched: that v is
simply not visible to the loss.  Pi is an integral of strain energy, and
strain energy in a slender tensile specimen is dominated by axial stretch.
If a 30% error in v costs less energy than the operator's own residual error
in Pi, then no optimiser minimising Pi has any reason to fix v, and every
lever tried so far was working on the wrong quantity.

Method
------
Perturb the finite-element solution by a controlled relative L2 error in one
component at a time, holding the Dirichlet conditions exactly:

    v:  v -> v * (1 + d)                    (v(x,0) = 0 is preserved by
                                             construction; rel error = d)
    u:  u -> u + d * ||u|| * phi / ||phi||,  phi = xi (1 - xi)
                                            (u(0,y) = 0 and u(L,y) = u_delta
                                             are preserved; rel error = d)

and report the resulting dPi/Pi.  Both perturbations are admissible, so Pi
can only rise, and the comparison is of curvature: how steeply the objective
punishes the same relative error in each component.

Run with:
    python -m verification.energy_sensitivity --geos 0,11,19
"""

from __future__ import annotations

import argparse

import numpy as np

from config import (LOADING_CONFIG, MATERIAL_CONFIG, TRAINING_CONFIG,
                    get_fillet_geometry)
from geometry.banks import VAL_BANK_SEED, build_geometry_bank
from physics.weak_form import reference_energy
from verification.fem.mesh import build_mesh, default_h
from verification.fem.solver import solve as fem_solve

MU, LAM = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"]
STATE = MATERIAL_CONFIG["state"]


class _Perturbed:
    """Minimal stand-in for a FemSolution: reference_energy needs mesh and u."""

    def __init__(self, mesh, u):
        self.mesh, self.u = mesh, u


def sensitivities(sol, fi, deltas):
    """dPi/Pi for a relative L2 error `d` in v alone, then in u alone."""
    Pi0 = reference_energy(sol, MU, LAM, STATE)
    u0 = sol.u
    xi = sol.mesh.nodes[:, 0] / fi["L_half"]
    phi = xi * (1.0 - xi)
    phi_n = np.linalg.norm(phi)
    nu_x = np.linalg.norm(u0[:, 0])

    out = {"v": [], "u": []}
    for d in deltas:
        uv = u0.copy()
        uv[:, 1] = u0[:, 1] * (1.0 + d)
        out["v"].append(
            (reference_energy(_Perturbed(sol.mesh, uv), MU, LAM, STATE)
             - Pi0) / Pi0)

        uu = u0.copy()
        uu[:, 0] = u0[:, 0] + d * nu_x * phi / phi_n
        out["u"].append(
            (reference_energy(_Perturbed(sol.mesh, uu), MU, LAM, STATE)
             - Pi0) / Pi0)
    return Pi0, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geos", default="0,2,11,13,19")
    ap.add_argument("--deltas", default="0.05,0.10,0.20,0.35")
    ap.add_argument("--h-factor", type=float, default=2.5)
    args = ap.parse_args()

    deltas = [float(d) for d in args.deltas.split(",")]
    bank = build_geometry_bank(TRAINING_CONFIG["bank_val_size"],
                              TRAINING_CONFIG["bank_geo_ranges"],
                              holes_on=False, seed=VAL_BANK_SEED)
    u_max = LOADING_CONFIG["u_max"]

    print("Energy cost of a controlled field error  (dPi/Pi, both admissible)")
    print("=" * 86)
    head = "".join(f"{f'd={d:.2f}':>12}" for d in deltas)
    print(f"{'geo':>4}{'taper':>7}{'component':>12}{head}")
    print("-" * 86)
    ratios = []
    for gi in [int(g) for g in args.geos.split(",")]:
        gmesh, _ = bank[gi]
        fi = get_fillet_geometry(gmesh.params)
        mesh = build_mesh(gmesh.params, h=default_h(fi) * args.h_factor)
        sol = fem_solve(mesh, u_max, MU, LAM, STATE, n_steps=1, verbose=False)
        _, s = sensitivities(sol, fi, deltas)
        taper = fi["H_gauge"] / fi["H_grip"]
        for comp in ("u", "v"):
            cells = "".join(f"{x:>12.3e}" for x in s[comp])
            print(f"{gi if comp == 'u' else '':>4}"
                  f"{taper if comp == 'u' else float('nan'):>7.3f}"
                  .replace("nan", "   ") + f"{comp:>12}{cells}")
        ratios.append([su / sv for su, sv in zip(s["u"], s["v"])])
        print()
    r = np.array(ratios).mean(axis=0)
    print("-" * 86)
    print(f"{'':>23}{'u/v':>12}" + "".join(f"{x:>12.1f}" for x in r))
    print("\n  How many times more energy the same relative error costs in the")
    print("  axial component than in the transverse one.")


if __name__ == "__main__":
    main()
