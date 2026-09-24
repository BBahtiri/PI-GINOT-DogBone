#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specimen families for the per-family operators (Phase 8, Tier 1).

A family bundles everything that used to be hard-wired to the dog-bone:
its parameters and their training box, the shape (for the FEM mesh and the
energy quadrature), the exact Dirichlet specification for the network
(geometry/adf.py), the Dirichlet sets for the FEM, the coordinate extent the
decoder normalises by, and the definition of its stress concentration factor

    K_t = max Cauchy vm over the peak region / (N / A_gross)

N is the axial force through the specimen -- the grip reaction for the FEM,
the section integral of P11 at x = 0.75 L for the network, equal in
equilibrium -- and A_gross the gross section per unit thickness.  At small
load this is the handbook's gross-section K_t.

Why not a far-field band mean, like the dog-bone's gauge mean: the first
Level-3 run used one ([0.6 L, 0.85 L]) and clamping the grip moved it by
~1% -- the band sits inside the clamped grip's end zone at L/W = 1.5, and
the single notch is clamped by definition.  The force is the same through
every section whatever the grip does.  The band mean is kept as
``kt_band`` for comparison.  The single notch excludes the grip end from the
peak search: its clamped corners are weakly singular (Williams 1952; 64.4
deg threshold for this material).

Parameters are dimensionless ratios plus the width W, so the training box is
a box in the quantities that set K_t.  Every family is loaded, like the
dog-bone, by u_delta = LOADING_CONFIG["u_max"] at its grip.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from geometry import adf
from geometry.shapes import (DoubleNotchPlate, OpenHolePlate,
                             RigidInclusionPlate, SingleNotchPlate)


class Family:
    name: str = ""
    keys: Tuple[str, ...] = ()
    box: Dict[str, Tuple[float, float]] = {}
    oor: Dict[str, Tuple[float, float]] = {}     # the one direction left
    strata: str = ""                             # IR set stratified on this
    seed_base: int = 0

    # --- geometry ---------------------------------------------------------
    def dims(self, p: dict) -> dict:
        raise NotImplementedError

    def shape(self, p: dict):
        raise NotImplementedError

    def extent(self, p: dict) -> Tuple[float, float]:
        d = self.dims(p)
        return d["L"], d["H"]

    # --- boundary conditions ---------------------------------------------
    def spec(self, p: dict, u_delta: float) -> adf.DirichletSpec:
        d = self.dims(p)
        return adf.quarter_tension(d["L"], d["H"], u_delta)

    def fem_dirichlet(self, mesh, u_delta: float):
        """Constrained DOFs and a values-of-load-fraction callable."""
        b = mesh.boundary
        items = ([(2 * i, 0.0) for i in b["left_symmetry"]]
                 + [(2 * i, u_delta) for i in b["right_grip"]]
                 + [(2 * i + 1, 0.0) for i in b["bottom"]])
        return _pack(items)

    # --- K_t regions --------------------------------------------------------
    def nominal_mask(self, cent: np.ndarray, p: dict) -> np.ndarray:
        L = self.dims(p)["L"]
        return (cent[:, 0] >= 0.6 * L) & (cent[:, 0] <= 0.85 * L)

    def peak_mask(self, cent: np.ndarray, p: dict) -> np.ndarray:
        return np.ones(len(cent), bool)

    def gross_height(self, p: dict) -> float:
        return self.extent(p)[1]

    def section_x(self, p: dict) -> float:
        return 0.75 * self.dims(p)["L"]

    def kt(self, vm: np.ndarray, cent: np.ndarray, N: float, p: dict):
        """(K_t, nominal stress, index of the peak element)."""
        pm = self.peak_mask(cent, p)
        nominal = float(N / self.gross_height(p))
        k = int(np.flatnonzero(pm)[np.argmax(vm[pm])])
        return float(vm[k] / nominal), nominal, k

    def kt_band(self, vm, cent, areas, p):
        nm, pm = self.nominal_mask(cent, p), self.peak_mask(cent, p)
        nominal = float(np.average(vm[nm], weights=areas[nm]))
        return float(vm[pm].max() / nominal)

    # --- sets ------------------------------------------------------------
    def _draw(self, rng, box):
        return {k: float(rng.uniform(*box[k])) for k in self.keys}

    def bank(self, n: int = 64) -> List[dict]:
        rng = np.random.default_rng(self.seed_base + 500)
        return [self._draw(rng, self.box) for _ in range(n)]

    def val(self, n: int = 2) -> List[dict]:
        rng = np.random.default_rng(self.seed_base + 550)
        return [self._draw(rng, self.box) for _ in range(n)]

    def in_range(self, n: int = 12) -> List[dict]:
        """Stratified: n/3 per third of the ``strata`` parameter's range,
        drawn inside a 5% margin of the training box."""
        rng = np.random.default_rng(self.seed_base + 600)
        m = {k: (lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
             for k, (lo, hi) in self.box.items()}
        lo, hi = m[self.strata]
        edges = np.linspace(lo, hi, 4)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            for _ in range(n // 3):
                p = self._draw(rng, m)
                p[self.strata] = float(rng.uniform(a, b))
                out.append(p)
        return out

    def out_of_range(self, n: int = 8) -> List[dict]:
        rng = np.random.default_rng(self.seed_base + 700)
        box = dict(self.box)
        box.update(self.oor)
        return [self._draw(rng, box) for _ in range(n)]


def _pack(items):
    d = {}
    for dof, val in items:
        d[int(dof)] = float(val)
    con = np.array(sorted(d), dtype=np.int64)
    full = np.array([d[c] for c in con])
    # every prescribed value here is 0 or u_delta, so scaling by the load
    # fraction is exact
    return con, (lambda frac: full * frac)


# --------------------------------------------------------------------------
class OpenHole(Family):
    name, strata, seed_base = "open_hole", "dW", 10
    keys = ("W", "dW", "LW")
    box = {"W": (16.0, 26.0), "dW": (0.2, 0.5), "LW": (1.5, 2.5)}
    oor = {"dW": (0.55, 0.60)}

    def dims(self, p):
        H = 0.5 * p["W"]
        return {"L": p["LW"] * p["W"], "H": H, "r": p["dW"] * H}

    def shape(self, p):
        d = self.dims(p)
        return OpenHolePlate(d["L"], d["H"], d["r"])


class Inclusion(OpenHole):
    name, seed_base = "inclusion", 20

    def shape(self, p):
        d = self.dims(p)
        return RigidInclusionPlate(d["L"], d["H"], d["r"])

    def spec(self, p, u_delta):
        d = self.dims(p)
        return adf.quarter_tension(d["L"], d["H"], u_delta,
                                   arc=((0.0, 0.0), d["r"]))

    def fem_dirichlet(self, mesh, u_delta):
        b = mesh.boundary
        items = ([(2 * i, 0.0) for i in b["left_symmetry"]]
                 + [(2 * i, u_delta) for i in b["right_grip"]]
                 + [(2 * i + 1, 0.0) for i in b["bottom"]]
                 + [(2 * i + c, 0.0) for i in b["inclusion"] for c in (0, 1)])
        return _pack(items)


class DoubleNotch(Family):
    name, strata, seed_base = "double_notch", "rt", 30
    keys = ("W", "tW", "rt", "LW")
    box = {"W": (16.0, 26.0), "tW": (0.1, 0.3), "rt": (0.25, 1.0),
           "LW": (1.5, 2.5)}
    oor = {"rt": (0.15, 0.25)}

    def dims(self, p):
        t = p["tW"] * p["W"]
        return {"L": p["LW"] * p["W"], "H": 0.5 * p["W"], "t": t,
                "rho": p["rt"] * t}

    def shape(self, p):
        d = self.dims(p)
        return DoubleNotchPlate(d["L"], d["H"], d["t"], d["rho"])


class SingleNotch(Family):
    name, strata, seed_base = "single_notch", "tW", 40
    keys = ("W", "tW", "rt", "LW")
    box = {"W": (16.0, 26.0), "tW": (0.1, 0.3), "rt": (0.25, 1.0),
           "LW": (2.0, 3.0)}
    oor = {"tW": (0.30, 0.35)}

    def dims(self, p):
        t = p["tW"] * p["W"]
        return {"L": p["LW"] * p["W"], "H": 0.5 * p["W"], "t": t,
                "rho": p["rt"] * t}

    def shape(self, p):
        d = self.dims(p)
        return SingleNotchPlate(d["L"], d["H"], d["t"], d["rho"])

    def extent(self, p):
        d = self.dims(p)
        return d["L"], 2.0 * d["H"]

    def spec(self, p, u_delta):
        d = self.dims(p)
        return adf.half_clamped_tension(d["L"], 2.0 * d["H"], u_delta)

    def fem_dirichlet(self, mesh, u_delta):
        b = mesh.boundary
        items = ([(2 * i, 0.0) for i in b["left_symmetry"]]
                 + [(2 * i, u_delta) for i in b["right_grip"]]
                 + [(2 * i + 1, 0.0) for i in b["right_grip"]])
        return _pack(items)

    def peak_mask(self, cent, p):
        return cent[:, 0] <= 0.5 * self.dims(p)["L"]


FAMILIES = {f.name: f for f in (OpenHole(), Inclusion(), DoubleNotch(),
                                SingleNotch())}
