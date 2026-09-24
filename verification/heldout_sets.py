#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fresh held-out sets, split by the range of the bank the model trained on.

Phase 6.21 found that two of the eight validation geometries used since
Phase 5 (taper 0.77, 0.82) sit outside the 64-geometry training bank's taper
range (0.261-0.747).  Under the uniform parameter sampler P(taper > 0.75) is
2.9%, so a 64-bank expects 1.9 such geometries; this one drew none.  The
high-taper corner needs W_gauge high and W_grip low at once, so it is thin,
and every raw parameter of those two geometries is inside its own training
range -- they are outside the *joint* distribution, not outside any single
coordinate.

The post-hoc split that exposed this was drawn after the failures were seen.
These sets are the clean version: new seeds that have never been used for any
hypothesis, both sets fixed before any model is scored on them.

  in-range (IR)       12 geometries.  Every raw parameter inside the training
                      bank's drawn min/max, and taper in [0.30, 0.70] -- a
                      margin of 0.04-0.05 inside the bank's 0.261-0.747 so no
                      member sits on the edge.  Stratified, four per third of
                      the band: an unstratified first draw landed 0.308-0.588
                      and left the upper third empty, which would have made
                      the in-range side easy and the contrast with the
                      out-of-range side wider than the test intends.  Changed
                      before any model was scored on either set.
  out-of-range (OOR)  8 geometries.  Every raw parameter inside the training
                      bank's drawn min/max, and taper >= 0.78 -- beyond the
                      bank's 0.747 by a margin, so extrapolation in the joint
                      geometry and in no single coordinate.

Both are drawn from the project's own sampler (sample_geometry_params), so
the non-taper parameters follow the distribution the model was trained on,
conditioned on the taper band.
"""

from __future__ import annotations

import numpy as np

import config as cfg
from config import TRAINING_CONFIG
from geometry.banks import TRAIN_BANK_SEED, build_geometry_bank
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params

IR_SEED, OOR_SEED = 300, 301
N_IR, N_OOR = 12, 8
IR_TAPER = (0.30, 0.70)
OOR_TAPER_MIN = 0.78
KEYS = ("L_total", "W_grip", "W_gauge", "R_fillet")


def training_box(n_bank: int = 64) -> dict:
    """Per-parameter min/max actually drawn in the training bank."""
    ps = build_geometry_bank(n_bank, TRAINING_CONFIG["bank_geo_ranges"],
                             holes_on=False, seed=TRAIN_BANK_SEED,
                             mesh_only=True)
    box = {k: (min(p[k] for p in ps), max(p[k] for p in ps)) for k in KEYS}
    tap = [p["W_gauge"] / p["W_grip"] for p in ps]
    box["taper"] = (min(tap), max(tap))
    return box


def _inside(p, box):
    return all(box[k][0] <= p[k] <= box[k][1] for k in KEYS)


def _draw(seed, n, keep, max_draws=200000):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(max_draws):
        p = sample_geometry_params(rng, TRAINING_CONFIG["bank_geo_ranges"])
        if keep(p):
            out.append(p)
            if len(out) == n:
                return out
    raise RuntimeError(f"only {len(out)} of {n} geometries after {max_draws}")


def param_sets(n_bank: int = 64):
    box = training_box(n_bank)
    taper = lambda p: p["W_gauge"] / p["W_grip"]
    edges = np.linspace(IR_TAPER[0], IR_TAPER[1], 4)
    ir = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        ir += _draw(IR_SEED + 10 * j, N_IR // 3,
                    lambda p, lo=lo, hi=hi: _inside(p, box)
                    and lo <= taper(p) < hi)
    oor = _draw(OOR_SEED, N_OOR, lambda p: _inside(p, box)
                and taper(p) >= OOR_TAPER_MIN)
    return ir, oor, box


def entries(params_list, seed):
    """(mesh, collocation) for each geometry, frozen by (seed, index)."""
    n_seg = cfg.COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
    root = np.random.SeedSequence(seed + 1000)
    out = []
    for p, child in zip(params_list, root.spawn(len(params_list))):
        m_seed, c_seed = child.spawn(2)
        mesh = generate_dogbone(p, n_pts_per_segment=n_seg,
                                rng=np.random.default_rng(m_seed))
        out.append((mesh, sample_collocation_points(
            mesh, rng=np.random.default_rng(c_seed))))
    return out


if __name__ == "__main__":
    ir, oor, box = param_sets()
    print("training bank, drawn ranges:")
    for k, (lo, hi) in box.items():
        print(f"  {k:>9}: {lo:7.3f} .. {hi:7.3f}")
    for name, ps in (("in-range", ir), ("out-of-range", oor)):
        t = [p["W_gauge"] / p["W_grip"] for p in ps]
        print(f"\n{name}: {len(ps)} geometries, taper {min(t):.3f} .. {max(t):.3f}")
        for i, p in enumerate(ps):
            print(f"  {i:>2}  " + "  ".join(f"{k}={p[k]:6.2f}" for k in KEYS)
                  + f"  taper={p['W_gauge'] / p['W_grip']:.3f}")
