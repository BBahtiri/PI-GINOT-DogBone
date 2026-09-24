#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Level 0 (geometry): are the boundary points, normals and measures right?

The manufactured-solution check in ``verification/mms.py`` compares P.N using
the mesh's own normals on both sides, so it cannot see a normal that points
the wrong way -- the error cancels.  This module closes that gap by comparing
the geometry itself against its closed form.

A wrong normal is the worst kind of defect here: every traction term in the
loss is ``P.N``, so flipping a sign turns "traction-free" into a constraint
that pushes the field somewhere else entirely, and the training curve looks
perfectly healthy while doing it.  Likewise, boundary points are sampled
proportional to arc length so the penalty terms approximate a surface
integral; if that weighting is off, the fillet is silently under- or
over-represented relative to the flat faces.

Checks
------
1. Every boundary point lies on the analytic boundary (signed distance ~ 0).
2. Normals are unit length.
3. Normals point *outward*: stepping a small distance along +N leaves the
   domain and along -N stays inside, tested with ``_point_in_dogbone``, the
   same predicate the interior sampler uses.
4. Flat-segment normals equal their exact constant value; fillet normals
   equal (center - p)/R, the outward normal of a concave arc.
5. Sampled arc length matches R*dtheta, and the polyline length of every
   segment matches its closed form.
6. The boundary sampler distributes points in proportion to arc length.
7. The five segments form a closed loop end to end.

Run with:  python -m verification.geometry_checks
"""

from __future__ import annotations

import numpy as np

from config import (COLLOCATION_CONFIG, GEOMETRY_DEFAULT, GEOMETRY_RANGES,
                    get_fillet_geometry)
from geometry.banks import val_bank_params
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import _point_in_dogbone, generate_dogbone

# Exact constant outward normals of the four non-arc segments.
FLAT_NORMALS = {
    "bottom": (0.0, -1.0),
    "right_grip": (1.0, 0.0),
    "gauge_top": (0.0, 1.0),
    "left_symmetry": (-1.0, 0.0),
}


def exact_segment_lengths(fi: dict) -> dict:
    """Closed-form length of each boundary segment."""
    R, dx = fi["R_fillet"], fi["dx"]
    return {
        "bottom": fi["L_half"],
        "right_grip": fi["H_grip"],
        "right_arc": R * np.arcsin(min(dx / R, 1.0)),
        "gauge_top": fi["x_g"],
        "left_symmetry": fi["H_gauge"],
    }


def _signed_distance_to_boundary(seg_name: str, pts: np.ndarray,
                                 fi: dict) -> np.ndarray:
    """Distance from each point to the analytic curve carrying that segment."""
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    if seg_name == "bottom":
        return np.abs(y)
    if seg_name == "right_grip":
        return np.abs(x - fi["L_half"])
    if seg_name == "gauge_top":
        return np.abs(y - fi["H_gauge"])
    if seg_name == "left_symmetry":
        return np.abs(x)
    xc, yc = fi["arc_center"]
    return np.abs(np.hypot(x - xc, y - yc) - fi["R_fillet"])


def check_one_geometry(params: dict, n_per_segment: int = None,
                       step_frac: float = 1e-4,
                       measure_consistent: bool = None) -> dict:
    """All geometry checks for a single specimen, as relative errors."""
    if n_per_segment is None:
        n_per_segment = COLLOCATION_CONFIG["n_boundary_per_segment"]
    mesh = generate_dogbone(params, n_pts_per_segment=n_per_segment,
                            rng=np.random.default_rng(0),
                            measure_consistent=measure_consistent)
    fi = mesh.fillet_info
    L0 = fi["L_half"]
    exact_len = exact_segment_lengths(fi)
    step = step_frac * L0

    out = {"on_boundary": 0.0, "unit_normals": 0.0, "normal_direction": 0.0,
           "outward_fail": 0, "inward_fail": 0, "length": 0.0,
           "loop_gap": 0.0, "sampler_measure": 0.0, "pc_density_ratio": 0.0,
           "boundary_shortfall": 0.0}

    seg_by_name = {s.name: s for s in mesh.boundary_segments}

    for name, seg in seg_by_name.items():
        pts = seg.points.astype(float)
        nrm = seg.normals.astype(float)

        # 1. points lie on the analytic curve
        out["on_boundary"] = max(
            out["on_boundary"],
            float(_signed_distance_to_boundary(name, pts, fi).max() / L0))

        # 2. unit length
        out["unit_normals"] = max(
            out["unit_normals"],
            float(np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max()))

        # 4. exact normal direction
        if name in FLAT_NORMALS:
            ref = np.broadcast_to(np.array(FLAT_NORMALS[name]), nrm.shape)
        else:
            xc, yc = fi["arc_center"]
            d = np.stack([xc - pts[:, 0], yc - pts[:, 1]], axis=-1)
            ref = d / np.linalg.norm(d, axis=1, keepdims=True)
        out["normal_direction"] = max(
            out["normal_direction"], float(np.abs(nrm - ref).max()))

        # 3. outwardness, against the interior predicate.  Corner points sit
        #    on two segments at once, so a step along one segment's normal can
        #    legitimately stay inside; skip the two endpoints of each segment.
        core = slice(2, -2)
        p_out = pts[core] + step * nrm[core]
        p_in = pts[core] - step * nrm[core]
        out["outward_fail"] += int(
            _point_in_dogbone(p_out[:, 0], p_out[:, 1], fi).sum())
        out["inward_fail"] += int(
            (~_point_in_dogbone(p_in[:, 0], p_in[:, 1], fi)).sum())

        # 5. polyline length vs closed form
        poly = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        out["length"] = max(out["length"],
                            abs(poly - exact_len[name]) / exact_len[name])

    # 7. closed loop: each segment ends where the next begins
    order = ["bottom", "right_grip", "right_arc", "gauge_top", "left_symmetry"]
    for a, b in zip(order, order[1:] + order[:1]):
        gap = np.linalg.norm(seg_by_name[a].points[-1].astype(float)
                             - seg_by_name[b].points[0].astype(float))
        out["loop_gap"] = max(out["loop_gap"], float(gap / L0))

    # 6. measure consistency, at the settings actually shipped in config.py.
    #    Two separate quantities:
    #      pc_density_ratio   -- how uneven the encoder's boundary point cloud
    #                            is in points per millimetre (want ~1)
    #      sampler_measure    -- how far the physics collocation points are
    #                            from an arc-length-proportional split
    #      boundary_shortfall -- fraction of the requested boundary points the
    #                            sampler could not deliver, because a segment's
    #                            pool is smaller than its arc-length share
    total_len = sum(exact_len.values())
    dens = {k: len(seg_by_name[k].points) / exact_len[k] for k in exact_len}
    out["pc_density_ratio"] = max(dens.values()) / min(dens.values())

    n_req = COLLOCATION_CONFIG["n_total_boundary"]
    coll = sample_collocation_points(
        mesh, n_interior=1, n_total_boundary=n_req,
        n_boundary_pc=COLLOCATION_CONFIG["n_boundary_pc"],
        rng=np.random.default_rng(1))
    n_total = sum(len(v) for v in coll.boundary_pts.values())
    out["boundary_shortfall"] = 1.0 - n_total / n_req
    for name, pts in coll.boundary_pts.items():
        got = len(pts) / n_total
        want = exact_len[name] / total_len
        out["sampler_measure"] = max(out["sampler_measure"], abs(got - want))

    return out


def run_all(n_geometries: int = 24, measure_consistent: bool = None) -> dict:
    """Worst case over the validation bank plus the default geometry.

    Extremes matter more than the average here: a near-tangent fillet
    (R barely above dH) is where an arc parametrisation goes wrong first.
    """
    params = [dict(GEOMETRY_DEFAULT)] + val_bank_params(n_geometries)

    # Deliberate corner cases: the shallowest and the deepest fillet the
    # parameter ranges allow.
    lo_R = GEOMETRY_RANGES["R_fillet"][0]
    hi_R = GEOMETRY_RANGES["R_fillet"][1]
    params += [
        dict(L_total=40.0, W_grip=26.0, W_gauge=6.0, R_fillet=hi_R, holes=[]),
        dict(L_total=70.0, W_grip=16.5, W_gauge=16.0, R_fillet=lo_R, holes=[]),
    ]

    worst = {}
    per_geo = []
    for p in params:
        r = check_one_geometry(p, measure_consistent=measure_consistent)
        per_geo.append(r)
        for k, v in r.items():
            worst[k] = max(worst.get(k, 0.0), v)
    return {"worst": worst, "n_geometries": len(params), "per_geometry": per_geo}


GEOMETRY_TOL = 1e-5


def main():
    rep = run_all()
    w = rep["worst"]
    print("Level 0 — boundary geometry, normals and measure")
    print("=" * 68)
    print(f"\nWorst case over {rep['n_geometries']} geometries "
          "(bank + default + two corner cases)\n")
    print("  Exactness of the boundary description")
    for label, key, unit in [
        ("points lie on the analytic curve", "on_boundary", "rel. to L_half"),
        ("normals are unit length", "unit_normals", "absolute"),
        ("normals match the closed form", "normal_direction", "absolute"),
        ("segment length vs closed form", "length", "relative"),
        ("boundary loop closes", "loop_gap", "rel. to L_half"),
    ]:
        print(f"    {label:<36} {w[key]:.3e}   ({unit})")
    print(f"    {'points outside after +N step':<36} "
          f"{int(w['outward_fail']):d}         (count, want 0)")
    print(f"    {'points inside after -N step':<36} "
          f"{int(w['inward_fail']):d}         (count, want 0)")

    print("\n  Measure consistency, at the shipped config settings")
    mc = COLLOCATION_CONFIG.get("boundary_measure_consistent", False)
    print(f"    boundary_measure_consistent = {mc}")
    for label, key, unit in [
        ("encoder PC density max/min", "pc_density_ratio", "ratio, want 1"),
        ("collocation vs arc-length split", "sampler_measure", "fraction"),
        ("requested boundary pts not returned", "boundary_shortfall",
         "fraction"),
    ]:
        print(f"    {label:<36} {w[key]:.3f}   ({unit})")

    print("\n" + "=" * 68)
    bad = [k for k in ("on_boundary", "unit_normals", "normal_direction",
                       "length", "loop_gap") if w[k] > GEOMETRY_TOL]
    bad += [k for k in ("outward_fail", "inward_fail") if w[k] > 0]
    if bad:
        print("FAIL: " + ", ".join(bad))
    else:
        print("Boundary description: all checks pass.")
    if w["pc_density_ratio"] > 1.5 or w["sampler_measure"] > 0.02:
        print("Measure consistency: NOT satisfied at these settings — set\n"
              "  COLLOCATION_CONFIG['boundary_measure_consistent'] = True\n"
              "  to share the same point budget out by arc length instead.")
    else:
        print("Measure consistency: satisfied.")


if __name__ == "__main__":
    main()
