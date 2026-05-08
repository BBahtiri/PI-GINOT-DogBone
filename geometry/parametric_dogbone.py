#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric DogBone specimen geometry generator — pure NumPy (no gmsh).

Generates 2D quarter-model (double symmetry about y=0 and x=0) DogBone
geometries with:
  - Variable grip/gauge widths and total length
  - Single circular fillet arc connecting grip to gauge section

No-hole single-topology operator.  Holes are permanently disabled.

The boundary is defined analytically (lines + circular arc), so no mesher
is required.  Interior collocation points are generated via rejection
sampling inside the polygonal/arc boundary.

Returns structured data suitable for:
  - Boundary point cloud extraction (for GINOT encoder)
  - Interior/boundary collocation point sampling (for physics loss)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    GEOMETRY_DEFAULT, GEOMETRY_RANGES, HOLE_CONFIG, COLLOCATION_CONFIG,
    validate_geometry, get_fillet_geometry,
)


@dataclass
class BoundarySegment:
    """A labelled segment of the domain boundary."""
    name: str                       # e.g. 'bottom', 'right_grip', 'right_arc', ...
    points: np.ndarray              # (N, 2) coordinates
    normals: np.ndarray             # (N, 2) outward unit normals
    bc_type: str                    # 'symmetry_y', 'dirichlet', 'traction_free',
                                    # 'symmetry_x'


@dataclass
class DogBoneMesh:
    """Complete geometry data for a single DogBone instance (mesh-free)."""
    params: dict                    # Geometry parameters that produced this
    nodes: np.ndarray               # (N_nodes, 2) — union of interior + boundary
    elements: np.ndarray            # empty (no mesh); kept for API compat
    interior_nodes: np.ndarray      # (N_int, 2) interior point coordinates
    boundary_segments: List[BoundarySegment]
    boundary_pc: np.ndarray         # (N_bnd, 2) full boundary point cloud
    boundary_normals: np.ndarray    # (N_bnd, 2) normals for boundary_pc
    fillet_info: dict               # Derived fillet geometry


# Analytical boundary sampling

def _sample_line(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    """Sample n equispaced points on the line segment p0 -> p1 (inclusive)."""
    t = np.linspace(0, 1, n).reshape(-1, 1)
    return ((1 - t) * p0 + t * p1).astype(np.float32)


def _sample_arc(
    center: Tuple[float, float],
    radius: float,
    theta_start: float,
    theta_end: float,
    n: int,
) -> np.ndarray:
    """Sample n equispaced points on a circular arc (angles in radians)."""
    thetas = np.linspace(theta_start, theta_end, n)
    x = center[0] + radius * np.cos(thetas)
    y = center[1] + radius * np.sin(thetas)
    return np.stack([x, y], axis=-1).astype(np.float32)


def _arc_outward_normals(
    pts: np.ndarray,
    center: Tuple[float, float],
    outward_sign: float = -1.0,
) -> np.ndarray:
    """Compute unit normals for arc points.

    outward_sign = -1  -> normals point *toward* center (concave fillet)
    outward_sign = +1  -> normals point *away* from center
    """
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.maximum(dist, 1e-10)
    normals = np.stack([outward_sign * dx / dist,
                        outward_sign * dy / dist], axis=-1)
    return normals.astype(np.float32)


def _sample_circle(
    center: Tuple[float, float],
    radius: float,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n points on a full circle.  Returns (points, inward normals).

    Kept for API compatibility but should never be called in no-hole mode.
    """
    thetas = np.linspace(0, 2 * math.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(thetas)
    y = center[1] + radius * np.sin(thetas)
    pts = np.stack([x, y], axis=-1).astype(np.float32)
    # Inward normals (toward center = outward from solid domain)
    normals = _arc_outward_normals(pts, center, outward_sign=-1.0)
    return pts, normals


# Quarter-model geometry generation (no-hole topology)

def generate_dogbone(
    params: Optional[dict] = None,
    n_pts_per_segment: int = 100,
    n_interior: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> DogBoneMesh:
    """Generate a DogBone quarter-model from geometry parameters.

    Quarter model: upper-right quarter with symmetry at x=0 and y=0.
    No-hole topology only.

    Boundary segments (counter-clockwise):
      1. bottom        : (0,0) -> (L_half, 0)         symmetry_y
      2. right_grip    : (L_half, 0) -> (L_half, H_grip)  dirichlet
      3. right_arc     : fillet from grip to gauge    traction_free
      4. gauge_top     : (x_g, H_gauge) -> (0, H_gauge)  traction_free
      5. left_symmetry : (0, H_gauge) -> (0, 0)       symmetry_x

    Args:
        params: Geometry dict with keys: L_total, W_grip, W_gauge, R_fillet.
        n_pts_per_segment: Points per boundary segment.
        n_interior: Interior collocation points.
        rng: NumPy random generator.
        verbose: Print debug info.

    Returns:
        DogBoneMesh with all geometry data.
    """
    if params is None:
        params = GEOMETRY_DEFAULT.copy()
    if n_interior is None:
        n_interior = COLLOCATION_CONFIG["n_interior"]
    if rng is None:
        rng = np.random.default_rng()

    if not validate_geometry(params):
        raise ValueError(f"Invalid geometry parameters: {params}")

    # Hard guard: reject any non-empty holes
    holes = []
    if len(params.get("holes", [])) > 0:
        raise ValueError(
            "This PI-GINOT dogbone setup is configured for no-hole "
            "geometries only."
        )

    fillet = get_fillet_geometry(params)

    L_half  = fillet["L_half"]
    H_grip  = fillet["H_grip"]
    H_gauge = fillet["H_gauge"]
    R       = fillet["R_fillet"]
    x_g     = fillet["x_g"]
    arc_center = fillet["arc_center"]

    # Arc angular extent
    #   Arc center = (x_g, H_gauge + R).
    #   P3 = (L_half, H_grip) -> theta_start = atan2(dH - R, dx_fillet)
    #   P4 = (x_g, H_gauge)   -> theta_end   = -pi/2
    dH = fillet["dH"]
    dx_fillet = fillet["dx"]

    theta_start = math.atan2(dH - R, dx_fillet)
    theta_end   = -math.pi / 2.0

    n = n_pts_per_segment

    # Sample each boundary segment (counter-clockwise)
    segments: List[BoundarySegment] = []

    # 1) Bottom (horizontal symmetry, y=0): (0,0) -> (L_half,0)
    pts_bottom = _sample_line(np.array([0.0, 0.0]), np.array([L_half, 0.0]), n)
    norm_bottom = np.tile(np.array([0.0, -1.0], dtype=np.float32), (n, 1))
    segments.append(BoundarySegment("bottom", pts_bottom, norm_bottom, "symmetry_y"))

    # 2) Right grip: (L_half,0) -> (L_half, H_grip)
    pts_right = _sample_line(np.array([L_half, 0.0]), np.array([L_half, H_grip]), n)
    norm_right = np.tile(np.array([1.0, 0.0], dtype=np.float32), (n, 1))
    segments.append(BoundarySegment("right_grip", pts_right, norm_right, "dirichlet"))

    # 3) Right fillet arc: P3(L_half, H_grip) -> P4(x_g, H_gauge)
    pts_arc = _sample_arc(arc_center, R, theta_start, theta_end, n)
    norm_arc = _arc_outward_normals(pts_arc, arc_center, outward_sign=-1.0)
    segments.append(BoundarySegment("right_arc", pts_arc, norm_arc, "traction_free"))

    # 4) Gauge top: (x_g, H_gauge) -> (0, H_gauge)
    pts_gauge = _sample_line(np.array([x_g, H_gauge]), np.array([0.0, H_gauge]), n)
    norm_gauge = np.tile(np.array([0.0, 1.0], dtype=np.float32), (n, 1))
    segments.append(BoundarySegment("gauge_top", pts_gauge, norm_gauge, "traction_free"))

    # 5) Left symmetry (vertical symmetry, x=0): (0, H_gauge) -> (0, 0)
    pts_left = _sample_line(np.array([0.0, H_gauge]), np.array([0.0, 0.0]), n)
    norm_left = np.tile(np.array([-1.0, 0.0], dtype=np.float32), (n, 1))
    segments.append(BoundarySegment("left_symmetry", pts_left, norm_left, "symmetry_x"))

    # Boundary point cloud (all segments concatenated)
    all_bnd_pts = np.concatenate([s.points for s in segments], axis=0)
    all_bnd_normals = np.concatenate([s.normals for s in segments], axis=0)

    # Interior points via rejection sampling
    interior_nodes = _sample_biased_interior(
        fillet, holes, n_interior, rng
    )

    # Union of all points
    all_nodes = np.concatenate([interior_nodes, all_bnd_pts], axis=0)

    return DogBoneMesh(
        params=params,
        nodes=all_nodes,
        elements=np.empty((0, 3), dtype=int),
        interior_nodes=interior_nodes,
        boundary_segments=segments,
        boundary_pc=all_bnd_pts,
        boundary_normals=all_bnd_normals,
        fillet_info=fillet,
    )


# Interior point rejection sampling (quarter-model)

def _rejection_sample_interior(
    fillet: dict,
    holes: list,
    n: int,
    rng: np.random.Generator,
    oversampling: float = 2.5,
) -> np.ndarray:
    """Generate interior points via rejection sampling in the quarter domain."""
    L_half = fillet["L_half"]
    H_grip = fillet["H_grip"]

    collected = []
    n_collected = 0

    while n_collected < n:
        batch = max(100, int(oversampling * (n - n_collected)))
        x = rng.uniform(0, L_half, size=batch)
        y = rng.uniform(0, H_grip, size=batch)

        inside = _point_in_dogbone(x, y, fillet)

        # Exclude holes (no-op with holes=[])
        for cx, cy, r in holes:
            dist_sq = (x - cx)**2 + (y - cy)**2
            inside &= dist_sq > r**2

        good = np.stack([x[inside], y[inside]], axis=-1)
        if len(good) > 0:
            collected.append(good)
            n_collected += len(good)

    return np.concatenate(collected, axis=0)[:n].astype(np.float32)


def _point_in_dogbone(
    x: np.ndarray,
    y: np.ndarray,
    fillet: dict,
) -> np.ndarray:
    """Vectorised test: is (x, y) inside the DogBone quarter-model?

    Two x-zones:
      gauge:        x in [0, x_g]       -> y <= H_gauge
      fillet/grip:  x in [x_g, L_half]  -> arc constraint (dist >= R)
    """
    L_half = fillet["L_half"]
    H_grip = fillet["H_grip"]
    H_gauge = fillet["H_gauge"]
    R = fillet["R_fillet"]
    x_g = fillet["x_g"]
    ac = fillet["arc_center"]

    inside = (x >= 0) & (x <= L_half) & (y >= 0) & (y <= H_grip)

    # Gauge zone: y <= H_gauge
    in_gauge = (x <= x_g)
    inside &= ~in_gauge | (y <= H_gauge)

    # Fillet zone: y must be below the arc
    in_fillet = (x > x_g)
    dist_sq = (x - ac[0])**2 + (y - ac[1])**2
    inside &= ~in_fillet | (dist_sq >= R**2)

    return inside


def _sample_biased_interior(
    fillet: dict,
    holes: list,
    n: int,
    rng: np.random.Generator,
    fillet_fraction: float = 0.5,
) -> np.ndarray:
    """Mixed interior sampler: uniform + fillet-biased.

    Holes are permanently disabled, so hole-biased sampling is removed.
    """
    n_fillet = int(n * fillet_fraction)
    n_uniform = n - n_fillet

    parts = []

    # Uniform interior
    if n_uniform > 0:
        pts_u = _rejection_sample_interior(fillet, holes, n_uniform, rng)
        parts.append(pts_u)

    # Fillet-biased: sample near the single fillet zone
    if n_fillet > 0:
        L_half = fillet["L_half"]
        H_grip = fillet["H_grip"]
        x_g = fillet["x_g"]
        fillet_width = max(L_half - x_g, 2.0)

        collected = []
        n_collected = 0
        while n_collected < n_fillet:
            batch = max(100, int(3.0 * (n_fillet - n_collected)))
            x_center = (x_g + L_half) / 2.0
            x = x_center + rng.normal(0, fillet_width * 0.4, size=batch)
            y = rng.uniform(0, H_grip, size=batch)
            x = np.clip(x, 0, L_half)
            y = np.clip(y, 0, H_grip)

            inside = _point_in_dogbone(x, y, fillet)
            # holes is [] so this loop is a no-op
            for cx, cy, r in holes:
                inside &= (x - cx)**2 + (y - cy)**2 > r**2
            good = np.stack([x[inside], y[inside]], axis=-1)
            if len(good) > 0:
                collected.append(good)
                n_collected += len(good)
        parts.append(np.concatenate(collected, axis=0)[:n_fillet])

    return np.concatenate(parts, axis=0)[:n].astype(np.float32)

# Parametric sampling (no-hole only)

def sample_geometry_params(
    rng: np.random.Generator,
    geometry_ranges: Optional[dict] = None,
    holes_enabled: bool = False,      # kept only for API compatibility
    hole_config: Optional[dict] = None,
    max_attempts: int = 100,
) -> dict:
    """Sample a valid random DogBone geometry parameter set.

    Always produces no-hole geometries.  The holes_enabled and hole_config
    arguments are kept for API compatibility but are ignored.
    """
    if geometry_ranges is None:
        geometry_ranges = GEOMETRY_RANGES

    for _ in range(max_attempts):
        params = {}
        for key, (lo, hi) in geometry_ranges.items():
            params[key] = float(rng.uniform(lo, hi))

        params["holes"] = []

        if validate_geometry(params):
            return params

    raise RuntimeError(
        f"Could not sample valid geometry in {max_attempts} attempts."
    )


def sample_batch(
    batch_size: int,
    rng: np.random.Generator,
    geometry_ranges: Optional[dict] = None,
    holes_enabled: bool = False,
) -> List[dict]:
    """Sample a batch of valid geometry parameter sets."""
    return [
        sample_geometry_params(rng, geometry_ranges, holes_enabled)
        for _ in range(batch_size)
    ]


def plot_dogbone(mesh: DogBoneMesh, show_normals: bool = True, ax=None):
    """Plot the DogBone boundary with labelled segments and normals."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Interior points
    if len(mesh.interior_nodes) > 0:
        ax.scatter(mesh.interior_nodes[:, 0], mesh.interior_nodes[:, 1],
                   s=0.3, c="lightgray", alpha=0.5, label="interior")

    # Boundary segments
    colors = {
        "bottom": "blue", "right_grip": "red", "right_arc": "orange",
        "gauge_top": "green", "left_symmetry": "purple",
    }

    for seg in mesh.boundary_segments:
        color = colors.get(seg.name, "brown")
        ax.plot(seg.points[:, 0], seg.points[:, 1], ".-", color=color,
                markersize=2, linewidth=1.0, label=seg.name)

        if show_normals:
            step = max(1, len(seg.points) // 12)
            scale = mesh.fillet_info["L_half"] * 0.03
            ax.quiver(
                seg.points[::step, 0], seg.points[::step, 1],
                seg.normals[::step, 0] * scale,
                seg.normals[::step, 1] * scale,
                color=color, angles="xy", scale_units="xy", scale=1,
                width=0.003, alpha=0.8,
            )

    ax.set_aspect("equal")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7,
              loc="upper left", ncol=2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    L_half = mesh.fillet_info["L_half"]
    H = mesh.fillet_info["H_grip"]
    ax.set_title(f"DogBone quarter-model  (L_half={L_half:.1f}, W_grip={2*H:.1f})")
    ax.autoscale_view()
    return ax
