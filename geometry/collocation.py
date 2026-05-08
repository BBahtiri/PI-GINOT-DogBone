#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collocation point sampling from DogBone mesh data.

Provides:
  - Interior point sampling (for equilibrium PDE residual)
  - Boundary point sampling per segment (arc-length weighted)
  - Boundary point cloud assembly (for GINOT geometry encoder input)
  - Coordinate normalization utilities

Boundary condition mapping (quarter-model, double symmetry):
  ┌──────────────────┬──────────────────────┬─────────────────────────────┐
  │ Boundary         │ Hard BC (architecture)│ Soft traction (loss)        │
  ├──────────────────┼──────────────────────┼─────────────────────────────┤
  │ Bottom (y=0)     │ v = 0  (via y·φ_v)   │ (P·N)_x = 0  (x-dir only) │
  │ Left sym (x=0)   │ u = 0  (via x/L·…)   │ (P·N)_y = 0  (y-dir only) │
  │ Right grip (x=L) │ u = u_δ (hard BC)    │ (P·N)_y = 0  (y-dir only) │
  │ Gauge top        │ —                    │ P·N = 0   (both dirs)       │
  │ Arc              │ —                    │ P·N = 0   (both dirs)       │
  │ Holes            │ —                    │ P·N = 0   (both dirs)       │
  └──────────────────┴──────────────────────┴─────────────────────────────┘
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COLLOCATION_CONFIG, NORMALIZATION_CONFIG

from .parametric_dogbone import DogBoneMesh


@dataclass
class CollocationData:
    """All collocation point data needed for one DogBone geometry instance."""
    # Geometry encoder input
    boundary_pc: np.ndarray             # (N_pc, 2)
    boundary_pc_normals: np.ndarray     # (N_pc, 2)

    # Interior points for equilibrium loss
    interior_pts: np.ndarray            # (N_int, 2)

    # Boundary points per segment
    boundary_pts: Dict[str, np.ndarray]
    boundary_normals: Dict[str, np.ndarray]
    boundary_bc_types: Dict[str, str]

    # Full traction-free boundary (arc, gauge top, holes): P·N = 0 both dirs
    traction_free_pts: np.ndarray       # (N_trac, 2)
    traction_free_normals: np.ndarray   # (N_trac, 2)
    traction_free_tags: np.ndarray      # (N_trac,) int: 0=gauge_top, 1=arc, 2=hole

    # Partial traction boundary (symmetry + grip): single component of P·N = 0
    partial_traction_pts: np.ndarray        # (N_part, 2)
    partial_traction_normals: np.ndarray    # (N_part, 2)
    partial_traction_dirs: np.ndarray       # (N_part,)  0=x-dir, 1=y-dir

    # Displacement BC boundary (right grip for u_delta)
    dirichlet_pts: np.ndarray           # (N_dir, 2)

    # Normalization bounds
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    # Original geometry parameters
    params: dict


# Partial traction direction per segment in the quarter-model:
#   bottom (y=0):       v=0 is hard -> enforce trac_x (dir=0)
#   left_symmetry (x=0): u=0 is hard -> enforce trac_y (dir=1)
#   right_grip (x=L):  u=u_delta is hard -> enforce trac_y (dir=1)
_PARTIAL_TRACTION_MAP = {
    "bottom":         0,   # symmetry at y=0: v=0 hard -> enforce (P·N)_x = 0
    "left_symmetry":  1,   # symmetry at x=0: u=0 hard -> enforce (P·N)_y = 0
    "right_grip":     1,   # loaded grip: u=u_delta hard -> enforce (P·N)_y = 0
}


def _segment_length(points: np.ndarray) -> float:
    """Estimate segment arc length from its sampled points."""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def sample_collocation_points(
    mesh: DogBoneMesh,
    n_interior: Optional[int] = None,
    n_total_boundary: Optional[int] = None,
    n_boundary_pc: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> CollocationData:
    """Sample collocation points from a DogBone mesh.

    Boundary points are sampled proportional to segment arc length
    (measure-consistent) rather than equal per segment.

    Args:
        mesh: DogBoneMesh from generate_dogbone().
        n_interior: Number of interior points. Defaults to config.
        n_total_boundary: Total boundary points distributed by arc length.
        n_boundary_pc: Size of boundary PC for encoder. Defaults to config.
        rng: Random generator.

    Returns:
        CollocationData with all point sets ready for physics loss.
    """
    if n_interior is None:
        n_interior = COLLOCATION_CONFIG["n_interior"]
    if n_total_boundary is None:
        n_total_boundary = COLLOCATION_CONFIG.get("n_total_boundary", 1200)
    if n_boundary_pc is None:
        n_boundary_pc = COLLOCATION_CONFIG["n_boundary_pc"]
    if rng is None:
        rng = np.random.default_rng()

    # Interior points
    interior_pts = _subsample(mesh.interior_nodes, n_interior, rng)

    # Segment arc lengths for proportional allocation
    seg_names = [seg.name for seg in mesh.boundary_segments]
    seg_lengths = np.array([_segment_length(seg.points) for seg in mesh.boundary_segments])

    total_length = seg_lengths.sum()
    if total_length < 1e-12:
        total_length = 1.0

    # Multinomial allocation: exact total with minimum 10 per segment
    n_segs = len(seg_names)
    min_per_seg = 10
    remaining = n_total_boundary - n_segs * min_per_seg
    if remaining < 0:
        remaining = 0
    fracs = seg_lengths / total_length
    raw = fracs * remaining
    extra = np.floor(raw).astype(int)
    left = remaining - extra.sum()
    if left > 0:
        # Largest-remainder: give +1 to segments with biggest fractional parts
        order = np.argsort(-(raw - extra))
        extra[order[:left]] += 1
    seg_counts = {name: min_per_seg + int(e) for name, e in zip(seg_names, extra)}

    # Sample boundary points proportional to arc length
    boundary_pts = {}
    boundary_normals = {}
    boundary_bc_types = {}

    for seg in mesh.boundary_segments:
        n_seg = max(1, seg_counts[seg.name])

        if len(seg.points) > n_seg:
            idx = rng.choice(len(seg.points), size=n_seg, replace=False)
            pts = seg.points[idx]
            norms = seg.normals[idx]
        else:
            pts = seg.points.copy()
            norms = seg.normals.copy()

        boundary_pts[seg.name] = pts.astype(np.float32)
        boundary_normals[seg.name] = norms.astype(np.float32)
        boundary_bc_types[seg.name] = seg.bc_type

    # Boundary point cloud for encoder input
    if len(mesh.boundary_pc) > n_boundary_pc:
        idx = rng.choice(len(mesh.boundary_pc), size=n_boundary_pc, replace=False)
        boundary_pc = mesh.boundary_pc[idx]
        boundary_pc_normals = mesh.boundary_normals[idx]
    else:
        boundary_pc = mesh.boundary_pc.copy()
        boundary_pc_normals = mesh.boundary_normals.copy()

    # Full traction-free points (both components of P·N = 0)
    # Tag each point: 0=gauge_top, 1=arc, 2=hole
    trac_pts_list = []
    trac_norm_list = []
    trac_tag_list = []
    for name, bc_type in boundary_bc_types.items():
        if bc_type in ("traction_free", "hole"):
            n_pts = len(boundary_pts[name])
            trac_pts_list.append(boundary_pts[name])
            trac_norm_list.append(boundary_normals[name])
            if "arc" in name:
                tag = 1   # fillet arc
            elif bc_type == "hole":
                tag = 2   # hole boundary
            else:
                tag = 0   # gauge_top (flat)
            trac_tag_list.append(np.full(n_pts, tag, dtype=np.int32))

    if trac_pts_list:
        traction_free_pts = np.concatenate(trac_pts_list, axis=0)
        traction_free_normals = np.concatenate(trac_norm_list, axis=0)
        traction_free_tags = np.concatenate(trac_tag_list, axis=0)
    else:
        traction_free_pts = np.empty((0, 2), dtype=np.float32)
        traction_free_normals = np.empty((0, 2), dtype=np.float32)
        traction_free_tags = np.empty((0,), dtype=np.int32)

    # Partial traction points (single component of P·N = 0 on symmetry/grip)
    part_pts_list = []
    part_norm_list = []
    part_dir_list = []
    for name, bc_type in boundary_bc_types.items():
        if name in _PARTIAL_TRACTION_MAP:
            trac_dir = _PARTIAL_TRACTION_MAP[name]
            n_pts = len(boundary_pts[name])
            part_pts_list.append(boundary_pts[name])
            part_norm_list.append(boundary_normals[name])
            part_dir_list.append(np.full(n_pts, trac_dir, dtype=np.int32))

    if part_pts_list:
        partial_traction_pts = np.concatenate(part_pts_list, axis=0)
        partial_traction_normals = np.concatenate(part_norm_list, axis=0)
        partial_traction_dirs = np.concatenate(part_dir_list, axis=0)
    else:
        partial_traction_pts = np.empty((0, 2), dtype=np.float32)
        partial_traction_normals = np.empty((0, 2), dtype=np.float32)
        partial_traction_dirs = np.empty((0,), dtype=np.int32)

    # Dirichlet BC points (right grip)
    dirichlet_pts = boundary_pts.get("right_grip", np.empty((0, 2), dtype=np.float32))

    # Normalization bounds
    all_pts = mesh.nodes
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    return CollocationData(
        boundary_pc=boundary_pc.astype(np.float32),
        boundary_pc_normals=boundary_pc_normals.astype(np.float32),
        interior_pts=interior_pts,
        boundary_pts=boundary_pts,
        boundary_normals=boundary_normals,
        boundary_bc_types=boundary_bc_types,
        traction_free_pts=traction_free_pts,
        traction_free_normals=traction_free_normals,
        traction_free_tags=traction_free_tags,
        partial_traction_pts=partial_traction_pts,
        partial_traction_normals=partial_traction_normals,
        partial_traction_dirs=partial_traction_dirs,
        dirichlet_pts=dirichlet_pts,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        params=mesh.params,
    )


# Coordinate normalization utilities

def normalize_coords(pts: np.ndarray, coll: CollocationData) -> np.ndarray:
    """Normalize coordinates to [-1, 1] based on domain bounds."""
    x_norm = 2.0 * (pts[:, 0] - coll.x_min) / (coll.x_max - coll.x_min) - 1.0
    y_norm = 2.0 * (pts[:, 1] - coll.y_min) / (coll.y_max - coll.y_min) - 1.0
    return np.stack([x_norm, y_norm], axis=-1).astype(np.float32)


def denormalize_coords(pts_norm: np.ndarray, coll: CollocationData) -> np.ndarray:
    """Inverse of normalize_coords: [-1, 1] to physical coordinates."""
    x_phys = (pts_norm[:, 0] + 1.0) / 2.0 * (coll.x_max - coll.x_min) + coll.x_min
    y_phys = (pts_norm[:, 1] + 1.0) / 2.0 * (coll.y_max - coll.y_min) + coll.y_min
    return np.stack([x_phys, y_phys], axis=-1).astype(np.float32)


def _subsample(
    pts: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample n points from pts without replacement."""
    if len(pts) == 0:
        return np.empty((0, pts.shape[1] if pts.ndim > 1 else 2), dtype=np.float32)
    if len(pts) <= n:
        return pts.copy().astype(np.float32)
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx].astype(np.float32)


def collocation_to_torch(coll: CollocationData, device: str = "cpu") -> dict:
    """Convert CollocationData numpy arrays to torch tensors."""
    import torch

    def _to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def _to_int_tensor(arr):
        return torch.tensor(arr, dtype=torch.long, device=device)

    return {
        "boundary_pc": _to_tensor(coll.boundary_pc),
        "boundary_pc_normals": _to_tensor(coll.boundary_pc_normals),
        "interior_pts": _to_tensor(coll.interior_pts),
        "traction_free_pts": _to_tensor(coll.traction_free_pts),
        "traction_free_normals": _to_tensor(coll.traction_free_normals),
        "partial_traction_pts": _to_tensor(coll.partial_traction_pts),
        "partial_traction_normals": _to_tensor(coll.partial_traction_normals),
        "partial_traction_dirs": _to_int_tensor(coll.partial_traction_dirs),
        "dirichlet_pts": _to_tensor(coll.dirichlet_pts),
        "x_min": coll.x_min,
        "x_max": coll.x_max,
        "y_min": coll.y_min,
        "y_max": coll.y_max,
        "params": coll.params,
        "boundary_segments": {
            name: {
                "pts": _to_tensor(coll.boundary_pts[name]),
                "normals": _to_tensor(coll.boundary_normals[name]),
                "bc_type": coll.boundary_bc_types[name],
            }
            for name in coll.boundary_pts
        },
    }
