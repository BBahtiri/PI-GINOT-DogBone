#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent verification grid for PI-GINOT reliability diagnostics.

Key design constraints (§3 of design doc):
  - Seed MUST differ from any training / validation seed
  - Sampling density MUST differ from training collocation
  - Includes stratified regions: deep interior, near-boundary layer,
    fillet-adjacent, and symmetry axes
  - Section slice x-positions MUST be disjoint from training slices
  - Checkpoint records training seeds; this module reads them and
    picks guaranteed-disjoint seeds
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


from .constants import (
    TRAINING_SECTION_XI as _TRAINING_XI,
    VERIFICATION_SECTION_XI as _VERIF_XI,
    AGENT_SEEDS,
    verify_seed_disjoint,
)

# Section slices sourced from central registry (constants.py enforces disjointness)
TRAINING_SECTION_XI = np.array(_TRAINING_XI)
VERIFICATION_SECTION_XI = np.array(_VERIF_XI)


@dataclass
class VerificationGrid:
    """Independent point set for reliability diagnostics.

    All arrays are numpy float32, in physical (un-normalized) coordinates.
    """
    # Stratified interior points
    interior_pts: np.ndarray            # [N_int, 2]
    near_boundary_pts: np.ndarray       # [N_bnd, 2]
    fillet_pts: np.ndarray              # [N_fil, 2]
    symmetry_x_pts: np.ndarray          # [N_sx, 2]  (near x=0)
    symmetry_y_pts: np.ndarray          # [N_sy, 2]  (near y=0)

    # Union of all points
    all_pts: np.ndarray                 # [N_total, 2]

    # Section slices
    section_xi: np.ndarray              # fractional x-positions
    section_x_positions: np.ndarray     # physical x-positions [mm]

    # Metadata
    seed: int
    n_total: int
    fillet_info: dict

    @property
    def L_half(self) -> float:
        return self.fillet_info["L_half"]


def _point_in_dogbone(
    x: np.ndarray,
    y: np.ndarray,
    fillet: dict,
) -> np.ndarray:
    """Vectorised point-in-domain test for the DogBone quarter-model."""
    L_half = fillet["L_half"]
    H_grip = fillet["H_grip"]
    H_gauge = fillet["H_gauge"]
    R = fillet["R_fillet"]
    x_g = fillet["x_g"]
    ac = fillet["arc_center"]

    inside = (x >= 0) & (x <= L_half) & (y >= 0) & (y <= H_grip)
    in_gauge = x <= x_g
    inside &= ~in_gauge | (y <= H_gauge)
    in_fillet = x > x_g
    dist_sq = (x - ac[0]) ** 2 + (y - ac[1]) ** 2
    inside &= ~in_fillet | (dist_sq >= R ** 2)
    return inside


def _rejection_sample(
    fillet: dict,
    n: int,
    rng: np.random.Generator,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Rejection sample n points inside the DogBone quarter-model.

    Optionally constrain to a sub-rectangle [x_lo, x_hi] x [y_lo, y_hi].
    """
    L_half = fillet["L_half"]
    H_grip = fillet["H_grip"]

    x_lo = x_range[0] if x_range else 0.0
    x_hi = x_range[1] if x_range else L_half
    y_lo = y_range[0] if y_range else 0.0
    y_hi = y_range[1] if y_range else H_grip

    collected = []
    n_got = 0
    while n_got < n:
        batch = max(100, int(3.0 * (n - n_got)))
        x = rng.uniform(x_lo, x_hi, size=batch)
        y = rng.uniform(y_lo, y_hi, size=batch)
        mask = _point_in_dogbone(x, y, fillet)
        good = np.stack([x[mask], y[mask]], axis=-1)
        if len(good) > 0:
            collected.append(good)
            n_got += len(good)
    return np.concatenate(collected, axis=0)[:n].astype(np.float32)


def _section_width(x: float, fillet: dict) -> float:
    """Compute the half-width at position x in the quarter-model."""
    H_gauge = fillet["H_gauge"]
    R = fillet["R_fillet"]
    x_g = fillet["x_g"]
    ac = fillet["arc_center"]

    if x <= x_g:
        return H_gauge
    else:
        dx = x - ac[0]
        dy_sq = R ** 2 - dx ** 2
        if dy_sq <= 0:
            return fillet["H_grip"]
        return ac[1] - math.sqrt(dy_sq)


def build_verification_grid(
    fillet_info: dict,
    seed: int = AGENT_SEEDS["verification_grid"],
    n_deep_interior: int = 800,
    n_near_boundary: int = 400,
    n_fillet: int = 600,
    n_symmetry_x: int = 100,
    n_symmetry_y: int = 100,
    training_seeds: Optional[List[int]] = None,
) -> VerificationGrid:
    """Build an independent verification grid for one geometry.

    Args:
        fillet_info: Dict from config.get_fillet_geometry().
        seed: RNG seed for verification (must differ from training seeds).
        n_deep_interior: Points in interior away from boundaries.
        n_near_boundary: Points in a thin boundary layer.
        n_fillet: Points concentrated near the fillet transition.
        n_symmetry_x: Points near x=0 symmetry axis.
        n_symmetry_y: Points near y=0 symmetry axis.
        training_seeds: Seeds used during training (from checkpoint).
            If provided, asserts non-overlap.

    Returns:
        VerificationGrid with all stratified points.
    """
    # Enforce seed disjointness against both the central registry
    # and the checkpoint's recorded training seeds
    verify_seed_disjoint(seed, "verification_grid")
    if training_seeds is not None:
        assert seed not in training_seeds, (
            f"Verification seed {seed} overlaps with checkpoint training seeds "
            f"{training_seeds}. Pick a disjoint seed."
        )

    rng = np.random.default_rng(seed)

    L_half = fillet_info["L_half"]
    H_grip = fillet_info["H_grip"]
    H_gauge = fillet_info["H_gauge"]
    x_g = fillet_info["x_g"]

    # Margin for "near boundary" vs "deep interior"
    margin = min(L_half, H_grip) * 0.08

    # 1. Deep interior: exclude points within `margin` of any boundary
    interior_pts = _rejection_sample(
        fillet_info, n_deep_interior, rng,
        x_range=(margin, L_half - margin),
        y_range=(margin, H_gauge - margin),  # stay in gauge for safety
    )

    # 2. Near-boundary layer: points within `margin` of the boundary
    #    Sample from full domain then keep only those near edges
    near_bnd_pts = _sample_near_boundary(
        fillet_info, n_near_boundary, rng, margin
    )

    # 3. Fillet-adjacent: concentrated near the fillet zone
    fillet_width = max(L_half - x_g, 2.0)
    fillet_center_x = (x_g + L_half) / 2.0
    fillet_pts = _sample_fillet_region(
        fillet_info, n_fillet, rng, fillet_center_x, fillet_width
    )

    # 4. Symmetry x-axis: points near x=0
    sym_x_pts = _rejection_sample(
        fillet_info, n_symmetry_x, rng,
        x_range=(0.0, margin * 2),
        y_range=(margin, H_gauge - margin),
    )

    # 5. Symmetry y-axis: points near y=0
    sym_y_pts = _rejection_sample(
        fillet_info, n_symmetry_y, rng,
        x_range=(margin, L_half - margin),
        y_range=(0.0, margin * 2),
    )

    # Union
    all_pts = np.concatenate(
        [interior_pts, near_bnd_pts, fillet_pts, sym_x_pts, sym_y_pts],
        axis=0,
    )

    # Section slices (disjoint from training)
    section_x = VERIFICATION_SECTION_XI * L_half

    return VerificationGrid(
        interior_pts=interior_pts,
        near_boundary_pts=near_bnd_pts,
        fillet_pts=fillet_pts,
        symmetry_x_pts=sym_x_pts,
        symmetry_y_pts=sym_y_pts,
        all_pts=all_pts,
        section_xi=VERIFICATION_SECTION_XI,
        section_x_positions=section_x,
        seed=seed,
        n_total=len(all_pts),
        fillet_info=fillet_info,
    )


def _sample_near_boundary(
    fillet: dict,
    n: int,
    rng: np.random.Generator,
    margin: float,
) -> np.ndarray:
    """Sample points near the boundary of the domain."""
    L_half = fillet["L_half"]
    H_grip = fillet["H_grip"]

    collected = []
    n_got = 0
    while n_got < n:
        batch = max(200, int(5.0 * (n - n_got)))
        x = rng.uniform(0, L_half, size=batch)
        y = rng.uniform(0, H_grip, size=batch)

        inside = _point_in_dogbone(x, y, fillet)

        # Check proximity to boundary: at least one coordinate near edge
        near_x0 = x < margin
        near_xL = x > L_half - margin
        near_y0 = y < margin
        # Near top boundary depends on position
        near_top = np.zeros(batch, dtype=bool)
        in_gauge = x <= fillet["x_g"]
        near_top[in_gauge] = y[in_gauge] > fillet["H_gauge"] - margin
        near_top[~in_gauge] = True  # fillet zone is always "near boundary"

        near_any = near_x0 | near_xL | near_y0 | near_top
        mask = inside & near_any

        good = np.stack([x[mask], y[mask]], axis=-1)
        if len(good) > 0:
            collected.append(good)
            n_got += len(good)

    return np.concatenate(collected, axis=0)[:n].astype(np.float32)


def _sample_fillet_region(
    fillet: dict,
    n: int,
    rng: np.random.Generator,
    center_x: float,
    width: float,
) -> np.ndarray:
    """Sample points concentrated near the fillet arc."""
    H_grip = fillet["H_grip"]
    L_half = fillet["L_half"]

    collected = []
    n_got = 0
    while n_got < n:
        batch = max(200, int(3.0 * (n - n_got)))
        x = center_x + rng.normal(0, width * 0.4, size=batch)
        y = rng.uniform(0, H_grip, size=batch)
        x = np.clip(x, 0, L_half)
        y = np.clip(y, 0, H_grip)

        mask = _point_in_dogbone(x, y, fillet)
        good = np.stack([x[mask], y[mask]], axis=-1)
        if len(good) > 0:
            collected.append(good)
            n_got += len(good)

    return np.concatenate(collected, axis=0)[:n].astype(np.float32)
