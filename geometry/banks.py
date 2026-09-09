#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed geometry banks, shared by the trainer and the evaluation code.

The bank RNG is consumed by three calls per geometry -- the parameter draw,
the mesh generation and the collocation sampling -- so the parameter sequence
depends on whether meshes are built at all.  Any code that needs "the
validation bank" must therefore go through this one function rather than
re-drawing parameters itself, or it will silently evaluate on a different set
of specimens than the one that was trained and validated against.
"""

from typing import List, Optional

import numpy as np

from config import COLLOCATION_CONFIG
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params

# Bank seeds, fixed for the life of the project.
TRAIN_BANK_SEED = 100
VAL_BANK_SEED = 200


def build_geometry_bank(n: int, geo_ranges: Optional[dict], holes_on: bool,
                        seed: int, mesh_only: bool = False) -> List:
    """Pre-generate a fixed bank of geometries.

    Args:
        n: number of geometries.
        geo_ranges: sampling ranges, or None for the config defaults.
        holes_on: whether to sample holes (permanently False for this operator).
        seed: bank RNG seed.
        mesh_only: True returns parameter dicts and never touches the mesh
            generator, so the meshes and collocation can be regenerated fresh
            each epoch.  False returns (mesh, coll) pairs with frozen
            collocation.  Note that the two modes yield *different* parameter
            sequences, because mesh generation advances the same RNG.

    Returns:
        list of dicts (mesh_only=True) or of (DogBoneMesh, CollocationData).
    """
    bank_rng = np.random.default_rng(seed)
    bank = []
    for _ in range(n):
        params = sample_geometry_params(
            bank_rng, geometry_ranges=geo_ranges, holes_enabled=holes_on,
        )
        if mesh_only:
            bank.append(params)
        else:
            n_bnd_seg = COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
            mesh = generate_dogbone(params, n_pts_per_segment=n_bnd_seg,
                                    rng=bank_rng)
            coll = sample_collocation_points(mesh, rng=bank_rng)
            bank.append((mesh, coll))
    return bank


def train_bank_params(n: int, geo_ranges: Optional[dict] = None) -> List[dict]:
    """Parameters of the training bank, exactly as the trainer builds it.

    The trainer uses mesh_only=True for the train bank (collocation is
    regenerated each epoch), so the parameter sequence differs from the
    validation bank's -- see build_geometry_bank.
    """
    return build_geometry_bank(n, geo_ranges, holes_on=False,
                               seed=TRAIN_BANK_SEED, mesh_only=True)


def val_bank_params(n: int, geo_ranges: Optional[dict] = None) -> List[dict]:
    """Parameters of the frozen validation bank, exactly as the trainer builds it."""
    bank = build_geometry_bank(n, geo_ranges, holes_on=False,
                               seed=VAL_BANK_SEED, mesh_only=False)
    return [mesh.params for mesh, _ in bank]
