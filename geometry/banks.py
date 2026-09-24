#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed geometry banks, shared by the trainer and the evaluation code.

A geometry's parameters depend only on (seed, index).  That sounds like it
should go without saying, and it did not: the bank originally drew parameters,
meshes and collocation points from one shared RNG, so the parameter sequence
depended on how much randomness the mesh and collocation happened to consume.
Changing ``n_interior`` changed which specimens a seed produced, and
"validation geometry 11" named different shapes in different runs -- which is
how the Phase 2-4 studies ended up evaluating on a narrow taper band while
believing they had chosen a spread.  Independent spawned sub-streams fix it;
``legacy_rng=True`` reproduces the old behaviour for results measured under it.
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
                        seed: int, mesh_only: bool = False,
                        legacy_rng: bool = False) -> List:
    """Pre-generate a fixed bank of geometries.

    Args:
        n: number of geometries.
        geo_ranges: sampling ranges, or None for the config defaults.
        holes_on: whether to sample holes (permanently False for this operator).
        seed: bank RNG seed.
        mesh_only: True returns parameter dicts and never touches the mesh
            generator, so the meshes and collocation can be regenerated fresh
            each epoch.  False returns (mesh, coll) pairs with frozen
            collocation.  Both modes now yield the SAME parameter sequence.
        legacy_rng: reproduce the pre-fix single-stream draw, under which the
            parameter sequence depended on the collocation settings and on
            mesh_only.  Every result measured before this fix used it.

    Returns:
        list of dicts (mesh_only=True) or of (DogBoneMesh, CollocationData).
    """
    # One independent stream per geometry, and three independent sub-streams
    # within each, so a geometry's PARAMETERS depend only on (seed, index).
    #
    # The single shared stream this replaces made the parameter sequence
    # depend on how much randomness the mesh and collocation happened to
    # consume -- so changing `n_interior`, `n_boundary_per_segment` or
    # `mesh_only` silently produced a different set of specimens from the same
    # seed, and "validation geometry 11" meant different shapes in different
    # runs.  Every result measured before this fix refers to the legacy draw;
    # pass ``legacy_rng=True`` to reproduce one.
    root = np.random.SeedSequence(seed)
    bank = []
    for child in root.spawn(n):
        if legacy_rng:
            break
        p_seed, m_seed, c_seed = child.spawn(3)
        params = sample_geometry_params(
            np.random.default_rng(p_seed), geometry_ranges=geo_ranges,
            holes_enabled=holes_on,
        )
        if mesh_only:
            bank.append(params)
            continue
        n_bnd_seg = COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
        mesh = generate_dogbone(params, n_pts_per_segment=n_bnd_seg,
                                rng=np.random.default_rng(m_seed))
        coll = sample_collocation_points(
            mesh, rng=np.random.default_rng(c_seed))
        bank.append((mesh, coll))
    if not legacy_rng:
        return bank

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
