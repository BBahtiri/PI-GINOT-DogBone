#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sampling a finite-element field at arbitrary points.

Nodal quantities (displacement) are interpolated with the element's own P1
shape functions, which is exact for the discrete solution.  Element
quantities (stress, constant per element on a CST) are looked up by
containing element -- no smoothing or nodal averaging, so a reported peak is
a value the reference actually holds rather than an artefact of
post-processing.
"""

from __future__ import annotations

import weakref

import numpy as np
from scipy.spatial import Delaunay

from verification.fem.mesh import FemMesh

# Keyed by id(mesh), and every entry carries a weak reference to the mesh it
# was built for.  The id alone is not an identity: CPython reuses the id of a
# freed object for the next one allocated.  Before this fix, a caller that
# loaded a reference solution, used it and let it go -- the scoring loops in
# supervised_ceiling and mixed_pilot, which unpickle one solve per geometry --
# could be handed the locator and element map of a *previous* mesh.  When the
# stale mesh had more elements than the current one that raised IndexError,
# which is how it was found; when it had fewer, it silently returned the wrong
# elements.  The only quantity routed through here is the FEM section force
# (axial_resultant), so only N_fem / N_rel_err could be corrupted -- K_t, the
# stress fields and the displacements are all read at the reference's own
# centroids and nodes.  Callers that kept every solution alive for the whole
# process (verification.operator_study._REF_CACHE) never reused an id.
_CACHE = {}


def _entry(mesh: FemMesh):
    """(Delaunay locator, Delaunay-simplex -> mesh-element map) for ``mesh``."""
    key = id(mesh)
    hit = _CACHE.get(key)
    if hit is not None and hit[0]() is mesh:
        return hit[1], hit[2]
    tri = Delaunay(mesh.nodes)
    lookup = {tuple(sorted(t)): e for e, t in enumerate(mesh.tris)}
    mapping = np.array([lookup.get(tuple(sorted(t)), -1) for t in tri.simplices],
                       dtype=np.int64)
    _CACHE[key] = (weakref.ref(mesh), tri, mapping)
    return tri, mapping


def _locator(mesh: FemMesh) -> Delaunay:
    """A point-location structure over the mesh's own triangles."""
    return _entry(mesh)[0]


def barycentric(mesh: FemMesh, pts: np.ndarray):
    """Containing element and barycentric coordinates for each point.

    Returns (elem_index, lambdas); elem_index is -1 where the point lies
    outside every element.
    """
    tri, mapping = _entry(mesh)
    simp = tri.find_simplex(pts)

    # tri.simplices is Delaunay's own triangulation of the node set, which
    # includes elements outside the domain; map back to the mesh's elements.
    elem = np.full(len(pts), -1, dtype=np.int64)
    lam = np.zeros((len(pts), 3))

    ok = simp >= 0
    elem[ok] = mapping[simp[ok]]

    good = elem >= 0
    if good.any():
        t = mesh.tris[elem[good]]
        p = mesh.nodes[t]
        v0 = p[:, 1] - p[:, 0]
        v1 = p[:, 2] - p[:, 0]
        v2 = pts[good] - p[:, 0]
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        lam[good] = np.stack([1.0 - l1 - l2, l1, l2], axis=-1)
    return elem, lam


def sample_nodal(mesh: FemMesh, nodal: np.ndarray,
                 pts: np.ndarray) -> np.ndarray:
    """P1-interpolate a nodal field. NaN outside the mesh."""
    elem, lam = barycentric(mesh, pts)
    out = np.full((len(pts),) + nodal.shape[1:], np.nan)
    ok = elem >= 0
    if ok.any():
        vals = nodal[mesh.tris[elem[ok]]]           # (n, 3, ...)
        w = lam[ok]
        out[ok] = np.einsum("na,na...->n...", w, vals)
    return out


def sample_elementwise(mesh: FemMesh, elem_vals: np.ndarray,
                       pts: np.ndarray) -> np.ndarray:
    """Look up a per-element field. NaN outside the mesh."""
    elem, _ = barycentric(mesh, pts)
    out = np.full((len(pts),) + elem_vals.shape[1:], np.nan)
    ok = elem >= 0
    out[ok] = elem_vals[elem[ok]]
    return out
