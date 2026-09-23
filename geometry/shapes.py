#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specimen families beyond the dog-bone, and a mesher that takes any of them
(Phase 8.2).

``geometry/triangulation.py`` meshes the dog-bone and is left untouched -- it
is verified and every reference in the project depends on it.  This module
repeats its construction (graded boundary nodes, structured offset layers
along the curved feature, a jittered lattice thinned by a size field,
Delaunay, centroid filter, guarded Laplacian smoothing, sliver removal)
against a small ``Shape`` interface instead of the dog-bone's fixed segments.

Segment names follow the FEM solver's convention, so
``verification.fem.solver.solve`` runs unchanged on any shape: the Dirichlet
sets are ``left_symmetry`` (u = 0), ``right_grip`` (u = u_delta) and
``bottom`` (v = 0).  ``info["L_half"]`` seeds its affine initial guess.

Families (quarter models in x-tension unless stated):

  OpenHolePlate(L, H, r)          central circular hole, radius r
                                  (width W = 2H, d/W = r/H)

The double-edge notch, the bonded rigid inclusion and the single-edge notch
of the Phase 8 plan follow the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.spatial import Delaunay

from geometry.triangulation import TriMesh


class Shape:
    """Interface: boundary geometry, membership, and where to refine."""
    name = "shape"
    segments: tuple = ()

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def feature_distance(self, pts: np.ndarray) -> np.ndarray:
        """Distance to the stress-concentration feature (refinement target)."""
        raise NotImplementedError

    def boundary(self, size) -> Dict[str, np.ndarray]:
        """Nodes per segment, given a callable size(pts) -> local h."""
        raise NotImplementedError

    def offset_layers(self, h_min: float, n: int) -> List[np.ndarray]:
        return []

    @property
    def feature_length(self) -> float:
        raise NotImplementedError

    @property
    def info(self) -> dict:
        raise NotImplementedError


def _graded_line(p0, p1, size, n_probe=4000):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n_probe)
    pts = p0 + t[:, None] * (p1 - p0)
    length = np.linalg.norm(p1 - p0)
    dens = np.cumsum(1.0 / size(pts)) * (length / n_probe)
    n = max(2, int(round(dens[-1])))
    tt = np.interp(np.linspace(0.0, dens[-1], n), dens, t)
    return p0 + tt[:, None] * (p1 - p0)


@dataclass
class OpenHolePlate(Shape):
    """Quarter of a plate 2L x 2H with a central hole of radius r.

    Segments (counter-clockwise): bottom (r, 0) -> (L, 0); right_grip;
    top (L, H) -> (0, H); left_symmetry (0, H) -> (0, r); hole (0, r) -> (r, 0).
    """
    L: float
    H: float
    r: float
    name = "open_hole"
    segments = ("bottom", "right_grip", "top", "left_symmetry", "hole")

    def inside(self, x, y):
        return ((x >= -1e-12) & (x <= self.L + 1e-12) & (y >= -1e-12)
                & (y <= self.H + 1e-12) & (x * x + y * y >= self.r ** 2 - 1e-9))

    def feature_distance(self, pts):
        return np.abs(np.hypot(pts[:, 0], pts[:, 1]) - self.r)

    @property
    def feature_length(self):
        return self.r

    def boundary(self, size):
        L, H, r = self.L, self.H, self.r
        h_arc = float(size(np.array([[0.0, r]]))[0])
        n_arc = max(8, int(round(0.5 * np.pi * r / h_arc)))
        th = np.linspace(0.5 * np.pi, 0.0, n_arc)
        return {
            "bottom": _graded_line((r, 0.0), (L, 0.0), size),
            "right_grip": _graded_line((L, 0.0), (L, H), size),
            "top": _graded_line((L, H), (0.0, H), size),
            "left_symmetry": _graded_line((0.0, H), (0.0, r), size),
            "hole": np.stack([r * np.cos(th), r * np.sin(th)], -1),
        }

    def offset_layers(self, h_min, n):
        out = []
        for k in range(1, n + 1):
            R = self.r + k * h_min
            m = max(4, int(round(0.5 * np.pi * R / h_min)) - k)
            th = np.linspace(0.5 * np.pi, 0.0, m)[1:-1]
            p = np.stack([R * np.cos(th), R * np.sin(th)], -1)
            out.append(p[self.inside(p[:, 0], p[:, 1])])
        return out

    @property
    def info(self):
        return {"family": self.name, "L_half": self.L, "H_grip": self.H,
                "r": self.r, "d_over_W": self.r / self.H}


# --------------------------------------------------------------------------
def _smooth(nodes, tris, fixed, shape, n_sweeps=12, relax=0.6):
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    for _ in range(n_sweeps):
        acc = np.zeros_like(nodes)
        cnt = np.zeros(len(nodes))
        np.add.at(acc, edges[:, 0], nodes[edges[:, 1]])
        np.add.at(cnt, edges[:, 0], 1.0)
        target = acc / np.maximum(cnt, 1.0)[:, None]
        cand = nodes.copy()
        move = ~fixed
        cand[move] = (1 - relax) * nodes[move] + relax * target[move]
        out = move & ~shape.inside(cand[:, 0], cand[:, 1])
        cand[out] = nodes[out]
        p = cand[tris]
        a2 = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
              - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        bad = a2 <= 1e-12
        if bad.any():
            rev = np.unique(tris[bad].ravel())
            cand[rev] = nodes[rev]
        nodes = cand
    return nodes


def build_shape_mesh(shape: Shape, h: float, refine: float = 6.0,
                     decay: float = 0.6, n_offset_layers: int = 3,
                     seed: int = 0, min_quality: float = 0.05,
                     n_smooth: int = 12) -> TriMesh:
    """Graded triangulation of any ``Shape``; same recipe as the dog-bone's."""
    rng = np.random.default_rng(seed)
    lf = shape.feature_length

    def size(pts):
        w = np.exp(-(shape.feature_distance(pts) / (decay * lf)) ** 2)
        return h / (1.0 + (refine - 1.0) * w)

    bnd = shape.boundary(size)
    boundary_pts = np.concatenate([bnd[s] for s in shape.segments], axis=0)
    h_min = h / refine
    layers = shape.offset_layers(h_min, n_offset_layers)

    info = shape.info
    xmax, ymax = boundary_pts.max(0)
    xmin, ymin = boundary_pts.min(0)
    interior = []
    for j, y in enumerate(np.arange(ymin + 0.5 * h_min, ymax, h_min)):
        xs = np.arange(xmin + (j % 2) * 0.5 * h_min, xmax, h_min)
        p = np.stack([xs, np.full_like(xs, y)], -1)
        p = p[shape.inside(p[:, 0], p[:, 1])]
        if not len(p):
            continue
        p = p[rng.random(len(p)) < (h_min / size(p)) ** 2]
        if len(p):
            p = p + rng.normal(0.0, 0.15 * h_min, size=p.shape)
            interior.append(p[shape.inside(p[:, 0], p[:, 1])])

    nodes = np.unique(np.round(np.concatenate([boundary_pts] + layers + interior), 9),
                      axis=0)
    # keep interior nodes off the boundary: drop any closer than h_loc/4 to a
    # boundary node (Delaunay would otherwise make slivers along the edges)
    is_b = np.zeros(len(nodes), bool)
    lookup = {tuple(np.round(p, 9)): True for p in boundary_pts}
    for i, p in enumerate(nodes):
        is_b[i] = tuple(np.round(p, 9)) in lookup
    from scipy.spatial import cKDTree
    tree = cKDTree(boundary_pts)
    d, _ = tree.query(nodes)
    keep = is_b | (d > 0.25 * size(nodes))
    nodes = nodes[keep]

    def triangulate(nodes):
        simp = Delaunay(nodes).simplices
        c = nodes[simp].mean(1)
        return simp[shape.inside(c[:, 0], c[:, 1])]

    simp = triangulate(nodes)
    if n_smooth:
        bset = {tuple(np.round(p, 9)) for p in boundary_pts}
        fixed = np.array([tuple(np.round(p, 9)) in bset for p in nodes])
        nodes = _smooth(nodes, simp, fixed, shape, n_sweeps=n_smooth)
        simp = triangulate(nodes)

    p = nodes[simp]
    tw = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
          - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    flip = tw < 0
    simp[flip] = simp[flip][:, [0, 2, 1]]
    tw = np.abs(tw)
    areas = 0.5 * tw
    p = nodes[simp]
    e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], 1)
    q = 4.0 * np.sqrt(3.0) * areas / np.maximum((e ** 2).sum(-1).sum(-1), 1e-300)
    good = q >= min_quality
    simp, areas, tw = simp[good], areas[good], tw[good]
    p = nodes[simp]
    grads = np.empty((len(simp), 3, 2))
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        grads[:, a, 0] = (p[:, b, 1] - p[:, c, 1]) / tw
        grads[:, a, 1] = (p[:, c, 0] - p[:, b, 0]) / tw

    lookup = {tuple(np.round(n, 9)): i for i, n in enumerate(nodes)}
    boundary = {s: np.array([i for i in (lookup.get(tuple(np.round(pt, 9)))
                                         for pt in bnd[s]) if i is not None],
                            dtype=np.int64) for s in shape.segments}
    used = np.zeros(len(nodes), bool)
    used[simp.ravel()] = True
    if not used.all():
        remap = -np.ones(len(nodes), np.int64)
        remap[used] = np.arange(used.sum())
        nodes, simp = nodes[used], remap[simp]
        boundary = {k: remap[v][remap[v] >= 0] for k, v in boundary.items()}
    return TriMesh(params=dict(info), fillet_info=info, nodes=nodes,
                   tris=simp.astype(np.int32), areas=areas, grads=grads,
                   boundary=boundary, h_target=h)
