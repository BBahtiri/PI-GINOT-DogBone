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
    def feature_disc(self):
        """(cx, cy, R): a disc containing the refinement feature."""
        raise NotImplementedError

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

    @property
    def feature_disc(self):
        return (0.0, 0.0, self.r)

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
                     n_smooth: int = 12, grade: float = None) -> TriMesh:
    """Graded triangulation of any ``Shape``; same recipe as the dog-bone's.

    Size field: ``h`` far away and ``h / refine`` on the feature.  With
    ``grade`` the size grows linearly away from the feature,
    h_loc = min(h, h / refine + grade * d), so the resolution *around* a
    small feature scales with the feature.  Without it the dog-bone's
    Gaussian blend over ``decay * feature_length`` is used; at a large
    ``refine`` that blend is so abrupt that the field within one notch radius
    of the root is under-resolved -- 6.6% low on a shallow semicircular notch
    (Phase 8.2c, first Level-3 run).
    """
    rng = np.random.default_rng(seed)
    lf = shape.feature_length

    def size(pts):
        d = shape.feature_distance(pts)
        if grade is not None:
            return np.minimum(h, h / refine + grade * d)
        w = np.exp(-(d / (decay * lf)) ** 2)
        return h / (1.0 + (refine - 1.0) * w)

    bnd = shape.boundary(size)
    boundary_pts = np.concatenate([bnd[s] for s in shape.segments], axis=0)
    h_min = h / refine
    layers = shape.offset_layers(h_min, n_offset_layers)

    info = shape.info
    xmax, ymax = boundary_pts.max(0)
    xmin, ymin = boundary_pts.min(0)
    # Interior nodes, level by level: a lattice of spacing s is laid only
    # where the target size lies in [s, 2s), and only inside the box that can
    # contain such points (the feature disc grown by the distance at which
    # the size reaches 2s).  A single lattice at the finest spacing over the
    # whole specimen -- the dog-bone mesher's approach -- costs tens of
    # millions of candidates at a sharp notch root.  Within a level the
    # thinning is the same as before: keep with probability (s / h_loc)^2.
    cx, cy, cr = shape.feature_disc
    delta = decay * lf

    def reach(h_t):
        if grade is not None:
            return np.inf if h_t >= h else max(0.0, (h_t - h / refine) / grade)
        w = (h / h_t - 1.0) / max(refine - 1.0, 1e-12)
        if w >= 1.0:
            return 0.0
        if w <= 0.0:
            return np.inf
        return delta * np.sqrt(-np.log(w))

    levels, s_l = [], h
    while s_l > h_min * 1.0001:
        levels.append(s_l)
        s_l *= 0.5
    levels.append(h_min)
    interior = []
    for li, s_l in enumerate(levels):
        hi_size = np.inf if li == 0 else 2.0 * s_l
        lo_size = 0.0 if li == len(levels) - 1 else s_l
        d = reach(hi_size) if np.isfinite(hi_size) else np.inf
        if np.isfinite(d):
            bx0, bx1 = max(xmin, cx - cr - d), min(xmax, cx + cr + d)
            by0, by1 = max(ymin, cy - cr - d), min(ymax, cy + cr + d)
        else:
            bx0, bx1, by0, by1 = xmin, xmax, ymin, ymax
        for j, y in enumerate(np.arange(by0 + 0.5 * s_l, by1, s_l)):
            xs = np.arange(bx0 + (j % 2) * 0.5 * s_l, bx1, s_l)
            p = np.stack([xs, np.full_like(xs, y)], -1)
            p = p[shape.inside(p[:, 0], p[:, 1])]
            if not len(p):
                continue
            hl = size(p)
            p, hl = p[(hl >= lo_size) & (hl < hi_size)], hl[(hl >= lo_size) & (hl < hi_size)]
            if not len(p):
                continue
            p = p[rng.random(len(p)) < (s_l / hl) ** 2]
            if len(p):
                p = p + rng.normal(0.0, 0.15 * s_l, size=p.shape)
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


# --------------------------------------------------------------------------
# Phase 8.2c: the other three families
# --------------------------------------------------------------------------
@dataclass
class RigidInclusionPlate(OpenHolePlate):
    """Quarter plate with a bonded rigid inclusion of radius r at the centre.

    Same geometry as the open-hole plate; the arc is a Dirichlet boundary
    (u = v = 0 -- by symmetry the inclusion does not move), not a free edge.
    The segment is called ``inclusion`` so no code can mistake it for a hole.
    """
    name = "inclusion"
    segments = ("bottom", "right_grip", "top", "left_symmetry", "inclusion")

    def boundary(self, size):
        b = super().boundary(size)
        b["inclusion"] = b.pop("hole")
        return b

    @property
    def info(self):
        i = super().info
        i["family"] = self.name
        return i


def _notch_boundary(x0_top, y_top, t, rho, size, flank_name="flank",
                    root_name="notch"):
    """Right half of a U-notch cut down from y_top at x = 0 (quarter/half
    models keep x >= 0): straight flank x = rho from y_top down to the root
    centre, then the quarter arc of radius rho to (0, y_top - t)."""
    yc = y_top - t + rho
    h_root = float(size(np.array([[0.0, y_top - t]]))[0])
    flank = (_graded_line((rho, y_top), (rho, yc), size)
             if t - rho > 1e-9 else np.array([[rho, y_top]]))
    n_arc = max(8, int(round(0.5 * np.pi * rho / h_root)))
    th = np.linspace(0.0, -0.5 * np.pi, n_arc)
    root = np.stack([rho * np.cos(th), yc + rho * np.sin(th)], -1)
    return {flank_name: flank, root_name: root}


def _in_notch(x, y, y_top, t, rho):
    yc = y_top - t + rho
    return ((x < rho) & (y > yc)) | (x * x + (y - yc) ** 2 < rho * rho)


@dataclass
class DoubleNotchPlate(Shape):
    """Quarter of a plate of width 2H with a U-notch of depth t and root
    radius rho (rho <= t) in each edge, at mid-length.

    Segments (counter-clockwise): bottom (0,0) -> (L,0); right_grip;
    top (L,H) -> (rho,H); flank (rho,H) -> (rho, H-t+rho); notch (root arc,
    to (0, H-t)); left_symmetry (0, H-t) -> (0,0).
    """
    L: float
    H: float
    t: float
    rho: float
    name = "double_notch"
    segments = ("bottom", "right_grip", "top", "flank", "notch", "left_symmetry")

    def inside(self, x, y):
        eps = 1e-9
        box = (x >= -eps) & (x <= self.L + eps) & (y >= -eps) & (y <= self.H + eps)
        return box & ~_in_notch(x + eps, y - eps, self.H, self.t, self.rho)

    def feature_distance(self, pts):
        yc = self.H - self.t + self.rho
        return np.abs(np.hypot(pts[:, 0], pts[:, 1] - yc) - self.rho)

    @property
    def feature_length(self):
        return self.rho

    @property
    def feature_disc(self):
        return (0.0, self.H - self.t + self.rho, self.rho)

    def boundary(self, size):
        L, H, t, rho = self.L, self.H, self.t, self.rho
        b = {"bottom": _graded_line((0.0, 0.0), (L, 0.0), size),
             "right_grip": _graded_line((L, 0.0), (L, H), size),
             "top": _graded_line((L, H), (rho, H), size)}
        b.update(_notch_boundary(0.0, H, t, rho, size))
        b["left_symmetry"] = _graded_line((0.0, H - t), (0.0, 0.0), size)
        return b

    def offset_layers(self, h_min, n):
        yc = self.H - self.t + self.rho
        out = []
        for k in range(1, n + 1):
            R = self.rho + k * h_min
            m = max(4, int(round(0.5 * np.pi * R / h_min)) - k)
            th = np.linspace(0.0, -0.5 * np.pi, m)[1:-1]
            p = np.stack([R * np.cos(th), yc + R * np.sin(th)], -1)
            out.append(p[self.inside(p[:, 0], p[:, 1])])
        return out

    @property
    def info(self):
        return {"family": self.name, "L_half": self.L, "H_grip": self.H,
                "t": self.t, "rho": self.rho}


@dataclass
class SingleNotchPlate(Shape):
    """Half of a plate of width W = 2H with one U-notch (depth t, root radius
    rho) in its top edge at mid-length -- no symmetry about the length axis,
    so the model spans the full width, shifted to y in [0, 2H].

    Segments: bottom (0,0) -> (L,0), FREE; right_grip (clamped in the FEM:
    u = u_delta, v = 0); top (L,2H) -> (rho,2H); flank; notch;
    left_symmetry (0, 2H-t) -> (0,0), u = 0 and shear-free.
    """
    L: float
    H: float
    t: float
    rho: float
    name = "single_notch"
    segments = ("bottom", "right_grip", "top", "flank", "notch", "left_symmetry")

    def inside(self, x, y):
        eps = 1e-9
        top = 2.0 * self.H
        box = (x >= -eps) & (x <= self.L + eps) & (y >= -eps) & (y <= top + eps)
        return box & ~_in_notch(x + eps, y - eps, top, self.t, self.rho)

    def feature_distance(self, pts):
        yc = 2.0 * self.H - self.t + self.rho
        return np.abs(np.hypot(pts[:, 0], pts[:, 1] - yc) - self.rho)

    @property
    def feature_length(self):
        return self.rho

    @property
    def feature_disc(self):
        return (0.0, 2.0 * self.H - self.t + self.rho, self.rho)

    def boundary(self, size):
        L, top, t, rho = self.L, 2.0 * self.H, self.t, self.rho
        b = {"bottom": _graded_line((0.0, 0.0), (L, 0.0), size),
             "right_grip": _graded_line((L, 0.0), (L, top), size),
             "top": _graded_line((L, top), (rho, top), size)}
        b.update(_notch_boundary(0.0, top, t, rho, size))
        b["left_symmetry"] = _graded_line((0.0, top - t), (0.0, 0.0), size)
        return b

    def offset_layers(self, h_min, n):
        yc = 2.0 * self.H - self.t + self.rho
        out = []
        for k in range(1, n + 1):
            R = self.rho + k * h_min
            m = max(4, int(round(0.5 * np.pi * R / h_min)) - k)
            th = np.linspace(0.0, -0.5 * np.pi, m)[1:-1]
            p = np.stack([R * np.cos(th), yc + R * np.sin(th)], -1)
            out.append(p[self.inside(p[:, 0], p[:, 1])])
        return out

    @property
    def info(self):
        return {"family": self.name, "L_half": self.L, "H_grip": 2.0 * self.H,
                "t": self.t, "rho": self.rho}
