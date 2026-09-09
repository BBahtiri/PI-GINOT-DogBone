#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graded triangulation of the quarter dog-bone.

Two consumers, one mesher:

* the finite-element reference in ``verification/fem`` (which is where this
  code started, and which is why it is verified against the patch test, a
  closed-form uniaxial solution and a mesh-convergence study);
* the energy quadrature in ``physics/weak_form.py``, which needs
  ``integral Psi dV`` on a point-cloud-defined domain and cannot get it from
  Monte Carlo without paying variance that competes with the physics signal
  it is trying to minimise.

Sharing one mesher means the energy is integrated on the same discretisation
the reference solves on, so an integration artefact cannot masquerade as a
modelling difference.

Priorities differ from ``geometry/collocation.py``: node placement is
deterministic and structured rather than random, elements are graded towards
the fillet where the stress gradient lives, and every element must be
well-shaped because a sliver poisons the stress it carries.

Construction
------------
1. Boundary nodes on all five segments at a target spacing, refined on the
   arc and on the parts of the neighbouring flat faces near it.
2. Interior nodes from a jittered structured lattice, with local spacing
   h(x, y) that shrinks towards the fillet, plus offset layers just inside
   the arc so the boundary layer is resolved.
3. Delaunay over the union, then drop triangles whose centroid is outside
   the domain (this removes the concave-hull artefacts along the arc) and
   drop slivers.

The centroid test uses ``_point_in_dogbone`` -- the same predicate the
collocation sampler and the verification checks use -- so "inside" means the
same thing everywhere in the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay

from config import get_fillet_geometry, validate_geometry
from geometry.parametric_dogbone import _point_in_dogbone

# Segment names, matching geometry/parametric_dogbone.py.
SEGMENTS = ("bottom", "right_grip", "right_arc", "gauge_top", "left_symmetry")


@dataclass
class TriMesh:
    """A conforming triangulation of the quarter model."""
    params: dict
    fillet_info: dict
    nodes: np.ndarray                  # (n_node, 2) float64
    tris: np.ndarray                   # (n_elem, 3) int32, counter-clockwise
    areas: np.ndarray                  # (n_elem,) float64, reference areas
    grads: np.ndarray                  # (n_elem, 3, 2) P1 shape-function grads
    boundary: Dict[str, np.ndarray]    # segment name -> node indices
    h_target: float                    # nominal element size away from fillet

    @property
    def n_node(self) -> int:
        return len(self.nodes)

    @property
    def n_elem(self) -> int:
        return len(self.tris)

    def quality(self) -> dict:
        """Shape quality: 1 is equilateral, 0 is degenerate.

        q = 4*sqrt(3)*A / (l1^2 + l2^2 + l3^2), the standard radius-ratio
        surrogate.  A reference solution is only as good as its worst
        elements, so this is reported rather than assumed.
        """
        p = self.nodes[self.tris]
        e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]],
                     axis=1)
        l2 = (e ** 2).sum(-1).sum(-1)
        q = 4.0 * np.sqrt(3.0) * self.areas / np.maximum(l2, 1e-300)
        return {"min": float(q.min()), "mean": float(q.mean()),
                "p01": float(np.percentile(q, 1)),
                "n_below_0.3": int((q < 0.3).sum())}


def default_h(fi: dict, per_length: float = 38.0,
              across_gauge: float = 8.0) -> float:
    """A default element size that resolves the *thinnest* dimension.

    Sizing on the specimen length alone is not enough: the geometry ranges
    allow a 3 mm gauge half-height on a 35 mm half-length, where L_half/38
    leaves three elements across the gauge and the solve is unusable (the
    section force then varies by tens of percent along x, which is how the
    failure shows up).  Take the tighter of the two requirements.
    """
    return min(fi["L_half"] / per_length, fi["H_gauge"] / across_gauge)


def _arc_point(theta, fi):
    """Point on the fillet arc at angle theta measured from the arc centre."""
    xc, yc = fi["arc_center"]
    R = fi["R_fillet"]
    return np.stack([xc + R * np.cos(theta), yc + R * np.sin(theta)], axis=-1)


def _arc_angles(fi):
    """Angular extent of the fillet arc: (theta_start, theta_end)."""
    import math
    return (math.atan2(fi["dH"] - fi["R_fillet"], fi["dx"]), -math.pi / 2.0)


def _size_field(pts: np.ndarray, fi: dict, h: float, refine: float,
                decay: float) -> np.ndarray:
    """Target element size at each point.

    ``h`` away from the fillet, ``h/refine`` on it, blending over a distance
    ``decay * R``.  Distance is measured to the arc itself (not to a box), so
    the refinement follows the curve.
    """
    xc, yc = fi["arc_center"]
    R = fi["R_fillet"]
    # Distance to the arc = | |p - c| - R |, valid inside the arc's angular
    # wedge; outside it, distance to the nearer arc endpoint.
    d_centre = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc)
    th = np.arctan2(pts[:, 1] - yc, pts[:, 0] - xc)
    th0, th1 = _arc_angles(fi)
    lo, hi = min(th0, th1), max(th0, th1)
    inside_wedge = (th >= lo) & (th <= hi)
    d_arc = np.abs(d_centre - R)
    ends = _arc_point(np.array([th0, th1]), fi)
    d_end = np.minimum(np.hypot(pts[:, 0] - ends[0, 0], pts[:, 1] - ends[0, 1]),
                       np.hypot(pts[:, 0] - ends[1, 0], pts[:, 1] - ends[1, 1]))
    d = np.where(inside_wedge, d_arc, d_end)

    w = np.exp(-(d / (decay * R)) ** 2)          # 1 on the arc, 0 far away
    return h / (1.0 + (refine - 1.0) * w)


def _boundary_nodes(fi: dict, h: float, refine: float, decay: float):
    """Nodes on each segment, spaced by the local size field."""
    import math
    L_half, H_grip = fi["L_half"], fi["H_grip"]
    H_gauge, x_g, R = fi["H_gauge"], fi["x_g"], fi["R_fillet"]

    def graded_line(p0, p1, n_probe=4000):
        """Place nodes along a line at the local target spacing."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        t = np.linspace(0.0, 1.0, n_probe)
        pts = p0 + t[:, None] * (p1 - p0)
        length = np.linalg.norm(p1 - p0)
        hs = _size_field(pts, fi, h, refine, decay)
        # Arc-length reparametrisation: cumulative 1/h gives node positions.
        dens = np.cumsum(1.0 / hs) * (length / n_probe)
        n = max(2, int(round(dens[-1])))
        want = np.linspace(0.0, dens[-1], n)
        tt = np.interp(want, dens, t)
        return p0 + tt[:, None] * (p1 - p0)

    th0, th1 = _arc_angles(fi)
    n_arc = max(8, int(round(R * abs(th1 - th0) * refine / h)))
    arc = _arc_point(np.linspace(th0, th1, n_arc), fi)

    segs = {
        "bottom": graded_line((0.0, 0.0), (L_half, 0.0)),
        "right_grip": graded_line((L_half, 0.0), (L_half, H_grip)),
        "right_arc": arc,
        "gauge_top": graded_line((x_g, H_gauge), (0.0, H_gauge)),
        "left_symmetry": graded_line((0.0, H_gauge), (0.0, 0.0)),
    }
    return segs


def _smooth(nodes: np.ndarray, tris: np.ndarray, fixed: np.ndarray,
            fi: dict, n_sweeps: int = 12, relax: float = 0.6) -> np.ndarray:
    """Laplacian smoothing of interior nodes, with an inversion guard.

    Delaunay on a jittered lattice leaves a tail of poorly shaped triangles,
    and a sliver carries a meaningless stress.  Each sweep moves every free
    node a fraction of the way to the centroid of its neighbours, and any move
    that would leave the domain or invert an element is rejected.
    """
    n_node = len(nodes)
    # Neighbour sums via edge accumulation.
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    for _ in range(n_sweeps):
        acc = np.zeros((n_node, 2))
        cnt = np.zeros(n_node)
        np.add.at(acc, edges[:, 0], nodes[edges[:, 1]])
        np.add.at(cnt, edges[:, 0], 1.0)
        cnt = np.maximum(cnt, 1.0)
        target = acc / cnt[:, None]
        cand = nodes.copy()
        move = ~fixed
        cand[move] = (1 - relax) * nodes[move] + relax * target[move]

        # Reject moves that leave the domain.
        outside = move & ~_point_in_dogbone(cand[:, 0], cand[:, 1], fi)
        cand[outside] = nodes[outside]

        # Reject moves that invert or collapse any element.
        p = cand[tris]
        a2 = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
              - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
        bad = a2 <= 1e-12
        if bad.any():
            revert = np.unique(tris[bad].ravel())
            cand[revert] = nodes[revert]
        nodes = cand
    return nodes


def build_mesh(params: dict, h: float = 0.5, refine: float = 6.0,
               decay: float = 0.45, n_offset_layers: int = 3,
               seed: int = 0, min_quality: float = 0.05,
               n_smooth: int = 12) -> TriMesh:
    """Build a graded triangulation of the quarter model.

    Args:
        params: geometry parameters.
        h: nominal element size [mm] away from the fillet.
        refine: element-size reduction factor on the fillet.
        decay: blend width for the refinement, in units of R_fillet.
        n_offset_layers: structured layers just inside the arc, so the
            boundary layer is resolved rather than left to Delaunay.
        seed: jitter seed for the interior lattice.
        min_quality: drop triangles below this shape quality.

    Returns:
        FemMesh with counter-clockwise triangles, areas and P1 gradients.
    """
    params = {**params, "holes": params.get("holes", [])}
    if not validate_geometry(params):
        raise ValueError(f"Invalid geometry: {params}")
    fi = get_fillet_geometry(params)
    rng = np.random.default_rng(seed)

    bnd = _boundary_nodes(fi, h, refine, decay)
    boundary_pts = np.concatenate([bnd[s] for s in SEGMENTS], axis=0)

    # Offset layers just inside the fillet arc.
    th0, th1 = _arc_angles(fi)
    n_arc = len(bnd["right_arc"])
    layers = []
    h_arc = h / refine
    for k in range(1, n_offset_layers + 1):
        # Inward = towards the arc centre is *outside* the material, so step
        # away from the centre.
        R_k = fi["R_fillet"] + k * h_arc
        th = np.linspace(th0, th1, max(4, n_arc - 2 * k))
        xc, yc = fi["arc_center"]
        pts = np.stack([xc + R_k * np.cos(th), yc + R_k * np.sin(th)], -1)
        keep = _point_in_dogbone(pts[:, 0], pts[:, 1], fi)
        if keep.any():
            layers.append(pts[keep])

    # Interior lattice, rejected against the domain and thinned by the local
    # size field.  Rows are offset by half a cell so the raw lattice is closer
    # to equilateral than a square grid would be.
    h_min = h / refine
    ys = np.arange(h_min * 0.5, fi["H_grip"], h_min)
    interior = []
    for j, y in enumerate(ys):
        xs = np.arange((j % 2) * h_min * 0.5, fi["L_half"], h_min)
        if len(xs) == 0:
            continue
        p = np.stack([xs, np.full_like(xs, y)], -1)
        p = p[_point_in_dogbone(p[:, 0], p[:, 1], fi)]
        if len(p) == 0:
            continue
        # Keep a point with probability (h_min / h_local)^2 -- area density.
        h_loc = _size_field(p, fi, h, refine, decay)
        keep = rng.random(len(p)) < (h_min / h_loc) ** 2
        p = p[keep]
        if len(p):
            p = p + rng.normal(0.0, 0.15 * h_min, size=p.shape)
            p = p[_point_in_dogbone(p[:, 0], p[:, 1], fi)]
            interior.append(p)

    nodes = np.concatenate([boundary_pts] + layers + interior, axis=0)
    # Deduplicate: Delaunay is unhappy with coincident points.
    nodes = np.unique(np.round(nodes, 9), axis=0)

    n_boundary = len(boundary_pts)
    tri = Delaunay(nodes)
    simp = tri.simplices
    cent = nodes[simp].mean(axis=1)
    simp = simp[_point_in_dogbone(cent[:, 0], cent[:, 1], fi)]

    # Smooth, then re-triangulate: moving nodes can improve the connectivity
    # too, and a second Delaunay is cheap next to the solve that follows.
    if n_smooth:
        lookup0 = {tuple(np.round(n, 9)): i for i, n in enumerate(nodes)}
        fixed = np.zeros(len(nodes), dtype=bool)
        for pt in boundary_pts:
            i = lookup0.get(tuple(np.round(pt, 9)))
            if i is not None:
                fixed[i] = True
        nodes = _smooth(nodes, simp, fixed, fi, n_sweeps=n_smooth)
        tri = Delaunay(nodes)
        simp = tri.simplices
        cent = nodes[simp].mean(axis=1)
        simp = simp[_point_in_dogbone(cent[:, 0], cent[:, 1], fi)]

    # Orient counter-clockwise and compute areas.
    p = nodes[simp]
    twice_area = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                  - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    flip = twice_area < 0
    simp[flip] = simp[flip][:, [0, 2, 1]]
    twice_area = np.abs(twice_area)
    areas = 0.5 * twice_area

    # Drop slivers before computing gradients.
    p = nodes[simp]
    e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], 1)
    q = 4.0 * np.sqrt(3.0) * areas / np.maximum((e ** 2).sum(-1).sum(-1), 1e-300)
    good = q >= min_quality
    simp, areas, twice_area = simp[good], areas[good], twice_area[good]

    # P1 shape-function gradients: grad(N_a) = [y_b - y_c, x_c - x_b] / (2A).
    p = nodes[simp]
    grads = np.empty((len(simp), 3, 2))
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        grads[:, a, 0] = (p[:, b, 1] - p[:, c, 1]) / twice_area
        grads[:, a, 1] = (p[:, c, 0] - p[:, b, 0]) / twice_area

    # Boundary node indices, by matching against the segment node lists.
    lookup = {tuple(np.round(n, 9)): i for i, n in enumerate(nodes)}
    boundary = {}
    for s in SEGMENTS:
        idx = [lookup.get(tuple(np.round(pt, 9))) for pt in bnd[s]]
        boundary[s] = np.array([i for i in idx if i is not None], dtype=np.int64)

    # Drop any node no element uses (possible after sliver removal).
    used = np.zeros(len(nodes), dtype=bool)
    used[simp.ravel()] = True
    if not used.all():
        remap = -np.ones(len(nodes), dtype=np.int64)
        remap[used] = np.arange(used.sum())
        nodes = nodes[used]
        simp = remap[simp]
        boundary = {k: remap[v][remap[v] >= 0] for k, v in boundary.items()}

    return TriMesh(params=params, fillet_info=fi, nodes=nodes,
                   tris=simp.astype(np.int32), areas=areas, grads=grads,
                   boundary=boundary, h_target=h)


def summarize(mesh: TriMesh) -> str:
    q = mesh.quality()
    return (f"nodes {mesh.n_node:7d}  elems {mesh.n_elem:7d}  "
            f"h={mesh.h_target:.3f}  area={mesh.areas.sum():.4f}  "
            f"quality min={q['min']:.3f} p01={q['p01']:.3f} "
            f"mean={q['mean']:.3f}")


# ------------------------------------------------------- quadrature rules --

# Barycentric coordinates and weights for triangle quadrature.  Weights sum to
# 1 and are multiplied by the element area.
QUADRATURE_RULES = {
    # Degree 1: centroid.  Exact for a linear integrand.
    1: (np.array([[1 / 3, 1 / 3, 1 / 3]]), np.array([1.0])),
    # Degree 2: the standard symmetric 3-point interior rule.
    2: (np.array([[2 / 3, 1 / 6, 1 / 6],
                  [1 / 6, 2 / 3, 1 / 6],
                  [1 / 6, 1 / 6, 2 / 3]]), np.full(3, 1 / 3)),
    # Degree 3: 4-point (centroid plus three), the classic Strang rule.
    3: (np.array([[1 / 3, 1 / 3, 1 / 3],
                  [0.6, 0.2, 0.2],
                  [0.2, 0.6, 0.2],
                  [0.2, 0.2, 0.6]]),
        np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])),
}


def quadrature(mesh: TriMesh, degree: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrature points and weights for integrating over the domain.

    Args:
        mesh: the triangulation.
        degree: polynomial degree the rule integrates exactly (1, 2 or 3).

    Returns:
        pts: (n_elem * n_q, 2) physical coordinates.
        w:   (n_elem * n_q,) weights that sum to the domain area.
    """
    if degree not in QUADRATURE_RULES:
        raise ValueError(f"no rule of degree {degree}; have "
                         f"{sorted(QUADRATURE_RULES)}")
    bary, wq = QUADRATURE_RULES[degree]
    verts = mesh.nodes[mesh.tris]                      # (M, 3, 2)
    pts = np.einsum("qa,mad->mqd", bary, verts)        # (M, n_q, 2)
    w = mesh.areas[:, None] * wq[None, :]              # (M, n_q)
    return pts.reshape(-1, 2), w.reshape(-1)


# ------------------------------------------------------------------ cache --

_MESH_CACHE: Dict[tuple, TriMesh] = {}


def _key(params: dict, h: float, refine: float, decay: float,
         n_smooth: int) -> tuple:
    return (round(params["L_total"], 9), round(params["W_grip"], 9),
            round(params["W_gauge"], 9), round(params["R_fillet"], 9),
            round(float(h), 9), round(float(refine), 6),
            round(float(decay), 6), int(n_smooth))


def cached_mesh(params: dict, h: Optional[float] = None, refine: float = 6.0,
                decay: float = 0.45, n_smooth: int = 12) -> TriMesh:
    """Triangulate once per geometry and reuse.

    Meshing costs a Delaunay, a smoothing sweep and a second Delaunay -- a
    fraction of a second, but it is per geometry per epoch if not cached, and
    the geometry bank is fixed. ``h`` defaults to ``default_h``.
    """
    from config import get_fillet_geometry
    fi = get_fillet_geometry(params)
    if h is None:
        h = default_h(fi)
    k = _key(params, h, refine, decay, n_smooth)
    hit = _MESH_CACHE.get(k)
    if hit is None:
        hit = build_mesh(params, h=h, refine=refine, decay=decay,
                         n_smooth=n_smooth)
        _MESH_CACHE[k] = hit
    return hit


_H_CACHE: Dict[tuple, float] = {}


def h_for_budget(params: dict, target_elems: int, tol: float = 0.25,
                 max_tries: int = 6) -> float:
    """Element size giving roughly ``target_elems`` triangles.

    The energy integrand is far smoother than the stress field, so the
    quadrature mesh does not need the reference mesh's resolution -- and at
    the reference resolution it does not fit in memory, since every quadrature
    point is a forward pass through the decoder.  Sizing to a point budget
    keeps the cost comparable to the collocation set it replaces, and
    ``EnergyLoss.integration_error`` reports what the choice costs.

    Grading makes the count-versus-h relation only approximately h^-2, so this
    bisects rather than solving in closed form.  Meshes are cached, so the few
    extra builds are paid once per geometry.
    """
    from config import get_fillet_geometry
    ck = (round(params["L_total"], 9), round(params["W_grip"], 9),
          round(params["W_gauge"], 9), round(params["R_fillet"], 9),
          int(target_elems))
    if ck in _H_CACHE:
        return _H_CACHE[ck]
    fi = get_fillet_geometry(params)
    area = fi["x_g"] * fi["H_gauge"] + (fi["L_half"] - fi["x_g"]) * fi["H_grip"]
    h = float(np.sqrt(2.0 * area / max(target_elems, 1)))
    lo, hi = h / 8.0, h * 8.0
    for _ in range(max_tries):
        n = cached_mesh(params, h=h).n_elem
        if abs(n - target_elems) <= tol * target_elems:
            break
        if n > target_elems:
            lo = h
        else:
            hi = h
        h = float(np.sqrt(lo * hi)) if n > target_elems else float(
            np.sqrt(lo * hi))
        h = float(np.sqrt(max(lo, 1e-6) * hi))
    _H_CACHE[ck] = h
    return h


def clear_cache() -> None:
    _MESH_CACHE.clear()
    _H_CACHE.clear()
