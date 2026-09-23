#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Approximate distance functions for exact Dirichlet conditions (Phase 8.1).

The construction is Sukumar & Srivastava (2022, CMAME 389:114333), in the form
Berrone, Canuto, Pintore & Sukumar (2023, Heliyon 9:e18820) found most
accurate: every Dirichlet piece gets a distance function *normalised to first
order* (phi = 0 on the piece, d phi / dn = 1 there), the pieces are joined by
R-equivalence, and piecewise data are lifted by transfinite interpolation
with the same functions as weights.  Per displacement component c,

    w_c(X) = g_c(X) + phi_c(X) / l_c * N_c(X)

    phi_c = R(phi_1, ..., phi_k),     R = (sum_i phi_i^-1)^-1          (m = 1)
    g_c   = sum_i W_i g_i,            W_i = phi_i^-1 / sum_j phi_j^-1

where N_c is the network output and l_c a characteristic length that keeps
the multiplier dimensionless.  Both R and W are evaluated in product form, so
nothing is divided by a vanishing phi:

    R   = prod_i phi_i / sum_j prod_{i != j} phi_i
    W_i = prod_{k != i} phi_k / sum_j prod_{k != j} phi_k

**The dog-bone ansatz is the special case.**  With pieces {x = 0 : 0} and
{x = L : u_delta} for u and {y = 0 : 0} for v:

    R(x, L - x) = x (L - x) / L,      W_grip = x / L

so w_u = u_delta x/L + x(L - x)/L^2 N_u = u_delta xi + xi(1 - xi) N_u and
w_v = (y/H) N_v -- ``PhysicsDecoder._apply_hard_bc``, identical in exact
arithmetic (in floating point the two agree to rounding: 9e-17 relative in
float64, 2.5e-8 in float32, not bit for bit).  For two *parallel* pieces the
normalised form and the plain product x(L - x) differ only by the constant L,
so for the dog-bone the distinction is moot.

What the general form adds is what the next geometries need: trimmed segments
(a Dirichlet condition on part of an edge, e.g. a clamped grip region),
circles and arcs (a bonded rigid inclusion), and a defined behaviour where
pieces meet.  There the two forms genuinely differ.  R-equivalence keeps
d phi / dn = 1 along each piece up to the corner, at the price of a Laplacian
that is unbounded at the vertex (Berrone et al. 2023, App. A.1) -- harmless to
the energy form, which uses first derivatives only.  A plain product's normal
derivative falls linearly to zero at the corner.  Which is better depends on
the solution: where two zero-data pieces meet at 90 degrees the exact
gradient itself vanishes at the corner, so the product's decay is the right
shape and R-equivalence asks the network to learn N -> 0 there.  Neither is
assumed better here; Phase 8 measures it (ablation A-ADF).

Numerics.  ``segment`` is not differentiable exactly on its segment (it is
|f|-like there) and its square roots have infinite slope at zero; points
exactly on the segment get the one-sided, domain-side gradient ``n_in`` when
it is supplied, and zero otherwise -- never NaN.  Division guards use the
dtype's own ``tiny``, so float32 is safe.

Primitives return non-negative values in the domain for ``segment`` and
``circle(outside=True)``; ``line`` is signed, positive on the domain side.
All functions take X of shape [..., 2] and are differentiable (torch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Union

import torch

def _tiny(t: torch.Tensor) -> torch.Tensor:
    return torch.full_like(t, torch.finfo(t.dtype).tiny)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def line(X: torch.Tensor, p0: Sequence[float], n: Sequence[float]) -> torch.Tensor:
    """Signed distance to the line through ``p0`` with unit normal ``n``.

    ``n`` points into the domain, so the value is positive there.  Exact
    distance, normalised to every order.
    """
    p0 = torch.as_tensor(p0, dtype=X.dtype, device=X.device)
    n = torch.as_tensor(n, dtype=X.dtype, device=X.device)
    n = n / torch.linalg.norm(n)
    return ((X - p0) * n).sum(-1)


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt with a zero (not NaN) gradient at x = 0."""
    pos = x > 0
    return torch.where(pos, torch.sqrt(torch.where(pos, x, torch.ones_like(x))),
                       torch.zeros_like(x))


def segment(X: torch.Tensor, p0: Sequence[float], p1: Sequence[float],
            n_in: Sequence[float] = None) -> torch.Tensor:
    """Trimmed ADF of the segment p0 -> p1 (Sukumar & Srivastava 2022, sec. 3;
    eqs. 4-6 of arXiv:2104.08426).

    f = signed distance to the segment's line, t = trimming function (positive
    inside the circle that has the segment as diameter),
    phi = sqrt(f^2 + ((sqrt(t^2 + f^4) - t) / 2)^2).  Zero only on the
    segment itself, first-order normalised in its interior.

    ``n_in`` (unit normal pointing into the domain) sets the gradient at
    points exactly on the segment, where phi has a kink: the one-sided limit
    from the domain side is n_in.  Without it that gradient is 0.
    """
    p0 = torch.as_tensor(p0, dtype=X.dtype, device=X.device)
    p1 = torch.as_tensor(p1, dtype=X.dtype, device=X.device)
    d = p1 - p0
    L = torch.linalg.norm(d)
    xc = 0.5 * (p0 + p1)
    r = X - p0
    f = (r[..., 0] * d[1] - r[..., 1] * d[0]) / L
    t = ((0.5 * L) ** 2 - ((X - xc) ** 2).sum(-1)) / L
    varphi = _safe_sqrt(t * t + f ** 4)
    q = f * f + (0.5 * (varphi - t)) ** 2
    phi = _safe_sqrt(q)
    if n_in is not None:
        n = torch.as_tensor(n_in, dtype=X.dtype, device=X.device)
        n = n / torch.linalg.norm(n)
        # adds exactly 0 to the value; supplies grad = n_in where phi = 0
        phi = phi + torch.where(q > 0, torch.zeros_like(q),
                                ((X - X.detach()) * n).sum(-1))
    return phi


def circle(X: torch.Tensor, c: Sequence[float], r: float,
           outside: bool = True) -> torch.Tensor:
    """ADF of a circle: (|X - c|^2 - r^2) / (2 r), positive outside.

    Normalised to first order (|grad| = |X - c| / r = 1 on the circle).  Set
    ``outside=False`` when the domain is the disc.
    """
    c = torch.as_tensor(c, dtype=X.dtype, device=X.device)
    v = (((X - c) ** 2).sum(-1) - r * r) / (2.0 * r)
    return v if outside else -v


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------
def _leave_one_out_products(phis: List[torch.Tensor]) -> List[torch.Tensor]:
    """prod_{k != i} phi_k for every i, without division."""
    k = len(phis)
    if k == 1:
        return [torch.ones_like(phis[0])]
    pre = [torch.ones_like(phis[0])]
    for p in phis[:-1]:
        pre.append(pre[-1] * p)
    suf = [torch.ones_like(phis[0])]
    for p in reversed(phis[1:]):
        suf.append(suf[-1] * p)
    suf = suf[::-1]
    return [pre[i] * suf[i] for i in range(k)]


def r_equivalence(phis: List[torch.Tensor]) -> torch.Tensor:
    """(sum_i phi_i^-1)^-1 in product form: zero on every piece, normalised."""
    if len(phis) == 1:
        return phis[0]
    loo = _leave_one_out_products(phis)
    num = loo[0] * phis[0]                         # prod of all
    den = sum(loo)
    return num / torch.where(den == 0, _tiny(den), den)


def naive_product(phis: List[torch.Tensor]) -> torch.Tensor:
    """prod_i phi_i -- the un-normalised alternative, kept for comparison."""
    out = phis[0]
    for p in phis[1:]:
        out = out * p
    return out


def tfi_weights(phis: List[torch.Tensor]) -> List[torch.Tensor]:
    """Transfinite-interpolation weights, m = 1: W_i = 1 on piece i, 0 on the others.

    At a point where several pieces meet every product-form weight is 0/0.
    There the weights are shared equally among the pieces that vanish, so the
    lift returns their common value -- correct whenever the data agree at the
    corner (they must, for the Dirichlet problem to be well posed).  An
    earlier version returned 0 there, which was right only for zero data.
    """
    if len(phis) == 1:
        return [torch.ones_like(phis[0])]
    loo = _leave_one_out_products(phis)
    den = sum(loo)
    corner = den == 0
    safe = torch.where(corner, torch.ones_like(den), den)
    zero = [(p == 0).to(p.dtype) for p in phis]
    nz = torch.clamp(sum(zero), min=1.0)
    return [torch.where(corner, z / nz, l / safe) for l, z in zip(loo, zero)]


# --------------------------------------------------------------------------
# Per-component Dirichlet specification
# --------------------------------------------------------------------------
Value = Union[float, Callable[[torch.Tensor], torch.Tensor]]


@dataclass
class Piece:
    """One Dirichlet piece: its ADF and the prescribed value on it."""
    adf: Callable[[torch.Tensor], torch.Tensor]
    value: Value = 0.0
    name: str = ""

    def g(self, X: torch.Tensor) -> torch.Tensor:
        if callable(self.value):
            return self.value(X)
        return torch.full(X.shape[:-1], float(self.value), dtype=X.dtype,
                          device=X.device)


@dataclass
class DirichletSpec:
    """Dirichlet pieces per component (0 = u, 1 = v) and multiplier lengths."""
    pieces: Dict[int, List[Piece]] = field(default_factory=dict)
    length: Dict[int, float] = field(default_factory=dict)

    def phi(self, X: torch.Tensor, c: int, combine=r_equivalence) -> torch.Tensor:
        ps = self.pieces.get(c, [])
        if not ps:
            return torch.ones(X.shape[:-1], dtype=X.dtype, device=X.device)
        return combine([p.adf(X) for p in ps])

    def lift(self, X: torch.Tensor, c: int) -> torch.Tensor:
        ps = self.pieces.get(c, [])
        if not ps:
            return torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        W = tfi_weights([p.adf(X) for p in ps])
        return sum(w * p.g(X) for w, p in zip(W, ps))

    def apply(self, X: torch.Tensor, N: torch.Tensor,
              combine=r_equivalence) -> torch.Tensor:
        """Displacement [..., 2] from network outputs N [..., 2]."""
        out = []
        for c in (0, 1):
            ell = self.length.get(c, 1.0)
            out.append(self.lift(X, c) + self.phi(X, c, combine) / ell * N[..., c])
        return torch.stack(out, -1)


# --------------------------------------------------------------------------
# The specifications the plan needs
# --------------------------------------------------------------------------
def quarter_tension(L: float, H: float, u_delta: float,
                    arc: tuple = None) -> DirichletSpec:
    """Quarter model in x-tension: u = 0 on x = 0, u = u_delta on x = L,
    v = 0 on y = 0 -- the dog-bone, open-hole and double-notch families.

    ``arc = (c, r)`` adds a bonded rigid inclusion: u = v = 0 on the circle.
    ``H`` is the multiplier length for v (H_grip for the dog-bone).
    """
    pu = [Piece(lambda X: line(X, (0.0, 0.0), (1.0, 0.0)), 0.0, "x=0"),
          Piece(lambda X: line(X, (L, 0.0), (-1.0, 0.0)), u_delta, "x=L")]
    pv = [Piece(lambda X: line(X, (0.0, 0.0), (0.0, 1.0)), 0.0, "y=0")]
    if arc is not None:
        c, r = arc
        pu.insert(1, Piece(lambda X: circle(X, c, r), 0.0, "inclusion"))
        pv.append(Piece(lambda X: circle(X, c, r), 0.0, "inclusion"))
    return DirichletSpec({0: pu, 1: pv}, {0: L, 1: H})


def half_clamped_tension(L: float, H: float, u_delta: float) -> DirichletSpec:
    """Half model without the y-symmetry (single-edge notch): u = 0 on the
    symmetry plane x = 0; the grip x = L is clamped, u = u_delta and v = 0,
    which also removes the rigid translation in y."""
    pu = [Piece(lambda X: line(X, (0.0, 0.0), (1.0, 0.0)), 0.0, "x=0"),
          Piece(lambda X: line(X, (L, 0.0), (-1.0, 0.0)), u_delta, "x=L")]
    pv = [Piece(lambda X: line(X, (L, 0.0), (-1.0, 0.0)), 0.0, "x=L")]
    return DirichletSpec({0: pu, 1: pv}, {0: L, 1: L})


def clamped_grip_region(L: float, H: float, x_s: float, u_delta: float) -> DirichletSpec:
    """Phase 7 B2's grip: symmetry planes as before, and u = u_delta, v = 0 on
    the top face over x_s <= x <= L only -- a trimmed segment, because the
    rest of that face is free."""
    grip = (lambda X: segment(X, (x_s, H), (L, H), n_in=(0.0, -1.0)))
    pu = [Piece(lambda X: line(X, (0.0, 0.0), (1.0, 0.0)), 0.0, "x=0"),
          Piece(grip, u_delta, "grip region")]
    pv = [Piece(lambda X: line(X, (0.0, 0.0), (0.0, 1.0)), 0.0, "y=0"),
          Piece(grip, 0.0, "grip region")]
    return DirichletSpec({0: pu, 1: pv}, {0: L, 1: H})
