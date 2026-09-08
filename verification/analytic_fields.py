#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closed-form displacement fields wrapped in the model interface.

Why this exists
---------------
Nothing in the repository checks the physics *operators* against a known
answer.  The training loss can only say that a network fits the points it was
handed; it cannot say whether ``equilibrium_residual`` computes Div P, whether
``traction`` pairs the right stress and normal indices, or whether the fillet
normals point outward.  A sign error in any of those is invisible to training
-- the network simply converges to the wrong field and every downstream number
is quietly wrong.

``AnalyticModel`` presents a closed-form field through the same interface
``PI_GINOT`` exposes, so the *actual production code path* can be driven with
a field whose exact answer is known.  Nothing is reimplemented: the residual,
traction and resultant code under test is the code that trains.

    field = HomogeneousUniaxial(l1=1.05, mu=MU, lam=LAM)
    f_x, f_y, _ = equilibrium_residual(AnalyticModel(field), pts, None,
                                       u_d, x_m, MU, LAM, y_m, "plane stress")
    # exact answer: 0

Fields are evaluated in float64 by default -- these are verification
tolerances, not training tolerances, and float32 would bottom out at ~1e-6
long before the operators themselves do.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


class AnalyticModel:
    """Adapter presenting a closed-form displacement field as a PI_GINOT.

    Only the three methods the physics code actually calls are provided:
    ``encode`` (returns a dummy latent), ``decode`` and
    ``predict_with_grad_latent``.  Gradients come from autograd on the
    closed-form expression, so the differentiation path exercised here is the
    same one the trained model goes through.

    Args:
        field: callable ``(X, Y) -> (u, v)`` on tensors of shape [B, N, 1].
        dtype: working precision; float64 by default.
    """

    def __init__(self, field: Callable, dtype: torch.dtype = torch.float64):
        self.field = field
        self.dtype = dtype
        self.training = False

    # -- interface the physics code uses ---------------------------------
    def encode(self, boundary_pc, x_max, y_max=None, sample_ids=None):
        """No geometry conditioning: the field is prescribed, not inferred."""
        return None

    def decode(self, query_pts, geometry_latent=None, u_delta=None,
               x_max=None, y_max=None):
        X = query_pts[..., 0:1]
        Y = query_pts[..., 1:2]
        u, v = self.field(X, Y)
        return torch.cat([u, v], dim=-1)

    def predict_with_grad_latent(self, query_pts, geometry_latent=None,
                                 u_delta=None, x_max=None, y_max=None):
        assert query_pts.requires_grad, (
            "query_pts must have requires_grad=True for AD."
        )
        uv = self.decode(query_pts)
        u = uv[..., 0:1]
        v = uv[..., 1:2]
        ones = torch.ones_like(u)
        du = torch.autograd.grad(u, query_pts, ones, create_graph=True,
                                 retain_graph=True)[0]
        dv = torch.autograd.grad(v, query_pts, ones, create_graph=True,
                                 retain_graph=True)[0]
        return uv, du[..., 0:1], du[..., 1:2], dv[..., 0:1], dv[..., 1:2]

    # -- convenience ------------------------------------------------------
    def query(self, pts: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Turn an (N, 2) numpy array into a differentiable query tensor."""
        return torch.tensor(np.asarray(pts), dtype=self.dtype,
                            device=device).unsqueeze(0).requires_grad_(True)

    def eval_mode_args(self, x_max: float, y_max: float, u_delta: float = 1.0,
                       device: str = "cpu") -> tuple:
        """The (u_delta, x_max, y_max) tensors the physics signatures expect."""
        t = lambda v: torch.tensor([v], dtype=self.dtype, device=device)
        return t(u_delta), t(x_max), t(y_max)


# ---------------------------------------------------------------- fields --

class AffineField:
    """u = (F - I) X, i.e. a spatially constant deformation gradient.

    Any hyperelastic stress is then constant in space, so Div P must vanish
    identically -- the finite-strain patch test in its purest form.
    """

    def __init__(self, F: np.ndarray):
        F = np.asarray(F, dtype=float)
        assert F.shape == (2, 2)
        self.F = F
        self.H = F - np.eye(2)          # displacement gradient

    def __call__(self, X, Y):
        h = self.H
        u = h[0, 0] * X + h[0, 1] * Y
        v = h[1, 0] * X + h[1, 1] * Y
        return u, v

    def __repr__(self):
        return f"AffineField(F={self.F.tolist()})"


class RigidTranslation:
    """u = const.  F = I everywhere, so P must be exactly zero."""

    def __init__(self, ux: float = 0.3, uy: float = -0.7):
        self.ux, self.uy = ux, uy

    def __call__(self, X, Y):
        return (torch.full_like(X, self.ux), torch.full_like(Y, self.uy))

    def __repr__(self):
        return f"RigidTranslation(ux={self.ux}, uy={self.uy})"


class RigidRotation:
    """A finite rigid rotation.  F = Q, so J = 1 and the stress must vanish.

    Distinct from translation: the displacement gradient is *not* zero, so
    this catches a constitutive law that is only accidentally correct at
    F = I, and it is the classic objectivity check.
    """

    def __init__(self, theta: float = 0.35):
        self.theta = theta
        c, s = np.cos(theta), np.sin(theta)
        self.Q = np.array([[c, -s], [s, c]])

    def __call__(self, X, Y):
        c, s = np.cos(self.theta), np.sin(self.theta)
        return (c * X - s * Y - X, s * X + c * Y - Y)

    def __repr__(self):
        return f"RigidRotation(theta={self.theta})"


class HomogeneousUniaxial(AffineField):
    """Laterally-free uniaxial extension at axial stretch ``l1``.

    The lateral stretch is the plane-stress value, so on a rectangle every
    face except the loaded ones is traction-free and the exact resultant is
    known in closed form.
    """

    def __init__(self, l1: float, mu: float, lam: float):
        from physics.uniaxial import uniaxial_lateral_stretch
        beta = float(uniaxial_lateral_stretch(l1, mu, lam))
        super().__init__(np.diag([l1, beta]))
        self.l1, self.beta = l1, beta
        self.mu, self.lam = mu, lam

    @property
    def P11(self) -> float:
        from physics.uniaxial import uniaxial_P11
        return float(uniaxial_P11(self.l1, self.mu, self.lam))

    def __repr__(self):
        return (f"HomogeneousUniaxial(l1={self.l1:.6f}, "
                f"beta={self.beta:.6f}, P11={self.P11:.6f})")


class CallableField:
    """Wrap an arbitrary ``(X, Y) -> (u, v)`` torch expression."""

    def __init__(self, fn: Callable, name: str = "callable"):
        self.fn, self.name = fn, name

    def __call__(self, X, Y):
        return self.fn(X, Y)

    def __repr__(self):
        return f"CallableField({self.name})"
