#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Equilibrium and traction in the REFERENCE configuration — PyTorch.

Uses the 1st Piola–Kirchhoff stress P throughout:
  - Div(P) = 0       interior equilibrium (w.r.t. material coords X)
  - P · N  = 0       traction-free boundary (N = reference normal)

This is frame-consistent: all derivatives are w.r.t. the undeformed
(reference) coordinates X, and P is the correct stress for that frame.

Supports both plane strain (F33=1) and plane stress (P33=0) via the
stress_state parameter passed through to neo_hookean.
"""

import torch

from .neo_hookean import first_piola_kirchhoff_stress


# Core helpers
def _predict_piola(
    model,
    query_pts: torch.Tensor,
    geometry_latent: torch.Tensor,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    mu: float,
    lam: float,
    y_max: torch.Tensor = None,
    stress_state: str = "plane strain",
):
    """Run model forward → displacement grads → 1st Piola stress P.

    Returns:
        P11, P12, P21, P22, detF: each [B, N, 1].
    """
    assert query_pts.requires_grad, "query_pts must have requires_grad=True"

    uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
        query_pts, geometry_latent, u_delta, x_max, y_max
    )
    P11, P12, P21, P22, detF = first_piola_kirchhoff_stress(
        du_dx, du_dy, dv_dx, dv_dy, mu, lam, stress_state
    )
    return P11, P12, P21, P22, detF


# Equilibrium: Div(P) = 0
def equilibrium_residual(
    model,
    query_pts: torch.Tensor,
    geometry_latent: torch.Tensor,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    mu: float,
    lam: float,
    y_max: torch.Tensor = None,
    stress_state: str = "plane strain",
) -> tuple:
    """Compute equilibrium residual Div(P) = 0 at interior points.

    Reference-frame momentum balance:
        ∂P₁₁/∂X + ∂P₁₂/∂Y = 0   (x-direction)
        ∂P₂₁/∂X + ∂P₂₂/∂Y = 0   (y-direction)

    Returns:
        f_x, f_y: [B, N, 1] equilibrium residuals (should → 0).
        detF:     [B, N, 1] determinant of F (for barrier loss).
    """
    P11, P12, P21, P22, detF = _predict_piola(
        model, query_pts, geometry_latent, u_delta, x_max, mu, lam,
        y_max, stress_state,
    )

    ones = torch.ones_like(P11)

    # P is NOT symmetric → need 4 separate grad calls
    dP11 = torch.autograd.grad(
        P11, query_pts, ones, create_graph=True, retain_graph=True
    )[0]  # [B, N, 2]
    dP12 = torch.autograd.grad(
        P12, query_pts, ones, create_graph=True, retain_graph=True
    )[0]
    dP21 = torch.autograd.grad(
        P21, query_pts, ones, create_graph=True, retain_graph=True
    )[0]
    dP22 = torch.autograd.grad(
        P22, query_pts, ones, create_graph=True, retain_graph=True
    )[0]

    # Div(P)_i = ∂P_{iJ}/∂X_J
    f_x = dP11[..., 0:1] + dP12[..., 1:2]   # ∂P11/∂X + ∂P12/∂Y
    f_y = dP21[..., 0:1] + dP22[..., 1:2]   # ∂P21/∂X + ∂P22/∂Y

    return f_x, f_y, detF


# Traction: t = P · N
def traction(
    P11: torch.Tensor,
    P12: torch.Tensor,
    P21: torch.Tensor,
    P22: torch.Tensor,
    normals: torch.Tensor,
) -> tuple:
    """Compute reference traction t = P · N.

    t_i = P_{iJ} N_J

    Args:
        P11, P12, P21, P22: [B, N, 1] 1st Piola stress components.
        normals:            [B, N, 2] reference outward unit normals N.

    Returns:
        trac_x, trac_y: [B, N, 1] traction components.
    """
    N_x = normals[..., 0:1]
    N_y = normals[..., 1:2]

    trac_x = P11 * N_x + P12 * N_y
    trac_y = P21 * N_x + P22 * N_y

    return trac_x, trac_y


# Boundary stress computation
def boundary_piola(
    model,
    bnd_pts: torch.Tensor,
    geometry_latent: torch.Tensor,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    mu: float,
    lam: float,
    y_max: torch.Tensor = None,
    stress_state: str = "plane strain",
) -> tuple:
    """Compute 1st Piola stress at boundary collocation points.

    Returns:
        P11, P12, P21, P22, detF: each [B, N, 1].
    """
    P11, P12, P21, P22, detF = _predict_piola(
        model, bnd_pts, geometry_latent, u_delta, x_max, mu, lam,
        y_max, stress_state,
    )
    return P11, P12, P21, P22, detF


# Traction residuals
def traction_residual_full(
    model,
    bnd_pts: torch.Tensor,
    normals: torch.Tensor,
    geometry_latent: torch.Tensor,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    mu: float,
    lam: float,
    y_max: torch.Tensor = None,
    stress_state: str = "plane strain",
) -> tuple:
    """Full traction residual P · N = 0 (both components).

    For traction-free boundaries: arcs, gauge top, holes.

    Returns:
        trac_x, trac_y: [B, N, 1] traction components.
        detF:            [B, N, 1] for barrier loss on boundary.
    """
    P11, P12, P21, P22, detF = boundary_piola(
        model, bnd_pts, geometry_latent, u_delta, x_max, mu, lam,
        y_max, stress_state,
    )
    trac_x, trac_y = traction(P11, P12, P21, P22, normals)
    return trac_x, trac_y, detF


def traction_residual_partial(
    model,
    bnd_pts: torch.Tensor,
    normals: torch.Tensor,
    dirs: torch.Tensor,
    geometry_latent: torch.Tensor,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    mu: float,
    lam: float,
    y_max: torch.Tensor = None,
    stress_state: str = "plane strain",
) -> torch.Tensor:
    """Partial traction residual (single component per point).

    For grip faces and symmetry line where one displacement is
    hard-enforced and only the complementary traction must vanish.

    Args:
        dirs: [B, N] or [N] int tensor. 0 → enforce trac_x, 1 → trac_y.

    Returns:
        trac_partial: [B, N, 1].
    """
    P11, P12, P21, P22, _ = boundary_piola(
        model, bnd_pts, geometry_latent, u_delta, x_max, mu, lam,
        y_max, stress_state,
    )
    trac_x, trac_y = traction(P11, P12, P21, P22, normals)

    if dirs.dim() == 1:
        dirs = dirs.unsqueeze(0).expand(trac_x.shape[0], -1)
    dirs = dirs.unsqueeze(-1)  # [B, N, 1]

    trac_partial = torch.where(dirs == 0, trac_x, trac_y)
    return trac_partial
