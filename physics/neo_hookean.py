#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo-Hookean hyperelastic constitutive law — PyTorch port.

All functions operate on batched displacement-gradient tensors [B, N, 1]
and return stress components in the same shape.

Neo-Hookean strain energy (compressible, 3D):
    W = (μ/2)(tr(F^T F) - 3) - μ ln(J) + (λ/2)(ln J)²
    where J = det(F), F = I + ∇u

First Piola–Kirchhoff stress (reference configuration):
    P = μ F + (λ ln J − μ) F^{-T}

Cauchy stress (current configuration, for post-processing only):
    σ = (1/J) P F^T

Equilibrium and traction BCs use P in the reference frame:
    Div(P) = 0          (interior equilibrium)
    P · N  = 0          (traction-free boundaries, N = reference normal)

Plane stress support:
    For thin specimens, F33 is solved from P33 = 0 using Newton iteration
    (5 steps, differentiable through PyTorch autograd).  The 3D determinant
    J = J2D * F33 replaces J2D in the log term.
"""

import torch

# Minimum detF to prevent log(0) or log(negative) during early training.
_DETF_MIN = 1e-6


def deformation_gradient(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
) -> tuple:
    """Build the 2D deformation gradient F = I + ∇u.

    Args:
        du_dx, du_dy, dv_dx, dv_dy: [B, N, 1] displacement gradients.

    Returns:
        F11, F12, F21, F22, detF: each [B, N, 1].
    """
    F11 = du_dx + 1.0
    F12 = du_dy
    F21 = dv_dx
    F22 = dv_dy + 1.0
    detF = F11 * F22 - F12 * F21
    return F11, F12, F21, F22, detF


def _solve_F33_plane_stress(J2D: torch.Tensor, mu: float, lam: float,
                            n_iter: int = 5) -> torch.Tensor:
    """Solve P33 = 0 for F33 via Newton iteration (differentiable).

    For Neo-Hookean:
        P33 = μ F33 + (λ ln(J2D·F33) − μ) / F33
    Setting P33 = 0 and solving for F33.

    Newton update:
        g   = μ F33 + (λ ln(J2D·F33) − μ) / F33
        g'  = μ + (λ − λ ln(J2D·F33) + μ) / F33²
        F33 ← F33 − g / g'

    Args:
        J2D:    [B, N, 1] in-plane determinant (may be < 0 at init).
        mu:     Shear modulus.
        lam:    Lamé second parameter.
        n_iter: Number of Newton iterations (default 5).

    Returns:
        F33: [B, N, 1] out-of-plane stretch satisfying P33 ≈ 0.
    """
    J2D_safe = torch.clamp(J2D, min=_DETF_MIN)
    F33 = torch.ones_like(J2D_safe)

    for _ in range(n_iter):
        J3D = J2D_safe * F33
        J3D = torch.clamp(J3D, min=_DETF_MIN)
        logJ = torch.log(J3D)

        # P33 = μ F33 + (λ ln J − μ) / F33
        g = mu * F33 + (lam * logJ - mu) / F33
        # dP33/dF33 = μ + (λ − λ ln J + μ) / F33²
        dg = mu + (lam - lam * logJ + mu) / (F33 ** 2 + 1e-12)
        F33 = F33 - g / (dg + 1e-12)
        # Clamp to keep F33 physical
        F33 = torch.clamp(F33, min=0.01)

    return F33


def first_piola_kirchhoff_stress(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str = "plane strain",
) -> tuple:
    """Compute 1st Piola–Kirchhoff stress P for Neo-Hookean material.

    P = μ F + (λ ln J − μ) F^{-T}

    For plane strain:  F33 = 1, J = J2D.
    For plane stress:  F33 solved from P33 = 0, J = J2D · F33.

    In both cases, the in-plane F^{-T} components are computed from
    the 2D submatrix (divided by J2D, not J3D).

    Args:
        du_dx, du_dy, dv_dx, dv_dy: [B, N, 1] displacement gradients.
        mu:  Shear modulus (Lamé first parameter).
        lam: Lamé second parameter.
        stress_state: 'plane strain' or 'plane stress'.

    Returns:
        P11, P12, P21, P22, detF: each [B, N, 1].
        detF is the in-plane determinant J2D (for barrier loss).
    """
    F11, F12, F21, F22, detF = deformation_gradient(du_dx, du_dy, dv_dx, dv_dy)

    J2D = detF
    J2D_safe = torch.clamp(J2D, min=_DETF_MIN)

    # In-plane F^{-T} (always uses J2D, not J3D):
    invFT11 =  F22 / J2D_safe
    invFT12 = -F21 / J2D_safe
    invFT21 = -F12 / J2D_safe
    invFT22 =  F11 / J2D_safe

    if stress_state == "plane stress":
        F33 = _solve_F33_plane_stress(J2D, mu, lam)
        J3D = J2D_safe * F33
        J3D = torch.clamp(J3D, min=_DETF_MIN)
        logJ = torch.log(J3D)
    else:
        # Plane strain: F33 = 1, J = J2D
        logJ = torch.log(J2D_safe)

    coeff = lam * logJ - mu          # (λ ln J − μ)

    P11 = mu * F11 + coeff * invFT11
    P12 = mu * F12 + coeff * invFT12
    P21 = mu * F21 + coeff * invFT21
    P22 = mu * F22 + coeff * invFT22

    return P11, P12, P21, P22, detF


def cauchy_stress(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str = "plane strain",
) -> tuple:
    """Compute Cauchy stress σ = (1/J) P Fᵀ (for post-processing only).

    NOT used for PDE residuals — use first_piola_kirchhoff_stress() instead.

    For plane stress, J3D = J2D · F33 is used for the 1/J factor.

    Args:
        du_dx, du_dy, dv_dx, dv_dy: [B, N, 1] displacement gradients.
        mu, lam: Lamé parameters.
        stress_state: 'plane strain' or 'plane stress'.

    Returns:
        S11, S22, S12, detF: each [B, N, 1].
    """
    F11, F12, F21, F22, detF = deformation_gradient(du_dx, du_dy, dv_dx, dv_dy)
    P11, P12, P21, P22, _ = first_piola_kirchhoff_stress(
        du_dx, du_dy, dv_dx, dv_dy, mu, lam, stress_state
    )

    J2D = detF
    J2D_safe = torch.clamp(J2D, min=_DETF_MIN)

    if stress_state == "plane stress":
        F33 = _solve_F33_plane_stress(J2D, mu, lam)
        J3D = J2D_safe * F33
        inv_J = 1.0 / torch.clamp(J3D, min=_DETF_MIN)
    else:
        inv_J = 1.0 / J2D_safe

    # σ = (1/J) P Fᵀ
    S11 = inv_J * (P11 * F11 + P12 * F12)
    S22 = inv_J * (P21 * F21 + P22 * F22)
    S12 = inv_J * (P11 * F21 + P12 * F22)

    return S11, S22, S12, detF


def full_stress_state(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str = "plane strain",
) -> tuple:
    """Compute full Cauchy stress including out-of-plane σ₃₃ (post-processing).

    For plane strain Neo-Hookean:
        F33 = 1  →  det(F_3D) = det(F_2D) = J
        σ₃₃ = λ ln(J) / J

    For plane stress:
        σ₃₃ = 0  (by definition; F33 solved from P33=0)

    Returns:
        S11, S22, S33, S12, detF: each [B, N, 1].
    """
    S11, S22, S12, detF = cauchy_stress(
        du_dx, du_dy, dv_dx, dv_dy, mu, lam, stress_state
    )
    detF_safe = torch.clamp(detF, min=_DETF_MIN)

    if stress_state == "plane strain":
        S33 = lam * torch.log(detF_safe) / detF_safe
    else:
        S33 = torch.zeros_like(S11)

    return S11, S22, S33, S12, detF


def von_mises_stress(
    S11: torch.Tensor,
    S22: torch.Tensor,
    S33: torch.Tensor,
    S12: torch.Tensor,
) -> torch.Tensor:
    """Compute von Mises equivalent stress (post-processing).

    σ_vm = √( σ₁₁² + σ₂₂² + σ₃₃² − σ₁₁σ₂₂ − σ₂₂σ₃₃ − σ₁₁σ₃₃ + 3σ₁₂² )
    """
    return torch.sqrt(
        S11**2 + S22**2 + S33**2
        - S11 * S22 - S22 * S33 - S11 * S33
        + 3.0 * S12**2
        + 1e-12
    )
