#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closed-form uniaxial references for the compressible Neo-Hookean law.

Used to build an *absolute* target for the axial force resultant
N(x) = 2 * integral of P11 over the reference half-section, which the
section-resultant loss in ``training/trainer.py`` anchors against.

Laterally-free uniaxial plane stress
------------------------------------
For F = diag(l1, beta, beta) with P22 = P33 = 0 the transverse equation is

    mu*beta^2 + lam*ln(l1*beta^2) - mu = 0

Substituting t = beta^2 makes it a scalar Lambert-type equation

    mu*t + lam*ln(t) = mu - lam*ln(l1)

solved here by damped Newton (monotone in t, so this is unconditionally
convergent from t = 1).  The axial nominal stress is then

    P11 = mu*l1 + (lam*ln(J) - mu)/l1,     J = l1*t

Why the small-strain anchor is wrong -- and in which direction
-------------------------------------------------------------
The previous target was ``N = E * (u_delta/L_half) * 2*H_gauge``.  It carries
two independent errors that pull in *opposite* directions:

1. Linearisation.  At a fixed stretch, E*eps overstates the true nominal
   stress by ~2.0-3.5% over this operating range (eps ~ 0.029-0.050).

2. Constant section.  Far larger, and the other way.  The dog-bone is not a
   uniform bar: the grip and fillet region is much stiffer than the gauge, so
   the gauge must stretch by *more* than the average u_delta/L_half to make
   up the prescribed total elongation, and the true resultant is
   correspondingly higher.

Netting the two out against a proper varying-section solve, the old anchor
*undershoots* by 5-30% depending on geometry (worst for short specimens with
a long fillet, e.g. L_total=40, W_grip=26, W_gauge=6, R_fillet=20 -> -30%).
Correcting only the linearisation, as a uniform-bar finite-strain formula
does, moves the target the wrong way and makes the total error slightly worse.

``section_resultant_series`` therefore solves the 1D varying-section problem:
N is constant along x at equilibrium, so with h(X) the reference half-height,

    P11(X) = N / (2*h(X))    ->    l1(X) = P11^{-1}(N/(2*h(X)))
    integral over [0, L_half] of (l1(X) - 1) dX = u_delta

is one scalar equation for N.  It reduces to the uniform result when the
gauge spans the whole half-length, and it still assumes a locally uniaxial
state (exact in the gauge, approximate through the fillet) -- but it captures
the effect that dominates the error.
"""

import numpy as np

__all__ = [
    "uniaxial_lateral_stretch",
    "uniaxial_P11",
    "stretch_for_P11",
    "section_resultant_uniform",
    "section_resultant_series",
    "section_resultant_small_strain",
    "section_half_height",
]


def uniaxial_lateral_stretch(l1, mu: float, lam: float,
                             n_iter: int = 60, tol: float = 1e-14):
    """Lateral stretch beta for laterally-free uniaxial extension.

    Solves ``mu*t + lam*ln(t) = mu - lam*ln(l1)`` for t = beta^2 by Newton
    from t = 1.  The left-hand side is strictly increasing in t, so the
    iteration is globally convergent once steps are kept positive.

    Args:
        l1:  axial stretch (scalar or array), > 0.
        mu:  shear modulus.
        lam: Lame second parameter.

    Returns:
        beta (same shape as l1).
    """
    l1 = np.asarray(l1, dtype=float)
    rhs = mu - lam * np.log(l1)
    t = np.ones_like(l1)
    for _ in range(n_iter):
        f = mu * t + lam * np.log(t) - rhs
        df = mu + lam / t
        step = f / df
        t_new = np.maximum(t - step, 1e-12)
        if np.max(np.abs(t_new - t)) < tol * np.max(np.abs(t_new)):
            t = t_new
            break
        t = t_new
    return np.sqrt(t)


def uniaxial_P11(l1, mu: float, lam: float):
    """Axial 1st Piola-Kirchhoff stress for laterally-free uniaxial extension.

    P11 = mu*l1 + (lam*ln(J) - mu)/l1  with  J = l1 * beta^2.
    """
    l1 = np.asarray(l1, dtype=float)
    beta = uniaxial_lateral_stretch(l1, mu, lam)
    J = l1 * beta * beta
    return mu * l1 + (lam * np.log(J) - mu) / l1


def stretch_for_P11(p11, mu: float, lam: float,
                    lo: float = 1.0, hi: float = 5.0,
                    n_iter: int = 80, tol: float = 1e-12):
    """Invert ``uniaxial_P11``: axial stretch that produces a given P11.

    P11 is strictly increasing in l1 over the tensile range, so plain
    bisection is used -- robust and derivative-free, and the cost is
    irrelevant since results are cached per geometry.

    Args:
        p11: target nominal stress (scalar or array), >= 0.
        lo, hi: bracket on the axial stretch.

    Returns:
        l1 (same shape as p11).
    """
    p11 = np.asarray(p11, dtype=float)
    a = np.full_like(p11, lo)
    b = np.full_like(p11, hi)
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        too_small = uniaxial_P11(m, mu, lam) < p11
        a = np.where(too_small, m, a)
        b = np.where(too_small, b, m)
        if np.max(b - a) < tol:
            break
    return 0.5 * (a + b)


def section_half_height(X, fillet_info: dict):
    """Reference half-height h(X) of the quarter model, vectorised.

    Constant ``H_gauge`` over the gauge, then the fillet arc.  Mirrors
    ``PI_GINOT_Trainer._section_width`` but accepts arrays.
    """
    X = np.asarray(X, dtype=float)
    x_g = fillet_info["x_g"]
    H_gauge = fillet_info["H_gauge"]
    H_grip = fillet_info["H_grip"]
    xc, yc = fillet_info["arc_center"]
    R = fillet_info["R_fillet"]

    arg = R * R - (X - xc) ** 2
    arc = np.where(arg > 0.0, yc - np.sqrt(np.maximum(arg, 0.0)), H_grip)
    return np.where(X <= x_g, H_gauge, arc)


def section_resultant_small_strain(fillet_info: dict, u_delta: float,
                                   E: float) -> float:
    """Legacy anchor: N = E * (u_delta/L_half) * 2*H_gauge.

    Kept only so the ablation in the paper has the original baseline to
    compare against.  See the module docstring for why it is biased.
    """
    return E * (u_delta / fillet_info["L_half"]) * 2.0 * fillet_info["H_gauge"]


def section_resultant_uniform(fillet_info: dict, u_delta: float,
                              mu: float, lam: float) -> float:
    """Finite-strain anchor assuming a uniform bar at the average stretch.

    Corrects the linearisation but not the varying section, so it is the
    *smaller* of the two corrections and moves the target the wrong way.
    Provided for the ablation.
    """
    L_half = fillet_info["L_half"]
    l1 = 1.0 + u_delta / L_half
    return float(2.0 * fillet_info["H_gauge"] * uniaxial_P11(l1, mu, lam))


def section_resultant_series(fillet_info: dict, u_delta: float,
                             mu: float, lam: float,
                             n_quad: int = 400, n_iter: int = 60,
                             tol: float = 1e-10) -> float:
    """Finite-strain anchor for the actual varying section.

    Solves ``integral (l1(X) - 1) dX = u_delta`` for the constant resultant N,
    where ``l1(X) = P11^{-1}(N / (2 h(X)))``.  Elongation is monotone in N, so
    bisection is used; the bracket is widened until it contains the root.

    Returns:
        N [N], the axial force resultant of the half-section.
    """
    L_half = fillet_info["L_half"]
    X = np.linspace(0.0, L_half, n_quad)
    h = section_half_height(X, fillet_info)

    def elongation(N):
        l1 = stretch_for_P11(N / (2.0 * h), mu, lam)
        return float(np.trapezoid(l1 - 1.0, X))

    # Bracket: the uniform-bar answer is a lower bound in practice; grow until
    # the elongation exceeds the target.
    N_lo = 0.0
    N_hi = max(section_resultant_uniform(fillet_info, u_delta, mu, lam), 1.0)
    for _ in range(40):
        if elongation(N_hi) >= u_delta:
            break
        N_hi *= 2.0
    else:
        raise RuntimeError("section_resultant_series: could not bracket N")

    for _ in range(n_iter):
        N_mid = 0.5 * (N_lo + N_hi)
        if elongation(N_mid) < u_delta:
            N_lo = N_mid
        else:
            N_hi = N_mid
        if N_hi - N_lo < tol * max(N_hi, 1.0):
            break
    return 0.5 * (N_lo + N_hi)
