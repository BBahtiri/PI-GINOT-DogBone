#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo-Hookean hyperelastic constitutive law — PyTorch port.

All functions operate on batched displacement-gradient tensors [B, N, 1]
and return stress components in the same shape.

Neo-Hookean strain energy (compressible, 3D):
    W = (mu/2)(tr(F^T F) - 3) - mu ln(J) + (lam/2)(ln J)^2
    where J = det(F), F = I + grad(u)

First Piola-Kirchhoff stress (reference configuration):
    P = mu F + (lam ln J - mu) F^{-T}

Cauchy stress (current configuration, for post-processing only):
    sigma = (1/J) P F^T

Equilibrium and traction BCs use P in the reference frame:
    Div(P) = 0          (interior equilibrium)
    P . N  = 0          (traction-free boundaries, N = reference normal)

Plane stress support
--------------------
For a thin specimen, F33 is solved from P33 = 0.  Multiplying P33 = 0 by
F33 gives the scalar residual actually solved here:

    r(F33; J2D) = mu*F33^2 + lam*ln(J2D*F33) - mu = 0

The solve runs in two stages: 12 Newton steps under ``torch.no_grad()`` take
the residual to machine precision, then 2 *differentiable* Newton steps from
that converged root reattach F33 to the autograd graph.  The graph therefore
carries 2 clean steps instead of 5 clamped ones, and the values are exact
rather than ~1e-11.

The analytic first derivative is

    dF33/dJ2D = -(lam/J2D) / (2*mu*F33 + lam/F33)

which at J2D = 1, F33 = 1 gives -lam/(2*mu+lam) — exactly the small-strain
plane-stress value -nu/(1-nu).  Note that a custom autograd Function which
returns only this expression is *not* sufficient: ``equilibrium_residual``
differentiates P once w.r.t. x and the optimiser differentiates again w.r.t.
the parameters, so the second derivative of the closure is on the training
path and a detached-IFT backward reports it as zero (~14% error).  See
``solve_F33_plane_stress``.

The 3D determinant J3D = J2D * F33 replaces J2D in the log term.
"""

import warnings

import torch

# Minimum detF to prevent log(0) or log(negative) during early training.
_DETF_MIN = 1e-6

# Lower clamp applied to F33 during the Newton solve.  Hitting this bound
# means the solve did not converge to a physical root.
_F33_MIN = 1e-3

# Convergence diagnostics for the plane-stress closure.  Collecting them
# requires .item() calls, which force a host sync, so they are gathered only
# every ``F33_DIAG_EVERY`` solves.  Set to 1 in tests, 0 to disable entirely.
F33_DIAG_EVERY = 50
_F33_CALL_COUNT = 0

# Populated in-place by the most recent *checked* solve, so callers and tests
# can inspect convergence without changing any return signature.  May be stale
# between checks when F33_DIAG_EVERY > 1.
LAST_F33_SOLVE = {
    "n_iter": 0,
    "max_residual": float("nan"),
    "n_clamped": 0,
    "max_step": float("nan"),
    "checked_call": -1,
}

# Warn at most once per process if the closure fails to converge.
_F33_WARNED = False


def require_stress_state(stress_state):
    """Validate the stress state, refusing to silently pick a constitutive model.

    Every physics entry point used to default to ``'plane strain'``.  Any new
    call site that forgot the argument would therefore have switched the
    constitutive model without warning.  The default is now ``None`` and this
    helper turns the omission into a loud error.
    """
    if stress_state is None:
        raise ValueError(
            "stress_state must be given explicitly ('plane strain' or "
            "'plane stress') — pass MATERIAL_CONFIG['state']."
        )
    if stress_state not in ("plane strain", "plane stress"):
        raise ValueError(
            f"Unknown stress_state {stress_state!r}; expected "
            "'plane strain' or 'plane stress'."
        )
    return stress_state


def deformation_gradient(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
) -> tuple:
    """Build the 2D deformation gradient F = I + grad(u).

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


def _newton_F33(J: torch.Tensor, mu, lam, n_iter: int) -> tuple:
    """Newton iteration on r(F33) = 0.  Caller controls grad mode.

    Returns (F33, last_step); ``last_step`` is a convergence diagnostic.
    """
    F33 = torch.ones_like(J)
    step = torch.zeros_like(J)
    for _ in range(n_iter):
        r = mu * F33 * F33 + lam * torch.log(J * F33) - mu
        dr = 2.0 * mu * F33 + lam / F33
        step = r / dr
        F33 = torch.clamp(F33 - step, min=_F33_MIN)
    return F33, step


def solve_F33_plane_stress(J2D: torch.Tensor, mu, lam,
                           n_iter: int = 12,
                           n_correct: int = 2) -> torch.Tensor:
    """Out-of-plane stretch F33 satisfying P33 = 0 (plane stress closure).

    Two-stage solve:

    1. ``n_iter`` Newton steps under ``torch.no_grad()`` drive the residual to
       machine precision.  No autograd graph is built, so this stage is free
       in both memory and graph depth.
    2. ``n_correct`` *differentiable* Newton steps starting from that converged
       root.  Their values are unchanged (r ~ 0, so each step moves F33 by
       ~1e-16), but they reconnect F33 to the graph with the correct
       derivatives.

    Why not a pure implicit-function-theorem backward?  An IFT backward built
    from *detached* saved tensors is exact to first order but reports
    d2F33/dJ2D2 = 0.  ``equilibrium_residual`` differentiates P once w.r.t. x
    and the optimiser then differentiates again w.r.t. the parameters, so the
    second derivative of the closure is on the training path: a pure-IFT
    backward silently biases it by ~14% (measured against the exact Lambert-W
    solution).  One differentiable Newton step is equivalent to IFT and has
    the same defect; two steps recover the second derivative to ~1e-8
    (finite-difference reference precision) and a third changes nothing.  See
    ``tests/test_physics.py::test_F33_second_derivative``.

    Cost versus the previous implementation: values are exact instead of
    ~1e-11, and the graph carries 2 clean Newton steps instead of 5 with
    clamps and epsilon guards.

    Args:
        J2D:       [B, N, 1] in-plane determinant.  Values below ``_DETF_MIN``
                   (possible early in training) are clamped.
        mu:        Shear modulus (float or tensor).
        lam:       Lame second parameter (float or tensor).
        n_iter:    Detached Newton iterations (default 12).
        n_correct: Differentiable correction steps (default 2 — do not lower
                   this to 1, see above).

    Returns:
        F33: [B, N, 1] out-of-plane stretch satisfying P33 ~ 0.
    """
    global _F33_CALL_COUNT, _F33_WARNED

    J = torch.clamp(J2D, min=_DETF_MIN)

    with torch.no_grad():
        F33, step = _newton_F33(J.detach(), mu, lam, n_iter)

        _F33_CALL_COUNT += 1
        if F33_DIAG_EVERY and _F33_CALL_COUNT % F33_DIAG_EVERY == 0:
            resid = mu * F33 * F33 + lam * torch.log(J.detach() * F33) - mu
            mu_abs = float(torch.as_tensor(mu).abs().max())
            max_resid = float((resid.abs().max() / mu_abs).item())
            n_clamped = int((F33 <= _F33_MIN * (1.0 + 1e-12)).sum().item())

            LAST_F33_SOLVE.update(
                n_iter=n_iter,
                max_residual=max_resid,
                n_clamped=n_clamped,
                max_step=float(step.abs().max().item()),
                checked_call=_F33_CALL_COUNT,
            )

            # Newton cannot beat the working precision: float32 bottoms out
            # around 1e-7 relative, float64 around 1e-15.
            tol = 64.0 * torch.finfo(J.dtype).eps
            if (max_resid > tol or n_clamped > 0) and not _F33_WARNED:
                _F33_WARNED = True
                warnings.warn(
                    "plane-stress F33 closure did not converge cleanly "
                    f"(max |r|/mu = {max_resid:.3e} > tol {tol:.1e}, "
                    f"{n_clamped} points clamped at the lower bound). "
                    "Deformation may be unphysical; check the J barrier.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # Differentiable correction steps: values unchanged, derivatives attached.
    for _ in range(n_correct):
        r = mu * F33 * F33 + lam * torch.log(J * F33) - mu
        dr = 2.0 * mu * F33 + lam / F33
        F33 = torch.clamp(F33 - r / dr, min=_F33_MIN)

    return F33


def strain_energy_density(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str = None,
) -> torch.Tensor:
    """Neo-Hookean strain energy density.

        W = (mu/2)(tr C - 3) - mu ln J + (lam/2)(ln J)^2

    For plane strain, F33 = 1 and J = J2D.

    For plane stress this returns the **reduced** energy: F33 is eliminated by
    the P33 = 0 closure, so W is a function of the in-plane F alone.  That is
    the correct object to minimise, and it has a property worth stating,
    because it is what makes the energy form and the stress form consistent:

        dW_reduced/dF = dW/dF + (dW/dF33)(dF33/dF) = dW/dF = P

    The second term vanishes because F33 is chosen to make dW/dF33 = P33 = 0.
    That is the envelope theorem, and it means the gradient of the reduced
    energy *is* the plane-stress first Piola-Kirchhoff stress -- no separate
    derivation, and no inconsistency between the two formulations.  Autograd
    still differentiates through the closure, but that path is multiplied by a
    residual the solver drives to ~1e-16, so it contributes nothing.
    ``tests/test_physics.py::test_energy_gradient_is_the_stress`` checks it.

    Args:
        du_dx, du_dy, dv_dx, dv_dy: [B, N, 1] displacement gradients.
        mu, lam: Lame parameters.
        stress_state: 'plane strain' or 'plane stress'.  Required.

    Returns:
        W: [B, N, 1] energy density [MPa] (energy per unit reference volume).
    """
    require_stress_state(stress_state)

    F11, F12, F21, F22, J2D = deformation_gradient(du_dx, du_dy, dv_dx, dv_dy)
    J2D_safe = torch.clamp(J2D, min=_DETF_MIN)
    trC_2d = F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2

    if stress_state == "plane stress":
        F33 = solve_F33_plane_stress(J2D, mu, lam)
        J = torch.clamp(J2D_safe * F33, min=_DETF_MIN)
        trC = trC_2d + F33 ** 2
    else:
        J = J2D_safe
        trC = trC_2d + 1.0

    logJ = torch.log(J)
    return 0.5 * mu * (trC - 3.0) - mu * logJ + 0.5 * lam * logJ ** 2


def first_piola_kirchhoff_stress(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str = None,
    return_J3D: bool = False,
) -> tuple:
    """Compute 1st Piola-Kirchhoff stress P for a Neo-Hookean material.

    P = mu F + (lam ln J - mu) F^{-T}

    For plane strain:  F33 = 1, J = J2D.
    For plane stress:  F33 solved from P33 = 0, J = J2D * F33.

    In both cases, the in-plane F^{-T} components are computed from
    the 2D submatrix (divided by J2D, not J3D).

    Args:
        du_dx, du_dy, dv_dx, dv_dy: [B, N, 1] displacement gradients.
        mu:  Shear modulus (Lame first parameter).
        lam: Lame second parameter.
        stress_state: 'plane strain' or 'plane stress'.  Required.
        return_J3D: if True, also return the 3D determinant J3D = J2D * F33
            (equal to J2D under plane strain).  The J barrier must guard J3D,
            since F33 < 1 in tension makes J3D the more critical quantity.

    Returns:
        P11, P12, P21, P22, detF                (return_J3D=False)
        P11, P12, P21, P22, detF, J3D           (return_J3D=True)
        each [B, N, 1].  detF is the in-plane determinant J2D.
    """
    require_stress_state(stress_state)

    F11, F12, F21, F22, detF = deformation_gradient(du_dx, du_dy, dv_dx, dv_dy)

    J2D = detF
    J2D_safe = torch.clamp(J2D, min=_DETF_MIN)

    # In-plane F^{-T} (always uses J2D, not J3D):
    invFT11 = F22 / J2D_safe
    invFT12 = -F21 / J2D_safe
    invFT21 = -F12 / J2D_safe
    invFT22 = F11 / J2D_safe

    if stress_state == "plane stress":
        F33 = solve_F33_plane_stress(J2D, mu, lam)
        J3D = torch.clamp(J2D_safe * F33, min=_DETF_MIN)
        logJ = torch.log(J3D)
    else:
        # Plane strain: F33 = 1, J = J2D
        J3D = J2D_safe
        logJ = torch.log(J2D_safe)

    coeff = lam * logJ - mu          # (lam ln J - mu)

    P11 = mu * F11 + coeff * invFT11
    P12 = mu * F12 + coeff * invFT12
    P21 = mu * F21 + coeff * invFT21
    P22 = mu * F22 + coeff * invFT22

    if return_J3D:
        return P11, P12, P21, P22, detF, J3D
    return P11, P12, P21, P22, detF


def cauchy_stress(
    du_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dx: torch.Tensor,
    dv_dy: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str = None,
) -> tuple:
    """Compute Cauchy stress sigma = (1/J) P F^T (for post-processing only).

    NOT used for PDE residuals — use first_piola_kirchhoff_stress() instead.

    For plane stress, J3D = J2D * F33 is used for the 1/J factor.  J3D is
    taken from the P computation rather than re-solving the closure, which
    halves the cost of every post-processing path.

    Args:
        du_dx, du_dy, dv_dx, dv_dy: [B, N, 1] displacement gradients.
        mu, lam: Lame parameters.
        stress_state: 'plane strain' or 'plane stress'.  Required.

    Returns:
        S11, S22, S12, detF: each [B, N, 1].
    """
    require_stress_state(stress_state)

    F11, F12, F21, F22, detF = deformation_gradient(du_dx, du_dy, dv_dx, dv_dy)
    P11, P12, P21, P22, _, J3D = first_piola_kirchhoff_stress(
        du_dx, du_dy, dv_dx, dv_dy, mu, lam, stress_state, return_J3D=True
    )

    inv_J = 1.0 / torch.clamp(J3D, min=_DETF_MIN)

    # sigma = (1/J) P F^T
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
    stress_state: str = None,
) -> tuple:
    """Compute full Cauchy stress including out-of-plane sigma_33.

    For plane strain Neo-Hookean:
        F33 = 1  ->  det(F_3D) = det(F_2D) = J
        sigma_33 = lam ln(J) / J

    For plane stress:
        sigma_33 = 0  (by definition; F33 solved from P33 = 0)

    Returns:
        S11, S22, S33, S12, detF: each [B, N, 1].
    """
    require_stress_state(stress_state)

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

    sigma_vm = sqrt( s11^2 + s22^2 + s33^2 - s11 s22 - s22 s33 - s11 s33
                     + 3 s12^2 )
    """
    return torch.sqrt(
        S11**2 + S22**2 + S33**2
        - S11 * S22 - S22 * S33 - S11 * S33
        + 3.0 * S12**2
        + 1e-12
    )
