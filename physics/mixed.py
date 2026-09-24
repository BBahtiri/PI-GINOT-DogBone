#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixed u-P by stress potentials: equilibrium and tractions by construction.

Why this module exists
----------------------
Phase 6.4 established that the stress-concentration failure is in the
*objective*, not the architecture.  Supervised on stress, this network
reaches K_t to 0.39% (R^2 +0.98); supervised on displacement to 0.2% on the
very specimens it is scored on, it gives R^2 -0.55, worse than a constant.
Stress is a derivative of the displacement output, and the concentration
lives entirely in that derivative, which no displacement-only objective
weights properly.

Phase 6.5 then closed the cheap alternative.  Adding the equilibrium
residual to the energy collapsed the field to uniform strain -- predicted K_t
1.000015-1.000036 on all eight geometries -- because a uniform stress is an
exact null of Div P = 0.  A residual term cannot buy stress sensitivity: the
trivial solution satisfies it.

So stress has to be a primary variable, and equilibrium has to be something
the representation *cannot* violate rather than something the optimiser is
asked to learn.

The construction
----------------
The first Piola-Kirchhoff stress is not symmetric at finite strain, so each
of its rows is a separate divergence-free vector field.  In two dimensions a
divergence-free field is the curl of a scalar, so with one potential per row

    P_i1 =  d(phi_i)/dY ,     P_i2 = -d(phi_i)/dX ,     i = 1, 2

the momentum balance Div P = 0 holds identically for *any* phi_1, phi_2 --
the two mixed partials cancel.  Angular momentum (P F^T symmetric) is not
built in; it comes through the constitutive tie below, because P(F) from a
hyperelastic potential satisfies it automatically.

The traction on a boundary with outward normal N and counter-clockwise
tangent T = (-N_Y, N_X) is then

    t_i = P_iJ N_J = grad(phi_i) . T = d(phi_i)/ds

-- the tangential derivative of the potential along the boundary.  Every
natural boundary condition becomes a statement about the potential's value:

    traction-free                   ->  phi_1, phi_2 constant along the edge
    shear-free symmetry plane       ->  one of them constant along the edge
    resultant across a cut A -> B   ->  phi_i(B) - phi_i(A)

Mapped onto the quarter model (the fillet arc runs all the way to the grip,
so the top boundary gauge_top + right_arc is one connected traction-free
curve y = H(X) from (0, H_gauge) to (L_half, H_grip)):

    top, y = H(X)       traction-free       phi_1 = phi_2 = 0   (fixes gauge)
    bottom, y = 0       shear-free, t_1=0   phi_1 = c_1         (constant)
    left, X = 0         shear-free, t_2=0   phi_2 = 0           (continuity
                                                                 at the corner)
    right grip          u Dirichlet         no condition -- reaction

The axial force through any vertical section is then

    F_x(X) = integral_0^H(X) P_11 dY = phi_1(X, H) - phi_1(X, 0) = -c_1

for every X.  **Section-force constancy is exact by construction.**  That is
the quantity Phase 0 found at 6.5% CV, that the resultant anchor and its
calibration existed to enforce, and that has needed a dedicated loss term in
every configuration since.  Here it cannot fail.

The ansatz, with xi = X / L_half and eta = Y / H(X), both in [0, 1]:

    phi_1 = s_F [ -(1 + h)(1 - eta)  +  eta (1 - eta) psi_1(X, Y) ]
    phi_2 = s_F [  xi (1 - eta) psi_2(X, Y) ]

with psi_1, psi_2 two extra decoder channels, h a per-geometry scalar read
off the pooled latent, and s_F = E (u_delta / L_half) H_gauge the small-strain
bar force, so every learned quantity is O(1).  The base term alone is the
one-dimensional bar solution: on the gauge P_11 = s_F (1 + h) / H and P_12 = 0,
and over the fillet a shear P_12 = s_F (1 + h) Y H'(X) / H^2 that follows the
taper.  psi_1 and psi_2 add everything two-dimensional, including the
concentration.

Why the Phase 6.5 collapse is now impossible
--------------------------------------------
A uniform P that is traction-free on the arc must vanish, because the arc's
normals span both directions.  So P_phi cannot be uniform unless it is zero,
and zero stress contradicts the stretch the grip imposes through the
constitutive tie.  The degenerate direction the hybrid fell into does not
exist in this representation.

The loss
--------
With equilibrium, tractions and Dirichlet data all exact by construction, the
only thing left to enforce is that the two stress fields agree:

    L_c = integral || P_phi - P(F(u)) ||^2 dA  /  (sigma_nom^2 |Omega|)

If L_c = 0 then P(F(u)) is equilibrated and traction-free, so u solves the
boundary-value problem.  It is a complete formulation on its own -- a
least-squares mixed method with the hard part built into the representation
rather than penalised -- and it needs only first derivatives of both fields,
the same order as the energy form.

Scope and the one real limitation
---------------------------------
H(X) here is the analytic top boundary of the parametric dog-bone, which is
exact and cheap and is this paper's scope.  An arbitrary geometry would need
an approximate distance function in its place (R-functions; Sukumar and
Srivastava's construction is the standard one), and that is where a general
version of this would have to go next.
"""

from __future__ import annotations

import torch

from physics.neo_hookean import (first_piola_kirchhoff_stress,
                                 strain_energy_density)
from physics.weak_form import EnergyLoss


def half_height(X: torch.Tensor, fi: dict) -> torch.Tensor:
    """Reference half-height H(X), differentiable in X.

    Constant H_gauge up to the tangency point x_g, then the fillet arc, which
    ends exactly at (L_half, H_grip) for every geometry in both banks
    (R / dH >= 1.149 measured, and >= 1 is what the arc needs).  C1 at x_g:
    the arc is tangent to the gauge there, so H' is continuous and only H''
    jumps -- and H'' never enters the stress, which needs first derivatives
    of the potentials only.
    """
    xc, yc = fi["arc_center"]
    R = fi["R_fillet"]
    arg = torch.clamp(R * R - (X - xc) ** 2, min=1e-12)
    arc = yc - torch.sqrt(arg)
    return torch.where(X <= fi["x_g"], torch.full_like(X, fi["H_gauge"]), arc)


def mixed_fields(model, pts: torch.Tensor, geometry_latent, u_delta, x_max,
                 y_max, fi: dict, mu, lam, stress_state: str, E: float,
                 create_graph: bool = True) -> dict:
    """Both stress fields at ``pts`` from one pass of the shared trunk.

    ``pts`` must require grad.  Returns the displacement, the four in-plane
    components of P from the potentials and from the constitutive law, the
    axial force -c_1, and the determinants for the barrier.
    """
    dec = model.decoder
    if not getattr(dec, "mixed", False):
        raise RuntimeError("mixed_fields needs a decoder built with "
                           "DECODER_CONFIG['mixed'] = True")

    feats, z_pool = dec.features(pts, geometry_latent, x_max, y_max)
    raw = dec.output_scale * dec.output_proj(feats)
    uv = dec._apply_hard_bc(raw, pts, u_delta, x_max, y_max)
    psi = dec.stress_proj(feats)                       # [B, N, 2]
    h = dec.force_head(z_pool)                         # [B, 1]

    X, Y = pts[..., 0:1], pts[..., 1:2]
    L_half = fi["L_half"]
    ud = float(u_delta.reshape(-1)[0])
    s_F = E * (ud / L_half) * fi["H_gauge"]

    H = half_height(X, fi)
    xi, eta = X / L_half, Y / H
    one_h = 1.0 + h.unsqueeze(1)                       # [B, 1, 1]
    phi1 = s_F * (-one_h * (1.0 - eta) + eta * (1.0 - eta) * psi[..., 0:1])
    phi2 = s_F * (xi * (1.0 - eta) * psi[..., 1:2])

    def grad(f):
        return torch.autograd.grad(f, pts, torch.ones_like(f),
                                   create_graph=create_graph,
                                   retain_graph=True)[0]

    g1, g2 = grad(phi1), grad(phi2)
    gu, gv = grad(uv[..., 0:1]), grad(uv[..., 1:2])
    P11p, P12p = g1[..., 1:2], -g1[..., 0:1]
    P21p, P22p = g2[..., 1:2], -g2[..., 0:1]

    P11, P12, P21, P22, J2D, J3D = first_piola_kirchhoff_stress(
        gu[..., 0:1], gu[..., 1:2], gv[..., 0:1], gv[..., 1:2],
        mu, lam, stress_state, return_J3D=True)

    return {"uv": uv, "phi1": phi1, "phi2": phi2,
            "grad_u": (gu[..., 0:1], gu[..., 1:2], gv[..., 0:1], gv[..., 1:2]),
            "P_phi": (P11p, P12p, P21p, P22p),
            "P_F": (P11, P12, P21, P22),
            "axial_force": s_F * one_h.reshape(-1),   # F_x = -c_1
            "sigma_nom": E * ud / L_half,
            "J2D": J2D, "J3D": J3D}


class MixedLoss(EnergyLoss):
    """Constitutive gap between the two stress fields, on the energy quadrature.

    Subclasses EnergyLoss for three things it already gets right: the
    stratified, resampled quadrature (a fixed rule collapses -- Phase 2), the
    determinant barrier, and a place in the trainer's energy branch, which is
    where the geometry parameters this loss needs are threaded.

    ``w_energy`` > 0 adds the total potential energy back.  The default is
    zero because the gap alone is a complete formulation and the cleaner
    test; the energy is kept available because a least-squares functional can
    be worse-conditioned than the minimum principle it replaces, and if it is,
    that is the first thing to try.
    """

    def __init__(self, *args, E: float, w_energy: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.E = float(E)
        self.register_buffer("w_energy", torch.tensor(float(w_energy)))

    def forward(self, model, params: dict, boundary_pc, u_delta, x_max,
                y_max=None, sample_ids=None, geometry_latent=None,
                trac_free_pts=None, trac_free_normals=None,
                trac_free_tags=None, interior_pts=None) -> dict:
        from config import get_fillet_geometry
        if geometry_latent is None:
            if hasattr(model, "set_geometry"):
                model.set_geometry(params)
            geometry_latent = model.encode(boundary_pc, x_max, y_max,
                                           sample_ids=sample_ids)
        fi = get_fillet_geometry(params)
        pts, w, area, _ = self.quad_points(params, u_delta.device)
        q = pts.clone().requires_grad_(True)
        f = mixed_fields(model, q, geometry_latent, u_delta, x_max, y_max,
                         fi, self.mu, self.lam, self.stress_state, self.E)

        gap = sum((a - b) ** 2 for a, b in zip(f["P_phi"], f["P_F"]))
        L_c = (gap[0, :, 0] * w).sum() / (f["sigma_nom"] ** 2 * area)

        J_guard = torch.minimum(f["J2D"], f["J3D"])
        L_barrier = torch.mean(torch.relu(self.j_min - J_guard) ** 2)
        loss = L_c + self.w_bar * L_barrier

        out = {"loss": loss, "geometry_latent": geometry_latent,
               "L_c": L_c, "L_c_log": float(L_c.detach()),
               "L_barrier": L_barrier,
               "L_barrier_log": float(L_barrier.detach()),
               "axial_force_log": float(f["axial_force"].detach().mean()),
               "min_J2D_log": float(f["J2D"].detach().min()),
               "min_J3D_log": float(f["J3D"].detach().min()),
               "area": area}
        if float(self.w_energy) > 0.0:
            # From the gradients the gap already computed, on the same
            # quadrature points.  Calling self.energy() instead would cost a
            # second forward pass and -- because the points are resampled on
            # every call -- integrate the two terms over different samples.
            Psi = strain_energy_density(*f["grad_u"], self.mu, self.lam,
                                        self.stress_state)
            Pi = (Psi[0, :, 0] * w).sum()
            L_energy = Pi / (self.E_scale * area)
            out["loss"] = out["loss"] + self.w_energy * L_energy
            out["L_energy_log"] = float(L_energy.detach())
        # Keys the trainer's epoch summary reads for the energy form.
        out.setdefault("L_energy_log", 0.0)
        out["Pi_log"] = out["L_energy_log"]
        out["L_trac_top_log"] = out["L_trac_arc_log"] = 0.0
        return out


def cauchy_from_P(P, grad_u, J3D):
    """In-plane Cauchy stress sigma = P F^T / J from a first Piola stress.

    Plane stress, so P_i3 = 0 and the in-plane block needs only the in-plane
    F.  Returns (s11, s22, s12, asym) where s12 is the symmetrised shear and
    ``asym`` = s12 - s21 before symmetrising.  For P from the constitutive law
    asym vanishes identically; for P from the potentials it measures the
    angular-momentum balance the representation does not build in, and so is
    reported rather than discarded.
    """
    P11, P12, P21, P22 = P
    dudx, dudy, dvdx, dvdy = grad_u
    F11, F12, F21, F22 = 1.0 + dudx, dudy, dvdx, 1.0 + dvdy
    s11 = (P11 * F11 + P12 * F12) / J3D
    s12a = (P11 * F21 + P12 * F22) / J3D
    s21a = (P21 * F11 + P22 * F12) / J3D
    s22 = (P21 * F21 + P22 * F22) / J3D
    return s11, s22, 0.5 * (s12a + s21a), s12a - s21a
