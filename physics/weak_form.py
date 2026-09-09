#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Total potential energy as the training loss.

Why this replaces five penalty terms
------------------------------------
The total potential energy of a hyperelastic body is

    Pi = integral_Omega Psi(F) dV  -  integral_Gamma_t t_bar . u dS

and in this quarter model **every** boundary is one of: hard Dirichlet
(``right_grip``, and the two symmetry planes, all baked into the decoder),
traction-free (``gauge_top``, ``right_arc``), or a symmetry plane.  There is no
applied traction anywhere, so the second integral vanishes identically and

    Pi = integral_Omega Psi(F) dV

is the whole loss.  Minimising it subject to the hard Dirichlet conditions
gives equilibrium ``Div P = 0`` in the interior *and* ``P.N = 0`` on the free
boundaries as **natural** boundary conditions -- satisfied at the minimum
rather than fought for by penalty weights.  One term replaces ``L_eq``,
``L_trac_top``, ``L_trac_arc``, ``L_trac_hole`` and ``L_part``.

That matters here for a specific, measured reason.  ``docs/phase1_transverse_
study.md`` showed the traction-free condition on the gauge top is the only
term that sets the lateral contraction, that it carried the lowest weight in
the configuration, and that raising it cuts the transverse error by up to
2.1x. In the energy form that condition is not weighted at all -- it is a
consequence of the minimum -- so the failure mode has no way to occur. That is
a falsifiable prediction, and ``verification/energy_study.py`` tests it.

Two further properties worth having
-----------------------------------
*The loss is an error bound.*  Any field satisfying the Dirichlet conditions
has ``Pi >= Pi_exact``, so ``Pi_model - Pi_reference`` is a one-sided measure
of how far the model is from the solution -- unlike a residual, which can be
small for the wrong reason.

*Only first derivatives are needed.*  ``Psi`` depends on ``grad u`` alone, so
the loss requires one differentiation of the network instead of the two that
``Div P`` needs.  The optimiser's backward pass is then second-order overall
rather than third.

Quadrature, and why the points must move
----------------------------------------
``integral Psi dV`` is taken on the triangulation from
``geometry/triangulation.py`` -- the same mesher the finite-element reference
uses, so an integration artefact cannot masquerade as a modelling difference.

A **fixed** quadrature rule does not work here, and the failure is
spectacular rather than subtle.  Minimising a finite sum is not minimising an
integral: once the field is near the true minimiser, the only way left to
reduce the sum is to exploit the fixed points, and a network with this much
capacity does exactly that.  Measured on validation geometry 0 with a cached
degree-2 rule on 1233 triangles:

    epoch  100   Pi/Pi_ref = 0.992   u 5.9e-03   v 0.277   vm 0.085
    epoch  200   Pi/Pi_ref = 0.657   u 5.0e-02   v 0.379   vm 1.032
    epoch  300   Pi/Pi_ref = 0.007   u 2.3e-01   v 0.992   vm 2.701

At epoch 100 the energy form beats every penalty configuration tried.  By
epoch 300 it has driven the *sum* to nearly zero while the *field* is three
times worse than where it started.  Note that ``Pi < Pi_ref`` is impossible
for a field meeting the Dirichlet conditions, so the energy is its own
collapse detector -- ``EnergyLoss.collapse_ratio`` reports it.

The fix keeps the triangulation but resamples the points inside it each call:
one or more uniformly distributed points per triangle, weighted by that
triangle's exact area.  This is *stratified* Monte Carlo, which answers both
objections to plain Monte Carlo -- it is unbiased by construction (uniform
within each cell, exact cell measures) and stratification by element keeps the
variance far below the unstratified estimator, because the mesh is already
graded to where the integrand varies.  And there is no fixed point set left to
exploit.

The plan for this phase specified caching the triangulation "so a bank
geometry is triangulated once, not once per epoch".  The triangulation should
indeed be cached -- it is.  The quadrature *points* must not be.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from geometry.triangulation import cached_mesh, quadrature
from physics.neo_hookean import (deformation_gradient, require_stress_state,
                                 solve_F33_plane_stress, strain_energy_density)


class EnergyLoss(nn.Module):
    """Total potential energy of the quarter model, nondimensionalised.

    Args:
        mu, lam: Lame parameters.
        stress_state: 'plane strain' or 'plane stress'.  Required.
        quad_degree: fixed-rule degree (1 centroid, 2 three-point, 3
            four-point), used when ``resample`` is False.
        resample: draw fresh stratified points inside the cached triangles on
            every call.  **Leave this on.**  With a fixed rule the optimiser
            drives the quadrature sum to zero while the field diverges; see
            the module docstring for the measured trajectory.
        n_per_elem: stratified samples per triangle when resampling.  3 gives
            the same point count as the degree-2 fixed rule.
        h: element size for the quadrature mesh.  None sizes it to
            ``n_elem_target`` triangles instead, because every quadrature point
            is a decoder forward pass: at the reference mesh's resolution the
            integral does not fit in memory, and the energy integrand is far
            smoother than the stress it is integrated from.
        n_elem_target: triangle budget when ``h`` is None.  1400 triangles at
            degree 2 is 4200 points, comparable to the 4000 collocation points
            the energy replaces.
        w_barrier, j_min: the determinant barrier, unchanged from PhysicsLoss.
            The energy already diverges as J -> 0, but only logarithmically,
            and early in training J can go negative outright, where the energy
            is undefined rather than large.
        E_scale: characteristic stress used to nondimensionalise Pi.
    """

    def __init__(self, mu: float, lam: float, stress_state: str = None,
                 quad_degree: int = 2, h: Optional[float] = None,
                 n_elem_target: int = 1400, resample: bool = True,
                 n_per_elem: int = 3, seed: int = 0,
                 w_barrier: float = 1e3, j_min: float = 0.05,
                 E_scale: float = 760.0):
        super().__init__()
        self.mu = mu
        self.lam = lam
        self.stress_state = require_stress_state(stress_state)
        self.quad_degree = quad_degree
        self.h = h
        self.n_elem_target = n_elem_target
        self.resample = resample
        self.n_per_elem = n_per_elem
        self._rng = np.random.default_rng(seed)
        self.j_min = j_min
        self.E_scale = E_scale
        self.register_buffer("w_bar", torch.tensor(float(w_barrier)))
        self._quad_cache = {}

    # ------------------------------------------------------------------
    def _mesh_h(self, params: dict) -> float:
        """Element size for this geometry's quadrature mesh."""
        if self.h is not None:
            return self.h
        from geometry.triangulation import h_for_budget
        return h_for_budget(params, self.n_elem_target)

    def _mesh(self, params: dict):
        """The cached triangulation for a geometry."""
        return cached_mesh(params, h=self._mesh_h(params))

    def _stratified(self, mesh):
        """Uniform points inside each triangle, with exact area weights.

        Barycentric coordinates from the standard reflection trick: draw
        r1, r2 ~ U(0,1) and fold the half that falls outside the simplex.
        Unbiased for the element integral, so the sum is an unbiased estimate
        of ``integral Psi dV`` with variance reduced by the mesh's own grading.
        """
        n_e, k = mesh.n_elem, self.n_per_elem
        r1 = self._rng.random((n_e, k))
        r2 = self._rng.random((n_e, k))
        flip = (r1 + r2) > 1.0
        r1 = np.where(flip, 1.0 - r1, r1)
        r2 = np.where(flip, 1.0 - r2, r2)
        bary = np.stack([1.0 - r1 - r2, r1, r2], axis=-1)     # (M, k, 3)
        verts = mesh.nodes[mesh.tris]                          # (M, 3, 2)
        pts = np.einsum("mkа,mad->mkd".replace("а", "a"), bary, verts)
        w = np.repeat(mesh.areas[:, None] / k, k, axis=1)
        return pts.reshape(-1, 2), w.reshape(-1)

    def quad_points(self, params: dict, device, dtype=torch.float32):
        """Quadrature points and weights for a geometry.

        The mesh is cached; the points are not, unless ``resample`` is off.
        """
        mesh = self._mesh(params)
        area = float(mesh.areas.sum())
        if self.resample:
            pts, w = self._stratified(mesh)
            return (torch.tensor(pts, dtype=dtype, device=device).unsqueeze(0),
                    torch.tensor(w, dtype=dtype, device=device), area, mesh)

        key = (id(device), str(dtype),
               round(params["L_total"], 9), round(params["W_grip"], 9),
               round(params["W_gauge"], 9), round(params["R_fillet"], 9))
        hit = self._quad_cache.get(key)
        if hit is None:
            pts, w = quadrature(mesh, self.quad_degree)
            hit = (
                torch.tensor(pts, dtype=dtype, device=device).unsqueeze(0),
                torch.tensor(w, dtype=dtype, device=device), area, mesh,
            )
            self._quad_cache[key] = hit
        return hit

    # ------------------------------------------------------------------
    def energy(self, model, params: dict, geometry_latent, u_delta, x_max,
               y_max, return_fields: bool = False):
        """Pi = integral Psi dV, plus the determinant statistics."""
        device = u_delta.device
        pts, w, area, _ = self.quad_points(params, device)
        q = pts.clone().requires_grad_(True) if not self.resample \
            else pts.requires_grad_(True)

        _, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
            q, geometry_latent, u_delta, x_max, y_max)
        Psi = strain_energy_density(du_dx, du_dy, dv_dx, dv_dy,
                                    self.mu, self.lam, self.stress_state)
        Pi = (Psi[0, :, 0] * w).sum()

        _, _, _, _, J2D = deformation_gradient(du_dx, du_dy, dv_dx, dv_dy)
        if self.stress_state == "plane stress":
            J3D = J2D * solve_F33_plane_stress(J2D, self.mu, self.lam)
        else:
            J3D = J2D
        if return_fields:
            return Pi, J2D, J3D, area, Psi
        return Pi, J2D, J3D, area

    # ------------------------------------------------------------------
    def forward(self, model, params: dict, boundary_pc, u_delta, x_max,
                y_max=None, sample_ids=None, geometry_latent=None) -> dict:
        """Energy loss for one geometry.

        ``params`` selects the cached quadrature mesh; ``boundary_pc`` is the
        encoder input, exactly as in PhysicsLoss.
        """
        if geometry_latent is None:
            geometry_latent = model.encode(boundary_pc, x_max, y_max,
                                           sample_ids=sample_ids)

        Pi, J2D, J3D, area = self.energy(model, params, geometry_latent,
                                         u_delta, x_max, y_max)

        # Nondimensional: energy per unit volume, in units of E.
        L_energy = Pi / (self.E_scale * area)

        J_guard = torch.minimum(J2D, J3D)
        L_barrier = torch.mean(torch.relu(self.j_min - J_guard) ** 2)

        loss = L_energy + self.w_bar * L_barrier
        return {
            "loss": loss,
            "geometry_latent": geometry_latent,
            "L_energy": L_energy,
            "L_barrier": L_barrier,
            "Pi": Pi,
            "Pi_log": float(Pi.detach()),
            "L_energy_log": float(L_energy.detach()),
            "L_barrier_log": float(L_barrier.detach()),
            "min_J2D_log": float(J2D.detach().min()),
            "min_J3D_log": float(J3D.detach().min()),
            "area": area,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def integration_error(self, model, params: dict, geometry_latent, u_delta,
                          x_max, y_max, refine: float = 2.0) -> dict:
        """Compare Pi on the training mesh against a refined one.

        The plan requires this figure: the quadrature error must sit well
        below the physics signal being optimised, or a "better" energy is just
        a better-integrated one.  Reported as a relative difference, and also
        against a higher-degree rule on the same mesh, which separates mesh
        resolution from quadrature order.
        """
        h0 = self._mesh_h(params)

        def pi_for(h, degree, chunk=4096):
            mesh = cached_mesh(params, h=h)
            pts, w = quadrature(mesh, degree)
            total = 0.0
            for lo in range(0, len(pts), chunk):
                hi = min(lo + chunk, len(pts))
                q = torch.tensor(pts[lo:hi], dtype=x_max.dtype,
                                 device=x_max.device
                                 ).unsqueeze(0).requires_grad_(True)
                with torch.enable_grad():
                    _, a, b, c, d = model.predict_with_grad_latent(
                        q, geometry_latent, u_delta, x_max, y_max)
                    Psi = strain_energy_density(a, b, c, d, self.mu, self.lam,
                                                self.stress_state)
                wt = torch.tensor(w[lo:hi], dtype=x_max.dtype,
                                  device=x_max.device)
                total += float((Psi[0, :, 0].detach() * wt).sum())
                del q, a, b, c, d, Psi
            return total

        base = pi_for(h0, self.quad_degree)
        finer_mesh = pi_for(h0 / refine, self.quad_degree)
        higher_rule = pi_for(h0, min(self.quad_degree + 1, 3))
        return {
            "Pi": base,
            "Pi_refined_mesh": finer_mesh,
            "Pi_higher_rule": higher_rule,
            "rel_mesh": abs(base - finer_mesh) / max(abs(finer_mesh), 1e-300),
            "rel_rule": abs(base - higher_rule) / max(abs(higher_rule), 1e-300),
        }


def collapse_ratio(pi_model: float, pi_reference: float) -> float:
    """Pi_model / Pi_reference.

    Any field satisfying the Dirichlet conditions has ``Pi >= Pi_exact``, and
    the reference is itself an upper bound on ``Pi_exact``, so a ratio
    meaningfully below 1 means the quadrature sum has stopped approximating
    the integral rather than that the model has beaten the reference.  Values
    under about 0.95 should be treated as quadrature collapse, not progress.
    """
    return pi_model / pi_reference


def reference_energy(sol, mu: float, lam: float, stress_state: str) -> float:
    """Pi of a finite-element solution, for use as the comparison value.

    Any admissible field has Pi >= Pi_exact, so the reference energy is itself
    an upper bound and ``Pi_model - Pi_reference`` is a signed measure of which
    discretisation is closer to the true minimiser.
    """
    from verification.fem.solver import deformation_gradients

    F = deformation_gradients(sol.mesh, sol.u)
    g = [torch.tensor(F[:, i, j] - (1.0 if i == j else 0.0),
                      dtype=torch.float64).reshape(1, -1, 1)
         for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    Psi = strain_energy_density(*g, mu, lam, stress_state)
    w = torch.tensor(sol.mesh.areas, dtype=torch.float64)
    return float((Psi[0, :, 0] * w).sum())
