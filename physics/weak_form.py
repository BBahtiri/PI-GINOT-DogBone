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
from physics.equilibrium import equilibrium_residual
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
                 E_scale: float = 760.0,
                 w_trac_top: float = 0.0, w_trac_arc: float = 0.0,
                 S0: float = 100.0):
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
        # Traction-free residuals, redundant at the exact minimiser.
        #
        # Phase 2's argument was that these are *natural* BCs of the energy
        # form -- satisfied at the minimum rather than fought for by weights --
        # and that is true.  It is also insufficient.  A natural BC is only
        # free if the optimiser reaches the minimum, and the transverse
        # direction has 183x less curvature than the axial one
        # (docs/phase3_transverse_conditioning.md), so it does not.  These
        # terms cannot bias the converged answer, because they vanish there;
        # what they change is the approach, in the one direction the energy
        # barely constrains.  Default 0.0, so the energy form as measured in
        # Phase 2 is unchanged unless asked for.
        self.register_buffer("w_trac_top", torch.tensor(float(w_trac_top)))
        self.register_buffer("w_trac_arc", torch.tensor(float(w_trac_arc)))
        self.S0 = S0
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

    def quad_points(self, params: dict, device, dtype=None):
        """Quadrature points and weights for a geometry.

        The mesh is cached; the points are not, unless ``resample`` is off.

        ``dtype`` resolves to torch's default *at call time*.  It used to be
        ``dtype=torch.get_default_dtype()`` in the signature -- introduced in
        Phase 6.2 by a mechanical substitution for the hard-coded float32 --
        and a default argument is evaluated once, at import.  Every caller
        imports this module before any run sets its precision, so the points
        stayed float32 under float64, the model promoted them internally, and
        autograd handed the displacement gradients back at the leaf's dtype:
        float32.  The strain energy of the Phase 6.4 "fp64" arm was built
        from float32-rounded gradients.
        """
        if dtype is None:
            dtype = torch.get_default_dtype()
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
    def _traction_terms(self, model, geometry_latent, u_delta, x_max, y_max,
                        pts, normals, tags):
        """Mean squared nondimensional traction residual, split gauge-top/arc."""
        zero = torch.zeros((), device=u_delta.device if
                           torch.is_tensor(u_delta) else None)
        if pts is None or pts.shape[1] == 0:
            return zero, zero
        from physics.equilibrium import traction_residual_full
        tx, ty, _, _ = traction_residual_full(
            model, pts, normals, geometry_latent, u_delta, x_max,
            self.mu, self.lam, y_max, self.stress_state, return_J3D=True)
        res = ((tx / self.S0) ** 2 + (ty / self.S0) ** 2).squeeze(-1)
        top, arc = (tags == 0), (tags == 1)
        return (res[:, top].mean() if top.any() else zero,
                res[:, arc].mean() if arc.any() else zero)

    def forward(self, model, params: dict, boundary_pc, u_delta, x_max,
                y_max=None, sample_ids=None, geometry_latent=None,
                trac_free_pts=None, trac_free_normals=None,
                trac_free_tags=None, interior_pts=None) -> dict:
        """Energy loss for one geometry.

        ``params`` selects the cached quadrature mesh; ``boundary_pc`` is the
        encoder input, exactly as in PhysicsLoss.

        ``interior_pts`` is accepted and ignored here.  The energy uses its
        own quadrature mesh and has no use for the collocation points, but
        ``HybridLoss`` does, and the trainer threads them unconditionally for
        the same reason it threads the traction points: passing them only on
        the branch that needs them is how the branches drift apart.
        """
        if geometry_latent is None:
            # A model conditioned on geometry parameters rather than on the
            # boundary point cloud needs to know which geometry this is, and
            # encode()'s signature has no way to say.  The loss does know --
            # `params` selects the quadrature mesh -- so it tells the model
            # when the model asks to be told.  See
            # verification/supervised_ceiling.OracleConditioned.
            if hasattr(model, "set_geometry"):
                model.set_geometry(params)
            geometry_latent = model.encode(boundary_pc, x_max, y_max,
                                           sample_ids=sample_ids)

        Pi, J2D, J3D, area = self.energy(model, params, geometry_latent,
                                         u_delta, x_max, y_max)

        # Nondimensional: energy per unit volume, in units of E.
        L_energy = Pi / (self.E_scale * area)

        J_guard = torch.minimum(J2D, J3D)
        L_barrier = torch.mean(torch.relu(self.j_min - J_guard) ** 2)

        if self.w_trac_top > 0.0 or self.w_trac_arc > 0.0:
            L_trac_top, L_trac_arc = self._traction_terms(
                model, geometry_latent, u_delta, x_max, y_max,
                trac_free_pts, trac_free_normals, trac_free_tags)
        else:
            L_trac_top = L_trac_arc = torch.zeros((), device=L_energy.device)

        loss = (L_energy + self.w_bar * L_barrier
                + self.w_trac_top * L_trac_top + self.w_trac_arc * L_trac_arc)
        return {
            "loss": loss,
            "geometry_latent": geometry_latent,
            "L_energy": L_energy,
            "L_barrier": L_barrier,
            "L_trac_top": L_trac_top,
            "L_trac_arc": L_trac_arc,
            "L_trac_top_log": float(L_trac_top.detach()),
            "L_trac_arc_log": float(L_trac_arc.detach()),
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


class HybridLoss(EnergyLoss):
    """The energy functional with the equilibrium residual added back.

    Why this exists
    ---------------
    ``loss_form`` has always been ``energy`` XOR ``penalty``.  Phase 2
    compared them and the energy form won, and the equilibrium residual went
    out of the objective with the form that carried it.  Nobody ran the sum.

    Phase 6.4 made that gap worth closing.  Supervising this architecture on
    *stress* reaches a concentration factor of 0.39% and ``K_t`` R^2 +0.98;
    supervising it on *displacement* to 0.2% -- on the very specimens it is
    scored on, so with no generalisation, conditioning, capacity or
    optimisation gap left to blame -- still gives 3.15%, worse than
    predicting a constant, at R^2 -0.55.  Stress is a derivative of the
    output, and the concentration lives entirely in that derivative.  So the
    objective needs a term whose gradient is sensitive to the stress, and the
    equilibrium residual is the only such term the physics offers without
    changing the architecture.

    What is and is not added
    ------------------------
    Only ``L_eq``.  The traction conditions stay natural BCs of the energy
    minimum: Phase 2 measured that they work that way, and reintroducing them
    as penalties would bring back the weighting problem the energy form
    removed.  The determinant barrier is already in ``EnergyLoss``.

    The residual is nondimensionalised by ``L0 / S0`` exactly as in
    ``PhysicsLoss``, so ``w_eq`` means the same thing in both and a weight
    that worked there is a sensible starting point here.

    The caveat worth stating before the sweep
    -----------------------------------------
    ``L_eq`` is a volume average, like the energy.  A peak over a small
    region contributes to it in proportion to that region's share of the
    sample, so this may be just as blind to the concentration as the energy
    is.  If so, the fix is not a larger weight but a sampler that puts points
    where the residual is large -- which is Step 3, and this is the cheapest
    setting in which to test it.
    """

    def __init__(self, *args, w_eq: float = 1.0, L0: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("w_eq", torch.tensor(float(w_eq)))
        self.L0 = float(L0)

    def forward(self, model, params: dict, boundary_pc, u_delta, x_max,
                y_max=None, sample_ids=None, geometry_latent=None,
                trac_free_pts=None, trac_free_normals=None,
                trac_free_tags=None, interior_pts=None) -> dict:
        out = super().forward(
            model, params, boundary_pc, u_delta, x_max, y_max, sample_ids,
            geometry_latent, trac_free_pts, trac_free_normals,
            trac_free_tags)

        if interior_pts is None or float(self.w_eq) == 0.0:
            # Reported as zero rather than omitted, so a w_eq=0 arm produces
            # the same log keys as the others and is directly comparable to
            # the plain energy form it should reproduce.
            out["L_eq"] = torch.zeros((), device=out["loss"].device)
            out["L_eq_log"] = 0.0
            return out

        eq_scale = self.L0 / self.S0
        f_x, f_y, _, _ = equilibrium_residual(
            model, interior_pts, out["geometry_latent"], u_delta, x_max,
            self.mu, self.lam, y_max, self.stress_state, return_J3D=True)
        L_eq = torch.mean((eq_scale * f_x) ** 2 + (eq_scale * f_y) ** 2)

        out["loss"] = out["loss"] + self.w_eq * L_eq
        out["L_eq"] = L_eq
        out["L_eq_log"] = float(L_eq.detach())
        return out
