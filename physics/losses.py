#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-informed loss functions for PI-GINOT.

Assembles the complete loss from:
  1. Equilibrium residual:   Div(P) = 0       on interior
  2. Full traction-free BC:  P · N = 0        on arcs, gauge top, holes
  3. Partial traction BC:    (P·N)_i = 0      on grips, symmetry line
  4. detF barrier:           ReLU(J_min - J)² to prevent inverted elements

Uses encode-once pattern: geometry latent is computed once and reused
for all query point sets per geometry.

Includes gradient-based adaptive loss weighting.

Nondimensionalization:
  Equilibrium residual is scaled by (L0 / S0) so that Div(P)·L0/S0 is O(1).
  Traction residual is scaled by (1 / S0) so that P·N/S0 is O(1).
  This improves conditioning by removing the raw MPa magnitudes.
"""

import torch
import torch.nn as nn

from .equilibrium import (
    equilibrium_residual,
    traction_residual_full,
    traction_residual_partial,
)


class PhysicsLoss(nn.Module):
    """Combined physics loss for PI-GINOT training.

    Loss = w_eq·L_eq + w_top·L_top + w_arc·L_arc + w_part·L_part + w_bar·L_barrier

    Args:
        mu, lam:            Lamé parameters.
        w_equilibrium:      Initial weight for equilibrium loss.
        w_trac_top:         Weight for gauge-top traction loss.
        w_trac_arc:         Weight for fillet-arc traction loss (> w_top).
        w_traction_partial: Initial weight for partial traction loss.
        w_barrier:          Weight for detF barrier loss.
        j_min:              Minimum allowed J (det F) for barrier.
        adaptive_beta:      EMA blending factor for adaptive weights.
        L0:                 Characteristic length for nondimensionalization.
        S0:                 Characteristic stress for nondimensionalization.
        stress_state:       'plane strain' or 'plane stress'.
    """

    def __init__(
        self,
        mu: float,
        lam: float,
        w_equilibrium: float = 1.0,
        w_trac_top: float = 10.0,
        w_trac_arc: float = 50.0,
        w_traction_partial: float = 10.0,
        w_barrier: float = 1e3,
        j_min: float = 0.1,
        adaptive_beta: float = 0.1,
        L0: float = 50.0,
        S0: float = 760.0,
        stress_state: str = "plane strain",
    ):
        super().__init__()
        self.mu = mu
        self.lam = lam
        self.j_min = j_min
        self.stress_state = stress_state

        # Nondimensionalization scales
        self.L0 = L0
        self.S0 = S0

        # Loss weights (updated by adaptive scheme)
        self.register_buffer("w_eq", torch.tensor(w_equilibrium))
        self.register_buffer("w_trac_top", torch.tensor(w_trac_top))
        self.register_buffer("w_trac_arc", torch.tensor(w_trac_arc))
        self.register_buffer("w_part", torch.tensor(w_traction_partial))
        self.register_buffer("w_bar", torch.tensor(w_barrier))
        self.beta = adaptive_beta

    def forward(
        self,
        model: nn.Module,
        interior_pts: torch.Tensor,
        trac_free_pts: torch.Tensor,
        trac_free_normals: torch.Tensor,
        trac_free_tags: torch.Tensor,
        partial_trac_pts: torch.Tensor,
        partial_trac_normals: torch.Tensor,
        partial_trac_dirs: torch.Tensor,
        boundary_pc: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
        sample_ids: torch.Tensor = None,
    ) -> dict:
        """Compute all physics loss components.

        Uses encode-once: boundary_pc → geometry_latent, reused for
        interior, traction-free, and partial-traction queries.

        The detF barrier covers both interior and traction-free boundary
        points, ensuring physical admissibility near holes and fillets.

        Returns dict with 'loss' (total), individual losses (graph-attached
        for adaptive weighting), and detached copies for logging.
        """
        # Encode geometry once
        geometry_latent = model.encode(boundary_pc, x_max, y_max,
                                      sample_ids=sample_ids)

        # Nondimensionalization factors
        eq_scale = self.L0 / self.S0      # Scale for Div(P): [MPa/mm] → O(1)
        trac_scale = 1.0 / self.S0        # Scale for P·N:    [MPa]   → O(1)

        # 1. Equilibrium: Div(P) = 0
        f_x, f_y, detF_int = equilibrium_residual(
            model, interior_pts, geometry_latent, u_delta, x_max,
            self.mu, self.lam, y_max, self.stress_state,
        )
        L_eq = torch.mean((eq_scale * f_x)**2 + (eq_scale * f_y)**2)

        # 2. Full traction-free BC: P · N = 0 (split by segment)
        detF_bnd = None
        _zero = torch.tensor(0.0, device=interior_pts.device)
        if trac_free_pts.shape[1] > 0:
            trac_x, trac_y, detF_bnd = traction_residual_full(
                model, trac_free_pts, trac_free_normals,
                geometry_latent, u_delta, x_max, self.mu, self.lam,
                y_max, self.stress_state,
            )
            residual = (trac_scale * trac_x)**2 + (trac_scale * trac_y)**2
            # [B, N, 1] — squeeze last dim for masking
            res_flat = residual.squeeze(-1)  # [B, N]

            # Split by segment tag (0=gauge_top, 1=arc, 2=hole)
            mask_top  = (trac_free_tags == 0)  # [N]
            mask_arc  = (trac_free_tags == 1)
            mask_hole = (trac_free_tags == 2)

            L_trac_top  = res_flat[:, mask_top].mean()  if mask_top.any()  else _zero
            L_trac_arc  = res_flat[:, mask_arc].mean()  if mask_arc.any()  else _zero
            L_trac_hole = res_flat[:, mask_hole].mean() if mask_hole.any() else _zero
        else:
            L_trac_top  = _zero
            L_trac_arc  = _zero
            L_trac_hole = _zero

        # 3. Partial traction BC
        if partial_trac_pts.shape[1] > 0:
            trac_partial = traction_residual_partial(
                model, partial_trac_pts, partial_trac_normals,
                partial_trac_dirs,
                geometry_latent, u_delta, x_max, self.mu, self.lam,
                y_max, self.stress_state,
            )
            L_part = torch.mean((trac_scale * trac_partial)**2)
        else:
            L_part = torch.tensor(0.0, device=interior_pts.device)

        # 4. detF barrier on interior + traction-free boundary
        detF_all = [detF_int]
        if detF_bnd is not None:
            detF_all.append(detF_bnd)
        detF_cat = torch.cat(detF_all, dim=1)
        L_barrier = torch.mean(torch.relu(self.j_min - detF_cat) ** 2)

        # Weighted total (arcs weighted higher than flat top)
        loss = (self.w_eq * L_eq
                + self.w_trac_top * L_trac_top
                + self.w_trac_arc * (L_trac_arc + L_trac_hole)
                + self.w_part * L_part
                + self.w_bar * L_barrier)

        return {
            "loss": loss,
            # Graph-attached (for adaptive weighting / downstream use)
            "geometry_latent": geometry_latent,
            "L_eq": L_eq,
            "L_trac_top": L_trac_top,
            "L_trac_arc": L_trac_arc,
            "L_trac_hole": L_trac_hole,
            "L_part": L_part,
            "L_barrier": L_barrier,
            # Detached scalars for logging
            "L_eq_log": L_eq.item(),
            "L_trac_top_log": L_trac_top.item(),
            "L_trac_arc_log": L_trac_arc.item(),
            "L_trac_hole_log": L_trac_hole.item(),
            "L_part_log": L_part.item(),
            "L_barrier_log": L_barrier.item(),
            "w_eq": self.w_eq.item(),
            "w_trac_top": self.w_trac_top.item(),
            "w_trac_arc": self.w_trac_arc.item(),
            "w_part": self.w_part.item(),
        }

    @torch.no_grad()
    def update_adaptive_weights(
        self,
        model: nn.Module,
        interior_pts: torch.Tensor,
        trac_free_pts: torch.Tensor,
        trac_free_normals: torch.Tensor,
        trac_free_tags: torch.Tensor,
        partial_trac_pts: torch.Tensor,
        partial_trac_normals: torch.Tensor,
        partial_trac_dirs: torch.Tensor,
        boundary_pc: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
        sample_ids: torch.Tensor = None,
    ):
        """Update loss weights via gradient-based adaptive balancing.

        Computes max|∇L_eq| / mean|∇L_trac| w.r.t. model parameters,
        then EMA-blends into current weights.
        """
        with torch.enable_grad():
            loss_dict = self.forward(
                model, interior_pts, trac_free_pts, trac_free_normals,
                trac_free_tags,
                partial_trac_pts, partial_trac_normals, partial_trac_dirs,
                boundary_pc, u_delta, x_max, y_max, sample_ids,
            )

            grads_eq = torch.autograd.grad(
                loss_dict["L_eq"], model.parameters(),
                retain_graph=True, allow_unused=True,
            )
            max_grad_eq = max(
                g.abs().max().item() for g in grads_eq if g is not None
            )

            L_trac_total = loss_dict["L_trac_arc"] + loss_dict["L_trac_top"] + loss_dict["L_part"]
            if L_trac_total.item() > 0:
                grads_trac = torch.autograd.grad(
                    L_trac_total, model.parameters(),
                    retain_graph=False, allow_unused=True,
                )
                mean_grad_trac = torch.stack([
                    g.abs().mean() for g in grads_trac if g is not None
                ]).mean().item()

                if mean_grad_trac > 1e-12:
                    adaptive_w = max_grad_eq / mean_grad_trac
                    # Safety clamp: prevent runaway when traction grads are tiny
                    adaptive_w = max(1.0, min(adaptive_w, 100.0))
                    new_w = (1 - self.beta) * self.w_trac_arc.item() + self.beta * adaptive_w
                    self.w_trac_arc.fill_(new_w)
                    self.w_trac_top.fill_(new_w / 5.0)  # keep 5:1 arc:top ratio
                    self.w_part.fill_(new_w)
