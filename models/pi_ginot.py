#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Informed Geometry-Informed Neural Operator Transformer (PI-GINOT).

Top-level model that composes:
  1. GeometryEncoder  — encodes boundary point cloud → latent geometry tokens
  2. PhysicsDecoder   — decodes query points + geometry latent → displacement

Supports an encode-once pattern: call encode() once per geometry, then
decode() / predict_with_grad_latent() multiple times with different
query point sets (interior, traction, partial) for efficiency.
"""

import torch
import torch.nn as nn

from .geometry_encoder import GeometryEncoder
from .physics_decoder import PhysicsDecoder



class GeometryAuxHead(nn.Module):
    """Predict normalised geometry parameters from pooled latent.

    Outputs 8 values: L_total, W_grip, W_gauge, R_fillet,
    hole_present (sigmoid), cx, cy, r.
    For no-hole cases, cx/cy/r regression loss is masked.
    """

    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Args: z [B, n_tok, dim]. Returns: [B, d_out]."""
        z_pool = z.mean(dim=1)
        return self.net(z_pool)


class PI_GINOT(nn.Module):
    """Full PI-GINOT model: geometry encoder + physics decoder.

    Args:
        encoder_config: dict for GeometryEncoder (ENCODER_CONFIG).
        decoder_config: dict for PhysicsDecoder (DECODER_CONFIG).
    """

    def __init__(self, encoder_config: dict, decoder_config: dict):
        super().__init__()

        assert encoder_config["out_c"] == decoder_config["embed_dim"], (
            f"Encoder out_c ({encoder_config['out_c']}) must match "
            f"decoder embed_dim ({decoder_config['embed_dim']})"
        )

        self.encoder = GeometryEncoder(encoder_config)
        self.decoder = PhysicsDecoder(decoder_config)

        # Geometry auxiliary head (latent supervision)
        self.geom_aux_head = GeometryAuxHead(
            d_in=encoder_config["out_c"],
            d_hidden=128,
            d_out=8,
        )

    # Encode-once pattern
    def encode(
        self,
        boundary_pc: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
        sample_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Encode boundary PC → geometry latent (call once per geometry).

        Normalises boundary_pc to [-1, 1] before encoding so that the
        encoder's ball-query radius operates in a canonical scale.

        Args:
            boundary_pc: [B, N_bnd, 2] raw physical coordinates.
            x_max:       [B] or [B, 1] specimen length for normalisation.
            y_max:       [B] or [B, 1] specimen half-height.

        Returns:
            geometry_latent: [B, n_latent, embed_dim].
        """
        bpc_norm = self._normalise_pc(boundary_pc, x_max, y_max)
        return self.encoder(bpc_norm, sample_ids=sample_ids)

    def decode(
        self,
        query_pts: torch.Tensor,
        geometry_latent: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode with pre-computed geometry latent.

        Args:
            query_pts:        [B, N_q, 2] physical coords.
            geometry_latent:  [B, n_latent, embed_dim].
            u_delta, x_max:   [B] or [B, 1].
            y_max:            [B] or [B, 1] specimen half-height.

        Returns:
            uv: [B, N_q, 2] displacement with hard BCs.
        """
        return self.decoder(query_pts, geometry_latent, u_delta, x_max, y_max)

    # Full forward (encode + decode in one call)
    def forward(
        self,
        query_pts: torch.Tensor,
        boundary_pc: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
    ) -> torch.Tensor:
        """Full forward pass: normalise + encode + decode."""
        geometry_latent = self.encode(boundary_pc, x_max, y_max)
        return self.decode(query_pts, geometry_latent, u_delta, x_max, y_max)

    # AD-enabled prediction (for physics loss)
    def predict_with_grad(
        self,
        query_pts: torch.Tensor,
        boundary_pc: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
    ) -> tuple:
        """Forward + displacement gradients (encodes boundary_pc)."""
        geometry_latent = self.encode(boundary_pc, x_max, y_max)
        return self.predict_with_grad_latent(
            query_pts, geometry_latent, u_delta, x_max, y_max
        )

    def predict_with_grad_latent(
        self,
        query_pts: torch.Tensor,
        geometry_latent: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
    ) -> tuple:
        """Forward + displacement gradients with pre-computed latent.

        Returns:
            uv, du_dx, du_dy, dv_dx, dv_dy: each appropriately shaped.
        """
        assert query_pts.requires_grad, (
            "query_pts must have requires_grad=True for AD."
        )

        uv = self.decode(query_pts, geometry_latent, u_delta, x_max, y_max)
        u = uv[..., 0:1]
        v = uv[..., 1:2]

        ones = torch.ones_like(u)

        du_dxy = torch.autograd.grad(
            u, query_pts, grad_outputs=ones,
            create_graph=True, retain_graph=True,
        )[0]

        dv_dxy = torch.autograd.grad(
            v, query_pts, grad_outputs=ones,
            create_graph=True, retain_graph=True,
        )[0]

        du_dx = du_dxy[..., 0:1]
        du_dy = du_dxy[..., 1:2]
        dv_dx = dv_dxy[..., 0:1]
        dv_dy = dv_dxy[..., 1:2]

        return uv, du_dx, du_dy, dv_dx, dv_dy

    def _normalise_pc(
        self,
        boundary_pc: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
    ) -> torch.Tensor:
        """Normalise boundary PC from physical coords to [-1, 1] per axis.

        Uses x_max for x-axis and y_max for y-axis so both span [-1, 1].
        Consistent with the decoder's coordinate normalisation.
        """
        if x_max.dim() == 1:
            x_max = x_max.unsqueeze(-1)  # [B, 1]
        x_max = x_max.unsqueeze(-1)      # [B, 1, 1]

        if y_max is None:
            y_scale = x_max
        else:
            if y_max.dim() == 1:
                y_max = y_max.unsqueeze(-1)
            y_scale = y_max.unsqueeze(-1)  # [B, 1, 1]

        x = boundary_pc[..., 0:1]
        y = boundary_pc[..., 1:2]
        x_norm = 2.0 * x / x_max - 1.0
        y_norm = 2.0 * y / y_scale - 1.0

        return torch.cat([x_norm, y_norm], dim=-1)

    def predict_geometry(self, geometry_latent: torch.Tensor) -> torch.Tensor:
        """Predict geometry parameters from latent. Returns [B, 8]."""
        return self.geom_aux_head(geometry_latent)

    def count_parameters(self) -> dict:
        enc = self.encoder.count_parameters()
        dec = self.decoder.count_parameters()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "encoder": enc,
            "decoder": dec,
            "total": total,
            "trainable": trainable,
        }
