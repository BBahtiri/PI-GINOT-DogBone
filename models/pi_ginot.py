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



class PI_GINOT(nn.Module):
    """Full PI-GINOT model: geometry encoder + physics decoder.

    Args:
        encoder_config: dict for GeometryEncoder (ENCODER_CONFIG).
        decoder_config: dict for PhysicsDecoder (DECODER_CONFIG).
    """

    # Legacy state_dict prefixes that no longer exist on this model.  Kept so
    # checkpoints trained before the head was removed still load.
    LEGACY_PREFIXES = ("geom_aux_head.",)

    @classmethod
    def strip_legacy_keys(cls, state_dict: dict) -> tuple:
        """Drop parameters belonging to removed submodules.

        Returns (clean_state_dict, dropped_key_names).
        """
        dropped = [k for k in state_dict
                   if k.startswith(cls.LEGACY_PREFIXES)]
        if not dropped:
            return state_dict, []
        clean = {k: v for k, v in state_dict.items() if k not in dropped}
        return clean, dropped

    def __init__(self, encoder_config: dict, decoder_config: dict,
                 aux_head: bool = False):
        super().__init__()

        assert encoder_config["out_c"] == decoder_config["embed_dim"], (
            f"Encoder out_c ({encoder_config['out_c']}) must match "
            f"decoder embed_dim ({decoder_config['embed_dim']})"
        )

        self.encoder = GeometryEncoder(encoder_config)
        self.decoder = PhysicsDecoder(decoder_config)

        # Geometry auxiliary head, built only when asked so that checkpoints
        # without it load unchanged.  It predicts the four dimensionless shape
        # descriptors (geometry/descriptors.py) from the pooled latent -- the
        # pooled vector because that is what FiLM conditioning consumes and
        # what the probe finds most decodable, and dimensionless because the
        # encoder's input is scale-normalised and cannot determine millimetres.
        #
        # The point is not to add information.  It is to give the encoder a
        # direct gradient for extracting geometry, which the physics loss
        # cannot supply: the objective is 183x less sensitive to the transverse
        # field than the axial one, so the reward for learning to read a fillet
        # out of a point cloud arrives through the thinnest channel in the
        # problem.  See docs/phase4_geometry_encoder.md.
        self.geom_aux_head = None
        if aux_head:
            d = encoder_config["out_c"]
            self.geom_aux_head = nn.Sequential(
                nn.Linear(d, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU(),
                nn.Linear(128, 4),
            )

    def predict_geometry(self, geometry_latent: torch.Tensor):
        """Standardised dimensionless shape descriptors from the latent."""
        if self.geom_aux_head is None:
            raise RuntimeError("model was built without aux_head=True")
        return self.geom_aux_head(geometry_latent.mean(dim=1))

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
