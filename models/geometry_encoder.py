#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry Encoder (Branch network) for PI-GINOT.

Wraps GINOT's PointCloudPerceiverChannelsEncoder with config-driven
hyperparameters.  Takes 2D boundary point clouds and produces latent
geometry tokens consumed by the physics decoder via cross-attention.

Pipeline:
    Boundary PC [B, N_bnd, 2]
        → NeRF positional encoding + linear projection
        → FPS sampling + ball-query grouping  (PointSetEmbedding)
        → Cross-attention with full positional embeddings
        → Self-attention stack
        → Output projection + tanh
    Geometry Latent Z [B, n_point, out_c]
"""

import torch
import torch.nn as nn

from .modules.point_encoding import PointCloudPerceiverChannelsEncoder


class GeometryEncoder(nn.Module):
    """Thin config-driven wrapper around PointCloudPerceiverChannelsEncoder.

    Args:
        config: dict matching ENCODER_CONFIG keys from config.py.
                Required keys: input_channels, out_c, width, n_point,
                n_sample, radius, d_hidden, num_heads, cross_attn_layers,
                self_attn_layers, fps_method, dropout.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.encoder = PointCloudPerceiverChannelsEncoder(
            input_channels=config["input_channels"],
            out_c=config["out_c"],
            width=config["width"],
            n_point=config["n_point"],
            n_sample=config["n_sample"],
            radius=config["radius"],
            d_hidden=list(config["d_hidden"]),
            num_heads=config["num_heads"],
            cross_attn_layers=config["cross_attn_layers"],
            self_attn_layers=config["self_attn_layers"],
            fps_method=config["fps_method"],
            dropout=config.get("dropout", 0.0),
        )

    def forward(self, boundary_pc: torch.Tensor,
               sample_ids: torch.Tensor = None) -> torch.Tensor:
        """Encode a batch of 2D boundary point clouds into geometry latents.

        Args:
            boundary_pc:  [B, N_bnd, 2]  boundary point cloud coordinates
                          (should be normalised to roughly [-1, 1]).
            sample_ids:   [B] optional integer IDs for deterministic FPS
                          caching. When provided, FPS is deterministic
                          and indices are cached by ID.

        Returns:
            latent: [B, n_point, out_c]  geometry latent tokens.
        """
        return self.encoder(boundary_pc, sample_ids=sample_ids)

    @property
    def latent_dim(self) -> int:
        """Dimension of each latent token (= out_c)."""
        return self.config["out_c"]

    @property
    def n_latent_tokens(self) -> int:
        """Number of latent tokens output (= n_point)."""
        return self.config["n_point"]

    def count_parameters(self) -> dict:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
