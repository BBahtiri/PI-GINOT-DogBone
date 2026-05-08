#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics-Informed Solution Decoder (Trunk network) for PI-GINOT.

Pipeline:
    Query points (x, y) [B, N_q, 2]   — physical coordinates
        → Normalise to [-1, 1]         — differentiable, preserves AD graph
        → NeRF positional encoding     — configurable max_deg (default 6)
        → Linear projection → Q embeddings [B, N_q, embed_dim]
        → For each layer i:
            Cross-Attention(Q, K=Z, V=Z)   — token-level geometry conditioning
            FiLM(Q, z_pool)                — global geometry modulation
        → Linear projection → raw (φ_u, φ_v) [B, N_q, 2]
        → Scale by output_scale        — keeps initial corrections small
        → HardBCLayer (uses physical coords) → (u, v) [B, N_q, 2]

FiLM conditioning:
    z_pool = mean(Z, dim=1)           — pooled geometry summary [B, D]
    gamma  = Linear(z_pool)           — per-feature scale
    beta   = Linear(z_pool)           — per-feature shift
    h      = gamma * h + beta         — affine modulation

    Initialised near identity (gamma≈1, beta≈0) so that the initial
    solution is unchanged from the hard-BC baseline.

Hard BC enforcement (normalised correction):
    ξ = x / L_half
    u(x, y) = u_δ · ξ  +  ξ · (1 − ξ) · φ_u(x, y)
    v(x, y) = (y / H_grip) · φ_v(x, y)

This guarantees exactly:
    u(0, y)       = 0      (vertical symmetry plane, x=0)
    u(L_half, y)  = u_δ    (right grip, prescribed displacement)
    v(x, 0)       = 0      (horizontal symmetry plane, y=0)

The correction terms ξ(1-ξ) and y/H_grip are O(1), preventing large
displacements at random initialization.
"""

import torch
import torch.nn as nn

from .modules.transformer import MLP, ResidualCrossAttentionBlock
from .modules.point_position_embedding import PosEmbLinear


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for geometry conditioning.

    Applies an affine transform h → gamma(z) * h + beta(z) where gamma
    and beta are learned functions of the pooled geometry latent.

    Initialised near identity (gamma≈1, beta≈0) so the initial solution
    is the unmodulated hard-BC baseline.

    Args:
        cond_dim:   Dimension of the conditioning vector (z_pool).
        hidden_dim: Dimension of the features to modulate.
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, hidden_dim)
        self.beta_proj = nn.Linear(cond_dim, hidden_dim)

        # Near-identity init: gamma ≈ 1, beta ≈ 0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, z_pool: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            h:      [B, N, D]  feature tensor to modulate.
            z_pool: [B, D]     pooled geometry conditioning vector.

        Returns:
            [B, N, D]  modulated features.
        """
        gamma = self.gamma_proj(z_pool).unsqueeze(1)  # [B, 1, D]
        beta = self.beta_proj(z_pool).unsqueeze(1)    # [B, 1, D]
        return gamma * h + beta


class PhysicsDecoder(nn.Module):
    """Geometry-conditioned displacement decoder with hard BCs.

    Uses both cross-attention (token-level) and FiLM (global) conditioning
    to ensure the decoder actively uses geometry information.

    Args:
        config: dict matching DECODER_CONFIG keys from config.py.
                Required keys: embed_dim, num_heads, cross_attn_layers,
                in_channels, out_channels.
                Optional keys: pe_max_deg (default 6), output_scale (default 1e-3).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        n_cross   = config["cross_attn_layers"]
        in_ch     = config["in_channels"]      # 2 for (x, y)
        out_ch    = config["out_channels"]      # 2 for (u, v)

        # Configurable PE frequency cap (default 6 → max freq 2^5 = 32)
        pe_max_deg = config.get("pe_max_deg", 6)

        # Output correction scale (keeps initial corrections small)
        self.output_scale = config.get("output_scale", 1e-3)

        # Query-point encoder: NeRF PE + projection → embed_dim
        self.q_posenc = PosEmbLinear("nerf", in_ch, embed_dim, max_deg=pe_max_deg)
        self.q_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Cross-attention stack: fuse geometry latent Z into queries
        self.cross_attn_blocks = nn.ModuleList([
            ResidualCrossAttentionBlock(width=embed_dim, heads=num_heads)
            for _ in range(n_cross)
        ])

        # FiLM layers: global geometry modulation after each cross-attention
        self.film_layers = nn.ModuleList([
            FiLMLayer(cond_dim=embed_dim, hidden_dim=embed_dim)
            for _ in range(n_cross)
        ])

        # Output projection: embed_dim → (φ_u, φ_v)
        # Tiny random init so the initial solution is exactly the hard-BC
        # baseline (u = u_δ·ξ, v = 0) with zero correction.
        self.output_proj = nn.Linear(embed_dim, out_ch)
        nn.init.normal_(self.output_proj.weight, std=1e-5)
        nn.init.zeros_(self.output_proj.bias)

        # Direct pooled-latent bias injection
        # Adds a geometry-conditioned shift to queries before cross-attention,
        # strengthening the decoder's dependence on the geometry latent.
        self.z_bias = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Near-zero init so initial behavior is unchanged
        nn.init.zeros_(self.z_bias[-1].weight)
        nn.init.zeros_(self.z_bias[-1].bias)

    def forward(
        self,
        query_pts: torch.Tensor,
        geometry_latent: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict displacement field at query points.

        Args:
            query_pts:        [B, N_q, 2]  physical (x, y) coordinates.
            geometry_latent:  [B, n_latent, embed_dim]  from GeometryEncoder.
            u_delta:          [B, 1] or scalar — prescribed displacement.
            x_max:            [B, 1] or scalar — specimen total length.
            y_max:            [B, 1] or scalar — specimen half-height (H_grip).
                              If None, defaults to x_max (isotropic scaling).

        Returns:
            uv: [B, N_q, 2]  displacement (u, v) with hard BCs enforced.
        """
        if y_max is None:
            y_max = x_max

        # Normalise physical coords to [-1, 1] for PE + attention
        query_norm = self._normalise_coords(query_pts, x_max, y_max)

        # Query encoding: NeRF PE + projection
        q = self.q_posenc(query_norm)                # [B, N_q, embed_dim]
        q = self.q_proj(q)                           # [B, N_q, embed_dim]

        # Pool geometry latent for FiLM conditioning
        z_pool = geometry_latent.mean(dim=1)         # [B, embed_dim]

        # Direct latent bias injection
        q = q + self.z_bias(z_pool).unsqueeze(1)     # [B, N_q, embed_dim]

        # Cross-attention + FiLM stack
        for block, film in zip(self.cross_attn_blocks, self.film_layers):
            q = block(q, geometry_latent)             # [B, N_q, embed_dim]
            q = film(q, z_pool)                       # [B, N_q, embed_dim]

        # Raw network output (scaled to keep initial corrections small)
        raw = self.output_scale * self.output_proj(q)  # [B, N_q, 2]

        # Hard Dirichlet BC enforcement (physical coords)
        uv = self._apply_hard_bc(raw, query_pts, u_delta, x_max, y_max)

        return uv

    def _normalise_coords(
        self,
        query_pts: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor,
    ) -> torch.Tensor:
        """Normalise physical (x, y) to [-1, 1] per axis.

        Uses x_max for x and y_max for y so both axes span the full
        [-1, 1] range.  Differentiable linear map — AD through it is exact.
        """
        if x_max.dim() == 1:
            x_max = x_max.unsqueeze(-1)   # [B, 1]
        x_max = x_max.unsqueeze(-1)       # [B, 1, 1]

        if y_max.dim() == 1:
            y_max = y_max.unsqueeze(-1)
        y_max = y_max.unsqueeze(-1)       # [B, 1, 1]

        x = query_pts[..., 0:1]
        y = query_pts[..., 1:2]

        x_norm = 2.0 * x / x_max - 1.0   # [0, x_max] → [-1, 1]
        y_norm = 2.0 * y / y_max - 1.0   # [0, y_max] → [-1, 1]

        return torch.cat([x_norm, y_norm], dim=-1)

    def _apply_hard_bc(
        self,
        raw: torch.Tensor,
        query_pts: torch.Tensor,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor,
    ) -> torch.Tensor:
        """Apply hard Dirichlet BCs via normalised distance functions.

        Using normalised coordinates:
            ξ = x / L_half,   η = y / H_grip

            u = u_δ · ξ + ξ · (1 − ξ) · φ_u
            v = η · φ_v

        The v-branch uses y_max (= H_grip) so that the transverse
        displacement uses the full [0, 1] range, giving Poisson
        contraction the same learning capacity as axial extension.

        Guarantees:
            u(0, y)       = 0      exactly  (vertical symmetry)
            u(L_half, y)  = u_δ    exactly  (right grip)
            v(x, 0)       = 0      exactly  (horizontal symmetry)
        """
        x = query_pts[..., 0:1]     # [B, N_q, 1]
        y = query_pts[..., 1:2]     # [B, N_q, 1]
        phi_u = raw[..., 0:1]       # [B, N_q, 1]
        phi_v = raw[..., 1:2]       # [B, N_q, 1]

        if u_delta.dim() == 1:
            u_delta = u_delta.unsqueeze(-1)
        if x_max.dim() == 1:
            x_max = x_max.unsqueeze(-1)
        if y_max.dim() == 1:
            y_max = y_max.unsqueeze(-1)
        u_delta = u_delta.unsqueeze(-1)          # [B, 1, 1]
        x_max = x_max.unsqueeze(-1)              # [B, 1, 1]
        y_max = y_max.unsqueeze(-1)              # [B, 1, 1]

        # Normalised coordinates
        xi = x / x_max                # ξ ∈ [0, 1]
        eta = y / y_max               # η ∈ [0, 1] (uses H_grip, not L)

        # u = u_δ·ξ + ξ·(1-ξ)·φ_u   → correction ≤ 0.25·|φ_u|
        u = u_delta * xi + xi * (1.0 - xi) * phi_u
        # v = η · φ_v                → correction ≤ 1.0·|φ_v|
        v = eta * phi_v

        return torch.cat([u, v], dim=-1)         # [B, N_q, 2]

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
