#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reliability diagnostics for PI-GINOT predictions.

Core diagnostics:
  1. Normalized equilibrium residual (Div(P) = 0) on independent grid
  2. Normalized traction residual (P·N = 0) split by segment tag
  3. Section-force CV on disjoint verification slices
  4. Latent-swap sensitivity against fixed reference latents
  5. Hard-BC baseline collapse detector (on full verification grid)
  6. Geometry distance to training bank (NN + Mahalanobis)

All residual normalization uses INPUT-ONLY scales (§2.1):
    stress_scale = E * u_delta / L_half
    length_scale = L_half
Never uses predicted stress values for normalization.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .constants import AGENT_SEEDS, VERIFICATION_SECTION_XI
from .schemas import (
    NormalizationScales,
    ReliabilityMetrics,
    GeometryParams,
)
from .verification import (
    VerificationGrid,
    _section_width,
)


# ---------------------------------------------------------------------------
# 1. Normalized equilibrium residual
# ---------------------------------------------------------------------------

def compute_equilibrium_residual(
    model: nn.Module,
    query_pts: torch.Tensor,       # [1, N, 2] requires_grad=True
    geometry_latent: torch.Tensor,  # [1, n_tok, dim]
    u_delta: torch.Tensor,         # [1]
    x_max: torch.Tensor,           # [1]
    y_max: torch.Tensor,           # [1]
    mu: float,
    lam: float,
    stress_state: str,
    scales: NormalizationScales,
) -> Dict[str, float]:
    """Compute normalized Div(P) residual on verification points.

    Returns dict with:
        mean_normalized: mean |Div(P)|^2 * (L/S)^2
        max_normalized:  max |Div(P)|^2 * (L/S)^2
        frac_detF_neg:   fraction of points with detF <= 0
        frac_detF_low:   fraction of points with detF < 0.1
        detF_values:     raw detF numpy array
    """
    from physics.neo_hookean import first_piola_kirchhoff_stress

    assert query_pts.requires_grad

    uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
        query_pts, geometry_latent, u_delta, x_max, y_max
    )

    P11, P12, P21, P22, detF = first_piola_kirchhoff_stress(
        du_dx, du_dy, dv_dx, dv_dy, mu, lam, stress_state
    )

    # Div(P) = (dP11/dx + dP12/dy, dP21/dx + dP22/dy)
    ones = torch.ones_like(P11)

    dP11_dxy = torch.autograd.grad(P11, query_pts, grad_outputs=ones,
                                    create_graph=False, retain_graph=True)[0]
    dP12_dxy = torch.autograd.grad(P12, query_pts, grad_outputs=ones,
                                    create_graph=False, retain_graph=True)[0]
    dP21_dxy = torch.autograd.grad(P21, query_pts, grad_outputs=ones,
                                    create_graph=False, retain_graph=True)[0]
    dP22_dxy = torch.autograd.grad(P22, query_pts, grad_outputs=ones,
                                    create_graph=False, retain_graph=False)[0]

    f_x = dP11_dxy[..., 0:1] + dP12_dxy[..., 1:2]  # [1, N, 1]
    f_y = dP21_dxy[..., 0:1] + dP22_dxy[..., 1:2]

    # Normalize: multiply by (length_scale / stress_scale)
    norm_factor = scales.length_scale / scales.stress_scale
    f_x_norm = f_x * norm_factor
    f_y_norm = f_y * norm_factor

    residual_sq = (f_x_norm ** 2 + f_y_norm ** 2).squeeze(-1)  # [1, N]
    mean_res = residual_sq.mean().item()
    max_res = residual_sq.max().item()

    # detF statistics
    detF_np = detF.squeeze(-1).detach().cpu().numpy().flatten()
    frac_neg = float(np.mean(detF_np <= 0))
    frac_low = float(np.mean(detF_np < 0.1))

    return {
        "mean_normalized": mean_res,
        "max_normalized": max_res,
        "frac_detF_neg": frac_neg,
        "frac_detF_low": frac_low,
        "detF_values": detF_np,
        "uv": uv.detach(),
    }


# ---------------------------------------------------------------------------
# 2. Normalized traction residual — split by segment tag
# ---------------------------------------------------------------------------

def compute_traction_residual_by_tag(
    model: nn.Module,
    trac_pts: torch.Tensor,         # [1, N, 2] requires_grad=True
    trac_normals: torch.Tensor,     # [1, N, 2]
    trac_tags: torch.Tensor,        # [N] long: 0=gauge_top, 1=arc, 2=hole
    geometry_latent: torch.Tensor,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    y_max: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str,
    scales: NormalizationScales,
) -> Dict[str, float]:
    """Compute normalized P·N traction residual, split by boundary segment.

    Returns dict with per-segment mean normalized |P·N|^2 / stress_scale^2.
    Keys: gauge_top, arc, hole (may be 0.0 if segment has no points).
    """
    from physics.equilibrium import boundary_piola

    P11, P12, P21, P22, _ = boundary_piola(
        model, trac_pts, geometry_latent, u_delta, x_max,
        mu, lam, y_max, stress_state,
    )

    N_x = trac_normals[..., 0:1]
    N_y = trac_normals[..., 1:2]

    # P · N
    trac_x = P11 * N_x + P12 * N_y
    trac_y = P21 * N_x + P22 * N_y

    trac_scale = 1.0 / scales.stress_scale
    residual = (trac_x * trac_scale) ** 2 + (trac_y * trac_scale) ** 2
    res_flat = residual.squeeze(-1)  # [1, N]

    # Split by tag
    result = {}
    tag_names = {0: "gauge_top", 1: "arc", 2: "hole"}
    for tag_val, tag_name in tag_names.items():
        mask = trac_tags == tag_val
        if mask.any():
            result[tag_name] = res_flat[:, mask].mean().item()
        else:
            result[tag_name] = 0.0

    # Also provide the pooled mean for backward compat
    result["mean_normalized"] = res_flat.mean().item()

    return result


def compute_partial_traction_residual(
    model: nn.Module,
    pt_pts: torch.Tensor,           # [1, N, 2] requires_grad=True
    pt_normals: torch.Tensor,       # [1, N, 2]
    pt_dirs: torch.Tensor,          # [N] long: 0=x-dir, 1=y-dir
    geometry_latent: torch.Tensor,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    y_max: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str,
    scales: NormalizationScales,
) -> float:
    """Compute normalized partial traction residual.

    For symmetry/grip boundaries where only one component of P·N = 0.
    Returns mean normalized residual.
    """
    from physics.equilibrium import boundary_piola

    P11, P12, P21, P22, _ = boundary_piola(
        model, pt_pts, geometry_latent, u_delta, x_max,
        mu, lam, y_max, stress_state,
    )

    N_x = pt_normals[..., 0:1]
    N_y = pt_normals[..., 1:2]

    trac_x = P11 * N_x + P12 * N_y  # [1, N, 1]
    trac_y = P21 * N_x + P22 * N_y

    trac_scale = 1.0 / scales.stress_scale

    # Select the constrained component per point
    # dirs=0 → x-component constrained, dirs=1 → y-component
    x_mask = (pt_dirs == 0).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
    selected = torch.where(x_mask, trac_x, trac_y)
    residual = (selected * trac_scale) ** 2

    return residual.mean().item()


# ---------------------------------------------------------------------------
# 3. Section-force CV on verification slices
# ---------------------------------------------------------------------------

def compute_section_force_cv(
    model: nn.Module,
    geometry_latent: torch.Tensor,
    fillet_info: dict,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    y_max: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str,
    device: torch.device,
    section_x_positions: Optional[np.ndarray] = None,
    n_y_pts: int = 128,
) -> Tuple[float, List[float], float]:
    """Compute section-force coefficient of variation.

    Uses VERIFICATION slice positions (disjoint from training).

    Returns:
        cv: coefficient of variation of axial resultants
        resultants: list of N(x) values at each slice
    """
    from physics.neo_hookean import first_piola_kirchhoff_stress

    L_half = fillet_info["L_half"]

    if section_x_positions is None:
        section_x_positions = np.array(VERIFICATION_SECTION_XI) * L_half

    resultants = []
    for x_k in section_x_positions:
        w_k = _section_width(float(x_k), fillet_info)
        y_vals = np.linspace(0.01 * w_k, 0.99 * w_k, n_y_pts)
        dy = w_k / n_y_pts

        pts = np.stack([np.full(n_y_pts, float(x_k)), y_vals], axis=-1)
        query = torch.tensor(pts, dtype=torch.float32,
                             device=device).unsqueeze(0).requires_grad_(True)

        with torch.enable_grad():
            uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
                query, geometry_latent, u_delta, x_max, y_max
            )
            P11, _, _, _, detF = first_piola_kirchhoff_stress(
                du_dx, du_dy, dv_dx, dv_dy, mu, lam, stress_state
            )

        # Axial resultant N(x) = 2 * dy * sum(P11)  (quarter → half symmetry)
        N_k = 2.0 * dy * P11[0, :, 0].sum().item()
        resultants.append(N_k)

    N_arr = np.array(resultants)
    N_mean = float(np.mean(N_arr))
    cv = float(np.std(N_arr) / (np.abs(N_mean) + 1e-12))

    return cv, resultants, N_mean


# ---------------------------------------------------------------------------
# 4. Latent-swap sensitivity against fixed references
# ---------------------------------------------------------------------------

def compute_swap_sensitivity(
    model: nn.Module,
    geometry_latent: torch.Tensor,   # [1, n_tok, dim]
    reference_latents: List[torch.Tensor],  # list of [1, n_tok, dim]
    query_pts: torch.Tensor,         # [1, N, 2]
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    y_max: torch.Tensor,
    x_max_physical: float,
    u_max: float,
) -> float:
    """Compute latent-swap sensitivity against a fixed reference set.

    Uses pre-encoded reference latents (from training bank, selected
    by greedy farthest-point) for consistent comparison.

    Returns:
        mean_swap_du: mean relative change in correction field when
                      swapping geometry latent for each reference.
    """
    # Prediction with correct latent
    uv_correct = model.decode(query_pts, geometry_latent, u_delta, x_max, y_max)
    u_correct = uv_correct[0, :, 0].detach().cpu().numpy()

    # Hard-BC baseline
    pts_np = query_pts[0].detach().cpu().numpy()
    u_base = u_max * pts_np[:, 0] / x_max_physical
    du_correct = u_correct - u_base
    norm_du = max(np.linalg.norm(du_correct), 1e-12)

    swap_vals = []
    for ref_z in reference_latents:
        uv_swap = model.decode(query_pts, ref_z, u_delta, x_max, y_max)
        u_swap = uv_swap[0, :, 0].detach().cpu().numpy()
        du_swap = u_swap - u_base
        swap_vals.append(np.linalg.norm(du_correct - du_swap) / norm_du)

    return float(np.mean(swap_vals)) if swap_vals else 0.0


# ---------------------------------------------------------------------------
# 5. Hard-BC baseline collapse detector (§2.4)
# ---------------------------------------------------------------------------

def detect_baseline_collapse(
    model: nn.Module,
    query_pts: torch.Tensor,         # [1, N, 2]
    geometry_latent: torch.Tensor,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    y_max: torch.Tensor,
    x_max_physical: float,
    u_max: float,
    threshold: float = 0.01,
) -> Tuple[float, bool]:
    """Detect if the decoder has collapsed to the hard-BC baseline.

    If the decoder's correction term vanishes, u ≈ u_δ * x / L_half
    everywhere — a uniform-strain solution. This satisfies equilibrium
    nearly exactly, but misses fillet concentration.

    Returns:
        correction_magnitude: ||u_pred - u_baseline|| / ||u_baseline||
        is_collapsed: True if correction_magnitude < threshold
    """
    uv = model.decode(query_pts, geometry_latent, u_delta, x_max, y_max)
    u_pred = uv[0, :, 0].detach().cpu().numpy()
    v_pred = uv[0, :, 1].detach().cpu().numpy()

    pts_np = query_pts[0].detach().cpu().numpy()
    u_base = u_max * pts_np[:, 0] / x_max_physical
    v_base = np.zeros_like(u_base)

    pred_vec = np.stack([u_pred, v_pred], axis=-1)
    base_vec = np.stack([u_base, v_base], axis=-1)

    norm_base = max(np.linalg.norm(base_vec), 1e-12)
    correction_magnitude = float(
        np.linalg.norm(pred_vec - base_vec) / norm_base
    )

    return correction_magnitude, correction_magnitude < threshold


# ---------------------------------------------------------------------------
# 6. Geometry distance to training bank
# ---------------------------------------------------------------------------

def compute_geometry_distances(
    query_params: GeometryParams,
    train_bank_params: List[Tuple[float, float, float, float]],
    param_ranges: Dict[str, Tuple[float, float]],
) -> Tuple[float, float, bool]:
    """Compute distance of query geometry to training bank.

    Uses:
        - Normalized nearest-neighbor in [0,1]^4
        - Mahalanobis distance using training bank covariance
        - Inside-box check against training ranges

    Returns:
        nn_distance: normalized NN distance
        mahalanobis: Mahalanobis distance
        inside_box: whether all params are within training ranges
    """
    q = np.array(query_params.as_tuple())

    # Normalize to [0, 1] using param ranges
    ranges = np.array([
        param_ranges["L_total"],
        param_ranges["W_grip"],
        param_ranges["W_gauge"],
        param_ranges["R_fillet"],
    ])
    lo = ranges[:, 0]
    hi = ranges[:, 1]
    span = hi - lo
    span[span < 1e-12] = 1.0

    q_norm = (q - lo) / span
    bank_norm = np.array([(np.array(p) - lo) / span for p in train_bank_params])

    # NN distance
    dists = np.linalg.norm(bank_norm - q_norm, axis=1)
    nn_dist = float(np.min(dists))

    # Mahalanobis distance
    if len(bank_norm) > 4:
        cov = np.cov(bank_norm.T)
        try:
            cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(4))
            diff = q_norm - bank_norm.mean(axis=0)
            mahal = float(np.sqrt(diff @ cov_inv @ diff))
        except np.linalg.LinAlgError:
            mahal = nn_dist * 10.0  # fallback
    else:
        mahal = nn_dist * 10.0

    # Inside box
    inside = bool(np.all(q >= lo) and np.all(q <= hi))

    return nn_dist, mahal, inside


# ---------------------------------------------------------------------------
# Full reliability assessment
# ---------------------------------------------------------------------------

def run_full_reliability(
    model: nn.Module,
    geometry_latent: torch.Tensor,
    verification_grid: VerificationGrid,
    fillet_info: dict,
    u_delta: torch.Tensor,
    x_max: torch.Tensor,
    y_max: torch.Tensor,
    mu: float,
    lam: float,
    stress_state: str,
    scales: NormalizationScales,
    device: torch.device,
    reference_latents: List[torch.Tensor],
    query_geometry: GeometryParams,
    train_bank_params: List[Tuple[float, float, float, float]],
    param_ranges: Dict[str, Tuple[float, float]],
    u_max: float,
    trac_free_pts: Optional[torch.Tensor] = None,
    trac_free_normals: Optional[torch.Tensor] = None,
    trac_free_tags: Optional[torch.Tensor] = None,
    partial_trac_pts: Optional[torch.Tensor] = None,
    partial_trac_normals: Optional[torch.Tensor] = None,
    partial_trac_dirs: Optional[torch.Tensor] = None,
) -> ReliabilityMetrics:
    """Run the complete reliability assessment pipeline.

    This is the single entry point that computes ALL reliability metrics
    for one geometry prediction.
    """
    model.eval()

    x_max_physical = float(x_max.item())

    # 1. Equilibrium residual on verification grid
    vg_pts = torch.tensor(
        verification_grid.all_pts, dtype=torch.float32, device=device
    ).unsqueeze(0).requires_grad_(True)

    with torch.enable_grad():
        eq_result = compute_equilibrium_residual(
            model, vg_pts, geometry_latent, u_delta, x_max, y_max,
            mu, lam, stress_state, scales,
        )

    # 2. Traction residuals split by segment tag
    trac_gauge = 0.0
    trac_arc = 0.0
    trac_partial = 0.0

    if (trac_free_pts is not None and trac_free_normals is not None
            and trac_free_tags is not None):
        trac_pts_g = trac_free_pts.requires_grad_(True)
        with torch.enable_grad():
            trac_result = compute_traction_residual_by_tag(
                model, trac_pts_g, trac_free_normals, trac_free_tags,
                geometry_latent, u_delta, x_max, y_max,
                mu, lam, stress_state, scales,
            )
        trac_gauge = trac_result["gauge_top"]
        trac_arc = trac_result["arc"]

    if (partial_trac_pts is not None and partial_trac_normals is not None
            and partial_trac_dirs is not None):
        pt_pts_g = partial_trac_pts.requires_grad_(True)
        with torch.enable_grad():
            trac_partial = compute_partial_traction_residual(
                model, pt_pts_g, partial_trac_normals, partial_trac_dirs,
                geometry_latent, u_delta, x_max, y_max,
                mu, lam, stress_state, scales,
            )

    # 3. Section-force CV
    section_cv, resultants, N_mean = compute_section_force_cv(
        model, geometry_latent, fillet_info,
        u_delta, x_max, y_max,
        mu, lam, stress_state, device,
    )

    # 4. Latent-swap sensitivity (subset of verification grid)
    n_swap_pts = min(500, verification_grid.n_total)
    swap_rng = np.random.default_rng(verification_grid.seed + 1)
    swap_idx = swap_rng.choice(
        verification_grid.n_total, n_swap_pts, replace=False
    )
    swap_pts = torch.tensor(
        verification_grid.all_pts[swap_idx],
        dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        swap_sensitivity = compute_swap_sensitivity(
            model, geometry_latent, reference_latents,
            swap_pts, u_delta, x_max, y_max,
            x_max_physical, u_max,
        )

    # 5. Baseline collapse detection — on FULL verification grid, NOT swap subset
    collapse_pts = torch.tensor(
        verification_grid.all_pts, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        correction_mag, _ = detect_baseline_collapse(
            model, collapse_pts, geometry_latent,
            u_delta, x_max, y_max,
            x_max_physical, u_max,
        )

    # 6. Geometry distances
    nn_dist, mahal_dist, inside_box = compute_geometry_distances(
        query_geometry, train_bank_params, param_ranges,
    )

    # Section force anchor ratio: N_mean / N_target_nominal
    # N_target_nominal = E * (u_max / L_half) * (2 * H_gauge) — input-only
    H_gauge = fillet_info["H_gauge"]
    L_half = fillet_info["L_half"]
    N_target_nominal = (scales.stress_scale) * (2.0 * H_gauge)
    anchor_ratio = abs(N_mean) / max(abs(N_target_nominal), 1e-12)

    return ReliabilityMetrics(
        normalized_equilibrium_residual=eq_result["mean_normalized"],
        max_normalized_equilibrium_residual=eq_result["max_normalized"],
        frac_detF_negative=eq_result["frac_detF_neg"],
        frac_detF_low=eq_result["frac_detF_low"],
        section_force_cv=section_cv,
        section_resultants=resultants,
        section_force_mean_N=N_mean,
        section_force_anchor_ratio=anchor_ratio,
        latent_swap_sensitivity=swap_sensitivity,
        correction_magnitude=correction_mag,
        traction_residual_gauge_top=trac_gauge,
        traction_residual_arc=trac_arc,
        traction_residual_partial=trac_partial,
        geometry_nn_distance=nn_dist,
        geometry_mahalanobis=mahal_dist,
        inside_training_box=inside_box,
        verification_grid_seed=verification_grid.seed,
        n_verification_points=verification_grid.n_total,
    )
