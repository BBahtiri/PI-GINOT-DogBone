#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confidence gate logic for PI-GINOT agent.

Implements the four-level reliability classification:
  REJECT  — hard failures, prediction not usable
  LOW     — significant concerns, only trends interpretable
  MEDIUM  — moderate concerns, qualified interpretation
  HIGH    — all checks pass, full precision output

Rejection criteria (any one triggers REJECT):
  1. detF <= 0 at > 1% of verification points
  2. section_force_cv > 0.50
  3. latent_swap_sensitivity < 0.01 (frozen encoder)
  4. geometry outside training box
  5. decoder collapsed to hard-BC baseline (correction_magnitude < 0.01)

Threshold calibration follows the design doc §2.5 response behavior table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .schemas import (
    ReliabilityLevel,
    ReliabilityMetrics,
    RESPONSE_BEHAVIOR,
)


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

@dataclass
class GateThresholds:
    """Configurable thresholds for each gate.

    Defaults follow the design doc. Can be overridden per-agent
    for tighter/looser gates.
    """
    # REJECT triggers
    frac_detF_neg_reject: float = 0.01      # > 1% negative detF
    section_cv_reject: float = 0.50         # CV > 50%
    swap_sensitivity_reject: float = 0.01   # < 1% relative change
    require_inside_box: bool = True         # must be within training ranges
    correction_collapse_threshold: float = 0.01  # < 1% correction

    # LOW thresholds
    eq_residual_low: float = 1.0            # normalized eq residual
    section_cv_low: float = 0.20            # CV > 20%
    swap_sensitivity_low: float = 0.05      # < 5%
    frac_detF_neg_low: float = 0.001        # > 0.1%
    geometry_mahal_low: float = 4.0         # Mahalanobis > 4

    # MEDIUM thresholds
    eq_residual_medium: float = 0.1         # normalized eq residual
    section_cv_medium: float = 0.10         # CV > 10%
    swap_sensitivity_medium: float = 0.10   # < 10%
    correction_magnitude_medium: float = 0.05
    geometry_mahal_medium: float = 2.5      # Mahalanobis > 2.5

    # Section force anchor ratio (N_mean / N_target)
    # Catches magnitude collapse: model gets constant N(x) but wrong scale
    anchor_ratio_reject_lo: float = 0.2     # below 20% of expected
    anchor_ratio_reject_hi: float = 5.0     # above 500% of expected
    anchor_ratio_low_lo: float = 0.5        # 50%
    anchor_ratio_low_hi: float = 2.0        # 200%
    anchor_ratio_medium_lo: float = 0.8     # 80%
    anchor_ratio_medium_hi: float = 1.25    # 125%

    # HIGH: everything below medium thresholds


DEFAULT_THRESHOLDS = GateThresholds()


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_gates(
    metrics: ReliabilityMetrics,
    thresholds: GateThresholds = DEFAULT_THRESHOLDS,
) -> Tuple[ReliabilityLevel, List[str]]:
    """Evaluate all confidence gates and classify the prediction.

    Returns:
        level: ReliabilityLevel enum
        reasons: List of human-readable reasons for the classification
    """
    rejection_reasons: List[str] = []
    concerns: List[str] = []

    # --- REJECTION GATES (any one = REJECT) ---

    # Gate 1: detF negative
    if metrics.frac_detF_negative > thresholds.frac_detF_neg_reject:
        rejection_reasons.append(
            f"detF <= 0 at {100*metrics.frac_detF_negative:.1f}% of "
            f"verification points (threshold: {100*thresholds.frac_detF_neg_reject:.1f}%)"
        )

    # Gate 2: Section force imbalance
    if metrics.section_force_cv > thresholds.section_cv_reject:
        rejection_reasons.append(
            f"Section-force CV = {100*metrics.section_force_cv:.1f}% "
            f"(threshold: {100*thresholds.section_cv_reject:.0f}%)"
        )

    # Gate 3: Frozen encoder (swap sensitivity too low)
    if metrics.latent_swap_sensitivity < thresholds.swap_sensitivity_reject:
        rejection_reasons.append(
            f"Latent-swap sensitivity = {metrics.latent_swap_sensitivity:.4f} "
            f"(minimum: {thresholds.swap_sensitivity_reject:.3f}) — "
            f"encoder may be ignoring geometry input"
        )

    # Gate 4: Outside training box
    if thresholds.require_inside_box and not metrics.inside_training_box:
        rejection_reasons.append(
            "Geometry parameters outside training range (extrapolation)"
        )

    # Gate 5: Baseline collapse (§2.4)
    if metrics.correction_magnitude < thresholds.correction_collapse_threshold:
        rejection_reasons.append(
            f"Decoder collapsed to hard-BC baseline "
            f"(correction_magnitude = {metrics.correction_magnitude:.4f}, "
            f"threshold: {thresholds.correction_collapse_threshold:.3f})"
        )

    # Gate 6: Section force magnitude collapse
    ar = metrics.section_force_anchor_ratio
    if ar < thresholds.anchor_ratio_reject_lo or ar > thresholds.anchor_ratio_reject_hi:
        rejection_reasons.append(
            f"Section-force anchor ratio = {ar:.3f} "
            f"(valid range: [{thresholds.anchor_ratio_reject_lo}, "
            f"{thresholds.anchor_ratio_reject_hi}]) — "
            f"axial force magnitude is wrong"
        )

    if rejection_reasons:
        return ReliabilityLevel.REJECT, rejection_reasons

    # --- LOW GATES ---
    low_reasons: List[str] = []

    if metrics.normalized_equilibrium_residual > thresholds.eq_residual_low:
        low_reasons.append(
            f"High equilibrium residual: {metrics.normalized_equilibrium_residual:.3e} "
            f"(threshold: {thresholds.eq_residual_low:.1e})"
        )

    if metrics.section_force_cv > thresholds.section_cv_low:
        low_reasons.append(
            f"Section-force CV = {100*metrics.section_force_cv:.1f}% "
            f"(LOW threshold: {100*thresholds.section_cv_low:.0f}%)"
        )

    if metrics.latent_swap_sensitivity < thresholds.swap_sensitivity_low:
        low_reasons.append(
            f"Low swap sensitivity: {metrics.latent_swap_sensitivity:.4f}"
        )

    if metrics.frac_detF_negative > thresholds.frac_detF_neg_low:
        low_reasons.append(
            f"detF <= 0 at {100*metrics.frac_detF_negative:.2f}% of points"
        )

    if metrics.geometry_mahalanobis > thresholds.geometry_mahal_low:
        low_reasons.append(
            f"Mahalanobis distance = {metrics.geometry_mahalanobis:.2f} "
            f"(threshold: {thresholds.geometry_mahal_low:.1f})"
        )

    if (ar < thresholds.anchor_ratio_low_lo
            or ar > thresholds.anchor_ratio_low_hi):
        low_reasons.append(
            f"Section-force anchor ratio = {ar:.3f} "
            f"(LOW range: [{thresholds.anchor_ratio_low_lo}, "
            f"{thresholds.anchor_ratio_low_hi}])"
        )

    if low_reasons:
        return ReliabilityLevel.LOW, low_reasons

    # --- MEDIUM GATES ---
    medium_reasons: List[str] = []

    if metrics.normalized_equilibrium_residual > thresholds.eq_residual_medium:
        medium_reasons.append(
            f"Moderate equilibrium residual: "
            f"{metrics.normalized_equilibrium_residual:.3e}"
        )

    if metrics.section_force_cv > thresholds.section_cv_medium:
        medium_reasons.append(
            f"Section-force CV = {100*metrics.section_force_cv:.1f}%"
        )

    if metrics.latent_swap_sensitivity < thresholds.swap_sensitivity_medium:
        medium_reasons.append(
            f"Moderate swap sensitivity: {metrics.latent_swap_sensitivity:.4f}"
        )

    if metrics.correction_magnitude < thresholds.correction_magnitude_medium:
        medium_reasons.append(
            f"Small correction magnitude: {metrics.correction_magnitude:.4f}"
        )

    if metrics.geometry_mahalanobis > thresholds.geometry_mahal_medium:
        medium_reasons.append(
            f"Geometry Mahalanobis = {metrics.geometry_mahalanobis:.2f}"
        )

    if (ar < thresholds.anchor_ratio_medium_lo
            or ar > thresholds.anchor_ratio_medium_hi):
        medium_reasons.append(
            f"Section-force anchor ratio = {ar:.3f} "
            f"(MEDIUM range: [{thresholds.anchor_ratio_medium_lo}, "
            f"{thresholds.anchor_ratio_medium_hi}])"
        )

    if medium_reasons:
        return ReliabilityLevel.MEDIUM, medium_reasons

    # --- HIGH ---
    return ReliabilityLevel.HIGH, ["All reliability checks passed"]


def get_response_behavior(level: ReliabilityLevel) -> dict:
    """Return the response behavior constraints for a confidence level."""
    return RESPONSE_BEHAVIOR[level].copy()
