#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured data models for the PI-GINOT reliability-aware agent.

Defines enums, dataclasses, and typed containers for:
  - Reliability levels with explicit response behavior
  - Prediction provenance (checkpoint lineage)
  - Reliability metrics (normalized residuals, gates)
  - Full inference results with diagnostics
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# ---------------------------------------------------------------------------
# Reliability levels
# ---------------------------------------------------------------------------

class ReliabilityLevel(enum.Enum):
    """Four-tier reliability classification.

    Each level maps to explicit response behavior constraints.
    """
    REJECT = "reject"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Response behavior table (§2.5 of design doc)
RESPONSE_BEHAVIOR: Dict[ReliabilityLevel, dict] = {
    ReliabilityLevel.HIGH: {
        "numbers": "full precision",
        "plots": "all",
        "interpretation": "full",
        "optimization": "allowed",
        "retraining_eligible": False,
        "sig_figs": None,  # unlimited
    },
    ReliabilityLevel.MEDIUM: {
        "numbers": "2 significant figures",
        "plots": "all + residual overlay",
        "interpretation": "qualified (with caveats)",
        "optimization": "allowed with penalty",
        "retraining_eligible": True,  # optional
        "sig_figs": 2,
    },
    ReliabilityLevel.LOW: {
        "numbers": "1 significant figure, ranges",
        "plots": "residual overlay only",
        "interpretation": "minimal, only trends",
        "optimization": "blocked",
        "retraining_eligible": True,  # recommended
        "sig_figs": 1,
    },
    ReliabilityLevel.REJECT: {
        "numbers": "none",
        "plots": "diagnostic only",
        "interpretation": "none",
        "optimization": "blocked",
        "retraining_eligible": True,  # required before retry
        "sig_figs": 0,
    },
}


# ---------------------------------------------------------------------------
# Material parameters (input-only, never from predictions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialParams:
    """Immutable material parameter container."""
    E: float       # Young's modulus [MPa]
    nu: float      # Poisson's ratio [-]
    mu: float      # Shear modulus [MPa]
    lam: float     # First Lamé parameter [MPa]
    state: str     # 'plane strain' or 'plane stress'


# ---------------------------------------------------------------------------
# Geometry parameters
# ---------------------------------------------------------------------------

@dataclass
class GeometryParams:
    """Parametric DogBone geometry specification."""
    L_total: float      # Full specimen length [mm]
    W_grip: float       # Full grip width [mm]
    W_gauge: float      # Full gauge width [mm]
    R_fillet: float     # Fillet radius [mm]

    @property
    def L_half(self) -> float:
        return self.L_total / 2.0

    @property
    def H_grip(self) -> float:
        return self.W_grip / 2.0

    @property
    def H_gauge(self) -> float:
        return self.W_gauge / 2.0

    def as_dict(self) -> dict:
        return {
            "L_total": self.L_total,
            "W_grip": self.W_grip,
            "W_gauge": self.W_gauge,
            "R_fillet": self.R_fillet,
            "holes": [],
        }

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Parameter vector for distance computations."""
        return (self.L_total, self.W_grip, self.W_gauge, self.R_fillet)


# ---------------------------------------------------------------------------
# Normalization scales (input-only, never from predictions — §2.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizationScales:
    """Physically-motivated scales depending ONLY on inputs and material.

    stress_scale = E * u_delta / L_half   (small-strain 1D nominal)
    length_scale = L_half

    These are honest: a model that under-predicts stress will NOT
    under-predict these scales.
    """
    stress_scale: float     # [MPa]
    length_scale: float     # [mm]

    @classmethod
    def from_inputs(
        cls,
        E: float,
        u_delta: float,
        L_half: float,
    ) -> "NormalizationScales":
        return cls(
            stress_scale=E * u_delta / L_half,
            length_scale=L_half,
        )


# ---------------------------------------------------------------------------
# Prediction provenance (§4.1)
# ---------------------------------------------------------------------------

@dataclass
class PredictionProvenance:
    """Full lineage of a single prediction for reproducibility.

    Answers: 'why did yesterday's prediction differ from today's?'
    """
    checkpoint_path: str
    checkpoint_id: str                      # hash or filename
    checkpoint_training_date: Optional[str]
    checkpoint_training_epochs: int
    checkpoint_best_val_raw: float
    checkpoint_gate_status: Dict[str, float]  # ckpt_swap_du, ckpt_section_cv

    stress_formulation: str                 # 'plane strain' or 'plane stress'
    material_params: MaterialParams
    loading_u_delta: float

    inference_timestamp: str
    verification_grid_seed: int
    training_rng_seeds: List[int]           # from checkpoint

    geometry_params: GeometryParams
    # NOTE: geometry distances live in ReliabilityMetrics (single source of truth).
    # Provenance records WHAT was predicted; ReliabilityMetrics records HOW reliable it was.

    @classmethod
    def create_timestamp(cls) -> str:
        return datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Reliability metrics
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityMetrics:
    """All reliability diagnostics for one prediction.

    Every metric that feeds into the confidence gate is collected here.
    Residuals are normalized by input-only scales.
    """
    # Normalized equilibrium residual: mean |Div(P)|^2 * (L0/S0)^2
    normalized_equilibrium_residual: float

    # Max normalized equilibrium residual (worst-case point)
    max_normalized_equilibrium_residual: float

    # Fraction of verification points with detF <= 0
    frac_detF_negative: float

    # Fraction of verification points with detF < 0.1
    frac_detF_low: float

    # Section force coefficient of variation (on verification slices)
    section_force_cv: float

    # Axial resultants at each verification slice [N]
    section_resultants: List[float]

    # Latent swap sensitivity (against fixed reference set)
    latent_swap_sensitivity: float

    # Hard-BC baseline collapse magnitude: ||u_pred - u_baseline|| / ||u_baseline||
    correction_magnitude: float

    # Traction residuals (normalized)
    traction_residual_gauge_top: float
    traction_residual_arc: float
    traction_residual_partial: float

    # Geometry distance metrics
    geometry_nn_distance: float             # NN in normalized param space
    geometry_mahalanobis: float             # Mahalanobis distance

    # Inside training box?
    inside_training_box: bool

    # Section force magnitude (catches constant-but-wrong-magnitude N(x))
    section_force_mean_N: float             # mean axial resultant [N]
    section_force_anchor_ratio: float       # N_mean / N_target_nominal

    # Verification grid info
    verification_grid_seed: int
    n_verification_points: int


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Complete output of a single reliability-aware inference.

    Contains field predictions, reliability metrics, confidence level,
    rejection reasons, provenance, and response behavior constraints.
    """
    # Core predictions (numpy arrays on CPU)
    query_points: object        # np.ndarray [N, 2] physical coords
    displacement_u: object      # np.ndarray [N]
    displacement_v: object      # np.ndarray [N]
    correction_u: object        # np.ndarray [N] = u_pred - u_baseline
    stress_P11: object          # np.ndarray [N] 1st Piola-Kirchhoff
    stress_P22: object          # np.ndarray [N]
    stress_P12: object          # np.ndarray [N]
    cauchy_S11: object          # np.ndarray [N] Cauchy stress
    cauchy_S22: object          # np.ndarray [N]
    cauchy_S12: object          # np.ndarray [N]
    strain_E11: object          # np.ndarray [N] Green-Lagrange
    det_F: object               # np.ndarray [N] deformation determinant

    # Reliability
    reliability: ReliabilityMetrics
    confidence_level: ReliabilityLevel
    rejection_reasons: List[str]

    # Response behavior (derived from confidence level)
    response_behavior: dict

    # Provenance
    provenance: PredictionProvenance

    # Normalization scales used
    scales: NormalizationScales

    def is_rejected(self) -> bool:
        return self.confidence_level == ReliabilityLevel.REJECT

    def summary_dict(self) -> dict:
        """Compact summary for JSON serialization."""
        return {
            "confidence_level": self.confidence_level.value,
            "rejected": self.is_rejected(),
            "rejection_reasons": self.rejection_reasons,
            "normalized_eq_residual": self.reliability.normalized_equilibrium_residual,
            "section_force_cv": self.reliability.section_force_cv,
            "latent_swap_sensitivity": self.reliability.latent_swap_sensitivity,
            "correction_magnitude": self.reliability.correction_magnitude,
            "frac_detF_negative": self.reliability.frac_detF_negative,
            "inside_training_box": self.reliability.inside_training_box,
            "geometry_nn_distance": self.reliability.geometry_nn_distance,
            "stress_scale_MPa": self.scales.stress_scale,
            "length_scale_mm": self.scales.length_scale,
            "checkpoint_id": self.provenance.checkpoint_id,
            "verification_grid_seed": self.reliability.verification_grid_seed,
        }
