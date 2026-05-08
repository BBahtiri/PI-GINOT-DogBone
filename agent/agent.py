#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PI-GINOT Reliability-Aware Agent.

Main agent class providing:
  - predict(): Full inference with reliability diagnostics
  - interpret(): Natural language interpretation of results
  - optimize(): Geometry optimization with reliability penalties
  - refine(): Residual-driven adaptive fine-tuning
  - health_check(): Model health assessment on benchmark set

Response formatting respects the confidence-level behavior table:
  HIGH   → full precision, all plots, full interpretation
  MEDIUM → 2 sig figs, qualified interpretation, residual overlays
  LOW    → 1 sig fig, ranges only, trends only, optimization blocked
  REJECT → no numbers, diagnostic only, blocked
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schemas import (
    GeometryParams,
    ReliabilityLevel,
    InferenceResult,
    RESPONSE_BEHAVIOR,
)
from .constants import AGENT_SEEDS
from .inference import InferenceEngine
from .gates import GateThresholds
from .refinement import ResidualDrivenRefinement
from .health_check import ModelHealthCheck


# ---------------------------------------------------------------------------
# Number formatting per confidence level
# ---------------------------------------------------------------------------

def _format_value(
    value: float,
    unit: str,
    level: ReliabilityLevel,
) -> str:
    """Format a numeric value according to confidence-level constraints.

    Dispatches on level first to avoid operator-precedence ambiguity.
    """
    if level == ReliabilityLevel.REJECT:
        return "[withheld — insufficient reliability]"

    sig_figs = RESPONSE_BEHAVIOR[level]["sig_figs"]

    if sig_figs == 1:
        # 1 significant figure + range indicator
        magnitude = 10 ** int(np.floor(np.log10(abs(value) + 1e-30)))
        rounded = round(value / magnitude) * magnitude
        return f"~{rounded:.0f} {unit}"
    elif sig_figs == 2:
        return f"{value:.2g} {unit}"
    else:
        # Full precision (HIGH: sig_figs=None)
        if abs(value) < 0.01 or abs(value) > 1e4:
            return f"{value:.4e} {unit}"
        else:
            return f"{value:.4f} {unit}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PI_GINOT_Agent:
    """Reliability-aware inference agent for PI-GINOT.

    Wraps the InferenceEngine with:
      - Structured JSON output
      - Response formatting per confidence level
      - Geometry optimization with reliability gates
      - Adaptive refinement orchestration
      - Model health monitoring

    Args:
        checkpoint_path: Path to trained PI-GINOT checkpoint.
        device: 'cpu', 'cuda', or 'auto'.
        gate_thresholds: Optional custom reliability thresholds.
        verification_seed: Seed for independent verification grid.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        gate_thresholds: Optional[GateThresholds] = None,
        verification_seed: int = 99999,
    ):
        self.engine = InferenceEngine.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            gate_thresholds=gate_thresholds,
            verification_seed=verification_seed,
        )
        self.checkpoint_path = checkpoint_path
        self._history: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # predict(): Core prediction with full reliability
    # -----------------------------------------------------------------

    def predict(
        self,
        geometry: GeometryParams,
        n_query_points: int = 4000,
    ) -> Dict[str, Any]:
        """Run reliability-aware prediction and return structured output.

        Args:
            geometry: DogBone geometry specification.
            n_query_points: Interior points for field evaluation.

        Returns:
            Dict with keys:
                status: 'ok' | 'rejected'
                confidence_level: 'high' | 'medium' | 'low' | 'reject'
                summary: Human-readable summary
                diagnostics: Full reliability metrics
                fields: Formatted field statistics (respects confidence level)
                provenance: Checkpoint and inference metadata
                response_behavior: What the agent is allowed to show
                raw_result: InferenceResult object (for programmatic use)
        """
        result = self.engine.predict(geometry, n_query_points)
        level = result.confidence_level

        # Build formatted output
        output = {
            "status": "rejected" if result.is_rejected() else "ok",
            "confidence_level": level.value,
            "rejection_reasons": result.rejection_reasons,
            "summary": self._build_summary(result),
            "diagnostics": self._build_diagnostics(result),
            "fields": self._build_field_stats(result),
            "provenance": self._build_provenance(result),
            "response_behavior": result.response_behavior,
            "raw_result": result,
        }

        # Log to history
        self._history.append({
            "geometry": geometry.as_dict(),
            "confidence_level": level.value,
            "eq_residual": result.reliability.normalized_equilibrium_residual,
            "section_cv": result.reliability.section_force_cv,
            "provenance": result.provenance.checkpoint_id,
        })

        return output

    def _build_summary(self, result: InferenceResult) -> str:
        """Build human-readable summary respecting confidence level."""
        level = result.confidence_level
        geo = result.provenance.geometry_params
        r = result.reliability

        if level == ReliabilityLevel.REJECT:
            reasons = "; ".join(result.rejection_reasons)
            return (
                f"PREDICTION REJECTED for DogBone geometry "
                f"(L={geo.L_total:.1f}, W_grip={geo.W_grip:.1f}, "
                f"W_gauge={geo.W_gauge:.1f}, R={geo.R_fillet:.1f}).\n"
                f"Reasons: {reasons}\n"
                f"No field values are provided. Diagnostic plots only."
            )

        # Confidence qualifier
        qualifier = {
            ReliabilityLevel.HIGH: "",
            ReliabilityLevel.MEDIUM: " (moderate confidence — see caveats)",
            ReliabilityLevel.LOW: " (LOW confidence — trends only)",
        }[level]

        peak_S11 = _format_value(
            float(np.max(np.abs(result.cauchy_S11))), "MPa", level
        )
        peak_u = _format_value(
            float(np.max(np.abs(result.displacement_u))), "mm", level
        )

        return (
            f"DogBone prediction{qualifier}: "
            f"L={geo.L_total:.1f}, W_grip={geo.W_grip:.1f}, "
            f"W_gauge={geo.W_gauge:.1f}, R={geo.R_fillet:.1f} mm.\n"
            f"Peak |σ₁₁| = {peak_S11}, max |u| = {peak_u}.\n"
            f"Equilibrium residual (normalized): "
            f"{r.normalized_equilibrium_residual:.3e}, "
            f"section-force CV: {100*r.section_force_cv:.1f}%."
        )

    def _build_diagnostics(self, result: InferenceResult) -> dict:
        """Full reliability diagnostics dict."""
        r = result.reliability
        return {
            "normalized_equilibrium_residual": r.normalized_equilibrium_residual,
            "max_normalized_eq_residual": r.max_normalized_equilibrium_residual,
            "frac_detF_negative": r.frac_detF_negative,
            "frac_detF_below_0.1": r.frac_detF_low,
            "section_force_cv": r.section_force_cv,
            "section_resultants_N": r.section_resultants,
            "latent_swap_sensitivity": r.latent_swap_sensitivity,
            "correction_magnitude": r.correction_magnitude,
            "traction_residual_gauge_top": r.traction_residual_gauge_top,
            "traction_residual_arc": r.traction_residual_arc,
            "traction_residual_partial": r.traction_residual_partial,
            "geometry_nn_distance": r.geometry_nn_distance,
            "geometry_mahalanobis": r.geometry_mahalanobis,
            "inside_training_box": r.inside_training_box,
            "verification_grid_seed": r.verification_grid_seed,
            "n_verification_points": r.n_verification_points,
            "stress_scale_MPa": result.scales.stress_scale,
            "length_scale_mm": result.scales.length_scale,
        }

    def _build_field_stats(self, result: InferenceResult) -> dict:
        """Field statistics formatted per confidence level."""
        level = result.confidence_level

        if level == ReliabilityLevel.REJECT:
            return {"note": "Field values withheld due to rejection."}

        stats = {}
        for name, arr in [
            ("u_max_mm", np.max(np.abs(result.displacement_u))),
            ("v_max_mm", np.max(np.abs(result.displacement_v))),
            ("correction_u_max_mm", np.max(np.abs(result.correction_u))),
            ("sigma_11_max_MPa", np.max(np.abs(result.cauchy_S11))),
            ("sigma_22_max_MPa", np.max(np.abs(result.cauchy_S22))),
            ("sigma_12_max_MPa", np.max(np.abs(result.cauchy_S12))),
            ("P11_max_MPa", np.max(np.abs(result.stress_P11))),
            ("E11_max", np.max(np.abs(result.strain_E11))),
            ("detF_min", np.min(result.det_F)),
            ("detF_max", np.max(result.det_F)),
        ]:
            unit = name.split("_")[-1] if "_" in name else ""
            stats[name] = _format_value(float(arr), unit, level)

        return stats

    def _build_provenance(self, result: InferenceResult) -> dict:
        """Provenance metadata for reproducibility."""
        p = result.provenance
        return {
            "checkpoint_id": p.checkpoint_id,
            "checkpoint_path": p.checkpoint_path,
            "training_epochs": p.checkpoint_training_epochs,
            "best_val_raw": p.checkpoint_best_val_raw,
            "gate_status_at_save": p.checkpoint_gate_status,
            "stress_formulation": p.stress_formulation,
            "loading_u_delta_mm": p.loading_u_delta,
            "inference_timestamp": p.inference_timestamp,
            "verification_grid_seed": p.verification_grid_seed,
            "geometry_nn_distance": result.reliability.geometry_nn_distance,
            "geometry_mahalanobis": result.reliability.geometry_mahalanobis,
        }

    # -----------------------------------------------------------------
    # interpret(): Natural-language interpretation
    # -----------------------------------------------------------------

    def interpret(self, result_dict: Dict[str, Any]) -> str:
        """Generate natural-language interpretation of prediction results.

        Respects confidence-level constraints on what can be stated.
        """
        level = ReliabilityLevel(result_dict["confidence_level"])
        diag = result_dict["diagnostics"]
        fields = result_dict["fields"]

        lines = [result_dict["summary"], ""]

        if level == ReliabilityLevel.REJECT:
            lines.append("Action required: resolve rejection reasons before "
                         "using this prediction. Consider:")
            lines.append("  - Checking geometry parameters are within "
                         "training ranges")
            lines.append("  - Running residual-driven refinement (agent.refine())")
            lines.append("  - Inspecting diagnostic plots for systematic issues")
            return "\n".join(lines)

        # Reliability assessment
        lines.append("Reliability assessment:")
        lines.append(f"  Equilibrium residual (normalized): "
                     f"{diag['normalized_equilibrium_residual']:.3e}")
        lines.append(f"  Section-force CV: "
                     f"{100*diag['section_force_cv']:.1f}%")
        lines.append(f"  Latent-swap sensitivity: "
                     f"{diag['latent_swap_sensitivity']:.4f}")
        lines.append(f"  Correction magnitude: "
                     f"{diag['correction_magnitude']:.4f}")
        lines.append(f"  Geometry NN distance: "
                     f"{diag['geometry_nn_distance']:.4f}")

        if level == ReliabilityLevel.LOW:
            lines.append("")
            lines.append("WARNING: Low confidence. Only trends are "
                         "interpretable. Do not use individual values "
                         "for design decisions.")
        elif level == ReliabilityLevel.MEDIUM:
            lines.append("")
            lines.append("Note: Moderate confidence. Values are approximate "
                         "(2 significant figures). Residual overlays "
                         "recommended for spatial assessment.")

        if level in (ReliabilityLevel.HIGH, ReliabilityLevel.MEDIUM):
            lines.append("")
            lines.append("Field summary:")
            for k, v in fields.items():
                if k != "note":
                    lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # optimize(): Geometry optimization with reliability gates
    # -----------------------------------------------------------------

    def optimize(
        self,
        objective: str = "minimize_peak_sigma11",
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_candidates: int = 50,
        target_strain: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run geometry optimization with reliability-aware evaluation.

        Uses random search with reliability filtering.  Candidates that
        don't reach at least MEDIUM confidence are penalized or excluded.

        SCALING CAVEAT:
            The model was trained at u_delta = 1.0 mm, giving a nominal
            training strain of u_max / L_half (geometry-dependent, typically
            ~2-4%).  If target_strain is set, linear elastic scaling is
            applied: stress_scaled = stress_predicted * (target_strain /
            training_strain).  This is only valid when BOTH the training
            strain and target strain are in the small-strain regime.  The
            Neo-Hookean material is nonlinear, so scaling from a 4%
            prediction to 2% introduces some error from the nonlinear term.

        Args:
            objective: 'minimize_peak_sigma11' or 'maximize_correction_range'
            param_bounds: Custom parameter ranges (default: training ranges)
            n_candidates: Number of geometry candidates to evaluate
            target_strain: If set, scales stress linearly from the training
                           strain.  Capped at ±50% of training strain.

        Returns:
            Dict with best geometry, objective value, and reliability info.
        """
        from config import GEOMETRY_RANGES, LOADING_CONFIG

        if param_bounds is None:
            param_bounds = {
                "L_total": GEOMETRY_RANGES["L_total"],
                "W_grip": GEOMETRY_RANGES["W_grip"],
                "W_gauge": GEOMETRY_RANGES["W_gauge"],
                "R_fillet": GEOMETRY_RANGES["R_fillet"],
            }

        rng = np.random.default_rng(AGENT_SEEDS["optimize_rng"])
        u_max = LOADING_CONFIG["u_max"]
        candidates = []
        scaling_warnings = []

        for i in range(n_candidates):
            params = {}
            for key, (lo, hi) in param_bounds.items():
                params[key] = float(rng.uniform(lo, hi))
            params["holes"] = []

            from config import validate_geometry
            if not validate_geometry(params):
                continue

            geo = GeometryParams(**{k: v for k, v in params.items()
                                    if k != "holes"})

            try:
                result = self.engine.predict(geo, n_query_points=2000)
            except Exception:
                continue

            if result.is_rejected():
                continue

            # Compute scaling factor
            training_strain = u_max / geo.L_half
            if target_strain is not None:
                strain_ratio = target_strain / training_strain
                # Cap at ±50% of training strain
                if strain_ratio < 0.5 or strain_ratio > 1.5:
                    if len(scaling_warnings) < 3:
                        scaling_warnings.append(
                            f"Geometry L_half={geo.L_half:.1f}: "
                            f"training_strain={training_strain:.4f}, "
                            f"target={target_strain:.4f}, "
                            f"ratio={strain_ratio:.2f} outside [0.5, 1.5]"
                        )
                    continue  # skip — scaling unreliable
                scale = strain_ratio
            else:
                scale = 1.0

            if objective == "minimize_peak_sigma11":
                obj_val = float(np.max(np.abs(result.cauchy_S11))) * scale
            elif objective == "maximize_correction_range":
                obj_val = -float(np.max(np.abs(result.correction_u))) * scale
            else:
                raise ValueError(f"Unknown objective: {objective}")

            # Reliability penalty
            penalty = 1.0
            if result.confidence_level == ReliabilityLevel.MEDIUM:
                penalty = 1.1
            elif result.confidence_level == ReliabilityLevel.LOW:
                penalty = 2.0

            candidates.append({
                "geometry": geo,
                "objective_raw": obj_val,
                "objective_penalized": obj_val * penalty,
                "confidence_level": result.confidence_level.value,
                "eq_residual": result.reliability.normalized_equilibrium_residual,
                "section_cv": result.reliability.section_force_cv,
                "training_strain": training_strain,
            })

        if not candidates:
            return {
                "status": "error",
                "message": "No valid candidates found. All geometries were "
                           "rejected by reliability gates.",
                "scaling_warnings": scaling_warnings,
            }

        candidates.sort(key=lambda c: c["objective_penalized"])
        best = candidates[0]

        scaling_disclosure = (
            f"Linear scaling applied from training strain "
            f"({best['training_strain']:.4f}) to target_strain "
            f"({target_strain:.4f}).  The Neo-Hookean material is "
            f"nonlinear; this introduces some error.  Both strains "
            f"should be in the small-strain regime for this to be valid."
            if target_strain is not None else
            f"Direct prediction at u_delta = {u_max} mm "
            f"(training strain ~ {best['training_strain']:.4f})."
        )

        return {
            "status": "ok",
            "objective": objective,
            "best_geometry": best["geometry"].as_dict(),
            "best_objective_value": best["objective_raw"],
            "best_confidence_level": best["confidence_level"],
            "n_candidates_evaluated": len(candidates),
            "n_candidates_rejected": n_candidates - len(candidates),
            "scaling_disclosure": scaling_disclosure,
            "scaling_warnings": scaling_warnings,
            "all_candidates": candidates[:10],
        }

    # -----------------------------------------------------------------
    # refine(): Residual-driven adaptive refinement
    # -----------------------------------------------------------------

    def refine(
        self,
        geometry: GeometryParams,
        n_epochs: int = 100,
        lr: float = 1e-5,
    ) -> Dict[str, Any]:
        """Run residual-driven fine-tuning for a specific geometry.

        Fine-tuning improves intrinsic physics consistency on this geometry.
        This reduces observable residuals but does NOT guarantee reduction
        of systematic bias, since no external reference is used.

        Always evaluates on a held-out benchmark before and after.

        Returns:
            Dict with before/after reliability metrics.
        """
        refiner = ResidualDrivenRefinement(
            engine=self.engine,
            geometry=geometry,
            n_epochs=n_epochs,
            lr=lr,
        )
        return refiner.run()

    # -----------------------------------------------------------------
    # health_check(): Model health monitoring
    # -----------------------------------------------------------------

    def health_check(
        self,
        n_benchmark: int = 20,
        benchmark_seed: int = 88888,
    ) -> Dict[str, Any]:
        """Run model health check on a fixed benchmark geometry set.

        Returns reliability metrics for each benchmark geometry,
        aggregate statistics, and pass/fail assessment.
        """
        checker = ModelHealthCheck(
            engine=self.engine,
            n_benchmark=n_benchmark,
            benchmark_seed=benchmark_seed,
        )
        return checker.run()

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def history(self) -> List[Dict[str, Any]]:
        """Return prediction history for this session."""
        return self._history.copy()

    def to_json(self, result_dict: Dict[str, Any]) -> str:
        """Serialize prediction result to JSON (excluding numpy arrays)."""
        serializable = {
            k: v for k, v in result_dict.items()
            if k != "raw_result"
        }
        return json.dumps(serializable, indent=2, default=str)
