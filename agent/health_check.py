#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model health check for PI-GINOT (§4.2 of design doc).

Runs reliability diagnostics on a fixed benchmark geometry set.
Designed to be run independently of user queries (nightly/weekly).

Provides:
  - Per-geometry reliability metrics
  - Aggregate pass/fail statistics
  - Regression detection vs. historical baselines
  - Time-series–ready output for health monitoring dashboards
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import validate_geometry
from geometry.parametric_dogbone import sample_geometry_params

from .constants import AGENT_SEEDS
from .schemas import GeometryParams, ReliabilityLevel


class ModelHealthCheck:
    """Fixed-benchmark health assessment for PI-GINOT.

    Evaluates a deterministic set of benchmark geometries and
    produces a structured health report.

    Args:
        engine: InferenceEngine instance.
        n_benchmark: Number of benchmark geometries.
        benchmark_seed: Fixed seed for reproducible benchmark set.
    """

    def __init__(
        self,
        engine,  # InferenceEngine
        n_benchmark: int = 20,
        benchmark_seed: int = AGENT_SEEDS["health_check_benchmark"],
    ):
        self.engine = engine
        self.n_benchmark = n_benchmark
        self.benchmark_seed = benchmark_seed
        self._benchmark_geos = self._build_benchmark_set()

    def _build_benchmark_set(self) -> List[GeometryParams]:
        """Build a fixed, reproducible set of benchmark geometries."""
        rng = np.random.default_rng(self.benchmark_seed)
        geos = []
        attempts = 0
        while len(geos) < self.n_benchmark and attempts < self.n_benchmark * 10:
            attempts += 1
            try:
                p = sample_geometry_params(rng)
                geo = GeometryParams(
                    L_total=p["L_total"],
                    W_grip=p["W_grip"],
                    W_gauge=p["W_gauge"],
                    R_fillet=p["R_fillet"],
                )
                geos.append(geo)
            except Exception:
                continue
        return geos

    def run(self) -> Dict[str, Any]:
        """Run health check on all benchmark geometries.

        Returns:
            Dict with:
                timestamp: ISO timestamp of the check
                checkpoint_id: which model was evaluated
                n_geometries: number of benchmark geometries
                per_geometry: list of per-geometry metrics
                aggregate: summary statistics
                pass_fail: overall health assessment
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        checkpoint_id = self.engine.checkpoint_meta["checkpoint_id"]

        per_geometry = []
        levels = []
        eq_residuals = []
        section_cvs = []
        swap_sensitivities = []
        correction_mags = []
        n_rejected = 0

        for i, geo in enumerate(self._benchmark_geos):
            try:
                result = self.engine.predict(geo, n_query_points=2000)
                r = result.reliability
                entry = {
                    "index": i,
                    "geometry": geo.as_dict(),
                    "confidence_level": result.confidence_level.value,
                    "rejected": result.is_rejected(),
                    "eq_residual": r.normalized_equilibrium_residual,
                    "section_cv": r.section_force_cv,
                    "swap_sensitivity": r.latent_swap_sensitivity,
                    "correction_magnitude": r.correction_magnitude,
                    "frac_detF_negative": r.frac_detF_negative,
                    "geometry_nn_distance": r.geometry_nn_distance,
                }
                per_geometry.append(entry)
                levels.append(result.confidence_level)
                eq_residuals.append(r.normalized_equilibrium_residual)
                section_cvs.append(r.section_force_cv)
                swap_sensitivities.append(r.latent_swap_sensitivity)
                correction_mags.append(r.correction_magnitude)
                if result.is_rejected():
                    n_rejected += 1
            except Exception as e:
                per_geometry.append({
                    "index": i,
                    "geometry": geo.as_dict(),
                    "error": str(e),
                })

        # Aggregate statistics
        n_eval = len(eq_residuals)
        aggregate = {}
        if n_eval > 0:
            aggregate = {
                "n_evaluated": n_eval,
                "n_rejected": n_rejected,
                "rejection_rate": n_rejected / n_eval,
                "eq_residual_mean": float(np.mean(eq_residuals)),
                "eq_residual_std": float(np.std(eq_residuals)),
                "eq_residual_max": float(np.max(eq_residuals)),
                "section_cv_mean": float(np.mean(section_cvs)),
                "section_cv_max": float(np.max(section_cvs)),
                "swap_sensitivity_mean": float(np.mean(swap_sensitivities)),
                "swap_sensitivity_min": float(np.min(swap_sensitivities)),
                "correction_magnitude_mean": float(np.mean(correction_mags)),
                "correction_magnitude_min": float(np.min(correction_mags)),
                "level_distribution": {
                    level.value: levels.count(level)
                    for level in ReliabilityLevel
                },
            }
        else:
            aggregate = {
                "n_evaluated": 0,
                "error": "No geometries could be evaluated",
            }

        # Pass/fail assessment
        healthy = True
        health_notes = []

        if n_eval == 0:
            healthy = False
            health_notes.append("No geometries could be evaluated")
        else:
            if aggregate["rejection_rate"] > 0.20:
                healthy = False
                health_notes.append(
                    f"High rejection rate: {100*aggregate['rejection_rate']:.0f}%"
                )
            if aggregate["eq_residual_mean"] > 1.0:
                healthy = False
                health_notes.append(
                    f"Mean equilibrium residual too high: "
                    f"{aggregate['eq_residual_mean']:.3e}"
                )
            if aggregate["swap_sensitivity_min"] < 0.01:
                healthy = False
                health_notes.append(
                    f"Swap sensitivity too low: "
                    f"{aggregate['swap_sensitivity_min']:.4f}"
                )
            if aggregate["correction_magnitude_min"] < 0.01:
                healthy = False
                health_notes.append(
                    f"Baseline collapse detected: correction_magnitude = "
                    f"{aggregate['correction_magnitude_min']:.4f}"
                )

        if not health_notes:
            health_notes.append("All health checks passed")

        return {
            "timestamp": timestamp,
            "checkpoint_id": checkpoint_id,
            "n_geometries": self.n_benchmark,
            "per_geometry": per_geometry,
            "aggregate": aggregate,
            "pass_fail": {
                "healthy": healthy,
                "notes": health_notes,
            },
        }

    def save_report(
        self,
        report: Dict[str, Any],
        output_dir: str = "docs",
    ) -> str:
        """Save health report as JSON and markdown.

        Returns path to the markdown report.
        """
        os.makedirs(output_dir, exist_ok=True)

        # JSON
        json_path = os.path.join(output_dir, "health_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Markdown
        md_path = os.path.join(output_dir, "reliability_report_v1.md")
        lines = [
            "# PI-GINOT Model Health Report",
            "",
            f"**Timestamp:** {report['timestamp']}",
            f"**Checkpoint:** {report['checkpoint_id']}",
            f"**Benchmark size:** {report['n_geometries']} geometries",
            "",
            "## Summary",
            "",
            f"**Status:** {'HEALTHY' if report['pass_fail']['healthy'] else 'UNHEALTHY'}",
            "",
        ]

        for note in report["pass_fail"]["notes"]:
            lines.append(f"- {note}")

        if "aggregate" in report and "n_evaluated" in report["aggregate"]:
            agg = report["aggregate"]
            lines.extend([
                "",
                "## Aggregate Metrics",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                f"| Geometries evaluated | {agg['n_evaluated']} |",
                f"| Rejection rate | {100*agg.get('rejection_rate', 0):.1f}% |",
                f"| Mean eq. residual | {agg.get('eq_residual_mean', 0):.4e} |",
                f"| Max eq. residual | {agg.get('eq_residual_max', 0):.4e} |",
                f"| Mean section-force CV | {100*agg.get('section_cv_mean', 0):.1f}% |",
                f"| Mean swap sensitivity | {agg.get('swap_sensitivity_mean', 0):.4f} |",
                f"| Min correction magnitude | {agg.get('correction_magnitude_min', 0):.4f} |",
            ])

            if "level_distribution" in agg:
                lines.extend([
                    "",
                    "## Confidence Level Distribution",
                    "",
                    "| Level | Count |",
                    "| --- | --- |",
                ])
                for level, count in agg["level_distribution"].items():
                    lines.append(f"| {level} | {count} |")

        lines.extend([
            "",
            "---",
            "",
            "*This report was generated automatically by the PI-GINOT "
            "reliability agent. All metrics are computed on an independent "
            "verification grid with disjoint seeds from training.*",
        ])

        with open(md_path, "w") as f:
            f.write("\n".join(lines))

        return md_path
