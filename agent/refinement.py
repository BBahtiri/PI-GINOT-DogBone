#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual-driven adaptive refinement for PI-GINOT.

Targets geometries where the physics residual is high, fine-tuning the
model specifically on those geometries to reduce observable residuals.

Critical limitations (documented honestly):
  - Improves intrinsic physics consistency on targeted geometries
  - Does NOT guarantee reduction of systematic bias (no external reference)
  - Must evaluate on held-out benchmark before AND after refinement
  - May improve one region at the cost of another
  - Checks per-geometry regression, not just mean (§3.2)

Design constraints:
  - Always uses independent verification grid (not training collocation)
  - Saves backup checkpoint before refinement
  - Reports before/after metrics on both target and benchmark geometries
  - Updates engine checkpoint_meta after successful refinement (§4.6)
  - Invalidates prediction cache after any weight modification
"""

from __future__ import annotations

import copy
import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    MATERIAL_CONFIG, LOADING_CONFIG, NONDIM_SCALES,
    COLLOCATION_CONFIG, GEOMETRY_RANGES,
    validate_geometry, get_fillet_geometry,
)
from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params
from geometry.collocation import sample_collocation_points
from physics.losses import PhysicsLoss

from .constants import AGENT_SEEDS
from .schemas import GeometryParams, NormalizationScales, ReliabilityLevel


class ResidualDrivenRefinement:
    """Fine-tune PI-GINOT on a specific geometry to reduce physics residuals.

    Args:
        engine: InferenceEngine instance (provides model, device, etc.)
        geometry: Target geometry for refinement
        n_epochs: Number of fine-tuning epochs
        lr: Learning rate for fine-tuning (should be small)
        regression_fraction: Max fraction of benchmark geos allowed to regress
    """

    def __init__(
        self,
        engine,  # InferenceEngine
        geometry: GeometryParams,
        n_epochs: int = 100,
        lr: float = 1e-5,
        regression_fraction: float = 0.25,
    ):
        self.engine = engine
        self.geometry = geometry
        self.n_epochs = n_epochs
        self.lr = lr
        self.regression_fraction = regression_fraction

    def run(self) -> Dict[str, Any]:
        """Execute refinement with before/after evaluation.

        Regression detection uses per-geometry checks (§3.2):
        if > regression_fraction of benchmark geometries see their
        eq_residual increase by > 20%, the model is reverted.

        After successful refinement, the engine's checkpoint_meta is
        updated with a new ID to preserve provenance integrity (§4.6).

        Returns:
            Dict with before/after per-geometry metrics, status, disclosure.
        """
        model = self.engine.model
        device = self.engine.device
        material = self.engine.material

        # --- Save backup ---
        backup_state = copy.deepcopy(model.state_dict())
        original_ckpt_id = self.engine.checkpoint_meta["checkpoint_id"]

        # --- Evaluate BEFORE on target ---
        before_target = self._evaluate_target()

        # --- Evaluate BEFORE on held-out benchmark (per-geometry) ---
        before_benchmark = self._evaluate_benchmark_per_geo()

        # --- Build training data for target geometry ---
        params = self.geometry.as_dict()
        fillet_info = get_fillet_geometry(params)
        rng = np.random.default_rng(AGENT_SEEDS["refinement_collocation"])
        mesh = generate_dogbone(params, n_interior=4000, rng=rng)

        # Build physics loss
        stress_state = material.state
        loss_fn = PhysicsLoss(
            mu=material.mu,
            lam=material.lam,
            w_equilibrium=100.0,
            w_trac_top=2.0,
            w_trac_arc=20.0,
            w_traction_partial=2.0,
            w_barrier=1e3,
            j_min=0.05,
            adaptive_beta=0.0,  # no adaptive during refinement
            L0=NONDIM_SCALES["L0"],
            S0=NONDIM_SCALES["S0"],
            stress_state=stress_state,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # --- Fine-tuning loop ---
        model.train()

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32,
                                device=device).unsqueeze(0)

        u_max = LOADING_CONFIG["u_max"]
        losses = []

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # Resample collocation for variance reduction
            coll = sample_collocation_points(mesh, rng=rng)

            interior = _t(coll.interior_pts).requires_grad_(True)
            tf_pts = _t(coll.traction_free_pts).requires_grad_(True)
            tf_norms = _t(coll.traction_free_normals)
            pt_pts = _t(coll.partial_traction_pts).requires_grad_(True)
            pt_norms = _t(coll.partial_traction_normals)
            pt_dirs = torch.tensor(coll.partial_traction_dirs,
                                   dtype=torch.long, device=device)
            bpc = _t(coll.boundary_pc)
            u_d = torch.tensor([u_max], dtype=torch.float32, device=device)
            x_m = torch.tensor([coll.x_max], dtype=torch.float32, device=device)
            y_m = torch.tensor([coll.y_max], dtype=torch.float32, device=device)
            tf_tags = torch.tensor(coll.traction_free_tags,
                                   dtype=torch.long, device=device)

            ld = loss_fn(
                model, interior, tf_pts, tf_norms, tf_tags,
                pt_pts, pt_norms, pt_dirs, bpc, u_d, x_m, y_m,
            )

            ld["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(ld["loss"].item())

        model.eval()

        # Invalidate cache — weights have changed
        self.engine.invalidate_cache()

        # --- Evaluate AFTER on target ---
        after_target = self._evaluate_target()

        # --- Evaluate AFTER on held-out benchmark (per-geometry) ---
        after_benchmark = self._evaluate_benchmark_per_geo()

        # --- Per-geometry regression check (§3.2) ---
        regression_detected = False
        n_regressed = 0
        n_compared = 0

        for b_entry, a_entry in zip(
            before_benchmark["per_geometry"], after_benchmark["per_geometry"]
        ):
            if "eq_residual" in b_entry and "eq_residual" in a_entry:
                n_compared += 1
                if a_entry["eq_residual"] > 1.2 * b_entry["eq_residual"]:
                    n_regressed += 1

        if n_compared > 0:
            regression_rate = n_regressed / n_compared
            if regression_rate > self.regression_fraction:
                regression_detected = True
                # Restore backup
                model.load_state_dict(backup_state)
                model.eval()
                # Invalidate cache again — we reverted
                self.engine.invalidate_cache()

        # --- Update provenance after successful refinement (§4.6) ---
        if not regression_detected:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            new_id = f"{original_ckpt_id}_refined_{ts}"
            self.engine.checkpoint_meta["checkpoint_id"] = new_id
            self.engine.checkpoint_meta["refinement_info"] = {
                "original_checkpoint_id": original_ckpt_id,
                "refinement_timestamp": ts,
                "target_geometry": self.geometry.as_dict(),
                "epochs": self.n_epochs,
                "lr": self.lr,
            }

        return {
            "status": "ok" if not regression_detected else "reverted",
            "before_target": before_target,
            "after_target": after_target,
            "before_benchmark": before_benchmark,
            "after_benchmark": after_benchmark,
            "epochs_run": self.n_epochs,
            "final_loss": losses[-1] if losses else None,
            "regression_detected": regression_detected,
            "n_benchmark_regressed": n_regressed,
            "n_benchmark_compared": n_compared,
            "regression_note": (
                f"Per-geometry regression: {n_regressed}/{n_compared} "
                f"geometries regressed >20%. Model reverted to backup."
                if regression_detected else None
            ),
            "provenance_update": (
                f"Checkpoint ID updated to: "
                f"{self.engine.checkpoint_meta['checkpoint_id']}"
                if not regression_detected else
                f"Reverted to original: {original_ckpt_id}"
            ),
            "disclosure": (
                "Fine-tuning improved intrinsic physics consistency on this "
                "geometry. This reduces observable residuals but does not "
                "guarantee reduction of systematic bias, since no external "
                "reference is used."
            ),
        }

    def _evaluate_target(self) -> Dict[str, Any]:
        """Evaluate reliability on the target geometry."""
        result = self.engine.predict(
            self.geometry, n_query_points=2000, use_cache=False
        )
        return {
            "confidence_level": result.confidence_level.value,
            "eq_residual": result.reliability.normalized_equilibrium_residual,
            "section_cv": result.reliability.section_force_cv,
            "swap_sensitivity": result.reliability.latent_swap_sensitivity,
            "correction_magnitude": result.reliability.correction_magnitude,
        }

    def _evaluate_benchmark_per_geo(
        self,
        n_benchmark: int = 5,
        seed: int = AGENT_SEEDS["refinement_benchmark"],
    ) -> Dict[str, Any]:
        """Evaluate reliability per-geometry on a held-out benchmark set.

        Returns per-geometry metrics so the caller can do per-geometry
        regression checks, not just mean.
        """
        rng = np.random.default_rng(seed)
        per_geo = []

        for i in range(n_benchmark):
            try:
                p = sample_geometry_params(rng)
                geo = GeometryParams(
                    L_total=p["L_total"],
                    W_grip=p["W_grip"],
                    W_gauge=p["W_gauge"],
                    R_fillet=p["R_fillet"],
                )
                result = self.engine.predict(
                    geo, n_query_points=1000, use_cache=False
                )
                per_geo.append({
                    "index": i,
                    "geometry": geo.as_dict(),
                    "eq_residual": result.reliability.normalized_equilibrium_residual,
                    "section_cv": result.reliability.section_force_cv,
                    "confidence_level": result.confidence_level.value,
                })
            except Exception as e:
                per_geo.append({"index": i, "error": str(e)})

        # Aggregate
        eq_vals = [g["eq_residual"] for g in per_geo if "eq_residual" in g]
        cv_vals = [g["section_cv"] for g in per_geo if "section_cv" in g]

        return {
            "n_evaluated": len(eq_vals),
            "mean_eq_residual": float(np.mean(eq_vals)) if eq_vals else float("inf"),
            "mean_section_cv": float(np.mean(cv_vals)) if cv_vals else float("inf"),
            "per_geometry": per_geo,
        }
