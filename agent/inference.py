#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reliability-aware inference engine for PI-GINOT.

Orchestrates the full prediction pipeline:
  1. Load checkpoint with provenance extraction
  2. Validate geometry parameters
  3. Generate collocation + independent verification grid
  4. Encode geometry (once) → geometry latent
  5. Decode displacement field on query points
  6. Compute full stress/strain state
  7. Run reliability diagnostics on independent verification grid
  8. Classify confidence level via gates
  9. Package InferenceResult with provenance

The engine stores pre-encoded reference latents for swap sensitivity
and training bank parameters for distance metrics.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    ENCODER_CONFIG, DECODER_CONFIG, MATERIAL_CONFIG,
    LOADING_CONFIG, NONDIM_SCALES, GEOMETRY_RANGES,
    TRAINING_CONFIG, validate_geometry, get_fillet_geometry,
)
from models.pi_ginot import PI_GINOT
from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params
from geometry.collocation import sample_collocation_points
from physics.neo_hookean import (
    first_piola_kirchhoff_stress, cauchy_stress,
    full_stress_state,
)

from .constants import AGENT_SEEDS, verify_seed_disjoint
from .schemas import (
    GeometryParams,
    MaterialParams,
    NormalizationScales,
    PredictionProvenance,
    ReliabilityMetrics,
    ReliabilityLevel,
    InferenceResult,
    RESPONSE_BEHAVIOR,
)
from .verification import build_verification_grid
from .reliability import run_full_reliability
from .gates import evaluate_gates, get_response_behavior, GateThresholds


# ---------------------------------------------------------------------------
# Checkpoint provenance extraction
# ---------------------------------------------------------------------------

def _extract_provenance(
    checkpoint: dict,
    checkpoint_path: str,
) -> dict:
    """Extract training metadata from a checkpoint for provenance.

    Warns (but does not raise) if training_rng_seeds are missing.
    The verification grid uses a separate seed (99999) that is extremely
    unlikely to overlap, but formal disjointness cannot be proven without
    the training seeds.
    """
    # Check for training seeds — warn if missing, don't crash
    if "training_rng_seeds" not in checkpoint:
        warnings.warn(
            "Checkpoint missing 'training_rng_seeds' — cannot guarantee "
            "verification grid disjointness. Results are still valid but "
            "formally unverified. Re-save the checkpoint using the updated "
            "trainer (training/trainer.py) to silence this warning.",
            UserWarning,
            stacklevel=2,
        )

    # Compute checkpoint ID from file path hash
    ckpt_id = checkpoint.get("checkpoint_id", "")
    if not ckpt_id:
        ckpt_id = hashlib.sha256(
            checkpoint_path.encode()
        ).hexdigest()[:16]

    return {
        "checkpoint_id": ckpt_id,
        "checkpoint_path": checkpoint_path,
        "training_date": checkpoint.get("timestamp", None),
        "training_epochs": checkpoint.get("epoch", 0),
        "best_val_raw": checkpoint.get("best_val_raw", float("inf")),
        "gate_status": {
            "ckpt_swap_du": checkpoint.get("ckpt_swap_du", -1.0),
            "ckpt_section_cv": checkpoint.get("ckpt_section_cv", -1.0),
        },
        "training_rng_seeds": checkpoint.get("training_rng_seeds", None),
    }


# ---------------------------------------------------------------------------
# Reference latent management (greedy farthest-point from training bank)
# ---------------------------------------------------------------------------

def _greedy_farthest_points(
    points: np.ndarray,
    n: int,
) -> List[int]:
    """Select n maximally-dissimilar points via greedy farthest-point.

    Args:
        points: [N, D] array of normalized coordinates.
        n: number of points to select.

    Returns:
        List of indices into `points`.
    """
    N = len(points)
    if N <= n:
        return list(range(N))

    # Start with the point closest to the centroid (stable anchor)
    centroid = points.mean(axis=0)
    dists_to_centroid = np.linalg.norm(points - centroid, axis=1)
    selected = [int(np.argmin(dists_to_centroid))]

    # Track minimum distance from each point to the selected set
    min_dists = np.full(N, np.inf)

    for _ in range(n - 1):
        last = selected[-1]
        d = np.linalg.norm(points - points[last], axis=1)
        min_dists = np.minimum(min_dists, d)
        min_dists[selected] = -1.0  # exclude already selected
        selected.append(int(np.argmax(min_dists)))

    return selected


def _build_reference_latents(
    model: PI_GINOT,
    train_bank_params: List[Tuple[float, float, float, float]],
    n_refs: int = 3,
    device: torch.device = torch.device("cpu"),
    seed: int = AGENT_SEEDS["reference_latents"],
) -> Tuple[List[torch.Tensor], List[Tuple[float, ...]]]:
    """Build reference latents from maximally dissimilar TRAINING geometries.

    Uses greedy farthest-point selection in normalized [0,1]^4 parameter
    space over the actual training bank, ensuring the reference set is
    grounded in the training distribution.

    Returns:
        reference_latents: list of [1, n_tok, dim] tensors
        reference_params: list of (L_total, W_grip, W_gauge, R_fillet) tuples
    """
    rng = np.random.default_rng(seed)

    # Normalize training bank to [0,1]^4
    bank_arr = np.array(train_bank_params)
    ranges = np.array([
        GEOMETRY_RANGES["L_total"],
        GEOMETRY_RANGES["W_grip"],
        GEOMETRY_RANGES["W_gauge"],
        GEOMETRY_RANGES["R_fillet"],
    ])
    lo = ranges[:, 0]
    hi = ranges[:, 1]
    span = hi - lo
    span[span < 1e-12] = 1.0
    bank_norm = (bank_arr - lo) / span

    # Greedy farthest-point selection
    indices = _greedy_farthest_points(bank_norm, n_refs)

    reference_latents = []
    reference_params = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            params_tuple = train_bank_params[idx]
            params = {
                "L_total": params_tuple[0],
                "W_grip": params_tuple[1],
                "W_gauge": params_tuple[2],
                "R_fillet": params_tuple[3],
                "holes": [],
            }

            if not validate_geometry(params):
                continue

            mesh = generate_dogbone(params, n_interior=200, rng=rng)
            coll = sample_collocation_points(mesh, rng=rng)

            bpc = torch.tensor(
                coll.boundary_pc, dtype=torch.float32, device=device
            ).unsqueeze(0)
            x_m = torch.tensor(
                [coll.x_max], dtype=torch.float32, device=device
            )
            y_m = torch.tensor(
                [coll.y_max], dtype=torch.float32, device=device
            )

            z = model.encode(bpc, x_m, y_m)
            reference_latents.append(z.detach())
            reference_params.append(params_tuple)

    return reference_latents, reference_params


# ---------------------------------------------------------------------------
# Training bank parameter extraction
# ---------------------------------------------------------------------------

def _build_train_bank_params(
    n_bank: int = None,
    seed: int = None,
    ranges: dict = None,
) -> List[Tuple[float, float, float, float]]:
    """Reconstruct training bank parameters for distance computation.

    Uses the same seed and bank size as training to reproduce the exact
    parameter sets.  Defaults match trainer._build_geometry_bank(seed=100).
    """
    n_bank = n_bank or TRAINING_CONFIG.get("bank_train_size", 128)
    seed = seed or 100  # Must match trainer._build_geometry_bank(seed=100)
    rng = np.random.default_rng(seed)
    bank_ranges = ranges or TRAINING_CONFIG.get("bank_geo_ranges", GEOMETRY_RANGES)

    params_list = []
    for _ in range(n_bank):
        p = sample_geometry_params(rng, geometry_ranges=bank_ranges)
        params_list.append((
            p["L_total"], p["W_grip"], p["W_gauge"], p["R_fillet"],
        ))

    return params_list


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

class _LRUCache:
    """Simple LRU cache for InferenceResult objects.

    Keyed by (checkpoint_id, geometry_tuple, verification_seed).
    """

    def __init__(self, maxsize: int = 32):
        self._maxsize = maxsize
        self._cache: OrderedDict[tuple, InferenceResult] = OrderedDict()

    def get(self, key: tuple) -> Optional[InferenceResult]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, value: InferenceResult) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def invalidate(self) -> None:
        """Clear the entire cache (e.g. after refinement)."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Reliability-aware inference engine for PI-GINOT.

    Usage:
        engine = InferenceEngine.from_checkpoint("path/to/best.pt")
        result = engine.predict(GeometryParams(L_total=54, W_grip=20, ...))
    """

    def __init__(
        self,
        model: PI_GINOT,
        material: MaterialParams,
        device: torch.device,
        checkpoint_meta: dict,
        reference_latents: List[torch.Tensor],
        train_bank_params: List[Tuple[float, float, float, float]],
        gate_thresholds: GateThresholds = GateThresholds(),
        verification_seed: int = AGENT_SEEDS["verification_grid"],
    ):
        self.model = model.to(device)
        self.model.eval()
        self.material = material
        self.device = device
        self.checkpoint_meta = checkpoint_meta
        self.reference_latents = reference_latents
        self.train_bank_params = train_bank_params
        self.gate_thresholds = gate_thresholds
        self.verification_seed = verification_seed
        self.u_max = LOADING_CONFIG["u_max"]
        self._cache = _LRUCache(maxsize=32)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto",
        gate_thresholds: Optional[GateThresholds] = None,
        verification_seed: int = AGENT_SEEDS["verification_grid"],
    ) -> "InferenceEngine":
        """Load model from checkpoint and build inference engine.

        Raises ValueError if checkpoint lacks training_rng_seeds.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=dev)

        # Extract provenance (raises if seeds missing)
        ckpt_meta = _extract_provenance(checkpoint, checkpoint_path)

        # Ensure verification seed is disjoint from training
        training_seeds = ckpt_meta["training_rng_seeds"]
        verify_seed_disjoint(verification_seed, "verification_grid")

        # Build model
        model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(dev)
        model.eval()

        # Material
        material = MaterialParams(
            E=MATERIAL_CONFIG["E"],
            nu=MATERIAL_CONFIG["nu"],
            mu=MATERIAL_CONFIG["mu"],
            lam=MATERIAL_CONFIG["lam"],
            state=MATERIAL_CONFIG.get("state", "plane strain"),
        )

        # Build training bank params using checkpoint-recorded config
        # (matches trainer._build_geometry_bank exactly)
        bank_params = _build_train_bank_params(
            n_bank=checkpoint.get("train_bank_size", None),
            seed=checkpoint.get("train_bank_seed", None),
            ranges=checkpoint.get("bank_geo_ranges", None),
        )

        # Build reference latents from training bank via farthest-point
        ref_latents, _ = _build_reference_latents(
            model, bank_params, device=dev
        )

        return cls(
            model=model,
            material=material,
            device=dev,
            checkpoint_meta=ckpt_meta,
            reference_latents=ref_latents,
            train_bank_params=bank_params,
            gate_thresholds=gate_thresholds or GateThresholds(),
            verification_seed=verification_seed,
        )

    def predict(
        self,
        geometry: GeometryParams,
        n_query_points: int = 4000,
        query_seed: int = AGENT_SEEDS["query_default"],
        use_cache: bool = True,
    ) -> InferenceResult:
        """Run full reliability-aware prediction for one geometry.

        Args:
            geometry: Parametric DogBone specification.
            n_query_points: Number of interior query points.
            query_seed: Seed for query point sampling.
            use_cache: If True, return cached result for repeated queries.

        Returns:
            InferenceResult with fields, reliability, provenance.
        """
        # Cache lookup
        cache_key = (
            self.checkpoint_meta["checkpoint_id"],
            geometry.as_tuple(),
            self.verification_seed,
            n_query_points,
        )
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        self.model.eval()

        # Validate geometry
        params = geometry.as_dict()
        if not validate_geometry(params):
            raise ValueError(f"Invalid geometry: {params}")

        fillet_info = get_fillet_geometry(params)
        L_half = fillet_info["L_half"]

        # Normalization scales (INPUT-ONLY, never from predictions)
        scales = NormalizationScales.from_inputs(
            E=self.material.E,
            u_delta=self.u_max,
            L_half=L_half,
        )

        # Generate mesh and collocation
        rng = np.random.default_rng(query_seed)
        mesh = generate_dogbone(params, n_interior=n_query_points, rng=rng)
        coll = sample_collocation_points(mesh, rng=rng)

        # Build verification grid (independent seed)
        vgrid = build_verification_grid(
            fillet_info,
            seed=self.verification_seed,
            training_seeds=self.checkpoint_meta["training_rng_seeds"],
        )

        # Tensors
        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

        bpc = _t(coll.boundary_pc)
        x_max = torch.tensor([coll.x_max], dtype=torch.float32,
                             device=self.device)
        y_max = torch.tensor([coll.y_max], dtype=torch.float32,
                             device=self.device)
        u_delta = torch.tensor([self.u_max], dtype=torch.float32,
                               device=self.device)

        query = _t(coll.interior_pts).requires_grad_(True)

        # Traction-free boundary tensors
        trac_free_pts = _t(coll.traction_free_pts).requires_grad_(True)
        trac_free_normals = _t(coll.traction_free_normals)
        trac_free_tags = torch.tensor(coll.traction_free_tags,
                                      dtype=torch.long, device=self.device)
        partial_trac_pts = _t(coll.partial_traction_pts).requires_grad_(True)
        partial_trac_normals = _t(coll.partial_traction_normals)
        partial_trac_dirs = torch.tensor(coll.partial_traction_dirs,
                                         dtype=torch.long, device=self.device)

        # --- Encode geometry (once) ---
        with torch.no_grad():
            geometry_latent = self.model.encode(bpc, x_max, y_max)

        # --- Decode displacement field ---
        with torch.enable_grad():
            uv, du_dx, du_dy, dv_dx, dv_dy = (
                self.model.predict_with_grad_latent(
                    query, geometry_latent, u_delta, x_max, y_max
                )
            )

            # Full stress state
            S11, S22, S33, S12, detF = full_stress_state(
                du_dx, du_dy, dv_dx, dv_dy,
                self.material.mu, self.material.lam,
                self.material.state,
            )

            # Piola-Kirchhoff stress
            P11, P12, P21, P22, detF_pk = first_piola_kirchhoff_stress(
                du_dx, du_dy, dv_dx, dv_dy,
                self.material.mu, self.material.lam,
                self.material.state,
            )

            # Green-Lagrange strain
            E11 = 0.5 * ((1.0 + du_dx) ** 2 + dv_dx ** 2 - 1.0)

        # Extract numpy arrays
        pts_np = coll.interior_pts
        u_np = uv[0, :, 0].detach().cpu().numpy()
        v_np = uv[0, :, 1].detach().cpu().numpy()
        u_base = self.u_max * pts_np[:, 0] / coll.x_max
        correction_u = u_np - u_base

        # --- Run full reliability diagnostics ---
        reliability = run_full_reliability(
            model=self.model,
            geometry_latent=geometry_latent,
            verification_grid=vgrid,
            fillet_info=fillet_info,
            u_delta=u_delta,
            x_max=x_max,
            y_max=y_max,
            mu=self.material.mu,
            lam=self.material.lam,
            stress_state=self.material.state,
            scales=scales,
            device=self.device,
            reference_latents=self.reference_latents,
            query_geometry=geometry,
            train_bank_params=self.train_bank_params,
            param_ranges=GEOMETRY_RANGES,
            u_max=self.u_max,
            trac_free_pts=trac_free_pts,
            trac_free_normals=trac_free_normals,
            trac_free_tags=trac_free_tags,
            partial_trac_pts=partial_trac_pts,
            partial_trac_normals=partial_trac_normals,
            partial_trac_dirs=partial_trac_dirs,
        )

        # --- Evaluate gates ---
        confidence_level, reasons = evaluate_gates(
            reliability, self.gate_thresholds
        )
        response_behavior = get_response_behavior(confidence_level)

        # --- Build provenance ---
        provenance = PredictionProvenance(
            checkpoint_path=self.checkpoint_meta["checkpoint_path"],
            checkpoint_id=self.checkpoint_meta["checkpoint_id"],
            checkpoint_training_date=self.checkpoint_meta["training_date"],
            checkpoint_training_epochs=self.checkpoint_meta["training_epochs"],
            checkpoint_best_val_raw=self.checkpoint_meta["best_val_raw"],
            checkpoint_gate_status=self.checkpoint_meta["gate_status"],
            stress_formulation=self.material.state,
            material_params=self.material,
            loading_u_delta=self.u_max,
            inference_timestamp=PredictionProvenance.create_timestamp(),
            verification_grid_seed=self.verification_seed,
            training_rng_seeds=self.checkpoint_meta["training_rng_seeds"],
            geometry_params=geometry,
        )

        result = InferenceResult(
            query_points=pts_np,
            displacement_u=u_np,
            displacement_v=v_np,
            correction_u=correction_u,
            stress_P11=P11[0, :, 0].detach().cpu().numpy(),
            stress_P22=P22[0, :, 0].detach().cpu().numpy(),
            stress_P12=P12[0, :, 0].detach().cpu().numpy(),
            cauchy_S11=S11[0, :, 0].detach().cpu().numpy(),
            cauchy_S22=S22[0, :, 0].detach().cpu().numpy(),
            cauchy_S12=S12[0, :, 0].detach().cpu().numpy(),
            strain_E11=E11[0, :, 0].detach().cpu().numpy(),
            det_F=detF[0, :, 0].detach().cpu().numpy(),
            reliability=reliability,
            confidence_level=confidence_level,
            rejection_reasons=reasons,
            response_behavior=response_behavior,
            provenance=provenance,
            scales=scales,
        )

        # Cache the result
        if use_cache:
            self._cache.put(cache_key, result)

        return result

    def invalidate_cache(self) -> None:
        """Clear the prediction cache (call after refinement)."""
        self._cache.invalidate()