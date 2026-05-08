#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central constants and seed registry for the PI-GINOT agent.

All reserved RNG seeds are declared here to prevent collisions between
training, verification, and diagnostic subsystems.  Any new seed usage
MUST be registered here.

The verification grid's independence guarantee depends on its seed being
disjoint from every training seed.  This registry is the single source
of truth for that invariant.

Seed inventory was derived by grepping for `default_rng` and `seed=`
across trainer.py, main.py, and all diagnostic methods.
"""

from __future__ import annotations

from typing import Dict, FrozenSet


# ---------------------------------------------------------------------------
# Training seeds (recorded in checkpoints, must never be reused by agent)
# ---------------------------------------------------------------------------

# Every seed used by trainer.py, main.py, or any training-time code path.
# Organized by call site with line references.
TRAINING_SEEDS: Dict[str, int] = {
    # trainer.__init__  (L106)
    "main_rng":                 42,
    # trainer.single_geometry_mode  (L142)
    "fixed_geometry":            0,
    # trainer._build_geometry_bank for train  (L167, L213)
    "train_bank":              100,
    # trainer._build_geometry_bank for val no-hole  (L183, L213)
    "val_bank_nohole":         200,
    # trainer._save_geometry_plot bank loop  (L259)
    "plot_rng_bank":           999,
    # trainer._build_validation_set  (L477)
    "val_set_builder":       12345,
    # trainer._save_field_plots  (L1245) — uses epoch as seed (0..N)
    # We register the range as a sentinel; see TRAINING_SEED_RANGES below.
    # trainer._save_profile_plots  (L1364)
    "profile_dense_rng":       999,
    # trainer.run_boundary_resampling_diagnostic  (L1576)
    # Uses 1000+k for k=0..n_trials; see TRAINING_SEED_RANGES.
    # trainer.run_boundary_permutation_diagnostic  (L1638)
    # Uses 2000+k for k=0..n_trials; see TRAINING_SEED_RANGES.
    # main.py inspect_initial_state  (L45)
    "main_inspect":            123,
    # main.py geometry bank preview  (L56)
    "main_bank_preview":       100,
}

# Ranges of seeds used during training diagnostics.
# No agent seed may fall within any of these [lo, hi] intervals.
TRAINING_SEED_RANGES = [
    (0, 2000),      # field_plot epochs (0..N), resampling (1000+k), permutation (2000+k)
]

# Flat set for fast membership checks (excludes ranges — those are checked separately)
TRAINING_SEED_SET: FrozenSet[int] = frozenset(TRAINING_SEEDS.values())


def _in_training_range(seed: int) -> bool:
    """Check if seed falls within any training diagnostic range."""
    for lo, hi in TRAINING_SEED_RANGES:
        if lo <= seed <= hi:
            return True
    return False


# ---------------------------------------------------------------------------
# Agent seeds (must be disjoint from TRAINING_SEED_SET and ranges)
# ---------------------------------------------------------------------------

AGENT_SEEDS: Dict[str, int] = {
    "verification_grid":        99999,
    "health_check_benchmark":   88888,
    "reference_latents":        77777,
    "optimize_rng":             54321,
    "refinement_collocation":   33333,
    # Intentionally same as health_check — refinement evaluates on the
    # exact same benchmark set so before/after comparisons are consistent
    # with the health report.
    "refinement_benchmark":     88888,
    # Changed from 12345 to avoid collision with val_set_builder
    "query_default":            67890,
}

# Verify no collisions with training seeds at import time
_point_collisions = TRAINING_SEED_SET & frozenset(AGENT_SEEDS.values())
assert not _point_collisions, (
    f"Agent seed collision with training seeds: {_point_collisions}. "
    f"Pick different values in AGENT_SEEDS."
)

# Verify no agent seed falls in a training range
for _name, _seed in AGENT_SEEDS.items():
    assert not _in_training_range(_seed), (
        f"Agent seed '{_name}' = {_seed} falls within training range "
        f"{TRAINING_SEED_RANGES}. Pick a value outside these ranges."
    )


def all_training_seeds_list() -> list:
    """Return a flat list of training seeds for checkpoint storage."""
    return sorted(set(TRAINING_SEEDS.values()))


def verify_seed_disjoint(seed: int, context: str = "") -> None:
    """Assert that a seed is not in the training set or any training range.

    Raises AssertionError with a descriptive message if violated.
    """
    assert seed not in TRAINING_SEED_SET, (
        f"Seed {seed} ({context}) collides with training seed set "
        f"{TRAINING_SEED_SET}. Pick a disjoint value."
    )
    assert not _in_training_range(seed), (
        f"Seed {seed} ({context}) falls within training diagnostic range "
        f"{TRAINING_SEED_RANGES}. Pick a value outside these ranges."
    )


# ---------------------------------------------------------------------------
# Training section slice positions (from trainer.py)
# Verification slices MUST be disjoint from these.
# ---------------------------------------------------------------------------

TRAINING_SECTION_XI = (0.05, 0.20, 0.42, 0.65, 0.82, 0.95)
VERIFICATION_SECTION_XI = (0.10, 0.35, 0.55, 0.75, 0.90)

_overlap = set(TRAINING_SECTION_XI) & set(VERIFICATION_SECTION_XI)
assert not _overlap, (
    f"Verification section slices overlap training slices: {_overlap}"
)
