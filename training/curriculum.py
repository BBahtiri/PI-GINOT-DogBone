#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry curriculum scheduler for PI-GINOT training.

Progressively widens the geometry parameter ranges and enables holes
as training advances, following the phase definitions in config.py.
"""

from typing import Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CURRICULUM_CONFIG, GEOMETRY_RANGES


class CurriculumScheduler:
    """Returns geometry sampling parameters based on the current epoch.

    When curriculum is disabled, always returns the full GEOMETRY_RANGES.

    Args:
        config: Dict matching CURRICULUM_CONFIG from config.py.
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = CURRICULUM_CONFIG
        self.enabled = config.get("enabled", False)
        self.phases = config.get("phases", [])

    def get_params(self, epoch: int) -> Tuple[dict, bool]:
        """Return (geometry_ranges, holes_enabled) for the given epoch.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            geometry_ranges: Dict of (min, max) per geometry parameter.
            holes_enabled:   Whether to sample holes for this epoch.
        """
        if not self.enabled or not self.phases:
            return GEOMETRY_RANGES, False

        # Find the active phase for this epoch
        for phase in self.phases:
            lo, hi = phase["epoch_range"]
            if lo <= epoch < hi:
                return phase["geometry_ranges"], phase.get("holes_enabled", False)

        # Past all defined phases → use the last phase's settings
        last = self.phases[-1]
        return last["geometry_ranges"], last.get("holes_enabled", False)

    def phase_name(self, epoch: int) -> str:
        """Return the human-readable name of the active phase."""
        if not self.enabled or not self.phases:
            return "full"
        for phase in self.phases:
            lo, hi = phase["epoch_range"]
            if lo <= epoch < hi:
                return phase["name"]
        return self.phases[-1]["name"]
