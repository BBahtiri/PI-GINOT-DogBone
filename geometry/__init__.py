"""Geometry generation and collocation point sampling for parametric DogBone specimens."""

from .parametric_dogbone import (
    generate_dogbone, sample_geometry_params, sample_batch,
    plot_dogbone, DogBoneMesh, BoundarySegment,
)
from .collocation import sample_collocation_points, CollocationData, collocation_to_torch
