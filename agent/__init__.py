"""
PI-GINOT Reliability-Aware Agent Package.

Provides a reliability-aware inference interface for the Physics-Informed
Geometry-Informed Neural Operator Transformer (PI-GINOT) on parametric
DogBone specimens.

Quick start:
    from agent import PI_GINOT_Agent, GeometryParams

    agent = PI_GINOT_Agent("checkpoints/best.pt")
    result = agent.predict(GeometryParams(
        L_total=54.0, W_grip=20.0, W_gauge=10.0, R_fillet=12.0
    ))
    print(agent.interpret(result))

Architecture:
    constants.py     → Seed registry, section slice positions
    schemas.py       → Data models (ReliabilityMetrics, InferenceResult, ...)
    verification.py  → Independent verification grid (disjoint from training)
    reliability.py   → Normalized residual computation + diagnostics
    gates.py         → Confidence level classification (reject/low/medium/high)
    inference.py     → Model loading, encode-once inference, full pipeline
    agent.py         → Agent orchestration (predict/interpret/optimize/refine)
    refinement.py    → Residual-driven adaptive fine-tuning
    health_check.py  → Benchmark-based model health monitoring
"""

from .constants import (
    AGENT_SEEDS,
    TRAINING_SEED_SET,
    verify_seed_disjoint,
)

from .schemas import (
    ReliabilityLevel,
    ReliabilityMetrics,
    PredictionProvenance,
    InferenceResult,
    GeometryParams,
    MaterialParams,
    NormalizationScales,
    RESPONSE_BEHAVIOR,
)

from .gates import (
    GateThresholds,
    evaluate_gates,
    get_response_behavior,
)

from .inference import InferenceEngine

from .agent import PI_GINOT_Agent

__all__ = [
    # Main agent
    "PI_GINOT_Agent",
    # Inference engine
    "InferenceEngine",
    # Data models
    "ReliabilityLevel",
    "ReliabilityMetrics",
    "PredictionProvenance",
    "InferenceResult",
    "GeometryParams",
    "MaterialParams",
    "NormalizationScales",
    "RESPONSE_BEHAVIOR",
    # Gates
    "GateThresholds",
    "evaluate_gates",
    "get_response_behavior",
    # Constants
    "AGENT_SEEDS",
    "TRAINING_SEED_SET",
    "verify_seed_disjoint",
]
