#!/usr/bin/env python3
"""
LangChain tool wrappers around the PI-GINOT inference stack.

Each tool returns (message_str, [custom_outputs]) — the message goes into
the LLM conversation and the custom_outputs are rendered by Streamlit.

Includes:
- Predict_DogBone: reliability-aware field prediction
- Optimize_Geometry: reliability-gated geometry search
- Refine_Model: fine-tuning with regression protection
- Run_Health_Check: benchmark model diagnostics
- Visualize_Geometry: geometry outline rendering
- Save_Geometry: persist named geometry to user's library (FIX #8)
- Recall_Geometry: recall a saved geometry by name (FIX #8)
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from typing import Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent import PI_GINOT_Agent, GeometryParams
from config import validate_geometry, get_fillet_geometry

from .network_components import (
    FieldPlotOutput,
    TableOutput,
    GeometryPlotOutput,
    DiagnosticsOutput,
    OptimizationTraceOutput,
)



_PI_AGENT: Optional[PI_GINOT_Agent] = None


def init_pi_agent(checkpoint_path: str, device: str = "auto"):
    """Initialize the PI-GINOT agent singleton."""
    global _PI_AGENT
    _PI_AGENT = PI_GINOT_Agent(checkpoint_path, device=device)
    return _PI_AGENT


def get_pi_agent() -> PI_GINOT_Agent:
    """Get the initialized PI-GINOT agent."""
    if _PI_AGENT is None:
        raise RuntimeError("init_pi_agent() not called")
    return _PI_AGENT



_LTM = None


def set_ltm(ltm):
    """Set the long-term memory reference for geometry library tools."""
    global _LTM
    _LTM = ltm


def get_ltm():
    """Get the current long-term memory instance."""
    return _LTM



class GeometryInput(BaseModel):
    L_total: float = Field(..., description="Full specimen length [mm] (40-70)")
    W_grip: float = Field(..., description="Full grip width [mm] (16-26)")
    W_gauge: float = Field(..., description="Full gauge width [mm] (6-14)")
    R_fillet: float = Field(..., description="Fillet radius [mm] (8-20)")


class OptimizeInput(BaseModel):
    objective: str = Field(
        "minimize_peak_sigma11",
        description="'minimize_peak_sigma11' or 'maximize_correction_range'",
    )
    n_candidates: int = Field(30, description="Number of geometries to evaluate")
    target_strain: Optional[float] = Field(
        None, description="Target strain (applies linear scaling)"
    )


class RefineInput(GeometryInput):
    n_epochs: int = Field(100)
    lr: float = Field(1e-5)


class SaveGeometryInput(BaseModel):
    name: str = Field(..., description="Name to save the geometry under")
    L_total: float = Field(..., description="Full specimen length [mm]")
    W_grip: float = Field(..., description="Full grip width [mm]")
    W_gauge: float = Field(..., description="Full gauge width [mm]")
    R_fillet: float = Field(..., description="Fillet radius [mm]")
    notes: str = Field("", description="Optional notes about this geometry")


class RecallGeometryInput(BaseModel):
    name: str = Field(..., description="Name of the geometry to recall")



@tool("Predict_DogBone", args_schema=GeometryInput)
def predict_dogbone(L_total: float, W_grip: float, W_gauge: float, R_fillet: float):
    """Run PI-GINOT prediction on a DogBone geometry. Returns displacement/stress/strain
    fields with reliability diagnostics and a confidence level (reject/low/medium/high).

    Use this when the user provides geometry parameters or asks for a prediction.
    The output includes a scatter plot of sigma_11 and a diagnostics card."""
    agent = get_pi_agent()
    geo = GeometryParams(L_total=L_total, W_grip=W_grip, W_gauge=W_gauge, R_fillet=R_fillet)

    try:
        result = agent.predict(geo)
    except Exception as e:
        return f"Prediction failed: {e}"

    level = result["confidence_level"]

    summary = (
        f"Prediction complete for DogBone "
        f"(L={L_total}, W_grip={W_grip}, W_gauge={W_gauge}, R={R_fillet}).\n"
        f"Confidence: **{level.upper()}**\n\n"
        f"{result['summary']}"
    )

    outputs = []

    # Stress-strain field plots (only if not rejected)
    if level != "reject":
        raw = result["raw_result"]
        pts = raw.query_points
        stride = max(1, len(pts) // 500)  # cap at ~500 points for rendering

        # Define all fields to plot: (attribute, display_name, unit)
        field_specs = [
            ("cauchy_S11", "σ₁₁ (Cauchy)", "MPa"),
            ("cauchy_S22", "σ₂₂ (Cauchy)", "MPa"),
            ("cauchy_S12", "σ₁₂ (Cauchy)", "MPa"),
            ("strain_E11", "E₁₁ (Green-Lagrange)", "—"),
        ]

        for attr, display_name, unit in field_specs:
            field_data = getattr(raw, attr, None)
            if field_data is None:
                continue
            outputs.append(FieldPlotOutput(
                title=f"{display_name} [{unit}] — {level.upper()} confidence",
                points=[{"x": float(p[0]), "y": float(p[1]), "value": float(v)}
                        for p, v in zip(pts[::stride], field_data[::stride])],
                field_name=attr,
                unit=unit,
                confidence_level=level,
            ))

    # Diagnostics card (always)
    outputs.append(DiagnosticsOutput(
        confidence_level=level,
        metrics=result["diagnostics"],
        rejection_reasons=result["rejection_reasons"],
        response_behavior=result["response_behavior"],
    ))

    # Table of field statistics
    if level != "reject":
        outputs.append(TableOutput(
            name="Field Statistics",
            data=[{"field": k, "value": v} for k, v in result["fields"].items()],
        ))

    return summary, outputs


@tool("Optimize_Geometry", args_schema=OptimizeInput)
def optimize_geometry(objective: str, n_candidates: int, target_strain: Optional[float]):
    """Run reliability-gated geometry optimization via random search.

    Use this when the user wants to find the best DogBone geometry for a given objective.
    Returns the best geometry, the objective value, and a trace of all candidates."""
    agent = get_pi_agent()
    try:
        result = agent.optimize(
            objective=objective,
            n_candidates=n_candidates,
            target_strain=target_strain,
        )
    except Exception as e:
        return f"Optimization failed: {e}"

    if result["status"] == "error":
        return result["message"]

    msg = (
        f"Optimization complete.\n"
        f"**Best geometry**: L={result['best_geometry']['L_total']:.1f}, "
        f"W_grip={result['best_geometry']['W_grip']:.1f}, "
        f"W_gauge={result['best_geometry']['W_gauge']:.1f}, "
        f"R={result['best_geometry']['R_fillet']:.1f} mm\n"
        f"**Objective** ({objective}): {result['best_objective_value']:.3f}\n"
        f"**Confidence**: {result['best_confidence_level']}\n"
        f"**Evaluated**: {result['n_candidates_evaluated']} valid / "
        f"{result['n_candidates_rejected']} rejected\n\n"
        f"*{result['scaling_disclosure']}*"
    )

    outputs = [
        OptimizationTraceOutput(
            iterations=[{
                "L_total": c["geometry"].L_total,
                "W_grip": c["geometry"].W_grip,
                "W_gauge": c["geometry"].W_gauge,
                "R_fillet": c["geometry"].R_fillet,
                "objective": c["objective_raw"],
                "confidence": c["confidence_level"],
            } for c in result.get("all_candidates", [])],
            objective=objective,
            best_geometry=result["best_geometry"],
        ),
        TableOutput(
            name="Top 10 Candidates",
            data=[{
                "rank": i + 1,
                "L": f"{c['geometry'].L_total:.1f}",
                "W_grip": f"{c['geometry'].W_grip:.1f}",
                "W_gauge": f"{c['geometry'].W_gauge:.1f}",
                "R": f"{c['geometry'].R_fillet:.1f}",
                "objective": f"{c['objective_raw']:.3f}",
                "confidence": c["confidence_level"],
            } for i, c in enumerate(result.get("all_candidates", [])[:10])],
        ),
    ]

    return msg, outputs


@tool("Refine_Model", args_schema=RefineInput)
def refine_model(L_total: float, W_grip: float, W_gauge: float, R_fillet: float,
                 n_epochs: int, lr: float):
    """Fine-tune PI-GINOT on a specific geometry to reduce physics residuals.

    Use this ONLY when a prediction came back with LOW or REJECT confidence and the user
    explicitly wants to refine. The model has per-geometry regression protection."""
    agent = get_pi_agent()
    geo = GeometryParams(L_total=L_total, W_grip=W_grip, W_gauge=W_gauge, R_fillet=R_fillet)
    try:
        result = agent.refine(geo, n_epochs=n_epochs, lr=lr)
    except Exception as e:
        return f"Refinement failed: {e}"

    if result["regression_detected"]:
        msg = (
            f"⚠️ Refinement reverted: {result['n_benchmark_regressed']}/"
            f"{result['n_benchmark_compared']} benchmark geometries regressed >20%.\n"
            f"{result['regression_note']}"
        )
    else:
        before = result["before_target"]["eq_residual"]
        after = result["after_target"]["eq_residual"]
        msg = (
            f"✓ Refinement successful.\n"
            f"- Target eq_residual: {before:.3e} → {after:.3e}\n"
            f"- Confidence: {result['before_target']['confidence_level']} → "
            f"{result['after_target']['confidence_level']}\n"
            f"- {result['provenance_update']}\n\n"
            f"*{result['disclosure']}*"
        )

    return msg, []


@tool("Run_Health_Check")
def run_health_check():
    """Run benchmark health check on a fixed set of geometries. Use this when the user
    asks about overall model quality or reliability trends."""
    agent = get_pi_agent()
    try:
        report = agent.health_check(n_benchmark=10)
    except Exception as e:
        return f"Health check failed: {e}"

    agg = report["aggregate"]
    status = "HEALTHY" if report["pass_fail"]["healthy"] else "UNHEALTHY"
    msg = (
        f"Model Health: **{status}**\n"
        f"- Evaluated: {agg.get('n_evaluated', 0)} benchmarks\n"
        f"- Rejection rate: {100 * agg.get('rejection_rate', 0):.1f}%\n"
        f"- Mean eq residual: {agg.get('eq_residual_mean', 0):.3e}\n"
        f"- Notes: {'; '.join(report['pass_fail']['notes'])}"
    )

    outputs = [
        TableOutput(
            name="Per-Geometry Results",
            data=[{
                "idx": r["index"],
                "confidence": r.get("confidence_level", "—"),
                "eq_residual": f"{r.get('eq_residual', 0):.2e}",
                "section_cv": f"{100 * r.get('section_cv', 0):.1f}%",
                "swap": f"{r.get('swap_sensitivity', 0):.3f}",
            } for r in report["per_geometry"] if "error" not in r],
        ),
    ]
    return msg, outputs


@tool("Visualize_Geometry", args_schema=GeometryInput)
def visualize_geometry(L_total: float, W_grip: float, W_gauge: float, R_fillet: float):
    """Show the DogBone geometry outline without running a prediction.
    Useful when the user wants to see what a geometry looks like."""
    params = {
        "L_total": L_total, "W_grip": W_grip,
        "W_gauge": W_gauge, "R_fillet": R_fillet, "holes": [],
    }
    if not validate_geometry(params):
        return f"Invalid geometry: {params}"

    from geometry.parametric_dogbone import generate_dogbone
    import numpy as np

    rng = np.random.default_rng(0)
    mesh = generate_dogbone(params, n_interior=200, rng=rng)

    bnd = [{"x": float(p[0]), "y": float(p[1])} for p in mesh.boundary_pc]
    interior = [{"x": float(p[0]), "y": float(p[1])} for p in mesh.interior_nodes[:150]]

    return (
        f"Geometry visualized. L_half={mesh.fillet_info['L_half']:.1f}, "
        f"H_grip={mesh.fillet_info['H_grip']:.1f} mm."
    ), [
        GeometryPlotOutput(
            params=params, boundary_points=bnd, interior_points=interior
        ),
    ]



@tool("Save_Geometry", args_schema=SaveGeometryInput)
def save_geometry(name: str, L_total: float, W_grip: float, W_gauge: float,
                  R_fillet: float, notes: str = ""):
    """Save a DogBone geometry to the user's library under a given name.

    Use this when the user wants to remember a geometry for later —
    e.g., after optimization finds a good candidate, or the user has a
    standard geometry they'll re-use across sessions."""
    ltm = get_ltm()
    if ltm is None:
        return "Memory system not available. Cannot save geometry.", []

    params = {
        "L_total": L_total, "W_grip": W_grip,
        "W_gauge": W_gauge, "R_fillet": R_fillet,
    }
    ltm.save_geometry(name, params, notes)
    return (
        f"✓ Saved geometry '{name}': L={L_total}, W_grip={W_grip}, "
        f"W_gauge={W_gauge}, R={R_fillet} mm."
        + (f"\n  Notes: {notes}" if notes else "")
    ), []


@tool("Recall_Geometry", args_schema=RecallGeometryInput)
def recall_geometry(name: str):
    """Recall a saved DogBone geometry by name from the user's library.

    Use this when the user references a geometry by name (e.g., 'my optimized one',
    'the standard specimen') instead of providing explicit dimensions."""
    ltm = get_ltm()
    if ltm is None:
        return "Memory system not available. Cannot recall geometry.", []

    geo = ltm.get_geometry(name)
    if geo is None:
        # List available geometries for the user
        available = ltm.list_geometries()
        if available:
            return (
                f"No geometry named '{name}' in library. "
                f"Available: {', '.join(available)}"
            ), []
        return f"No geometry named '{name}' in library (library is empty).", []

    p = geo["params"]
    msg = (
        f"Recalled '{name}': L_total={p['L_total']}, W_grip={p['W_grip']}, "
        f"W_gauge={p['W_gauge']}, R_fillet={p['R_fillet']} mm"
    )
    if geo.get("notes"):
        msg += f"\n  Notes: {geo['notes']}"
    msg += f"\n  Saved: {geo.get('saved_at', 'unknown')}"
    return msg, []



ALL_TOOLS = [
    predict_dogbone,
    optimize_geometry,
    refine_model,
    run_health_check,
    visualize_geometry,
    save_geometry,
    recall_geometry,
]

# Subset for agents that don't need memory tools
PREDICTION_TOOLS = [predict_dogbone, visualize_geometry, save_geometry, recall_geometry]
OPTIMIZATION_TOOLS = [optimize_geometry, predict_dogbone]
DIAGNOSTICS_TOOLS = [predict_dogbone, refine_model, run_health_check]
