#!/usr/bin/env python3
"""
PI-GINOT Design Studio — Live interactive geometry exploration (Enhanced).

Premium features:
- Race the Optimizer: 30-second human-vs-AI challenge
- Reliability Heatmap: spatial view of trust across parameter space
- "What the Agent Sees" overlay: verification slices, training bank proximity
- Comparison mode: side-by-side geometry delta view
- Session stats dashboard
- Auto-detected stress hotspot callouts
- Parameter radar chart (geometry vs training bank)
- Shareable design links
"""

import streamlit as st
import sys
import os
import uuid
import time
import math
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents import tools
from agent.schemas import GeometryParams
from config import GEOMETRY_RANGES, GEOMETRY_DEFAULT, validate_geometry, get_fillet_geometry


# Enhanced CSS
st.markdown("""
<style>
    /* Confidence pills */
    .conf-pill {
        display: inline-block; padding: 4px 12px; border-radius: 12px;
        font-weight: 600; font-size: 0.85em;
    }
    .conf-high { background: #d1fae5; color: #065f46; }
    .conf-medium { background: #fef3c7; color: #92400e; }
    .conf-low { background: #fed7aa; color: #9a3412; }
    .conf-reject { background: #fee2e2; color: #991b1b; }

    /* Design card */
    .design-card {
        border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px;
        background: linear-gradient(135deg, #f9fafb 0%, #ffffff 100%);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Hero metric */
    .hero-delta {
        font-size: 2.5em; font-weight: 700;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }

    /* Session stats */
    .session-stat {
        display: inline-block; padding: 6px 12px; margin: 2px;
        background: #f3f4f6; border-radius: 8px;
        font-size: 0.85em; color: #374151;
    }
    .session-stat b { color: #111827; }

    /* Race the optimizer */
    .race-timer {
        font-size: 3em; font-weight: 800;
        font-family: 'Courier New', monospace;
        color: #dc2626;
        text-align: center;
    }
    .race-score {
        font-size: 1.2em; font-weight: 600;
        padding: 10px; border-radius: 8px;
        text-align: center;
    }
    .race-user { background: #dbeafe; color: #1e40af; }
    .race-ai { background: #fef3c7; color: #92400e; }
    .race-winner {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        animation: pulse-gold 1.5s infinite;
    }
    @keyframes pulse-gold {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    /* Hotspot callout */
    .hotspot-badge {
        display: inline-block; padding: 3px 10px; border-radius: 10px;
        background: #fee2e2; color: #991b1b; font-weight: 600;
        font-size: 0.85em; animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* Slider labels bolder */
    .stSlider label { font-weight: 600 !important; }

    /* Feature tabs */
    .feature-tab {
        padding: 8px 16px; margin: 2px; border-radius: 8px;
        cursor: pointer; border: 1px solid #e5e7eb;
        background: white; font-weight: 500;
    }
    .feature-tab.active {
        background: #3b82f6; color: white; border-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)



def confidence_pill(level):
    """Render a colored confidence badge."""
    icons = {"high": "🟢", "medium": "🟡", "low": "🟠", "reject": "🔴"}
    return (
        f'<span class="conf-pill conf-{level}">{icons.get(level, "⚪")} '
        f'{level.upper()}</span>'
    )


def _draw_outline(fig, L, W_grip, W_gauge, R, color="#1f2937", width=2):
    """Draw the dogbone quarter-model outline as a line overlay."""
    L_half = L / 2
    H_grip = W_grip / 2
    H_gauge = W_gauge / 2
    dH = H_grip - H_gauge
    if R <= dH:
        return
    dx = math.sqrt(dH * (2 * R - dH))
    x_g = L_half - dx
    arc_cy = H_gauge + R

    xs = [0, L_half, L_half]
    ys = [0, 0, H_grip]

    theta_start = math.atan2(dH - R, dx)
    theta_end = -math.pi / 2
    for t in np.linspace(theta_start, theta_end, 30):
        xs.append(x_g + R * math.cos(t))
        ys.append(arc_cy + R * math.sin(t))

    xs.extend([0, 0])
    ys.extend([H_gauge, 0])

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=color, width=width),
        showlegend=False, hoverinfo="skip",
    ))


def _find_hotspot(pts_x, pts_y, values):
    """Return coords of peak absolute value point."""
    if len(values) == 0:
        return None, None, None
    idx = int(np.argmax(np.abs(values)))
    return pts_x[idx], pts_y[idx], values[idx]


def _build_commentary(result, L, W_grip, W_gauge, R):
    """Generate contextual commentary based on current geometry + result."""
    level = result["confidence"]
    sigma = result["peak_sigma_11"]
    cv = result["section_cv"]

    lines = []

    if level == "high":
        lines.append("✅ **High confidence** — this prediction is trustworthy.")
    elif level == "medium":
        lines.append(
            "⚠️ **Moderate confidence** — use these numbers with caveats."
        )
    elif level == "low":
        lines.append(
            "🟠 **Low confidence** — only trends are reliable, not absolute values."
        )
    else:
        lines.append("🚫 **Rejected** — this geometry breaks a reliability gate.")

    if level != "reject":
        dH = (W_grip - W_gauge) / 2
        K_t_est = 1 + 2 * np.sqrt(dH / R) if R > 0 else 1
        lines.append(
            f"\n📐 **Stress concentration:** Estimated K_t ≈ {K_t_est:.2f}."
        )

        if R < 10:
            lines.append("\n💡 **Tip:** Small R → sharp concentrations. Try R > 12.")
        elif R > 17:
            lines.append("\n💡 **Tip:** Large R is gentle but costs gauge length.")

    if cv > 0.1:
        lines.append(
            f"\n⚠️ **Section force imbalance** ({100*cv:.1f}%) — model inconsistency."
        )

    return "\n".join(lines)


def _geometry_hash(L, W_grip, W_gauge, R):
    """Short hash for geometry identity."""
    s = f"{L:.2f}_{W_grip:.2f}_{W_gauge:.2f}_{R:.2f}"
    return hashlib.md5(s.encode()).hexdigest()[:6]



if "studio_sliders" not in st.session_state:
    st.session_state.studio_sliders = {
        "L_total": GEOMETRY_DEFAULT["L_total"],
        "W_grip": GEOMETRY_DEFAULT["W_grip"],
        "W_gauge": GEOMETRY_DEFAULT["W_gauge"],
        "R_fillet": GEOMETRY_DEFAULT["R_fillet"],
    }

if "saved_designs" not in st.session_state:
    st.session_state.saved_designs = []

if "baseline_result" not in st.session_state:
    st.session_state.baseline_result = None

if "session_stats" not in st.session_state:
    st.session_state.session_stats = {
        "n_predictions": 0,
        "n_high": 0,
        "n_medium": 0,
        "n_low": 0,
        "n_reject": 0,
        "best_sigma": float("inf"),
        "best_params": None,
        "hottest_kt": 0,
        "hottest_params": None,
        "start_time": time.time(),
    }

# Race the Optimizer state
if "race_mode" not in st.session_state:
    st.session_state.race_mode = False
if "race_started_at" not in st.session_state:
    st.session_state.race_started_at = None
if "race_user_best" not in st.session_state:
    st.session_state.race_user_best = None
if "race_ai_best" not in st.session_state:
    st.session_state.race_ai_best = None
if "race_history" not in st.session_state:
    st.session_state.race_history = []

# Comparison mode
if "compare_mode" not in st.session_state:
    st.session_state.compare_mode = False
if "compare_snapshot" not in st.session_state:
    st.session_state.compare_snapshot = None

# Overlay toggles
if "show_overlay" not in st.session_state:
    st.session_state.show_overlay = False
if "show_hotspot" not in st.session_state:
    st.session_state.show_hotspot = True

# Parse URL params for shared links
query_params = st.query_params
if "L" in query_params and "first_load" not in st.session_state:
    try:
        st.session_state.studio_sliders = {
            "L_total": float(query_params["L"]),
            "W_grip": float(query_params["Wg"]),
            "W_gauge": float(query_params["Wga"]),
            "R_fillet": float(query_params["R"]),
        }
        st.session_state.first_load = True
    except (ValueError, KeyError):
        pass

# Initialize PI-GINOT
if "pi_agent_initialized" not in st.session_state:
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "checkpoints", "best.pt"
    )
    try:
        with st.spinner("Loading PI-GINOT..."):
            tools.init_pi_agent(checkpoint_path, device="auto")
        st.session_state.pi_agent_initialized = True
        st.session_state.pi_agent_error = None
    except Exception as e:
        st.session_state.pi_agent_initialized = True
        st.session_state.pi_agent_error = str(e)



@st.cache_data(show_spinner=False, max_entries=500)
def cached_predict(L_total: float, W_grip: float, W_gauge: float, R_fillet: float):
    """Cache predictions by rounded geometry params."""
    try:
        agent = tools.get_pi_agent()
        geo = GeometryParams(
            L_total=L_total, W_grip=W_grip, W_gauge=W_gauge, R_fillet=R_fillet
        )
        result = agent.predict(geo, n_query_points=2000)

        raw = result["raw_result"]
        return {
            "status": "ok",
            "confidence": result["confidence_level"],
            "rejection_reasons": result["rejection_reasons"],
            "x": raw.query_points[:, 0].tolist(),
            "y": raw.query_points[:, 1].tolist(),
            "sigma_11": raw.cauchy_S11.tolist(),
            "u": raw.displacement_u.tolist(),
            "peak_sigma_11": float(np.max(np.abs(raw.cauchy_S11))),
            "peak_u": float(np.max(np.abs(raw.displacement_u))),
            "eq_residual": result["diagnostics"]["normalized_equilibrium_residual"],
            "section_cv": result["diagnostics"]["section_force_cv"],
            "swap_sens": result["diagnostics"]["latent_swap_sensitivity"],
            "correction_mag": result["diagnostics"]["correction_magnitude"],
            "inside_box": result["diagnostics"]["inside_training_box"],
            "nn_distance": result["diagnostics"]["geometry_nn_distance"],
            "mahalanobis": result["diagnostics"].get("geometry_mahalanobis", 0),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@st.cache_data(show_spinner=False, max_entries=100)
def compute_reliability_heatmap(L_total: float, W_grip: float,
                                R_range: tuple, Wga_range: tuple,
                                n_grid: int = 12):
    """Compute a grid of (R_fillet, W_gauge) → (confidence, peak_sigma)."""
    R_vals = np.linspace(R_range[0], R_range[1], n_grid)
    Wga_vals = np.linspace(Wga_range[0], Wga_range[1], n_grid)

    conf_map = {"high": 3, "medium": 2, "low": 1, "reject": 0}
    confidence_grid = np.full((n_grid, n_grid), -1, dtype=float)
    sigma_grid = np.full((n_grid, n_grid), np.nan)

    for i, R in enumerate(R_vals):
        for j, Wga in enumerate(Wga_vals):
            params = {"L_total": L_total, "W_grip": W_grip,
                      "W_gauge": Wga, "R_fillet": R, "holes": []}
            if not validate_geometry(params):
                continue
            r = cached_predict(
                round(L_total / 0.5) * 0.5, round(W_grip / 0.5) * 0.5,
                round(Wga / 0.5) * 0.5, round(R / 0.5) * 0.5,
            )
            if r["status"] == "ok":
                confidence_grid[i, j] = conf_map[r["confidence"]]
                sigma_grid[i, j] = r["peak_sigma_11"]

    return {
        "R_vals": R_vals.tolist(),
        "Wga_vals": Wga_vals.tolist(),
        "confidence": confidence_grid.tolist(),
        "sigma": sigma_grid.tolist(),
    }


def round_for_cache(val, step):
    """Round slider values to cache-friendly resolution."""
    return round(val / step) * step


def update_session_stats(result, params):
    """Update session statistics from a prediction result."""
    s = st.session_state.session_stats
    s["n_predictions"] += 1
    s[f"n_{result['confidence']}"] += 1

    if result["confidence"] in ("high", "medium"):
        if result["peak_sigma_11"] < s["best_sigma"]:
            s["best_sigma"] = result["peak_sigma_11"]
            s["best_params"] = dict(params)

    dH = (params["W_grip"] - params["W_gauge"]) / 2
    kt = 1 + 2 * np.sqrt(dH / params["R_fillet"]) if params["R_fillet"] > 0 else 1
    if kt > s["hottest_kt"]:
        s["hottest_kt"] = kt
        s["hottest_params"] = dict(params)



st.title("🎨 PI-GINOT Design Studio")
st.caption(
    "Interactive physics-informed design exploration. "
    "Drag sliders for live predictions, race the AI, explore reliability maps."
)

if st.session_state.get("pi_agent_error"):
    st.error(f"⚠️ Model not loaded: {st.session_state.pi_agent_error}")
    st.stop()

# Session stats bar
s = st.session_state.session_stats
session_time = int(time.time() - s["start_time"])
st.markdown(
    f'<div>'
    f'<span class="session-stat">📊 Session: <b>{s["n_predictions"]}</b> predictions</span>'
    f'<span class="session-stat">🟢 High: <b>{s["n_high"]}</b></span>'
    f'<span class="session-stat">🟡 Med: <b>{s["n_medium"]}</b></span>'
    f'<span class="session-stat">🟠 Low: <b>{s["n_low"]}</b></span>'
    f'<span class="session-stat">🔴 Rej: <b>{s["n_reject"]}</b></span>'
    + (f'<span class="session-stat">🏆 Best σ₁₁: <b>{s["best_sigma"]:.1f} MPa</b></span>'
       if s["best_sigma"] < float("inf") else "")
    + (f'<span class="session-stat">🔥 Hottest K_t: <b>{s["hottest_kt"]:.2f}</b></span>'
       if s["hottest_kt"] > 0 else "")
    + f'<span class="session-stat">⏱️ {session_time//60}m {session_time%60}s</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.divider()


mode = st.radio(
    "Mode",
    ["🎨 Explore", "🏁 Race the Optimizer", "🗺️ Reliability Map",
     "⚖️ Compare Two Designs"],
    horizontal=True,
    label_visibility="collapsed",
)



if mode == "🎨 Explore":
    left, center, right = st.columns([1, 2, 1])

    # ─── LEFT: Controls ───
    with left:
        st.subheader("🎛️ Geometry")

        L_lo, L_hi = GEOMETRY_RANGES["L_total"]
        Wg_lo, Wg_hi = GEOMETRY_RANGES["W_grip"]
        Wga_lo, Wga_hi = GEOMETRY_RANGES["W_gauge"]
        R_lo, R_hi = GEOMETRY_RANGES["R_fillet"]

        L = st.slider("**L_total** [mm]", float(L_lo), float(L_hi),
                      st.session_state.studio_sliders["L_total"],
                      step=0.5, key="slider_L")
        W_grip = st.slider("**W_grip** [mm]", float(Wg_lo), float(Wg_hi),
                           st.session_state.studio_sliders["W_grip"],
                           step=0.5, key="slider_Wgrip")
        W_gauge = st.slider("**W_gauge** [mm]", float(Wga_lo), float(Wga_hi),
                            st.session_state.studio_sliders["W_gauge"],
                            step=0.5, key="slider_Wgauge")
        R = st.slider("**R_fillet** [mm]", float(R_lo), float(R_hi),
                      st.session_state.studio_sliders["R_fillet"],
                      step=0.5, key="slider_R")

        st.session_state.studio_sliders = {
            "L_total": L, "W_grip": W_grip, "W_gauge": W_gauge, "R_fillet": R,
        }

        params = {"L_total": L, "W_grip": W_grip, "W_gauge": W_gauge,
                  "R_fillet": R, "holes": []}
        is_valid = validate_geometry(params)

        if not is_valid:
            st.error(
                "⚠️ Invalid geometry:\n"
                "- W_gauge < W_grip\n"
                "- R_fillet > (W_grip − W_gauge) / 2\n"
                "- Gauge length > 2 mm"
            )

        st.divider()

        # Overlay controls
        st.caption("**Overlays**")
        st.session_state.show_overlay = st.checkbox(
            "🔍 Show what the agent sees",
            value=st.session_state.show_overlay,
            help="Visualize verification slices and training bank proximity",
        )
        st.session_state.show_hotspot = st.checkbox(
            "⚡ Highlight stress hotspot",
            value=st.session_state.show_hotspot,
        )

        st.divider()

        # Actions
        col_a, col_b = st.columns(2)
        if col_a.button("📸 Snapshot", use_container_width=True):
            st.session_state.baseline_result = cached_predict(
                round_for_cache(L, 0.5), round_for_cache(W_grip, 0.5),
                round_for_cache(W_gauge, 0.5), round_for_cache(R, 0.5),
            )
            st.session_state.baseline_params = dict(st.session_state.studio_sliders)
            st.toast("📸 Snapshot taken", icon="📸")

        if col_b.button("🔄 Reset", use_container_width=True):
            st.session_state.studio_sliders = {
                "L_total": GEOMETRY_DEFAULT["L_total"],
                "W_grip": GEOMETRY_DEFAULT["W_grip"],
                "W_gauge": GEOMETRY_DEFAULT["W_gauge"],
                "R_fillet": GEOMETRY_DEFAULT["R_fillet"],
            }
            st.rerun()

        # Share button
        if is_valid:
            share_url = (
                f"?L={L}&Wg={W_grip}&Wga={W_gauge}&R={R}"
            )
            st.caption(f"🔗 **Share this design:** Copy URL with `{share_url}`")

        # Derived quantities
        with st.expander("📐 Derived quantities"):
            if is_valid:
                L_half = L / 2
                H_grip = W_grip / 2
                H_gauge = W_gauge / 2
                dH = H_grip - H_gauge
                if R > dH:
                    dx = math.sqrt(dH * (2 * R - dH))
                    x_g = L_half - dx
                    st.metric("Gauge length", f"{x_g:.1f} mm")
                    st.metric("Width ratio", f"{W_grip/W_gauge:.2f}")
                    K_t = 1 + 2 * np.sqrt(dH / R)
                    st.metric("Est. K_t (Neuber)", f"{K_t:.2f}")

    # ─── CENTER: The canvas ───
    with center:
        if not is_valid:
            st.warning("👈 Adjust sliders to a valid geometry")
        else:
            result = cached_predict(
                round_for_cache(L, 0.5), round_for_cache(W_grip, 0.5),
                round_for_cache(W_gauge, 0.5), round_for_cache(R, 0.5),
            )

            if result["status"] == "ok":
                # Update session stats
                update_session_stats(result, st.session_state.studio_sliders)

                # Hero metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                delta_sigma = None
                delta_u = None
                if st.session_state.baseline_result and \
                        st.session_state.baseline_result.get("status") == "ok":
                    b = st.session_state.baseline_result
                    delta_sigma = result["peak_sigma_11"] - b["peak_sigma_11"]
                    delta_u = result["peak_u"] - b["peak_u"]

                mcol1.metric(
                    "Peak σ₁₁",
                    f"{result['peak_sigma_11']:.1f} MPa",
                    delta=f"{delta_sigma:+.1f} MPa" if delta_sigma is not None else None,
                    delta_color="inverse",
                )
                mcol2.metric(
                    "Max |u|",
                    f"{result['peak_u']:.3f} mm",
                    delta=f"{delta_u:+.3f}" if delta_u is not None else None,
                )
                mcol3.metric("Eq. residual", f"{result['eq_residual']:.2e}")
                mcol4.markdown(
                    f"**Confidence**<br>{confidence_pill(result['confidence'])}",
                    unsafe_allow_html=True,
                )

                # Main plot
                pts_x = result["x"]
                pts_y = result["y"]
                sigma = result["sigma_11"]

                n_show = 800
                stride = max(1, len(pts_x) // n_show)
                pts_x_s = pts_x[::stride]
                pts_y_s = pts_y[::stride]
                sigma_s = sigma[::stride]

                color_scale = "Viridis" if result["confidence"] != "reject" else "Reds"

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pts_x_s, y=pts_y_s,
                    mode="markers",
                    marker=dict(
                        size=7, color=sigma_s, colorscale=color_scale,
                        colorbar=dict(title="σ₁₁ [MPa]", thickness=15, len=0.75),
                        opacity=0.85, line=dict(width=0),
                    ),
                    hovertemplate=(
                        "x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>"
                        "σ₁₁=%{marker.color:.2f} MPa<extra></extra>"
                    ),
                    name="σ₁₁ field",
                ))

                _draw_outline(fig, L, W_grip, W_gauge, R)

                # Loading arrow
                L_half = L / 2
                fig.add_annotation(
                    x=L_half * 1.08, y=W_grip / 4,
                    ax=L_half * 0.98, ay=W_grip / 4,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowwidth=3,
                    arrowcolor="#1f2937",
                )
                fig.add_annotation(
                    x=L_half * 1.12, y=W_grip / 4,
                    text="<b>u_δ = 1 mm</b>",
                    showarrow=False, font=dict(size=11, color="#1f2937"),
                )

                # ⚡ HOTSPOT CALLOUT
                if st.session_state.show_hotspot and result["confidence"] != "reject":
                    hx, hy, hv = _find_hotspot(pts_x, pts_y, sigma)
                    if hx is not None:
                        dH = (W_grip - W_gauge) / 2
                        kt = 1 + 2 * np.sqrt(dH / R) if R > 0 else 1
                        fig.add_trace(go.Scatter(
                            x=[hx], y=[hy],
                            mode="markers",
                            marker=dict(
                                size=25, color="rgba(239, 68, 68, 0)",
                                line=dict(color="#dc2626", width=3),
                                symbol="circle",
                            ),
                            showlegend=False, hoverinfo="skip",
                        ))
                        fig.add_annotation(
                            x=hx, y=hy,
                            text=f"⚡ <b>Peak: {abs(hv):.0f} MPa</b><br>K_t ≈ {kt:.2f}",
                            showarrow=True, arrowhead=2,
                            ax=40, ay=-40,
                            bgcolor="rgba(254, 226, 226, 0.9)",
                            bordercolor="#dc2626",
                            borderwidth=1,
                            font=dict(size=10, color="#991b1b"),
                        )

                # 🔍 "WHAT THE AGENT SEES" OVERLAY
                if st.session_state.show_overlay:
                    # Verification slices (where section CV is computed)
                    verification_xi = [0.10, 0.35, 0.55, 0.75, 0.90]
                    for xi in verification_xi:
                        x_slice = xi * L_half
                        w_slice = (W_gauge / 2 if x_slice < L_half * 0.7
                                   else W_grip / 2)
                        fig.add_shape(
                            type="line",
                            x0=x_slice, x1=x_slice,
                            y0=0, y1=w_slice,
                            line=dict(color="rgba(124, 58, 237, 0.5)",
                                      width=2, dash="dash"),
                        )
                    fig.add_annotation(
                        x=L_half * 0.10, y=W_grip / 2 * 1.1,
                        text="🔍 <b>Verification slices</b><br>(where section CV is checked)",
                        showarrow=False,
                        font=dict(size=9, color="#7c3aed"),
                        bgcolor="rgba(237, 233, 254, 0.9)",
                    )

                if result["confidence"] in ("low", "reject"):
                    fig.add_shape(
                        type="rect", xref="paper", yref="paper",
                        x0=0, y0=0, x1=1, y1=1,
                        line=dict(color="rgba(239, 68, 68, 0.8)", width=4),
                        fillcolor="rgba(0,0,0,0)",
                    )

                fig.update_layout(
                    title=dict(
                        text=(
                            f"σ₁₁ field · L={L:.1f} · W_grip={W_grip:.1f} · "
                            f"W_gauge={W_gauge:.1f} · R={R:.1f} mm"
                        ),
                        font=dict(size=14),
                    ),
                    xaxis=dict(title="x [mm]", scaleanchor="y", scaleratio=1),
                    yaxis=dict(title="y [mm]"),
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=40),
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})

                if result["confidence"] == "reject":
                    with st.container(border=True):
                        st.error("🚫 **Prediction rejected**")
                        for reason in result["rejection_reasons"]:
                            st.markdown(f"- {reason}")

            else:
                st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")

    # ─── RIGHT: Commentary, radar, saved designs ───
    with right:
        if is_valid and result.get("status") == "ok":
            st.subheader("🎯 Agent Commentary")
            commentary = _build_commentary(result, L, W_grip, W_gauge, R)
            with st.container(border=True):
                st.markdown(commentary)

            # Parameter radar chart
            st.subheader("📡 Geometry Fingerprint")
            bank_means = {  # approximate training bank centers
                "L_total": (sum(GEOMETRY_RANGES["L_total"]) / 2),
                "W_grip": (sum(GEOMETRY_RANGES["W_grip"]) / 2),
                "W_gauge": (sum(GEOMETRY_RANGES["W_gauge"]) / 2),
                "R_fillet": (sum(GEOMETRY_RANGES["R_fillet"]) / 2),
            }

            # Normalize to [0, 1]
            def _norm(val, key):
                lo, hi = GEOMETRY_RANGES[key]
                return (val - lo) / (hi - lo)

            current_norm = [
                _norm(L, "L_total"),
                _norm(W_grip, "W_grip"),
                _norm(W_gauge, "W_gauge"),
                _norm(R, "R_fillet"),
            ]
            bank_norm = [_norm(bank_means[k], k) for k in
                         ["L_total", "W_grip", "W_gauge", "R_fillet"]]

            categories = ["L_total", "W_grip", "W_gauge", "R_fillet"]

            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=bank_norm + [bank_norm[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Training center",
                line_color="rgba(156, 163, 175, 0.8)",
                fillcolor="rgba(156, 163, 175, 0.2)",
            ))
            radar.add_trace(go.Scatterpolar(
                r=current_norm + [current_norm[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Current",
                line_color="#3b82f6",
                fillcolor="rgba(59, 130, 246, 0.3)",
            ))
            radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=260,
                margin=dict(l=40, r=40, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(radar, use_container_width=True,
                            config={"displayModeBar": False})

            # OOD indicator
            nn_dist = result.get("nn_distance", 0)
            if nn_dist > 0.3:
                st.warning(f"⚠️ Distance to nearest training geometry: {nn_dist:.3f} (far)")
            elif nn_dist > 0.15:
                st.info(f"🟡 Distance to nearest training geometry: {nn_dist:.3f} (moderate)")
            else:
                st.success(f"✅ Distance to nearest training geometry: {nn_dist:.3f} (close)")

        st.divider()
        st.subheader("💾 Saved Designs")

        if st.button("⭐ Save this design", use_container_width=True,
                     disabled=not is_valid or result.get("status") != "ok"):
            st.session_state.saved_designs.append({
                "id": _geometry_hash(L, W_grip, W_gauge, R),
                "timestamp": time.time(),
                "params": dict(st.session_state.studio_sliders),
                "peak_sigma": result.get("peak_sigma_11"),
                "confidence": result.get("confidence"),
            })
            st.toast("💾 Design saved", icon="⭐")

        if st.session_state.saved_designs:
            for i, design in enumerate(reversed(st.session_state.saved_designs[-5:])):
                with st.container(border=True):
                    cols = st.columns([3, 1])
                    cols[0].markdown(
                        f"**#{design['id']}** "
                        f"· σ={design['peak_sigma']:.1f} MPa "
                        f"· {confidence_pill(design['confidence'])}",
                        unsafe_allow_html=True,
                    )
                    if cols[1].button("▲ Load", key=f"load_{design['id']}_{i}"):
                        st.session_state.studio_sliders = dict(design["params"])
                        st.rerun()
        else:
            st.caption("No saved designs yet.")



elif mode == "🏁 Race the Optimizer":
    st.subheader("🏁 Race the Optimizer")
    st.markdown(
        "**Challenge:** Minimize peak σ₁₁ with the sliders in **30 seconds**. "
        "Then watch the AI do it. Can you beat a physics-informed neural operator?"
    )

    if not st.session_state.race_mode:
        # Start screen
        col_start, _ = st.columns([1, 1])
        with col_start:
            st.info(
                "**Rules:**\n"
                "1. You have 30 seconds to find the lowest peak σ₁₁\n"
                "2. Move sliders freely — every valid geometry counts\n"
                "3. After time's up, the AI runs its optimizer\n"
                "4. Best confidence ≥ MEDIUM required for both"
            )

            if st.button("🚀 Start the Race!", type="primary",
                         use_container_width=True):
                st.session_state.race_mode = True
                st.session_state.race_started_at = time.time()
                st.session_state.race_user_best = None
                st.session_state.race_ai_best = None
                st.session_state.race_history = []
                st.session_state.studio_sliders = {
                    "L_total": GEOMETRY_DEFAULT["L_total"],
                    "W_grip": GEOMETRY_DEFAULT["W_grip"],
                    "W_gauge": GEOMETRY_DEFAULT["W_gauge"],
                    "R_fillet": GEOMETRY_DEFAULT["R_fillet"],
                }
                st.rerun()

    else:
        # Race in progress
        elapsed = time.time() - st.session_state.race_started_at
        remaining = max(0, 30 - elapsed)

        race_over = remaining <= 0 and st.session_state.race_user_best is not None

        if not race_over and remaining > 0:
            # Active race
            col_timer, col_score = st.columns([1, 2])

            with col_timer:
                color = "#dc2626" if remaining < 10 else "#059669"
                st.markdown(
                    f'<div class="race-timer" style="color:{color};">'
                    f'{remaining:.1f}s</div>',
                    unsafe_allow_html=True,
                )
                st.progress(remaining / 30)

            with col_score:
                if st.session_state.race_user_best:
                    st.markdown(
                        f'<div class="race-score race-user">'
                        f'🧑 <b>Your best: {st.session_state.race_user_best["sigma"]:.1f} MPa</b>'
                        f' ({st.session_state.race_user_best["confidence"].upper()})'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="race-score race-user">'
                        '🧑 Your best: — (try a geometry!)'
                        '</div>',
                        unsafe_allow_html=True,
                    )

            col_sliders, col_vis = st.columns([1, 2])

            with col_sliders:
                st.caption("**🎛️ Adjust Quickly!**")
                L_lo, L_hi = GEOMETRY_RANGES["L_total"]
                Wg_lo, Wg_hi = GEOMETRY_RANGES["W_grip"]
                Wga_lo, Wga_hi = GEOMETRY_RANGES["W_gauge"]
                R_lo, R_hi = GEOMETRY_RANGES["R_fillet"]

                L = st.slider("L_total", float(L_lo), float(L_hi),
                              st.session_state.studio_sliders["L_total"],
                              step=0.5, key="race_L")
                W_grip = st.slider("W_grip", float(Wg_lo), float(Wg_hi),
                                   st.session_state.studio_sliders["W_grip"],
                                   step=0.5, key="race_Wgrip")
                W_gauge = st.slider("W_gauge", float(Wga_lo), float(Wga_hi),
                                    st.session_state.studio_sliders["W_gauge"],
                                    step=0.5, key="race_Wgauge")
                R = st.slider("R_fillet", float(R_lo), float(R_hi),
                              st.session_state.studio_sliders["R_fillet"],
                              step=0.5, key="race_R")

                st.session_state.studio_sliders = {
                    "L_total": L, "W_grip": W_grip,
                    "W_gauge": W_gauge, "R_fillet": R,
                }

                params = {"L_total": L, "W_grip": W_grip,
                          "W_gauge": W_gauge, "R_fillet": R, "holes": []}
                is_valid = validate_geometry(params)

                if is_valid:
                    r = cached_predict(
                        round_for_cache(L, 0.5), round_for_cache(W_grip, 0.5),
                        round_for_cache(W_gauge, 0.5), round_for_cache(R, 0.5),
                    )
                    if r["status"] == "ok":
                        current_sigma = r["peak_sigma_11"]
                        current_conf = r["confidence"]

                        st.metric("Current σ₁₁", f"{current_sigma:.1f} MPa",
                                  delta=(
                                      f"{current_sigma - st.session_state.race_user_best['sigma']:+.1f}"
                                      if st.session_state.race_user_best else None
                                  ),
                                  delta_color="inverse")
                        st.markdown(confidence_pill(current_conf),
                                    unsafe_allow_html=True)

                        st.session_state.race_history.append({
                            "time": elapsed,
                            "sigma": current_sigma,
                            "confidence": current_conf,
                        })

                        # Update user best (MEDIUM or HIGH only)
                        if current_conf in ("high", "medium"):
                            if (st.session_state.race_user_best is None
                                    or current_sigma < st.session_state.race_user_best["sigma"]):
                                st.session_state.race_user_best = {
                                    "sigma": current_sigma,
                                    "confidence": current_conf,
                                    "params": dict(st.session_state.studio_sliders),
                                }
                                st.balloons()
                else:
                    st.warning("Invalid — fix constraints!")

            with col_vis:
                if is_valid and r["status"] == "ok":
                    pts_x = r["x"][::3]
                    pts_y = r["y"][::3]
                    sigma = r["sigma_11"][::3]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pts_x, y=pts_y, mode="markers",
                        marker=dict(size=6, color=sigma, colorscale="Viridis",
                                    colorbar=dict(title="σ₁₁", thickness=12)),
                        hoverinfo="skip",
                    ))
                    _draw_outline(fig, L, W_grip, W_gauge, R)
                    fig.update_layout(
                        height=320, xaxis=dict(scaleanchor="y"),
                        margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

            # Force a rerun every second for timer update
            time.sleep(0.5)
            st.rerun()

        else:
            # Race complete → run AI
            if st.session_state.race_ai_best is None:
                st.subheader("⏱️ Time's up! Now the AI takes its turn...")

                progress_bar = st.progress(0.0)
                status = st.empty()
                vis_placeholder = st.empty()

                ai_candidates = []
                rng = np.random.default_rng(int(time.time()) % 10000)

                n_tries = 40
                for i in range(n_tries):
                    for _ in range(30):
                        cand = {
                            "L_total": float(rng.uniform(*GEOMETRY_RANGES["L_total"])),
                            "W_grip": float(rng.uniform(*GEOMETRY_RANGES["W_grip"])),
                            "W_gauge": float(rng.uniform(*GEOMETRY_RANGES["W_gauge"])),
                            "R_fillet": float(rng.uniform(*GEOMETRY_RANGES["R_fillet"])),
                            "holes": [],
                        }
                        if validate_geometry(cand):
                            break
                    if not validate_geometry(cand):
                        continue

                    r = cached_predict(
                        round_for_cache(cand["L_total"], 0.5),
                        round_for_cache(cand["W_grip"], 0.5),
                        round_for_cache(cand["W_gauge"], 0.5),
                        round_for_cache(cand["R_fillet"], 0.5),
                    )
                    if r["status"] != "ok" or r["confidence"] not in ("high", "medium"):
                        continue

                    ai_candidates.append({
                        "sigma": r["peak_sigma_11"],
                        "confidence": r["confidence"],
                        "params": {k: v for k, v in cand.items() if k != "holes"},
                    })

                    progress_bar.progress((i + 1) / n_tries)
                    best_so_far = min(ai_candidates, key=lambda c: c["sigma"])
                    status.write(
                        f"AI candidate {i+1}/{n_tries}: "
                        f"best so far = {best_so_far['sigma']:.1f} MPa"
                    )

                if ai_candidates:
                    st.session_state.race_ai_best = min(
                        ai_candidates, key=lambda c: c["sigma"]
                    )

                progress_bar.empty()
                status.empty()
                st.rerun()

            # Final results screen
            st.subheader("🏆 Results!")

            user_best = st.session_state.race_user_best
            ai_best = st.session_state.race_ai_best

            col_user, col_vs, col_ai = st.columns([2, 1, 2])

            if user_best and ai_best:
                user_wins = user_best["sigma"] < ai_best["sigma"]
                margin = abs(user_best["sigma"] - ai_best["sigma"])
                pct = 100 * margin / max(user_best["sigma"], ai_best["sigma"])

                with col_user:
                    winner_class = "race-winner" if user_wins else "race-user"
                    st.markdown(
                        f'<div class="race-score {winner_class}">'
                        f'<b>🧑 You</b><br>'
                        f'<span style="font-size:2em;">{user_best["sigma"]:.1f}</span> MPa<br>'
                        f'{confidence_pill(user_best["confidence"])}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    p = user_best["params"]
                    st.caption(
                        f"L={p['L_total']:.1f} · W_grip={p['W_grip']:.1f} · "
                        f"W_gauge={p['W_gauge']:.1f} · R={p['R_fillet']:.1f}"
                    )

                with col_vs:
                    st.markdown("<div style='text-align:center; "
                                "font-size:3em; margin-top:40px;'>⚔️</div>",
                                unsafe_allow_html=True)

                with col_ai:
                    winner_class = "race-winner" if not user_wins else "race-ai"
                    st.markdown(
                        f'<div class="race-score {winner_class}">'
                        f'<b>🤖 PI-GINOT</b><br>'
                        f'<span style="font-size:2em;">{ai_best["sigma"]:.1f}</span> MPa<br>'
                        f'{confidence_pill(ai_best["confidence"])}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    p = ai_best["params"]
                    st.caption(
                        f"L={p['L_total']:.1f} · W_grip={p['W_grip']:.1f} · "
                        f"W_gauge={p['W_gauge']:.1f} · R={p['R_fillet']:.1f}"
                    )

                st.divider()
                if user_wins:
                    st.success(
                        f"🎉 **You won by {margin:.1f} MPa ({pct:.1f}%)!** "
                        f"Impressive — you outperformed PI-GINOT's random search."
                    )
                else:
                    st.info(
                        f"🤖 **AI wins by {margin:.1f} MPa ({pct:.1f}%).** "
                        f"PI-GINOT evaluated ~40 candidates in seconds."
                    )

                # Race history chart
                if st.session_state.race_history:
                    hist_df = pd.DataFrame(st.session_state.race_history)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist_df["time"], y=hist_df["sigma"],
                        mode="markers+lines",
                        marker=dict(
                            size=6,
                            color=[
                                {"high": "#059669", "medium": "#f59e0b",
                                 "low": "#ea580c", "reject": "#dc2626"}[c]
                                for c in hist_df["confidence"]
                            ],
                        ),
                        name="Your exploration",
                    ))
                    fig.add_hline(
                        y=ai_best["sigma"], line_dash="dash",
                        line_color="#f59e0b",
                        annotation_text=f"🤖 AI best: {ai_best['sigma']:.1f} MPa",
                    )
                    fig.update_layout(
                        title="Your exploration path",
                        xaxis_title="Time [s]",
                        yaxis_title="Peak σ₁₁ [MPa]",
                        height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            elif user_best is None:
                st.error("You didn't find a valid geometry with confidence ≥ MEDIUM!")
                if ai_best:
                    st.info(f"AI found: {ai_best['sigma']:.1f} MPa")

            if st.button("🔄 Race Again", type="primary"):
                st.session_state.race_mode = False
                st.session_state.race_user_best = None
                st.session_state.race_ai_best = None
                st.rerun()



elif mode == "🗺️ Reliability Map":
    st.subheader("🗺️ Reliability Heatmap")
    st.markdown(
        "Explore how confidence and peak stress vary across the parameter space. "
        "Fix two parameters, see how the model behaves across the other two."
    )

    map_left, map_right = st.columns([1, 3])

    with map_left:
        st.caption("**Fixed parameters:**")
        L_fixed = st.slider("L_total", *GEOMETRY_RANGES["L_total"],
                            GEOMETRY_DEFAULT["L_total"], step=1.0)
        W_grip_fixed = st.slider("W_grip", *GEOMETRY_RANGES["W_grip"],
                                 GEOMETRY_DEFAULT["W_grip"], step=1.0)

        st.caption("**Grid resolution:**")
        n_grid = st.select_slider("Points per axis",
                                  options=[8, 10, 12, 15, 20],
                                  value=12)

        metric = st.radio("Show", ["Confidence", "Peak σ₁₁"],
                          horizontal=False)

        if st.button("🔄 Recompute map", type="primary",
                     use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption(
            f"ℹ️ Computing {n_grid}×{n_grid} = {n_grid**2} predictions. "
            f"Results are cached."
        )

    with map_right:
        with st.spinner(f"Computing {n_grid}×{n_grid} heatmap..."):
            heatmap_data = compute_reliability_heatmap(
                L_fixed, W_grip_fixed,
                GEOMETRY_RANGES["R_fillet"],
                GEOMETRY_RANGES["W_gauge"],
                n_grid=n_grid,
            )

        R_vals = heatmap_data["R_vals"]
        Wga_vals = heatmap_data["Wga_vals"]

        if metric == "Confidence":
            z_data = np.array(heatmap_data["confidence"])
            z_text = np.where(
                z_data == 3, "H",
                np.where(z_data == 2, "M",
                         np.where(z_data == 1, "L",
                                  np.where(z_data == 0, "R", "—"))),
            )

            fig = go.Figure(data=go.Heatmap(
                x=Wga_vals, y=R_vals, z=z_data,
                colorscale=[
                    [0.0, "#dc2626"],  # reject
                    [0.25, "#ea580c"],  # low
                    [0.5, "#f59e0b"],  # medium
                    [0.75, "#10b981"],  # high
                    [1.0, "#059669"],
                ],
                zmin=-0.5, zmax=3.5,
                text=z_text, texttemplate="%{text}",
                textfont=dict(size=10),
                colorbar=dict(
                    title="Confidence",
                    tickvals=[0, 1, 2, 3],
                    ticktext=["Reject", "Low", "Medium", "High"],
                ),
                hovertemplate=(
                    "W_gauge=%{x:.1f}<br>R_fillet=%{y:.1f}<br>"
                    "Level: %{text}<extra></extra>"
                ),
            ))
        else:  # Peak σ₁₁
            z_data = np.array(heatmap_data["sigma"])
            fig = go.Figure(data=go.Heatmap(
                x=Wga_vals, y=R_vals, z=z_data,
                colorscale="Viridis",
                colorbar=dict(title="Peak σ₁₁ [MPa]"),
                hovertemplate=(
                    "W_gauge=%{x:.1f}<br>R_fillet=%{y:.1f}<br>"
                    "σ₁₁=%{z:.1f} MPa<extra></extra>"
                ),
            ))

        # Overlay the user's current geometry as a cursor
        current = st.session_state.studio_sliders
        fig.add_trace(go.Scatter(
            x=[current["W_gauge"]], y=[current["R_fillet"]],
            mode="markers+text",
            marker=dict(size=20, color="white",
                        line=dict(color="black", width=3), symbol="circle"),
            text=["You are here"],
            textposition="top center",
            textfont=dict(size=10, color="black"),
            name="Current",
            hoverinfo="skip",
        ))

        fig.update_layout(
            title=(f"Reliability map @ L={L_fixed:.1f}, "
                   f"W_grip={W_grip_fixed:.1f}"),
            xaxis_title="W_gauge [mm]",
            yaxis_title="R_fillet [mm]",
            height=550,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "💡 **How to read this:** Green zones = trustworthy predictions. "
            "Red zones = geometries where the model breaks reliability gates. "
            "The white circle shows your current slider position — "
            "stay in green zones for publication-quality results."
        )



elif mode == "⚖️ Compare Two Designs":
    st.subheader("⚖️ Side-by-Side Comparison")

    if st.session_state.compare_snapshot is None:
        st.info(
            "**Step 1:** Set up **Design A** below and click "
            "**'Lock as Design A'**. Then adjust sliders to create Design B."
        )
    else:
        p = st.session_state.compare_snapshot["params"]
        st.success(
            f"**✅ Design A locked:** L={p['L_total']:.1f}, W_grip={p['W_grip']:.1f}, "
            f"W_gauge={p['W_gauge']:.1f}, R={p['R_fillet']:.1f}. "
            f"Now adjust sliders for Design B."
        )

    cmp_left, cmp_right = st.columns([1, 3])

    with cmp_left:
        L_lo, L_hi = GEOMETRY_RANGES["L_total"]
        Wg_lo, Wg_hi = GEOMETRY_RANGES["W_grip"]
        Wga_lo, Wga_hi = GEOMETRY_RANGES["W_gauge"]
        R_lo, R_hi = GEOMETRY_RANGES["R_fillet"]

        L = st.slider("L_total", float(L_lo), float(L_hi),
                      st.session_state.studio_sliders["L_total"],
                      step=0.5, key="cmp_L")
        W_grip = st.slider("W_grip", float(Wg_lo), float(Wg_hi),
                           st.session_state.studio_sliders["W_grip"],
                           step=0.5, key="cmp_Wgrip")
        W_gauge = st.slider("W_gauge", float(Wga_lo), float(Wga_hi),
                            st.session_state.studio_sliders["W_gauge"],
                            step=0.5, key="cmp_Wgauge")
        R = st.slider("R_fillet", float(R_lo), float(R_hi),
                      st.session_state.studio_sliders["R_fillet"],
                      step=0.5, key="cmp_R")

        st.session_state.studio_sliders = {
            "L_total": L, "W_grip": W_grip, "W_gauge": W_gauge, "R_fillet": R,
        }

        params = {"L_total": L, "W_grip": W_grip, "W_gauge": W_gauge,
                  "R_fillet": R, "holes": []}
        is_valid = validate_geometry(params)

        if is_valid:
            result_b = cached_predict(
                round_for_cache(L, 0.5), round_for_cache(W_grip, 0.5),
                round_for_cache(W_gauge, 0.5), round_for_cache(R, 0.5),
            )
        else:
            result_b = None

        st.divider()

        if st.button("🔒 Lock as Design A", use_container_width=True,
                     disabled=not is_valid or (result_b and result_b["status"] != "ok")):
            st.session_state.compare_snapshot = {
                "params": dict(st.session_state.studio_sliders),
                "result": result_b,
            }
            st.toast("Design A locked!", icon="🔒")
            st.rerun()

        if st.button("🗑️ Clear Design A", use_container_width=True,
                     disabled=st.session_state.compare_snapshot is None):
            st.session_state.compare_snapshot = None
            st.rerun()

    with cmp_right:
        if st.session_state.compare_snapshot is None:
            if is_valid and result_b and result_b["status"] == "ok":
                # Show single (A to-be) preview
                pts_x = result_b["x"][::3]
                pts_y = result_b["y"][::3]
                sigma = result_b["sigma_11"][::3]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pts_x, y=pts_y, mode="markers",
                    marker=dict(size=6, color=sigma, colorscale="Viridis",
                                colorbar=dict(title="σ₁₁")),
                ))
                _draw_outline(fig, L, W_grip, W_gauge, R)
                fig.update_layout(
                    title=f"Design A (preview): σ₁₁ max = {result_b['peak_sigma_11']:.1f} MPa",
                    xaxis=dict(scaleanchor="y"), height=400,
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Side-by-side
            result_a = st.session_state.compare_snapshot["result"]
            params_a = st.session_state.compare_snapshot["params"]

            if not is_valid or not result_b or result_b["status"] != "ok":
                st.warning("Adjust sliders to a valid Design B to see comparison.")
            else:
                # Unified color scale
                sigma_a = np.array(result_a["sigma_11"])
                sigma_b = np.array(result_b["sigma_11"])
                vmin = min(sigma_a.min(), sigma_b.min())
                vmax = max(sigma_a.max(), sigma_b.max())

                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=("Design A", "Design B", "Δ (B − A)"),
                    horizontal_spacing=0.08,
                )

                stride = max(1, len(result_a["x"]) // 400)

                # A
                fig.add_trace(go.Scatter(
                    x=result_a["x"][::stride], y=result_a["y"][::stride],
                    mode="markers",
                    marker=dict(size=5, color=sigma_a[::stride],
                                colorscale="Viridis", cmin=vmin, cmax=vmax,
                                showscale=False),
                    hoverinfo="skip",
                ), row=1, col=1)

                # B
                fig.add_trace(go.Scatter(
                    x=result_b["x"][::stride], y=result_b["y"][::stride],
                    mode="markers",
                    marker=dict(size=5, color=sigma_b[::stride],
                                colorscale="Viridis", cmin=vmin, cmax=vmax,
                                colorbar=dict(title="σ₁₁", x=0.63)),
                    hoverinfo="skip",
                ), row=1, col=2)

                # Delta (interpolate B onto A's points — simple nearest-neighbor)
                # For visualization, just use B's field minus interpolated A
                # Quick approximation: assume same points
                pts_a = np.stack([result_a["x"], result_a["y"]], axis=-1)
                pts_b = np.stack([result_b["x"], result_b["y"]], axis=-1)

                # Simple grid-based interpolation
                from scipy.spatial import cKDTree
                tree_a = cKDTree(pts_a)
                _, idx = tree_a.query(pts_b, k=1)
                sigma_a_interp = sigma_a[idx]
                delta = sigma_b - sigma_a_interp

                fig.add_trace(go.Scatter(
                    x=result_b["x"][::stride], y=result_b["y"][::stride],
                    mode="markers",
                    marker=dict(size=5, color=delta[::stride],
                                colorscale="RdBu_r",
                                cmin=-np.abs(delta).max(),
                                cmax=np.abs(delta).max(),
                                colorbar=dict(title="Δσ₁₁", x=1.02)),
                    hoverinfo="skip",
                ), row=1, col=3)

                _draw_outline(fig, **{k: params_a[k] for k in
                                      ["L_total", "W_grip", "W_gauge", "R_fillet"]})
                # Skip adding outlines to subplots 2,3 — Plotly's add_trace
                # with subplot index is complex for Scatter lines. Use shapes:
                for col_idx in [2, 3]:
                    L_b = L if col_idx >= 2 else params_a["L_total"]
                    # (simple fallback: omit outline on B and delta subplots)
                    pass

                fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
                fig.update_xaxes(scaleanchor="y2", scaleratio=1, row=1, col=2)
                fig.update_xaxes(scaleanchor="y3", scaleratio=1, row=1, col=3)

                fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10),
                                  plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

                # Comparison metrics
                st.divider()
                mcol1, mcol2, mcol3 = st.columns(3)

                delta_peak = result_b["peak_sigma_11"] - result_a["peak_sigma_11"]
                pct_change = 100 * delta_peak / result_a["peak_sigma_11"]

                mcol1.metric(
                    "Design A peak σ₁₁",
                    f"{result_a['peak_sigma_11']:.1f} MPa",
                )
                mcol2.metric(
                    "Design B peak σ₁₁",
                    f"{result_b['peak_sigma_11']:.1f} MPa",
                    delta=f"{delta_peak:+.1f} MPa ({pct_change:+.1f}%)",
                    delta_color="inverse",
                )

                if abs(pct_change) < 2:
                    verdict = "🟰 **Nearly identical** — geometry change has minimal effect."
                elif delta_peak < 0:
                    verdict = f"✅ **Design B is {abs(pct_change):.1f}% better** at reducing stress concentration."
                else:
                    verdict = f"❌ **Design B is {pct_change:.1f}% worse** — reconsider the change."

                mcol3.markdown(f"### Verdict\n{verdict}")

                # Comparison table
                st.divider()
                st.subheader("📋 Detailed Comparison")
                comparison_df = pd.DataFrame([
                    {
                        "Parameter": "L_total",
                        "Design A": f"{params_a['L_total']:.1f}",
                        "Design B": f"{st.session_state.studio_sliders['L_total']:.1f}",
                    },
                    {
                        "Parameter": "W_grip",
                        "Design A": f"{params_a['W_grip']:.1f}",
                        "Design B": f"{st.session_state.studio_sliders['W_grip']:.1f}",
                    },
                    {
                        "Parameter": "W_gauge",
                        "Design A": f"{params_a['W_gauge']:.1f}",
                        "Design B": f"{st.session_state.studio_sliders['W_gauge']:.1f}",
                    },
                    {
                        "Parameter": "R_fillet",
                        "Design A": f"{params_a['R_fillet']:.1f}",
                        "Design B": f"{st.session_state.studio_sliders['R_fillet']:.1f}",
                    },
                    {
                        "Parameter": "Peak σ₁₁ [MPa]",
                        "Design A": f"{result_a['peak_sigma_11']:.2f}",
                        "Design B": f"{result_b['peak_sigma_11']:.2f}",
                    },
                    {
                        "Parameter": "Confidence",
                        "Design A": result_a["confidence"].upper(),
                        "Design B": result_b["confidence"].upper(),
                    },
                    {
                        "Parameter": "Eq. residual",
                        "Design A": f"{result_a['eq_residual']:.2e}",
                        "Design B": f"{result_b['eq_residual']:.2e}",
                    },
                    {
                        "Parameter": "Section CV",
                        "Design A": f"{100*result_a['section_cv']:.1f}%",
                        "Design B": f"{100*result_b['section_cv']:.1f}%",
                    },
                ])
                st.dataframe(comparison_df, use_container_width=True,
                             hide_index=True)
