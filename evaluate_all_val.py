#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the last saved PI-GINOT checkpoint on ALL validation geometries
and produce per-geometry field plots.

Usage:
    python evaluate_all_val.py
    python evaluate_all_val.py --checkpoint checkpoints/best.pt
    python evaluate_all_val.py --out_dir checkpoints/eval_fields
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    ENCODER_CONFIG, DECODER_CONFIG, MATERIAL_CONFIG,
    TRAINING_CONFIG, LOADING_CONFIG, COLLOCATION_CONFIG,
)
from models.pi_ginot import PI_GINOT
from physics.losses import PhysicsLoss
from physics.neo_hookean import full_stress_state
from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params
from geometry.collocation import sample_collocation_points


def build_val_bank(config):
    """Reproduce the trainer's validation geometry bank (seed=200)."""
    bank_ranges = config.get("bank_geo_ranges", None)
    n_val = config.get("bank_val_size", 8)
    bank_rng = np.random.default_rng(200)  # same seed as trainer

    bank = []
    for _ in range(n_val):
        params = sample_geometry_params(
            bank_rng, geometry_ranges=bank_ranges, holes_enabled=False,
        )
        n_bnd_seg = COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
        mesh = generate_dogbone(
            params, n_pts_per_segment=n_bnd_seg, rng=bank_rng,
        )
        coll = sample_collocation_points(mesh, rng=bank_rng)
        bank.append((mesh, coll))
    return bank


@torch.no_grad()
def evaluate_geometry(model, mesh, coll, device):
    """Run the model on one geometry and return numpy field arrays."""
    mu = MATERIAL_CONFIG["mu"]
    lam = MATERIAL_CONFIG["lam"]
    state = MATERIAL_CONFIG.get("state", "plane strain")

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

    pts_np = coll.interior_pts
    # Need grads for stress computation
    with torch.enable_grad():
        query = _t(pts_np).requires_grad_(True)
        bpc = _t(coll.boundary_pc)
        u_delta = torch.tensor(
            [LOADING_CONFIG["u_max"]], dtype=torch.float32, device=device
        )
        x_max = torch.tensor([coll.x_max], dtype=torch.float32, device=device)
        y_max = torch.tensor([coll.y_max], dtype=torch.float32, device=device)

        z = model.encode(bpc, x_max, y_max)
        uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
            query, z, u_delta, x_max, y_max,
        )
        S11, S22, S33, S12, detF = full_stress_state(
            du_dx, du_dy, dv_dx, dv_dy, mu, lam, state,
        )
        E11 = 0.5 * ((1.0 + du_dx) ** 2 + dv_dx ** 2 - 1.0)

    u_np = uv[0, :, 0].detach().cpu().numpy()
    u_base = LOADING_CONFIG["u_max"] * pts_np[:, 0] / coll.x_max

    return {
        "x": pts_np[:, 0],
        "y": pts_np[:, 1],
        "u": u_np,
        "delta_u": u_np - u_base,
        "v": uv[0, :, 1].detach().cpu().numpy(),
        "S11": S11[0, :, 0].detach().cpu().numpy(),
        "E11": E11[0, :, 0].detach().cpu().numpy(),
        "params": mesh.params,
    }


def plot_single_geometry(res, geo_idx, out_dir):
    """Save a 1x5 field plot (u, delta_u, v, sigma_11, E_11) for one geometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    col_defs = [
        ("u",       "u [mm]",     "jet"),
        ("delta_u", "Δu [mm]",    "jet"),
        ("v",       "v [mm]",     "jet"),
        ("S11",     "σ₁₁ [MPa]", "jet"),
        ("E11",     "E₁₁ [-]",   "jet"),
    ]
    fmt = FuncFormatter(lambda val, pos: f"{val:.4f}")

    fig, axes = plt.subplots(1, 5, figsize=(5 * 4.5, 3.5), squeeze=False)
    for col, (key, label, cmap) in enumerate(col_defs):
        ax = axes[0, col]
        vals = res[key]
        sc = ax.scatter(
            res["x"], res["y"], c=vals, cmap=cmap, s=5, edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, format=fmt)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title(label, fontsize=10)

    p = res["params"]
    fig.suptitle(
        f"Val geometry {geo_idx:02d}  —  "
        f"L={p['L_total']:.1f}  W_grip={p['W_grip']:.1f}  "
        f"W_gauge={p['W_gauge']:.1f}  R={p['R_fillet']:.1f}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fname = f"val_geo_{geo_idx:02d}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary_grid(all_results, out_dir):
    """Save one large grid figure: N_val rows x 5 columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    col_defs = [
        ("u",       "u [mm]",     "jet"),
        ("delta_u", "Δu [mm]",    "jet"),
        ("v",       "v [mm]",     "jet"),
        ("S11",     "σ₁₁ [MPa]", "jet"),
        ("E11",     "E₁₁ [-]",   "jet"),
    ]
    fmt = FuncFormatter(lambda val, pos: f"{val:.4f}")

    n_rows = len(all_results)
    n_cols = 5
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 2.8), squeeze=False,
    )

    for row, res in enumerate(all_results):
        for col, (key, label, cmap) in enumerate(col_defs):
            ax = axes[row, col]
            vals = res[key]
            sc = ax.scatter(
                res["x"], res["y"], c=vals, cmap=cmap, s=2, edgecolors="none",
            )
            fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02, format=fmt)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=5)

            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                p = res["params"]
                ax.set_ylabel(
                    f"L={p['L_total']:.0f} W={p['W_gauge']:.0f} R={p['R_fillet']:.0f}",
                    fontsize=7,
                )

    fig.suptitle(
        f"All {n_rows} validation geometries  —  last.pt",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_dir, "val_all_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PI-GINOT on all validation geometries"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/last.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--out_dir", type=str, default="checkpoints/eval_fields",
        help="Output directory for field plots",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cpu or cuda",
    )
    args, _ = parser.parse_known_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Build model and load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    n_params = model.count_parameters()["trainable"]
    print(f"  Loaded epoch {epoch}  |  {n_params:,} parameters  |  {device}")

    # Build validation bank (same as trainer)
    config = TRAINING_CONFIG.copy()
    val_bank = build_val_bank(config)
    print(f"  Validation bank: {len(val_bank)} geometries")

    # Evaluate all geometries
    all_results = []
    for i, (mesh, coll) in enumerate(val_bank):
        p = mesh.params
        print(f"  [{i+1:2d}/{len(val_bank)}] "
              f"L={p['L_total']:.1f}  W_grip={p['W_grip']:.1f}  "
              f"W_gauge={p['W_gauge']:.1f}  R={p['R_fillet']:.1f}  ... ",
              end="", flush=True)
        res = evaluate_geometry(model, mesh, coll, device)
        all_results.append(res)

        # Per-geometry plot
        path = plot_single_geometry(res, i, args.out_dir)
        print(f"saved → {path}")

    # Summary grid plot
    summary_path = plot_summary_grid(all_results, args.out_dir)
    print(f"\nSummary grid saved → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
