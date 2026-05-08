#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
showcase_gif_v2.py — Scientific visualization of PI-GINOT predictions on
parametric DogBone specimens.

Each frame shows three specimens with distinct geometric parameters,
evaluated by the trained PI-GINOT model. Fields displayed: axial
displacement u, axial 1st Piola-Kirchhoff stress sigma_11, and axial
Green-Lagrange strain E_11.
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
import imageio.v2 as imageio

from config import (
    ENCODER_CONFIG, DECODER_CONFIG, MATERIAL_CONFIG,
    LOADING_CONFIG, TRAINING_CONFIG, COLLOCATION_CONFIG,
)
from models.pi_ginot import PI_GINOT
from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params
from geometry.collocation import sample_collocation_points
from physics.neo_hookean import full_stress_state


# Colour palette for white-background scientific figures
BG_COLOR        = "#ffffff"
PANEL_COLOR     = "#ffffff"
TEXT_PRIMARY    = "#1a1a1a"
TEXT_SECONDARY  = "#555555"
TEXT_DIM        = "#999999"
BORDER_COLOR    = "#cccccc"
RULE_COLOR      = "#e5e5e5"

def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavuserif",
        "axes.facecolor": PANEL_COLOR,
        "figure.facecolor": BG_COLOR,
        "savefig.facecolor": BG_COLOR,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.7,
        "axes.labelcolor": TEXT_PRIMARY,
        "xtick.color": TEXT_PRIMARY,
        "ytick.color": TEXT_PRIMARY,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })


def sample_diverse_triplet(rng, bank_ranges):
    """Sample 3 geometries from distinct L_total tercile bins for visual variety."""
    L_lo, L_hi = bank_ranges["L_total"]
    L_bins = [
        (L_lo,                       L_lo + (L_hi - L_lo) / 3),
        (L_lo + (L_hi - L_lo) / 3,   L_lo + 2 * (L_hi - L_lo) / 3),
        (L_lo + 2 * (L_hi - L_lo) / 3, L_hi),
    ]
    rng.shuffle(L_bins)

    triplet = []
    for (lo, hi) in L_bins:
        ranges_mod = dict(bank_ranges)
        ranges_mod["L_total"] = (lo, hi)
        p = sample_geometry_params(rng, geometry_ranges=ranges_mod,
                                   holes_enabled=False)
        triplet.append(p)
    return triplet


def load_model(ckpt_path, device):
    """Load PI-GINOT checkpoint and return the model in eval mode."""
    print(f"Loading checkpoint: {ckpt_path}")
    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded from epoch {epoch}  |  "
          f"{model.count_parameters()['trainable']:,} parameters")
    return model, epoch


def evaluate_geometry(model, params, device, rng):
    """Generate mesh, run encode/decode, compute stress/strain, return fields."""
    mu  = MATERIAL_CONFIG["mu"]
    lam = MATERIAL_CONFIG["lam"]
    state = MATERIAL_CONFIG.get("state", "plane strain")
    u_max = LOADING_CONFIG["u_max"]
    n_bnd = COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)

    mesh = generate_dogbone(params, n_pts_per_segment=n_bnd,
                            n_interior=3500, rng=rng)
    coll = sample_collocation_points(mesh, rng=rng)

    def _t(arr, req=False):
        t = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
        return t.requires_grad_(req) if req else t

    pts   = coll.interior_pts
    query = _t(pts, req=True)
    bpc   = _t(coll.boundary_pc)
    u_d   = torch.tensor([u_max], dtype=torch.float32, device=device)
    x_m   = torch.tensor([coll.x_max], dtype=torch.float32, device=device)
    y_m   = torch.tensor([coll.y_max], dtype=torch.float32, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.enable_grad():
        z = model.encode(bpc, x_m, y_m)
        uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
            query, z, u_d, x_m, y_m
        )
        S11, S22, S33, S12, detF = full_stress_state(
            du_dx, du_dy, dv_dx, dv_dy, mu, lam, state,
        )
        E11 = 0.5 * ((1.0 + du_dx)**2 + dv_dx**2 - 1.0)

    if device == "cuda":
        torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "mesh":     mesh,
        "x":        pts[:, 0],
        "y":        pts[:, 1],
        "u":        uv[0, :, 0].detach().cpu().numpy(),
        "v":        uv[0, :, 1].detach().cpu().numpy(),
        "E11":      E11[0, :, 0].detach().cpu().numpy(),
        "S11":      S11[0, :, 0].detach().cpu().numpy(),
        "params":   params,
        "infer_ms": infer_ms,
    }


def draw_field_panel(ax, data, field_key, label, cmap, max_extent):
    """Scatter-plot one field on a single axes with geometry outline."""
    vals = data[field_key]
    vmin, vmax = vals.min(), vals.max()

    sc = ax.scatter(
        data["x"], data["y"], c=vals, cmap=cmap,
        s=5, edgecolors="none", vmin=vmin, vmax=vmax, rasterized=True,
    )

    for seg in data["mesh"].boundary_segments:
        ax.plot(seg.points[:, 0], seg.points[:, 1],
                color=TEXT_PRIMARY, linewidth=0.9, alpha=0.9)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.88, pad=0.012,
                        fraction=0.05, aspect=14)
    cbar.ax.tick_params(labelsize=7, colors=TEXT_PRIMARY)
    cbar.outline.set_edgecolor("#333333")
    cbar.outline.set_linewidth(0.5)

    ax.set_title(label, fontsize=9, color=TEXT_PRIMARY, pad=4)
    ax.set_aspect("equal")
    ax.set_xlim(-max_extent * 0.02, max_extent * 1.02)
    fi = data["mesh"].fillet_info
    ax.set_ylim(-fi["H_grip"] * 0.06, fi["H_grip"] * 1.15)
    ax.tick_params(axis="both", length=2)
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.4,
            color=TEXT_DIM)
    ax.set_axisbelow(True)


def draw_header(fig, gs_slot, epoch, n_solved, total_specimens,
                avg_ms, n_params, device_name):
    """Top-of-frame header with model info and running metrics."""
    ax = fig.add_subplot(gs_slot)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.axhline(y=0.02, xmin=0.005, xmax=0.995,
               color=BORDER_COLOR, linewidth=0.6)

    ax.text(0.005, 0.78,
            "Physics-Informed Geometry-Informed Neural Operator "
            "Transformer (PI-GINOT)",
            fontsize=12, fontweight="bold", color=TEXT_PRIMARY,
            va="center")
    ax.text(0.005, 0.45,
            "Predicted fields on parametric quarter-model dogbone specimens.  "
            "Neo-Hookean hyperelasticity, plane stress.",
            fontsize=9, color=TEXT_SECONDARY, va="center", style="italic")

    meta = [
        ("checkpoint",  f"epoch {epoch}"),
        ("parameters",  f"{n_params:,}"),
        ("specimens",   f"{n_solved} / {total_specimens}"),
        ("mean wall-clock",  f"{avg_ms:.1f} ms"),
        ("device",      device_name),
    ]
    x_start = 0.005
    x_step  = 0.20
    for i, (k, v) in enumerate(meta):
        x = x_start + i * x_step
        ax.text(x, 0.18, f"{k}:",
                fontsize=8, color=TEXT_DIM, va="center")
        ax.text(x, 0.18, f"    {v}",
                fontsize=8.5, color=TEXT_PRIMARY, va="center",
                fontweight="bold")

    return ax


def render_frame(triplet_data, frame_idx, total_frames, epoch,
                 n_solved, avg_ms, n_params, device_name, output_path):
    """Render one GIF frame: header + 3x3 grid (specimens x fields)."""

    max_extent = max(d["mesh"].fillet_info["L_half"] for d in triplet_data)

    fig = plt.figure(figsize=(18, 10), dpi=110)
    gs = gridspec.GridSpec(
        4, 3,
        height_ratios=[0.30, 1.0, 1.0, 1.0],
        hspace=0.36, wspace=0.22,
        left=0.05, right=0.97, top=0.96, bottom=0.05,
    )

    draw_header(fig, gs[0, :], epoch, n_solved, total_frames * 3,
                avg_ms, n_params, device_name)

    field_specs = [
        ("u",   r"$u$  [mm]",              "turbo"),
        ("S11", r"$\sigma_{11}$  [MPa]",   "viridis"),
        ("E11", r"$E_{11}$  [–]",          "inferno"),
    ]

    labels = ["Specimen A", "Specimen B", "Specimen C"]

    for col, data in enumerate(triplet_data):
        p = data["params"]

        for row, (key, label, cmap) in enumerate(field_specs):
            ax = fig.add_subplot(gs[row + 1, col])
            draw_field_panel(ax, data, key, label, cmap, max_extent)

            if row == 0:
                header = (f"{labels[col]}    "
                          f"$L$={p['L_total']:.1f},  "
                          f"$W_{{grip}}$={p['W_grip']:.1f},  "
                          f"$W_{{gauge}}$={p['W_gauge']:.1f},  "
                          f"$R$={p['R_fillet']:.1f}  mm    "
                          f"({data['infer_ms']:.0f} ms)")
                ax.text(0.5, 1.22, header,
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=9, color=TEXT_PRIMARY, fontweight="bold")
                ax.set_ylabel("y [mm]", fontsize=8)
            else:
                ax.set_ylabel("y [mm]", fontsize=8)

            if row == 2:
                ax.set_xlabel("x [mm]", fontsize=8)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    fig.savefig(output_path, dpi=110, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)


def render_title_frame(path, epoch, n_params, device_name, total_specimens):
    """Opening card: model name, architecture summary, evaluation setup."""
    fig, ax = plt.subplots(figsize=(18, 10), dpi=110, facecolor=BG_COLOR)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.80, "PI-GINOT",
            ha="center", va="center", fontsize=46, fontweight="bold",
            color=TEXT_PRIMARY, family="serif")
    ax.text(0.5, 0.72,
            "Physics-Informed Geometry-Informed Neural Operator Transformer",
            ha="center", va="center", fontsize=16, color=TEXT_SECONDARY)

    ax.axhline(0.66, xmin=0.28, xmax=0.72,
               color=BORDER_COLOR, linewidth=1.0)

    ax.text(0.5, 0.58,
            "Predictions of displacement and stress fields\n"
            "on parametric dogbone specimens.",
            ha="center", va="center", fontsize=15, color=TEXT_PRIMARY,
            linespacing=1.6)

    box_txt = (
        "Architecture      Boundary point-cloud encoder + cross-attention decoder\n"
        "Training loss     PINN residuals (equilibrium, tractions, section resultant)\n"
        "Material          Neo-Hookean, plane stress\n"
        f"Checkpoint        epoch {epoch},  {n_params:,} parameters\n"
        f"Evaluation        {total_specimens} random specimens,  {device_name}\n"
        "Loaded from       checkpoints/last.pt  (no retraining)"
    )
    ax.text(0.5, 0.32, box_txt,
            ha="center", va="center", fontsize=11, color=TEXT_PRIMARY,
            family="monospace", linespacing=1.7,
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f5f5f5",
                      edgecolor=BORDER_COLOR, linewidth=0.8))

    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def render_end_frame(path, total_specimens, avg_ms, n_params,
                     epoch, device_name):
    """Closing card: aggregate statistics and method notes."""
    fig, ax = plt.subplots(figsize=(18, 10), dpi=110, facecolor=BG_COLOR)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.82, "Summary",
            ha="center", va="center", fontsize=32, fontweight="bold",
            color=TEXT_PRIMARY, family="serif")

    ax.axhline(0.75, xmin=0.40, xmax=0.60,
               color=BORDER_COLOR, linewidth=1.0)

    stats = [
        (f"{total_specimens}",              "specimens evaluated"),
        (f"{avg_ms:.1f} ms",                "mean wall-clock"),
        (f"{n_params / 1e6:.2f} M",         "trainable parameters"),
        (f"{epoch}",                        "training epochs"),
    ]
    for i, (val, lbl) in enumerate(stats):
        x = 0.10 + 0.225 * i + 0.10
        ax.text(x, 0.56, val,
                ha="center", va="center", fontsize=26, fontweight="bold",
                color=TEXT_PRIMARY)
        ax.text(x, 0.48, lbl,
                ha="center", va="center", fontsize=10, color=TEXT_SECONDARY)

    notes = (
        "Each specimen is parameterized by (L, W_grip, W_gauge, R_fillet)\n"
        "and sampled uniformly from the training parameter bank ranges.\n"
        "Model evaluated in inference mode; no weight updates applied.\n"
        f"Hardware: {device_name}."
    )
    ax.text(0.5, 0.22, notes,
            ha="center", va="center", fontsize=11, color=TEXT_SECONDARY,
            linespacing=1.7, family="monospace")

    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="PI-GINOT showcase GIF")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.pt")
    parser.add_argument("--n_triplets", type=int, default=12)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="pi_ginot_showcase_v2.gif")
    parser.add_argument("--frame_dir", type=str, default="showcase_frames_v2")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--title_hold", type=float, default=2.5)
    args, _ = parser.parse_known_args()

    setup_style()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_name = ("GPU ("
                   f"{torch.cuda.get_device_name(0)})" if device == "cuda"
                   else "CPU")
    os.makedirs(args.frame_dir, exist_ok=True)

    model, epoch = load_model(args.ckpt, device=device)
    n_params = model.count_parameters()["trainable"]

    rng = np.random.default_rng(args.seed)
    bank_ranges = TRAINING_CONFIG.get("bank_geo_ranges", None)
    triplets = [sample_diverse_triplet(rng, bank_ranges)
                for _ in range(args.n_triplets)]

    total_specimens = args.n_triplets * 3
    print(f"\nGenerated {args.n_triplets} triplets "
          f"({total_specimens} specimens total)")

    frame_paths = []

    title_path = os.path.join(args.frame_dir, "00_title.png")
    render_title_frame(title_path, epoch, n_params, device_name,
                       total_specimens)
    frame_paths.append(title_path)
    print("Title frame rendered.")

    cumulative_ms = 0.0
    cumulative_count = 0

    for i, triplet in enumerate(triplets):
        print(f"  Triplet [{i+1:2d}/{args.n_triplets}]")
        triplet_data = []
        for j, params in enumerate(triplet):
            data = evaluate_geometry(model, params, device, rng)
            triplet_data.append(data)
            cumulative_ms += data["infer_ms"]
            cumulative_count += 1
            print(f"     specimen {j+1}: L={params['L_total']:5.1f}  "
                  f"W_gauge={params['W_gauge']:4.1f}  "
                  f"R={params['R_fillet']:5.1f}   "
                  f"({data['infer_ms']:.1f} ms)")

        avg_ms = cumulative_ms / cumulative_count
        frame_path = os.path.join(args.frame_dir, f"frame_{i+1:03d}.png")
        render_frame(triplet_data, frame_idx=i, total_frames=args.n_triplets,
                     epoch=epoch, n_solved=cumulative_count,
                     avg_ms=avg_ms, n_params=n_params,
                     device_name=device_name, output_path=frame_path)
        frame_paths.append(frame_path)

    final_avg_ms = cumulative_ms / cumulative_count
    end_path = os.path.join(args.frame_dir, "99_end.png")
    render_end_frame(end_path, total_specimens, final_avg_ms, n_params,
                     epoch, device_name)
    frame_paths.append(end_path)

    # Assemble GIF with per-frame durations (title/end cards held longer)
    print(f"\nAssembling GIF: {args.output}")
    frame_duration_ms = 1000.0 / args.fps
    images = []
    durations = []
    target_size = None

    for i, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("RGB")
        if target_size is None:
            w, h = img.size
            if w > 1400:
                scale = 1400 / w
                target_size = (int(w * scale), int(h * scale))
            else:
                target_size = (w, h)
        img = img.resize(target_size, Image.LANCZOS)
        images.append(np.array(img))

        if i == 0 or i == len(frame_paths) - 1:
            durations.append(args.title_hold * 1000.0)
        else:
            durations.append(frame_duration_ms)

    imageio.mimsave(args.output, images, duration=durations, loop=0)

    # Optional MP4 export (requires imageio-ffmpeg)
    mp4_path = args.output.replace(".gif", ".mp4")
    try:
        with imageio.get_writer(
            mp4_path, fps=args.fps, codec="libx264",
            quality=8, pixelformat="yuv420p",
        ) as writer:
            for img, dur in zip(images, durations):
                n_repeat = max(1, int(round(dur * args.fps)))
                for _ in range(n_repeat):
                    writer.append_data(img)
        print(f"  MP4:  {mp4_path}")
    except Exception as e:
        print(f"  (MP4 skipped: {e})")

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nDone.  {args.output}  ({size_mb:.1f} MB)  "
          f"{total_specimens} specimens")


if __name__ == "__main__":
    main()
