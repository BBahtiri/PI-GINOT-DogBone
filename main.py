#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PI-GINOT — Physics-Informed Geometry-Informed Neural Operator Transformer.

Entry point for training and evaluation on parametric DogBone specimens.

Usage:
    python main.py --mode train --epochs 1000 --lr 1e-4
    python main.py --mode train --resume checkpoints/last.pt
"""

import argparse
import torch
import numpy as np

from config import (
    ENCODER_CONFIG, DECODER_CONFIG, MATERIAL_CONFIG,
    TRAINING_CONFIG, NONDIM_SCALES, RANDOM_SEED,
    GEOMETRY_DEFAULT,
)
from models.pi_ginot import PI_GINOT
from physics.losses import PhysicsLoss
from training.trainer import PI_GINOT_Trainer


# Initial state diagnostic
def inspect_initial_state(model, device="cpu"):
    """Run one forward pass before training and print diagnostic stats.

    Checks that the model starts in a physically sane regime:
    - Small displacement gradients
    - detF ≈ 1 (near identity)
    - Bounded stress
    - Bounded equilibrium residual

    Uses GEOMETRY_DEFAULT when single_geometry mode is enabled so
    the diagnostic is consistent with the geometry used during training.
    """
    from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params
    from geometry.collocation import sample_collocation_points
    from physics.neo_hookean import first_piola_kirchhoff_stress
    from physics.equilibrium import equilibrium_residual

    rng = np.random.default_rng(123)

    stress_state = MATERIAL_CONFIG.get("state", "plane strain")

    # Use geometry consistent with the training mode
    if TRAINING_CONFIG.get("single_geometry", False):
        mesh = generate_dogbone(GEOMETRY_DEFAULT, rng=rng)
        coll = sample_collocation_points(mesh, rng=rng)
    elif (TRAINING_CONFIG.get("use_train_geometry_bank", False)
          or TRAINING_CONFIG.get("use_val_geometry_bank", False)):
        # Bank mode: build the first train-bank geometry deterministically
        bank_rng = np.random.default_rng(100)  # same seed as trainer bank
        bank_ranges = TRAINING_CONFIG.get("bank_geo_ranges", None)
        params = sample_geometry_params(bank_rng, geometry_ranges=bank_ranges,
                                        holes_enabled=False)
        mesh = generate_dogbone(params, rng=bank_rng)
        coll = sample_collocation_points(mesh, rng=bank_rng)
    else:
        params = sample_geometry_params(rng)
        mesh = generate_dogbone(params, rng=rng)
        coll = sample_collocation_points(mesh, rng=rng)

    def _t(arr, req=False):
        t = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
        return t.requires_grad_(req) if req else t

    interior = _t(coll.interior_pts[:500], req=True)
    bpc = _t(coll.boundary_pc)
    u_delta = torch.tensor([1.0], dtype=torch.float32, device=device)
    x_max = torch.tensor([coll.x_max], dtype=torch.float32, device=device)
    y_max = torch.tensor([coll.y_max], dtype=torch.float32, device=device)

    mu = MATERIAL_CONFIG["mu"]
    lam = MATERIAL_CONFIG["lam"]

    model.eval()
    z = model.encode(bpc, x_max, y_max)
    uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
        interior, z, u_delta, x_max, y_max
    )

    P11, P12, P21, P22, detF = first_piola_kirchhoff_stress(
        du_dx, du_dy, dv_dx, dv_dy, mu, lam, stress_state
    )

    f_x, f_y, _ = equilibrium_residual(
        model, interior, z, u_delta, x_max, mu, lam, y_max, stress_state
    )

    print("=" * 60)
    print(f"INITIAL STATE DIAGNOSTIC (before training) [{stress_state}]")
    print("=" * 60)
    print(f"  |u| max:       {uv[...,0].abs().max().item():.4f}")
    print(f"  |v| max:       {uv[...,1].abs().max().item():.4f}")
    print(f"  |∇u| max:      {max(du_dx.abs().max().item(), du_dy.abs().max().item(), dv_dx.abs().max().item(), dv_dy.abs().max().item()):.4e}")
    print(f"  detF min/mean:  {detF.min().item():.4f} / {detF.mean().item():.4f}")
    frac_neg = (detF < 0.0).float().mean().item()
    frac_low = (detF < 0.1).float().mean().item()
    print(f"  detF < 0:       {100*frac_neg:.1f}%")
    print(f"  detF < 0.1:     {100*frac_low:.1f}%")
    print(f"  |P| max:        {max(P11.abs().max().item(), P12.abs().max().item(), P21.abs().max().item(), P22.abs().max().item()):.4e}")
    print(f"  |Div(P)| max:   {max(f_x.abs().max().item(), f_y.abs().max().item()):.4e}")

    L0 = NONDIM_SCALES["L0"]
    S0 = NONDIM_SCALES["S0"]
    L_eq = torch.mean((L0/S0 * f_x)**2 + (L0/S0 * f_y)**2)
    print(f"  L_eq (nondim):  {L_eq.item():.4e}")
    print("=" * 60)

    if frac_neg > 0.01:
        print("  ⚠ WARNING: >1% of points have detF < 0 at initialization!")
    elif L_eq.item() > 1e6:
        print("  ⚠ WARNING: L_eq is still very large — check output_scale.")
    else:
        print("  ✓ Initial state looks physically sane.")
    print()
    model.train()


def main():
    parser = argparse.ArgumentParser(description="PI-GINOT Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train"], help="Run mode")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu or cuda")
    args, _ = parser.parse_known_args()

    # Seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Config overrides
    config = TRAINING_CONFIG.copy()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    # Model
    model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG)

    # Stress state
    stress_state = MATERIAL_CONFIG.get("state", "plane strain")
    print(f"  Stress formulation: {stress_state}")

    # Loss
    loss_fn = PhysicsLoss(
        mu=MATERIAL_CONFIG["mu"],
        lam=MATERIAL_CONFIG["lam"],
        w_equilibrium=config["w_equilibrium"],
        w_trac_top=config["w_trac_top"],
        w_trac_arc=config["w_trac_arc"],
        w_traction_partial=config["w_traction_partial"],
        w_barrier=config["w_barrier"],
        j_min=config["barrier_delta"],
        adaptive_beta=config["adaptive_beta"],
        L0=NONDIM_SCALES["L0"],
        S0=NONDIM_SCALES["S0"],
        stress_state=stress_state,
    )

    # Trainer
    trainer = PI_GINOT_Trainer(
        model=model, loss_fn=loss_fn, config=config,
        device=device, save_dir=args.save_dir,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.mode == "train":
        # Run diagnostic before training
        inspect_initial_state(model, device=device)
        trainer.fit()


if __name__ == "__main__":
    main()
