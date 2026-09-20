#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified configuration for Physics-Informed GINOT on parametric DogBone specimens.

Merges:
  - GINOT geometry encoder hyperparameters
  - Physics-informed solution decoder settings
  - Neo-Hookean material parameters (from PINN-DogBone)
  - Parametric geometry ranges for DogBone variants
  - Collocation point counts and training settings

Unit system (consistent with PINN-DogBone):
    Length: mm
    Force:  N
    Stress: MPa  (= N/mm^2)

Stress formulation:
    The physics engine uses 1st Piola–Kirchhoff stress P in the reference
    configuration.  Equilibrium: Div(P) = 0.  Traction: P · N = 0.
    Current setting: plane stress.  F33 is solved from P33 = 0 by a Newton
    closure (physics/neo_hookean.solve_F33_plane_stress) and the 3D
    determinant J3D = J2D·F33 enters the log term.  Set
    MATERIAL_CONFIG["state"] = "plane strain" for the F33 = 1 formulation.

Topology:
    No-hole single-topology operator.  Holes are permanently disabled.
"""

import math

# Geometry encoder (branch) — PointCloudPerceiverChannelsEncoder
ENCODER_CONFIG = {
    "input_channels": 2,            # 2D point clouds (x, y)
    "out_c": 64,                    # Output embedding dimension
    "width": 64,                    # Hidden dimension in encoder
    "n_point": 32,                  # FPS sampled points from boundary PC
    "n_sample": 18,                 # Neighbours per point in ball query
    "radius": 0.15,                 # Ball query radius (normalised coords [-1,1])
    "d_hidden": [64, 64],           # MLP hidden dims in PointSetEmbedding
    "num_heads": 4,                 # Attention heads
    "cross_attn_layers": 1,         # Cross-attention layers in encoder
    "self_attn_layers": 3,          # Self-attention layers in encoder
    "fps_method": "fps",            # Farthest point sampling method
    "dropout": 0.0,                 # Dropout rate
}

# Solution decoder (trunk) — modified GINOT trunk with hard BCs
DECODER_CONFIG = {
    "embed_dim": 64,                # Must match encoder out_c
    "num_heads": 4,                 # Attention heads in cross-attention
    "cross_attn_layers": 6,         # Cross-attention layers in decoder
    "in_channels": 2,               # Query point dimension (x, y)
    "out_channels": 2,              # Output: (u, v) displacements
    # Stability settings
    "pe_max_deg": 6,                # NeRF PE max degree (2^5=32 max freq, was 15->16384)
    "output_scale": 1.0,            # Full scale from start — no warmup
}

# Material parameters — Neo-Hookean hyperelasticity
MATERIAL_CONFIG = {
    "E": 760.0,                     # [MPa] Young's modulus
    "nu": 0.23,                     # [-]   Poisson's ratio
    "state": "plane stress",        # 'plane strain' or 'plane stress'
}

# Derived Lame parameters (computed from E, nu)
_E = MATERIAL_CONFIG["E"]
_nu = MATERIAL_CONFIG["nu"]
MATERIAL_CONFIG["mu"] = _E / (2.0 * (1.0 + _nu))
MATERIAL_CONFIG["lam"] = (_E * _nu) / ((1.0 + _nu) * (1.0 - 2.0 * _nu))

# Nondimensionalization scales
NONDIM_SCALES = {
    "L0": 50.0,                     # [mm]  Characteristic length
    "S0": MATERIAL_CONFIG["E"],     # [MPa] Characteristic stress (= E)
}

# Parametric DogBone geometry — default shape and sampling ranges
#
# Quarter-model (double symmetry about y=0 and x=0):
#   Only the upper-right quarter of the specimen is modelled.
#   x=0 is the mid-length vertical symmetry plane.
#   y=0 is the mid-height horizontal symmetry plane.
#   Domain: [0, L_half] x [0, H_grip]
#
# Shape topology (counter-clockwise boundary):
#   P1(0,0) -> P2(L_half,0)           : bottom (symmetry, y=0)
#   P2(L_half,0) -> P3(L_half,H_grip) : right grip edge (displacement BC)
#   P3 -> P4 via fillet arc            : fillet (traction-free)
#   P4(x_g, H_gauge) -> P5(0, H_gauge): gauge top (traction-free)
#   P5(0, H_gauge) -> P1(0,0)         : left symmetry (symmetry, x=0)
#
# Fillet geometry:
#   L_half = L_total / 2
#   dH = H_grip - H_gauge
#   dx = sqrt(dH * (2*R_fillet - dH))
#   x_g = L_half - dx
GEOMETRY_DEFAULT = {
    "L_total": 54.0,                # [mm] Full specimen length (quarter uses L/2)
    "W_grip": 20.0,                 # [mm] Full grip width (H_grip = W_grip/2)
    "W_gauge": 10.0,                # [mm] Full gauge width (H_gauge = W_gauge/2)
    "R_fillet": 12.0,               # [mm] Fillet arc radius
    "holes": [],                    # Permanently empty — no-hole topology
}

# Uniform sampling ranges for parametric variants during training
GEOMETRY_RANGES = {
    "L_total": (40.0, 70.0),        # [mm]
    "W_grip": (16.0, 26.0),         # [mm]
    "W_gauge": (6.0, 14.0),         # [mm]
    "R_fillet": (8.0, 20.0),        # [mm]
}

# Hole configuration — permanently disabled for no-hole operator
HOLE_CONFIG = {
    "enabled": False,
    "max_holes": 0,
    "r_range": (0.0, 0.0),
    "margin": 0.0,
}

# Collocation points
COLLOCATION_CONFIG = {
    "n_interior": 4000,             # Interior collocation points per geometry
    "n_boundary_per_segment": 400,  # Boundary points per segment (5 segs * 400 = 2000 pool)
    "n_boundary_pc": 320,           # Boundary point cloud size for encoder
    "n_total_boundary": 1600,       # Total boundary points for physics loss

    # Boundary points are placed uniformly per segment by default, so points
    # per millimetre varies 5x (up to 9.5x over the geometry ranges) between
    # the 5 mm symmetry face and the 27 mm bottom face.  That biases the
    # encoder's view of the shape, and it starves the boundary collocation
    # sampler's pool on the longest segment, which then cannot receive its
    # arc-length share (1398 of 1600 points are returned at these settings).
    # True shares the same total out by arc length instead.  Default False so
    # existing checkpoints and results stay interpretable; flip it as an
    # ablation.  See verification/geometry_checks.py.
    "boundary_measure_consistent": False,
}

# Coordinate normalization
NORMALIZATION_CONFIG = {
    "normalize_coords": True,       # Normalize (x,y) to [-1, 1] before decoder
    "normalize_pc": True,           # Normalize boundary PC to [-1, 1] for encoder
}

# Training
TRAINING_CONFIG = {
    # Optimizer
    "optimizer": "adam",             # 'adam' or 'adamw'
    "learning_rate": 1e-4,          # Bumped up: need stronger gradients to escape baseline
    "weight_decay": 0.0,            # Weight decay (for AdamW)
    "grad_clip_norm": 1.0,          # Gradient clipping norm (tighter for physics)

    # Scheduler
    "scheduler": "reduce_on_plateau",
    "scheduler_factor": 0.7,        # LR reduction factor
    "scheduler_patience": 10,       # Patience before reducing LR

    # Epochs
    "epochs": 1500,                 # No-hole operator training
    "print_every": 10,              # Print loss every N epochs
    "val_every": 25,                # Validate every N epochs

    # Batching
    "batch_size": 8,                # Sample 8 from bank of 128 each epoch
    "num_workers": 4,               # DataLoader workers

    # Loss formulation.
    #   'penalty' — weighted residuals: Div P = 0 plus traction penalties
    #               (the original formulation; every number reported up to
    #               Phase 1 was produced with it)
    #   'energy'  — total potential energy Pi = integral Psi dV.  Every
    #               boundary here is hard Dirichlet, traction-free or a
    #               symmetry plane, so the external work term vanishes and
    #               this one term replaces w_equilibrium, w_trac_top,
    #               w_trac_arc and w_traction_partial; the traction
    #               conditions become natural BCs of the minimum.
    #               Measured against the FEM reference on three geometries:
    #               u 12.5x better, von Mises 5.6x, section force <= 0.5%,
    #               and 3.4-5.1x cheaper per epoch.  See
    #               docs/phase2_energy_form.md.
    #               Default stays 'penalty' until the operator is retrained,
    #               so existing results remain comparable.
    "loss_form": "penalty",
    # 'hybrid':     the energy functional with the equilibrium residual added
    #               back.  loss_form had always been one form or the other;
    #               Phase 2 compared them, the energy form won, and L_eq left
    #               the objective with the form that carried it.  Phase 6.4
    #               made the sum worth trying: supervising this architecture
    #               on stress reaches K_t R^2 +0.98, while a 0.2% displacement
    #               fit on the very specimens it is scored on still gives
    #               R^2 -0.55 -- so the objective needs a term sensitive to
    #               the stress, and L_eq is the only one the physics offers
    #               without changing the architecture.  Tractions stay
    #               natural BCs.  See physics/weak_form.py::HybridLoss.
    "w_equilibrium_hybrid": 100.0,  # same nondimensionalisation as
                                    # w_equilibrium, so the penalty form's
                                    # value is the sensible starting point
    "energy_quad_degree": 2,        # fixed-rule degree (unused when resampling)
    "energy_n_elem": 1400,          # quadrature triangles per geometry
    "energy_resample": True,        # stratified resampling — DO NOT disable:
                                    # a fixed rule collapses (Pi/Pi_ref -> 0.007
                                    # while the field gets 3x worse)
    "energy_n_per_elem": 3,         # stratified samples per triangle
    # Traction-free residuals under the ENERGY form.  These are redundant at
    # the exact minimiser -- Pi's natural BCs are exactly P.N = 0 on these
    # segments -- so they cannot bias the converged answer.  What they change
    # is the approach.  docs/phase3_transverse_conditioning.md measures the
    # same relative field error costing 183x more energy in the axial
    # direction than the transverse one, so the transverse component sits in
    # the objective's near-null space and "satisfied at the minimum" never
    # arrives.  The gauge-top term is the one Phase 1 identified as the only
    # carrier of transverse information.  0.0 = off, i.e. the energy form
    # exactly as Phase 2 measured it.
    "energy_w_trac_top": 0.0,
    "energy_w_trac_arc": 0.0,
    # Section anchor under the energy form.  This is a WEIGHT, not a gate: the
    # energy loss is O(5e-4) nondimensional against a penalty loss of O(1), so
    # sharing `w_resultant` (10.0) would make the anchor dominate by orders of
    # magnitude for no stated reason.  0.0 = off, which is what Phase 2 shipped
    # on the strength of a <=0.5% force error with no anchor at all -- measured
    # on ONE fine-tuned specimen.  The operator study (docs/phase2_operator.md)
    # shows that justification does not transfer: with capacity shared across a
    # bank the energy form's force error is 3.5% against the penalty form's
    # 2.5%.  Natural BCs are only satisfied AT the minimum.
    "w_resultant_energy": 0.0,
    # Weight on the ABSOLUTE anchor under the energy form, kept separate from
    # the consistency weight above because the two terms differ by ~1000x in
    # magnitude: the consistency terms start near 0.4, the anchor is a squared
    # relative error (~3e-4 at a 2% miss).  A single scalar over the sum
    # cannot tune both -- any weight that makes the level matter makes the
    # consistency terms dominate the whole loss.  The first anchor sweep
    # (docs/phase2_operator.md) varied only the sum and so appeared to test the
    # absolute anchor while it contributed ~0.01% of the signal.
    "w_resultant_abs_energy": 0.0,

    # Geometry auxiliary loss: supervise the ENCODER to predict the four
    # dimensionless shape descriptors (geometry/descriptors.py) from its
    # latent.  Targets are known for free when a training geometry is
    # generated, so this costs no reference solution.
    #
    # The motivation is measured, not assumed.  The encoder carries the taper
    # ratio at probe R^2 0.974 and the fillet at 0.03-0.30, and making the
    # fillet more available (4x the FPS centroids, decodability 0.156 -> 0.451)
    # changes no field metric by more than 1%.  So it is not that the encoder
    # cannot see the fillet -- it never learns to, because the only reward for
    # doing so arrives through a transverse channel the objective attenuates
    # 183x.  This term pays for the extraction directly.
    # See docs/phase4_geometry_encoder.md.  0.0 = off.
    "w_geom_aux": 0.0,

    # Geometry conditioning path.
    #   'pointcloud' — the GeometryEncoder over the boundary point cloud
    #                  (default; generalises beyond the parametric family)
    #   'parameters' — an MLP over the four dimensionless shape descriptors
    #                  (geometry/descriptors.py), valid only for this
    #                  parametric family, where the four numbers are known
    #
    # This is the single largest lever measured in the project.  At equal
    # budget, on eight held-out geometries spanning taper 0.31-0.82 and over
    # three seeds, 'parameters' beats 'pointcloud' by 3.32 +- 0.01 on the
    # transverse displacement, 3.58 +- 0.31 on the axial one and 2.25 +- 0.14
    # on von Mises; trained out to 1600 epochs the transverse ratio is 5.30.
    # Phase 4 ruled out closing that gap by repairing the encoder in place --
    # sampling resolution, a supervised geometry head, and both together each
    # buy at most 1.21x.  See docs/phase4_geometry_encoder.md and
    # docs/phase5_hardening.md.
    #
    # Default is 'pointcloud' so existing checkpoints and results stay valid.
    "conditioning": "pointcloud",

    # Physics loss weights (initial, before adaptive balancing)
    "w_equilibrium": 100.0,         # Raised 10x: must dominate to break uniform-strain baseline
    "w_trac_top": 2.0,              # Reduced: equilibrium + resultant should drive learning
                                    # NOTE: this is the only term that sets the
                                    # lateral contraction (P.N=0 on gauge_top
                                    # means P22=P12=0), and it is the lowest
                                    # weight here -- 50x below w_equilibrium.
                                    # Raising it to 100 cuts the transverse
                                    # displacement error 1.46x on average over
                                    # 7 geometries spanning the taper range --
                                    # 2.1x where the operator is weakest, ~1.0x
                                    # where it is already good -- at no axial
                                    # cost and ~1pp on the section force.
                                    # Not changed by default: all the evidence
                                    # is single-geometry training, and this is
                                    # a training-wide knob. Ablation row. See
                                    # docs/phase1_transverse_study.md.
    "w_trac_arc": 20.0,             # Reduced: was over-weighting local arc at expense of global balance
    "w_traction_partial": 2.0,      # Reduced for same reason
    "w_barrier": 1e3,               # detF barrier weight (strong during warmup)
    "w_resultant": 10.0,            # Raised: primary tool for enforcing N(gauge)=N(grip)
    "n_resultant_slices": 16,       # Number of vertical slices for resultant
    "n_resultant_y_pts": 128,       # Points per slice for integration

    # Section-resultant anchoring (see physics/uniaxial.py).  Errors below are
    # against the finite-element reference (verification/fem) on the
    # 24-geometry validation bank -- NOT against each other.  Measuring the
    # anchors against one another is what produced the withdrawn "5-30% low"
    # figure for the legacy form: `series` itself runs ~2.5% high, so the two
    # errors were being added instead of netted.
    #                                          mean |err|   max |err|
    #   'series'       — 1D varying-section finite-strain solve  [default]
    #                                              2.49%       5.07%
    #   'series_calibrated' — the same, x0.97517 fitted against the reference
    #                    on the TRAINING bank, so the validation numbers below
    #                    are held out.  Makes the anchor weakly
    #                    reference-dependent, so it is opt-in.
    #                                              0.74%       2.54%
    #   'uniform'      — finite-strain uniform bar at average stretch.
    #                    Corrects the linearisation but not the varying
    #                    section, which is the larger term and the other way,
    #                    so it is worse than doing nothing.   [ablation]
    #                                              6.02%      13.46%
    #   'small_strain' — legacy E*(u/L)*2H                     [ablation]
    #                                              3.60%      10.96%
    #   'none'         — rely on reaction self-consistency alone
    "resultant_anchor": "series",
    "w_resultant_anchor": 0.5,      # Weight of the absolute anchor term
    "w_resultant_reaction": 0.5,    # Weight of the grip reaction-consistency term
                                    # (0.5 keeps the total anchoring magnitude
                                    #  comparable to the old single 0.5*L_anchor,
                                    #  so the bias fix is not confounded by a
                                    #  change in how hard the term pulls)

    # Barrier warmup (exponentially ramp down barrier after early locking)
    "barrier_warmup_epochs": 300,
    "barrier_final_weight": 1.0,
    "barrier_delta": 0.05,          # detF < delta triggers barrier — used for single-geo diagnostic
    "adaptive_beta": 0.1,           # EMA blending factor for adaptive weights
    "adaptive_update_every": 100,   # Update adaptive weights every N epochs
    "adaptive_start_epoch": 200,    # Don't update adaptive weights before this epoch

    # EMA-smoothed loss for LR scheduler (avoids noisy single-batch reactions)
    "ema_alpha": 0.1,               # EMA blending factor: ema = (1-a)*ema + a*new

    # Checkpointing
    "save_best_only": True,
    "checkpoint_dir": "checkpoints",

    # Checkpoint gating thresholds
    "ckpt_min_swap_du": 0.03,       # Relaxed for Stage A
    "ckpt_max_section_cv": 0.15,    # Relaxed for Stage A

    # Geometry visualization
    "save_geo": True,               # Save geometry PNGs on the first epoch

    # Field visualization
    "plot_every": 50,               # Save displacement/stress field plots every N epochs (0=off)

    # Single-geometry mode (operator bypass)
    "single_geometry": False,       # False = geometry bank mode

    # Geometry bank
    "use_train_geometry_bank": True,   # Fixed train bank for consistent exposure
    "use_val_geometry_bank": True,     # Fixed validation bank
    "bank_train_size": 128,            # Training bank size (128 geometries)
    "bank_val_size": 24,               # Validation bank size
    "hole_probability": 0.0,           # Permanently zero — no holes
    "resample_train_collocation": True, # Resample collocation for train bank each epoch
    "bank_geo_ranges": {             # Full 4D parameter range
        "L_total": (40.0, 70.0),
        "W_grip": (16.0, 26.0),
        "W_gauge": (6.0, 14.0),
        "R_fillet": (8.0, 20.0),
    },
}

# Curriculum learning — permanently disabled for no-hole operator
CURRICULUM_CONFIG = {
    "enabled": False,
    "phases": [],
}

# Loading configuration
LOADING_CONFIG = {
    "u_max": 1.0,                   # [mm] Maximum applied displacement
}

# Output and reproducibility
OUTPUT_CONFIG = {
    "save_weights": True,
    "save_plots": True,
    "plot_dpi": 200,
    "results_dir": "results",
}

RANDOM_SEED = 42


# Geometry constraint validation (quarter-model)
def validate_geometry(params: dict) -> bool:
    """Check that a geometry parameterization is physically valid.

    Quarter-model constraints:
      - W_gauge < W_grip  (gauge narrower than grip)
      - R_fillet > (W_grip - W_gauge) / 2  (arc can span the height difference)
      - Gauge length > 0  (fillet fits within the half-length)

    Holes are permanently disabled — any non-empty holes list is rejected.
    """
    H_grip = params["W_grip"] / 2.0
    H_gauge = params["W_gauge"] / 2.0
    dH = H_grip - H_gauge
    R = params["R_fillet"]
    L = params["L_total"]
    L_half = L / 2.0

    if dH <= 0:
        return False
    if R <= dH:
        return False

    dx = math.sqrt(dH * (2.0 * R - dH))
    x_g = L_half - dx               # gauge-fillet transition
    if x_g < 2.0:                   # minimum 2 mm gauge
        return False

    # Reject any geometry that somehow has holes
    if len(params.get("holes", [])) > 0:
        return False

    return True


def get_fillet_geometry(params: dict) -> dict:
    """Compute derived fillet quantities for the quarter-model.

    Returns dict with: L_half, H_grip, H_gauge, dH, dx, x_g, gauge_length,
    and arc center coordinates for the single (right-side) fillet.
    """
    H_grip = params["W_grip"] / 2.0
    H_gauge = params["W_gauge"] / 2.0
    dH = H_grip - H_gauge
    R = params["R_fillet"]
    L = params["L_total"]
    L_half = L / 2.0
    dx = math.sqrt(dH * (2.0 * R - dH))

    x_g = L_half - dx

    arc_center = (x_g, H_gauge + R)

    return {
        "L_total": L,
        "L_half": L_half,
        "H_grip": H_grip,
        "H_gauge": H_gauge,
        "dH": dH,
        "dx": dx,
        "x_g": x_g,
        "gauge_length": x_g,
        "arc_center": arc_center,
        "R_fillet": R,
    }
