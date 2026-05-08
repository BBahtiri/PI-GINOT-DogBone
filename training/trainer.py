#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-geometry training loop for PI-GINOT.

Each training step:
  1. Samples B random DogBone geometries (via curriculum scheduler)
  2. Saves each geometry as PNG (first epoch only)
  3. Generates collocation points for each geometry
  4. Runs the full PI-GINOT forward pass (encode once, decode many)
  5. Computes the physics loss (Div(P)=0 + P·N=0 + detF barrier)
  6. Backpropagates and updates model weights

Supports:
  - ADAM / AdamW optimizer with gradient clipping
  - ReduceLROnPlateau scheduler
  - Periodic adaptive loss-weight updates
  - Model checkpointing (best validation loss)
  - Training history logging
  - Geometry plot saving (first epoch only)
  - Periodic displacement / stress field plots
"""

import os
import json
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    TRAINING_CONFIG, MATERIAL_CONFIG, LOADING_CONFIG,
    ENCODER_CONFIG, DECODER_CONFIG, COLLOCATION_CONFIG,
    GEOMETRY_DEFAULT, NONDIM_SCALES, HOLE_CONFIG,
)
from geometry.parametric_dogbone import generate_dogbone, sample_geometry_params, plot_dogbone
from geometry.collocation import sample_collocation_points
from models.pi_ginot import PI_GINOT
from physics.losses import PhysicsLoss
from physics.neo_hookean import full_stress_state, first_piola_kirchhoff_stress
from .curriculum import CurriculumScheduler


class PI_GINOT_Trainer:
    """Training manager for the PI-GINOT model.

    Args:
        model:     PI_GINOT model instance.
        loss_fn:   PhysicsLoss instance.
        config:    Training config dict (defaults to TRAINING_CONFIG).
        device:    'cpu' or 'cuda'.
        save_dir:  Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: PI_GINOT,
        loss_fn: PhysicsLoss,
        config: Optional[dict] = None,
        device: str = "cpu",
        save_dir: str = "checkpoints",
    ):
        self.config = config or TRAINING_CONFIG
        self.device = torch.device(device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Geometry plot directory
        self.geo_dir = os.path.join(save_dir, "geometries")
        os.makedirs(self.geo_dir, exist_ok=True)

        # Field plot directory
        self.fields_dir = os.path.join(save_dir, "fields")
        os.makedirs(self.fields_dir, exist_ok=True)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)

        # Optimizer
        opt_name = self.config.get("optimizer", "adam")
        lr = self.config["learning_rate"]
        wd = self.config.get("weight_decay", 0.0)
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=wd
            )
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=self.config["scheduler_factor"],
            patience=self.config["scheduler_patience"],
        )

        # Curriculum
        self.curriculum = CurriculumScheduler()

        # RNG for reproducibility
        self.rng = np.random.default_rng(42)

        # Whether to save geometry PNGs on the first epoch
        self.save_geo = self.config.get("save_geo", True)

        # Field plot interval (0 = disabled)
        self.plot_every = self.config.get("plot_every", 100)

        # EMA-smoothed loss for LR scheduler
        self._ema_loss = None
        self._ema_alpha = self.config.get("ema_alpha", 0.1)

        # Fixed validation geometry set (built lazily)
        self._val_set = None
        self._val_signature = None
        self._val_every = self.config.get("val_every", 10)

        # History
        self.history = {
            "epoch": [], "loss": [], "raw_loss": [],
            "L_eq": [], "L_trac_top": [], "L_trac_arc": [],
            "L_trac_hole": [], "L_part": [], "L_barrier": [], "L_resultant": [],
            "w_eq": [], "w_trac_top": [], "w_trac_arc": [],
            "val_raw_loss": [], "val_hole_loss": [],
            "ckpt_swap_du": [], "ckpt_section_cv": [],
            "lr": [], "phase": [],
        }
        self.best_loss = float("inf")  # legacy (unused in Run 3)
        self.best_val_raw = float("inf")       # best val loss (always updated)
        self.best_gated_val_raw = float("inf")  # best val loss among gate-passing ckpts
        self.start_epoch = 0

        # Single-geometry mode (operator bypass)
        self.single_geometry = self.config.get("single_geometry", False)
        self._fixed_mesh = None
        if self.single_geometry:
            fixed_rng = np.random.default_rng(0)
            self._fixed_mesh = generate_dogbone(GEOMETRY_DEFAULT, rng=fixed_rng)
            print(f"  [Single-geometry mode] Fixed dogbone: "
                  f"L={GEOMETRY_DEFAULT['L_total']}, "
                  f"W_gauge={GEOMETRY_DEFAULT['W_gauge']}, "
                  f"R={GEOMETRY_DEFAULT['R_fillet']}")

        # Profile plot directory
        self.profiles_dir = os.path.join(save_dir, "profiles")
        os.makedirs(self.profiles_dir, exist_ok=True)

        # Geometry bank flags (Phase 2: separate train/val control)
        self._use_train_bank = self.config.get("use_train_geometry_bank", False)
        self._use_val_bank = self.config.get("use_val_geometry_bank", True)
        self._train_bank = []
        self._val_bank = []

        bank_ranges = self.config.get("bank_geo_ranges", None)

        # Resample training collocation each epoch?
        self._resample_train_coll = self.config.get("resample_train_collocation", True)

        if self._use_train_bank and not self.single_geometry:
            n_train = self.config.get("bank_train_size", 128)
            self._train_bank = self._build_geometry_bank(
                n_train, bank_ranges, holes_on=False, seed=100,
                mesh_only=self._resample_train_coll,
            )
            mode_str = "params-only (mesh+coll regenerated)" if self._resample_train_coll else "fixed coll"
            print(f"  [Train bank mode] {n_train} geometries ({mode_str})")
            for i, entry in enumerate(self._train_bank):
                p = entry if self._resample_train_coll else entry[0].params
                print(f"    train[{i:2d}]: L={p['L_total']:.1f} "
                      f"W_grip={p['W_grip']:.1f} "
                      f"W_gauge={p['W_gauge']:.1f} "
                      f"R={p['R_fillet']:.1f}")

        if self._use_val_bank and not self.single_geometry:
            n_val = self.config.get("bank_val_size", 8)
            # No-hole validation bank
            self._val_bank = self._build_geometry_bank(
                n_val, bank_ranges, holes_on=False, seed=200,
            )
            print(f"  [Val bank: no-hole] {n_val} fixed geometries")
            for i, (m, _) in enumerate(self._val_bank):
                p = m.params
                print(f"    val  [{i:2d}]: L={p['L_total']:.1f} "
                      f"W_grip={p['W_grip']:.1f} "
                      f"W_gauge={p['W_gauge']:.1f} "
                      f"R={p['R_fillet']:.1f}")

            # Hole validation bank — permanently disabled
            self._val_bank_hole = []

        if not self._use_train_bank and not self.single_geometry:
            print(f"  [Online random training] sampling from bank_geo_ranges")


    # Geometry bank construction
    def _build_geometry_bank(self, n, geo_ranges, holes_on, seed,
                             min_holes: int = 0, mesh_only: bool = False):
        """Pre-generate a fixed bank of geometries.

        When mesh_only=True (train bank), returns a list of geometry
        parameter dicts.  The mesh and collocation are regenerated fresh
        each epoch so interior + boundary points vary, giving better
        operator-learning coverage.

        When mesh_only=False (val bank), returns (mesh, coll) pairs with
        frozen collocation for clean progress tracking.
        """
        bank_rng = np.random.default_rng(seed)
        bank = []
        for _ in range(n):
            params = sample_geometry_params(
                bank_rng, geometry_ranges=geo_ranges,
                holes_enabled=holes_on,
            )
            if mesh_only:
                bank.append(params)
            else:
                n_bnd_seg = COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
                mesh = generate_dogbone(params, n_pts_per_segment=n_bnd_seg,
                                        rng=bank_rng)
                coll = sample_collocation_points(mesh, rng=bank_rng)
                bank.append((mesh, coll))
        return bank

    # Core training loop
    def fit(self, epochs: Optional[int] = None, print_every: Optional[int] = None):
        """Run the full training loop."""
        epochs = epochs or self.config["epochs"]
        print_every = print_every or self.config["print_every"]
        batch_size = self.config["batch_size"]
        u_max = LOADING_CONFIG["u_max"]
        clip_norm = self.config.get("grad_clip_norm", 0.5)
        adaptive_every = self.config.get("adaptive_update_every", 100)
        use_adaptive = self.config.get("use_adaptive_weights", True)

        adaptive_start = self.config.get("adaptive_start_epoch", 200)

        print(f"Training PI-GINOT for {epochs} epochs, batch_size={batch_size}, "
              f"device={self.device}")
        print(f"  Model params: {self.model.count_parameters()['trainable']:,}")
        print(f"  Material: mu={MATERIAL_CONFIG['mu']:.1f}, "
              f"lam={MATERIAL_CONFIG['lam']:.1f}")
        print(f"  Formulation: Div(P)=0, P·N=0 (reference configuration)")
        print(f"  Adaptive weighting starts at epoch {adaptive_start}")
        print(f"  LR scheduler uses EMA-smoothed validation loss (α={self._ema_alpha})")
        if self.save_geo:
            print(f"  Saving first-epoch geometry plots to {self.geo_dir}/")
        if self.plot_every > 0:
            print(f"  Saving field plots every {self.plot_every} epochs to {self.fields_dir}/")
        print()

        # Save all bank geometries upfront with descriptive filenames
        if self._use_train_bank and self.save_geo:
            plot_rng = np.random.default_rng(999)
            n_bnd_seg = COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
            for i, entry in enumerate(self._train_bank):
                if self._resample_train_coll:
                    m = generate_dogbone(entry, n_pts_per_segment=n_bnd_seg,
                                         rng=plot_rng)
                else:
                    m = entry[0]
                self._save_geometry_plot(m, epoch=0, geo_idx=i,
                                        prefix="train_bank")
            print(f"  Saved {len(self._train_bank)} train-bank geometry plots")

        if self._use_val_bank and self.save_geo and self._val_bank:
            for i, (m, _) in enumerate(self._val_bank):
                self._save_geometry_plot(m, epoch=0, geo_idx=i,
                                        prefix="val_nohole")
            print(f"  Saved {len(self._val_bank)} val-bank (no-hole) geometry plots")

        if self.save_geo and hasattr(self, "_val_bank_hole") and self._val_bank_hole:
            for i, (m, _) in enumerate(self._val_bank_hole):
                self._save_geometry_plot(m, epoch=0, geo_idx=i,
                                        prefix="val_hole")
            print(f"  Saved {len(self._val_bank_hole)} val-bank (hole) geometry plots")

        for epoch in range(self.start_epoch, epochs):

            # Output scale: fixed at 1.0 (no warmup — let network correct from epoch 0)
            self.model.decoder.output_scale = 1.0

            # Section-resultant: ON from epoch 0 (no ramp)
            # This is the primary force for breaking the uniform-strain baseline
            self._w_resultant_eff = self.config.get("w_resultant", 0.0)

            t0 = time.time()

            if self.single_geometry:
                geo_ranges = None   # not used — mesh is cached
                holes_on = False
                phase = "single"
            elif self._use_train_bank:
                geo_ranges = None   # not used — bank is pre-built
                holes_on = False
                phase = "train_bank"
            else:
                # Phase 2/3: online random sampling with mixed hole probability
                geo_ranges = self.config.get("bank_geo_ranges", None)
                hole_prob = self.config.get("hole_probability", 0.0)
                holes_enabled_global = HOLE_CONFIG.get("enabled", False)
                holes_on = holes_enabled_global and (hole_prob > 0)
                phase = "random+hole" if holes_on else "random"

            # Save geometry plots on first epoch (or bank upfront)
            save_geo_this_epoch = (
                self.save_geo
                and (epoch == self.start_epoch)
                and not self._use_train_bank  # bank saved separately below
            )

            batch_loss = self._train_one_epoch(
                batch_size, geo_ranges, holes_on, u_max, clip_norm,
                epoch=epoch, save_geo=save_geo_this_epoch,
            )

            # Unweighted raw loss (the true progress metric)
            raw_loss = (batch_loss["L_eq"] + batch_loss["L_trac_top"]
                        + batch_loss["L_trac_arc"] + batch_loss["L_trac_hole"]
                        + batch_loss["L_part"] + batch_loss["L_barrier"])
            L_res_batch = batch_loss.get("L_resultant", 0.0)
            w_res_eff = getattr(self, "_w_resultant_eff", 0.0)
            raw_total = raw_loss + w_res_eff * L_res_batch

            # Current adaptive weights
            w_eq_val = self.loss_fn.w_eq.item()
            w_top_val = self.loss_fn.w_trac_top.item()
            w_arc_val = self.loss_fn.w_trac_arc.item()

            # Periodic validation on fixed geometry set
            val_raw = None
            val_hole = None
            if (epoch + 1) % self._val_every == 0 or epoch == 0:
                val_raw = self._evaluate_validation(geo_ranges, holes_on, u_max)

                # Hole-bank validation (separate tracking)
                val_hole = None
                if self._val_bank_hole:
                    val_hole = self._evaluate_validation_hole(u_max)

                # EMA-smooth the validation loss for LR scheduler
                if self._ema_loss is None:
                    self._ema_loss = val_raw
                else:
                    self._ema_loss = ((1 - self._ema_alpha) * self._ema_loss
                                     + self._ema_alpha * val_raw)
                self.scheduler.step(self._ema_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Delayed adaptive weighting
            if (use_adaptive
                    and epoch >= adaptive_start
                    and (epoch + 1) % adaptive_every == 0):
                self._update_adaptive_weights(
                    batch_size, geo_ranges, holes_on, u_max
                )

            # Log
            self.history["epoch"].append(epoch)
            self.history["loss"].append(batch_loss["loss"])
            self.history["raw_loss"].append(raw_loss)
            self.history["L_eq"].append(batch_loss["L_eq"])
            self.history["L_trac_top"].append(batch_loss["L_trac_top"])
            self.history["L_trac_arc"].append(batch_loss["L_trac_arc"])
            self.history["L_trac_hole"].append(batch_loss["L_trac_hole"])
            self.history["L_part"].append(batch_loss["L_part"])
            self.history["L_barrier"].append(batch_loss["L_barrier"])
            self.history["L_resultant"].append(
                batch_loss.get("L_resultant_log", 0.0))
            self.history.setdefault("L_geom_aux", []).append(
                batch_loss.get("L_geom_aux_log", 0.0))
            self.history["w_eq"].append(w_eq_val)
            self.history["w_trac_top"].append(w_top_val)
            self.history["w_trac_arc"].append(w_arc_val)
            self.history["val_raw_loss"].append(val_raw)
            self.history["val_hole_loss"].append(
                val_hole if val_hole is not None else None)
            self.history["lr"].append(current_lr)
            self.history["phase"].append(phase)

            # Gated best-checkpoint selection (Run 3: dual tracking)
            if val_raw is not None:
                # Always track best val_raw
                improved_val = val_raw < self.best_val_raw
                if improved_val:
                    self.best_val_raw = val_raw

                # Run gates and track best gated checkpoint
                if self._use_val_bank and len(self._val_bank) >= 2:
                    swap_du_val, section_cv_val = self._checkpoint_gates()
                else:
                    swap_du_val, section_cv_val = (1.0, 0.0)
                min_swap = self.config.get("ckpt_min_swap_du", 0.03)
                max_cv = self.config.get("ckpt_max_section_cv", 0.15)
                gate_pass = (swap_du_val >= min_swap) and (section_cv_val <= max_cv)

                if gate_pass and val_raw < self.best_gated_val_raw:
                    self.best_gated_val_raw = val_raw
                    self._save_checkpoint(epoch, is_best=True)
                elif improved_val:
                    # Save as last-best-val even if gate fails
                    self._save_checkpoint(epoch, is_best=False)

                if not gate_pass and (epoch + 1) % print_every == 0:
                    status_str = "PASS" if gate_pass else "BLOCKED"
                    print(f"  [ckpt gate] swap_du={swap_du_val:.4f} "
                          f"(min={min_swap}), section_cv={section_cv_val:.4f} "
                          f"(max={max_cv}) — {status_str}")

                self.history["ckpt_swap_du"].append(swap_du_val)
                self.history["ckpt_section_cv"].append(section_cv_val)
            else:
                self.history["ckpt_swap_du"].append(None)
                self.history["ckpt_section_cv"].append(None)

            if (epoch + 1) % print_every == 0 or epoch == 0:
                dt = time.time() - t0
                val_str = f"val={val_raw:.3e}" if val_raw is not None else "val=—"
                val_h_str = f" val_h={val_hole:.3e}" if val_hole is not None else ""
                print(
                    f"Epoch {epoch+1:4d}/{epochs} | "
                    f"loss={batch_loss['loss']:.4e} raw={raw_loss:.4e} raw+res={raw_total:.4e} | "
                    f"L_eq={batch_loss['L_eq']:.3e} "
                    f"L_top={batch_loss['L_trac_top']:.3e} "
                    f"L_arc={batch_loss['L_trac_arc']:.3e} "
                    f"L_part={batch_loss['L_part']:.3e} "
                    f"L_bar={batch_loss['L_barrier']:.3e} L_res={batch_loss.get('L_resultant', 0.0):.3e} | "
                    f"w_top={w_top_val:.1f} w_arc={w_arc_val:.1f} w_res={w_res_eff:.2f} | "
                    f"{val_str}{val_h_str} lr={current_lr:.2e} | "
                    f"{phase} {dt:.1f}s"
                )

            # Periodic field plots
            if self.plot_every > 0 and (
                epoch == 0
                or (epoch + 1) % self.plot_every == 0
                or epoch == epochs - 1
            ):
                self._save_field_plots(epoch, geo_ranges, holes_on)
                if self.single_geometry:
                    self._save_profile_plots(epoch)
                if self._use_val_bank and (epoch + 1) % (self.plot_every * 2) == 0:
                    self.run_latent_diagnostics(epoch=epoch)

        # Run diagnostics at end of training
        if self._use_val_bank:
            self.run_latent_diagnostics(epoch=epochs - 1)
            print(f"\n  --- End-of-training stability diagnostics ---")
            for vi in range(min(len(self._val_bank), 4)):
                self.run_boundary_resampling_diagnostic(val_idx=vi)
                self.run_boundary_permutation_diagnostic(val_idx=vi)
            print(f"\n  --- End-of-training section-force diagnostics ---")
            for vi in range(min(len(self._val_bank), 4)):
                self.run_section_force_diagnostic(val_idx=vi)

        self._save_checkpoint(epochs - 1, is_best=False)
        self._save_history()
        print(f"\nTraining complete. best_val_raw={self.best_val_raw:.4e}, best_gated={self.best_gated_val_raw:.4e}")

    # Fixed validation set
    def _build_validation_set(self, geo_ranges, holes_on):
        """Build fixed validation geometries for clean progress tracking.

        In bank mode, returns the pre-built val bank.
        Otherwise uses a deterministic RNG so the same geometries are
        always used within a curriculum phase.
        """
        if self._use_val_bank and self._val_bank:
            return self._val_bank

        val_rng = np.random.default_rng(12345)
        n_val = self.config["batch_size"]
        val_set = []
        for _ in range(n_val):
            if self.single_geometry and self._fixed_mesh is not None:
                mesh = self._fixed_mesh
            else:
                params = sample_geometry_params(
                    val_rng, geometry_ranges=geo_ranges, holes_enabled=holes_on,
                )
                mesh = generate_dogbone(params, rng=val_rng)
            coll = sample_collocation_points(mesh, rng=val_rng)
            val_set.append((mesh, coll))
        return val_set

    @torch.no_grad()
    def _evaluate_validation(self, geo_ranges, holes_on, u_max):
        """Evaluate model on fixed validation geometries.

        Returns the mean unweighted raw loss (L_eq + L_trac + L_part + L_barrier)
        across all validation geometries.
        """
        # Lazily build or rebuild validation set when curriculum changes
        sig = (str(geo_ranges), holes_on, self.single_geometry, self._use_val_bank)
        if self._val_set is None or sig != self._val_signature:
            self._val_set = self._build_validation_set(geo_ranges, holes_on)
            self._val_signature = sig

        self.model.eval()
        raw_total = 0.0
        n = len(self._val_set)

        for val_idx, (mesh, coll) in enumerate(self._val_set):
            def _t(arr):
                return torch.tensor(arr, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)

            interior = _t(coll.interior_pts).requires_grad_(True)
            tf_pts = _t(coll.traction_free_pts).requires_grad_(True)
            tf_norms = _t(coll.traction_free_normals)
            pt_pts = _t(coll.partial_traction_pts).requires_grad_(True)
            pt_norms = _t(coll.partial_traction_normals)
            pt_dirs = torch.tensor(coll.partial_traction_dirs,
                                   dtype=torch.long, device=self.device)
            bpc = _t(coll.boundary_pc)
            u_d = torch.tensor([u_max], dtype=torch.float32, device=self.device)
            x_m = torch.tensor([coll.x_max], dtype=torch.float32, device=self.device)
            y_m = torch.tensor([coll.y_max], dtype=torch.float32, device=self.device)

            tf_tags = torch.tensor(coll.traction_free_tags,
                                   dtype=torch.long, device=self.device)

            # Deterministic FPS for bank val geometries
            v_sid = None
            if self._use_val_bank:
                v_sid = torch.tensor([100 + val_idx], dtype=torch.long,
                                     device=self.device)

            with torch.enable_grad():
                ld = self.loss_fn(
                    self.model, interior, tf_pts, tf_norms, tf_tags,
                    pt_pts, pt_norms, pt_dirs, bpc, u_d, x_m, y_m,
                    sample_ids=v_sid,
                )

                # Section-resultant loss (same as training)
                w_res = self.config.get("w_resultant", 0.0)
                L_res_val = 0.0
                if w_res > 0.0:
                    L_res_val = self._compute_section_resultant_loss(
                        self.model, ld["geometry_latent"], mesh,
                        u_d, x_m, y_m,
                        n_slices=self.config.get("n_resultant_slices", 10),
                        n_y_pts=self.config.get("n_resultant_y_pts", 64),
                    ).item()

            raw = (ld["L_eq_log"] + ld["L_trac_top_log"] + ld["L_trac_arc_log"]
                   + ld["L_trac_hole_log"] + ld["L_part_log"] + ld["L_barrier_log"]
                   + w_res * L_res_val)
            raw_total += raw / n

        self.model.train()
        return raw_total


    def _evaluate_validation_hole(self, u_max):
        """Evaluate model on fixed hole-bank validation geometries.

        Same logic as _evaluate_validation but uses _val_bank_hole.
        Returns mean raw loss including L_resultant.
        """
        if not self._val_bank_hole:
            return None

        self.model.eval()
        raw_total = 0.0
        n = len(self._val_bank_hole)

        for val_idx, (mesh, coll) in enumerate(self._val_bank_hole):
            def _t(arr):
                return torch.tensor(arr, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)

            interior = _t(coll.interior_pts).requires_grad_(True)
            tf_pts = _t(coll.traction_free_pts).requires_grad_(True)
            tf_norms = _t(coll.traction_free_normals)
            pt_pts = _t(coll.partial_traction_pts).requires_grad_(True)
            pt_norms = _t(coll.partial_traction_normals)
            pt_dirs = torch.tensor(coll.partial_traction_dirs,
                                   dtype=torch.long, device=self.device)
            bpc = _t(coll.boundary_pc)
            u_d = torch.tensor([u_max], dtype=torch.float32, device=self.device)
            x_m = torch.tensor([coll.x_max], dtype=torch.float32, device=self.device)
            y_m = torch.tensor([coll.y_max], dtype=torch.float32, device=self.device)
            tf_tags = torch.tensor(coll.traction_free_tags,
                                   dtype=torch.long, device=self.device)

            # Deterministic FPS for hole val geometries (offset by 200)
            v_sid = torch.tensor([200 + val_idx], dtype=torch.long,
                                 device=self.device)

            with torch.enable_grad():
                ld = self.loss_fn(
                    self.model, interior, tf_pts, tf_norms, tf_tags,
                    pt_pts, pt_norms, pt_dirs, bpc, u_d, x_m, y_m,
                    sample_ids=v_sid,
                )

                w_res = self.config.get("w_resultant", 0.0)
                L_res_val = 0.0
                if w_res > 0.0:
                    L_res_val = self._compute_section_resultant_loss(
                        self.model, ld["geometry_latent"], mesh,
                        u_d, x_m, y_m,
                        n_slices=self.config.get("n_resultant_slices", 10),
                        n_y_pts=self.config.get("n_resultant_y_pts", 64),
                    ).item()

            raw = (ld["L_eq_log"] + ld["L_trac_top_log"] + ld["L_trac_arc_log"]
                   + ld["L_trac_hole_log"] + ld["L_part_log"] + ld["L_barrier_log"]
                   + w_res * L_res_val)
            raw_total += raw / n

        self.model.train()
        return raw_total

    # Checkpoint gates: lightweight swap + section-CV check
    @torch.no_grad()
    def _checkpoint_gates(self, n_geo: int = 2) -> tuple:
        """Compute lightweight latent-swap and section-CV gate values.

        Uses the first `n_geo` no-hole val-bank geometries.
        Returns (mean_swap_du, mean_section_cv).
        """
        n_geo = min(n_geo, len(self._val_bank))
        if n_geo < 2:
            return (1.0, 0.0)  # pass by default if not enough geos

        self.model.eval()
        mu = MATERIAL_CONFIG["mu"]
        lam = MATERIAL_CONFIG["lam"]

        # Encode first n_geo val geometries
        latents = []
        colls = []
        for i in range(n_geo):
            mesh, coll = self._val_bank[i]
            bpc = torch.tensor(coll.boundary_pc, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
            x_m = torch.tensor([coll.x_max], dtype=torch.float32,
                               device=self.device)
            y_m = torch.tensor([coll.y_max], dtype=torch.float32,
                               device=self.device)
            sid = torch.tensor([100 + i], dtype=torch.long, device=self.device)
            z = self.model.encode(bpc, x_m, y_m, sample_ids=sid)
            latents.append(z)
            colls.append(coll)

        # Swap sensitivity (Δu correction only)
        swap_du_vals = []
        for i in range(n_geo):
            coll_i = colls[i]
            query = torch.tensor(coll_i.interior_pts, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            u_d = torch.tensor([LOADING_CONFIG["u_max"]], dtype=torch.float32,
                               device=self.device)
            x_m = torch.tensor([coll_i.x_max], dtype=torch.float32,
                               device=self.device)
            y_m = torch.tensor([coll_i.y_max], dtype=torch.float32,
                               device=self.device)

            # Correct latent
            uv_correct = self.model.decode(query, latents[i], u_d, x_m, y_m)
            u_correct = uv_correct[0, :, 0].cpu().numpy()
            u_base = LOADING_CONFIG["u_max"] * coll_i.interior_pts[:, 0] / coll_i.x_max
            du_correct = u_correct - u_base
            norm_du = max(np.linalg.norm(du_correct), 1e-12)

            # Swapped latents
            for j in range(n_geo):
                if j == i:
                    continue
                uv_swap = self.model.decode(query, latents[j], u_d, x_m, y_m)
                u_swap = uv_swap[0, :, 0].cpu().numpy()
                du_swap = u_swap - u_base
                swap_du_vals.append(np.linalg.norm(du_correct - du_swap) / norm_du)

        mean_swap_du = float(np.mean(swap_du_vals)) if swap_du_vals else 0.0

        # Section-force CV
        cv_vals = []
        for i in range(n_geo):
            cv = self.run_section_force_diagnostic(val_idx=i, n_slices=6,
                                                   use_hole_bank=False)
            if cv is not None:
                cv_vals.append(cv)

        mean_section_cv = float(np.mean(cv_vals)) if cv_vals else 0.0

        self.model.train()
        return (mean_swap_du, mean_section_cv)

    # Single epoch
    def _train_one_epoch(
        self, batch_size, geo_ranges, holes_on, u_max, clip_norm,
        epoch: int = 0, save_geo: bool = False,
    ) -> dict:
        """Train on one batch of randomly sampled geometries."""
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device)
        accum = {"L_eq": 0.0, "L_trac_top": 0.0, "L_trac_arc": 0.0,
                 "L_trac_hole": 0.0, "L_part": 0.0, "L_barrier": 0.0,
                 "L_resultant": 0.0}

        # Without-replacement for bank: shuffle all, take batch_size
        if self._use_train_bank:
            if not hasattr(self, "_bank_order") or len(self._bank_order) < batch_size:
                order = list(range(len(self._train_bank)))
                self.rng.shuffle(order)
                self._bank_order = list(order)
            epoch_indices = [self._bank_order.pop(0) for _ in range(batch_size)]
        else:
            epoch_indices = [None] * batch_size

        # Balanced topology: 2 no-hole + 2 hole (Run 2)
        holes_global = HOLE_CONFIG.get("enabled", False)
        if holes_global and not self._use_train_bank and not self.single_geometry:
            topologies = [False, False, True, True]
            self.rng.shuffle(topologies)
        else:
            topologies = [None] * batch_size  # None = use default logic

        for geo_idx, bidx in enumerate(epoch_indices):
            forced_holes = topologies[geo_idx] if geo_idx < len(topologies) else None
            loss_dict = self._compute_single_geometry_loss(
                geo_ranges, holes_on, u_max,
                epoch=epoch, geo_idx=geo_idx, save_geo=save_geo,
                bank_idx=bidx, forced_holes=forced_holes,
            )
            (loss_dict["loss"] / batch_size).backward()
            total_loss += loss_dict["loss"].detach() / batch_size
            for k in accum:
                accum[k] += loss_dict[f"{k}_log"] / batch_size

        nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            **accum,
        }

    # Single-geometry forward + physics loss
    def _compute_single_geometry_loss(
        self, geo_ranges, holes_on, u_max,
        epoch: int = 0, geo_idx: int = 0, save_geo: bool = False,
        bank_idx: int = None, forced_holes: bool = None,
    ) -> dict:
        """Sample one geometry, build collocation, compute physics loss."""
        idx = None  # bank index (None for non-bank paths)
        if self.single_geometry and self._fixed_mesh is not None:
            # Reuse fixed mesh, resample collocation for variance reduction
            mesh = self._fixed_mesh
            coll = sample_collocation_points(mesh, rng=self.rng)
        elif self._use_train_bank:
            # Use specified bank index (from without-replacement schedule)
            idx = bank_idx if bank_idx is not None else int(
                self.rng.integers(0, len(self._train_bank)))
            if self._resample_train_coll:
                params = self._train_bank[idx]
                n_bnd_seg = COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
                mesh = generate_dogbone(
                    params, n_pts_per_segment=n_bnd_seg,
                    n_interior=COLLOCATION_CONFIG["n_interior"],
                    rng=self.rng,
                )
                coll = sample_collocation_points(mesh, rng=self.rng)
            else:
                mesh, coll = self._train_bank[idx]
        else:
            # Per-sample hole decision (balanced topology in Run 2)
            if forced_holes is not None:
                this_sample_holes = forced_holes
            else:
                hole_prob = self.config.get("hole_probability", 0.0)
                holes_global = HOLE_CONFIG.get("enabled", False)
                this_sample_holes = holes_global and (self.rng.random() < hole_prob)
            if this_sample_holes:
                # Retry until we get at least 1 hole (up to 50 attempts)
                for _ha in range(50):
                    params = sample_geometry_params(
                        self.rng, geometry_ranges=geo_ranges,
                        holes_enabled=True,
                    )
                    if len(params.get("holes", [])) >= 1:
                        break
            else:
                params = sample_geometry_params(
                    self.rng, geometry_ranges=geo_ranges,
                    holes_enabled=False,
                )
            mesh = generate_dogbone(params, rng=self.rng)
            coll = sample_collocation_points(mesh, rng=self.rng)

        # Save geometry plot
        if save_geo:
            self._save_geometry_plot(mesh, epoch, geo_idx)

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

        interior = _t(coll.interior_pts).requires_grad_(True)
        tf_pts = _t(coll.traction_free_pts).requires_grad_(True)
        tf_norms = _t(coll.traction_free_normals)
        tf_tags = torch.tensor(coll.traction_free_tags,
                               dtype=torch.long, device=self.device)
        pt_pts = _t(coll.partial_traction_pts).requires_grad_(True)
        pt_norms = _t(coll.partial_traction_normals)
        pt_dirs = torch.tensor(coll.partial_traction_dirs,
                               dtype=torch.long, device=self.device)
        bpc = _t(coll.boundary_pc)
        u_delta = torch.tensor([u_max], dtype=torch.float32, device=self.device)
        x_max = torch.tensor([coll.x_max], dtype=torch.float32, device=self.device)
        y_max = torch.tensor([coll.y_max], dtype=torch.float32, device=self.device)

        # Deterministic FPS for bank geometries (sample_ids enables caching)
        sample_ids = None
        if self._use_train_bank and idx is not None:
            sample_ids = torch.tensor([idx], dtype=torch.long,
                                      device=self.device)

        loss_dict = self.loss_fn(
            self.model, interior, tf_pts, tf_norms, tf_tags,
            pt_pts, pt_norms, pt_dirs, bpc, u_delta, x_max, y_max,
            sample_ids=sample_ids,
        )

        # Section-resultant loss (axial force consistency)
        w_resultant = getattr(self, "_w_resultant_eff", self.config.get("w_resultant", 0.0))
        if w_resultant > 0.0:
            geometry_latent = loss_dict["geometry_latent"]
            L_resultant = self._compute_section_resultant_loss(
                self.model, geometry_latent, mesh, u_delta, x_max, y_max,
                n_slices=self.config.get("n_resultant_slices", 6),
                n_y_pts=self.config.get("n_resultant_y_pts", 24),
            )
            loss_dict["loss"] = loss_dict["loss"] + w_resultant * L_resultant
            loss_dict["L_resultant_log"] = L_resultant.item()
        else:
            loss_dict["L_resultant_log"] = 0.0


        # Geometry auxiliary loss (Run 2: latent supervision)
        w_geom_aux = self.config.get("w_geom_aux", 0.0)
        if w_geom_aux > 0.0:
            geometry_latent = loss_dict["geometry_latent"]
            geom_pred = self.model.predict_geometry(geometry_latent)  # [1, 8]

            # Build target from mesh params (normalised to ~[0,1])
            p = mesh.params
            from config import GEOMETRY_RANGES
            def _norm(val, key):
                lo, hi = GEOMETRY_RANGES[key]
                return (val - lo) / (hi - lo + 1e-12)

            holes = p.get("holes", [])
            has_hole = 1.0 if len(holes) > 0 else 0.0
            if has_hole > 0:
                cx, cy, r = holes[0]
                fi = mesh.fillet_info
                cx_n = cx / fi["L_half"]
                cy_n = cy / fi["H_grip"]
                r_n = (r - 0.6) / (1.5 - 0.6 + 1e-12)  # from HOLE_CONFIG r_range
            else:
                cx_n, cy_n, r_n = 0.0, 0.0, 0.0

            target = torch.tensor(
                [[_norm(p["L_total"], "L_total"),
                  _norm(p["W_grip"], "W_grip"),
                  _norm(p["W_gauge"], "W_gauge"),
                  _norm(p["R_fillet"], "R_fillet"),
                  has_hole, cx_n, cy_n, r_n]],
                dtype=torch.float32, device=self.device,
            )  # [1, 8]

            # MSE on first 4 (shape) + BCE on hole_present + masked MSE on cx,cy,r
            L_shape = (geom_pred[:, :4] - target[:, :4]).pow(2).mean()
            L_hole_cls = nn.functional.binary_cross_entropy_with_logits(
                geom_pred[:, 4], target[:, 4])
            if has_hole > 0:
                L_hole_reg = (geom_pred[:, 5:8] - target[:, 5:8]).pow(2).mean()
            else:
                L_hole_reg = torch.tensor(0.0, device=self.device)
            L_geom_aux = L_shape + L_hole_cls + L_hole_reg
            loss_dict["loss"] = loss_dict["loss"] + w_geom_aux * L_geom_aux
            loss_dict["L_geom_aux_log"] = L_geom_aux.item()
        else:
            loss_dict["L_geom_aux_log"] = 0.0

        return loss_dict

    # Geometry visualization

    # Section width + hole-aware interval helpers
    @staticmethod
    def _section_width(x_val: float, fillet_info: dict) -> float:
        """Compute the specimen half-height w(x) at a given x position."""
        x_g = fillet_info["x_g"]
        H_gauge = fillet_info["H_gauge"]
        if x_val <= x_g:
            return H_gauge
        xc, yc = fillet_info["arc_center"]
        R = fillet_info["R_fillet"]
        dx = x_val - xc
        arg = R**2 - dx**2
        if arg <= 0:
            return fillet_info["H_grip"]
        return yc - np.sqrt(arg)

    @staticmethod
    def _section_intervals(x_k: float, fillet_info: dict, holes: list) -> list:
        """Return solid material intervals at slice x=x_k, excluding holes.

        Args:
            x_k: x-position of the vertical slice.
            fillet_info: dict with specimen geometry info.
            holes: list of (cx, cy, r) tuples.

        Returns:
            List of (lo, hi) intervals covering only solid material.
        """
        w_k = PI_GINOT_Trainer._section_width(x_k, fillet_info)
        intervals = [(0.0, w_k)]

        for cx, cy, r in holes:
            dx = abs(x_k - cx)
            if dx >= r:
                continue
            dy = np.sqrt(r**2 - dx**2)
            a = max(0.0, cy - dy)
            b = min(w_k, cy + dy)
            new_intervals = []
            for lo, hi in intervals:
                if b <= lo or a >= hi:
                    new_intervals.append((lo, hi))
                else:
                    if lo < a:
                        new_intervals.append((lo, a))
                    if b < hi:
                        new_intervals.append((b, hi))
            intervals = new_intervals

        return intervals

    # Section-resultant loss (hole-aware, Run 2)
    def _compute_section_resultant_loss(
        self,
        model,
        geometry_latent: torch.Tensor,
        mesh,
        u_delta: torch.Tensor,
        x_max: torch.Tensor,
        y_max: torch.Tensor,
        n_slices: int = 6,
        n_y_pts: int = 32,
    ) -> torch.Tensor:
        """Compute axial force resultant N(x) at vertical slices.

        Hole-aware: integrates P11 only over solid material intervals,
        skipping the void region where a hole intersects the slice.

        Returns loss = Var(N)/mean(N)^2 + 2*adjacent_diff^2.
        """
        fi = mesh.fillet_info
        L_half = fi["L_half"]
        mu = MATERIAL_CONFIG["mu"]
        lam = MATERIAL_CONFIG["lam"]
        state = MATERIAL_CONFIG.get("state", "plane strain")
        holes = mesh.params.get("holes", [])

        if n_slices <= 6:
            xis = np.array([0.05, 0.20, 0.42, 0.65, 0.82, 0.95])[:n_slices]
        else:
            xis = np.array([0.05, 0.12, 0.20, 0.30, 0.42,
                            0.56, 0.70, 0.82, 0.90, 0.96])[:n_slices]
        x_positions = xis * L_half

        resultants = []
        for x_k in x_positions:
            intervals = self._section_intervals(x_k, fi, holes)
            if not intervals:
                continue  # fully inside a hole — skip this slice

            # Build query points across all solid intervals
            all_y = []
            all_dy = []
            for lo, hi in intervals:
                seg_len = hi - lo
                n_seg = max(2, int(n_y_pts * seg_len / fi.get("H_grip", 1.0)))
                y_edges = np.linspace(lo, hi, n_seg + 1)
                y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
                dy = seg_len / n_seg
                all_y.append(y_mid)
                all_dy.append(np.full(n_seg, dy))

            y_vals = np.concatenate(all_y)
            dy_vals = np.concatenate(all_dy)
            n_pts = len(y_vals)

            pts = np.stack([np.full(n_pts, x_k), y_vals], axis=-1)
            query = torch.tensor(pts, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            query.requires_grad_(True)

            uv, du_dx, du_dy, dv_dx, dv_dy = model.predict_with_grad_latent(
                query, geometry_latent, u_delta, x_max, y_max
            )
            P11, _, _, _, _ = first_piola_kirchhoff_stress(
                du_dx, du_dy, dv_dx, dv_dy, mu, lam, state
            )

            dy_t = torch.tensor(dy_vals, dtype=torch.float32,
                                device=self.device)
            N_k = 2.0 * (P11[0, :, 0] * dy_t).sum()
            resultants.append(N_k)

        if len(resultants) < 2:
            return torch.tensor(0.0, device=self.device)

        N_stack = torch.stack(resultants)
        N_mean = N_stack.mean()
        den = N_mean.abs() + 1e-12

        # Relative variance (existing)
        L_var = ((N_stack - N_mean) / den).pow(2).mean()
        L_adj = ((N_stack[1:] - N_stack[:-1]) / den).pow(2).mean()

        # Absolute force anchor (prevents magnitude collapse)
        # Small-strain 1D estimate: N_target = E · (u_δ/L) · (2·H_gauge)
        # This is approximate but keeps the force magnitude honest.
        E_mod = MATERIAL_CONFIG["E"]
        H_gauge = fi["H_gauge"]
        u_delta_val = u_delta.item() if isinstance(u_delta, torch.Tensor) else float(u_delta)
        N_target = E_mod * (u_delta_val / L_half) * (2.0 * H_gauge)
        L_anchor = ((N_mean - N_target) / N_target).pow(2)

        L_resultant = L_var + 2.0 * L_adj + 0.5 * L_anchor
        return L_resultant

    # Section-force diagnostic (no grad)
    @torch.no_grad()
    def run_section_force_diagnostic(self, val_idx: int = 0, n_slices: int = 8,
                                     use_hole_bank: bool = False):
        """Compute and print section-mean stress at vertical slices.

        For each val geometry, evaluates:
          - section-mean σ₁₁ (Cauchy stress)
          - section-mean E₁₁ (Green-Lagrange strain)
          - axial resultant N(x) = ∫ P₁₁ dy
          - du/dx deviation from baseline (u_delta/L_half)

        Prints a table comparing gauge vs grip sections.
        """
        bank = self._val_bank_hole if use_hole_bank else self._val_bank
        bank_label = "hole" if use_hole_bank else "no-hole"
        if not bank or val_idx >= len(bank):
            print(f"  [section diagnostic] No {bank_label} val bank — skipping")
            return

        self.model.eval()
        mesh, coll = bank[val_idx]
        fi = mesh.fillet_info
        L_half = fi["L_half"]
        mu = MATERIAL_CONFIG["mu"]
        lam = MATERIAL_CONFIG["lam"]
        u_max = LOADING_CONFIG["u_max"]

        x_m = torch.tensor([coll.x_max], dtype=torch.float32, device=self.device)
        y_m = torch.tensor([coll.y_max], dtype=torch.float32, device=self.device)
        u_d = torch.tensor([u_max], dtype=torch.float32, device=self.device)

        # Encode geometry (deterministic for val bank)
        bpc = torch.tensor(coll.boundary_pc, dtype=torch.float32,
                           device=self.device).unsqueeze(0)
        sid_offset = 200 if use_hole_bank else 100
        v_sid = torch.tensor([sid_offset + val_idx], dtype=torch.long, device=self.device)
        z = self.model.encode(bpc, x_m, y_m, sample_ids=v_sid)

        baseline_grad = u_max / L_half
        n_y = 64
        x_positions = np.linspace(0.05 * L_half, 0.95 * L_half, n_slices)

        print(f"  [Section diagnostic] val[{val_idx}]: "
              f"L_half={L_half:.1f}, W_gauge={fi['H_gauge']:.1f}, "
              f"W_grip={fi['H_grip']:.1f}")
        print(f"  {'x/L':>6s} {'w(x)':>6s} {'mean_E11':>10s} "
              f"{'E11_base':>10s} {'deviation':>10s} {'mean_P11':>10s} "
              f"{'N(x)':>10s}")
        print(f"  {'-'*70}")

        resultants = []
        for x_k in x_positions:
            w_k = self._section_width(x_k, fi)
            y_vals = np.linspace(0.01 * w_k, 0.99 * w_k, n_y)
            dy = w_k / n_y

            pts = np.stack([np.full(n_y, x_k), y_vals], axis=-1)
            query = torch.tensor(pts, dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            query.requires_grad_(True)

            with torch.enable_grad():
                uv, du_dx, du_dy, dv_dx, dv_dy = self.model.predict_with_grad_latent(
                    query, z, u_d, x_m, y_m
                )
                P11, _, _, _, detF = first_piola_kirchhoff_stress(
                    du_dx, du_dy, dv_dx, dv_dy, mu, lam,
                    MATERIAL_CONFIG.get("state", "plane strain"),
                )

            # Section means
            du_dx_mean = du_dx[0, :, 0].mean().item()
            E11_mean = du_dx_mean + 0.5 * (du_dx_mean**2)  # approx Green-Lagrange
            P11_mean = P11[0, :, 0].mean().item()  # first Piola-Kirchhoff
            N_k = 2.0 * dy * P11[0, :, 0].sum().item()  # axial resultant (quarter→half)
            deviation = (du_dx_mean - baseline_grad) / baseline_grad * 100

            resultants.append(N_k)
            region = "gauge" if x_k < fi["x_g"] else "fillet/grip"
            print(f"  {x_k/L_half:6.2f} {w_k:6.1f} {E11_mean:10.4f} "
                  f"{baseline_grad:10.4f} {deviation:+9.1f}% {P11_mean:10.2f} "
                  f"{N_k:10.2f}  ({region})")

        N_arr = np.array(resultants)
        cv = np.std(N_arr) / (np.mean(N_arr) + 1e-12)
        print(f"  {'-'*70}")
        print(f"  N(x) mean={np.mean(N_arr):.2f}, std={np.std(N_arr):.2f}, "
              f"CV={100*cv:.1f}%")
        if cv < 0.05:
            print(f"  ✓ Axial force roughly consistent (CV < 5%)")
        elif cv < 0.15:
            print(f"  ⚠ Moderate force imbalance (CV = {100*cv:.1f}%)")
        else:
            print(f"  ✗ Large force imbalance (CV = {100*cv:.1f}%) — "
                  f"model not redistributing axial stress correctly")

        self.model.train()
        return cv

    def _save_geometry_plot(self, mesh, epoch: int, geo_idx: int,
                            prefix: str = ""):
        """Save a PNG of the DogBone geometry with collocation points."""
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        plot_dogbone(mesh, show_normals=True, ax=ax)

        p = mesh.params
        ax.set_title(
            f"Epoch {epoch}, Geo {geo_idx}  |  "
            f"L={p['L_total']:.1f}  W_grip={p['W_grip']:.1f}  "
            f"W_gauge={p['W_gauge']:.1f}  R={p['R_fillet']:.1f}  "
            f"holes={len(p.get('holes', []))}"
        )

        if prefix:
            fname = f"{prefix}_{geo_idx:02d}.png"
        else:
            fname = f"epoch{epoch:04d}_geo{geo_idx:02d}.png"
        path = os.path.join(self.geo_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Displacement / stress field plots (all batch geometries)
    def _evaluate_single_geometry(self, mesh, coll, mu, lam, state,
                                  sample_ids=None):
        """Run model on one geometry and return numpy field arrays + mesh info.

        Returns dict with keys: x, y, u, delta_u, v, S11, E11, params.
        """
        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

        pts_np = coll.interior_pts
        query = _t(pts_np).requires_grad_(True)
        bpc = _t(coll.boundary_pc)
        u_delta = torch.tensor([LOADING_CONFIG["u_max"]],
                               dtype=torch.float32, device=self.device)
        x_max = torch.tensor([coll.x_max], dtype=torch.float32,
                             device=self.device)
        y_max = torch.tensor([coll.y_max], dtype=torch.float32,
                             device=self.device)

        with torch.enable_grad():
            z = self.model.encode(bpc, x_max, y_max, sample_ids=sample_ids)
            uv, du_dx, du_dy, dv_dx, dv_dy = self.model.predict_with_grad_latent(
                query, z, u_delta, x_max, y_max
            )
            S11, S22, S33, S12, detF = full_stress_state(
                du_dx, du_dy, dv_dx, dv_dy, mu, lam, state,
            )
            # Green-Lagrange strain E = 0.5*(F^T F - I)
            # E11 = 0.5*((1+du/dx)^2 + (dv/dx)^2 - 1)
            E11 = 0.5 * ((1.0 + du_dx)**2 + dv_dx**2 - 1.0)

        # Hard-BC baseline: u_base = u_delta * x / x_max
        u_np = uv[0, :, 0].detach().cpu().numpy()
        u_base = LOADING_CONFIG["u_max"] * pts_np[:, 0] / coll.x_max
        delta_u = u_np - u_base

        return {
            "x": pts_np[:, 0],
            "y": pts_np[:, 1],
            "u": u_np,
            "delta_u": delta_u,
            "v": uv[0, :, 1].detach().cpu().numpy(),
            "S11": S11[0, :, 0].detach().cpu().numpy(),
            "E11": E11[0, :, 0].detach().cpu().numpy(),
            "params": mesh.params,
        }

    def _save_field_plots(self, epoch: int, geo_ranges: dict, holes_on: bool):
        """Evaluate the model on batch_size geometries and save field plots.

        Produces a single PNG with batch_size rows × 5 columns:
            u(x,y)  |  Δu(x,y)  |  v(x,y)  |  σ₁₁  |  E₁₁

        Each row is a different geometry sampled from the current
        curriculum phase.  Uses scatter plots (same style as the
        original PINN-DogBone-FFN-KAN visualizations).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        batch_size = self.config["batch_size"]
        mu = MATERIAL_CONFIG["mu"]
        lam = MATERIAL_CONFIG["lam"]
        state = MATERIAL_CONFIG.get("state", "plane strain")

        # Use a separate RNG so field-plot sampling doesn't disturb training
        plot_rng = np.random.default_rng(epoch)

        # Sample and evaluate all geometries
        self.model.eval()
        results = []
        if self.single_geometry and self._fixed_mesh is not None:
            # Plot same geometry batch_size times (same mesh, diff collocation)
            for _ in range(batch_size):
                coll = sample_collocation_points(self._fixed_mesh, rng=plot_rng)
                results.append(self._evaluate_single_geometry(
                    self._fixed_mesh, coll, mu, lam, state))
        elif self._use_val_bank and self._val_bank:
            # Plot 2 no-hole + 2 hole val geometries for comparison
            n_nohole = min(2, len(self._val_bank))
            for i in range(n_nohole):
                mesh, coll = self._val_bank[i]
                v_sid = torch.tensor([100 + i], dtype=torch.long,
                                     device=self.device)
                results.append(self._evaluate_single_geometry(
                    mesh, coll, mu, lam, state, sample_ids=v_sid))
            if hasattr(self, "_val_bank_hole") and self._val_bank_hole:
                n_hole = min(2, len(self._val_bank_hole))
                for i in range(n_hole):
                    mesh, coll = self._val_bank_hole[i]
                    v_sid = torch.tensor([200 + i], dtype=torch.long,
                                         device=self.device)
                    results.append(self._evaluate_single_geometry(
                        mesh, coll, mu, lam, state, sample_ids=v_sid))
        else:
            for _ in range(batch_size):
                params = sample_geometry_params(
                    plot_rng, geometry_ranges=geo_ranges, holes_enabled=holes_on,
                )
                mesh = generate_dogbone(params, rng=plot_rng)
                coll = sample_collocation_points(mesh, rng=plot_rng)
                results.append(self._evaluate_single_geometry(
                    mesh, coll, mu, lam, state))
        self.model.train()

        # Layout: rows × 5 columns (2 no-hole + 2 hole when available)
        n_rows = len(results)
        n_cols = 5
        col_defs = [
            ("u",       "u [mm]",       "jet"),
            ("delta_u", "Δu [mm]",      "jet"),
            ("v",       "v [mm]",       "jet"),
            ("S11",     "σ₁₁ [MPa]",   "jet"),
            ("E11",     "E₁₁ [-]",     "jet"),
        ]
        fmt = FuncFormatter(lambda val, pos: f"{val:.4f}")

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 4.5, n_rows * 3),
            squeeze=False,
        )

        for row, res in enumerate(results):
            for col, (key, label, cmap) in enumerate(col_defs):
                ax = axes[row, col]
                vals = res[key]
                sc = ax.scatter(
                    res["x"], res["y"], c=vals,
                    cmap=cmap, s=5, edgecolors="none",
                )
                fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, format=fmt)
                ax.set_aspect("equal")
                ax.set_xlabel("X (mm)", fontsize=7)
                ax.set_ylabel("Y (mm)", fontsize=7)
                ax.tick_params(labelsize=6)

                # Column headers on the first row
                if row == 0:
                    ax.set_title(label, fontsize=10)

                # Row label on the first column
                if col == 0:
                    p = res["params"]
                    nh = len(p.get("holes", []))
                    hole_str = f"  H={nh}" if nh > 0 else ""
                    ax.set_ylabel(
                        f"L={p['L_total']:.0f}  W={p['W_gauge']:.0f}  "
                        f"R={p['R_fillet']:.0f}{hole_str}",
                        fontsize=8,
                    )

        fig.suptitle(
            f"Epoch {epoch + 1}  —  {n_rows} geometries",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        fname = f"fields_epoch{epoch + 1:04d}.png"
        out_path = os.path.join(self.fields_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Section-mean profile plots (single-geometry diagnostic)
    def _save_profile_plots(self, epoch: int):
        """Evaluate the fixed geometry and plot section-mean profiles along x.

        Produces a single PNG with 4 panels:
            Δu(x)  |  v(x)  |  σ₁₁(x)  |  E₁₁(x)

        Each curve shows the mean value across the y-section at each x,
        revealing whether the model develops fillet stress concentration.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._fixed_mesh is None:
            return

        mu = MATERIAL_CONFIG["mu"]
        lam = MATERIAL_CONFIG["lam"]
        state = MATERIAL_CONFIG.get("state", "plane strain")

        # Dense collocation for smooth profiles
        dense_rng = np.random.default_rng(999)
        coll = sample_collocation_points(self._fixed_mesh, rng=dense_rng)

        self.model.eval()
        res = self._evaluate_single_geometry(
            self._fixed_mesh, coll, mu, lam, state
        )
        self.model.train()

        x = res["x"]
        fields = {
            "delta_u": res["delta_u"],
            "v":       res["v"],
            "S11":     res["S11"],
            "E11":     res["E11"],
        }

        # Bin by x into ~40 sections
        n_bins = 40
        x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
        x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
        bin_idx = np.digitize(x, x_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        profiles = {}
        for key, vals in fields.items():
            means = np.zeros(n_bins)
            stds = np.zeros(n_bins)
            for b in range(n_bins):
                mask = bin_idx == b
                if mask.any():
                    means[b] = vals[mask].mean()
                    stds[b] = vals[mask].std()
            profiles[key] = (means, stds)

        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        plot_defs = [
            ("delta_u", "Δu [mm]",     "C0"),
            ("v",       "v [mm]",       "C1"),
            ("S11",     "σ₁₁ [MPa]",   "C2"),
            ("E11",     "E₁₁ [-]",     "C3"),
        ]

        # Mark fillet x-range
        from config import get_fillet_geometry
        fillet = get_fillet_geometry(self._fixed_mesh.params)

        for ax, (key, label, color) in zip(axes, plot_defs):
            means, stds = profiles[key]
            ax.plot(x_mid, means, color=color, linewidth=1.5)
            ax.fill_between(x_mid, means - stds, means + stds,
                            alpha=0.2, color=color)
            ax.axvspan(fillet["x_g"], fillet["L_half"], alpha=0.08,
                       color="gray", label="grip/fillet")
            ax.axvline(fillet["x_g"], ls="--", color="gray", lw=0.5)
            ax.set_xlabel("X (mm)", fontsize=8)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(f"Section-mean {label}", fontsize=10)
            ax.tick_params(labelsize=7)

        fig.suptitle(f"Epoch {epoch + 1}  —  section-mean profiles",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fname = f"profiles_epoch{epoch + 1:04d}.png"
        out_path = os.path.join(self.profiles_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Adaptive weight update
    def _update_adaptive_weights(self, batch_size, geo_ranges, holes_on, u_max):
        """Recompute adaptive loss weights on a fresh geometry sample."""
        if self.single_geometry and self._fixed_mesh is not None:
            mesh = self._fixed_mesh
            coll = sample_collocation_points(mesh, rng=self.rng)
        elif self._use_train_bank:
            idx = int(self.rng.integers(0, len(self._train_bank)))
            if self._resample_train_coll:
                params = self._train_bank[idx]
                n_bnd_seg = COLLOCATION_CONFIG.get("n_boundary_per_segment", 400)
                mesh = generate_dogbone(
                    params, n_pts_per_segment=n_bnd_seg,
                    n_interior=COLLOCATION_CONFIG["n_interior"],
                    rng=self.rng,
                )
                coll = sample_collocation_points(mesh, rng=self.rng)
            else:
                mesh, coll = self._train_bank[idx]
        else:
            params = sample_geometry_params(
                self.rng, geometry_ranges=geo_ranges, holes_enabled=holes_on
            )
            mesh = generate_dogbone(params, rng=self.rng)
            coll = sample_collocation_points(mesh, rng=self.rng)

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

        # Construct sample_ids for bank geometries
        _aw_sid = None
        if self._use_train_bank and idx is not None:
            _aw_sid = torch.tensor([idx], dtype=torch.long,
                                    device=self.device)

        self.loss_fn.update_adaptive_weights(
            self.model,
            _t(coll.interior_pts).requires_grad_(True),
            _t(coll.traction_free_pts).requires_grad_(True),
            _t(coll.traction_free_normals),
            torch.tensor(coll.traction_free_tags, dtype=torch.long,
                         device=self.device),
            _t(coll.partial_traction_pts).requires_grad_(True),
            _t(coll.partial_traction_normals),
            torch.tensor(coll.partial_traction_dirs, dtype=torch.long,
                         device=self.device),
            _t(coll.boundary_pc),
            torch.tensor([u_max], dtype=torch.float32, device=self.device),
            torch.tensor([coll.x_max], dtype=torch.float32, device=self.device),
            torch.tensor([coll.y_max], dtype=torch.float32, device=self.device),
            sample_ids=_aw_sid,
        )

    # Checkpointing
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "best_val_raw": self.best_val_raw,
            "best_gated_val_raw": self.best_gated_val_raw,
            "loss_fn_state_dict": self.loss_fn.state_dict(),
        }
        path = os.path.join(self.save_dir, "last.pt")
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.save_dir, "best.pt")
            torch.save(state, best_path)

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.best_val_raw = ckpt.get("best_val_raw", self.best_loss)
        self.best_gated_val_raw = ckpt.get("best_gated_val_raw", self.best_loss)
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {self.start_epoch}, "
              f"best_loss={self.best_loss:.4e}")


    # Boundary PC resampling sensitivity (Phase 2 diagnostic)
    @torch.no_grad()
    def run_boundary_resampling_diagnostic(self, val_idx: int = 0, n_trials: int = 8):
        """Test encoder stability under boundary point cloud resampling.

        Uses a FIXED query cloud (from the val bank) and only resamples
        the boundary PC.  This isolates encoder sensitivity from
        query-point variation.

        Target: < 2% relative spread.
        """
        if not self._val_bank or val_idx >= len(self._val_bank):
            print("  [resampling diagnostic] No val bank — skipping")
            return

        self.model.eval()
        mesh, coll_ref = self._val_bank[val_idx]

        # Fixed query cloud + domain bounds from the bank entry
        query = torch.tensor(coll_ref.interior_pts, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        x_m = torch.tensor([coll_ref.x_max], dtype=torch.float32,
                           device=self.device)
        y_m = torch.tensor([coll_ref.y_max], dtype=torch.float32,
                           device=self.device)
        u_d = torch.tensor([LOADING_CONFIG["u_max"]], dtype=torch.float32,
                           device=self.device)

        fields = []
        for k in range(n_trials):
            trial_rng = np.random.default_rng(1000 + k)
            coll_trial = sample_collocation_points(mesh, rng=trial_rng)

            # Only boundary PC changes — query cloud stays fixed
            bpc = torch.tensor(coll_trial.boundary_pc, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

            z = self.model.encode(bpc, x_m, y_m)
            uv = self.model.decode(query, z, u_d, x_m, y_m)
            fields.append(uv[0].cpu().numpy())

        ref = fields[0]
        ref_norm = max(np.linalg.norm(ref), 1e-12)
        spreads = [np.linalg.norm(f - ref) / ref_norm for f in fields[1:]]

        mean_spread = np.mean(spreads)
        max_spread = np.max(spreads)
        status = "✓" if mean_spread < 0.02 else "⚠"
        print(f"  [Resampling] val[{val_idx}]: mean={100*mean_spread:.2f}%, "
              f"max={100*max_spread:.2f}%  {status}")

        self.model.train()
        return mean_spread

    # Boundary PC permutation sensitivity (Phase 2 diagnostic)
    @torch.no_grad()
    def run_boundary_permutation_diagnostic(self, val_idx: int = 0, n_trials: int = 8):
        """Test encoder invariance under boundary point ordering.

        For the same geometry with the same boundary PC, shuffle the
        point ordering and re-encode. The predicted field should stay
        nearly identical if the encoder is permutation-stable.

        Target: < 1% relative change.
        """
        if not self._val_bank or val_idx >= len(self._val_bank):
            print("  [permutation diagnostic] No val bank — skipping")
            return

        self.model.eval()
        mesh, coll = self._val_bank[val_idx]

        bpc_np = coll.boundary_pc.copy()  # [N_bnd, 2]
        query = torch.tensor(coll.interior_pts, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        x_m = torch.tensor([coll.x_max], dtype=torch.float32,
                           device=self.device)
        y_m = torch.tensor([coll.y_max], dtype=torch.float32,
                           device=self.device)
        u_d = torch.tensor([LOADING_CONFIG["u_max"]], dtype=torch.float32,
                           device=self.device)

        # Reference: original ordering
        bpc_ref = torch.tensor(bpc_np, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        z_ref = self.model.encode(bpc_ref, x_m, y_m)
        uv_ref = self.model.decode(query, z_ref, u_d, x_m, y_m)
        ref = uv_ref[0].cpu().numpy()
        ref_norm = max(np.linalg.norm(ref), 1e-12)

        spreads = []
        for k in range(n_trials):
            perm_rng = np.random.default_rng(2000 + k)
            perm = perm_rng.permutation(bpc_np.shape[0])
            bpc_perm = torch.tensor(bpc_np[perm], dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            z_perm = self.model.encode(bpc_perm, x_m, y_m)
            uv_perm = self.model.decode(query, z_perm, u_d, x_m, y_m)
            f_perm = uv_perm[0].cpu().numpy()
            spreads.append(np.linalg.norm(f_perm - ref) / ref_norm)

        mean_spread = np.mean(spreads)
        max_spread = np.max(spreads)
        status = "✓" if mean_spread < 0.01 else "⚠"
        print(f"  [Permutation] val[{val_idx}]: mean={100*mean_spread:.2f}%, "
              f"max={100*max_spread:.2f}%  {status}")

        self.model.train()
        return mean_spread

    def _save_history(self):
        path = os.path.join(self.save_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    # Latent-swap diagnostic (Phase 1: is the encoder being used?)
    @torch.no_grad()
    def run_latent_diagnostics(self, epoch: int = -1):
        """Test whether the decoder actually uses the geometry latent.

        For each val-bank geometry i:
          1. Encode on its own boundary PC → z_i
          2. Decode on its OWN query cloud with correct z_i
          3. Decode on its OWN query cloud with each other z_j
          4. Compare geometry-sensitive fields: Δu, v, σ₁₁

        The key improvement over the naive metric: we subtract the hard-BC
        baseline u_base = u_δ·x/x_max, so the comparison is on the
        geometry-dependent correction only.

        Also computes pairwise latent distances ||z_i - z_j||.
        Saves results as a PNG and prints a summary table.
        """
        if not self._val_bank:
            print("  [latent diagnostic] No val bank — skipping")
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from physics.neo_hookean import full_stress_state

        self.model.eval()
        n_val = len(self._val_bank)
        mu = MATERIAL_CONFIG["mu"]
        lam = MATERIAL_CONFIG["lam"]
        state = MATERIAL_CONFIG.get("state", "plane strain")

        # Step 1: encode all val geometries
        latents = []   # list of [1, n_tok, dim] tensors
        colls = []
        for i, (mesh, coll) in enumerate(self._val_bank):
            bpc = torch.tensor(coll.boundary_pc, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
            x_m = torch.tensor([coll.x_max], dtype=torch.float32,
                               device=self.device)
            y_m = torch.tensor([coll.y_max], dtype=torch.float32,
                               device=self.device)
            sid = torch.tensor([100 + i], dtype=torch.long, device=self.device)
            z = self.model.encode(bpc, x_m, y_m, sample_ids=sid)
            latents.append(z)
            colls.append(coll)

        # Step 2: pairwise latent distances
        z_flat = [z.flatten().cpu().numpy() for z in latents]
        dist_matrix = np.zeros((n_val, n_val))
        for i in range(n_val):
            for j in range(n_val):
                dist_matrix[i, j] = np.linalg.norm(z_flat[i] - z_flat[j])

        print(f"\n{'='*60}")
        print(f"  LATENT-SWAP DIAGNOSTIC (epoch {epoch + 1})")
        print(f"{'='*60}")
        print(f"\n  Pairwise latent distances ||z_i - z_j||:")
        header = "       " + "".join(f"  val{j:d}  " for j in range(n_val))
        print(header)
        for i in range(n_val):
            row = f"  val{i} "
            for j in range(n_val):
                row += f"  {dist_matrix[i,j]:6.3f}"
            print(row)

        # Step 3: per-geometry decode with correct vs swapped latents
        # For each geometry i, decode on its OWN query cloud with each latent
        # Compare Δu (correction), v, and σ₁₁
        swap_du = np.zeros((n_val, n_val))   # Δu swap sensitivity
        swap_v = np.zeros((n_val, n_val))    # v swap sensitivity
        swap_s11 = np.zeros((n_val, n_val))  # σ₁₁ swap sensitivity

        def _decode_fields(coll_i, z_j):
            """Decode on geometry i's query cloud with latent z_j.
            Returns (delta_u, v, S11) as numpy arrays.
            """
            query = torch.tensor(coll_i.interior_pts, dtype=torch.float32,
                                 device=self.device).unsqueeze(0).requires_grad_(True)
            u_d = torch.tensor([LOADING_CONFIG["u_max"]], dtype=torch.float32,
                               device=self.device)
            x_m = torch.tensor([coll_i.x_max], dtype=torch.float32,
                               device=self.device)
            y_m = torch.tensor([coll_i.y_max], dtype=torch.float32,
                               device=self.device)

            with torch.enable_grad():
                uv, du_dx, du_dy, dv_dx, dv_dy = self.model.predict_with_grad_latent(
                    query, z_j, u_d, x_m, y_m
                )
                S11, _, _, _, _ = full_stress_state(
                    du_dx, du_dy, dv_dx, dv_dy, mu, lam, state,
                )

            u_np = uv[0, :, 0].detach().cpu().numpy()
            v_np = uv[0, :, 1].detach().cpu().numpy()
            s11_np = S11[0, :, 0].detach().cpu().numpy()

            # Subtract hard-BC baseline: u_base = u_δ * x / x_max
            x_pts = coll_i.interior_pts[:, 0]
            u_base = LOADING_CONFIG["u_max"] * x_pts / coll_i.x_max
            delta_u = u_np - u_base

            return delta_u, v_np, s11_np

        # Decode each geometry with each latent
        all_fields = {}  # (i, j) → (delta_u, v, S11)
        for i in range(n_val):
            for j in range(n_val):
                all_fields[(i, j)] = _decode_fields(colls[i], latents[j])

        # Compute swap sensitivity per field
        for i in range(n_val):
            du_i, v_i, s11_i = all_fields[(i, i)]
            norm_du = max(np.linalg.norm(du_i), 1e-12)
            norm_v = max(np.linalg.norm(v_i), 1e-12)
            norm_s11 = max(np.linalg.norm(s11_i), 1e-12)
            for j in range(n_val):
                du_j, v_j, s11_j = all_fields[(i, j)]
                swap_du[i, j] = np.linalg.norm(du_i - du_j) / norm_du
                swap_v[i, j] = np.linalg.norm(v_i - v_j) / norm_v
                swap_s11[i, j] = np.linalg.norm(s11_i - s11_j) / norm_s11

        # Print tables
        for name, matrix in [("Δu (correction)", swap_du),
                              ("v (transverse)", swap_v),
                              ("σ₁₁ (stress)", swap_s11)]:
            print(f"\n  Swap sensitivity on {name}:")
            print(header)
            for i in range(n_val):
                row = f"  val{i} "
                for j in range(n_val):
                    row += f"  {matrix[i,j]:6.4f}"
                print(row)

        # Summary: mean off-diagonal for each field
        mask = ~np.eye(n_val, dtype=bool)
        mean_du = np.mean(swap_du[mask])
        mean_v = np.mean(swap_v[mask])
        mean_s11 = np.mean(swap_s11[mask])

        print(f"\n  Mean off-diagonal swap sensitivity:")
        print(f"    Δu:  {mean_du:.4f}  ({100*mean_du:.1f}%)")
        print(f"    v:   {mean_v:.4f}  ({100*mean_v:.1f}%)")
        print(f"    σ₁₁: {mean_s11:.4f}  ({100*mean_s11:.1f}%)")

        best = max(mean_du, mean_v, mean_s11)
        if best < 0.01:
            print("  ⚠ WARNING: Decoder barely uses the latent! (<1% on all fields)")
        elif best < 0.05:
            print("  ~ WEAK: Decoder uses latent weakly (1-5% change)")
        elif best < 0.15:
            print("  ◐ MODERATE: Decoder uses latent moderately (5-15%)")
        else:
            print("  ✓ STRONG: Decoder uses geometry latent meaningfully (>15%)")

        # Step 4: Plot (6 panels: 2 rows × 3 cols)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        # Row 1: matrices (latent dist, Δu swap, σ₁₁ swap)
        for ax, matrix, title, cmap in [
            (axes[0, 0], dist_matrix, "Latent distance ||z_i - z_j||", "viridis"),
            (axes[0, 1], swap_du, "Δu swap sensitivity", "Reds"),
            (axes[0, 2], swap_s11, "σ₁₁ swap sensitivity", "Reds"),
        ]:
            vmin = 0 if cmap == "Reds" else None
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin)
            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(n_val))
            ax.set_yticks(range(n_val))
            ax.set_xticklabels([f"v{i}" for i in range(n_val)])
            ax.set_yticklabels([f"v{i}" for i in range(n_val)])
            for ii in range(n_val):
                for jj in range(n_val):
                    val = matrix[ii, jj]
                    fmt = f"{val:.2f}" if matrix is dist_matrix else f"{val:.3f}"
                    ax.text(jj, ii, fmt, ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)

        # Row 2: scatter comparisons for val[0] with each latent
        ref_pts = colls[0].interior_pts
        x_ref = ref_pts[:, 0]

        # Panel: Δu(x) for val[0] query with each latent
        for j in range(n_val):
            du_j, _, _ = all_fields[(0, j)]
            axes[1, 0].scatter(x_ref, du_j, s=1, alpha=0.3, label=f"z_{j}")
        axes[1, 0].set_xlabel("x [mm]")
        axes[1, 0].set_ylabel("Δu [mm]")
        axes[1, 0].set_title("Δu(x) on val[0], different latents", fontsize=10)
        axes[1, 0].legend(fontsize=7, markerscale=5)

        # Panel: v(x) for val[0] query with each latent
        for j in range(n_val):
            _, v_j, _ = all_fields[(0, j)]
            axes[1, 1].scatter(x_ref, v_j, s=1, alpha=0.3, label=f"z_{j}")
        axes[1, 1].set_xlabel("x [mm]")
        axes[1, 1].set_ylabel("v [mm]")
        axes[1, 1].set_title("v(x) on val[0], different latents", fontsize=10)
        axes[1, 1].legend(fontsize=7, markerscale=5)

        # Panel: σ₁₁(x) for val[0] query with each latent
        for j in range(n_val):
            _, _, s11_j = all_fields[(0, j)]
            axes[1, 2].scatter(x_ref, s11_j, s=1, alpha=0.3, label=f"z_{j}")
        axes[1, 2].set_xlabel("x [mm]")
        axes[1, 2].set_ylabel("σ₁₁ [MPa]")
        axes[1, 2].set_title("σ₁₁(x) on val[0], different latents", fontsize=10)
        axes[1, 2].legend(fontsize=7, markerscale=5)

        fig.suptitle(f"Latent Diagnostic — Epoch {epoch + 1}", fontsize=13,
                     fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fname = f"latent_diag_epoch{epoch + 1:04d}.png"
        out_path = os.path.join(self.save_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

        self.model.train()
