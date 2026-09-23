#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-family operators (Phase 8, Tier 1): the trainer loop, generalised.

``training/trainer.py`` is wired to the dog-bone throughout.  This module
reproduces its training mechanics exactly and takes the geometry from a
``geometry.families.Family`` instead:

  * Adam over every model parameter, lr 3e-4, grad-norm clip 1.0;
  * one optimiser step per "epoch" on a batch of 4 geometries drawn without
    replacement from a fixed 64-geometry bank (bank order from
    ``default_rng(42)``, independent of the run seed, as in the trainer);
  * ReduceLROnPlateau (factor 0.7, patience 10) stepped every 25 steps on an
    EMA (alpha 0.1) of the energy on two validation geometries -- the trainer
    validates at the start of every 25-epoch ``fit`` chunk;
  * the energy loss with stratified resampling on a ~1400-triangle
    quadrature mesh, barrier 1e3 at J 0.05, E_scale = E, quadrature RNG 0;
  * the model conditioned on the family's exact parameters (the Phase 6.7
    "oracle" arm), its decoder's hard Dirichlet layer replaced by the
    family's distance-function spec (geometry/adf.py) -- for the dog-bone the
    shipped layer is kept, and the spec is its exact special case anyway.

``--family dogbone`` runs the dog-bone through this loop, as the check that
the loop is equivalent to the trainer's (Phase 8 Tier 1, P-T4).

    python -m training.family_trainer --family open_hole --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from config import (DECODER_CONFIG, ENCODER_CONFIG, LOADING_CONFIG,
                    MATERIAL_CONFIG, NONDIM_SCALES, TRAINING_CONFIG)
from geometry.families import FAMILIES
from geometry.shapes import build_shape_mesh
from physics.weak_form import EnergyLoss
from training.manifest import set_deterministic, set_precision

MU, LAM, STATE = (MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"],
                  MATERIAL_CONFIG["state"])
UD = LOADING_CONFIG["u_max"]
N_ELEM_Q = TRAINING_CONFIG.get("energy_n_elem", 1400)
N_Q_FEATURE = 8          # quadrature elements per feature radius at the root
GRADE_Q = 0.25


class FamilyEnergyLoss(EnergyLoss):
    """The energy loss with the family's own quadrature mesh."""

    def __init__(self, family, **kw):
        super().__init__(**kw)
        self.family = family
        self._meshes = {}

    def _mesh(self, params):
        key = tuple(round(float(params[k]), 12) for k in self.family.keys)
        m = self._meshes.get(key)
        if m is None:
            sh = self.family.shape(params)
            L, Hs = self.family.extent(params)
            h = np.sqrt(L * Hs / (0.433 * 0.8 * self.n_elem_target))
            for _ in range(4):                      # a few secant steps on h
                m = build_shape_mesh(sh, h=h, refine=max(1.0, h * N_Q_FEATURE
                                                          / sh.feature_length),
                                     grade=GRADE_Q, n_smooth=6)
                h *= np.sqrt(m.n_elem / self.n_elem_target)
                if abs(m.n_elem / self.n_elem_target - 1.0) < 0.15:
                    break
            self._meshes[key] = m
        return m


def family_model(fam_name: str):
    from verification.supervised_ceiling import OracleConditioned
    if fam_name == "dogbone":
        return OracleConditioned(ENCODER_CONFIG, DECODER_CONFIG,
                                 TRAINING_CONFIG["bank_geo_ranges"])
    fam = FAMILIES[fam_name]
    model = OracleConditioned(ENCODER_CONFIG, DECODER_CONFIG, fam.box,
                              keys=fam.keys)
    cache = {}

    def apply_bc(raw, query_pts, u_delta, x_max, y_max):
        p = model._geo
        key = tuple(round(float(p[k]), 12) for k in fam.keys)
        spec = cache.get(key)
        if spec is None:
            spec = cache[key] = fam.spec(p, UD)
        uv = spec.apply(query_pts[0], raw[0])
        return uv.unsqueeze(0)

    model.decoder._apply_hard_bc = apply_bc
    return model


def _sets(fam_name):
    if fam_name == "dogbone":
        from geometry.banks import train_bank_params, val_bank_params
        from config import get_fillet_geometry
        ext = (lambda p: (get_fillet_geometry(p)["L_half"],
                          get_fillet_geometry(p)["H_grip"]))
        return train_bank_params(64), val_bank_params(2), ext
    fam = FAMILIES[fam_name]
    return fam.bank(64), fam.val(2), fam.extent


def build_loss(fam_name):
    kw = dict(mu=MU, lam=LAM, stress_state=STATE, n_elem_target=N_ELEM_Q,
              resample=True, n_per_elem=TRAINING_CONFIG.get("energy_n_per_elem", 3),
              w_barrier=TRAINING_CONFIG["w_barrier"],
              j_min=TRAINING_CONFIG["barrier_delta"],
              E_scale=NONDIM_SCALES["S0"], S0=NONDIM_SCALES["S0"])
    if fam_name == "dogbone":
        return EnergyLoss(**kw)
    return FamilyEnergyLoss(FAMILIES[fam_name], **kw)


def train(fam_name, seed, out_dir, steps=1600, batch=4, lr=3e-4, chunk=25,
          threads=1):
    torch.set_num_threads(threads)
    set_precision("float32")
    set_deterministic(seed)
    model = family_model(fam_name)
    loss_fn = build_loss(fam_name)
    bank, val, extent = _sets(fam_name)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=TRAINING_CONFIG["scheduler_factor"],
        patience=TRAINING_CONFIG["scheduler_patience"])
    clip = TRAINING_CONFIG["grad_clip_norm"]
    rng = np.random.default_rng(42)
    dt = torch.get_default_dtype()
    u_d = torch.tensor([UD], dtype=dt)
    bpc = torch.zeros(1, 1, 2, dtype=dt)          # ignored by the oracle model
    os.makedirs(out_dir, exist_ok=True)
    order, ema, hist = [], None, []
    t0 = time.time()

    def geo_loss(p):
        L, Hs = extent(p)
        return loss_fn(model, p, bpc, u_d, torch.tensor([L], dtype=dt),
                       torch.tensor([Hs], dtype=dt))

    for step in range(steps):
        if step % chunk == 0:
            model.eval()
            v = float(np.mean([(lambda o: o["L_energy_log"] + o["L_barrier_log"])(geo_loss(p))
                               for p in val]))
            ema = v if ema is None else (1 - 0.1) * ema + 0.1 * v
            sched.step(ema)
        model.train()
        opt.zero_grad()
        if len(order) < batch:
            order = list(range(len(bank)))
            rng.shuffle(order)
        idx = [order.pop(0) for _ in range(batch)]
        tot, e_sum, jmin = 0.0, 0.0, np.inf
        for i in idx:
            o = geo_loss(bank[i])
            (o["loss"] / batch).backward()
            tot += float(o["loss"].detach()) / batch
            e_sum += o["L_energy_log"] / batch
            jmin = min(jmin, o["min_J3D_log"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if (step + 1) % chunk == 0 or step == 0:
            rec = {"step": step + 1, "loss": tot, "L_energy": e_sum, "min_J3D": jmin,
                   "val_ema": ema, "lr": opt.param_groups[0]["lr"],
                   "wall_min": (time.time() - t0) / 60.0}
            hist.append(rec)
            print(f"  [{fam_name} s{seed}] step {step + 1:5d}  loss {tot:.6e}  "
                  f"J3D_min {jmin:.3f}  lr {rec['lr']:.1e}  "
                  f"{rec['wall_min']:.1f} min", flush=True)
        if (step + 1) % 100 == 0 or step + 1 == steps:
            torch.save({"model_state_dict": model.state_dict(), "step": step + 1,
                        "family": fam_name, "seed": seed},
                       os.path.join(out_dir, "last.pt"))
            json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"))
    wall = time.time() - t0
    json.dump({"family": fam_name, "seed": seed, "steps": steps,
               "wall_min": wall / 60.0, "final_loss": hist[-1]["loss"],
               "s_per_step": wall / steps},
              open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    print(f"  [{fam_name} s{seed}] done: {steps} steps in {wall / 60:.1f} min "
          f"({wall / steps:.2f} s/step)", flush=True)
    return model


def load(fam_name, path):
    model = family_model(fam_name)
    st = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(st, strict=True)
    return model.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["dogbone"] + list(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"verification/results/phase8/tier1/{a.family}_s{a.seed}"
    train(a.family, a.seed, out, steps=a.steps, threads=a.threads)
