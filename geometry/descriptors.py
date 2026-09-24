#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimensionless shape descriptors — the geometry an encoder can actually see.

``PI_GINOT.encode`` normalises the boundary point cloud to [-1, 1], by x_max in
x and y_max in y, before the encoder sees anything.  Absolute millimetres are
gone at that point, so a latent cannot carry ``L_total`` in mm and asking it to
is asking for the impossible.  The decoder recovers absolute sizes by combining
the shape with the ``x_max`` / ``y_max`` scalars it receives separately.

These are the quantities that survive the normalisation:

    Wg/Wgrip   the taper ratio -- the encoder already carries this well
               (probe R^2 0.974)
    R/L        fillet radius against specimen length
    R/Wgrip    fillet radius against grip width
    L/Wgrip    overall aspect

Two places must agree on this list or the project measures one thing and
trains another: ``verification/encoder_probe.py``, which asks what the latent
contains, and the geometry auxiliary loss, which supervises what it should
contain.  Hence one definition, here.

The historical note is worth keeping.  The original architecture shipped a
``geom_aux_head`` predicting eight values, the first four of which were the
*absolute* parameters normalised to their sampling ranges.  It was never
enabled -- ``w_geom_aux`` had no entry in the config, so the trainer's
``config.get("w_geom_aux", 0.0)`` was always zero -- and it was removed in
commit 2b0a79d as dead code.  Had it been switched on it would have been
training the encoder toward targets its input cannot determine.
"""

from __future__ import annotations

import numpy as np

from config import TRAINING_CONFIG

RATIO_NAMES = ("Wg_over_Wgrip", "R_over_L", "R_over_Wgrip", "L_over_Wgrip")


def ratios_from_params(params: dict) -> np.ndarray:
    """The four dimensionless descriptors of one geometry."""
    L = float(params["L_total"])
    Wgrip = float(params["W_grip"])
    Wg = float(params["W_gauge"])
    R = float(params["R_fillet"])
    return np.array([Wg / Wgrip, R / L, R / Wgrip, L / Wgrip], dtype=np.float64)


def _moments(n: int = 20000, seed: int = 0):
    """Mean and spread of each ratio over the sampling ranges.

    Computed by sampling rather than hard-coded so that changing
    ``bank_geo_ranges`` cannot silently leave the standardisation behind.
    """
    rng = np.random.default_rng(seed)
    r = TRAINING_CONFIG["bank_geo_ranges"]
    draw = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in r.items()}
    vals = np.stack([
        draw["W_gauge"] / draw["W_grip"],
        draw["R_fillet"] / draw["L_total"],
        draw["R_fillet"] / draw["W_grip"],
        draw["L_total"] / draw["W_grip"],
    ], axis=1)
    return vals.mean(axis=0), vals.std(axis=0)


RATIO_MEAN, RATIO_STD = _moments()


def standardised_ratios(params: dict) -> np.ndarray:
    """Ratios z-scored over the sampling ranges.

    Without this the four targets differ by an order of magnitude in scale --
    ``R/L`` runs about 0.2 and ``L/Wgrip`` about 2.7 -- so a plain MSE would
    spend almost all of its gradient on the aspect ratio, which the decoder
    already receives directly as x_max and y_max.
    """
    return (ratios_from_params(params) - RATIO_MEAN) / RATIO_STD
