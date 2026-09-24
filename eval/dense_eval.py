#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frozen dense physics-evaluation set.

Why this exists
---------------
The only physics numbers the training loop produces are losses on the *same*
collocation points the optimiser is fitting, resampled every epoch.  That
measures how well the model fits the points it was handed, not how well it
satisfies the PDE.  Reporting it as a physics metric would be reporting
training loss.

This module builds a set that is generated once from a locked seed, written to
disk, and never resampled: a Sobol-sequence interior at roughly ten times the
training density plus a dense deterministic boundary, per geometry.  All
reported ``||Div P||`` and ``||P.N||`` figures come from here.

Sobol rather than uniform random: the residual field is smooth and strongly
concentrated at the fillet, and a low-discrepancy set has far lower variance
for the same point count, so the metric does not wobble between rebuilds.

Usage
-----
    from eval.dense_eval import build_dense_eval_set, load_dense_eval_set
    build_dense_eval_set(params_list)          # once, committed as an artefact
    geoms = load_dense_eval_set()
    report = evaluate_dense(model, geoms, device="cuda")
    print(format_dense_report(report))
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from config import (COLLOCATION_CONFIG, LOADING_CONFIG, MATERIAL_CONFIG,
                    NONDIM_SCALES, get_fillet_geometry, validate_geometry)
from geometry.collocation import sample_collocation_points
from geometry.parametric_dogbone import (_point_in_dogbone, generate_dogbone)
from physics.equilibrium import equilibrium_residual, traction, boundary_piola

# Locked -- changing this invalidates every number computed from the set.
DENSE_EVAL_SEED = 31337
DEFAULT_PATH = os.path.join("eval", "dense_eval.npz")

# ~10x the training interior density (COLLOCATION_CONFIG["n_interior"] = 4000).
DEFAULT_N_INTERIOR = 40_000
DEFAULT_N_BOUNDARY_PER_SEGMENT = 2_000

# Segment tags, matching CollocationData.traction_free_tags.
TRACTION_FREE_SEGMENTS = ("gauge_top", "right_arc")
PARTIAL_TRACTION_DIRS = {"bottom": 0, "left_symmetry": 1, "right_grip": 1}


@dataclass
class DenseEvalGeometry:
    """One frozen geometry: dense interior, dense boundary, fixed encoder PC."""
    params: dict
    fillet_info: dict
    interior: np.ndarray                      # (N_int, 2)
    boundary_pts: Dict[str, np.ndarray]       # per segment, (N_seg, 2)
    boundary_normals: Dict[str, np.ndarray]   # per segment, (N_seg, 2)
    boundary_pc: np.ndarray                   # (N_pc, 2) encoder input
    x_max: float
    y_max: float


# ------------------------------------------------------------------ build --

def _sobol_interior(fillet: dict, n: int, seed: int) -> np.ndarray:
    """Sobol points inside the quarter dog-bone, by rejection.

    scipy's Sobol engine only keeps its balance properties when the total
    number of drawn points is a power of two, so a short draw is retried from
    a *fresh* engine with a larger exponent rather than topped up -- the
    result is still a single balanced 2**m block, and still deterministic in
    ``seed``.  The acceptance rate is the domain's share of its bounding box,
    which is at worst about 0.23 over the parameter ranges in use.
    """
    from scipy.stats import qmc

    L_half = fillet["L_half"]
    H_grip = fillet["H_grip"]

    m = max(10, int(np.ceil(np.log2(n / 0.15))))
    for _ in range(6):
        pts = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(m)
        x = pts[:, 0] * L_half
        y = pts[:, 1] * H_grip
        keep = _point_in_dogbone(x, y, fillet)
        if int(keep.sum()) >= n:
            good = np.stack([x[keep], y[keep]], axis=-1)
            return good[:n].astype(np.float32)
        m += 1
    raise RuntimeError(
        f"_sobol_interior: could not collect {n} interior points for "
        f"{fillet}; acceptance rate is unexpectedly low."
    )


def build_dense_eval_set(
    params_list: List[dict],
    n_interior: int = DEFAULT_N_INTERIOR,
    n_boundary_per_segment: int = DEFAULT_N_BOUNDARY_PER_SEGMENT,
    n_boundary_pc: Optional[int] = None,
    seed: int = DENSE_EVAL_SEED,
    path: str = DEFAULT_PATH,
) -> List[DenseEvalGeometry]:
    """Generate the frozen evaluation set and write it to ``path``.

    Args:
        params_list: geometry parameter dicts (e.g. the validation bank).
        n_interior: interior points per geometry.
        n_boundary_per_segment: boundary points per named segment.
        n_boundary_pc: encoder point-cloud size; defaults to the training value
            so the encoder sees the same kind of input it was trained on.
        seed: locked master seed; each geometry gets seed + index.
        path: npz destination.  Pass None to skip writing.

    Returns:
        The list of DenseEvalGeometry, in the order given.
    """
    if n_boundary_pc is None:
        n_boundary_pc = COLLOCATION_CONFIG["n_boundary_pc"]

    geoms: List[DenseEvalGeometry] = []
    for i, params in enumerate(params_list):
        params = {**params, "holes": params.get("holes", [])}
        if not validate_geometry(params):
            raise ValueError(f"dense eval geometry {i} is invalid: {params}")

        fi = get_fillet_geometry(params)
        mesh = generate_dogbone(params, n_pts_per_segment=n_boundary_per_segment,
                                rng=np.random.default_rng(seed + i))
        coll = sample_collocation_points(
            mesh, n_interior=1, n_total_boundary=n_boundary_pc,
            n_boundary_pc=n_boundary_pc,
            rng=np.random.default_rng(seed + 10_000 + i),
        )

        geoms.append(DenseEvalGeometry(
            params=params,
            fillet_info=fi,
            interior=_sobol_interior(fi, n_interior, seed + 20_000 + i),
            boundary_pts={s.name: s.points.astype(np.float32)
                          for s in mesh.boundary_segments},
            boundary_normals={s.name: s.normals.astype(np.float32)
                              for s in mesh.boundary_segments},
            boundary_pc=coll.boundary_pc.astype(np.float32),
            x_max=float(coll.x_max),
            y_max=float(coll.y_max),
        ))

    if path:
        save_dense_eval_set(geoms, path, seed=seed)
    return geoms


def save_dense_eval_set(geoms: List[DenseEvalGeometry], path: str,
                        seed: int = DENSE_EVAL_SEED) -> None:
    """Write the set to a single npz, flattening the per-segment dicts."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob: Dict[str, np.ndarray] = {
        "seed": np.array(seed),
        "n_geo": np.array(len(geoms)),
    }
    for i, g in enumerate(geoms):
        blob[f"g{i}/params"] = np.array(json.dumps(g.params))
        blob[f"g{i}/interior"] = g.interior
        blob[f"g{i}/boundary_pc"] = g.boundary_pc
        blob[f"g{i}/bounds"] = np.array([g.x_max, g.y_max])
        blob[f"g{i}/segments"] = np.array(list(g.boundary_pts.keys()))
        for name, pts in g.boundary_pts.items():
            blob[f"g{i}/seg/{name}/pts"] = pts
            blob[f"g{i}/seg/{name}/normals"] = g.boundary_normals[name]
    # Stored as float32: the model runs in float32, so float64 coordinates
    # would only double the artefact size.
    for k, v in list(blob.items()):
        if isinstance(v, np.ndarray) and v.dtype == np.float64:
            blob[k] = v.astype(np.float32)
    np.savez_compressed(path, **blob)


def load_dense_eval_set(path: str = DEFAULT_PATH) -> List[DenseEvalGeometry]:
    """Read a frozen set back.  Raises if it has not been built yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run build_dense_eval_set() once and commit "
            "the artefact so every reported number refers to the same points."
        )
    z = np.load(path, allow_pickle=False)
    geoms = []
    for i in range(int(z["n_geo"])):
        params = json.loads(str(z[f"g{i}/params"]))
        segments = [str(s) for s in z[f"g{i}/segments"]]
        x_max, y_max = z[f"g{i}/bounds"]
        geoms.append(DenseEvalGeometry(
            params=params,
            fillet_info=get_fillet_geometry(params),
            interior=z[f"g{i}/interior"],
            boundary_pts={n: z[f"g{i}/seg/{n}/pts"] for n in segments},
            boundary_normals={n: z[f"g{i}/seg/{n}/normals"] for n in segments},
            boundary_pc=z[f"g{i}/boundary_pc"],
            x_max=float(x_max), y_max=float(y_max),
        ))
    return geoms


# --------------------------------------------------------------- evaluate --

def _chunks(n: int, size: int):
    for lo in range(0, n, size):
        yield lo, min(lo + size, n)


def evaluate_dense(
    model,
    geoms: List[DenseEvalGeometry],
    device: str = "cpu",
    u_delta: Optional[float] = None,
    chunk: int = 8192,
    sample_id_offset: int = 500,
) -> List[dict]:
    """Physics residuals on the frozen set, per geometry.

    Metrics are nondimensional so they compare across geometries:
        eq_l2    : RMS ||Div P|| * L0 / S0
        eq_linf  : max ||Div P|| * L0 / S0
        trac_*   : RMS ||P.N|| / S0 on each traction-free segment
        part_*   : RMS of the enforced traction component / S0 on the
                   symmetry and grip faces

    ``create_graph`` is off throughout: the residual needs first derivatives
    of P only, and building the second-order graph here would waste memory for
    a quantity nobody backpropagates.
    """
    mu, lam = MATERIAL_CONFIG["mu"], MATERIAL_CONFIG["lam"]
    state = MATERIAL_CONFIG["state"]
    L0, S0 = NONDIM_SCALES["L0"], NONDIM_SCALES["S0"]
    u_delta = LOADING_CONFIG["u_max"] if u_delta is None else u_delta
    dev = torch.device(device)

    was_training = model.training
    model.eval()
    results = []
    try:
        for gi, g in enumerate(geoms):
            x_m = torch.tensor([g.x_max], dtype=torch.get_default_dtype(), device=dev)
            y_m = torch.tensor([g.y_max], dtype=torch.get_default_dtype(), device=dev)
            u_d = torch.tensor([u_delta], dtype=torch.get_default_dtype(), device=dev)
            sid = torch.tensor([sample_id_offset + gi], dtype=torch.long,
                               device=dev)
            bpc = torch.tensor(g.boundary_pc, dtype=torch.get_default_dtype(),
                               device=dev).unsqueeze(0)
            with torch.no_grad():
                z = model.encode(bpc, x_m, y_m, sample_ids=sid)

            # --- interior: ||Div P|| ---
            sq_sum, max_sq, n_int = 0.0, 0.0, 0
            for lo, hi in _chunks(len(g.interior), chunk):
                pts = torch.tensor(g.interior[lo:hi], dtype=torch.get_default_dtype(),
                                   device=dev).unsqueeze(0).requires_grad_(True)
                f_x, f_y, _ = equilibrium_residual(
                    model, pts, z, u_d, x_m, mu, lam, y_m, state,
                    create_graph=False)
                r2 = ((L0 / S0) ** 2) * (f_x ** 2 + f_y ** 2)
                r2 = r2.detach()
                sq_sum += float(r2.sum())
                max_sq = max(max_sq, float(r2.max()))
                n_int += r2.numel()

            res = {
                "params": g.params,
                "n_interior": n_int,
                "eq_l2": float(np.sqrt(sq_sum / max(n_int, 1))),
                "eq_linf": float(np.sqrt(max_sq)),
            }

            # --- boundary: ||P.N|| per segment ---
            for name, pts_np in g.boundary_pts.items():
                nrm_np = g.boundary_normals[name]
                sq, cnt = 0.0, 0
                for lo, hi in _chunks(len(pts_np), chunk):
                    pts = torch.tensor(pts_np[lo:hi], dtype=torch.get_default_dtype(),
                                       device=dev).unsqueeze(0).requires_grad_(True)
                    nrm = torch.tensor(nrm_np[lo:hi], dtype=torch.get_default_dtype(),
                                       device=dev).unsqueeze(0)
                    P11, P12, P21, P22, _ = boundary_piola(
                        model, pts, z, u_d, x_m, mu, lam, y_m, state)
                    tx, ty = traction(P11, P12, P21, P22, nrm)
                    if name in TRACTION_FREE_SEGMENTS:
                        t2 = (tx ** 2 + ty ** 2) / (S0 ** 2)
                    else:
                        comp = tx if PARTIAL_TRACTION_DIRS[name] == 0 else ty
                        t2 = (comp ** 2) / (S0 ** 2)
                    t2 = t2.detach()
                    sq += float(t2.sum())
                    cnt += t2.numel()
                key = ("trac_" if name in TRACTION_FREE_SEGMENTS else "part_")
                res[key + name] = float(np.sqrt(sq / max(cnt, 1)))
                res[f"n_{name}"] = cnt

            results.append(res)
    finally:
        if was_training:
            model.train()
    return results


def format_dense_report(results: List[dict]) -> str:
    """One line per geometry plus a mean row, for logs and the paper table."""
    keys = [k for k in results[0]
            if k.startswith(("eq_", "trac_", "part_"))]
    head = f"{'geo':>4} " + " ".join(f"{k:>16}" for k in keys)
    lines = [head, "-" * len(head)]
    for i, r in enumerate(results):
        lines.append(f"{i:>4} " + " ".join(f"{r[k]:>16.4e}" for k in keys))
    lines.append("-" * len(head))
    lines.append(f"{'mean':>4} " + " ".join(
        f"{np.mean([r[k] for r in results]):>16.4e}" for k in keys))
    return "\n".join(lines)


# -------------------------------------------------------------------- CLI --

def _default_params_list():
    """The validation bank, taken from the same builder the trainer uses."""
    from config import TRAINING_CONFIG
    from geometry.banks import val_bank_params

    return val_bank_params(TRAINING_CONFIG["bank_val_size"],
                           TRAINING_CONFIG["bank_geo_ranges"])


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true",
                    help="generate the frozen set and write it to --path")
    ap.add_argument("--checkpoint", default=None,
                    help="evaluate this checkpoint on the frozen set")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--n-interior", type=int, default=DEFAULT_N_INTERIOR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    if args.build:
        geoms = build_dense_eval_set(_default_params_list(),
                                     n_interior=args.n_interior,
                                     path=args.path)
        size_mb = os.path.getsize(args.path) / 1e6
        print(f"Wrote {len(geoms)} geometries to {args.path} "
              f"({size_mb:.1f} MB, seed {DENSE_EVAL_SEED})")
        print(f"  interior points per geometry: {len(geoms[0].interior):,}")

    if args.checkpoint:
        from config import DECODER_CONFIG, ENCODER_CONFIG
        from models.pi_ginot import PI_GINOT

        geoms = load_dense_eval_set(args.path)
        model = PI_GINOT(ENCODER_CONFIG, DECODER_CONFIG).to(args.device)
        ckpt = torch.load(args.checkpoint, map_location=args.device,
                          weights_only=False)
        state, dropped = PI_GINOT.strip_legacy_keys(ckpt["model_state_dict"])
        if dropped:
            print(f"  dropped {len(dropped)} legacy checkpoint key(s)")
        model.load_state_dict(state)
        print(format_dense_report(evaluate_dense(model, geoms,
                                                 device=args.device)))


if __name__ == "__main__":
    main()
