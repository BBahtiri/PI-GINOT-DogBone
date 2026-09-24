#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run manifest: everything needed to say what produced a checkpoint.

The repository currently records no hardware, no library versions, no git
state and no config snapshot alongside its checkpoints, so there is no way to
say which configuration produced ``checkpoints/best.pt``.  This module writes
a JSON manifest next to every run and stamps a short config hash into the
checkpoints themselves, which closes that gap for anything trained from now
on.

The config hash covers only the *scientific* configuration -- the encoder,
decoder, material, geometry, collocation and training dicts -- so two runs
that differ merely in output directory or print frequency still compare equal.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Optional

import numpy as np
import torch

MANIFEST_NAME = "run_manifest.json"

# Config dicts that define the experiment.  Anything not listed here is
# considered cosmetic and does not change the hash.
SCIENTIFIC_CONFIG_KEYS = (
    "ENCODER_CONFIG", "DECODER_CONFIG", "MATERIAL_CONFIG", "NONDIM_SCALES",
    "GEOMETRY_DEFAULT", "GEOMETRY_RANGES", "COLLOCATION_CONFIG",
    "NORMALIZATION_CONFIG", "TRAINING_CONFIG", "LOADING_CONFIG", "RANDOM_SEED",
)

# Keys inside TRAINING_CONFIG that do not affect the science.
_COSMETIC_TRAINING_KEYS = frozenset({
    "print_every", "plot_every", "save_geo", "checkpoint_dir", "num_workers",
    "save_best_only",
})


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(("git",) + args, capture_output=True, text=True,
                             timeout=10,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__))))
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict:
    """Commit, branch, dirty flag and the diffstat of uncommitted work."""
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
        "describe": _git("describe", "--always", "--dirty"),
    }


def environment() -> dict:
    """Interpreter, libraries, and the accelerator actually present."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        # Recorded because it is a run-level choice that changes results and
        # is not visible anywhere in the config dicts: everything downstream
        # builds its tensors at torch.get_default_dtype().
        "default_dtype": str(torch.get_default_dtype()).replace("torch.", ""),
    }
    try:
        import scipy
        env["scipy"] = scipy.__version__
    except ImportError:
        env["scipy"] = None
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        env["gpu_total_memory_gb"] = round(props.total_memory / 1e9, 2)
        env["gpu_capability"] = f"{props.major}.{props.minor}"
    return env


def collect_config(config_module) -> dict:
    """Snapshot the scientific config dicts from a module (usually ``config``)."""
    snap = {}
    for key in SCIENTIFIC_CONFIG_KEYS:
        if hasattr(config_module, key):
            snap[key] = getattr(config_module, key)
    return snap


def config_hash(snapshot: dict) -> str:
    """Stable sha256 over the scientific config, ignoring cosmetic keys."""
    trimmed = dict(snapshot)
    train = trimmed.get("TRAINING_CONFIG")
    if isinstance(train, dict):
        trimmed["TRAINING_CONFIG"] = {
            k: v for k, v in train.items() if k not in _COSMETIC_TRAINING_KEYS
        }
    blob = json.dumps(trimmed, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def build_manifest(config_module, extra: Optional[dict] = None) -> dict:
    """Assemble the manifest without writing it (used by tests)."""
    snapshot = collect_config(config_module)
    manifest = {
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "git": git_state(),
        "environment": environment(),
        "config_sha256": config_hash(snapshot),
        "config": snapshot,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(save_dir: str, config_module, extra: Optional[dict] = None,
                   filename: str = MANIFEST_NAME) -> dict:
    """Write the manifest into ``save_dir`` and return it.

    Args:
        save_dir: run directory (usually the checkpoint directory).
        config_module: the imported ``config`` module.
        extra: anything run-specific worth recording (device, CLI args, ...).

    Returns:
        The manifest dict, whose ``config_sha256`` should be stamped into
        every checkpoint the run writes.
    """
    manifest = build_manifest(config_module, extra=extra)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    return manifest


def set_precision(name: str) -> torch.dtype:
    """Set the working precision for everything built after this call.

    The training and evaluation paths construct their tensors at
    ``torch.get_default_dtype()`` rather than naming a dtype, so this one call
    moves the model parameters, the collocation points, the geometry inputs
    and the physics arithmetic together.  Calling it after a model exists does
    not convert that model, so it belongs at the top of a run.

    What float64 can and cannot be expected to do here is worth stating,
    because the literature's large effect sizes come from a mechanism this
    project does not have.  Xu et al. (arXiv 2505.10949) report 34-117x from
    the dtype alone, but those runs are L-BFGS, whose default
    ``tolerance_change`` of 1e-7 sits *below* float32 machine epsilon, so the
    inner loop stops on arithmetic rather than on convergence.  This trainer
    is Adam and has no such stopping rule, so that mechanism is absent and
    the effect here has to come from somewhere else -- the plane-stress
    Newton closure, which converges to ~1e-7 instead of ~1e-16 in single
    precision, or the gradient signal late in training falling below the
    float32 noise floor.  Both are plausible and neither is established;
    the flag exists so it can be measured rather than assumed.
    """
    dt = {"float32": torch.float32, "float64": torch.float64,
          "fp32": torch.float32, "fp64": torch.float64}.get(str(name).lower())
    if dt is None:
        raise ValueError(f"unknown precision {name!r}; "
                         "expected float32 or float64")
    torch.set_default_dtype(dt)
    return dt


def set_deterministic(seed: int, warn_only: bool = True) -> None:
    """Seed every RNG in play and ask torch for deterministic kernels.

    ``warn_only=True`` because some scatter/gather kernels used by
    ``models/modules/pointnet2_utils.py`` have no deterministic
    implementation; they warn instead of raising, so the rest of the run is
    still reproducible and the offenders are named in the log.

    CUBLAS_WORKSPACE_CONFIG has to be set before the CUDA context is created,
    so call this before any tensor touches the GPU.
    """
    import random

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
