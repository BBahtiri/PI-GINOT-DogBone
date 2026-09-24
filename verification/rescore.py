#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-score a finished checkpoint in the stress-concentration norms.

Every comparison made up to Phase 5 was scored in relative L2.  That norm is
an area-weighted average over the whole specimen, and the specimen is mostly
prismatic gauge where the state is nearly uniaxial -- so it is dominated by
the easy part of the field and is close to blind to the fillet peak, which is
the quantity the paper's claim is about.  Phase 6.1 added the right norms to
``eval.compare_reference`` (signed peak error, fillet L-infinity, and the
stress-concentration factor against the reference) and surfaced them in the
study harnesses.

This module applies them to checkpoints that already exist, so the record can
be corrected without spending a single training epoch.  It trains nothing and
touches no weights; it loads, evaluates, and prints.

Run with:
    python -m verification.rescore \
        --checkpoint verification/results/fixed_long/oracle-energy/last.pt \
        --conditioning oracle --tag oracle-energy
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from config import DECODER_CONFIG, ENCODER_CONFIG, TRAINING_CONFIG
from models.pi_ginot import PI_GINOT
from training.manifest import set_precision
from verification.operator_study import (REPORT_METRICS, evaluate,
                                         mean_metric)


def load(path: str, conditioning: str, device: str, enc_over: dict = None):
    enc = dict(ENCODER_CONFIG)
    if enc_over:
        enc.update(enc_over)
    if conditioning == "oracle":
        from verification.supervised_ceiling import OracleConditioned
        model = OracleConditioned(enc, DECODER_CONFIG,
                                  TRAINING_CONFIG["bank_geo_ranges"])
    else:
        model = PI_GINOT(enc, DECODER_CONFIG)
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model_state_dict"]
    state, dropped = PI_GINOT.strip_legacy_keys(state)
    # A checkpoint trained with the geometry auxiliary head carries its
    # weights; the head takes no part in the forward pass that produces the
    # fields, so loading without it is a faithful restore rather than a
    # silent truncation.  Anything else missing is a real mismatch and the
    # strict load below is what should catch it.
    aux = {k for k in state if k.startswith("geom_aux_head.")}
    for k in aux:
        state.pop(k)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={list(missing)} "
                         f"unexpected={list(unexpected)}")
    if dropped or aux:
        print(f"  dropped {len(dropped) + len(aux)} non-forward key(s)")
    return model.to(device), ck


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--conditioning", default="pointcloud",
                    choices=("pointcloud", "oracle"))
    ap.add_argument("--val", default="2,15,4,6,3,11,9,22")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--encoder", default=None,
                    help='JSON encoder overrides, e.g. \'{"n_point": 128}\'')
    ap.add_argument("--precision", default="float32",
                    choices=("float32", "float64"),
                    help="evaluate at this precision. A float32 checkpoint "
                         "loads into a float64 model (load_state_dict casts), "
                         "so this measures the forward path's arithmetic "
                         "without retraining anything")
    ap.add_argument("--h-factor", type=float, default=1.3,
                    help="reference element size = default_h * this. The "
                         "default 1.3 holds the reference's own K_t "
                         "discretisation error under 0.2%%; the 2.5 the "
                         "study harnesses use costs up to 0.85%%, which is "
                         "half the parameter-conditioned model's error and "
                         "biased low, so it is too coarse to score a "
                         "concentration factor against")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default=None,
                    help="json path; defaults to rescore.json next to the "
                         "checkpoint")
    args = ap.parse_args()

    set_precision(args.precision)
    tag = args.tag or os.path.basename(os.path.dirname(args.checkpoint))
    enc_over = json.loads(args.encoder) if args.encoder else None
    model, ck = load(args.checkpoint, args.conditioning, args.device,
                     enc_over)
    epoch = ck.get("epoch", -1)
    print(f"Re-scoring {tag} ({args.conditioning}, epoch {epoch}, "
          f"{args.precision}) in the stress-concentration norms")
    print("=" * 78, flush=True)

    val = [int(g) for g in args.val.split(",")]
    rows = evaluate(model, val, args.device,
                    ref_h_factor=args.h_factor)

    hdr = "".join(f"{k:>15}" for k in REPORT_METRICS) + f"{'|N err|':>9}"
    print("\n" + "=" * len(hdr))
    print(f"{tag} — mean over {len(rows)} held-out geometries "
          "(vm_peak / scf by magnitude)")
    print(hdr)
    print("-" * len(hdr))
    print("".join(f"{mean_metric(rows, k):>15.3e}" for k in REPORT_METRICS)
          + f"{100 * mean_metric(rows, 'N_err'):>8.1f}%")

    out = args.out or os.path.join(os.path.dirname(args.checkpoint),
                                   "rescore.json")
    with open(out, "w") as fh:
        json.dump({"tag": tag, "checkpoint": args.checkpoint,
                   "conditioning": args.conditioning, "epoch": epoch,
                   "ref_h_factor": args.h_factor,
                   "precision": args.precision,
                   "rows": rows}, fh, indent=2, default=float)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
