#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One lever at a time, on the operator, against the reference.

Phases 0-2 added four levers to the training path.  Each was justified by a
measurement -- but every one of those measurements was made *outside* the
operator: on a closed-form uniaxial solution, on a single fine-tuned
specimen, or on the collocation sampler in isolation.  A lever that is
demonstrably right in isolation can still fail to matter, or matter the
wrong way, once one network is fitting a whole geometry family.  This
module settles that, and it is the reason all four ship **off by default**:
the defaults encode what has been measured *here*, not what seemed right
when the lever was written.

The levers
----------
``loss_form``                 penalty (default) | energy
``resultant_anchor``          series (default) | series_calibrated | uniform
                              | small_strain | none
``boundary_measure_consistent``  False (default) | True
``w_trac_top``                2.0 (default) | raised

Protocol
--------
Identical to ``verification.operator_study`` and reusing its machinery, so
the arms here and the two forms there are directly comparable: same bank,
same wall-clock budget per arm, same held-out geometries (disjoint from the
trainer's own validation prefix), and a finite-element reference solved once
and shared by every arm.

Equal wall-clock is the right budget for the loss-form arm and a mild
handicap for the rest, which cost the same per epoch as the baseline; the
epoch counts are reported so that is visible rather than hidden.

Two arms are deliberately absent.  Under the energy form the trainer
switches the section anchor and the adaptive weighting off, and the
traction terms are natural BCs -- so ``resultant_anchor`` and
``w_trac_top`` have no meaning there.  Crossing them with ``energy`` would
produce arms that differ from their baseline in name only.

Run with:
    python -m verification.ablations --minutes 20
    python -m verification.ablations --arms baseline,anchor-legacy --minutes 20
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

import config as cfg
from config import COLLOCATION_CONFIG, TRAINING_CONFIG
from training.manifest import write_manifest
from verification.operator_study import evaluate, train_operator

# name -> (form, TRAINING_CONFIG overrides, COLLOCATION_CONFIG overrides, why)
ARMS = {
    "baseline": (
        "penalty", {}, {},
        "shipped defaults — the reference point for every other arm"),
    "energy": (
        "energy", {}, {},
        "total potential energy; traction conditions become natural BCs"),
    "wtop100": (
        "penalty", {"w_trac_top": 100.0}, {},
        "Phase 1's lever: the gauge-top term is the only thing setting "
        "lateral contraction, and it carried the lowest weight"),
    "anchor-legacy": (
        "penalty", {"resultant_anchor": "small_strain"}, {},
        "the anchor as shipped before Phase 0 — E*(u/L)*2H, 3.60% mean "
        "error against the reference"),
    "anchor-calibrated": (
        "penalty", {"resultant_anchor": "series_calibrated"}, {},
        "series x0.97517, fitted on the train bank — 0.74% mean error, but "
        "weakly reference-dependent, which is why it is opt-in"),
    "anchor-none": (
        "penalty", {"resultant_anchor": "none"}, {},
        "no absolute target; reaction self-consistency alone.  Separates "
        "'the anchor is biased' from 'the anchor is load-bearing'"),
    # The section anchor under the energy form.  Phase 2 shipped it off on the
    # strength of a <=0.5% force error with no anchor -- on ONE fine-tuned
    # specimen.  The operator study shows the energy form's force error is
    # *worse* than the penalty form's once capacity is shared (3.5% vs 2.5%),
    # which is what "natural BCs hold at the minimum" predicts for a network
    # that does not reach the minimum per geometry.  These arms ask whether
    # putting the anchor back recovers the force balance without giving up the
    # stress accuracy the energy form buys.
    #
    # The weight has to be swept rather than guessed: the energy loss is
    # O(5e-4) with only ~1e-4 of it reducible, while the anchor term starts
    # near 0.4, so the sensible range spans orders of magnitude and the
    # informative comparison is against `energy-w0` at the same epoch budget.
    "energy-w0": (
        "energy", {}, {}, "energy form, no anchor — control for the sweep"),
    "energy-anchor-1e-5": (
        "energy", {"w_resultant_energy": 1e-5}, {},
        "anchor at ~4% of the reducible energy"),
    "energy-anchor-1e-4": (
        "energy", {"w_resultant_energy": 1e-4}, {},
        "anchor at ~40% of the reducible energy"),
    "energy-anchor-1e-3": (
        "energy", {"w_resultant_energy": 1e-3}, {},
        "anchor dominant — expected to trade stress accuracy for force"),
    # The sweep above moves the force error only 4.2% -> 3.7%.  The likely
    # reason is that the anchor's TARGET is itself biased: `series` runs ~2.5%
    # high against the finite-element reference, so no weight on it can pull a
    # +4% error below +2.5%.  This arm swaps in the calibrated target (0.74%
    # mean error) at the sweep's best weight, which is the direct test.
    "energy-anchor-calib": (
        "energy", {"w_resultant_energy": 1e-4,
                   "resultant_anchor": "series_calibrated"}, {},
        "best sweep weight, but against a target that is not itself biased"),
    # The corrected absolute-anchor arms.  w_resultant_abs_energy weights the
    # anchor term directly, so these are the first arms that actually vary the
    # force LEVEL rather than the slice-to-slice consistency.  Targeting the
    # anchor at ~10-100% of the reducible energy (~1e-4) with L_anchor ~3e-4
    # puts the useful range around 0.03-3.
    "energy-abs-0.03": (
        "energy", {"w_resultant_abs_energy": 0.03,
                   "resultant_anchor": "series_calibrated"}, {},
        "absolute anchor at ~10% of the reducible energy"),
    "energy-abs-0.3": (
        "energy", {"w_resultant_abs_energy": 0.3,
                   "resultant_anchor": "series_calibrated"}, {},
        "absolute anchor comparable to the reducible energy"),
    "energy-abs-3": (
        "energy", {"w_resultant_abs_energy": 3.0,
                   "resultant_anchor": "series_calibrated"}, {},
        "absolute anchor dominant — the other end of the useful range"),
    # Phase 3: the traction-free residual restored to the energy form.
    #
    # Redundant at the exact minimiser -- it is a natural BC of Pi, and
    # test_energy_traction_terms_vanish_on_the_exact_solution checks that on
    # the reference -- so it cannot bias the converged answer.  What it changes
    # is the approach, in the direction the objective is 183x flatter
    # (docs/phase3_transverse_conditioning.md).
    #
    # The weight is swept around 183 rather than guessed: at w = 1 the term is
    # ~30% of the reducible energy, and equalising the transverse curvature
    # with the axial one needs roughly the curvature ratio.
    # The high sweep (10/100/1000) degraded everything monotonically while
    # barely moving v, and the reason is that the weight was sized from a
    # *value* ratio -- L_trac_top ~30% of the reducible energy at w = 1 -- when
    # what matters is curvature.  Even w = 10 puts the traction term ~7x the
    # reducible energy at the first epoch, so the whole high sweep was in the
    # over-dominant regime and never tested the hypothesis.  These arms sweep
    # below it.
    "energy-trac-0.03": (
        "energy", {"energy_w_trac_top": 0.03}, {},
        "gauge-top traction well below the energy's own scale"),
    "energy-trac-0.3": (
        "energy", {"energy_w_trac_top": 0.3}, {},
        "gauge-top traction at ~10% of the reducible energy"),
    "energy-trac-3": (
        "energy", {"energy_w_trac_top": 3.0}, {},
        "gauge-top traction comparable to the reducible energy"),
    "energy-trac-10": (
        "energy", {"energy_w_trac_top": 10.0}, {},
        "gauge-top traction restored, an order below the predicted weight"),
    "energy-trac-100": (
        "energy", {"energy_w_trac_top": 100.0}, {},
        "gauge-top traction at ~the measured curvature ratio"),
    "energy-trac-1000": (
        "energy", {"energy_w_trac_top": 1000.0}, {},
        "gauge-top traction an order above — expected to trade energy for it"),
    "measure-consistent": (
        "penalty", {}, {"boundary_measure_consistent": True},
        "boundary points allocated by arc length — the sampler otherwise "
        "varies density up to 17.5x between segments"),
}

METRICS = ("u", "v", "vm", "vm_fillet")


def run_arm(name: str, args, val_indices):
    form, over, coll_over, why = ARMS[name]
    print(f"\n  --- {name} ({form}) ---")
    print(f"      {why}", flush=True)

    coll_backup = copy.deepcopy(COLLOCATION_CONFIG)
    try:
        model, epochs, wall = train_operator(
            form, args.minutes, args.bank, args.batch, args.lr, args.device,
            args.seed, args.out, args.n_interior,
            overrides=over, colloc_overrides=coll_over, tag=name,
            max_epochs=args.epochs)
        rows = evaluate(model, val_indices, args.device)
    finally:
        # train_operator mutates the module-level collocation config; without
        # this every later arm inherits the previous arm's sampler.
        COLLOCATION_CONFIG.clear()
        COLLOCATION_CONFIG.update(coll_backup)
    return {"form": form, "why": why, "overrides": over,
            "colloc_overrides": coll_over, "epochs": epochs,
            "wall_s": wall, "rows": rows}


def report(results: dict):
    print("\n" + "=" * 94)
    print("Held-out geometries, mean over the set — ratio to baseline in "
          "parentheses (<1 is better)")
    print(f"{'arm':>20}{'epochs':>8}{'u':>11}{'v':>11}{'vm':>11}"
          f"{'vm fillet':>12}{'|N err|':>10}")
    print("-" * 94)

    def mean(r, k):
        return float(np.mean([x[k] for x in r["rows"]]))

    def mean_N(r):
        return float(np.mean([abs(x["N_err"]) for x in r["rows"]]))

    base = results.get("baseline")
    for name, r in results.items():
        cells = "".join(f"{mean(r, k):>11.3e}" if k != "vm_fillet"
                        else f"{mean(r, k):>12.3e}" for k in METRICS)
        print(f"{name:>20}{r['epochs']:>8d}{cells}{100 * mean_N(r):>9.1f}%")
        if base is not None and name != "baseline":
            ratios = "".join(
                f"{mean(r, k) / max(mean(base, k), 1e-30):>11.2f}"
                if k != "vm_fillet" else
                f"{mean(r, k) / max(mean(base, k), 1e-30):>12.2f}"
                for k in METRICS)
            print(f"{'(vs baseline)':>20}{'':>8}{ratios}"
                  f"{mean_N(r) / max(mean_N(base), 1e-30):>10.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--epochs", type=int, default=None,
                    help="fixed epoch budget instead of wall-clock; the "
                         "right budget when the arms cost the same per epoch")
    ap.add_argument("--bank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-interior", type=int, default=1500)
    ap.add_argument("--val", default="2,11,13,16,19")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--out", default="verification/results/ablations")
    args = ap.parse_args()

    names = [a.strip() for a in args.arms.split(",")]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; have {list(ARMS)}")
    if "baseline" not in names:
        print("  note: no 'baseline' arm — ratios will be omitted")

    val_indices = [int(g) for g in args.val.split(",")]
    n_seen = 2  # bank_val_size the trainer is given, see operator_study
    leaked = [g for g in val_indices if g < n_seen]
    if leaked:
        raise SystemExit(f"--val {leaked} overlaps the trainer's validation "
                         f"bank; pick indices >= {n_seen}")

    os.makedirs(args.out, exist_ok=True)
    write_manifest(args.out, cfg,
                   extra={"case": "ablations", "args": vars(args)})

    print("Ablations — one lever at a time, on the operator")
    print("=" * 94)
    print(f"  {len(names)} arms x {args.minutes:g} min, train bank "
          f"{args.bank}, batch {args.batch}, lr {args.lr}")
    print(f"  held-out geometries {val_indices}")

    results = {}
    for name in names:
        results[name] = run_arm(name, args, val_indices)
        with open(os.path.join(args.out, "result.json"), "w") as fh:
            json.dump(results, fh, indent=2, default=float)
    report(results)
    print(f"\nWrote {os.path.join(args.out, 'result.json')}")


if __name__ == "__main__":
    main()
