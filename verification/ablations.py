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

import torch

import config as cfg
from config import COLLOCATION_CONFIG, TRAINING_CONFIG
from training.manifest import write_manifest
from verification.operator_study import (REPORT_METRICS, evaluate,
                                         mean_metric, train_operator)

# Arms whose name starts with "oracle-" swap the point-cloud encoder for an
# MLP over the four exact geometry parameters (see
# verification/supervised_ceiling.OracleConditioned).  Supervised, that change
# takes held-out v from 3.04e-01 to 1.07e-02 at bank 64; these arms ask whether
# it survives physics training, which is the only version that matters since
# supervision needs a solve per geometry.

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
    # The encoder probe (verification/encoder_probe.py) shows the encoder
    # carries the taper ratio almost perfectly (R^2 0.97) and essentially
    # nothing about the fillet (R/L 0.30, R/Wgrip 0.17).  Quadrupling the FPS
    # centroid count takes fillet decodability in an *untrained* encoder from
    # 0.156 to 0.451 -- past what the trained encoder manages at 32 -- so this
    # asks whether that representational gain becomes a field-accuracy gain.
    "encoder-np128": (
        "energy", {"_encoder": {"n_point": 128}}, {},
        "energy form, 128 FPS centroids instead of 32"),
    # The geometry auxiliary head, restored with dimensionless targets.  If
    # the reading in 4.4 is right -- the encoder can see the fillet and never
    # learns to, because the transverse reward is 183x attenuated -- then
    # paying for the extraction directly should move v without any reference
    # solution.  If v does not move, that reading is wrong.
    # The aux term starts near 1.0 (z-scored targets, untrained head) against
    # a reducible energy of ~1e-4, so the weights below span subordinate to
    # dominant.  Dominant is not obviously wrong here: the aux gradient reaches
    # only the encoder and the aux head, and the encoder's physics gradient is
    # exactly what is failing, so letting the aux own the encoder is closer to
    # what oracle conditioning does than it is to drowning the loss.
    "geomaux-1e-3": (
        "energy", {"w_geom_aux": 1e-3}, {},
        "aux ~10x the reducible energy"),
    "geomaux-1e-1": (
        "energy", {"w_geom_aux": 1e-1}, {},
        "aux ~1000x the reducible energy"),
    "geomaux-10": (
        "energy", {"w_geom_aux": 10.0}, {},
        "aux owns the encoder outright"),
    # The missing cell.  n_point=128 makes the fillet ratio linearly available
    # (0.156 -> 0.451 in an untrained encoder) but on its own changes no field
    # metric; the auxiliary loss supplies a gradient for extraction but at
    # n_point=32 fails to raise R/L at any weight, even 10 where it dominates
    # the objective outright.  Neither alone works.  This asks whether they are
    # complementary -- availability plus a reason to use it.
    "geomaux-np128": (
        "energy", {"w_geom_aux": 1e-3, "_encoder": {"n_point": 128}}, {},
        "128 FPS centroids AND the geometry auxiliary loss"),
    # Mixed u-P by stress potentials added to the energy, in the trainer loop.
    # The pilot-loop result (+0.72 representability, +0.85 in range held-out)
    # was never run here, and the trainer loop's plain energy arm is the one
    # that reaches +0.93 in range -- so this is the arm the re-test needs.
    "oracle-mixed-energy": (
        "mixed", {"_decoder": {"mixed": True}, "w_energy_mixed": 100.0}, {},
        "energy + constitutive gap with exactly-equilibrated stress "
        "potentials, decoder conditioned on the exact geometry parameters"),
    "oracle-energy": (
        "energy", {}, {},
        "energy form, decoder conditioned on the exact geometry parameters"),
    # --- Phase 6 Step 1: precision and per-component output scaling -------
    # Both are cheap, and both arrived with a stated mechanism that the
    # measurement had to check before the arm was worth running.
    #
    # float64.  The literature's large effect (Xu et al., arXiv 2505.10949,
    # 34-117x) comes from L-BFGS runs whose default tolerance_change of 1e-7
    # sits below float32 machine epsilon, so the inner loop stops on
    # arithmetic.  This trainer is Adam and has no such rule, and evaluating
    # a finished checkpoint in float64 moves every stress-concentration
    # metric by less than 1e-3 relative -- so the forward path is not
    # precision-limited either.  What is left is training dynamics, which is
    # what this arm measures.
    #
    # Output scale.  The per-component scale is set from what the *heads*
    # must produce, not from the fields: the ansatz already divides v by
    # eta, leaving a 3.46x imbalance rather than the 20x the field norms
    # suggest.  0.289 = 1/3.46 equalises the two heads on the bank mean.
    "fp64": (
        "energy", {"_precision": "float64"}, {},
        "identical to energy-w0 in float64 — does precision change the "
        "training dynamics, given it does not change the forward pass"),
    "outscale": (
        "energy", {"_decoder": {"output_scale": [1.0, 0.289]}}, {},
        "per-component output scale, 1/3.46 on the transverse head, "
        "measured from the reference rather than assumed from the fields"),
    "fp64-outscale": (
        "energy",
        {"_precision": "float64",
         "_decoder": {"output_scale": [1.0, 0.289]}}, {},
        "both, in case either alone is masked by the other"),
    # --- Phase 6 Step 4 precursor: does the energy form recover any K_t
    # skill when the equilibrium residual is added back?  Phase 6.4 showed
    # the architecture can reach R^2 +0.98 under stress supervision and
    # -0.55 under a 0.2% displacement fit, so the objective needs a term
    # sensitive to stress.  L_eq is the only one available without a second
    # output head.  The weights bracket the penalty form's own w_equilibrium
    # of 100 by two orders of magnitude either side, because nothing is known
    # about the right scale when the term sits next to an energy rather than
    # next to other penalties.
    "hybrid-1": (
        "hybrid", {"w_equilibrium_hybrid": 1.0}, {},
        "energy + equilibrium, w_eq 1 -- a nudge"),
    "hybrid-100": (
        "hybrid", {"w_equilibrium_hybrid": 100.0}, {},
        "energy + equilibrium at the penalty form's own weight"),
    "hybrid-10000": (
        "hybrid", {"w_equilibrium_hybrid": 1e4}, {},
        "energy + equilibrium, w_eq 1e4 -- the residual dominates"),
    "hybrid-0": (
        "hybrid", {"w_equilibrium_hybrid": 0.0}, {},
        "control: the hybrid path at zero weight must reproduce energy-w0, "
        "or the plumbing is changing something it should not"),
    "measure-consistent": (
        "penalty", {}, {"boundary_measure_consistent": True},
        "boundary points allocated by arc length — the sampler otherwise "
        "varies density up to 17.5x between segments"),
}



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
            max_epochs=args.epochs,
            conditioning="oracle" if name.startswith("oracle-")
            else "pointcloud")
        rows = evaluate(model, val_indices, args.device,
                        ref_h_factor=args.h_factor)
    finally:
        # train_operator mutates the module-level collocation config; without
        # this every later arm inherits the previous arm's sampler.
        COLLOCATION_CONFIG.clear()
        COLLOCATION_CONFIG.update(coll_backup)
    return {"form": form, "why": why, "overrides": over,
            "colloc_overrides": coll_over, "epochs": epochs,
            "wall_s": wall, "rows": rows}


def report(results: dict):
    hdr = (f"{'arm':>20}{'epochs':>8}"
           + "".join(f"{k:>15}" for k in REPORT_METRICS) + f"{'|N err|':>9}")
    print("\n" + "=" * len(hdr))
    print("Held-out geometries, mean over the set — ratio to baseline on the "
          "line below (<1 is better)")
    print("vm_peak and scf are signed per geometry and averaged by magnitude "
          "here; vm_fillet_Linf is the")
    print("worst pointwise fillet error as a fraction of the reference peak.")
    print(hdr)
    print("-" * len(hdr))

    base = results.get("baseline")
    for name, r in results.items():
        cells = "".join(f"{mean_metric(r['rows'], k):>15.3e}"
                        for k in REPORT_METRICS)
        print(f"{name:>20}{r['epochs']:>8d}{cells}"
              f"{100 * mean_metric(r['rows'], 'N_err'):>8.1f}%")
        if base is not None and name != "baseline":
            rb = lambda k: (mean_metric(r["rows"], k)
                            / max(mean_metric(base["rows"], k), 1e-30))
            ratios = "".join(f"{rb(k):>15.2f}" for k in REPORT_METRICS)
            print(f"{'(vs baseline)':>20}{'':>8}{ratios}"
                  f"{rb('N_err'):>9.2f}")


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
    ap.add_argument("--h-factor", type=float, default=1.3,
                    help="reference element size = default_h * this. Phase "
                         "6.1 measured the reference's own K_t error at "
                         "0.85%% for 2.5 and 0.19%% for 1.3; everything "
                         "before that phase was scored at 2.5, so older "
                         "numbers are comparable to each other and not to "
                         "these")
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
