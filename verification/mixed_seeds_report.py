#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-seed verdict on energy vs mixed + energy, representability protocol.

Seed 0 comes from the original pilot runs; seeds 1 and 2 from
verification/results/mixed_seeds.  Within a seed the two arms share the model
initialisation and the geometry sampling order -- train() seeds everything
before building the model -- so the comparison is paired by seed, and at the
finest grain by (geometry, seed).

The energy arms of seeds 1 and 2 ran as two parallel single-thread processes;
every other arm ran as one two-thread process.  That changes reduction order
and nothing else -- one more source of run-to-run variation, applied to the
energy arm, and so if anything widening the energy spread the m+e arm is
compared against rather than favouring m+e.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

SRC = {
    ("energy", 0): ("verification/results/mixed_pilot/result.json", "energy"),
    ("m+e", 0): ("verification/results/mixed_energy/result.json", "mixed+energy"),
}
# Seeds 1 and 2: the energy arms finished in a first, parallel launch; the
# mixed + energy arms were restarted sequentially at the default two threads,
# because two single-thread m+e processes contended to 10 s/epoch against 3 s
# for one.  Hence two directories per seed.
for s in (1, 2):
    SRC[("energy", s)] = (
        f"verification/results/mixed_seeds/seed{s}/result_energy.json", "energy")
    SRC[("m+e", s)] = (
        f"verification/results/mixed_seeds/seed{s}_me/result.json", "mixed+energy")


def load():
    out = {}
    for key, (path, arm) in SRC.items():
        if os.path.exists(path):
            d = json.load(open(path))
            if arm in d:
                out[key] = d[arm]["rows"]
    return out


def r2(rows, k="scf_pinn"):
    ref = np.array([x["scf_ref"] for x in rows])
    p = np.array([x[k] for x in rows])
    return float(1 - ((p - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def main():
    R = load()
    seeds = sorted({s for (_, s) in R if ("energy", s) in R and ("m+e", s) in R})
    print(f"Energy vs mixed + energy — {len(seeds)} paired seeds: {seeds}")
    print("=" * 84)
    metrics = {
        "K_t R2 (from u)": lambda r: r2(r),
        "mean |K_t err| %": lambda r: 100 * np.mean([abs(x["scf"]) for x in r]),
        "v rel L2": lambda r: np.mean([x["v"] for x in r]),
        "vm rel L2": lambda r: np.mean([x["vm"] for x in r]),
        "|dx*| x_g": lambda r: np.mean([abs(x["x_peak_err"]) for x in r]),
    }
    print(f"{'metric':>20}" + "".join(f"{'energy s'+str(s):>11}" for s in seeds)
          + "".join(f"{'m+e s'+str(s):>10}" for s in seeds)
          + f"{'energy':>16}{'m+e':>16}")
    for name, f in metrics.items():
        e = [f(R[("energy", s)]) for s in seeds]
        m = [f(R[("m+e", s)]) for s in seeds]
        sd = lambda a: np.std(a, ddof=1) if len(a) > 1 else float("nan")
        print(f"{name:>20}" + "".join(f"{x:>11.3f}" for x in e)
              + "".join(f"{x:>10.3f}" for x in m)
              + f"{np.mean(e):>9.3f}±{sd(e):<6.3f}{np.mean(m):>9.3f}±{sd(m):<6.3f}")
    phi = [r2(R[("m+e", s)], "scf_pinn_phi") for s in seeds]
    print(f"{'m+e K_t R2 (phi)':>20}" + " " * (11 * len(seeds))
          + "".join(f"{x:>10.3f}" for x in phi)
          + f"{'':>16}{np.mean(phi):>9.3f}")

    # Paired at the finest grain: (geometry, seed).
    de, dm = [], []
    for s in seeds:
        for xe, xm in zip(R[("energy", s)], R[("m+e", s)]):
            assert xe["geo"] == xm["geo"]
            de.append(abs(xe["scf"])); dm.append(abs(xm["scf"]))
    de, dm = np.array(de), np.array(dm)
    diff = dm - de
    wins = int((diff < 0).sum())
    w = stats.wilcoxon(dm, de, alternative="less") if len(diff) >= 6 else None
    print("\nPaired by (geometry, seed), |K_t error|:")
    print(f"  mixed + energy better on {wins} of {len(diff)}; "
          f"mean change {100 * diff.mean():+.2f} pp "
          f"({100 * (dm.mean() / de.mean() - 1):+.0f}%)")
    if w is not None:
        print(f"  Wilcoxon signed-rank, one-sided (m+e smaller): p = {w.pvalue:.4f}")
    r2e = [r2(R[("energy", s)]) for s in seeds]
    r2m = [r2(R[("m+e", s)]) for s in seeds]
    print(f"  K_t R2 higher for m+e in {sum(m > e for e, m in zip(r2e, r2m))} "
          f"of {len(seeds)} seeds")


if __name__ == "__main__":
    main()
