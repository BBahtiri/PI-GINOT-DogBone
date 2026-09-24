#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free-edge traction residual per family (Phase 8 Tier 1, reported without a
prediction -- docs/phase8_tier1.md lists it as descriptive).

Phase 7 located the energy form's weakness at free edges: the natural
traction-free condition is only approximately met, and at a free-edge peak a
spurious normal stress moves von Mises away from |sigma_tt|.  This measures,
for each family's in-range operators:

  r_feature   RMS |P . N| / sigma_nom on the concentration feature's free
              boundary (hole, notch flank + root), at the FEM boundary nodes
  r_free      the same over every free boundary segment
  delta_peak  vm / |sigma_tt| - 1 at the network's largest boundary von Mises
              on the feature, in the deformed frame (Phase 7's delta_BC)

sigma_nom = N / A_gross from the FEM (the K_t denominator).  The rigid
inclusion's peak is on a Dirichlet boundary, so it has no feature entry.

    python -m verification.family_residual
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from geometry.families import FAMILIES
from training.family_trainer import load
from verification.bc_audit import _eval, _frame, _traction
from verification.family_fem import reaction, reference
from verification.family_score import ROOT, SEEDS, _done

FREE = {"open_hole": (("hole",), ("top", "hole")),
        "inclusion": ((), ("top",)),
        "double_notch": (("flank", "notch"), ("top", "flank", "notch")),
        "single_notch": (("flank", "notch"), ("top", "flank", "notch", "bottom"))}


def _normals(pts):
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([t[:, 1], -t[:, 0]], -1)       # outward for a CCW loop


def run():
    out = {}
    for name, fam in FAMILIES.items():
        feat, free = FREE[name]
        for s in SEEDS:
            ok, path = _done(name, s)
            if not ok:
                continue
            model = load(name, path)
            rows = []
            for i, p in enumerate(fam.in_range()):
                sol = reference(fam, p, f"ir{i}")
                s_nom = reaction(sol) / fam.gross_height(p)
                dt = torch.get_default_dtype()
                L, Hs = fam.extent(p)
                model.set_geometry(p)
                with torch.no_grad():
                    z = model.encode(torch.zeros(1, 1, 2, dtype=dt),
                                     torch.tensor([L], dtype=dt), torch.tensor([Hs], dtype=dt))
                lat = (z, torch.tensor([L], dtype=dt), torch.tensor([Hs], dtype=dt))
                segs = {}
                for seg in free:
                    idx = sol.mesh.boundary[seg]
                    pts = sol.mesh.nodes[idx]
                    if len(pts) < 3:
                        continue
                    N = _normals(pts)
                    core = slice(1, len(pts) - 1)
                    e = _eval(model, lat, pts[core])
                    tr = np.linalg.norm(_traction(e, N[core]), axis=1) / s_nom
                    segs[seg] = (e, N[core], tr)
                r_free = float(np.sqrt(np.mean(np.concatenate([v[2] ** 2 for v in segs.values()]))))
                row = {"r_free": r_free}
                fs = [segs[k] for k in feat if k in segs]
                if fs:
                    tr = np.concatenate([v[2] for v in fs])
                    row["r_feature"] = float(np.sqrt(np.mean(tr ** 2)))
                    vm = np.concatenate([v[0]["vm"] for v in fs])
                    k = int(np.argmax(vm))
                    e_all = {kk: np.concatenate([v[0][kk] for v in fs]) for kk in fs[0][0]}
                    e_pk = {kk: vv[k:k + 1] for kk, vv in e_all.items()}
                    N_pk = np.concatenate([v[1] for v in fs])[k:k + 1]
                    s_tt, s_nn, _ = (float(a[0]) for a in _frame(e_pk, N_pk))
                    row["delta_peak"] = float(e_pk["vm"][0] / abs(s_tt) - 1.0)
                    row["snn_over_stt"] = s_nn / s_tt
                rows.append(row)
            out[f"{name}|{s}"] = rows
            print(f"{name:13s} s{s}: r_free {100 * np.median([r['r_free'] for r in rows]):.2f}%"
                  + (f"  r_feature {100 * np.median([r['r_feature'] for r in rows]):.2f}%"
                     f"  delta_peak {100 * np.mean([r['delta_peak'] for r in rows]):+.2f}%"
                     f"  snn/stt {100 * np.median([r['snn_over_stt'] for r in rows]):+.2f}%"
                     if "r_feature" in rows[0] else ""), flush=True)
    json.dump(out, open(os.path.join(ROOT, "residuals.json"), "w"), indent=2)


if __name__ == "__main__":
    run()
