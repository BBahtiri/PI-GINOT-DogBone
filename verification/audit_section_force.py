#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Were any stored section-force errors corrupted by the id-reuse bug?

Recompute N_err with the fixed locator for every run whose scoring loop freed
its reference solutions, and compare with what was stored.  N_pinn never
touched the locator, so any difference is N_fem."""
import json, numpy as np, torch
from config import DECODER_CONFIG, ENCODER_CONFIG, LOADING_CONFIG, TRAINING_CONFIG
from eval.compare_reference import _pinn_resultant
from geometry.banks import VAL_BANK_SEED
from models.pi_ginot import PI_GINOT
from verification.supervised_ceiling import OracleConditioned, _entries, _solve_cached

VAL = [2, 15, 4, 6, 3, 11, 9, 22]
ents = _entries(TRAINING_CONFIG["bank_val_size"], bank_seed=VAL_BANK_SEED, pick=VAL)
RUNS = [  # (label, checkpoint, mixed decoder, result json, arm key or None)
    ("ceiling disp", "verification/results/ceiling/probe_disp/last.pt", False,
     "verification/results/ceiling/probe_disp/result.json", None),
    ("ceiling stress", "verification/results/ceiling/probe_stress/last.pt", False,
     "verification/results/ceiling/probe_stress/result.json", None),
    ("ceiling nopeak", "verification/results/ceiling/probe_stress_nopeak/last.pt", False,
     "verification/results/ceiling/probe_stress_nopeak/result.json", None),
    ("ceiling both", "verification/results/ceiling/probe_both/last.pt", False,
     "verification/results/ceiling/probe_both/result.json", None),
    ("pilot energy", "verification/results/mixed_pilot/energy.pt", True,
     "verification/results/mixed_pilot/result.json", "energy"),
    ("pilot mixed", "verification/results/mixed_pilot/mixed.pt", True,
     "verification/results/mixed_pilot/result.json", "mixed"),
    ("pilot m+e", "verification/results/mixed_energy/mixed+energy.pt", True,
     "verification/results/mixed_energy/result.json", "mixed+energy"),
]
d = torch.get_default_dtype()
print(f"{'run':>16}{'max |dN_err| (pp)':>20}{'stored mean |N|':>17}{'correct mean |N|':>18}")
for label, ck, mixed, res, arm in RUNS:
    m = OracleConditioned(ENCODER_CONFIG, dict(DECODER_CONFIG, mixed=mixed),
                          TRAINING_CONFIG["bank_geo_ranges"])
    st, _ = PI_GINOT.strip_legacy_keys(torch.load(ck, map_location="cpu",
                                                  weights_only=False)["model_state_dict"])
    m.load_state_dict({k: v for k, v in st.items() if not k.startswith("geom_aux")},
                      strict=True)
    m.eval()
    stored = json.load(open(res))
    rows = stored[arm]["rows"] if arm else stored["rows"]
    diffs, new = [], []
    for (gm, coll), gi, row in zip(ents, VAL, rows):
        assert row["geo"] == gi
        sol = _solve_cached(gm.params, f"val_{VAL_BANK_SEED}_{gi}", 1.3)
        fi = sol.mesh.fillet_info
        xs = np.array([0.15, 0.35, 0.55, 0.75, 0.92]) * fi["L_half"]
        nf = np.mean([sol.axial_resultant(x) for x in xs])
        m.set_geometry(gm.params)
        npn = np.mean([_pinn_resultant(m, gm.params, x, LOADING_CONFIG["u_max"],
                                       "cpu", 100 + gi, coll, fi) for x in xs])
        e = (npn - nf) / nf
        new.append(abs(e)); diffs.append(abs(e - row["N_err"]))
    print(f"{label:>16}{100*max(diffs):>19.3f}{100*np.mean([abs(r['N_err']) for r in rows]):>16.2f}%"
          f"{100*np.mean(new):>17.2f}%")
