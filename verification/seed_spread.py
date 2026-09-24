"""Phase 9.0 -- how much of the K_t error is where training happened to stop?

Evaluation only; reads the scored results already on disk.

For each family and split, the relative K_t error e[s, g] (seed s, geometry g)
is decomposed as

    mean_s,g e^2  =  mean_g m_g^2  +  mean_g var_s(e[., g])

where m_g is the seed-mean error on geometry g.  With only three seeds, m_g
still carries a third of the seed variance, so the *systematic* part (the error
all seeds share) is estimated without bias as

    systematic^2  =  mean_g m_g^2  -  mean_g s_g^2 / n_seeds      (s_g: ddof=1)

(the squared term is unbiased; its root, clipped at zero, is not), and the
seed part is sqrt(mean_g s_g^2).  "seed_share_of_mse" is the fraction of the
per-seed MSE removed by averaging the seeds' predictions.  Every seed sees the
same data stream (batch order from default_rng(42), quadrature from
EnergyLoss(seed=0)), so the seed changes the initialisation only.  An optimizer
that reached a unique minimum would remove the seed part; whether that minimum
is accurate is a separate question (Phase 9.0b / 9.2).

Also reports the learning-rate trajectory and the final validation energy of
every Tier-1 run: the plateau scheduler never fired, and seeds that disagree on
K_t by several per cent agree on the energy to a few parts in 1e4.

    python -m verification.seed_spread
"""
import glob
import json
import os

import numpy as np

TIER1 = "verification/results/phase8/tier1"
DOGBONE = "docs/phase6_7_retest_scores.json"
OUT = "docs/phase9_seed_spread.json"
FAMILIES = ["dogbone", "open_hole", "inclusion", "double_notch", "single_notch"]


def errors(fam, split, tier1, retest):
    if fam == "dogbone":
        E = [[g["err_u"] for g in retest[f"energy|{s}"][split]] for s in range(3)]
        K = [g["scf_ref"] for g in retest["energy|0"][split]]
    else:
        E = [[g["err"] for g in tier1[f"{fam}|{s}"][split]] for s in range(3)]
        K = [g["K_ref"] for g in tier1[f"{fam}|0"][split]]
    return np.array(E), np.array(K)


def decompose(E):
    n = E.shape[0]
    m = E.mean(0)
    s2 = E.var(0, ddof=1)
    rms = float(np.sqrt((E ** 2).mean()))
    seed = float(np.sqrt(s2.mean()))
    sys2 = float((m ** 2).mean() - s2.mean() / n)
    return {
        "rms": rms,
        "mean_abs": float(np.abs(E).mean()),
        "seed_sd": seed,
        "systematic": float(np.sqrt(max(sys2, 0.0))),
        "systematic_sq_raw": sys2,
        "seed_share_of_mse": float(E.var(0, ddof=0).mean() / (E ** 2).mean()),
        "ensemble_mean_abs": float(np.abs(m).mean()),
    }


def histories():
    rows = {}
    for p in sorted(glob.glob(f"{TIER1}/*_s*/history.json")):
        h = json.load(open(p))
        lr = np.array([r["lr"] for r in h])
        st = np.array([r["step"] for r in h])
        ve = np.array([r.get("val_ema", np.nan) for r in h])

        def at(s):
            return float(ve[min(np.searchsorted(st, s), len(st) - 1)])

        rows[os.path.basename(os.path.dirname(p))] = {
            "steps": int(st[-1]),
            "lr_start": float(lr[0]),
            "lr_end": float(lr[-1]),
            "lr_cuts": int((np.diff(lr) < 0).sum()),
            "val_ema": {s: at(s) for s in (400, 800, 1200, 1600)},
            "val_drop_last_400_rel": (at(1200) - at(1600)) / at(1600),
        }
    return rows


def main():
    tier1 = json.load(open(f"{TIER1}/scores.json"))
    retest = json.load(open(DOGBONE))
    out = {"decomposition": {}, "histories": histories(), "energy_tie": {}}
    print("family         split  mean|err|  rms    seed sd  systematic  seed share  3-seed avg |err|")
    for fam in FAMILIES:
        for split in ("IR", "OOR"):
            E, _ = errors(fam, split, tier1, retest)
            d = decompose(E)
            out["decomposition"][f"{fam}|{split}"] = d
            print(f"{fam:14s} {split:4s}  {d['mean_abs']*100:6.2f}%  {d['rms']*100:5.2f}%  "
                  f"{d['seed_sd']*100:5.2f}%   {d['systematic']*100:5.2f}%      "
                  f"{d['seed_share_of_mse']:4.2f}       {d['ensemble_mean_abs']*100:5.2f}%")
    h = out["histories"]
    print("\nlearning rate: start/end/cuts per run")
    for k, r in h.items():
        print(f"  {k:16s} {r['lr_start']:.1e} -> {r['lr_end']:.1e}  cuts {r['lr_cuts']}  "
              f"val energy still falling {r['val_drop_last_400_rel']*100:.2f}% over the last 400 steps")
    print("\nfinal validation energy across seeds vs K_t spread across seeds (in range)")
    for fam in FAMILIES[1:]:
        v = np.array([h[f"{fam}_s{s}"]["val_ema"][1600] for s in range(3)])
        spread = float((v.max() - v.min()) / v.mean())
        sd = out["decomposition"][f"{fam}|IR"]["seed_sd"]
        out["energy_tie"][fam] = {"val_energy_rel_spread": spread, "kt_seed_sd": sd}
        print(f"  {fam:14s} energy spread {spread*100:.3f}%   K_t seed sd {sd*100:.2f}%")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
