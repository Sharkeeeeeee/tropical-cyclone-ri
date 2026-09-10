"""Experiment A: is label provenance a confound in RI prediction?

Three tests, all runnable on the SHIPS predictors alone, so they do not wait on
the global imagery extraction. The vision-model versions reuse the same harness
once patches exist.

  A1  Quantization. Do best-track intensities pile up on the Dvorak CI table,
      and does that get stronger where there is no aircraft reconnaissance?

  A2  Rate convention. What 24 h intensity change does the Dvorak "typical"
      rate of +1.0 T per day permit, and how does that sit against the 30 kt
      RI threshold?

  A3  Cross-domain transfer. Train on recon-verified cases, test on
      Dvorak-derived ones and vice versa. A model that has learned cyclone
      physics should transfer. One that has learned an analyst's convention
      should not, and should fail asymmetrically.

    python -u scripts/17_experiment_a.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.evaluate import summary  # noqa: E402
from tcri.provenance import (CI_TO_KT, DVORAK_KT,  # noqa: E402
                             dvorak_rate_ceiling, quantization_contrast)

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)

ENV = ["per12", "vmax", "pot", "shrd", "t200", "rhlo", "d200", "cohc"]
IR = ["ir_pct_lt_m30", "ir_std_bt_0_200"]
GRID = np.arange(25, 156, 5)


def section(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def gbm(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def load():
    d = pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv",
                    converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    d = d[(d["SEASON"] >= 2000) & d["overwater_24h"]]
    return d.dropna(subset=ENV).reset_index(drop=True)


# ------------------------------------------------------------------- A1
def a1_quantization(_unused, n_boot):
    section("A1  quantization of best-track intensity onto the Dvorak table")
    # Quantization is a property of the labels, so this runs on the full label
    # table rather than the SHIPS-joined subset. Restricting it to cases that
    # happen to have environmental predictors would be an arbitrary filter on a
    # measurement that does not use them.
    d = pd.read_csv(PROCESSED_DIR / "ri_labels.csv",
                    converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    d = d[(d["SEASON"] >= 2000) & d["overwater_24h"]].reset_index(drop=True)
    print(f"full label table: {len(d):,} cases, "
          f"{d['SID'].nunique():,} storms\n")
    print(f"Dvorak CI winds, snapped to the 5 kt best-track grid: {DVORAK_KT}")
    print("A contrast of 0 means intensities are no more common on those "
          "values than beside them.\n")
    rows = []
    for label, sub in [("recon-verified (NA/EP)", d[d["recon_basin"]]),
                       ("Dvorak-derived (others)", d[~d["recon_basin"]])]:
        pt, (lo, hi) = quantization_contrast(sub["v0"].to_numpy(), GRID,
                                             n_boot=n_boot)
        rows.append({"sample": label, "n": len(sub), "contrast": pt,
                     "ci_lo": lo, "ci_hi": hi})
    t = pd.DataFrame(rows).set_index("sample")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    ratio = t["contrast"].iloc[1] / t["contrast"].iloc[0]
    print(f"\nquantization is {ratio:.2f}x stronger without reconnaissance")

    print("\nby basin:")
    rows = []
    for b, sub in d.groupby("BASIN"):
        if len(sub) < 500:
            continue
        pt, (lo, hi) = quantization_contrast(sub["v0"].to_numpy(), GRID,
                                             n_boot=n_boot)
        rows.append({"basin": b, "n": len(sub), "contrast": pt,
                     "ci_lo": lo, "ci_hi": hi,
                     "recon": "yes" if b in ("NA", "EP") else "no"})
    print(pd.DataFrame(rows).set_index("basin").to_string(
        float_format=lambda v: f"{v:.3f}"))
    print("\nNote: JTWC has used different CI-to-wind conversions in the west")
    print("Pacific, which blurs this signal rather than creating one. The test")
    print("is therefore conservative where it matters most.")
    return t


# ------------------------------------------------------------------- A2
def a2_rate_convention(d):
    section("A2  the Dvorak rate convention against the RI threshold")
    print("Intensity change over 24 h implied by a CI change of +1.0 (the")
    print("technique's typical rate) and +2.5 (its hard cap for T >= 4.0):\n")
    print(f"{'start (kt)':>11} {'CI':>5} {'+1.0 T/day':>12} {'+2.5 T/24h':>12}"
          f" {'typical >= 30 kt?':>18}")
    reach = []
    for kt in (35, 45, 55, 65, 77, 90, 102, 115):
        d1 = dvorak_rate_ceiling(kt, 1.0)
        d25 = dvorak_rate_ceiling(kt, 2.5)
        ci = [k for k, v in CI_TO_KT.items() if v == kt]
        reach.append(d1 >= 30)
        print(f"{kt:>11} {ci[0] if ci else float('nan'):>5.1f} "
              f"{d1:>+12.0f} {d25:>+12.0f} {'yes' if d1 >= 30 else 'no':>18}")
    print(f"\nThe typical rate reaches the 30 kt RI threshold at "
          f"{sum(reach)}/{len(reach)} starting intensities.")
    print("So recording an RI event requires the analyst to exceed the")
    print("technique's typical rate, though not its hard cap. The pressure is a")
    print("convention, not a ceiling -- which is why it can bias the labels")
    print("without making RI impossible to record.")

    obs = d[d["ri"] == 1]
    exceed = np.array([r.dv24 > dvorak_rate_ceiling(r.v0, 1.0)
                       for r in obs.itertuples()])
    print(f"\nOf {len(obs):,} recorded RI events, {100*exceed.mean():.1f}% "
          "exceed the typical rate implied by their starting intensity.")
    for label, sub in [("recon-verified", obs[obs["recon_basin"]]),
                       ("Dvorak-derived", obs[~obs["recon_basin"]])]:
        e = np.array([r.dv24 > dvorak_rate_ceiling(r.v0, 1.0)
                      for r in sub.itertuples()])
        print(f"  {label:<16} {100*e.mean():>5.1f}%  (n={len(sub):,})")


# ------------------------------------------------------------------- A3
def a3_transfer(d, seed, a_repeats=5):
    section("A3  cross-domain transfer between label-provenance groups")
    rec = d[d["recon_basin"]].reset_index(drop=True)
    dvo = d[~d["recon_basin"]].reset_index(drop=True)
    print(f"recon-verified : {len(rec):>7,} cases, "
          f"{100*rec['ri'].mean():.2f}% RI")
    print(f"Dvorak-derived : {len(dvo):>7,} cases, "
          f"{100*dvo['ri'].mean():.2f}% RI")
    print("\nROC AUC is the headline because it does not move with the base")
    print("rate, which differs between the two groups. PR AUC is reported for")
    print("completeness but is not comparable across them.\n")

    # Training-set size must be matched. The Dvorak-derived group has 50% more
    # cases, and "trained on more data transfers better" would otherwise be
    # indistinguishable from "learned physics rather than a convention". Every
    # model below is fitted on exactly n_train cases, sampled by storm so that
    # no storm is split, and repeated over several draws.
    n_train = int(0.8 * min(len(rec), len(dvo)))
    print(f"every model is fitted on exactly {n_train:,} cases, sampled by "
          f"storm and averaged over {a_repeats} draws\n")

    def sample_by_storm(sub, n, rng):
        sids = sub["SID"].unique()
        rng.shuffle(sids)
        sizes = sub.groupby("SID").size()
        taken, total = [], 0
        for s in sids:
            taken.append(s)
            total += int(sizes[s])
            if total >= n:
                break
        m = sub["SID"].isin(taken)
        return sub[m], sub[~m]

    rows = []
    for fname, feats in [("environment", ENV), ("environment + IR", ENV + IR)]:
        acc = {k: [] for k in (("recon", "recon"), ("recon", "Dvorak"),
                               ("Dvorak", "Dvorak"), ("Dvorak", "recon"))}
        for rep in range(a_repeats):
            rng = np.random.default_rng(seed + rep)
            for src_name, src, oth_name, oth in (("recon", rec, "Dvorak", dvo),
                                                 ("Dvorak", dvo, "recon", rec)):
                tr, held = sample_by_storm(src, n_train, rng)
                m = gbm(seed + rep).fit(tr[feats].to_numpy(float),
                                        tr["ri"].to_numpy(int))
                # Native: the part of the source domain this model never saw.
                p_native = m.predict_proba(held[feats].to_numpy(float))[:, 1]
                acc[(src_name, src_name)].append(
                    (held["ri"].to_numpy(int), p_native))
                # Foreign: all of the other domain, none of which it saw.
                p_foreign = m.predict_proba(oth[feats].to_numpy(float))[:, 1]
                acc[(src_name, oth_name)].append(
                    (oth["ri"].to_numpy(int), p_foreign))
        for (train, test), draws in acc.items():
            aucs = [roc_auc_score(y, p) for y, p in draws]
            prs = [summary(y, p, "")["PR_AUC"] for y, p in draws]
            bss = [summary(y, p, "")["BSS"] for y, p in draws]
            rows.append({
                "model": f"{fname}: {train} -> {test}",
                "n": int(np.mean([len(y) for y, _ in draws])),
                "base_rate": float(np.mean([y.mean() for y, _ in draws])),
                "ROC_AUC": float(np.mean(aucs)),
                "ROC_sd": float(np.std(aucs)),
                "PR_AUC": float(np.mean(prs)),
                "BSS": float(np.mean(bss)),
                "train": train, "test": test, "features": fname})
    res = pd.DataFrame(rows).set_index("model")
    print(res[["n", "base_rate", "ROC_AUC", "ROC_sd", "PR_AUC",
               "BSS"]].to_string(float_format=lambda v: f"{v:.4f}"))

    section("A3  transfer gaps (positive = skill lost when crossing domains)")
    print("A model that learned cyclone physics should lose little. One that")
    print("learned an analyst's convention should lose more in one direction")
    print("than the other.\n")
    for fname in ("environment", "environment + IR"):
        r = res[res["features"] == fname]
        def auc(a, b):
            return float(r[(r["train"] == a) & (r["test"] == b)]["ROC_AUC"].iloc[0])
        g_r = auc("recon", "recon") - auc("Dvorak", "recon")
        g_d = auc("Dvorak", "Dvorak") - auc("recon", "Dvorak")
        print(f"  {fname}")
        print(f"    testing on recon  : native {auc('recon','recon'):.4f}  "
              f"foreign {auc('Dvorak','recon'):.4f}  gap {g_r:+.4f}")
        print(f"    testing on Dvorak : native {auc('Dvorak','Dvorak'):.4f}  "
              f"foreign {auc('recon','Dvorak'):.4f}  gap {g_d:+.4f}")
        print(f"    asymmetry (recon-side gap minus Dvorak-side): "
              f"{g_r - g_d:+.4f}\n")
    return res


# ------------------------------------------------------------------- A2b
def a2b_ir_to_intensity(d, seed):
    """Is intensity more recoverable from imagery where it was derived from it?

    An earlier version of this test asked whether IR predictors could predict
    which intensities land on a Dvorak value. That was degenerate: the answer
    is a deterministic function of the intensity itself, so the control scored
    ROC AUC 0.999 and there was no headroom to measure anything. Discarded.

    The non-degenerate version of the same question: train a model to recover
    current intensity from IR predictors alone, and compare how well it does in
    each provenance group. If best-track intensity in the non-reconnaissance
    basins was itself read off this imagery, the imagery should recover it more
    tightly there. Physics alone gives no reason for such a gap.
    """
    section("A2b  is intensity more recoverable from imagery where it "
            "came from imagery?")
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    sub = d.dropna(subset=IR).reset_index(drop=True)
    print("Target: current intensity (kt). Features: the two SHIPS IR")
    print("predictors only -- no environment, no persistence.\n")

    rows = []
    n_match = min((sub["recon_basin"]).sum(), (~sub["recon_basin"]).sum())
    rng = np.random.default_rng(seed)
    for label, part in [("recon-verified", sub[sub["recon_basin"]]),
                        ("Dvorak-derived", sub[~sub["recon_basin"]])]:
        part = part.reset_index(drop=True)
        # Match sample sizes so that "more data fits better" cannot explain a gap.
        if len(part) > n_match:
            keep = rng.choice(len(part), n_match, replace=False)
            part = part.iloc[np.sort(keep)].reset_index(drop=True)
        X = part[IR].to_numpy(float)
        y = part["v0"].to_numpy(float)
        g = part["SEASON"].to_numpy()
        oof = np.full(len(part), np.nan)
        strat = (part["ri"].to_numpy(int))
        for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                           random_state=seed).split(X, strat, g):
            m = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, random_state=seed).fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        rows.append({"sample": label, "n": len(part),
                     "MAE_kt": mean_absolute_error(y, oof),
                     "R2": r2_score(y, oof),
                     "sd_of_target": float(np.std(y))})
    t = pd.DataFrame(rows).set_index("sample")
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    gap = t.loc["recon-verified", "MAE_kt"] - t.loc["Dvorak-derived", "MAE_kt"]
    print(f"\nintensity is recovered {gap:+.2f} kt more accurately in the")
    print("Dvorak-derived group (positive = tighter where labels came from IR)")
    print("\nCaveat: basins differ physically, so this is suggestive rather")
    print("than decisive on its own. It is the companion to A1 and A3, not a")
    print("substitute for them.")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5,
                    help="matched-size training draws for A3")
    a = ap.parse_args()

    d = load()
    section("sample")
    print(f"{len(d):,} cases, {d['SID'].nunique():,} storms, "
          f"seasons {int(d.SEASON.min())}-{int(d.SEASON.max())}, "
          f"{100*d['ri'].mean():.2f}% RI")

    q = a1_quantization(d, a.boot)
    a2_rate_convention(d)
    qp = a2b_ir_to_intensity(d, a.seed)
    res = a3_transfer(d, a.seed, a.repeats)

    REPORTS_DIR.mkdir(exist_ok=True)
    q.to_csv(REPORTS_DIR / "expA_quantization.csv")
    qp.to_csv(REPORTS_DIR / "expA_ir_to_intensity.csv")
    res.to_csv(REPORTS_DIR / "expA_transfer.csv")
    print(f"\nwrote {REPORTS_DIR / 'expA_quantization.csv'}")
    print(f"wrote {REPORTS_DIR / 'expA_ir_to_intensity.csv'}")
    print(f"wrote {REPORTS_DIR / 'expA_transfer.csv'}")


if __name__ == "__main__":
    main()
