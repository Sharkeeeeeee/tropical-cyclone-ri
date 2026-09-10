"""Does a learned representation beat hand-engineered cloud structure?

This is the question that decides how much deep learning is worth on this
problem, and it is not the same as "does imagery beat the environment" — the
structural features already answered that with a yes.

Two comparisons, and the distinction between them matters:

**Clean.** `CNN alone` against `structure only`. Both are honest out-of-fold
predictions from models that never saw the case they score, built on the same
rows with the same season-grouped folds. Nothing is stacked, so nothing leaks.
This is the headline.

**Stacked, and optimistic.** Feeding the CNN's out-of-fold probability to a
gradient-boosted model alongside the environment. A training row's CNN value
came from a network fitted on folds that overlap the booster's training set, so
the booster sees a feature that is slightly too good. Nested cross-validation
removes this and costs about five times the compute; it is budgeted for and not
done here. Every stacked number below is an upper bound.

    python -u scripts/19_final_comparison.py --tag recon_2000_2022
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.evaluate import reliability, summary, threshold_table  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

ENV = ["per12", "vmax", "pot", "shrd", "t200", "rhlo", "d200", "cohc"]
IR = ["ir_pct_lt_m30", "ir_std_bt_0_200"]
META = {"row", "SID", "ISO_TIME", "SEASON", "BASIN", "recon_basin",
        "v0", "dv24", "ri"}


def section(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def gbm(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="recon_2000_2022")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    section("load")
    st = pd.read_csv(PROCESSED_DIR / f"structure_{a.tag}.csv",
                     converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    sof = pd.read_csv(PROCESSED_DIR / f"structure_oof_{a.tag}.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    cnn_p = PROCESSED_DIR / f"cnn_oof_{a.tag}_seed{a.seed}.csv"
    if not cnn_p.exists():
        raise SystemExit(f"{cnn_p.name} not found — run 12_train_gpu.py first")
    cnn = pd.read_csv(cnn_p, converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    ships = pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv",
                        converters={"BASIN": str}, parse_dates=["ISO_TIME"],
                        usecols=["SID", "ISO_TIME"] + ENV + IR)

    emb = [c for c in cnn.columns if c.startswith("emb")]
    d = (st.merge(ships, on=["SID", "ISO_TIME"])
           .merge(cnn[["SID", "ISO_TIME", "p_cnn"] + emb], on=["SID", "ISO_TIME"])
           .merge(sof[["SID", "ISO_TIME", "p_environment",
                       "p_environment_plus_SHIPS_IR",
                       "p_environment_plus_structure", "p_structure_only"]],
                  on=["SID", "ISO_TIME"]))
    d = d.dropna(subset=ENV + ["p_cnn"]).reset_index(drop=True)
    STRUCT = [c for c in st.columns if c not in META]
    y = d["ri"].to_numpy(int)
    g = d["SEASON"].to_numpy()
    print(f"{len(d):,} cases on all three sources, {y.sum():,} RI "
          f"({100*y.mean():.2f}%), {d['SEASON'].nunique()} seasons")
    print(f"CNN embedding: {len(emb)} dimensions")

    section("clean comparison — no stacking, nothing leaks")
    print("Both are out-of-fold predictions from models that never saw the case")
    print("they score. This is the honest answer to whether a learned")
    print("representation reads more out of the imagery than the hand-built one.\n")
    clean = pd.DataFrame([
        summary(y, np.full(len(y), y.mean()), "climatology"),
        summary(y, d["p_structure_only"].to_numpy(), "structure only (141 features)"),
        summary(y, d["p_cnn"].to_numpy(), "CNN only (391k parameters)"),
    ]).set_index("model")
    print(clean[["n", "positives", "base_rate", "ROC_AUC", "PR_AUC",
                 "BSS"]].to_string(float_format=lambda v: f"{v:.4f}"))
    s_pr = clean.loc["structure only (141 features)", "PR_AUC"]
    c_pr = clean.loc["CNN only (391k parameters)", "PR_AUC"]
    print(f"\n  CNN minus structure, PR AUC: {c_pr - s_pr:+.4f}")
    if c_pr > s_pr:
        print("  The learned representation reads more out of the same pixels.")
    else:
        print("  The hand-engineered features read at least as much out of the")
        print("  same pixels as the network does. At ~880 positives that is the")
        print("  outcome a small-data problem would predict, and it bears")
        print("  directly on how much GPU time this problem deserves.")

    section("stacked on the environment — upper bounds, see the caveat above")
    def cv(cols):
        X = d[cols].to_numpy(float)
        oof = np.full(len(d), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                           random_state=a.seed).split(X, y, g):
            m = gbm(a.seed).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        return oof

    preds = {
        "environment": d["p_environment"].to_numpy(),
        "environment + SHIPS IR": d["p_environment_plus_SHIPS_IR"].to_numpy(),
        "environment + structure": d["p_environment_plus_structure"].to_numpy(),
    }
    for name, cols in [("environment + CNN probability", ENV + ["p_cnn"]),
                       ("environment + CNN embedding", ENV + emb),
                       ("environment + structure + CNN", ENV + STRUCT + ["p_cnn"])]:
        print(f"  fitting {name} ...", end="", flush=True)
        preds[name] = cv(cols)
        print(" done")

    res = pd.DataFrame([summary(y, p, k) for k, p in preds.items()]
                       ).set_index("model")
    print()
    print(res[["ROC_AUC", "PR_AUC", "BSS"]].to_string(
        float_format=lambda v: f"{v:.4f}"))

    section("what each imagery representation adds over the environment")
    base = res.loc["environment"]
    for k in res.index:
        if k == "environment":
            continue
        r = res.loc[k]
        print(f"  {k:<32} ROC {r.ROC_AUC-base.ROC_AUC:+.4f}   "
              f"PR {r.PR_AUC-base.PR_AUC:+.4f}   BSS {r.BSS-base.BSS:+.4f}")

    section("reliability of the best model")
    best = res["PR_AUC"].idxmax()
    print(f"{best}\n")
    print(reliability(y, preds[best]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    section(f"contingency: {best}")
    print(threshold_table(y, preds[best]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    out = d[["SID", "ISO_TIME", "SEASON", "BASIN", "v0", "dv24", "ri"]].copy()
    for k, p in preds.items():
        out["p_" + k.replace(" ", "_").replace("+", "plus")] = p
    out["p_structure_only"] = d["p_structure_only"]
    out["p_cnn_only"] = d["p_cnn"]
    out.to_csv(PROCESSED_DIR / f"final_oof_{a.tag}.csv", index=False)
    pd.concat([clean, res]).to_csv(REPORTS_DIR / f"final_comparison_{a.tag}.csv")
    print(f"\nwrote {PROCESSED_DIR / f'final_oof_{a.tag}.csv'}")
    print(f"wrote {REPORTS_DIR / f'final_comparison_{a.tag}.csv'}")


if __name__ == "__main__":
    main()
