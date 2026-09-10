"""Does hand-engineered cloud structure beat the two classical IR scalars?

This is the experiment that makes any later CNN result interpretable. The
baseline report showed the standard SHIPS IR predictors add ~0.019 PR AUC once
the environment is known. If a CNN beats that, one needs to know whether the
gain comes from spatial structure carrying information at all, or from deep
learning specifically. This run answers the first half with numpy features.

    GBM(environment)                          <- floor
    GBM(environment + SHIPS IR)               <- the classical bar
    GBM(environment + structure)              <- what structure alone buys
    GBM(environment + SHIPS IR + structure)   <- everything

Identical rows, identical stratified season-grouped folds.

    python -u scripts/11_structure_baseline.py --tag NA_2020 --group SID
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.evaluate import reliability, summary, threshold_table  # noqa: E402

ENV = ["per12", "vmax", "pot", "shrd", "t200", "rhlo", "d200", "cohc"]
IR = ["ir_pct_lt_m30", "ir_std_bt_0_200"]
META = {"row", "SID", "ISO_TIME", "SEASON", "BASIN", "recon_basin",
        "v0", "dv24", "ri"}
pd.set_option("display.width", 220)


def section(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def make_gbm(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="NA_2020")
    ap.add_argument("--group", default="SEASON")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--importance", action="store_true",
                    help="permutation importance for the structure block (slow)")
    a = ap.parse_args()

    section("data")
    st = pd.read_csv(PROCESSED_DIR / f"structure_{a.tag}.csv",
                     converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    ships = pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv",
                        converters={"BASIN": str}, parse_dates=["ISO_TIME"],
                        usecols=["SID", "ISO_TIME"] + ENV + IR)
    d = st.merge(ships, on=["SID", "ISO_TIME"], how="inner")
    d = d.dropna(subset=ENV).reset_index(drop=True)
    STRUCT = [c for c in st.columns if c not in META]
    y = d["ri"].to_numpy(dtype=int)
    print(f"cases {len(d):,}   RI {y.sum():,} ({100*y.mean():.2f}%)   "
          f"storms {d['SID'].nunique():,}")
    print(f"structure features: {len(STRUCT)}   environment: {len(ENV)}   "
          f"SHIPS IR: {len(IR)}")
    print(f"grouping folds by {a.group} ({d[a.group].nunique()} groups)")
    if y.sum() < 100:
        print("\n  NOTE: too few positives for these differences to be "
              "meaningful.\n  This run verifies the comparison, not the science.")

    groups = d[a.group].to_numpy()
    n_splits = min(a.folds, d[a.group].nunique())
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=a.seed)

    def cv(cols):
        X = d[cols].to_numpy(dtype=float)
        o = np.full(len(d), np.nan)
        for tr, te in gkf.split(X, y, groups=groups):
            m = make_gbm(a.seed)
            m.fit(X[tr], y[tr])
            o[te] = m.predict_proba(X[te])[:, 1]
        return o

    section("cross-validated comparison")
    specs = {
        "environment": ENV,
        "environment + SHIPS IR": ENV + IR,
        "environment + structure": ENV + STRUCT,
        "environment + SHIPS IR + structure": ENV + IR + STRUCT,
        "structure only": STRUCT,
    }
    preds = {"climatology": np.full(len(d), y.mean())}
    for name, cols in specs.items():
        print(f"  {name:<36s} {len(cols):>3d} features ...", end="", flush=True)
        preds[name] = cv(cols)
        print(" done")

    res = pd.DataFrame([summary(y, p, k) for k, p in preds.items()]
                       ).set_index("model")
    print()
    print(res.to_string(float_format=lambda v: f"{v:.4f}"))

    section("what each block adds on top of the environment")
    base = res.loc["environment"]
    for k in ("environment + SHIPS IR", "environment + structure",
              "environment + SHIPS IR + structure"):
        r = res.loc[k]
        print(f"  {k:<36s} ROC {r.ROC_AUC-base.ROC_AUC:+.4f}   "
              f"PR {r.PR_AUC-base.PR_AUC:+.4f}   BSS {r.BSS-base.BSS:+.4f}")
    ir_gain = res.loc["environment + SHIPS IR", "PR_AUC"] - base.PR_AUC
    st_gain = res.loc["environment + structure", "PR_AUC"] - base.PR_AUC
    print(f"\n  structure / SHIPS-IR gain ratio (PR AUC): "
          f"{st_gain/ir_gain:.2f}x" if abs(ir_gain) > 1e-9 else "")
    print("\n  If structure is worth much more than the two classical scalars,")
    print("  the imagery does carry RI-relevant information they discard, and a")
    print("  CNN has something to find. If it is worth about the same, the")
    print("  headroom is small whatever the model, and that is the finding.")

    best = "environment + SHIPS IR + structure"
    section(f"reliability: {best}")
    print(reliability(y, preds[best]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    section(f"contingency: {best}")
    print(threshold_table(y, preds[best]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    if a.importance:
        section("permutation importance, structure block (top 15)")
        X = d[ENV + IR + STRUCT].to_numpy(dtype=float)
        tr, te = next(iter(gkf.split(X, y, groups=groups)))
        m = make_gbm(a.seed).fit(X[tr], y[tr])
        pi = permutation_importance(m, X[te], y[te], n_repeats=5,
                                    random_state=a.seed, scoring="average_precision")
        imp = pd.Series(pi.importances_mean, index=ENV + IR + STRUCT)
        print(imp.sort_values(ascending=False).head(15).to_string(
            float_format=lambda v: f"{v:.4f}"))

    out = d[["SID", "ISO_TIME", "SEASON", "BASIN", "recon_basin",
             "v0", "dv24", "ri"]].copy()
    for k, p in preds.items():
        out["p_" + k.replace(" ", "_").replace("+", "plus")] = p
    p_out = PROCESSED_DIR / f"structure_oof_{a.tag}.csv"
    out.to_csv(p_out, index=False)
    res.to_csv(REPORTS_DIR / f"structure_summary_{a.tag}.csv")
    print(f"\nwrote {p_out}")
    print(f"wrote {REPORTS_DIR / f'structure_summary_{a.tag}.csv'}")


if __name__ == "__main__":
    main()
