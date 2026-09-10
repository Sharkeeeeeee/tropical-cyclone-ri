"""Why does this project's structural result disagree with McNeely et al. (2020)?

They built the ORB feature suite from GOES infrared, related it to rapid
intensification with a logistic lasso, and reported accuracy "comparable to
classifiers using environmental predictors alone", adding that "more complex
nonlinear machine learning methods did not perform better than the linear
logistic lasso model".

This project's structural features add 3.8x what the classical IR pair adds.
Three explanations are available and they are separable:

  1. Model class   -- logistic lasso against gradient boosting.
  2. Metric        -- accuracy against PR AUC and Brier skill. At a 7% base rate
                      accuracy is dominated by the negative class and can barely
                      move while the rare-event metrics move a lot.
  3. Feature set   -- deviation-angle variance is the strongest structural
                      feature here and is not part of ORB. Cannot be tested
                      without reimplementing ORB, so it is only bounded here by
                      removing DAV from our own set.

    python -u scripts/18_mcneely_check.py --tag recon_2000_2022
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.evaluate import summary  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

ENV = ["per12", "vmax", "pot", "shrd", "t200", "rhlo", "d200", "cohc"]
IR = ["ir_pct_lt_m30", "ir_std_bt_0_200"]
META = {"row", "SID", "ISO_TIME", "SEASON", "BASIN", "recon_basin",
        "v0", "dv24", "ri"}


def section(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def lasso():
    """Logistic regression with an L1 penalty -- the McNeely model class."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(penalty="l1", solver="liblinear",
                                  C=0.05, max_iter=5000)),
    ])


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

    st = pd.read_csv(PROCESSED_DIR / f"structure_{a.tag}.csv",
                     converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    ships = pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv",
                        converters={"BASIN": str}, parse_dates=["ISO_TIME"],
                        usecols=["SID", "ISO_TIME"] + ENV + IR)
    d = st.merge(ships, on=["SID", "ISO_TIME"], how="inner")
    d = d.dropna(subset=ENV).reset_index(drop=True)
    STRUCT = [c for c in st.columns if c not in META]
    NO_DAV = [c for c in STRUCT if not c.startswith("dav_")]
    y = d["ri"].to_numpy(int)
    g = d["SEASON"].to_numpy()

    section("sample")
    print(f"{len(d):,} cases, {y.sum():,} RI ({100*y.mean():.2f}%), "
          f"{d['SEASON'].nunique()} seasons")
    print(f"structure features: {len(STRUCT)}  (without DAV: {len(NO_DAV)})")
    print(f"\nAlways forecasting 'no RI' scores "
          f"{100*(1-y.mean()):.1f}% accuracy. That is the number the "
          f"comparison\nhas to beat, and it is why accuracy is a poor choice "
          f"of metric here.")

    def cv(cols, model_fn):
        X = d[cols].to_numpy(float)
        oof = np.full(len(d), np.nan)
        for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                           random_state=a.seed).split(X, y, g):
            m = model_fn()
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        return oof

    section("the same comparison under both model classes and both metrics")
    specs = [
        ("environment", ENV),
        ("environment + SHIPS IR", ENV + IR),
        ("environment + structure", ENV + STRUCT),
        ("environment + structure, no DAV", ENV + NO_DAV),
    ]
    rows = []
    for mname, mfn in (("logistic lasso", lasso), ("gradient boosting", gbm)):
        for fname, cols in specs:
            print(f"  {mname:<18} {fname:<34} ...", end="", flush=True)
            p = cv(cols, mfn)
            s = summary(y, p, f"{mname} | {fname}")
            # The metric McNeely reported, at the natural 0.5 cut.
            s["accuracy"] = accuracy_score(y, p >= 0.5)
            s["model_class"], s["features"] = mname, fname
            rows.append(s)
            print(" done")
    res = pd.DataFrame(rows).set_index("model")
    print()
    print(res[["accuracy", "ROC_AUC", "PR_AUC", "BSS"]].to_string(
        float_format=lambda v: f"{v:.4f}"))

    section("what structure adds, by model class and metric")
    print("If the disagreement is about model class, the gain should vanish for")
    print("the lasso. If it is about the metric, accuracy should stay flat while")
    print("PR AUC moves. Both can be true at once.\n")
    print(f"  {'model class':<20}{'metric':>10}{'env':>10}{'env+struct':>12}"
          f"{'gain':>9}")
    for mname in ("logistic lasso", "gradient boosting"):
        sub = res[res["model_class"] == mname]
        base = sub[sub["features"] == "environment"].iloc[0]
        full = sub[sub["features"] == "environment + structure"].iloc[0]
        for metric in ("accuracy", "PR_AUC", "BSS"):
            print(f"  {mname:<20}{metric:>10}{base[metric]:>10.4f}"
                  f"{full[metric]:>12.4f}{full[metric]-base[metric]:>+9.4f}")
        print()

    section("how much of the structural gain is deviation-angle variance?")
    for mname in ("logistic lasso", "gradient boosting"):
        sub = res[res["model_class"] == mname].set_index("features")
        base = sub.loc["environment", "PR_AUC"]
        full = sub.loc["environment + structure", "PR_AUC"]
        nodav = sub.loc["environment + structure, no DAV", "PR_AUC"]
        gain = full - base
        share = (full - nodav) / gain if abs(gain) > 1e-9 else np.nan
        print(f"  {mname:<20} gain {gain:+.4f}, of which DAV "
              f"{full-nodav:+.4f} ({100*share:.0f}%)")
    print("\nDAV is not in the ORB suite, so whatever share it carries is a")
    print("feature-set difference rather than a model or metric difference.")

    res.to_csv(REPORTS_DIR / f"mcneely_check_{a.tag}.csv")
    print(f"\nwrote {REPORTS_DIR / f'mcneely_check_{a.tag}.csv'}")


if __name__ == "__main__":
    main()
