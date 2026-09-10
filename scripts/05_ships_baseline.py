"""Reproduce a SHIPS-style statistical RI baseline. This is the number to beat.

Five forecasts are scored, and the interesting comparison is not model-vs-model
but the gap between the pairs with and without the GOES IR predictors. That gap
is exactly what the vision model is trying to improve on: how much the standard
hand-engineered infrared features add once the environment is already known.

Splits are grouped by season, so no storm and no season appears in both training
and evaluation. Out-of-fold predictions are written to disk so that the vision
model can later be scored on precisely the same cases.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.evaluate import reliability, summary, threshold_table  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)

# Kaplan & DeMaria SHIPS-RII predictors, minus the IR pair.
ENV = ["per12", "vmax", "pot", "shrd", "t200", "rhlo", "d200", "cohc"]
IR = ["ir_pct_lt_m30", "ir_std_bt_0_200"]

# Wider set for the gradient-boosted models. Anything downstream of t=0 is
# excluded: only analysis-time quantities may appear here.
WIDE_ENV = ENV + [
    "lat", "lon", "mslp", "csst", "cd20", "cd26", "dtl", "oage", "nage",
    "rsst", "dsst", "u200", "u20c", "v20c", "e000", "epos", "eneg", "epss",
    "enss", "rhmd", "rhhi", "z850", "refc", "pefc", "t000", "r000", "z000",
    "twac", "twxc", "g150", "g200", "g250", "v000", "v850", "v500", "v300",
    "tgrd", "tadv", "penc", "shdc", "sddc", "shgc", "divc", "t150", "t250",
    "shrs", "shrg", "penv", "vmpi", "vvav", "vmfx", "vvac", "he07", "he05",
    "o500", "o700", "cflx", "hist_20kt", "incv_m6",
    "mtpw_01", "mtpw_02", "mtpw_19", "mtpw_21",
]
WIDE_IR = IR + [
    "ir_avg_bt_0_200", "ir_avg_bt_100_300", "ir_std_bt_100_300",
    "ir_pct_lt_m10", "ir_pct_lt_m20", "ir_pct_lt_m40", "ir_pct_lt_m50",
    "ir_pct_lt_m60", "ir_max_bt_0_30", "ir_avg_bt_0_30", "ir_r_max_bt",
    "ir_min_bt_20_120", "ir_avg_bt_20_120", "ir_r_min_bt",
    "m1_ir_pct_lt_m30", "m1_ir_std_bt_0_200", "m1_ir_avg_bt_0_200",
    "m3_ir_pct_lt_m30", "m3_ir_std_bt_0_200", "m3_ir_avg_bt_0_200",
]

N_FOLDS = 5


def section(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def make_lr():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, class_weight=None)),
    ])


def make_gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)


def run_cv(df, feats, model_fn, groups):
    """Season-grouped out-of-fold probabilities."""
    X = df[feats].to_numpy(dtype=float)
    y = df["ri"].to_numpy(dtype=int)
    oof = np.full(len(df), np.nan)
    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=0)
    for tr, te in gkf.split(X, y, groups=groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    section("load joined table")
    df = pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv", converters={"BASIN": str, "SUBBASIN": str}, parse_dates=["ISO_TIME"])
    print(f"joined rows: {len(df):,}")

    # Evaluation sample: satellite era, over water for the whole window, and
    # dropping the final partial season, which IBTrACS has not finalised.
    last_season = int(df["SEASON"].max())
    d = df[(df["SEASON"] >= 2000) & (df["overwater_24h"])
           & (df["SEASON"] < last_season)].copy()
    d = d.dropna(subset=ENV)          # environmental core must be present
    d = d.reset_index(drop=True)
    print(f"evaluation sample: {len(d):,} cases, "
          f"{int(d['ri'].sum()):,} RI ({100*d['ri'].mean():.2f}%)")
    print(f"seasons {int(d['SEASON'].min())}-{int(d['SEASON'].max())}, "
          f"{d['SID'].nunique():,} storms")
    print(f"IR predictors present on {100*d['ir_pct_lt_m30'].notna().mean():.1f}% "
          f"of these cases")

    groups = d["SEASON"].to_numpy()
    wide_env = [c for c in WIDE_ENV if c in d.columns]
    wide_all = [c for c in WIDE_ENV + WIDE_IR if c in d.columns]

    section("fit (season-grouped 5-fold cross-validation)")
    specs = [
        ("LR  env only",        ENV,          make_lr),
        ("LR  env + IR (RII)",  ENV + IR,     make_lr),
        ("GBM env only",        wide_env,     make_gbm),
        ("GBM env + IR",        wide_all,     make_gbm),
    ]
    preds = {}
    for name, feats, fn in specs:
        print(f"  {name:<22s} {len(feats):>3d} features ...", end="", flush=True)
        preds[name] = run_cv(d, feats, fn, groups)
        print(" done")

    base = float(d["ri"].mean())
    preds["climatology"] = np.full(len(d), base)

    section("skill  (out-of-fold, all basins)")
    order = ["climatology", "LR  env only", "LR  env + IR (RII)",
             "GBM env only", "GBM env + IR"]
    rows = [summary(d["ri"], preds[k], k) for k in order]
    res = pd.DataFrame(rows).set_index("model")
    print(res.to_string(float_format=lambda v: f"{v:.4f}"))

    section("what the standard IR predictors add")
    for a, b in [("LR  env only", "LR  env + IR (RII)"),
                 ("GBM env only", "GBM env + IR")]:
        d_auc = res.loc[b, "ROC_AUC"] - res.loc[a, "ROC_AUC"]
        d_pr = res.loc[b, "PR_AUC"] - res.loc[a, "PR_AUC"]
        d_bss = res.loc[b, "BSS"] - res.loc[a, "BSS"]
        print(f"  {a:<20s} -> {b:<22s}  "
              f"ROC_AUC {d_auc:+.4f}   PR_AUC {d_pr:+.4f}   BSS {d_bss:+.4f}")
    print("\nThis is the margin the vision model has to improve on: the IR channel's"
          "\ncontribution once the large-scale environment is already accounted for.")

    best = max(order[1:], key=lambda k: res.loc[k, "PR_AUC"])
    section(f"reliability of the best baseline: {best}")
    print(reliability(d["ri"], preds[best]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    section(f"contingency table: {best}")
    print(threshold_table(d["ri"], preds[best]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    section(f"by basin: {best}")
    d["_p"] = preds[best]
    rows = []
    for b, sub in d.groupby("BASIN"):
        if sub["ri"].sum() < 20:
            continue
        s = summary(sub["ri"], sub["_p"], b)
        s["recon"] = "yes" if b in ("NA", "EP") else "no"
        rows.append(s)
    print(pd.DataFrame(rows).set_index("model").to_string(
        float_format=lambda v: f"{v:.4f}"))

    section(f"recon-verified vs Dvorak-derived: {best}")
    rows = []
    for k, sub in d.groupby("recon_basin"):
        rows.append(summary(sub["ri"], sub["_p"],
                            "recon-verified (NA/EP)" if k else "Dvorak-derived"))
    print(pd.DataFrame(rows).set_index("model").to_string(
        float_format=lambda v: f"{v:.4f}"))

    section("write")
    out = d[["SID", "USA_ATCF_ID", "ISO_TIME", "SEASON", "BASIN",
             "recon_basin", "v0", "dv24", "ri"]].copy()
    for k in order:
        out["p_" + k.replace(" ", "_").replace("+", "plus")
            .replace("(", "").replace(")", "")] = preds[k]
    p = PROCESSED_DIR / "baseline_oof.csv"
    out.to_csv(p, index=False)
    print(f"  out-of-fold predictions -> {p}")
    rp = REPORTS_DIR / "baseline_summary.csv"
    res.to_csv(rp)
    print(f"  skill summary           -> {rp}")


if __name__ == "__main__":
    main()
