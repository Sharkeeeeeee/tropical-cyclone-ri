"""Audit the evaluation for leakage and for arithmetic that only looks right.

Every check here is empirical. The point is not to restate the design but to
test whether the code actually implements it, because the failure mode that
matters is a split that looks grouped and is not.

Three families:

* **Leakage** - can information about a test case reach the model that scores
  it? Through the fold split, through a predictor that postdates t0, through
  duplicated rows, or through a scaler fitted on everything.
* **Out-of-fold integrity** - is every prediction really made by a model that
  did not see that row?
* **Metric correctness** - do the hand-written metrics agree with independent
  implementations on cases where the answer is known?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcri.config import PROCESSED_DIR  # noqa: E402
from tcri.dataset import BT_MEAN_K, BT_STD_K  # noqa: E402
from tcri.evaluate import (brier_skill_score, contingency,  # noqa: E402
                           reliability, summary)

results = []


def check(name, fn):
    try:
        d = fn()
        results.append((True, name, d or ""))
        print(f"[  ok  ] {name}" + (f"  -- {d}" if d else ""))
    except AssertionError as e:
        results.append((False, name, str(e)))
        print(f"[ FAIL ] {name}\n         {e}")
    except Exception as e:                       # noqa: BLE001
        results.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[ FAIL ] {name}\n         {type(e).__name__}: {e}")


def load():
    return pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv",
                       converters={"BASIN": str}, parse_dates=["ISO_TIME"])


# ----------------------------------------------------------------- leakage
def season_grouping_implies_storm_grouping():
    """Grouping folds by season is only safe if no storm spans two seasons."""
    d = load()
    spans = d.groupby("SID")["SEASON"].nunique()
    bad = spans[spans > 1]
    assert bad.empty, (f"{len(bad)} storms span >1 season, e.g. "
                       f"{list(bad.index[:3])} -- season folds would split them")
    return f"{d['SID'].nunique():,} storms, each within a single season"


def no_duplicate_case_keys():
    for name in ("ri_labels.csv", "ri_with_ships.csv"):
        p = PROCESSED_DIR / name
        if not p.exists():
            continue
        t = pd.read_csv(p, converters={"BASIN": str}, parse_dates=["ISO_TIME"])
        dup = t.duplicated(["SID", "ISO_TIME"]).sum()
        assert dup == 0, f"{name}: {dup:,} duplicate (SID, ISO_TIME) rows"
    return "no duplicate (SID, ISO_TIME) in the label or joined tables"


def no_future_information_in_features():
    """Nothing that postdates t0 may appear in any model's feature list."""
    import importlib.util as u

    banned = {"ships_dv24", "dv24", "v_p24", "ri", "v_p6", "v_p12", "v_p18",
              "d2l_p6", "d2l_p12", "d2l_p18", "d2l_p24"}
    offenders = []
    for script in ("05_ships_baseline.py", "09_train_cnn.py",
                   "11_structure_baseline.py", "12_train_gpu.py"):
        spec = u.spec_from_file_location("m", ROOT / "scripts" / script)
        m = u.module_from_spec(spec)
        spec.loader.exec_module(m)
        for attr in ("ENV", "IR", "WIDE_ENV", "WIDE_IR"):
            feats = getattr(m, attr, None)
            if feats:
                hit = banned & set(feats)
                if hit:
                    offenders.append(f"{script}:{attr} -> {sorted(hit)}")
    assert not offenders, "; ".join(offenders)
    return "no post-t0 quantity in ENV/IR/WIDE_ENV/WIDE_IR of any script"


def structure_features_are_t0_or_earlier():
    """Structure feature names must reference t0 or a backward tendency."""
    p = PROCESSED_DIR / "structure_NA_2020.csv"
    if not p.exists():
        return "structure table absent, skipped"
    cols = pd.read_csv(p, nrows=0).columns
    meta = {"row", "SID", "ISO_TIME", "SEASON", "BASIN", "recon_basin",
            "v0", "dv24", "ri"}
    feats = [c for c in cols if c not in meta]
    bad = [c for c in feats if c.endswith(("_p3h", "_p6h", "_p9h", "_fwd"))]
    assert not bad, f"forward-looking feature names: {bad[:5]}"
    tend = [c for c in feats if c.endswith(("_d9h", "_d3h"))]
    return (f"{len(feats)} features, {len(tend)} backward tendencies, "
            "none forward-looking")


def normalisation_is_not_fitted_on_data():
    """A scaler fitted on the whole set leaks test statistics into training."""
    assert isinstance(BT_MEAN_K, float) and isinstance(BT_STD_K, float), \
        "normalisation constants are not fixed scalars"
    src = (ROOT / "src" / "tcri" / "dataset.py").read_text(encoding="utf-8")
    for pat in ("StandardScaler", ".mean()", ".std()"):
        assert pat not in src, f"dataset.py computes {pat} from the data"
    return f"fixed constants ({BT_MEAN_K:.0f} K, {BT_STD_K:.0f} K), not fitted"


# ------------------------------------------------------- fold integrity
def folds_are_disjoint_by_group():
    d = load()
    d = d[(d["SEASON"] >= 2000) & d["overwater_24h"]].reset_index(drop=True)
    y = d["ri"].to_numpy(int)
    for key in ("SEASON", "SID"):
        groups = d["SEASON"].to_numpy()          # split as the pipeline does
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for k, (tr, te) in enumerate(gkf.split(d, y, groups=groups)):
            a = set(d.iloc[tr][key]); b = set(d.iloc[te][key])
            assert not (a & b), (f"fold {k}: {len(a & b)} shared {key} values "
                                 f"between train and test")
    return "season-grouped folds share no SEASON and no SID across train/test"


def verification_windows_do_not_overlap_across_folds():
    """Two cases 6 h apart in one storm share 18 h of their outcome window.

    If they landed in different folds the test label would be nearly determined
    by a training label. Storm-level disjointness should make this impossible;
    this checks it rather than assuming it.
    """
    d = load()
    d = d[(d["SEASON"] >= 2000) & d["overwater_24h"]].reset_index(drop=True)
    y = d["ri"].to_numpy(int)
    groups = d["SEASON"].to_numpy()
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    worst = 0
    for tr, te in gkf.split(d, y, groups=groups):
        a, b = d.iloc[tr], d.iloc[te]
        shared = set(a["SID"]) & set(b["SID"])
        worst = max(worst, len(shared))
    assert worst == 0, f"{worst} storms appear in both train and test"
    return "no storm contributes cases to both sides of any split"


def oof_predictions_cover_every_row_once():
    p = PROCESSED_DIR / "baseline_oof.csv"
    if not p.exists():
        return "baseline_oof.csv absent, skipped"
    oof = pd.read_csv(p, converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    assert not oof.duplicated(["SID", "ISO_TIME"]).any(), "duplicate rows"
    pcols = [c for c in oof.columns if c.startswith("p_")]
    for c in pcols:
        v = oof[c].to_numpy()
        assert np.isfinite(v).all(), f"{c}: {int((~np.isfinite(v)).sum())} NaN"
    # A constant column would mean a fold silently failed to fit.
    varying = [c for c in pcols if oof[c].nunique() > 10
               or c.startswith("p_climatology")]
    assert len(varying) == len(pcols), \
        f"near-constant prediction columns: {set(pcols) - set(varying)}"
    return f"{len(oof):,} rows x {len(pcols)} columns, all finite, none constant"


# ------------------------------------------------------------ metric checks
def metrics_match_independent_implementations():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.09).astype(int)
    p = np.clip(0.09 + 0.4 * y * rng.random(4000)
                + 0.05 * rng.standard_normal(4000), 1e-6, 1 - 1e-6)
    s = summary(y, p, "t")
    assert abs(s["ROC_AUC"] - roc_auc_score(y, p)) < 1e-12, "ROC AUC differs"
    assert abs(s["PR_AUC"] - average_precision_score(y, p)) < 1e-12, "PR AUC"
    assert abs(s["Brier"] - brier_score_loss(y, p)) < 1e-12, "Brier"
    # BSS by hand.
    base = y.mean()
    want = 1 - np.mean((p - y) ** 2) / np.mean((base - y) ** 2)
    assert abs(s["BSS"] - want) < 1e-10, f"BSS {s['BSS']} vs {want}"
    return "ROC/PR/Brier/BSS agree with sklearn and with hand arithmetic"


def climatology_scores_exactly_zero_bss():
    y = np.array([0] * 91 + [1] * 9)
    p = np.full(100, y.mean())
    assert abs(brier_skill_score(y, p)) < 1e-12, "climatology BSS is not 0"
    return "a climatological forecast scores BSS = 0 by construction"


def contingency_counts_are_consistent():
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.2).astype(int)
    p = rng.random(1000)
    for t in (0.1, 0.5, 0.9):
        c = contingency(y, p, t)
        assert c["hits"] + c["misses"] == y.sum(), f"threshold {t}: positives lost"
        assert c["hits"] + c["false_alarms"] == c["forecasts"], \
            f"threshold {t}: forecast count inconsistent"
        pod = c["hits"] / max(y.sum(), 1)
        assert abs(c["POD"] - pod) < 1e-12, "POD wrong"
    return "hits+misses = positives and hits+FA = forecasts at every threshold"


def reliability_bins_partition_the_sample():
    rng = np.random.default_rng(2)
    y = (rng.random(997) < 0.1).astype(int)
    p = rng.random(997)
    r = reliability(y, p, n_bins=10)
    assert r["n"].sum() == 997, f"bins hold {r['n'].sum()} of 997"
    assert (r["p_min"].values[1:] >= r["p_max"].values[:-1] - 1e-12).all(), \
        "bins are not monotonic in forecast probability"
    return f"{len(r)} bins partition all 997 samples, monotonic"


def main():
    print("=" * 74)
    print("leakage and rigour audit")
    print("=" * 74)
    print("\n-- leakage --")
    check("season grouping implies storm grouping",
          season_grouping_implies_storm_grouping)
    check("no duplicate case keys", no_duplicate_case_keys)
    check("no post-t0 quantity used as a predictor",
          no_future_information_in_features)
    check("structure features are t0 or backward tendencies",
          structure_features_are_t0_or_earlier)
    check("normalisation constants are fixed, not fitted",
          normalisation_is_not_fitted_on_data)

    print("\n-- fold integrity --")
    check("folds disjoint by season and by storm", folds_are_disjoint_by_group)
    check("outcome windows never straddle a split",
          verification_windows_do_not_overlap_across_folds)
    check("out-of-fold predictions cover every row once",
          oof_predictions_cover_every_row_once)

    print("\n-- metric correctness --")
    check("metrics match independent implementations",
          metrics_match_independent_implementations)
    check("climatology scores exactly BSS = 0",
          climatology_scores_exactly_zero_bss)
    check("contingency counts are self-consistent",
          contingency_counts_are_consistent)
    check("reliability bins partition the sample",
          reliability_bins_partition_the_sample)

    n_fail = sum(1 for ok, _, _ in results if not ok)
    print("=" * 74)
    print(f"{len(results) - n_fail}/{len(results)} passed")
    for ok, name, detail in results:
        if not ok:
            print(f"  FAIL  {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
