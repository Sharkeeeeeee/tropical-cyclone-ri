"""Metrics for a rare-event probabilistic forecast.

Nothing here reports accuracy: at a 9% base rate, always forecasting "no RI"
scores 91% and is useless. What matters is discrimination (ROC and PR AUC),
calibration (reliability, Brier), and skill against the climatological rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)


def brier_skill_score(y, p, base_rate=None) -> float:
    """1 - BS/BS_climatology. Zero means no better than forecasting the base rate."""
    base = float(np.mean(y)) if base_rate is None else base_rate
    bs = brier_score_loss(y, p)
    bs_clim = brier_score_loss(y, np.full_like(np.asarray(p, dtype=float), base))
    return 1.0 - bs / bs_clim if bs_clim > 0 else np.nan


def contingency(y, p, threshold: float) -> dict:
    yhat = np.asarray(p) >= threshold
    y = np.asarray(y).astype(bool)
    hits = int((yhat & y).sum())
    misses = int((~yhat & y).sum())
    fa = int((yhat & ~y).sum())
    pod = hits / (hits + misses) if hits + misses else np.nan
    far = fa / (hits + fa) if hits + fa else np.nan
    csi = hits / (hits + misses + fa) if hits + misses + fa else np.nan
    # Peirce skill score: POD minus probability of false detection.
    cn = int((~yhat & ~y).sum())
    pofd = fa / (fa + cn) if fa + cn else np.nan
    return {"threshold": threshold, "forecasts": int(yhat.sum()), "hits": hits,
            "misses": misses, "false_alarms": fa, "POD": pod, "FAR": far,
            "CSI": csi, "PSS": pod - pofd}


def summary(y, p, name: str = "") -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        "model": name,
        "n": len(y),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "BSS": brier_skill_score(y, p),
    }


def reliability(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins: with a rare event, equal-width bins leave most empty."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        rows.append({"bin": i + 1, "n": len(idx),
                     "p_min": p[idx].min(), "p_max": p[idx].max(),
                     "mean_forecast": p[idx].mean(),
                     "observed_freq": y[idx].mean()})
    return pd.DataFrame(rows)


def threshold_table(y, p, thresholds=(0.10, 0.15, 0.20, 0.30, 0.40, 0.50)):
    return pd.DataFrame([contingency(y, p, t) for t in thresholds])
