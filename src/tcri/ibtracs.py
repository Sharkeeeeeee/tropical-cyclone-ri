"""IBTrACS loading and 24 h rapid-intensification labels.

Three decisions worth knowing:

Intensity is USA_WIND, not WMO_WIND. WMO_WIND mixes 1-minute and 10-minute
averaging across agencies, so "30 kt in 24 h" would mean different things in
different basins.

Only synoptic hours 00/06/12/18 UTC are kept. IBTrACS carries 3-hourly rows but
off-synoptic intensity is interpolated, so including it inflates the sample
without adding information.

Samples are tropical systems (NATURE == 'TS') over water at depression strength
or above, following Kaplan & DeMaria. Extratropical stages are a different
problem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    IBTRACS_CSV,
    MIN_DIST2LAND_KM,
    MIN_INTENSITY_KT,
    RECON_BASINS,
    RI_THRESHOLD_KT,
    RI_WINDOW_H,
    SYNOPTIC_HOURS,
    TROPICAL_NATURES,
)

USECOLS = [
    "SID", "SEASON", "BASIN", "SUBBASIN", "NAME", "ISO_TIME", "NATURE",
    "USA_ATCF_ID",
    "LAT", "LON", "WMO_WIND", "TRACK_TYPE", "DIST2LAND", "LANDFALL",
    "USA_AGENCY", "USA_WIND", "USA_STATUS", "USA_SSHS",
]
NUMERIC = ["SEASON", "LAT", "LON", "WMO_WIND", "DIST2LAND", "LANDFALL",
           "USA_WIND", "USA_SSHS"]


def load_raw(path=IBTRACS_CSV) -> pd.DataFrame:
    """Read the IBTrACS CSV. Row 0 is the header, row 1 is a units row."""
    df = pd.read_csv(path, skiprows=[1], usecols=USECOLS, dtype=str,
                     keep_default_na=False, na_values=[""])
    for c in df.columns:
        df[c] = df[c].str.strip()
    df = df.replace({"": np.nan})
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], format="%Y-%m-%d %H:%M:%S",
                                    errors="coerce")
    return df


def to_synoptic(df: pd.DataFrame) -> pd.DataFrame:
    """Keep main tracks at synoptic hours, one row per (storm, time)."""
    df = df[df["TRACK_TYPE"] == "main"]
    df = df[df["ISO_TIME"].notna()]
    df = df[df["ISO_TIME"].dt.hour.isin(SYNOPTIC_HOURS)]
    df = df[df["ISO_TIME"].dt.minute.eq(0)]
    df = df.sort_values(["SID", "ISO_TIME"])
    df = df.drop_duplicates(subset=["SID", "ISO_TIME"], keep="first")
    return df.reset_index(drop=True)


def build_ri_samples(df: pd.DataFrame,
                     threshold_kt: float = RI_THRESHOLD_KT,
                     window_h: int = RI_WINDOW_H) -> pd.DataFrame:
    """One row per candidate forecast case, with the 24 h intensity change.

    A case is valid only if the same storm has a best-track fix at t0 and at
    t0 + window_h with a defined USA_WIND. Intermediate 6-hourly fixes are
    joined too, so that "stayed over water for the whole window" can be
    evaluated rather than assumed.
    """
    base = df.copy()
    key = ["SID", "ISO_TIME"]
    lookup = df[key + ["USA_WIND", "DIST2LAND", "NATURE", "LAT", "LON"]]

    # Join the future fixes at +6, +12, +18, +24 h.
    for lag in range(6, window_h + 1, 6):
        right = lookup.rename(columns={
            "ISO_TIME": "_t",
            "USA_WIND": f"v_p{lag}",
            "DIST2LAND": f"d2l_p{lag}",
            "NATURE": f"nature_p{lag}",
            "LAT": f"lat_p{lag}",
            "LON": f"lon_p{lag}",
        })
        base["_t"] = base["ISO_TIME"] + pd.Timedelta(hours=lag)
        base = base.merge(right, on=["SID", "_t"], how="left")
        base = base.drop(columns="_t")

    base = base.rename(columns={"USA_WIND": "v0", "DIST2LAND": "d2l_0"})
    base["dv24"] = base[f"v_p{window_h}"] - base["v0"]
    base["ri"] = (base["dv24"] >= threshold_kt).astype("Int64")
    base.loc[base["dv24"].isna(), "ri"] = pd.NA

    # Eligibility at t0.
    base["elig_t0"] = (
        base["NATURE"].isin(TROPICAL_NATURES)
        & base["v0"].notna()
        & (base["v0"] >= MIN_INTENSITY_KT)
        & base["d2l_0"].notna()
        & (base["d2l_0"] > MIN_DIST2LAND_KM)
    )
    # Verifiable: we actually know the answer at t0 + 24 h.
    base["verifiable"] = base[f"v_p{window_h}"].notna()

    # Over water for the entire window. Kept as a separate flag rather than
    # folded into eligibility, because land interaction is a modelling choice.
    d2l_cols = ["d2l_0"] + [f"d2l_p{l}" for l in range(6, window_h + 1, 6)]
    base["overwater_24h"] = (base[d2l_cols] > MIN_DIST2LAND_KM).all(axis=1)

    base["recon_basin"] = base["BASIN"].isin(RECON_BASINS)
    base["sample"] = base["elig_t0"] & base["verifiable"]
    return base


def label_episodes(samples: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive RI-positive fixes within a storm into single episodes.

    This is the number that governs how much the model can actually learn from:
    six-hourly RI cases inside one intensification burst are near-duplicates of
    each other, not independent observations.
    """
    pos = samples[(samples["sample"]) & (samples["ri"] == 1)].copy()
    pos = pos.sort_values(["SID", "ISO_TIME"])
    gap = pos.groupby("SID")["ISO_TIME"].diff()
    new_ep = (gap.isna()) | (gap > pd.Timedelta(hours=6))
    pos["episode_id"] = new_ep.cumsum()
    return pos
