"""Data-free checks of the pieces every result depends on.

None of these touch the network or data/, so they run in CI. They pin down
the RI definition, episode counting, the metrics, and the patch geometry --
the places where a library upgrade (pandas 3 changing datetime resolution,
for one) can silently change numbers without raising.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri import evaluate, gridsat, ibtracs, provenance  # noqa: E402


def synthetic_track(winds, sid="2005001N10100", basin="NA", d2l=500.0):
    t = pd.date_range("2005-08-26 00:00", periods=len(winds), freq="6h")
    return pd.DataFrame({
        "SID": sid, "SEASON": 2005, "BASIN": basin, "NAME": "TEST",
        "ISO_TIME": t, "NATURE": "TS", "LAT": 20.0, "LON": -60.0,
        "USA_WIND": np.asarray(winds, dtype=float), "DIST2LAND": d2l,
    })


# ------------------------------------------------------------- RI labels
def test_ri_label_uses_24h_change_and_threshold():
    #           00  06  12  18  00  06  12  18  00
    winds = [30, 35, 40, 50, 60, 65, 70, 75, 80]
    s = ibtracs.build_ri_samples(synthetic_track(winds)).set_index("ISO_TIME")
    t0 = pd.Timestamp("2005-08-26 00:00")
    assert s.loc[t0, "dv24"] == 30 and s.loc[t0, "ri"] == 1        # exactly 30
    t1 = pd.Timestamp("2005-08-26 06:00")
    assert s.loc[t1, "dv24"] == 30 and s.loc[t1, "ri"] == 1
    t3 = pd.Timestamp("2005-08-26 18:00")
    assert s.loc[t3, "dv24"] == 25 and s.loc[t3, "ri"] == 0
    # The last 24 h have no verifying fix: unknown, not negative.
    tail = s.iloc[-4:]
    assert tail["ri"].isna().all() and not tail["verifiable"].any()


def test_land_and_weak_cases_are_flagged():
    s = ibtracs.build_ri_samples(synthetic_track([20, 30, 40, 50, 60], d2l=0))
    assert not s["elig_t0"].any()           # over land (DIST2LAND == 0)
    assert not s["overwater_24h"].any()


def test_episodes_merge_consecutive_positives():
    winds = [30, 35, 40, 50, 60, 65, 70, 75, 80]
    s = ibtracs.build_ri_samples(synthetic_track(winds))
    ep = ibtracs.label_episodes(s)
    assert len(ep) == 3                      # 00, 06, 12 UTC are RI ...
    assert ep["episode_id"].nunique() == 1   # ... but one burst


def test_synoptic_filter_drops_offhour_and_duplicates():
    df = synthetic_track([30, 40, 50])
    extra = df.iloc[[0]].copy()
    extra["ISO_TIME"] = pd.Timestamp("2005-08-26 03:00")
    df = pd.concat([df, extra, df.iloc[[1]]])
    df["TRACK_TYPE"] = "main"
    out = ibtracs.to_synoptic(df)
    assert len(out) == 3 and (out["ISO_TIME"].dt.hour % 6 == 0).all()


# ------------------------------------------------------------- metrics
def test_metrics_on_known_cases():
    y = np.array([0] * 90 + [1] * 10)
    perfect = evaluate.summary(y, y.astype(float))
    assert perfect["ROC_AUC"] == 1 and perfect["PR_AUC"] == 1
    clim = evaluate.summary(y, np.full(100, 0.1))
    assert clim["BSS"] == pytest.approx(0.0, abs=1e-12)
    assert clim["PR_AUC"] == pytest.approx(0.1)


# ------------------------------------------------------------- provenance
def test_dvorak_table_values():
    assert list(provenance.is_dvorak_value([65, 60, 90, 100, 115])) == \
        [True, False, True, True, True]
    # Typical +1 T/day never reaches the 30 kt RI threshold (Experiment A2).
    for kt in (35, 45, 55, 65, 77, 90, 102, 115):
        assert provenance.dvorak_rate_ceiling(kt) < 30


# ------------------------------------------------------------- patches
def test_target_grid_is_fixed_in_km_not_degrees():
    for lat0 in (10.0, 35.0):
        lat, lon = gridsat.target_grid(lat0, -60.0, n=11, km=8.0)
        dlat_km = (lat[0, 5] - lat[1, 5]) * gridsat.KM_PER_DEG
        dlon_km = ((lon[5, 6] - lon[5, 5]) * gridsat.KM_PER_DEG
                   * np.cos(np.radians(lat[5, 5])))
        assert dlat_km == pytest.approx(8.0, rel=1e-6)
        assert dlon_km == pytest.approx(8.0, rel=1e-3)
        assert lat[5, 5] == pytest.approx(lat0)


def test_target_grid_wraps_dateline():
    _, lon = gridsat.target_grid(15.0, 179.9, n=21, km=8.0)
    assert lon.min() >= -180 and lon.max() < 180
    assert (lon < 0).any() and (lon > 0).any()


def test_storm_patch_samples_the_right_place():
    lats = np.arange(-30, 30.0001, 0.07)
    lons = np.arange(-180, 180, 0.07)
    field = np.add.outer(lats, np.zeros_like(lons)).astype(np.float32)
    da = xr.DataArray(field, coords={"lat": lats, "lon": lons},
                      dims=("lat", "lon"))
    patch, nanfrac = gridsat.storm_patch(da, 12.3, -45.6, n=33, km=8.0)
    assert nanfrac == 0
    assert patch[16, 16] == pytest.approx(12.3, abs=0.01)   # centre
    assert patch[0, 16] > patch[-1, 16]                      # north on top
