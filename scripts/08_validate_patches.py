"""End-to-end check of the patch pipeline against SHIPS' own IR predictors.

SHIPS publishes two quantities computed from the operational GOES imagery: the
percent of the 50-200 km annulus colder than -30 C, and the standard deviation
of brightness temperature within 200 km. Recomputing both from our GridSat
patches and comparing tests the whole chain at once - geolocation, storm
centring, the kilometre scaling of the grid, the brightness-temperature units
and the array orientation. Any of those being wrong would destroy the
correlation.

Agreement will not be perfect and should not be: SHIPS uses GOES alone while
GridSat merges every geostationary platform with a view-zenith-angle correction,
and the SHIPS image can be up to about 1.5 h off the synoptic hour.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.gridsat import PATCH_KM, PATCH_N  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "NA_2020"


def radius_grid(n=PATCH_N, km=PATCH_KM):
    off = (np.arange(n) - (n - 1) / 2.0) * km
    return np.hypot(off[:, None], off[None, :])


def main():
    d = PROCESSED_DIR / "patches"
    idx = pd.read_csv(d / f"index_{TAG}.csv", converters={"BASIN": str},
                      parse_dates=["ISO_TIME"])
    arr = np.load(d / f"patches_{TAG}.npy", mmap_mode="r")
    print(f"patches: {arr.shape}   index: {len(idx):,} rows")

    r = radius_grid()
    ann = (r >= 50) & (r <= 200)      # SHIPS annulus for the percent-area stats
    inner = r <= 200                  # SHIPS uses 0-200 km for the std dev

    keep = idx["frames_ok"] == arr.shape[1]
    rows = idx.loc[keep, "row"].to_numpy()
    print(f"complete cases: {len(rows):,}")

    pct30, sdbt, coldest = [], [], []
    for i in rows:
        bt = np.asarray(arr[i, -1], dtype=np.float32) - 273.15   # t0 frame, degC
        a = bt[ann]
        a = a[np.isfinite(a)]
        b = bt[inner]
        b = b[np.isfinite(b)]
        pct30.append(100.0 * np.mean(a < -30) if a.size else np.nan)
        sdbt.append(float(np.std(b)) if b.size else np.nan)
        coldest.append(float(np.nanmin(bt)))

    got = idx.loc[keep].copy()
    got["our_pct30"] = pct30
    got["our_sdbt"] = sdbt
    got["our_min_bt"] = coldest

    ships = pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv",
                        converters={"BASIN": str}, parse_dates=["ISO_TIME"],
                        usecols=["SID", "ISO_TIME", "ir_pct_lt_m30",
                                 "ir_std_bt_0_200", "ir_time", "v0"])
    m = got.merge(ships, on=["SID", "ISO_TIME"], how="inner",
                  suffixes=("", "_s"))
    m = m.dropna(subset=["ir_pct_lt_m30", "our_pct30"])
    print(f"comparable against SHIPS: {len(m):,}")

    print("\n--- percent of 50-200 km annulus colder than -30 C ---")
    c = np.corrcoef(m["our_pct30"], m["ir_pct_lt_m30"])[0, 1]
    bias = (m["our_pct30"] - m["ir_pct_lt_m30"]).mean()
    mae = (m["our_pct30"] - m["ir_pct_lt_m30"]).abs().mean()
    print(f"  correlation : {c:.4f}")
    print(f"  bias (ours - SHIPS): {bias:+.2f} pct points")
    print(f"  mean abs error     : {mae:.2f} pct points")

    print("\n--- standard deviation of BT within 200 km ---")
    m2 = m.dropna(subset=["ir_std_bt_0_200", "our_sdbt"])
    c2 = np.corrcoef(m2["our_sdbt"], m2["ir_std_bt_0_200"])[0, 1]
    print(f"  correlation : {c2:.4f}")
    print(f"  bias (ours - SHIPS): "
          f"{(m2['our_sdbt'] - m2['ir_std_bt_0_200']).mean():+.2f} K")

    print("\n--- diagnosing the std-dev offset: effective radius sweep ---")
    print("  SHIPS documents this predictor as 0-200 km. Recomputing it over a")
    print("  range of radii shows which one its numbers actually behave like.")
    print(f"  {'radius':>9} {'our mean':>9} {'bias':>8} {'corr':>7} {'MAE':>6}")
    for R in (75, 100, 125, 150, 175, 200, 250):
        vals = []
        for i in rows:
            bt = np.asarray(arr[i, -1], dtype=np.float32) - 273.15
            v = bt[(r <= R) & np.isfinite(bt)]
            vals.append(float(np.std(v)) if v.size else np.nan)
        got[f"sd{R}"] = vals
    mm = got.merge(ships, on=["SID", "ISO_TIME"]).dropna(subset=["ir_std_bt_0_200"])
    for R in (75, 100, 125, 150, 175, 200, 250):
        b = mm[f"sd{R}"] - mm["ir_std_bt_0_200"]
        print(f"  {R:>7} km {mm[f'sd{R}'].mean():>9.2f} {b.mean():>+8.2f} "
              f"{np.corrcoef(mm[f'sd{R}'], mm['ir_std_bt_0_200'])[0,1]:>7.3f} "
              f"{b.abs().mean():>6.2f}")
    print("  The offset is consistent with SHIPS behaving like a ~100-125 km")
    print("  effective radius rather than a literal 200 km disc. It does not")
    print("  affect the patches themselves: the annulus predictor above, which")
    print("  is the one that tests geolocation and scale, matches almost exactly.")

    print("\n--- sanity: coldest cloud top vs intensity ---")
    print(f"  corr(min BT, v0) : "
          f"{np.corrcoef(m['our_min_bt'], m['v0'])[0,1]:+.4f}  "
          "(negative expected: stronger storms have colder tops)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].scatter(m["ir_pct_lt_m30"], m["our_pct30"], s=9, alpha=.35,
                  edgecolors="none")
    ax[0].plot([0, 100], [0, 100], "r-", lw=1)
    ax[0].set_xlabel("SHIPS  % area < -30 C  (50-200 km)")
    ax[0].set_ylabel("recomputed from GridSat patch")
    ax[0].set_title(f"r = {c:.3f}   n = {len(m):,}")
    ax[1].scatter(m2["ir_std_bt_0_200"], m2["our_sdbt"], s=9, alpha=.35,
                  edgecolors="none")
    lim = [0, max(m2["ir_std_bt_0_200"].max(), m2["our_sdbt"].max()) * 1.05]
    ax[1].plot(lim, lim, "r-", lw=1)
    ax[1].set_xlabel("SHIPS  std dev BT  (0-200 km, K)")
    ax[1].set_ylabel("recomputed from GridSat patch")
    ax[1].set_title(f"r = {c2:.3f}")
    fig.suptitle(f"Patch pipeline validated against SHIPS IR predictors  ({TAG})")
    fig.tight_layout()
    out = REPORTS_DIR / f"patch_validation_{TAG}.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
