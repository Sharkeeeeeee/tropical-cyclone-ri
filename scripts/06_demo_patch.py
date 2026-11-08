"""Visual check of the patch extractor. Look at the images before trusting them.

Renders two cases from Hurricane Katrina (2005) as 4-frame sequences: the onset
of rapid intensification, and peak intensity a day later. The centre cross must
sit on the eye, the frames must be continuous in time, and the physical scale
must be identical across both rows despite the change in latitude.
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
from tcri.gridsat import (PATCH_KM, PATCH_N, open_time, sequence_times,  # noqa: E402
                          storm_patch)

CASES = [("2005-08-27 12:00", "RI onset"), ("2005-08-28 12:00", "peak")]


def main():
    lab = pd.read_csv(PROCESSED_DIR / "ri_labels.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    k = lab[(lab.NAME == "KATRINA") & (lab.SEASON == 2005)].set_index("ISO_TIME")

    half = PATCH_N * PATCH_KM / 2
    extent = [-half, half, -half, half]
    fig, axes = plt.subplots(len(CASES), 4, figsize=(15, 7.8))

    for r, (tstr, tag) in enumerate(CASES):
        t0 = pd.Timestamp(tstr)
        row = k.loc[t0]
        for c, t in enumerate(sequence_times(t0)):
            da = open_time(t)
            # Interpolate the storm centre linearly between best-track fixes.
            tv = t.as_unit("ns").value            # pandas 3 defaults to us
            tn = k.index.as_unit("ns").asi8
            lat = np.interp(tv, tn, k["LAT"])
            lon = np.interp(tv, tn, k["LON"])
            patch, nanfrac = storm_patch(da, float(lat), float(lon))
            bt = patch - 273.15

            ax = axes[r, c]
            ax.imshow(bt, cmap="gray_r", vmin=-85, vmax=25, extent=extent,
                      origin="upper", interpolation="nearest")
            ax.plot(0, 0, "r+", ms=13, mew=1.6)
            ax.set_title(f"{t:%m-%d %HZ}   min {np.nanmin(bt):.0f}"
                         + r"$\degree$C", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"Katrina, {tag}\n{row.v0:.0f} kt, "
                              f"dv24 {row.dv24:+.0f} kt", fontsize=9)
            ax.set_xticks([-400, 0, 400])
            ax.set_yticks([-400, 0, 400])
            ax.tick_params(labelsize=7)
            print(f"  {t:%Y-%m-%d %HZ}  centre {float(lat):5.2f},{float(lon):7.2f}  "
                  f"min BT {np.nanmin(bt):6.1f}C  missing {100*nanfrac:.2f}%")

    fig.suptitle("GridSat-B1 storm-centred IR patches  "
                 f"({PATCH_N}x{PATCH_N} at {PATCH_KM:.0f} km, axes in km)",
                 fontsize=11)
    fig.tight_layout()
    out = REPORTS_DIR / "patch_demo_katrina.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
