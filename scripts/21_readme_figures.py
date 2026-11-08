"""Figures for the top of the README.

1. reports/readme_katrina.gif -- Hurricane Katrina (2005) through its rapid
   intensification, as storm-centred GridSat-B1 IR patches beside the
   best-track intensity. Uses the same patch extractor as the training data.
2. reports/readme_results.png -- PR AUC of every model in the final
   reconnaissance-era comparison, read straight from the report CSV.

Needs the IBTrACS CSV (scripts/01_download_ibtracs.py) and network access to
NCEI OPeNDAP for ~28 GridSat frames (~0.1 MB each).
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import REPORTS_DIR, RI_THRESHOLD_KT  # noqa: E402
from tcri.gridsat import (PATCH_KM, PATCH_N, open_local, open_time,  # noqa: E402
                          storm_patch)
from tcri.ibtracs import load_raw  # noqa: E402

START, END = pd.Timestamp("2005-08-26 00:00"), pd.Timestamp("2005-08-29 09:00")

# Enhanced-IR palette: warm surfaces dark grey, cold cloud tops in colour.
_STOPS = [(-90, "#ff4fd8"), (-80, "#e8002a"), (-70, "#ffd400"),
          (-60, "#1fb24a"), (-50, "#1f5fd6"), (-40, "#38c8e8"),
          (-30, "#f2f2f2"), (30, "#101010")]
VMIN, VMAX = -90.0, 30.0
IR_CMAP = LinearSegmentedColormap.from_list(
    "enhanced_ir", [((t - VMIN) / (VMAX - VMIN), c) for t, c in _STOPS])

SSHS = [(64, "Cat 1"), (83, "Cat 2"), (96, "Cat 3"), (113, "Cat 4"),
        (137, "Cat 5")]


def katrina_track(path=None) -> pd.DataFrame:
    df = load_raw(path) if path else load_raw()
    k = df[(df.NAME == "KATRINA") & (df.SEASON == 2005) & (df.BASIN == "NA")
           & (df.TRACK_TYPE == "main")]
    k = k.dropna(subset=["ISO_TIME", "LAT", "LON"]).drop_duplicates("ISO_TIME")
    return k.set_index("ISO_TIME").sort_index()


def ri_windows(k: pd.DataFrame):
    """(start, end) of every 24 h window gaining >= the RI threshold."""
    syn = k[k.index.hour.isin([0, 6, 12, 18]) & (k.index.minute == 0)]
    v = syn["USA_WIND"]
    out = []
    for t0, v0 in v.items():
        t1 = t0 + pd.Timedelta(hours=24)
        if t1 in v.index and v[t1] - v0 >= RI_THRESHOLD_KT:
            out.append((t0, t1))
    return out


def merge(spans):
    """Union of overlapping (start, end) spans, so shading never stacks."""
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def make_gif(k: pd.DataFrame, gridsat_dir: Path | None = None) -> Path:
    times = pd.date_range(START, END, freq="3h")
    tnum = k.index.as_unit("ns").asi8
    syn = k[k.index.hour.isin([0, 6, 12, 18]) & (k.index.minute == 0)]
    win = merge(ri_windows(k))
    half = PATCH_N * PATCH_KM / 2
    extent = [-half, half, -half, half]

    frames = []
    for t in times:
        tv = t.as_unit("ns").value
        lat = float(np.interp(tv, tnum, k["LAT"]))
        lon = float(np.interp(tv, tnum, k["LON"]))
        wind = float(np.interp(tv, syn.index.as_unit("ns").asi8,
                               syn["USA_WIND"]))
        local = (gridsat_dir / f"GRIDSAT-B1.{t:%Y.%m.%d.%H}.v02r01.nc"
                 if gridsat_dir else None)
        da = open_local(local) if local and local.exists() else open_time(t)
        patch, _ = storm_patch(da, lat, lon)
        bt = patch - 273.15
        in_ri = any(a <= t <= b for a, b in win)
        print(f"  {t:%m-%d %HZ}  {lat:5.2f},{lon:7.2f}  {wind:4.0f} kt"
              f"  min BT {np.nanmin(bt):6.1f}C{'  RI' if in_ri else ''}")

        fig = plt.figure(figsize=(9.6, 4.8), dpi=100)
        ax = fig.add_axes([0.02, 0.10, 0.42, 0.78])
        im = ax.imshow(bt, cmap=IR_CMAP, vmin=VMIN, vmax=VMAX, extent=extent,
                       origin="upper", interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        halo = [patheffects.withStroke(linewidth=2.5, foreground="black")]
        ax.plot([-470, -270], [-470, -470], color="white", lw=2.5,
                path_effects=halo)
        ax.text(-370, -440, "200 km", color="white", ha="center",
                va="bottom", fontsize=8, path_effects=halo)
        ax.set_title(f"{t:%d %b %Y %H:%M} UTC", fontsize=10)
        cax = fig.add_axes([0.45, 0.10, 0.012, 0.78])
        cb = fig.colorbar(im, cax=cax, ticks=[-90, -70, -50, -30, 0, 30])
        cb.ax.tick_params(labelsize=7)
        cb.set_label("cloud-top temperature (°C)", fontsize=8)

        axp = fig.add_axes([0.58, 0.18, 0.39, 0.66])
        for a, b in win:
            axp.axvspan(a, b, color="#e8002a", alpha=0.12, lw=0)
        for kt, name in SSHS:
            axp.axhline(kt, color="0.85", lw=0.7, zorder=0)
            axp.text(END + pd.Timedelta(hours=3), kt + 1.5, name,
                     fontsize=7, color="0.55", ha="right")
        sel = syn[(syn.index >= START - pd.Timedelta(hours=6))
                  & (syn.index <= END + pd.Timedelta(hours=6))]
        axp.plot(sel.index, sel["USA_WIND"], color="0.25", lw=1.6)
        axp.plot([t], [wind], "o", ms=8,
                 color="#e8002a" if in_ri else "0.25")
        axp.set_xlim(START - pd.Timedelta(hours=3), END + pd.Timedelta(hours=3))
        axp.set_ylim(40, 160)
        axp.set_ylabel("max sustained wind (kt)", fontsize=9)
        axp.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        axp.xaxis.set_major_locator(matplotlib.dates.DayLocator())
        axp.tick_params(labelsize=8)
        for s in ("top", "right"):
            axp.spines[s].set_visible(False)
        axp.set_title("rapid intensification" if in_ri else " ",
                      fontsize=10, color="#e8002a", fontweight="bold")
        axp.text(0.0, -0.22, "shaded: 24 h windows gaining ≥ 30 kt",
                 transform=axp.transAxes, fontsize=7.5, color="0.4")

        fig.suptitle("Hurricane Katrina (2005) from geostationary infrared",
                     fontsize=12, x=0.5, y=0.985)
        fig.text(0.99, 0.01, "NOAA GridSat-B1 · IBTrACS", ha="right",
                 fontsize=7, color="0.5")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").copy())

    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                      dither=Image.Dither.NONE) for f in frames]
    durations = [260] * len(pal)
    durations[-1] = 1800
    out = REPORTS_DIR / "readme_katrina.gif"
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations,
                loop=0, optimize=True)
    return out


def make_results() -> Path:
    r = pd.read_csv(REPORTS_DIR / "final_comparison_recon_2000_2022.csv")
    base = float(r.loc[r.model == "climatology", "PR_AUC"].iloc[0])
    r = r[r.model != "climatology"].sort_values("PR_AUC")

    def colour(m):
        if m == "environment":
            return "0.55"
        if "structure" in m and "CNN" not in m:
            return "#1f5fd6"
        if "CNN" in m:
            return "#e8a200"
        return "0.75"

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=130)
    ax.barh(r.model, r.PR_AUC, color=[colour(m) for m in r.model])
    for y, v in enumerate(r.PR_AUC):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(base, color="0.3", ls=":", lw=1)
    ax.text(base + 0.004, -0.9, f"climatology {base:.3f}", fontsize=7.5,
            color="0.3")
    ax.set_xlim(0, 0.47)
    ax.set_ylim(-1.2, len(r) - 0.5)
    ax.set_xlabel("PR AUC (higher is better) · 12,339 cases, NA + EP 2000–2022, "
                  "season-grouped CV", fontsize=8)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Hand-engineered cloud structure beats a CNN on the same "
                 "patches", fontsize=10, loc="left")
    fig.tight_layout()
    out = REPORTS_DIR / "readme_results.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibtracs", type=Path, default=None,
                    help="IBTrACS CSV (default: the one 01_download fetched); "
                         "any file with the same header and Katrina's rows")
    ap.add_argument("--gridsat-dir", type=Path, default=None,
                    help="folder of whole GridSat-B1 files to use instead of "
                         "OPeNDAP, e.g. from the AWS mirror when NCEI is down")
    args = ap.parse_args()
    print(f"wrote {make_results()}")
    k = katrina_track(args.ibtracs)
    print(f"wrote {make_gif(k, args.gridsat_dir)}")


if __name__ == "__main__":
    main()
