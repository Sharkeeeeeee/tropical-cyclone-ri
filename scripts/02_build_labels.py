"""Build the RI label table and report the base-rate statistics.

Answers the question the proposal budget rests on: how many independent RI
events actually exist in the satellite record?
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import (MERGIR_START_YEAR, PROCESSED_DIR, REPORTS_DIR,  # noqa: E402
                         RI_THRESHOLD_KT)
from tcri.ibtracs import (build_ri_samples, label_episodes, load_raw,  # noqa: E402
                          to_synoptic)

pd.set_option("display.width", 200)


def section(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def main():
    section("load")
    raw = load_raw()
    print(f"raw rows                : {len(raw):>9,}")
    print(f"distinct storms         : {raw['SID'].nunique():>9,}")
    print(f"seasons                 : {int(raw['SEASON'].min())}-{int(raw['SEASON'].max())}")

    syn = to_synoptic(raw)
    print(f"synoptic main-track rows: {len(syn):>9,}")
    print(f"  with USA_WIND         : {syn['USA_WIND'].notna().sum():>9,} "
          f"({100*syn['USA_WIND'].notna().mean():.1f}%)")
    print(f"  with WMO_WIND         : {syn['WMO_WIND'].notna().sum():>9,} "
          f"({100*syn['WMO_WIND'].notna().mean():.1f}%)")

    section("build 24 h intensity-change cases")
    s = build_ri_samples(syn)
    print(f"eligible at t0          : {int(s['elig_t0'].sum()):>9,}")
    print(f"  and verifiable at +24h: {int(s['sample'].sum()):>9,}")

    d = s[s["sample"]].copy()
    d["ri"] = d["ri"].astype(int)

    section(f"base rate  (RI = dv24 >= {RI_THRESHOLD_KT:.0f} kt)")
    print(f"cases                   : {len(d):>9,}")
    print(f"RI-positive cases        : {int(d['ri'].sum()):>9,}")
    print(f"base rate               : {100*d['ri'].mean():>9.2f}%")
    ow = d[d["overwater_24h"]]
    print(f"\nrestricted to over-water for the full 24 h window:")
    print(f"cases                   : {len(ow):>9,}")
    print(f"RI-positive cases        : {int(ow['ri'].sum()):>9,}")
    print(f"base rate               : {100*ow['ri'].mean():>9.2f}%")

    section("independent RI episodes (consecutive 6-hourly hits merged)")
    for name, sub in [("all cases", d), ("over-water 24 h", ow)]:
        ep = label_episodes(sub)
        n_ep = ep["episode_id"].nunique()
        print(f"{name:<20s}  positives {int(sub['ri'].sum()):>6,}  ->  "
              f"episodes {n_ep:>6,}  ({int(sub['ri'].sum())/max(n_ep,1):.2f} fixes/episode)")

    section(f"MERGIR era only (>= {MERGIR_START_YEAR}), over water 24 h")
    m = ow[ow["SEASON"] >= MERGIR_START_YEAR]
    ep_m = label_episodes(m)
    print(f"storms                  : {m['SID'].nunique():>9,}")
    print(f"cases                   : {len(m):>9,}")
    print(f"RI-positive cases        : {int(m['ri'].sum()):>9,}")
    print(f"base rate               : {100*m['ri'].mean():>9.2f}%")
    print(f"independent episodes    : {ep_m['episode_id'].nunique():>9,}")

    section("by basin  (MERGIR era, over water 24 h)")
    g = m.groupby("BASIN").agg(
        storms=("SID", "nunique"), cases=("ri", "size"),
        ri_cases=("ri", "sum"), base_rate_pct=("ri", lambda x: 100 * x.mean()))
    ep_by_basin = label_episodes(m).groupby("BASIN")["episode_id"].nunique()
    g["episodes"] = ep_by_basin
    g["recon"] = ["yes" if b in ("NA", "EP") else "no" for b in g.index]
    print(g.sort_values("cases", ascending=False).to_string(
        float_format=lambda v: f"{v:.2f}"))

    section("recon-verified vs Dvorak-derived  (MERGIR era, over water 24 h)")
    r = m.groupby("recon_basin").agg(
        storms=("SID", "nunique"), cases=("ri", "size"),
        ri_cases=("ri", "sum"), base_rate_pct=("ri", lambda x: 100 * x.mean()))
    ep_r = label_episodes(m).groupby("recon_basin")["episode_id"].nunique()
    r["episodes"] = ep_r.reindex(r.index)          # align on the boolean key
    r.index = ["Dvorak-derived (no recon)" if not k else "recon-verified (NA/EP)"
               for k in r.index]
    print(r.to_string(float_format=lambda v: f"{v:.2f}"))

    section("per-season sanity check (over water 24 h, last 12 seasons)")
    y = ow.groupby("SEASON").agg(
        storms=("SID", "nunique"), cases=("ri", "size"), ri_cases=("ri", "sum"),
        base_rate_pct=("ri", lambda x: 100 * x.mean()))
    print(y.tail(12).to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nNote: the final season is partial -- IBTrACS is still being updated "
          "for it, and its cases must not be used for evaluation.")

    section("intensity-change distribution (over water 24 h, MERGIR era)")
    q = m["dv24"].describe(percentiles=[.01, .05, .25, .5, .75, .9, .95, .99])
    print(q.to_string(float_format=lambda v: f"{v:.1f}"))
    for thr in (25, 30, 35, 40, 45, 50, 55):
        n = int((m["dv24"] >= thr).sum())
        print(f"  dv24 >= {thr:>2d} kt : {n:>6,}  ({100*n/len(m):5.2f}%)")

    section("write outputs")
    cols = ["SID", "USA_ATCF_ID", "ISO_TIME", "SEASON", "BASIN", "SUBBASIN", "NAME", "NATURE",
            "LAT", "LON", "v0", "v_p24", "dv24", "ri", "d2l_0",
            "overwater_24h", "recon_basin", "USA_SSHS", "USA_AGENCY"]
    out = d[cols].sort_values(["SID", "ISO_TIME"])
    p = PROCESSED_DIR / "ri_labels.csv"
    out.to_csv(p, index=False)
    print(f"{len(out):,} labelled cases -> {p}")

    ep_all = label_episodes(d)[["SID", "ISO_TIME", "BASIN", "SEASON",
                                "v0", "v_p24", "dv24", "episode_id"]]
    pe = PROCESSED_DIR / "ri_episodes.csv"
    ep_all.to_csv(pe, index=False)
    print(f"{ep_all['episode_id'].nunique():,} episodes -> {pe}")


if __name__ == "__main__":
    main()
