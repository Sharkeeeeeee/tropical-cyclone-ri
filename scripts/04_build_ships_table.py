"""Parse every SHIPS basin file and join it to the IBTrACS RI labels.

The join key is (ATCF ID, synoptic time). SHIPS carries its own 24 h intensity
change on the DELV line, which gives a free consistency check: it and the
IBTrACS-derived dv24 come from the same best tracks and should agree almost
everywhere. Where they don't, something is wrong with the join.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import (PROCESSED_DIR, SHIPS_FILES, SHIPS_RAW_DIR,  # noqa: E402
                         SHIPS_TO_IBTRACS_BASIN)
from tcri.ships import parse_all  # noqa: E402


def section(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def main():
    section("parse SHIPS files")
    files = {}
    for basin, url in SHIPS_FILES.items():
        p = SHIPS_RAW_DIR / url.rsplit("/", 1)[1]
        if p.exists():
            files[basin] = p
        else:
            print(f"  [{basin}] MISSING {p.name} -- skipping")
    ships = parse_all(files)
    print(f"  total {len(ships):,} cases, {len(ships.columns)} columns")
    ships["basin"] = ships["basin_ships"].map(SHIPS_TO_IBTRACS_BASIN)

    cache = PROCESSED_DIR / "ships_t0.csv"
    ships.to_csv(cache, index=False)
    print(f"  cached -> {cache}")

    section("join to IBTrACS labels")
    lab = pd.read_csv(PROCESSED_DIR / "ri_labels.csv", converters={"BASIN": str, "SUBBASIN": str}, parse_dates=["ISO_TIME"])
    print(f"  labels    : {len(lab):>8,}")
    print(f"  ships     : {len(ships):>8,}")

    ships = ships.dropna(subset=["atcf_id", "time"])
    ships = ships.drop_duplicates(subset=["atcf_id", "time"], keep="first")

    j = lab.merge(ships, left_on=["USA_ATCF_ID", "ISO_TIME"],
                  right_on=["atcf_id", "time"], how="inner",
                  suffixes=("", "_ships"))
    print(f"  joined    : {len(j):>8,}  "
          f"({100*len(j)/len(lab):.1f}% of labelled cases)")

    section("consistency check: SHIPS DELV(+24) vs IBTrACS dv24")
    c = j.dropna(subset=["ships_dv24"])
    diff = c["ships_dv24"] - c["dv24"]
    print(f"  comparable cases : {len(c):>8,}")
    print(f"  exact agreement  : {100*(diff == 0).mean():>7.2f}%")
    print(f"  within 5 kt      : {100*(diff.abs() <= 5).mean():>7.2f}%")
    print(f"  mean / median    : {diff.mean():>7.2f} / {diff.median():.2f} kt")
    print(f"  max |difference| : {diff.abs().max():>7.1f} kt")
    if (diff == 0).mean() < 0.90:
        print("  WARNING: poor agreement -- the join or the label logic is wrong.")

    section("join rate and predictor availability by basin")
    lab_n = lab.groupby("BASIN").size().rename("labelled")
    j_n = j.groupby("BASIN").size().rename("joined")
    tbl = pd.concat([lab_n, j_n], axis=1)
    tbl["join_pct"] = 100 * tbl["joined"] / tbl["labelled"]
    for c_ in ["shrd", "pot", "cohc", "rhlo", "ir_pct_lt_m30"]:
        tbl[c_] = 100 * j.groupby("BASIN")[c_].apply(lambda s: s.notna().mean())
    print(tbl.to_string(float_format=lambda v: f"{v:.1f}"))
    print("\n(last five columns: % of joined cases with that predictor present)")

    section("write")
    out = PROCESSED_DIR / "ri_with_ships.csv"
    j.to_csv(out, index=False)
    print(f"  {len(j):,} rows x {len(j.columns)} cols -> {out}")
    m = j[(j["SEASON"] >= 2000) & (j["overwater_24h"])]
    print(f"  of which MERGIR era + over water 24 h: {len(m):,} "
          f"({int(m['ri'].sum()):,} RI, {100*m['ri'].mean():.2f}%)")


if __name__ == "__main__":
    main()
