"""Merge sharded patch extractions into one array and index.

Cloud extraction runs in shards of a few seasons each, so that a dropped session
costs at most one shard. This stitches them back together into the single
memmap + index pair the rest of the pipeline expects.

    python scripts/13_merge_shards.py --out recon_2000_2022 \
        --tags recon_2000_2002 recon_2003_2005 recon_2006_2008
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()

    d = PROCESSED_DIR / "patches"
    parts = []
    for t in a.tags:
        arr_p, idx_p = d / f"patches_{t}.npy", d / f"index_{t}.csv"
        if not arr_p.exists() or not idx_p.exists():
            raise SystemExit(f"missing shard {t}: need both {arr_p.name} "
                             f"and {idx_p.name}")
        arr = np.load(arr_p, mmap_mode="r")
        idx = pd.read_csv(idx_p, converters={"BASIN": str},
                          parse_dates=["ISO_TIME"])
        if len(idx) != arr.shape[0]:
            raise SystemExit(f"{t}: index has {len(idx)} rows but array has "
                             f"{arr.shape[0]}")
        parts.append((t, arr, idx))
        print(f"  {t:<24s} {arr.shape}  {len(idx):>7,} cases  "
              f"{int((idx['frames_ok'] == arr.shape[1]).sum()):>7,} complete")

    shapes = {p[1].shape[1:] for p in parts}
    if len(shapes) != 1:
        raise SystemExit(f"shards disagree on frame/patch shape: {shapes}")

    total = sum(p[1].shape[0] for p in parts)
    out_arr = d / f"patches_{a.out}.npy"
    if out_arr.exists():
        raise SystemExit(f"{out_arr.name} exists; delete it to rebuild")
    merged = np.lib.format.open_memmap(
        out_arr, mode="w+", dtype=np.float16,
        shape=(total,) + parts[0][1].shape[1:])

    rows = []
    at = 0
    for t, arr, idx in parts:
        n = arr.shape[0]
        merged[at:at + n] = arr[:]
        idx = idx.copy()
        idx["row"] = np.arange(at, at + n)
        idx["shard"] = t
        rows.append(idx)
        at += n
        print(f"  copied {t} -> rows {at-n:,}..{at-1:,}")
    merged.flush()

    index = pd.concat(rows, ignore_index=True)
    dup = index.duplicated(["SID", "ISO_TIME"]).sum()
    if dup:
        print(f"\n  WARNING: {dup:,} duplicate (SID, ISO_TIME) rows across "
              f"shards. Overlapping --seasons ranges?")
    out_idx = d / f"index_{a.out}.csv"
    index.to_csv(out_idx, index=False)

    complete = index["frames_ok"] == merged.shape[1]
    print(f"\nmerged {len(a.tags)} shards")
    print(f"  cases          : {total:,}")
    print(f"  complete       : {int(complete.sum()):,} "
          f"({100*complete.mean():.1f}%)")
    print(f"  RI (complete)  : {int(index.loc[complete, 'ri'].sum()):,} "
          f"({100*index.loc[complete, 'ri'].mean():.2f}%)")
    print(f"  seasons        : {int(index.SEASON.min())}-{int(index.SEASON.max())}")
    print(f"  array          : {out_arr}  "
          f"({out_arr.stat().st_size/2**30:.2f} GiB)")
    print(f"  index          : {out_idx}")


if __name__ == "__main__":
    main()
