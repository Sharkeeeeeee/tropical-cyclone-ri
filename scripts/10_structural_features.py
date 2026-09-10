"""Compute structural features for every extracted patch sequence.

CPU only, parallel over cases. Writes one CSV keyed by (SID, ISO_TIME) so it
joins straight onto the SHIPS table and the baseline out-of-fold predictions.

    python -u scripts/10_structural_features.py --tag NA_2020 --workers 8
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR  # noqa: E402
from tcri.gridsat import PATCH_KM  # noqa: E402
from tcri.structure import feature_names, sequence_features  # noqa: E402

_ARR = None
_TAG = None


def _init(tag):
    """Open the memmap once per worker rather than pickling patches around."""
    global _ARR, _TAG
    _TAG = tag
    _ARR = np.load(PROCESSED_DIR / "patches" / f"patches_{tag}.npy",
                   mmap_mode="r")


def _work(rows):
    out = []
    for r in rows:
        seq = np.asarray(_ARR[r], dtype=np.float32)
        out.append((r, sequence_features(seq, PATCH_KM)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="NA_2020")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()

    d = PROCESSED_DIR / "patches"
    idx = pd.read_csv(d / f"index_{a.tag}.csv", converters={"BASIN": str},
                      parse_dates=["ISO_TIME"])
    arr = np.load(d / f"patches_{a.tag}.npy", mmap_mode="r")
    keep = idx["frames_ok"] == arr.shape[1]
    rows = idx.loc[keep, "row"].to_numpy()
    names = feature_names(arr.shape[-1], PATCH_KM)
    print(f"patches {arr.shape}")
    print(f"complete cases: {len(rows):,}")
    print(f"features per case: {len(names)}")

    chunks = [rows[i:i + a.chunk] for i in range(0, len(rows), a.chunk)]
    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers,
                             initializer=_init, initargs=(a.tag,)) as ex:
        for j, res in enumerate(ex.map(_work, chunks), 1):
            for r, feats in res:
                results[r] = feats
            if j % 10 == 0 or j == len(chunks):
                el = time.time() - t0
                print(f"\r  {len(results):>6,}/{len(rows):,} cases  {el:6.1f}s "
                      f"({len(results)/el:.0f}/s)", end="", flush=True)
    print()

    feat = pd.DataFrame.from_dict(results, orient="index").reindex(columns=names)
    feat.index.name = "row"
    out = idx.loc[keep, ["SID", "ISO_TIME", "SEASON", "BASIN", "recon_basin",
                         "v0", "dv24", "ri", "row"]].set_index("row")
    out = out.join(feat)

    nan_rate = out[names].isna().mean()
    print(f"\nfeatures entirely present : "
          f"{int((nan_rate == 0).sum())}/{len(names)}")
    worst = nan_rate.sort_values(ascending=False).head(5)
    if worst.iloc[0] > 0:
        print("highest missing rates:")
        for k, v in worst.items():
            if v > 0:
                print(f"  {k:<28s} {100*v:5.2f}%")

    p = PROCESSED_DIR / f"structure_{a.tag}.csv"
    out.reset_index().to_csv(p, index=False)
    print(f"\n{len(out):,} rows x {len(names)} features -> {p}")


if __name__ == "__main__":
    main()
