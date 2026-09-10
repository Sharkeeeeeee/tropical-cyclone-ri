"""Extract storm-centred IR patch sequences for every labelled case.

Work is grouped by frame time, not by case: one GridSat file serves every storm
alive at that moment, and opening the remote file costs far more than reading a
box out of it. Storm centres come from the 3-hourly IBTrACS positions, so the
off-synoptic frames use the archive's own interpolation rather than ours.

Output is a single float16 memmap of shape (cases, frames, N, N) plus an index
CSV, so the training loader can slice it without unpacking anything.

    python -u scripts/07_extract_patches.py --basin NA --seasons 2020 --workers 8

Use -u when redirecting to a log: without it Python buffers stdout and a
running job looks like a dead one.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR, RAW_DIR  # noqa: E402
from tcri.gridsat import (PATCH_KM, PATCH_N, fetch_file, open_local,  # noqa: E402
                          open_time, sequence_times, storm_patch)
from tcri.ibtracs import load_raw  # noqa: E402

LAGS_H = (9, 6, 3, 0)
MAX_NAN_FRAC = 0.05          # reject a frame that is more than 5% missing


_SOURCE = "opendap"


def _init_worker(source: str):
    global _SOURCE
    _SOURCE = source


def worker(args):
    """Extract every patch needed at one frame time. Runs in its own process.

    With source='http' the whole global file is downloaded, read, and deleted
    before returning. Only a few tens of megabytes are ever on disk per worker,
    however large the total transfer.
    """
    ts_value, jobs = args
    ts = pd.Timestamp(ts_value)
    out = []
    tmp = None
    try:
        if _SOURCE in ("http", "s3"):
            fd, name = tempfile.mkstemp(suffix=".nc", prefix="gridsat_")
            os.close(fd)
            tmp = Path(name)
            fetch_file(ts, tmp, source=_SOURCE)
            da = open_local(tmp)         # .load() so the file can go now
        else:
            da = open_time(ts)
    except Exception as e:               # missing or unreadable file
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        return ([(r, f, None, np.nan) for r, f, _, _ in jobs],
                f"{ts}: {type(e).__name__}")
    try:
        for row, frame, lat, lon in jobs:
            try:
                patch, nanfrac = storm_patch(da, lat, lon)
                out.append((row, frame, patch.astype(np.float16), nanfrac))
            except Exception:
                out.append((row, frame, None, np.nan))
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return out, None


def build_positions(basin=None, seasons=None) -> pd.DataFrame:
    """3-hourly storm positions, used for the pre-t0 frames."""
    raw = load_raw()
    raw = raw[raw["TRACK_TYPE"] == "main"]
    raw = raw.dropna(subset=["ISO_TIME", "LAT", "LON"])
    raw = raw[raw["ISO_TIME"].dt.hour % 3 == 0]
    raw = raw[raw["ISO_TIME"].dt.minute == 0]
    return raw[["SID", "ISO_TIME", "LAT", "LON"]].drop_duplicates(
        ["SID", "ISO_TIME"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", nargs="*", default=None)
    ap.add_argument("--seasons", nargs="*", type=int, default=None)
    # Six is deliberate. NCEI's OPeNDAP endpoint started returning 502s
    # under ten concurrent readers, and a server that is refusing requests
    # is slower than a smaller pool that it answers.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue a previous run instead of starting over")
    ap.add_argument("--overwrite", action="store_true",
                    help="discard any existing output and start again")
    ap.add_argument("--source", default="opendap",
                    choices=["opendap", "http", "s3"],
                    help="opendap subsets server-side (~0.08 MB per patch); "
                         "http/s3 download whole 48 MB files and are ~650x the "
                         "transfer, for when OPeNDAP is down. s3 is the AWS "
                         "Open Data mirror -- prefer it, and prefer running "
                         "that variant on a cloud machine")
    a = ap.parse_args()

    lab = pd.read_csv(PROCESSED_DIR / "ri_labels.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    if a.basin:
        lab = lab[lab["BASIN"].isin(a.basin)]
    if a.seasons:
        lab = lab[lab["SEASON"].isin(a.seasons)]
    lab = lab[lab["overwater_24h"]].sort_values(["SID", "ISO_TIME"])
    if a.limit:
        lab = lab.head(a.limit)
    lab = lab.reset_index(drop=True)
    tag = a.tag or (("-".join(a.basin) if a.basin else "all") + "_" +
                    ("-".join(map(str, a.seasons)) if a.seasons else "all"))

    print(f"cases: {len(lab):,}  ({int(lab['ri'].sum()):,} RI, "
          f"{100*lab['ri'].mean():.2f}%)  storms: {lab['SID'].nunique():,}")

    pos = build_positions()
    pos_idx = pos.set_index(["SID", "ISO_TIME"])[["LAT", "LON"]]

    # One entry per (case, frame); grouped by the time it needs.
    tasks: dict[pd.Timestamp, list] = {}
    n_missing_pos = 0
    for row, r in lab.iterrows():
        for f, t in enumerate(sequence_times(r["ISO_TIME"], LAGS_H)):
            try:
                p = pos_idx.loc[(r["SID"], t)]
            except KeyError:
                n_missing_pos += 1
                continue
            tasks.setdefault(t, []).append(
                (row, f, float(p["LAT"]), float(p["LON"])))
    n_patches = sum(len(v) for v in tasks.values())
    print(f"frames needed: {n_patches:,} across {len(tasks):,} distinct times "
          f"({n_patches/max(len(tasks),1):.1f} per file)")
    if n_missing_pos:
        print(f"  {n_missing_pos:,} frames skipped: no best-track position "
              f"at that time (storm had not formed yet)")

    out_dir = PROCESSED_DIR / "patches"
    out_dir.mkdir(exist_ok=True)
    arr_path = out_dir / f"patches_{tag}.npy"
    nf_path = out_dir / f"nanfrac_{tag}.npy"
    shape = (len(lab), len(LAGS_H), PATCH_N, PATCH_N)

    def create_fresh():
        """Allocate the output array. Writes the checkpoint immediately.

        The checkpoint is saved before any work starts so that the array can
        never exist without one. A run that found an array but no checkpoint
        could not tell which frames were already done, and the old code
        silently rebuilt in that case -- which would discard hours of work
        without an error.
        """
        arr = np.lib.format.open_memmap(arr_path, mode="w+",
                                        dtype=np.float16, shape=shape)
        arr[:] = np.nan
        nanfrac = np.full(shape[:2], np.nan, dtype=np.float32)
        np.save(nf_path, nanfrac)
        return arr, nanfrac

    resumed = 0
    have_arr, have_ckpt = arr_path.exists(), nf_path.exists()

    if a.overwrite:
        if have_arr:
            print(f"--overwrite: discarding the existing {arr_path.name} "
                  f"({arr_path.stat().st_size/1e9:.2f} GB)")
        arr, nanfrac = create_fresh()

    elif a.resume:
        if not have_arr:
            print("--resume: nothing to resume from, starting fresh")
            arr, nanfrac = create_fresh()
        elif not have_ckpt:
            raise SystemExit(
                f"\n{arr_path.name} exists ({arr_path.stat().st_size/1e9:.2f} GB) "
                f"but {nf_path.name} does not, so there is no record of which\n"
                f"frames are already done. Another extraction may still be "
                f"running and writing to it.\n"
                f"Check for running processes first. To start over and discard "
                f"it, pass --overwrite.")
        else:
            arr = np.lib.format.open_memmap(arr_path, mode="r+")
            nanfrac = np.load(nf_path)
            if arr.shape != shape or nanfrac.shape != shape[:2]:
                raise SystemExit(
                    f"\nexisting arrays have shape {arr.shape}, but this "
                    f"selection needs {shape}.\nThe --basin/--seasons arguments "
                    f"must match the run being resumed, or pass --overwrite.")
            # A finite entry means that frame was already attempted.
            done_mask = np.isfinite(nanfrac)
            for t in list(tasks):
                remaining = [j for j in tasks[t] if not done_mask[j[0], j[1]]]
                resumed += len(tasks[t]) - len(remaining)
                if remaining:
                    tasks[t] = remaining
                else:
                    del tasks[t]
            print(f"resuming: {resumed:,} frames already done, "
                  f"{sum(len(v) for v in tasks.values()):,} to go")

    else:
        if have_arr:
            raise SystemExit(
                f"\n{arr_path.name} already exists "
                f"({arr_path.stat().st_size/1e9:.2f} GB).\n"
                f"Pass --resume to continue it, or --overwrite to discard it "
                f"and start again.\nRefusing to overwrite silently.")
        arr, nanfrac = create_fresh()

    items = [(t.to_datetime64(), v) for t, v in sorted(tasks.items())]
    if a.source in ("http", "s3"):
        gb = len(items) * 48 / 1e3
        print(f"\n  source={a.source}: downloading {len(items):,} whole files, "
              f"about {gb:.0f} GB of transfer.")
        print("  Only a few tens of MB are on disk at once, but check any "
              "data cap before a long run.\n")
    t_start = time.time()
    done = errors = 0
    with ProcessPoolExecutor(max_workers=a.workers,
                             initializer=_init_worker,
                             initargs=(a.source,)) as ex:
        futs = {ex.submit(worker, it): it[0] for it in items}
        for fut in as_completed(futs):
            res, err = fut.result()
            if err:
                errors += 1
                if errors <= 5:
                    print(f"  [warn] {err}")
            for row, frame, patch, nf in res:
                if patch is not None:
                    arr[row, frame] = patch
                nanfrac[row, frame] = nf
            done += 1
            if done % 200 == 0:          # checkpoint so a crash is resumable
                arr.flush()
                np.save(nf_path, nanfrac)
            if done % 50 == 0 or done == len(items):
                el = time.time() - t_start
                print(f"\r  {done:>5}/{len(items)} files  {el:6.1f}s  "
                      f"({done/el:.1f} files/s, eta "
                      f"{(len(items)-done)/max(done/el,1e-9)/60:.1f} min)",
                      end="", flush=True)
    print()
    arr.flush()
    np.save(nf_path, nanfrac)

    ok = np.isfinite(nanfrac) & (nanfrac <= MAX_NAN_FRAC)
    lab["frames_ok"] = ok.sum(axis=1)
    lab["max_nan_frac"] = np.nanmax(np.where(np.isfinite(nanfrac), nanfrac, 0), axis=1)
    lab["row"] = np.arange(len(lab))
    complete = lab["frames_ok"] == len(LAGS_H)

    print(f"\nfiles with errors     : {errors:,} / {len(items):,}")
    print(f"cases with all {len(LAGS_H)} frames: {int(complete.sum()):,} / {len(lab):,} "
          f"({100*complete.mean():.1f}%)")
    print(f"  of which RI         : {int(lab.loc[complete, 'ri'].sum()):,} "
          f"({100*lab.loc[complete, 'ri'].mean():.2f}%)")
    print(f"array: {arr_path}  {arr_path.stat().st_size/1e9:.2f} GB")

    idx_path = out_dir / f"index_{tag}.csv"
    lab.to_csv(idx_path, index=False)
    print(f"index: {idx_path}")


if __name__ == "__main__":
    main()
