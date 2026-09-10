"""Measure where training time actually goes on this machine.

A short run on a few hundred cases measures startup overhead, not throughput,
so it cannot be extrapolated to the full archive. This separates the two things
that matter:

* **Compute** - forward and backward on synthetic batches, nothing else. The
  ceiling the GPU can offer.
* **Data** - iterating the real memmap-backed dataset with no model at all. The
  ceiling the loader can offer.

Whichever is lower governs, and the gap between them says whether buying more
GPU would change anything. For this model the expected answer is that the data
path governs, which is the argument for spending grant money on ablations and
pretraining rather than on a bigger card.

    python -u scripts/14_benchmark_gpu.py --tag NA_2020
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.budget import limit_threads  # noqa: E402

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=0)
_k, _ = _pre.parse_known_args()
limit_threads(_k.threads or max(1, (os.cpu_count() or 4) // 2))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tcri.config import PROCESSED_DIR  # noqa: E402
from tcri.dataset import PatchDataset, patch_path  # noqa: E402
from tcri.model import PatchCNN, count_params  # noqa: E402


def section(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}", flush=True)


def bench_compute(device, batch, frames, width, amp_dtype, iters=40):
    model = PatchCNN(in_frames=frames, width=width).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    x = torch.randn(batch, frames, 128, 128, device=device)
    if device.type == "cuda":
        x = x.to(memory_format=torch.channels_last)
    y = (torch.rand(batch, device=device) < 0.09).float()

    for _ in range(5):                       # warm up cudnn autotuning
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            loss = lossf(model(x), y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            loss = lossf(model(x), y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    el = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda" else 0.0)
    return iters * batch / el, peak


def bench_loader(tag, batch, workers, frames, seconds=8.0):
    idx = pd.read_csv(PROCESSED_DIR / "patches" / f"index_{tag}.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    arr = np.load(patch_path(PROCESSED_DIR, tag), mmap_mode="r")
    rows = idx.loc[idx["frames_ok"] == arr.shape[1], "row"].to_numpy()
    ds = PatchDataset(patch_path(PROCESSED_DIR, tag), idx, rows,
                      augment=True, n_frames=frames)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=workers > 0,
                    prefetch_factor=4 if workers > 0 else None)
    it = iter(dl)
    next(it)                                  # pay worker startup first
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds:
        try:
            x, _, _ = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        n += len(x)
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(parents=[_pre])
    ap.add_argument("--tag", default="NA_2020")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--target-cases", type=int, default=14088,
                    help="dataset size to extrapolate to (NA+EP is 14,088)")
    a = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
    else:
        device, amp, name = torch.device("cpu"), None, "CPU"

    section("setup")
    print(f"device : {name}")
    print(f"amp    : {amp}")
    print(f"model  : {count_params(PatchCNN(a.frames, a.width)):,} parameters, "
          f"{a.frames} frames")

    section("compute ceiling (synthetic batches, no data loading)")
    print(f"{'batch':>7} {'samples/s':>11} {'peak VRAM':>11}")
    best_compute, best_batch = 0.0, 64
    for b in (32, 64, 128, 256):
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            sps, peak = bench_compute(device, b, a.frames, a.width, amp)
            print(f"{b:>7} {sps:>11.0f} {peak:>10.2f}G")
            if sps > best_compute:
                best_compute, best_batch = sps, b
        except RuntimeError as e:
            print(f"{b:>7}  failed: {str(e)[:50]}")
            break

    section("data ceiling (real patches, no model)")
    print(f"{'workers':>7} {'samples/s':>11}")
    best_data, best_workers = 0.0, 0
    for w in (0, 2, 4, 6):
        try:
            sps = bench_loader(a.tag, best_batch, w, a.frames)
            print(f"{w:>7} {sps:>11.0f}")
            if sps > best_data:
                best_data, best_workers = sps, w
        except Exception as e:
            print(f"{w:>7}  failed: {type(e).__name__} {str(e)[:40]}")

    section("verdict")
    governing = min(best_compute, best_data)
    which = "data loading" if best_data < best_compute else "compute"
    print(f"compute ceiling : {best_compute:>8.0f} samples/s  (batch {best_batch})")
    print(f"data ceiling    : {best_data:>8.0f} samples/s  (workers {best_workers})")
    print(f"governed by     : {which}  ->  {governing:.0f} samples/s")
    if best_compute > 0:
        print(f"GPU utilisation : {100*min(best_data,best_compute)/best_compute:.0f}% "
              "of what the card could do")

    section(f"extrapolation to {a.target_cases:,} cases")
    for epochs in (30, 40, 60):
        per_fold = a.target_cases * 0.8 * epochs / governing
        print(f"  {epochs:>2} epochs: {per_fold/60:6.1f} min/fold  ->  "
              f"{5*per_fold/3600:5.2f} h for 5-fold CV")
    print("\n  A duty cycle of 0.7 multiplies these by about 1.43.")
    if best_data < best_compute:
        print("  Data loading governs, so a faster GPU would not help. More")
        print("  workers, or pre-staging the array in RAM, would.")


if __name__ == "__main__":
    main()
