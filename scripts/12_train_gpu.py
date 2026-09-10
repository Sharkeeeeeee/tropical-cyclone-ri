"""Train the patch CNN on a single consumer GPU without monopolising the machine.

Defaults are deliberately polite: below-normal process priority, a quarter of
the cores, 60% of VRAM, and a 70% duty cycle. Expect roughly a 30% throughput
cost against an all-out run, in exchange for a laptop that stays usable and a
GPU that does not thermally throttle itself back down to a similar speed.

Every fold is checkpointed, so Ctrl+C is safe: rerun with --resume and it picks
up at the next unfinished fold.

    python -u scripts/12_train_gpu.py --tag global_2000_2022 --group SEASON
    python -u scripts/12_train_gpu.py --tag global_2000_2022 --duty 1.0 --workers 8

This needs a CUDA build of PyTorch. A CPU-only build reports no GPU; install the
matching wheel with, for CUDA 12.4:

    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.budget import (DutyCycle, describe, limit_threads,  # noqa: E402
                         limit_vram, set_process_priority)

# Thread limits must be set before torch initialises its pools.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--threads", type=int, default=0)
_known, _ = _pre.parse_known_args()
_THREADS = limit_threads(_known.threads or max(1, (os.cpu_count() or 4) // 4))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.dataset import PatchDataset, patch_path  # noqa: E402
from tcri.evaluate import summary  # noqa: E402
from tcri.model import PatchCNN, count_params  # noqa: E402

ENV = ["per12", "vmax", "pot", "shrd", "t200", "rhlo", "d200", "cohc"]


def section(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}", flush=True)


def build_loaders(path, idx, tr_rows, te_rows, a, pin):
    tr = PatchDataset(path, idx, tr_rows, augment=True, n_frames=a.frames)
    te = PatchDataset(path, idx, te_rows, augment=False, n_frames=a.frames)
    common = dict(num_workers=a.workers, pin_memory=pin,
                  persistent_workers=a.workers > 0,
                  prefetch_factor=2 if a.workers > 0 else None)
    return (DataLoader(tr, batch_size=a.batch, shuffle=True,
                       drop_last=len(tr) > a.batch, **common),
            DataLoader(te, batch_size=a.batch * 2, shuffle=False, **common))


def run_fold(path, idx, tr_rows, te_rows, a, device, amp_dtype, fold, n_folds):
    torch.manual_seed(a.seed + fold)
    np.random.seed(a.seed + fold)
    dl_tr, dl_te = build_loaders(path, idx, tr_rows, te_rows, a,
                                 pin=device.type == "cuda")

    model = PatchCNN(in_frames=a.frames, width=a.width).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = max(a.epochs * len(dl_tr), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    duty = DutyCycle(a.duty)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = n = 0
        for x, y, _ in dl_tr:
            with duty:
                x = x.to(device, non_blocking=True)
                if device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    loss = lossf(model(x), y)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                sched.step()
                tot += float(loss) * len(y)
                n += len(y)
        if a.verbose or ep == a.epochs - 1:
            el = time.time() - t0
            print(f"    fold {fold+1}/{n_folds} epoch {ep+1:>3}/{a.epochs}  "
                  f"loss {tot/max(n,1):.4f}  {el:6.1f}s  "
                  f"({n*(ep+1)/max(el,1e-9):.0f} samples/s)", flush=True)
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print(f"    stopping fold early at --max-minutes {a.max_minutes}",
                  flush=True)
            break

    model.eval()
    probs, embs, rows = [], [], []
    with torch.no_grad():
        for x, _, r in dl_te:
            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                f = model.features(x)
                p = torch.sigmoid(model.head(f).squeeze(-1))
            probs.append(p.float().cpu().numpy())
            embs.append(f.float().cpu().numpy())
            rows.append(r.numpy())
    return (np.concatenate(rows), np.concatenate(probs),
            np.concatenate(embs), duty.slept, time.time() - t0)


def main():
    ap = argparse.ArgumentParser(parents=[_pre])
    ap.add_argument("--tag", default="NA_2020")
    ap.add_argument("--group", default="SEASON")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    # Politeness knobs.
    ap.add_argument("--duty", type=float, default=0.70,
                    help="fraction of wall clock spent working (1.0 = flat out)")
    ap.add_argument("--vram", type=float, default=0.60,
                    help="fraction of VRAM this process may allocate")
    ap.add_argument("--priority", default="below",
                    choices=["idle", "below", "normal"])
    ap.add_argument("--workers", type=int, default=-1,
                    help="dataloader workers; -1 picks a quarter of the cores")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="give up on a fold after this long (0 = no limit)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.workers < 0:
        a.workers = max(0, min(4, (os.cpu_count() or 4) // 4))

    prio = set_process_priority(a.priority)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        vram = limit_vram(a.vram)
        name = torch.cuda.get_device_name(0)
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)
        torch.backends.cudnn.benchmark = True
    else:
        device, vram, name, amp_dtype = torch.device("cpu"), "n/a", "CPU", None
        print("\n  No CUDA device visible. torch reports "
              f"{torch.__version__!r}.\n"
              "  A '+cpu' build cannot see your GPU; install the CUDA wheel:\n"
              "    pip install --force-reinstall torch "
              "--index-url https://download.pytorch.org/whl/cu124\n"
              "  Continuing on CPU, which will be slow.\n")

    section("setup")
    print(f"device: {name}   amp: {amp_dtype}")
    print(describe(prio, _THREADS, vram, a.duty, a.workers))

    p_arr = patch_path(PROCESSED_DIR, a.tag)
    arr = np.load(p_arr, mmap_mode="r")
    idx = pd.read_csv(PROCESSED_DIR / "patches" / f"index_{a.tag}.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    ships = pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv",
                        converters={"BASIN": str}, parse_dates=["ISO_TIME"],
                        usecols=["SID", "ISO_TIME"] + ENV)
    idx = idx.merge(ships, on=["SID", "ISO_TIME"], how="left")
    d = idx[(idx["frames_ok"] == arr.shape[1])
            & idx[ENV].notna().all(axis=1)].reset_index(drop=True)
    rows = d["row"].to_numpy()
    y = d["ri"].to_numpy(dtype=int)
    print(f"patches {arr.shape}  ({p_arr.stat().st_size/2**30:.2f} GiB)")
    print(f"cases {len(d):,}   RI {y.sum():,} ({100*y.mean():.2f}%)   "
          f"groups: {d[a.group].nunique()} distinct {a.group}")

    ck_dir = PROCESSED_DIR / "cnn_folds"
    ck_dir.mkdir(exist_ok=True)
    groups = d[a.group].to_numpy()
    n_splits = min(a.folds, d[a.group].nunique())
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=a.seed)

    section("training")
    print(f"{count_params(PatchCNN(a.frames, a.width)):,} trainable parameters")
    oof = np.full(len(d), np.nan)
    embs = None
    slept = worked = 0.0
    for k, (tr_i, te_i) in enumerate(gkf.split(rows, y, groups=groups)):
        ck = ck_dir / f"{a.tag}_seed{a.seed}_fold{k}.npz"
        if a.resume and ck.exists():
            z = np.load(ck)
            r, p, e = z["rows"], z["probs"], z["embs"]
            print(f"  fold {k+1}/{n_splits}: loaded from checkpoint")
        else:
            r, p, e, sl, el = run_fold(p_arr, d, rows[tr_i], rows[te_i], a,
                                       device, amp_dtype, k, n_splits)
            slept += sl
            worked += el
            np.savez_compressed(ck, rows=r, probs=p, embs=e)
        pos = pd.Series(np.arange(len(d)), index=rows).loc[r].to_numpy()
        oof[pos] = p
        if embs is None:
            embs = np.full((len(d), e.shape[1]), np.nan, dtype=np.float32)
        embs[pos] = e

    if worked:
        print(f"\n  {worked/60:.1f} min wall clock, {slept/60:.1f} min of it "
              f"idling to hold the {a.duty:.0%} duty cycle")

    section("out-of-fold skill")
    got = np.isfinite(oof)
    res = pd.DataFrame([
        summary(y[got], np.full(got.sum(), y[got].mean()), "climatology"),
        summary(y[got], oof[got], "CNN alone"),
    ]).set_index("model")
    print(res.to_string(float_format=lambda v: f"{v:.4f}"))

    out = d[["SID", "ISO_TIME", "SEASON", "BASIN", "recon_basin",
             "v0", "dv24", "ri"]].copy()
    out["p_cnn"] = oof
    for j in range(embs.shape[1]):
        out[f"emb{j:02d}"] = embs[:, j]
    p_out = PROCESSED_DIR / f"cnn_oof_{a.tag}_seed{a.seed}.csv"
    out.to_csv(p_out, index=False)
    (REPORTS_DIR / f"cnn_run_{a.tag}_seed{a.seed}.json").write_text(
        json.dumps(vars(a) | {"device": name, "params": count_params(
            PatchCNN(a.frames, a.width))}, indent=2, default=str))
    print(f"\nwrote {p_out}")
    print("feed it to scripts/11_structure_baseline.py's comparison to score "
          "the CNN against the structure features on identical folds")


if __name__ == "__main__":
    main()
