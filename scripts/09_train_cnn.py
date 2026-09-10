"""Train the patch CNN under grouped cross-validation and score it against the
statistical baseline on exactly the same cases.

The comparison that matters is not CNN-versus-logistic-regression. The baseline
report showed that most of the spread between those models is capacity rather
than information, so the question is posed as: holding the large-scale
environment fixed, does the imagery add more than the two hand-engineered IR
summary statistics do?

    GBM(environment)                      <- floor
    GBM(environment + SHIPS IR features)  <- the bar
    GBM(environment + CNN output)         <- the claim

All three use identical rows and identical folds.

    python scripts/09_train_cnn.py --tag NA_2020 --group SID --epochs 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcri.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from tcri.dataset import PatchDataset, load_patches  # noqa: E402
from tcri.evaluate import reliability, summary  # noqa: E402
from tcri.model import PatchCNN, count_params  # noqa: E402

ENV = ["per12", "vmax", "pot", "shrd", "t200", "rhlo", "d200", "cohc"]
IR = ["ir_pct_lt_m30", "ir_std_bt_0_200"]


def section(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def make_gbm(seed=0):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def train_fold(arr, idx, tr_rows, te_rows, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_frames = args.frames or arr.shape[1]

    tr = PatchDataset(arr, idx, tr_rows, augment=True, n_frames=n_frames)
    te = PatchDataset(arr, idx, te_rows, augment=False, n_frames=n_frames)
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0,
                       drop_last=len(tr) > args.batch)
    dl_te = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=0)

    model = PatchCNN(in_frames=n_frames, width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * len(dl_tr), 1))
    # Plain BCE, no positive-class reweighting: the probabilities have to stay
    # calibrated, and reweighting would inflate them away from the base rate.
    lossf = nn.BCEWithLogitsLoss()

    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(y)
            n += len(y)
        if args.verbose:
            print(f"      epoch {ep+1:>2}/{args.epochs}  train loss {tot/max(n,1):.4f}")

    model.eval()
    probs, embs, rows = [], [], []
    with torch.no_grad():
        for x, _, r in dl_te:
            x = x.to(device)
            f = model.features(x)
            probs.append(torch.sigmoid(model.head(f).squeeze(-1)).cpu().numpy())
            embs.append(f.cpu().numpy())
            rows.append(r.numpy())
    return (np.concatenate(rows), np.concatenate(probs),
            np.concatenate(embs), count_params(model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="NA_2020")
    ap.add_argument("--group", default="SID",
                    help="SEASON for multi-season data, SID for a single season")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    section("data")
    arr, idx, rows = load_patches(PROCESSED_DIR, args.tag)
    print(f"patches {arr.shape}   complete cases {len(rows):,}")

    ships = pd.read_csv(PROCESSED_DIR / "ri_with_ships.csv",
                        converters={"BASIN": str}, parse_dates=["ISO_TIME"],
                        usecols=["SID", "ISO_TIME"] + ENV + IR)
    idx = idx.merge(ships, on=["SID", "ISO_TIME"], how="left")
    have_env = idx["row"].isin(rows) & idx[ENV].notna().all(axis=1)
    d = idx[have_env].reset_index(drop=True)
    rows = d["row"].to_numpy()
    y = d["ri"].to_numpy(dtype=int)
    print(f"with complete environment predictors: {len(d):,}  "
          f"({y.sum()} RI, {100*y.mean():.2f}%)")
    print(f"grouping folds by {args.group} "
          f"({d[args.group].nunique()} distinct groups)")
    print(f"device: {device}")

    if y.sum() < 60:
        print("\n  NOTE: this sample is far too small to train a useful vision\n"
              "  model. The point of this run is to verify the machinery -- data\n"
              "  loading, grouped folds, and the comparison harness -- not to\n"
              "  produce a meaningful skill number. Treat the CNN scores below as\n"
              "  noise.")

    section("cross-validated CNN")
    groups = d[args.group].to_numpy()
    n_splits = min(args.folds, len(np.unique(groups)))
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=args.seed)
    oof = np.full(len(d), np.nan)
    emb_dim = None
    embs = None
    t0 = time.time()
    for k, (tr_i, te_i) in enumerate(gkf.split(rows, y, groups=groups)):
        r, p, e, npar = train_fold(arr, d, rows[tr_i], rows[te_i], args, device)
        pos = pd.Series(np.arange(len(d)), index=rows).loc[r].to_numpy()
        oof[pos] = p
        if embs is None:
            emb_dim = e.shape[1]
            embs = np.full((len(d), emb_dim), np.nan, dtype=np.float32)
        embs[pos] = e
        print(f"  fold {k+1}/{n_splits}  train {len(tr_i):>4}  test {len(te_i):>4}  "
              f"test RI {int(y[te_i].sum()):>3}  "
              f"mean p {p.mean():.4f}")
    print(f"  {npar:,} trainable parameters, {time.time()-t0:.1f}s total")

    section("comparison on identical rows and folds")
    Xenv = d[ENV].to_numpy(dtype=float)
    Xir = d[ENV + IR].to_numpy(dtype=float)
    Xcnn = np.column_stack([Xenv, oof])

    def cv_gbm(X):
        o = np.full(len(d), np.nan)
        for tr_i, te_i in gkf.split(X, y, groups=groups):
            m = make_gbm(args.seed)
            m.fit(X[tr_i], y[tr_i])
            o[te_i] = m.predict_proba(X[te_i])[:, 1]
        return o

    preds = {
        "climatology": np.full(len(d), y.mean()),
        "CNN alone": oof,
        "GBM environment": cv_gbm(Xenv),
        "GBM environment + SHIPS IR": cv_gbm(Xir),
        "GBM environment + CNN": cv_gbm(Xcnn),
    }
    res = pd.DataFrame([summary(y, p, k) for k, p in preds.items()]).set_index("model")
    print(res.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nThe stacked row uses the CNN's out-of-fold probability as a feature,")
    print("so it carries the usual stacking optimism: a training row's CNN value")
    print("came from a model fitted on folds that overlap the GBM's training set.")
    print("Nested CV removes this and is the version that goes in the paper.")

    section("reliability, GBM environment + CNN")
    print(reliability(y, preds["GBM environment + CNN"], n_bins=5).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    out = d[["SID", "ISO_TIME", "SEASON", "BASIN", "v0", "dv24", "ri"]].copy()
    for k, p in preds.items():
        out["p_" + k.replace(" ", "_").replace("+", "plus")] = p
    for j in range(emb_dim):
        out[f"emb{j:02d}"] = embs[:, j]
    p_out = PROCESSED_DIR / f"cnn_oof_{args.tag}.csv"
    out.to_csv(p_out, index=False)
    res.to_csv(REPORTS_DIR / f"cnn_summary_{args.tag}.csv")
    print(f"\nwrote {p_out}")
    print(f"wrote {REPORTS_DIR / f'cnn_summary_{args.tag}.csv'}")


if __name__ == "__main__":
    main()
