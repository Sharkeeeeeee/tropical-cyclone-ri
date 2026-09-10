"""Pre-flight checks before committing to a long run.

Each check is something that would silently corrupt results rather than crash,
which is the class of bug worth spending time on before an unattended run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcri.config import PROCESSED_DIR  # noqa: E402
from tcri.dataset import (BT_MEAN_K, BT_STD_K, FILL_K, PatchDataset,  # noqa: E402
                          patch_path)
from tcri.evaluate import brier_skill_score, reliability, summary  # noqa: E402
from tcri.gridsat import PATCH_KM, PATCH_N, target_grid  # noqa: E402
from tcri.model import PatchCNN  # noqa: E402
from tcri.structure import geometry, sequence_features  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail or ""))
        print(f"[{PASS}] {name}" + (f"  -- {detail}" if detail else ""))
    except AssertionError as e:
        results.append((False, name, str(e)))
        print(f"[{FAIL}] {name}\n         {e}")
    except Exception as e:                       # noqa: BLE001
        results.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[{FAIL}] {name}\n         {type(e).__name__}: {e}")


# --------------------------------------------------------------- scripts load
def scripts_importable():
    """Execute each script's module body without calling main().

    Running `--help` was the obvious check and the wrong one: scripts that do
    not use argparse ignore the flag and run their whole job, which for the
    patch demo means waiting on a network service that may be down.
    Module-level execution catches syntax and import errors, which is what this
    check is for, and stops at the `if __name__ == "__main__"` guard.
    """
    probe = ("import importlib.util as u, sys; "
             "s = u.spec_from_file_location('probe', sys.argv[1]); "
             "m = u.module_from_spec(s); s.loader.exec_module(m)")
    bad = []
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    for p in scripts:
        if p.name == Path(__file__).name:
            continue
        r = subprocess.run([sys.executable, "-c", probe, str(p)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            tail = (r.stderr.strip().splitlines() or ["?"])[-1]
            bad.append(f"{p.name}: {tail[:70]}")
    assert not bad, "; ".join(bad)
    return f"{len(scripts) - 1} scripts execute their imports cleanly"


# ------------------------------------------------------------------- geometry
def patch_grid_is_metric():
    """A patch must span the same physical distance at every latitude."""
    for lat0 in (5.0, 25.0, 45.0):
        lat, lon = target_grid(lat0, -60.0)
        # North-south extent in km, from the corner rows.
        dy_km = (lat[0, PATCH_N // 2] - lat[-1, PATCH_N // 2]) * 111.32
        want = (PATCH_N - 1) * PATCH_KM
        assert abs(dy_km - want) < 1.0, f"lat {lat0}: {dy_km:.1f} vs {want} km"
        # East-west extent at the centre row, corrected for latitude.
        mid = PATCH_N // 2
        dx_km = ((lon[mid, -1] - lon[mid, 0])
                 * 111.32 * np.cos(np.radians(lat0)))
        assert abs(dx_km - want) < 6.0, f"lon {lat0}: {dx_km:.1f} vs {want} km"
    return "1016 km span held at 5N, 25N, 45N"


def dateline_is_continuous():
    lat, lon = target_grid(15.0, 179.5)
    assert lon.min() >= -180.0 and lon.max() < 180.0, "longitude not wrapped"
    # Crossing the seam must produce both signs, not a clamped edge.
    assert (lon > 170).any() and (lon < -170).any(), "seam not spanned"
    return "patch at 179.5E spans the dateline in both signs"


# -------------------------------------------------------------------- dataset
def hemisphere_mirror_applies():
    idx = pd.read_csv(PROCESSED_DIR / "patches" / "index_NA_2020.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    rows = idx.loc[idx["frames_ok"] == 4, "row"].to_numpy()[:1]
    p = patch_path(PROCESSED_DIR, "NA_2020")

    north = PatchDataset(p, idx, rows, augment=False)
    x_n, _, _ = north[0]

    flipped = idx.copy()
    flipped.loc[flipped["row"].isin(rows), "LAT"] *= -1     # pretend southern
    south = PatchDataset(p, index=flipped, rows=rows, augment=False)
    x_s, _, _ = south[0]

    assert np.allclose(x_n.numpy()[:, ::-1, :], x_s.numpy()), \
        "southern-hemisphere patch is not the north-south mirror"
    return "SH patches are mirrored, NH are not"


def normalisation_is_sane():
    idx = pd.read_csv(PROCESSED_DIR / "patches" / "index_NA_2020.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    rows = idx.loc[idx["frames_ok"] == 4, "row"].to_numpy()[:64]
    ds = PatchDataset(patch_path(PROCESSED_DIR, "NA_2020"), idx, rows)
    xs = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    assert np.isfinite(xs).all(), "NaN or inf survived normalisation"
    assert -4 < xs.min() and xs.max() < 4, f"range {xs.min():.2f}..{xs.max():.2f}"
    # The fill value must map to a plausible normalised number, not an outlier.
    fill = (FILL_K - BT_MEAN_K) / BT_STD_K
    assert abs(fill) < 2, f"fill maps to {fill:.2f}"
    return f"range {xs.min():.2f}..{xs.max():.2f}, fill -> {fill:.2f}"


def augmentation_preserves_content():
    idx = pd.read_csv(PROCESSED_DIR / "patches" / "index_NA_2020.csv",
                      converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    rows = idx.loc[idx["frames_ok"] == 4, "row"].to_numpy()[:1]
    p = patch_path(PROCESSED_DIR, "NA_2020")
    plain = PatchDataset(p, idx, rows, augment=False)[0][0].numpy()
    aug = PatchDataset(p, idx, rows, augment=True)
    seen = set()
    for _ in range(30):
        x = aug[0][0].numpy()
        assert x.shape == plain.shape, "augmentation changed shape"
        assert abs(float(x.sum()) - float(plain.sum())) < 1.0, \
            "rotation changed pixel content"
        seen.add(round(float(x[0, 0, 0]), 4))
    assert len(seen) > 1, "augmentation never varied the sample"
    return f"{len(seen)} distinct orientations, content preserved"


# ---------------------------------------------------------------------- model
def model_accepts_frame_counts():
    import torch
    out = []
    for f in (1, 2, 4):
        m = PatchCNN(in_frames=f, width=8)
        y = m(torch.randn(3, f, PATCH_N, PATCH_N))
        assert y.shape == (3,), f"frames={f} gave {tuple(y.shape)}"
        out.append(f)
    return f"forward ok for frames={out}"


def model_is_deterministic_in_eval():
    import torch
    m = PatchCNN(in_frames=4, width=8).eval()
    x = torch.randn(2, 4, PATCH_N, PATCH_N)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b), "eval mode is not deterministic (dropout left on?)"
    return "dropout disabled in eval"


# ------------------------------------------------------------------ structure
def structure_handles_missing_data():
    seq = np.full((4, PATCH_N, PATCH_N), 250.0, dtype=np.float32)
    seq[0, :40, :] = np.nan                    # a partly missing frame
    f = sequence_features(seq, PATCH_KM)
    assert len(f) > 100, f"only {len(f)} features"
    finite = sum(np.isfinite(v) for v in f.values())
    assert finite > 0.5 * len(f), "most features went NaN on a partial frame"

    allnan = np.full((4, PATCH_N, PATCH_N), np.nan, dtype=np.float32)
    f2 = sequence_features(allnan, PATCH_KM)    # must not raise
    assert len(f2) == len(f), "feature set changed shape on all-NaN input"
    return f"{len(f)} features; degrades to NaN without crashing"


def structure_geometry_matches_patch():
    g = geometry(PATCH_N, PATCH_KM)
    assert abs(g["r"].max() - np.hypot(508, 508)) < 1.0, "radius grid wrong"
    assert g["core"].sum() > 0 and g["rings"][(0, 50)].sum() > 0, "empty masks"
    return f"max radius {g['r'].max():.0f} km"


# ------------------------------------------------------------------- metrics
def metrics_handle_degenerate_input():
    y = np.zeros(50, dtype=int)
    p = np.random.rand(50)
    bss = brier_skill_score(y, p)
    assert not np.isfinite(bss) or bss <= 0, f"BSS {bss} with no positives"
    y2 = np.array([0] * 45 + [1] * 5)
    s = summary(y2, np.random.rand(50), "t")
    assert set(("ROC_AUC", "PR_AUC", "Brier", "BSS")) <= set(s), "missing metrics"
    r = reliability(y2, np.random.rand(50), n_bins=10)
    assert r["n"].sum() == 50, "reliability bins lost samples"
    return "no-positive and small-sample cases handled"


def oof_files_align_with_labels():
    p = PROCESSED_DIR / "baseline_oof.csv"
    if not p.exists():
        return "baseline_oof.csv absent, skipped"
    oof = pd.read_csv(p, converters={"BASIN": str}, parse_dates=["ISO_TIME"])
    assert not oof.duplicated(["SID", "ISO_TIME"]).any(), "duplicate keys"
    pcols = [c for c in oof.columns if c.startswith("p_")]
    for c in pcols:
        v = oof[c].to_numpy()
        assert np.isfinite(v).all(), f"{c} has NaN"
        assert 0 <= v.min() and v.max() <= 1, f"{c} outside [0,1]"
    assert oof["BASIN"].eq("NA").any(), \
        "no North Atlantic rows -- BASIN 'NA' parsed as missing again"
    return f"{len(oof):,} rows, {len(pcols)} prediction columns, all in [0,1]"


def main():
    print("=" * 72)
    print("pre-flight checks")
    print("=" * 72)
    for name, fn in [
        ("scripts import cleanly", scripts_importable),
        ("patch grid is metric, not degrees", patch_grid_is_metric),
        ("dateline patches stay continuous", dateline_is_continuous),
        ("southern-hemisphere mirror applies", hemisphere_mirror_applies),
        ("normalisation leaves no NaN, sane range", normalisation_is_sane),
        ("augmentation rotates without changing content",
         augmentation_preserves_content),
        ("model accepts 1/2/4 frames", model_accepts_frame_counts),
        ("model deterministic in eval", model_is_deterministic_in_eval),
        ("structure features survive missing data",
         structure_handles_missing_data),
        ("structure geometry matches patch geometry",
         structure_geometry_matches_patch),
        ("metrics handle degenerate input", metrics_handle_degenerate_input),
        ("saved out-of-fold predictions are well formed",
         oof_files_align_with_labels),
    ]:
        check(name, fn)

    n_fail = sum(1 for ok, _, _ in results if not ok)
    print("=" * 72)
    print(f"{len(results) - n_fail}/{len(results)} passed")
    if n_fail:
        print("\nfailures:")
        for ok, name, detail in results:
            if not ok:
                print(f"  {name}: {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
