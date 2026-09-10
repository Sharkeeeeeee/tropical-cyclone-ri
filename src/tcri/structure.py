"""Structural features from storm-centred IR patches.

Radial profiles, cold-cloud fractions, eye contrast, azimuthal wavenumbers and
deviation-angle variance, plus the 3 h and 9 h change in each -- about 140
features, a few milliseconds per patch.

These exist to separate "spatial structure carries signal" from "deep learning
extracts it". Without them a CNN result cannot distinguish the two.

Brightness temperature is degrees Celsius throughout; radii are kilometres.
"""
from __future__ import annotations

import numpy as np

RINGS = ((0, 50), (50, 100), (100, 200), (200, 300), (300, 500))
COLD_REGIONS = ((0, 100), (0, 200), (50, 200), (100, 300))
THRESHOLDS_C = (-30, -40, -50, -60, -70)
WN_RINGS = ((50, 150), (100, 300))
DAV_RADII = (200, 350)
N_AZIMUTH = 36
_GEOM_CACHE: dict = {}


def geometry(n: int, km: float) -> dict:
    """Radius, azimuth and ring masks for an n x n patch of `km`-spaced points."""
    key = (n, km)
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    off = (np.arange(n) - (n - 1) / 2.0) * km
    y = -off[:, None]                       # north positive; row 0 is the top
    x = off[None, :]
    r = np.hypot(y, x)
    theta = np.arctan2(y, x)                # radial direction at each pixel
    g = {
        "r": r, "theta": theta, "km": km, "n": n,
        "rings": {rr: (r >= rr[0]) & (r < rr[1]) for rr in RINGS},
        "cold": {rr: (r >= rr[0]) & (r < rr[1]) for rr in COLD_REGIONS},
        "wn": {rr: (r >= rr[0]) & (r < rr[1]) for rr in WN_RINGS},
        "dav": {R: r <= R for R in DAV_RADII},
        "core": r <= 30,
        "ring_around_core": (r > 30) & (r <= 80),
        "azbin": np.clip(((theta + np.pi) / (2 * np.pi) * N_AZIMUTH).astype(int),
                         0, N_AZIMUTH - 1),
    }
    _GEOM_CACHE[key] = g
    return g


def _safe(fn, values, default=np.nan):
    v = values[np.isfinite(values)]
    return float(fn(v)) if v.size else default


def _azimuthal_means(bt, mask, azbin):
    """Mean brightness temperature in each azimuthal sector of a ring."""
    out = np.full(N_AZIMUTH, np.nan)
    sel = mask & np.isfinite(bt)
    if not sel.any():
        return out
    b = azbin[sel]
    v = bt[sel]
    tot = np.bincount(b, weights=v, minlength=N_AZIMUTH)
    cnt = np.bincount(b, minlength=N_AZIMUTH)
    good = cnt > 0
    out[good] = tot[good] / cnt[good]
    return out


def _wavenumber_amplitudes(azmeans):
    """Normalised wavenumber-1 and -2 amplitudes around a ring."""
    v = azmeans.copy()
    if np.isnan(v).any():
        good = ~np.isnan(v)
        if good.sum() < N_AZIMUTH // 2:
            return np.nan, np.nan
        idx = np.arange(N_AZIMUTH)
        v[~good] = np.interp(idx[~good], idx[good], v[good], period=N_AZIMUTH)
    v = v - v.mean()
    f = np.fft.rfft(v)
    amp = 2.0 * np.abs(f) / N_AZIMUTH
    return float(amp[1]), float(amp[2])


def deviation_angle_variance(bt, g, radius, grad_min=0.05):
    """Piñeros/Ritchie DAV, in degrees squared.

    The brightness-temperature gradient direction is compared with the radial
    direction at every pixel inside `radius`. Gradient sign is ambiguous for
    this purpose, so deviations are folded onto [-90, 90). A perfectly
    axisymmetric cloud field gives zero variance.
    """
    filled = np.where(np.isfinite(bt), bt, np.nan)
    if not np.isfinite(filled).any():        # a wholly missing frame
        return np.nan
    gy, gx = np.gradient(np.nan_to_num(filled, nan=float(np.nanmean(filled))),
                         g["km"])
    gy = -gy                                 # row index increases southward
    mag = np.hypot(gx, gy)
    sel = g["dav"][radius] & (mag > grad_min) & np.isfinite(bt)
    if sel.sum() < 50:
        return np.nan
    dev = np.degrees(np.arctan2(gy[sel], gx[sel]) - g["theta"][sel])
    dev = (dev + 90.0) % 180.0 - 90.0
    return float(np.var(dev))


def frame_features(bt_celsius: np.ndarray, g: dict) -> dict:
    """Structural features for one frame. `bt_celsius` may contain NaN."""
    bt = bt_celsius
    f: dict[str, float] = {}

    for (a, b), m in g["rings"].items():
        v = bt[m]
        f[f"ring{a}_{b}_mean"] = _safe(np.mean, v)
        f[f"ring{a}_{b}_std"] = _safe(np.std, v)
        f[f"ring{a}_{b}_min"] = _safe(np.min, v)

    for (a, b), m in g["cold"].items():
        v = bt[m]
        v = v[np.isfinite(v)]
        for t in THRESHOLDS_C:
            f[f"cold{a}_{b}_lt{abs(t)}"] = (100.0 * np.mean(v < t)
                                            if v.size else np.nan)

    core = _safe(np.mean, bt[g["core"]])
    ring = _safe(np.mean, bt[g["ring_around_core"]])
    f["core_mean"] = core
    f["core_min"] = _safe(np.min, bt[g["core"]])
    # Positive when the centre is warmer than the surrounding ring: an eye.
    f["eye_contrast"] = core - ring if np.isfinite(core) and np.isfinite(ring) else np.nan

    for (a, b), m in g["wn"].items():
        az = _azimuthal_means(bt, m, g["azbin"])
        w1, w2 = _wavenumber_amplitudes(az)
        f[f"wn1_{a}_{b}"] = w1
        f[f"wn2_{a}_{b}"] = w2
        f[f"azspread_{a}_{b}"] = _safe(np.std, az)

    for R in DAV_RADII:
        f[f"dav_{R}"] = deviation_angle_variance(bt, g, R)

    # Where the deep convection sits relative to the centre.
    means = [f[f"ring{a}_{b}_mean"] for a, b in RINGS]
    if np.all(np.isfinite(means)):
        f["r_coldest_ring"] = float(RINGS[int(np.argmin(means))][0])
    else:
        f["r_coldest_ring"] = np.nan
    return f


def sequence_features(seq_kelvin: np.ndarray, km: float,
                      lags_h=(9, 6, 3, 0)) -> dict:
    """Features for a patch sequence: t0 values plus changes over the window.

    `seq_kelvin` is (frames, n, n), earliest first, in kelvin.
    """
    n = seq_kelvin.shape[-1]
    g = geometry(n, km)
    frames = [frame_features(np.asarray(s, dtype=np.float32) - 273.15, g)
              for s in seq_kelvin]

    out = {f"{k}": v for k, v in frames[-1].items()}
    keys = list(frames[-1])
    # Tendencies over the window.
    for lag_i, name in ((0, "d9h"), (len(frames) - 2, "d3h")):
        if 0 <= lag_i < len(frames) - 1:
            for k in keys:
                a, b = frames[-1][k], frames[lag_i][k]
                out[f"{k}_{name}"] = (a - b if np.isfinite(a) and np.isfinite(b)
                                      else np.nan)
    return out


def feature_names(n: int = 128, km: float = 8.0) -> list[str]:
    dummy = np.full((4, n, n), 250.0, dtype=np.float32)
    return list(sequence_features(dummy, km))
