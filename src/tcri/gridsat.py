"""Storm-centred IR patches from GridSat-B1.

GridSat-B1 is 3-hourly, 0.07 deg (~8 km), 70S-70N, 1980-present. Its synoptic
hours cover all four best-track times, so t0-9/-6/-3 need no interpolation.

Two things to know:

Data comes through NCEI's OPeNDAP server, subset server-side. A whole file is
~50 MB; one storm's box is ~0.08 MB. `irwin_cdr` is stored as a single
compressed chunk spanning the globe, so no client-side trick -- byte ranges
included -- can fetch less than that. When OPeNDAP is down the fallbacks pull
whole files from NCEI or the AWS Open Data mirror, at ~650x the transfer.

Patches are resampled onto a fixed kilometre grid, not cut as a degree box. A
degree box shrinks physically with latitude, and a model could learn latitude
from the framing alone.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates

OPENDAP = ("https://www.ncei.noaa.gov/thredds/dodsC/cdr/gridsat/"
           "{y}/GRIDSAT-B1.{y}.{m:02d}.{d:02d}.{h:02d}.v02r01.nc")

# Fallbacks, used when OPeNDAP is down. Both pull whole files.
HTTP_FILE = ("https://www.ncei.noaa.gov/data/"
             "geostationary-ir-channel-brightness-temperature-gridsat-b1/"
             "access/{y}/GRIDSAT-B1.{y}.{m:02d}.{d:02d}.{h:02d}.v02r01.nc")

# AWS Open Data mirror; anonymous, and faster than NCEI from most places.
S3_FILE = ("https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/"
           "data/{y}/GRIDSAT-B1.{y}.{m:02d}.{d:02d}.{h:02d}.v02r01.nc")

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (research; TC RI patch extraction)"}
FILE_SOURCES = {"http": HTTP_FILE, "s3": S3_FILE}

# The sparse-index coordinates carry millions of elements and are irrelevant
# here; reading them dominates the time to open a remote file.
DROP_VARS = ["sparse3ir", "sparse2ir", "sparse2wv", "sparse2vs", "satid_ir3",
             "irwin_2", "irwvp_2", "vschn_2", "satid_ir", "satid_wv",
             "satid_vs", "irwin_vza_adj", "satlat", "satlon", "satrad",
             "satname", "b1ufile", "b1file"]

KM_PER_DEG = 111.32
PATCH_N = 128      # 128 x 128 samples
PATCH_KM = 8.0     # 8 km spacing -> a 1024 km box, +/- 512 km from the centre


def url_for(ts: pd.Timestamp) -> str:
    return OPENDAP.format(y=ts.year, m=ts.month, d=ts.day, h=ts.hour)


def open_time(ts: pd.Timestamp, variable: str = "irwin_cdr",
              retries: int = 4, base_delay: float = 3.0) -> xr.DataArray:
    """Open one remote GridSat file, retrying transient failures.

    NCEI returns 502/503 under load. Without a retry each blip permanently
    drops a frame, since the extractor records the failure and moves on. The
    jitter stops a worker pool synchronising into a thundering herd.
    """
    delay = base_delay
    last: Exception | None = None
    for attempt in range(retries):
        try:
            ds = xr.open_dataset(url_for(ts), drop_variables=DROP_VARS)
            return ds[variable].isel(time=0)
        except Exception as e:                       # noqa: BLE001
            last = e
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, delay * 0.5))
                delay *= 2
    raise last if last else RuntimeError(f"could not open {ts}")


def http_url_for(ts: pd.Timestamp, source: str = "http") -> str:
    tmpl = FILE_SOURCES.get(source, HTTP_FILE)
    return tmpl.format(y=ts.year, m=ts.month, d=ts.day, h=ts.hour)


def fetch_file(ts: pd.Timestamp, dest: Path, retries: int = 4,
               base_delay: float = 3.0, source: str = "http") -> Path:
    """Download one whole GridSat file. Caller deletes it after use."""
    import requests

    delay = base_delay
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(http_url_for(ts, source), headers=HTTP_HEADERS,
                              stream=True, timeout=300) as r:
                r.raise_for_status()
                expect = int(r.headers.get("content-length", 0))
                got = 0
                with open(dest, "wb") as fh:
                    for block in r.iter_content(chunk_size=1 << 20):
                        fh.write(block)
                        got += len(block)
            if expect and got != expect:
                raise IOError(f"truncated: {got} of {expect} bytes")
            return dest
        except Exception as e:                       # noqa: BLE001
            last = e
            dest.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(delay + random.uniform(0, delay * 0.5))
                delay *= 2
    raise last if last else RuntimeError(f"could not fetch {ts}")


def open_local(path: Path, variable: str = "irwin_cdr") -> xr.DataArray:
    ds = xr.open_dataset(path, drop_variables=DROP_VARS)
    return ds[variable].isel(time=0).load()


def target_grid(lat0: float, lon0: float, n: int = PATCH_N,
                km: float = PATCH_KM):
    """Latitudes and longitudes of an n x n grid of `km`-spaced points.

    Row 0 is the northern edge, so the array is displayed in the same
    orientation as a conventional map.
    """
    off = (np.arange(n) - (n - 1) / 2.0) * km
    dy = -off[:, None]                      # north at the top
    dx = off[None, :]
    lat = lat0 + dy / KM_PER_DEG
    lat = np.clip(lat, -89.9, 89.9)
    # Longitude spacing uses each row's own latitude, so the patch is a local
    # equidistant grid rather than a box that stretches towards the poles.
    lon = lon0 + dx / (KM_PER_DEG * np.cos(np.radians(lat)))
    lon = (lon + 180.0) % 360.0 - 180.0     # wrap into [-180, 180)
    lat = np.broadcast_to(lat, lon.shape)
    return np.ascontiguousarray(lat), np.ascontiguousarray(lon)


def _window(da: xr.DataArray, lat, lon, pad: int = 3):
    """Fetch the smallest source window covering the target, dateline-safe.

    Returns the values plus the origin and step of a *monotonic* local axis, so
    that a longitude range spanning the dateline can be interpolated normally.
    """
    lats = da["lat"].values
    lons = da["lon"].values
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
    nlon = lons.size

    iy = np.clip(np.round((lat - lats[0]) / dlat).astype(int), 0, lats.size - 1)
    y0 = max(int(iy.min()) - pad, 0)
    y1 = min(int(iy.max()) + pad + 1, lats.size)

    ix = np.round((lon - lons[0]) / dlon).astype(int) % nlon
    # Unwrap the longitude indices: if they straddle the seam, shift the low
    # side up by a full revolution so the window is contiguous.
    span_direct = int(ix.max()) - int(ix.min())
    if span_direct > nlon // 2:
        ix = np.where(ix < nlon // 2, ix + nlon, ix)
    x0 = int(ix.min()) - pad
    x1 = int(ix.max()) + pad + 1

    if x1 <= nlon:
        vals = da.isel(lat=slice(y0, y1), lon=slice(x0 % nlon, x1)).values
    else:                                    # crosses the dateline: two reads
        a = da.isel(lat=slice(y0, y1), lon=slice(x0 % nlon, nlon)).values
        b = da.isel(lat=slice(y0, y1), lon=slice(0, x1 - nlon)).values
        vals = np.concatenate([a, b], axis=1)

    lat_org = lats[0] + y0 * dlat
    lon_org = lons[0] + x0 * dlon           # may exceed 180; consistent with ix
    return vals, lat_org, dlat, lon_org, dlon


def storm_patch(da: xr.DataArray, lat0: float, lon0: float,
                n: int = PATCH_N, km: float = PATCH_KM):
    """Bilinearly resample a storm-centred patch. Returns (patch, nan_fraction)."""
    lat, lon = target_grid(lat0, lon0, n, km)
    vals, lat_org, dlat, lon_org, dlon = _window(da, lat, lon)

    lon_u = lon.copy()
    if lon_org > 180.0 - 1e-9 or lon_org + dlon * vals.shape[1] > 180.0:
        lon_u = np.where(lon_u < lon_org - 1e-9, lon_u + 360.0, lon_u)

    fy = (lat - lat_org) / dlat
    fx = (lon_u - lon_org) / dlon

    vals = np.asarray(vals, dtype=np.float32)
    bad = ~np.isfinite(vals)
    filled = np.where(bad, 0.0, vals)

    out = map_coordinates(filled, [fy, fx], order=1, mode="nearest")
    # Interpolate the missing mask the same way, so partially-missing target
    # pixels are marked rather than silently filled with zero.
    miss = map_coordinates(bad.astype(np.float32), [fy, fx], order=1,
                           mode="nearest")
    out = np.where(miss > 0.5, np.nan, out).astype(np.float32)
    return out, float(np.isnan(out).mean())


def sequence_times(t0: pd.Timestamp, lags_h=(9, 6, 3, 0)):
    """Frame times for one case, earliest first."""
    return [t0 - pd.Timedelta(hours=h) for h in sorted(lags_h, reverse=True)]
