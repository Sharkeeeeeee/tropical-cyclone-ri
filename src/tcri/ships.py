"""Parser for the CIRA SHIPS developmental (lsdiag) files.

Each case is a block of fixed-width lines, HEAD to LAST. Every line is N fields
of 5 characters then a 4-character tag, where N is 31 in the 7-day files
(AL/CP/EP) and 21 in the 5-day ones (IO/SH/WP) -- read it off the TIME line
rather than hard-coding it. Most tags are time series with t=0 at field 2;
HEAD, HIST, PSLV, MTPW, IRXX, IR00, IRM1, IRM3 and LAST are not. The IR records
put their 20 documented values in fields 2..21. 9999 is missing.

Two gotchas: LON is degrees *west* x10 (converted here), and these are analysis
values rather than forecast values, so anything built on them is optimistic
relative to an operational forecast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MISSING = 9999

# Tag -> divisor needed to reach physical units. Tags absent here are kept raw.
TS_SCALE = {
    "LAT": 10, "LON": 10, "CSST": 10, "OAGE": 10, "NAGE": 10, "RSST": 10,
    "DSST": 10, "DSTA": 10, "U200": 10, "U20C": 10, "V20C": 10, "E000": 10,
    "EPOS": 10, "ENEG": 10, "EPSS": 10, "ENSS": 10, "T000": 10, "TWAC": 10,
    "TWXC": 10, "G150": 10, "G200": 10, "G250": 10, "V000": 10, "V850": 10,
    "V500": 10, "V300": 10, "PENC": 10, "SHDC": 10, "SHGC": 10, "T150": 10,
    "T200": 10, "T250": 10, "SHRD": 10, "SHRS": 10, "SHRG": 10, "PENV": 10,
    "HE07": 10, "HE05": 10, "VVAV": 100, "VMFX": 100, "VVAC": 100,
}

# Time-series predictors taken at t=0.
TS_AT_T0 = [
    "VMAX", "MSLP", "TYPE", "LAT", "LON", "CSST", "CD20", "CD26", "COHC",
    "DTL", "OAGE", "NAGE", "RSST", "DSST", "DSTA", "U200", "U20C", "V20C",
    "E000", "EPOS", "ENEG", "EPSS", "ENSS", "RHLO", "RHMD", "RHHI", "Z850",
    "D200", "REFC", "PEFC", "T000", "R000", "Z000", "TWAC", "TWXC", "G150",
    "G200", "G250", "V000", "V850", "V500", "V300", "TGRD", "TADV", "PENC",
    "SHDC", "SDDC", "SHGC", "DIVC", "T150", "T200", "T250", "SHRD", "SHTD",
    "SHRS", "SHTS", "SHRG", "PENV", "VMPI", "VVAV", "VMFX", "VVAC", "HE07",
    "HE05", "O500", "O700", "CFLX",
]

# IR00 fields 1..20 (documentation numbering) -> (column name, divisor).
IR_FIELDS = {
    1: ("ir_time", 1),
    2: ("ir_avg_bt_0_200", 10),
    3: ("ir_std_bt_0_200", 10),
    4: ("ir_avg_bt_100_300", 10),
    5: ("ir_std_bt_100_300", 10),
    6: ("ir_pct_lt_m10", 1),
    7: ("ir_pct_lt_m20", 1),
    8: ("ir_pct_lt_m30", 1),
    9: ("ir_pct_lt_m40", 1),
    10: ("ir_pct_lt_m50", 1),
    11: ("ir_pct_lt_m60", 1),
    12: ("ir_max_bt_0_30", 10),
    13: ("ir_avg_bt_0_30", 10),
    14: ("ir_r_max_bt", 1),
    15: ("ir_min_bt_20_120", 10),
    16: ("ir_avg_bt_20_120", 10),
    17: ("ir_r_min_bt", 1),
}


def _num(tok: str):
    tok = tok.strip()
    if not tok:
        return np.nan
    try:
        v = float(tok)
    except ValueError:
        return np.nan
    return np.nan if abs(v) >= MISSING else v


def _fields(line: str, ncol: int):
    return [line[i * 5:(i + 1) * 5] for i in range(ncol)]


def _parse_head(line: str) -> dict:
    """' EMA  191014 00   30   23.3  167.5 1008 CP012019 ... HEAD'

    Parsed from the right, because the storm-name field can be blank.
    """
    toks = line[:line.rfind("HEAD")].split()
    if len(toks) < 7:
        return {}
    atcf = toks[-1]
    mslp, lon, lat, vmax, hh, yymmdd = (toks[-2], toks[-3], toks[-4],
                                        toks[-5], toks[-6], toks[-7])
    name = " ".join(toks[:-7]) or None
    try:
        yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    except ValueError:
        return {}
    year = 1900 + yy if yy >= 70 else 2000 + yy
    try:
        ts = pd.Timestamp(year=year, month=mm, day=dd, hour=int(hh))
    except ValueError:
        return {}
    return {
        "atcf_id": atcf, "name": name, "time": ts,
        "head_vmax": _num(vmax), "head_lat": _num(lat), "head_lon": _num(lon),
        "head_mslp": _num(mslp),
    }


def parse_file(path: Path, basin: str) -> pd.DataFrame:
    """Return one row per SHIPS case, with predictors evaluated at t=0."""
    text = Path(path).read_text(errors="replace")
    lines = text.splitlines()

    # Field count from the first TIME line; tag sits immediately after.
    ncol = None
    for ln in lines[:20]:
        if ln.rstrip().endswith("TIME"):
            ncol = len(ln[:ln.rfind("TIME")].split())
            break
    if ncol is None:
        raise ValueError(f"no TIME line found in {path}")
    tag_at = ncol * 5

    def tag_of(ln: str) -> str:
        return ln[tag_at:tag_at + 5].strip()

    i0 = 2 + 24 // 6          # field index of forecast hour +24
    rows, cur = [], None
    for ln in lines:
        tag = tag_of(ln)
        if tag == "HEAD":
            cur = _parse_head(ln)
            if cur:
                cur["basin_ships"] = basin
            continue
        if cur is None:
            continue
        if tag == "LAST":
            rows.append(cur)
            cur = None
            continue

        if tag in TS_AT_T0:
            f = _fields(ln, ncol)
            v = _num(f[2])
            cur[tag.lower()] = v / TS_SCALE.get(tag, 1) if v == v else np.nan
        elif tag == "DELV":
            f = _fields(ln, ncol)
            cur["per12"] = _num(f[0])          # intensity change -12 h -> 0
            cur["ships_dv24"] = _num(f[i0])    # 0 -> +24 h, for cross-checking
        elif tag == "INCV":
            f = _fields(ln, ncol)
            cur["incv_m6"] = _num(f[1])        # -12 -> -6 h
        elif tag == "HIST":
            f = _fields(ln, ncol)
            cur["hist_20kt"] = _num(f[2])
        elif tag in ("IR00", "IRM1", "IRM3"):
            f = _fields(ln, ncol)
            pre = {"IR00": "", "IRM1": "m1_", "IRM3": "m3_"}[tag]
            for idx, (name, div) in IR_FIELDS.items():
                v = _num(f[1 + idx])           # doc item k -> field k+1
                cur[pre + name] = v / div if v == v else np.nan
        elif tag == "MTPW":
            f = _fields(ln, ncol)
            for k in (1, 2, 19, 21):           # 0-200 avg/std, %TPW<45mm, 0-500 avg
                v = _num(f[1 + k])
                cur[f"mtpw_{k:02d}"] = v / 10 if (v == v and k != 19) else v

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # SHIPS longitude is degrees WEST; IBTrACS is degrees east.
    for c in ("lon", "head_lon"):
        if c in df:
            df[c] = -df[c]
    df["pot"] = df["vmpi"] - df["vmax"]
    return df


def parse_all(files: dict[str, Path]) -> pd.DataFrame:
    out = []
    for basin, path in files.items():
        df = parse_file(path, basin)
        print(f"  [{basin}] {len(df):>7,} cases   {Path(path).name}")
        out.append(df)
    return pd.concat(out, ignore_index=True)
