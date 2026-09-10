"""Shared paths and constants for the RI pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- IBTrACS
IBTRACS_VERSION = "v04r01"
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    f"{IBTRACS_VERSION}/access/csv/ibtracs.since1980.list.{IBTRACS_VERSION}.csv"
)
IBTRACS_CSV = RAW_DIR / IBTRACS_URL.rsplit("/", 1)[1]

# ---------------------------------------------------------------- SHIPS
# CIRA SHIPS developmental data. AL/CP/EP run through 2023 and carry the full
# satellite predictor block; the JTWC basins (IO/SH/WP) stop at 2021 and have a
# reduced predictor set. Keep the difference visible instead of pooling it away.
_SHIPS_BASE = "https://rammb-data.cira.colostate.edu/ships/data"
SHIPS_FILES = {
    "AL": f"{_SHIPS_BASE}/AL/lsdiaga_1982_2023_sat_ts_7day.txt",
    "CP": f"{_SHIPS_BASE}/CP/lsdiagc_1982_2023_sat_ts_7day.txt",
    "EP": f"{_SHIPS_BASE}/EP/lsdiage_1982_2023_sat_ts_7day.txt",
    "IO": f"{_SHIPS_BASE}/IO/lsdiagi_1990_2021_5day.txt",
    "SH": f"{_SHIPS_BASE}/SH/lsdiags_1998_2021_5day.txt",
    "WP": f"{_SHIPS_BASE}/WP/lsdiagw_1990_2021_5day.txt",
}
# SHIPS basin code -> IBTrACS BASIN code.
SHIPS_TO_IBTRACS_BASIN = {"AL": "NA", "CP": "EP", "EP": "EP",
                          "IO": "NI", "SH": "SI", "WP": "WP"}
SHIPS_RAW_DIR = RAW_DIR / "ships"

# ---------------------------------------------------------------- RI definition
# Kaplan & DeMaria: intensity change over the following 24 h, 1-min sustained wind.
RI_THRESHOLD_KT = 30.0
RI_WINDOW_H = 24

# Synoptic hours only. IBTrACS carries 3-hourly interpolated rows, but best-track
# intensity is only actually assigned at 00/06/12/18 UTC; the rest is interpolation
# and would inflate the sample with rows that carry no independent information.
SYNOPTIC_HOURS = (0, 6, 12, 18)

# Sample restrictions (Kaplan & DeMaria 2010/2015 convention).
MIN_INTENSITY_KT = 25.0     # at least a depression at t0
MIN_DIST2LAND_KM = 0        # >0 km == over water
TROPICAL_NATURES = ("TS",)  # IBTrACS NATURE: TS = tropical, keeps out ET/DS/SS/MX

# Satellite era for the imagery we will actually pair against.
MERGIR_START_YEAR = 2000    # 4 km merged IR, 2000-present
GRIDSAT_START_YEAR = 1980   # 8 km GridSat-B1, 1980-present

# Basins that have routine aircraft reconnaissance. Best-track intensity elsewhere
# is largely Dvorak-derived from the same IR imagery we feed the model, so these
# two groups must be evaluated separately. See proposal section 4.
RECON_BASINS = ("NA", "EP")
