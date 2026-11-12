"""Directories, tariff, battery defaults and horizon bands.

Paths
-----
No location is hard-coded.  At import time the module resolves three roots,
each of which can be overridden by an environment variable:

======================  ==========================  ==========================
variable                default                     contents
======================  ==========================  ==========================
``EMSX_RAW_DIR``        ``<repo>/data/emsx``        the 70 per-site CSV files
``FORESURGERY_WORK``    ``<repo>/work``             all generated artefacts
``FORSURGERY_RESULTS``  ``$FORESURGERY_WORK/results``  parquet tables, reports
======================  ==========================  ==========================

``<repo>`` is the directory that contains ``pyproject.toml``; when the package
is installed non-editable and that directory cannot be found, the current
working directory is used instead.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = [
    "ROOT", "RAW_DIR", "WORK_DIR", "WAREHOUSE_DIR", "RESULTS_DIR",
    "MODELS_DIR", "FIGURES_DIR", "DT", "BANDS", "BAND_LABELS", "BATT",
    "TOU_SEGMENTS", "DEFAULT_C_DEM", "tou_price_vector", "battery_specs",
    "ensure_dirs",
]


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _guess_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() or (parent / "scripts").is_dir():
            return parent
    return Path.cwd()


def _resolve(var: str, default: str | Path, base: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else (base / default).resolve()


ROOT = _guess_root()

RAW_DIR = _resolve("EMSX_RAW_DIR", "data/emsx", ROOT)
WORK_DIR = _resolve("FORESURGERY_WORK", "work", ROOT)
WAREHOUSE_DIR = WORK_DIR / "warehouse"
RESULTS_DIR = _resolve("FORESURGERY_RESULTS", WORK_DIR / "results", ROOT)
MODELS_DIR = WORK_DIR / "models"
FIGURES_DIR = WORK_DIR / "figures"

# Kept as short aliases: the analysis scripts read ``RES`` everywhere.
RES = RESULTS_DIR
MODELS = MODELS_DIR
FIGS = FIGURES_DIR


def ensure_dirs() -> None:
    """Create every generated-artefact directory (idempotent)."""
    for d in (WAREHOUSE_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Discretisation
# --------------------------------------------------------------------------
DT = 0.25  # hours per step (15-min resolution, 96 steps = 24 h)


# --------------------------------------------------------------------------
# Horizon bands — the axis of the surgery. Steps are forecast offsets 0..95.
# --------------------------------------------------------------------------
BANDS = {
    "h0_2": (0, 8),       # 0-2 h
    "h2_6": (8, 24),      # 2-6 h
    "h6_12": (24, 48),    # 6-12 h
    "h12_24": (48, 96),   # 12-24 h
}
BAND_LIST = list(BANDS)
BAND_LABELS = ["0-2 h", "2-6 h", "6-12 h", "12-24 h"]


# --------------------------------------------------------------------------
# Battery — one unit per site, sized from the site's own load statistics.
# --------------------------------------------------------------------------
BATT = {
    "eta_c": 0.95,          # charge efficiency
    "eta_d": 0.95,          # discharge efficiency
    "soc_min": 0.10,        # fraction of E_cap
    "soc_max": 0.90,
    "soc_init": 0.50,
    "e_daily_share": 0.10,  # E_cap = share * mean daily consumption  [kWh]
    "e_hours": 4.0,         # P_rated = E_cap / e_hours              [kW]
}


def battery_specs(load: np.ndarray, e_share: float | None = None) -> dict:
    """Energy/power rating from the site's mean daily consumption."""
    share = BATT["e_daily_share"] if e_share is None else e_share
    daily_mean = float(np.nanmean(load)) * 96.0
    e_cap = max(share * daily_mean, 10.0)
    p_rated = e_cap / BATT["e_hours"]
    return {"p_kw": p_rated, "e_kwh": e_cap}


# --------------------------------------------------------------------------
# Time-of-use tariff on LOCAL time of day (period_id 0..95), EUR/kWh.
# Generic European industrial three-period structure; the paper's robustness
# check replaces it with the Italian F1/F2/F3 structure.
# --------------------------------------------------------------------------
TOU_SEGMENTS = [
    ((0, 7), 0.08),     # night off-peak
    ((7, 9), 0.18),     # morning peak
    ((9, 17), 0.12),    # daytime shoulder
    ((17, 21), 0.18),   # evening peak
    ((21, 24), 0.08),   # evening off-peak
]


def tou_price_vector() -> np.ndarray:
    """Price for each of the 96 local periods (period_id == local 15-min slot)."""
    price = np.zeros(96)
    for (h0, h1), p in TOU_SEGMENTS:
        price[h0 * 4:h1 * 4] = p
    return price


# --------------------------------------------------------------------------
# Demand charge, EUR per kW per day. 0.5 EUR/kW/day is the daily proxy for the
# ~15 EUR/kW/month charge that industrial sites in the dataset actually face.
# --------------------------------------------------------------------------
DEFAULT_C_DEM = 0.5
