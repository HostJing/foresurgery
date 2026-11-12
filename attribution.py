"""Cost-weighted band attribution and bootstrap intervals.

Why cost-weighted
-----------------
The naive share ``(c_base - c_heal) / (c_base - c_perfect)`` is unstable: when
the denominator is a few cents the ratio explodes and, over a fleet, such issue
times dominate the mean.  The paper therefore aggregates the *numerator and the
denominator separately*::

    share(h) = sum_t [c_base(t) - c_heal_h(t)] / sum_t [c_base(t) - c_perfect(t)]

over issue times whose regret is at least 5 % of the median base cost.  The
ratio of sums is the share of the fleet's recoverable regret that band ``h``
accounts for, which is the quantity a scheduling decision can act on.

Intervals come from a paired bootstrap over issue times (500 seeded resamples).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BAND_LIST

__all__ = ["VALID_EPS", "prepare_attribution", "cw_stats", "cw_share",
           "band_table", "regret_profile"]

VALID_EPS = 0.05  # regret must exceed 5 % of the median base cost


def prepare_attribution(df: pd.DataFrame, prefix: str = "heal",
                        valid_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn a surgery table into an attribution frame.

    Returns ``(h, kd)``: ``h`` holds one row per (issue time, axis, band) with
    ``recovered = denom - regret``, and ``kd`` the per-issue-time denominator
    table.  With ``valid_only`` (default) only issue times whose regret exceeds
    5 % of the median base cost are kept, which is the filter used throughout
    the paper.
    """
    lvl = df[df.scenario == "base"].groupby(["site_id", "i0"]).cost.first()
    per = df[df.scenario == "perfect"].groupby(["site_id", "i0"]).cost.first()
    kd = pd.DataFrame({"denom": lvl - per}).reset_index()
    med = float(df[df.scenario == "base"].cost.median())
    kd["valid"] = kd.denom > VALID_EPS * med
    kd["key"] = kd.site_id.astype(str) + ":" + kd.i0.astype(str)

    h = df[df.scenario.str.startswith(prefix)].copy()
    h = h.drop(columns=["denom"], errors="ignore")
    h["axis"] = np.where(h.scenario.str.startswith(f"{prefix}L"), "load", "pv")
    h["band"] = h.scenario.str.split("_", n=1).str[1]
    h["key"] = h.site_id.astype(str) + ":" + h.i0.astype(str)
    if "regret" not in h.columns:
        perf_map = df[df.scenario == "perfect"].set_index(["site_id", "i0"]).cost
        h["regret"] = h.cost - h.set_index(["site_id", "i0"]).index.map(perf_map)
    h = h.merge(kd[["key", "denom", "valid"]], on="key", how="inner")
    h["recovered"] = h.denom - h.regret
    return (h[h.valid] if valid_only else h), kd


def cw_stats(sub: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> tuple:
    """Cost-weighted share with a paired bootstrap CI over issue times."""
    denom, rec = sub.denom.sum(), sub.recovered.sum()
    share = rec / denom if denom > 1e-9 else np.nan
    keys = sub.key.unique()
    if len(keys) < 5:
        return share, np.nan, np.nan
    idx = sub.groupby("key")[["recovered", "denom"]].sum()
    rec_v, den_v = idx.recovered.to_numpy(), idx.denom.to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    R = rec_v[draws].sum(axis=1)
    D = den_v[draws].sum(axis=1)
    boot = np.divide(R, D, out=np.full(n_boot, np.nan), where=D > 1e-9)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return share, lo, hi


def cw_share(h: pd.DataFrame, axis: str, band: str) -> float:
    """Point estimate of the cost-weighted share, in percent."""
    sub = h[(h.axis == axis) & (h.band == band)]
    return 100 * sub.recovered.sum() / sub.denom.sum() if len(sub) else np.nan


def band_table(h: pd.DataFrame, valid_only: bool = True) -> pd.DataFrame:
    """Share [%], bootstrap CI and issue-time count for every (axis, band)."""
    sub = h[h.valid] if valid_only else h
    rows = []
    for axis in ("load", "pv"):
        for band in BAND_LIST:
            s, lo, hi = cw_stats(sub[(sub.axis == axis) & (sub.band == band)])
            rows.append({"axis": axis, "band": band, "share_pct": 100 * s,
                         "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                         "n_issue_times": sub[(sub.axis == axis)
                                              & (sub.band == band)].key.nunique()})
    return pd.DataFrame(rows)


def regret_profile(df: pd.DataFrame) -> pd.Series:
    """Per-site mean regret as a percentage of the perfect-foresight bill."""
    lvl = df[df.scenario == "base"].groupby(["site_id", "i0"]).cost.first()
    per = df[df.scenario == "perfect"].groupby(["site_id", "i0"]).cost.first()
    return 100 * (lvl - per).groupby(level=0).mean() / per.groupby(level=0).mean()
