"""The heal/degrade operator and the surgery campaign.

The operator acts on one horizon band of an issued forecast and leaves every
other band untouched::

    out[lo:hi] = actual[lo:hi] + k * (issued[lo:hi] - actual[lo:hi])

with ``k = 0`` healing the band (replace by the realised values), ``k = 1``
keeping the forecast as issued, and ``k > 1`` amplifying the issued error
k-fold about the realisation.  ``k = 1`` is therefore the identity, which makes
the heal and degrade campaigns directly comparable: both are the same operator
applied with a different dose.

An issue time ``t`` produces the scenario grid

    perfect | base | healL_{band} x4 | healP_{band} x4

where each scenario is dispatched with the same LP, tariff and battery, and
executed open-loop against the actuals.  Regret is cost minus the
perfect-foresight cost; the attribution of a band is the share of that regret
which healing the band recovers.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BANDS, BAND_LIST, DEFAULT_C_DEM, DT, RESULTS_DIR, battery_specs, tou_price_vector
from .data import load_site
from .dispatch import execute_open_loop, solve_dispatch

__all__ = ["band_surgery", "estimate_utc_offset_hours", "shifted_price",
           "parse_sites", "select_issues", "run_site", "run_campaign",
           "SCENARIOS"]

SCENARIOS = (["perfect", "base"]
             + [f"healL_{b}" for b in BAND_LIST]
             + [f"healP_{b}" for b in BAND_LIST])


# --------------------------------------------------------------------------
# The operator
# --------------------------------------------------------------------------
def band_surgery(base_fc: np.ndarray, actual: np.ndarray, band: str,
                 k: float = 0.0) -> np.ndarray:
    """Counterfactual surgery on one horizon band.

    ``k=0`` heal (replace with actual), ``k=1`` identity (as issued),
    ``k>1`` degrade (amplify the issued error k-fold about the actual).
    """
    lo, hi = BANDS[band]
    out = base_fc.copy()
    out[lo:hi] = actual[lo:hi] + k * (base_fc[lo:hi] - actual[lo:hi])
    return out


# --------------------------------------------------------------------------
# Local-clock alignment
# --------------------------------------------------------------------------
def estimate_utc_offset_hours(load: np.ndarray, ts: pd.DatetimeIndex) -> int:
    """Estimate the site's local UTC offset from the shape of its load.

    EMSx stores no local clock, but industrial sites load up between roughly
    07:00 and 19:00 local time.  The offset that maximises the correlation
    between hourly load and a working-hours template is used as the estimate;
    the tariff is then aligned to that clock.  The estimate is a nuisance
    parameter, not a result: the band ordering is unchanged for any offset that
    keeps the tariff peaks inside the same bands.
    """
    hourly = pd.Series(load, index=ts).resample("1h").mean()
    hod = hourly.index.hour.to_numpy()
    x = hourly.to_numpy()
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    best, best_off = -np.inf, 0
    for off in range(0, 24, 1):  # offsets east positive
        local_h = (hod + off) % 24
        tmpl = ((local_h >= 7) & (local_h < 19)).astype(float)
        tmpl = (tmpl - tmpl.mean()) / (tmpl.std() + 1e-9)
        r = float(np.nansum(x * tmpl) / len(x))
        if r > best:
            best, best_off = r, off
    return best_off


def shifted_price(off_hours: int) -> np.ndarray:
    """TOU price vector aligned to the site's local clock.

    The row grid is UTC; the local period of row step ``k`` (0..95 within a UTC
    day) is ``(k + 4*off) mod 96``, so the base vector is rolled by ``-4*off``.
    """
    return np.roll(tou_price_vector(), -off_hours * 4)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------
def parse_sites(s: str) -> list[int]:
    """Parse ``"1-70"`` or ``"1,5,9-12"`` into a sorted list of site ids."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def select_issues(seg: np.ndarray, max_issues: int) -> np.ndarray:
    """Issue rows spread evenly over the usable rows of one site."""
    ok = seg == np.roll(seg, -95)
    ok[-95:] = False
    cand = np.where(ok)[0]
    if len(cand) > max_issues > 0:
        return cand[np.linspace(0, len(cand) - 1, max_issues).astype(int)]
    return cand


# --------------------------------------------------------------------------
# One site
# --------------------------------------------------------------------------
def run_site(sid: int, stride: int = 1, c_dem: float = DEFAULT_C_DEM,
             max_issues: int = 180, e_share: float | None = None,
             degrade_k: float | None = None,
             warehouse_dir: Path | None = None) -> pd.DataFrame:
    """Run the full scenario grid for one site.

    ``degrade_k=None`` heals the band (k=0); ``degrade_k=k>1`` amplifies the
    issued error k-fold.  Returns one row per (issue time, scenario).
    """
    df, F, G = load_site(sid, warehouse_dir=warehouse_dir)
    act_l = df["actual_load"].to_numpy(dtype=np.float64)
    act_p = df["actual_pv"].to_numpy(dtype=np.float64)
    ts = pd.DatetimeIndex(df["ts"])

    sel = select_issues(df["seg_id"].to_numpy(), max_issues)
    off = estimate_utc_offset_hours(act_l, ts)
    price = shifted_price(off)
    b = battery_specs(act_l, e_share=e_share)
    k_band = 0.0 if degrade_k is None else degrade_k
    sc_prefix = "heal" if degrade_k is None else f"deg{degrade_k:g}"

    rows = []
    for i0 in sel:
        aL = act_l[i0:i0 + 96]
        aP = act_p[i0:i0 + 96]
        if np.isnan(aL).any() or np.isnan(aP).any():
            continue
        nl_act = aL - aP
        fL = F[i0].astype(np.float64)
        fP = G[i0].astype(np.float64)
        if np.isnan(fL).any() or np.isnan(fP).any():
            continue
        nl_fc = fL - fP

        # perfect foresight: the denominator of every share
        res_p = solve_dispatch(nl_act, price, b["p_kw"], b["e_kwh"], c_dem=c_dem)
        if res_p is None:
            continue
        c_perf = execute_open_loop(res_p[1], res_p[2], nl_act, price, c_dem=c_dem)

        # the forecast as issued
        res_b = solve_dispatch(nl_fc, price, b["p_kw"], b["e_kwh"], c_dem=c_dem)
        if res_b is None:
            continue
        c_base = execute_open_loop(res_b[1], res_b[2], nl_act, price, c_dem=c_dem)
        denom = c_base - c_perf

        base_rec = {
            "site_id": sid, "i0": int(i0), "ts": str(ts[i0]),
            "scenario": "perfect", "cost": c_perf, "regret": 0.0,
            "regret_share": 0.0,
            "err_h0_2": float(np.abs(fL[:8] - aL[:8]).mean()),
            "err_h2_6": float(np.abs(fL[8:24] - aL[8:24]).mean()),
            "err_h6_12": float(np.abs(fL[24:48] - aL[24:48]).mean()),
            "err_h12_24": float(np.abs(fL[48:96] - aL[48:96]).mean()),
            "perr_h0_2": float(np.abs(fP[:8] - aP[:8]).mean()),
            "perr_h2_6": float(np.abs(fP[8:24] - aP[8:24]).mean()),
            "perr_h6_12": float(np.abs(fP[24:48] - aP[24:48]).mean()),
            "perr_h12_24": float(np.abs(fP[48:96] - aP[48:96]).mean()),
            "load_mean": float(aL.mean()), "pv_mean": float(aP.mean()),
            "tz_off": off,
        }
        rows.append(base_rec)
        rows.append({
            "site_id": sid, "i0": int(i0), "ts": str(ts[i0]),
            "scenario": "base", "cost": c_base,
            "regret": c_base - c_perf,
            "regret_share": 1.0,
            "tz_off": off,
        })

        for axis, (fc_arr, act_arr) in (("L", (fL, aL)), ("P", (fP, aP))):
            for band in BAND_LIST:
                if axis == "L":
                    f2 = band_surgery(fL, aL, band, k=k_band)
                    nl_fc2 = f2 - fP
                else:
                    g2 = band_surgery(fP, aP, band, k=k_band)
                    nl_fc2 = fL - g2
                r = solve_dispatch(nl_fc2, price, b["p_kw"], b["e_kwh"], c_dem=c_dem)
                if r is None:
                    continue
                c = execute_open_loop(r[1], r[2], nl_act, price, c_dem=c_dem)
                rows.append({
                    "site_id": sid, "i0": int(i0), "ts": str(ts[i0]),
                    "scenario": f"{sc_prefix}{axis}_{band}", "cost": c,
                    "regret": c - c_perf,
                    "regret_share": (c_base - c) / denom if denom > 1e-9 else np.nan,
                    "tz_off": off,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Campaign
# --------------------------------------------------------------------------
def run_campaign(sites, c_dem: float = DEFAULT_C_DEM, max_issues: int = 180,
                 e_share: float | None = None, degrade_k: float | None = None,
                 tag: str = "full", out_dir: Path | None = None,
                 warehouse_dir: Path | None = None) -> pd.DataFrame:
    """Run :func:`run_site` over *sites* and write ``surgery_<tag>.parquet``."""
    out = RESULTS_DIR if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = []
    for k, sid in enumerate(sites, 1):
        t1 = time.time()
        try:
            df = run_site(sid, c_dem=c_dem, max_issues=max_issues,
                          e_share=e_share, degrade_k=degrade_k,
                          warehouse_dir=warehouse_dir)
            frames.append(df)
            n = len(df) // len(SCENARIOS)
            print(f"[{k:2d}/{len(sites)}] site {sid:3d}: {n} issue times "
                  f"({time.time() - t1:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 - keep the campaign going
            print(f"[{k:2d}/{len(sites)}] site {sid:3d}: FAILED {e}", flush=True)
    res = pd.concat(frames, ignore_index=True)
    fp = out / f"surgery_{tag}.parquet"
    res.to_parquet(fp, index=False)
    print(f"done: {len(res)} rows in {time.time() - t0:.0f}s -> {fp}")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Counterfactual forecast surgery on the issued EMSx archive.")
    ap.add_argument("--sites", default="1-70", help="e.g. 1-70 or 1,5,9-12")
    ap.add_argument("--max-issues", type=int, default=180,
                    help="issue times per site (evenly spread)")
    ap.add_argument("--c-dem", type=float, default=DEFAULT_C_DEM,
                    help="demand charge, EUR/kW/day")
    ap.add_argument("--e-share", type=float, default=None,
                    help="battery energy / mean daily load (default 0.10)")
    ap.add_argument("--degrade-k", type=float, default=None,
                    help="amplify the issued error k-fold instead of healing")
    ap.add_argument("--tag", default="full", help="output suffix")
    ap.add_argument("--warehouse", default=None, help="warehouse directory")
    args = ap.parse_args(argv)

    run_campaign(parse_sites(args.sites), c_dem=args.c_dem,
                 max_issues=args.max_issues, e_share=args.e_share,
                 degrade_k=args.degrade_k, tag=args.tag,
                 warehouse_dir=args.warehouse)


if __name__ == "__main__":
    main()
