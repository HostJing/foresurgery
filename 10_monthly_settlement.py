"""
A5: monthly demand-charge settlement (subset experiment).

Daily settlement is a proxy for the monthly demand charges used in practice.
This experiment re-runs the heal attribution over contiguous 30-day windows
with the demand charge assessed ONCE per month on the monthly peak import:

  - window: 2,880 consecutive 15-min steps (30 days), no NaNs, one segment
  - LP over the full window (one schedule, open-loop execution against
    actuals), demand-charge epigraph on the monthly peak, c_dem_month =
    30 x c_dem_day = 15 EUR/kW, so identical peaks cost the same as the
    daily benchmark
  - scenarios: perfect | base | healL x4 | healP x4
  - daily benchmark attribution on the same windows is obtained by summing
    the cost components day by day (recomputed here with the same battery)

Subset: first 15 sites with a valid window; 2 windows per site (seeded).

Run:  python monthly_settlement.py
Out : results/monthly_settlement.parquet, results/monthly_report.md
"""

import time

import numpy as np
import pandas as pd

from foresurgery.config import BANDS, battery_specs
from foresurgery.config import RESULTS_DIR
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import estimate_utc_offset_hours, shifted_price

RES = RESULTS_DIR
WIN = 96 * 30          # 30 days
C_DEM_MONTH = 0.5 * 30  # keep totals comparable with daily settlement
N_SITES = 30
WINDOWS = 2


def find_windows(seg, n):
    """Start indices of 2,880-step windows inside one segment, no NaN check
    deferred to caller; returns up to WINDOWS evenly spaced candidates."""
    ok = seg == np.roll(seg, -(WIN - 1))
    ok[n - WIN:] = False
    cand = np.where(ok)[0]
    if len(cand) == 0:
        return []
    if len(cand) <= WINDOWS:
        return list(cand)
    return list(cand[np.linspace(0, len(cand) - 1, WINDOWS).astype(int)])


def run_site(sid: int, seed: int) -> pd.DataFrame | None:
    df, F, G = load_site(sid)
    act_l = df["actual_load"].to_numpy(dtype=np.float64)
    act_p = df["actual_pv"].to_numpy(dtype=np.float64)
    ts = pd.DatetimeIndex(df["ts"])
    seg = df["seg_id"].to_numpy()
    price = np.tile(shifted_price(estimate_utc_offset_hours(act_l, ts)),
                    WIN // 96 + 1)[:WIN]
    b = battery_specs(act_l)

    rows = []
    for i0 in find_windows(seg, len(df)):
        aL = act_l[i0:i0 + WIN]
        aP = act_p[i0:i0 + WIN]
        if np.isnan(aL).any() or np.isnan(aP).any():
            continue
        # day-ahead forecast series: each day contributes the forecast
        # issued at its first row (the day-ahead schedule the operator
        # would actually hold for that day); healed bands are applied to
        # every day's day-ahead forecast
        fL_series = np.empty(WIN)
        fP_series = np.empty(WIN)
        ok_days = True
        for d in range(WIN // 96):
            r0 = i0 + 96 * d
            fL_day = F[r0].astype(np.float64)
            fP_day = G[r0].astype(np.float64)
            if np.isnan(fL_day).any() or np.isnan(fP_day).any():
                ok_days = False
                break
            fL_series[96 * d:96 * (d + 1)] = fL_day
            fP_series[96 * d:96 * (d + 1)] = fP_day
        if not ok_days:
            continue

        def dispatch_and_cost(nl_fc):
            r = solve_dispatch(nl_fc, price, b["p_kw"], b["e_kwh"],
                               c_dem=C_DEM_MONTH)
            if r is None:
                return None
            return execute_open_loop(r[1], r[2], aL - aP, price,
                                     c_dem=C_DEM_MONTH)

        scenarios = {"perfect": aL - aP, "base": fL_series - fP_series}
        for band, (lo, hi) in BANDS.items():
            fL2 = fL_series.copy()
            fP2 = fP_series.copy()
            for d in range(WIN // 96):
                r0 = i0 + 96 * d
                fL2[96 * d + lo:96 * d + hi] = act_l[r0 + lo:r0 + hi]
                fP2[96 * d + lo:96 * d + hi] = act_p[r0 + lo:r0 + hi]
            scenarios[f"healL_{band}"] = fL2 - fP_series
            scenarios[f"healP_{band}"] = fL_series - fP2

        costs = {}
        for name, nl_fc in scenarios.items():
            c = dispatch_and_cost(nl_fc)
            if c is None:
                costs = None
                break
            costs[name] = c
        if costs is None:
            continue
        denom = costs["base"] - costs["perfect"]
        for name, c in costs.items():
            rows.append({"site_id": sid, "i0": int(i0), "scenario": name,
                         "cost": c, "denom": denom})
        print(f"  site {sid} win@{i0} done denom={denom:.1f}", flush=True)
    return pd.DataFrame(rows) if rows else None


def main():
    out_frames = []
    t0 = time.time()
    used = 0
    for sid in range(1, 71):
        if used >= N_SITES:
            break
        try:
            fr = run_site(sid, 0)
        except Exception as e:
            print(f"  site {sid} FAILED {e}", flush=True)
            continue
        if fr is not None and fr.scenario.nunique() == 10:
            out_frames.append(fr)
            used += 1
            print(f"site {sid} accepted ({used}/{N_SITES}, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    res = pd.concat(out_frames, ignore_index=True)
    out = RESULTS_DIR / "monthly_settlement30.parquet"
    res.to_parquet(out, index=False)

    # attribution
    perf = res[res.scenario == "perfect"].set_index(["site_id", "i0"]).cost
    res["regret"] = [c - perf[(s, i)]
                     for s, i, c in zip(res.site_id, res.i0, res.cost)]
    med = float(res[res.scenario == "base"].cost.median())
    kd = res[res.scenario == "base"]
    kd = kd[kd.denom > 0.05 * med]
    keys = set(zip(kd.site_id, kd.i0))
    heal = res[res.scenario.str.startswith("heal")].copy()
    heal = heal[[tuple(k) in keys for k in zip(heal.site_id, heal.i0)]]
    heal["recovered"] = heal.denom - heal.regret
    heal["axis"] = np.where(heal.scenario.str.startswith("healL"),
                            "load", "pv")
    heal["band"] = heal.scenario.str.split("_", n=1).str[1]
    lines = [f"# A5 monthly settlement (30-day windows, c_dem_month=15 "
             f"EUR/kW; {len(kd)} valid windows, {res.site_id.nunique()} sites)\n",
             "| axis | 0-2h | 2-6h | 6-12h | 12-24h |", "|---|" + "---|" * 4]
    for axis in ["load", "pv"]:
        cells = []
        for band in BANDS:
            sub = heal[(heal.axis == axis) & (heal.band == band)]
            cells.append(f"{100 * sub.recovered.sum() / sub.denom.sum():.1f}"
                         if len(sub) else "--")
        lines.append(f"| {axis} | " + " | ".join(cells) + " |")
    lines.append("\n(daily-settlement reference on the full fleet: "
                 "load 1.3/2.8/10.5/19.4, pv 1.1/3.9/13.5/23.3)")
    txt = "\n".join(lines)
    (RES / "monthly_report.md").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
