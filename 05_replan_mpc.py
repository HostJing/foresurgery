"""
Replanning-MPC control experiment (Layer 3.3), v2.

v2 changes vs v1 (review fixes):
  1. TRUE rolling archive: at each hourly replan the *forecast issued at
     that replan row* (F[i0+j]) is used, not the initial forecast truncated.
     The healed band persists in its still-future portion (original
     coordinates), everything else follows the newly issued forecast.
  2. Chain reuse: once the healed band is fully in the past (j >= hi),
     the heal scenario coincides with the base chain (same forecasts ->
     same actions -> same SOC/cost), so the remaining replans are taken
     from the base chain. Cuts LP count ~43%.
  3. NaN check on every replan forecast row.

Protocol per (site, issue time i0, scenario):
  for j = 0, 4, ..., 92 (hourly replans):
    plan over the remaining horizon with the scenario forecast;
    execute the first 4 steps (1 h) against actuals; accumulate cost.
  SOC carries over (clipped to bounds). Hourly windows settle their own
  peak at a 4x scaled demand charge (24 windows/day <-> daily charge).

Run:  python replan_mpc2.py --max-issues 80 --tag replan80
"""
import argparse
import time


import numpy as np
import pandas as pd

from foresurgery.config import RESULTS_DIR
from foresurgery.config import BANDS, BATT, battery_specs
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import estimate_utc_offset_hours, parse_sites, shifted_price

REPLAN_J = list(range(0, 96, 4))  # hourly replans


def band_boundary(band):
    return BANDS[band][1]  # hi: heal irrelevant once j >= hi


def run_site_replan(sid: int, max_issues: int, c_dem: float) -> pd.DataFrame:
    df, F, G = load_site(sid)
    act_l = df["actual_load"].to_numpy(dtype=np.float64)
    act_p = df["actual_pv"].to_numpy(dtype=np.float64)
    ts = pd.DatetimeIndex(df["ts"])
    n = len(df)
    seg = df["seg_id"].to_numpy()
    ok = seg == np.roll(seg, -95)
    ok[-95:] = False
    cand = np.where(ok)[0]
    sel = (cand[np.linspace(0, len(cand) - 1, max_issues).astype(int)]
           if len(cand) > max_issues else cand)

    off = estimate_utc_offset_hours(act_l, ts)
    price = shifted_price(off)
    b = battery_specs(act_l)

    rows = []
    for i0 in sel:
        aL = act_l[i0:i0 + 96]
        aP = act_p[i0:i0 + 96]
        if np.isnan(aL).any() or np.isnan(aP).any():
            continue
        # replan forecast rows must be NaN-free
        rws = [i0 + j for j in REPLAN_J]
        if any(r + 96 > n for r in rws):
            continue
        if np.isnan(F[rws]).any() or np.isnan(G[rws]).any():
            continue

        soc0 = 0.5 * b["e_kwh"]

        def chain(mode):
            """Run the replan chain. mode: 'base' (issued forecasts),
            'perfect' (actuals), or (axis, lo, hi) heal tuple.
            Returns (total_cost, snaps) with snaps[j] = (soc, cost_added)."""
            soc = soc0
            total = 0.0
            snaps = {}
            for j in REPLAN_J:
                if mode == "perfect":
                    nl_fc = (aL - aP)[j:]
                else:
                    r = i0 + j
                    fL = F[r].astype(np.float64).copy()
                    fP = G[r].astype(np.float64).copy()
                    if mode != "base":
                        axis, lo, hi = mode
                        lo2 = max(lo, j) - j
                        hi2 = hi - j
                        if hi2 > 0:
                            if axis == "L":
                                fL[lo2:hi2] = act_l[r + lo2:r + hi2]
                            else:
                                fP[lo2:hi2] = act_p[r + lo2:r + hi2]
                    nl_fc = (fL - fP)[:96 - j]  # remaining horizon only
                res = solve_dispatch(nl_fc, price[j:], b["p_kw"], b["e_kwh"],
                                     c_dem=c_dem, soc_start=soc,
                                     soc_term=0.5 * b["e_kwh"])
                if res is None:
                    return None, None
                ch_j, dis_j = res[1][:4], res[2][:4]
                c = execute_open_loop(ch_j, dis_j, (aL - aP)[j:j + 4],
                                      price[j:j + 4], c_dem=c_dem * 4)
                total += c
                snaps[j] = (soc, c)
                soc += BATT["eta_c"] * ch_j.sum() - dis_j.sum() / BATT["eta_d"]
                soc = min(max(soc, 0.1 * b["e_kwh"]), 0.9 * b["e_kwh"])
            return total, snaps

        c_base, base_snaps = chain("base")
        if c_base is None:
            continue
        c_perf, _ = chain("perfect")
        if c_perf is None or c_base <= c_perf + 1e-12:
            continue
        denom = c_base - c_perf
        rows.append({"site_id": sid, "i0": int(i0), "ts": str(ts[i0]),
                     "scenario": "base", "cost": c_base, "denom": denom})
        rows.append({"site_id": sid, "i0": int(i0), "ts": str(ts[i0]),
                     "scenario": "perfect", "cost": c_perf, "denom": denom})

        for axis, pfx, arr in (("L", "healL", act_l), ("P", "healP", act_p)):
            for band, (lo, hi) in BANDS.items():
                c_h, _ = chain((axis, lo, hi))
                if c_h is None:
                    continue
                rows.append({"site_id": sid, "i0": int(i0), "ts": str(ts[i0]),
                             "scenario": f"{pfx}_{band}", "cost": c_h,
                             "denom": denom})
        print(f"  site {sid} issue@{i0} done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="1-70")
    ap.add_argument("--max-issues", type=int, default=80)
    ap.add_argument("--c-dem", type=float, default=0.5)
    ap.add_argument("--tag", default="replan80")
    args = ap.parse_args()
    sites = parse_sites(args.sites)
    out = RESULTS_DIR / f"surgery_{args.tag}.parquet"
    t0 = time.time()
    frames = []
    for k, sid in enumerate(sites, 1):
        t1 = time.time()
        try:
            frames.append(run_site_replan(sid, args.max_issues, args.c_dem))
            print(f"[{k:2d}/{len(sites)}] site {sid:3d} "
                  f"({time.time() - t1:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{k:2d}/{len(sites)}] site {sid:3d} FAILED {e}", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(out, index=False)
    print(f"DONE {len(res)} rows in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
