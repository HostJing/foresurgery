#!/usr/bin/env python
"""Quick check that the install, the warehouse and the LP all work.

Runs one issue time of one site through the full scenario grid and prints the
cost of each scenario. Nothing is written to disk. If this script produces
sensible numbers, every other script will run.

Usage
-----
    python scripts/00_smoke_test.py              # site 1
    python scripts/00_smoke_test.py --site 12
"""
import argparse
import time

import numpy as np

from foresurgery.config import BAND_LIST, DEFAULT_C_DEM, battery_specs, tou_price_vector
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import band_surgery, select_issues


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", type=int, default=1)
    ap.add_argument("--c-dem", type=float, default=DEFAULT_C_DEM)
    args = ap.parse_args(argv)

    df, F, G = load_site(args.site)
    act_l = df["actual_load"].to_numpy(dtype=np.float64)
    act_p = df["actual_pv"].to_numpy(dtype=np.float64)
    usable = select_issues(df["seg_id"].to_numpy(), 0)
    print(f"site {args.site}: {len(df)} rows, F{F.shape}, "
          f"{len(usable)} executable issue times")

    price = tou_price_vector()
    i0 = int(usable[len(usable) // 4])
    print(f"issue row {i0}, ts={df['ts'].iloc[i0]}")

    fc = F[i0].astype(np.float64)
    aL, aP = act_l[i0:i0 + 96], act_p[i0:i0 + 96]
    assert len(aL) == 96 and not np.isnan(aL).any(), "issue row is not usable"
    nl_act = aL - aP
    nl_fc = fc - G[i0].astype(np.float64)

    b = battery_specs(act_l)
    print(f"battery: P={b['p_kw']:.1f} kW, E={b['e_kwh']:.1f} kWh")

    t0 = time.time()
    plan = solve_dispatch(nl_act, price, b["p_kw"], b["e_kwh"], c_dem=args.c_dem)
    dt = time.time() - t0
    c_perf = execute_open_loop(plan[1], plan[2], nl_act, price, c_dem=args.c_dem)
    print(f"perfect foresight: LP {dt:.2f}s, planned={plan[0]:.2f}, "
          f"realised={c_perf:.2f}")
    if abs(plan[0] - c_perf) > 1e-6:
        print("  note: planned and realised differ because, in the default "
              "'legacy' formulation,\n        the peak variable shares the "
              "terminal-SOC slot (docs/LIMITATIONS.md).\n        "
              "FORESURGERY_PEAK_SLOT=own makes the two agree.")

    plan_b = solve_dispatch(nl_fc, price, b["p_kw"], b["e_kwh"], c_dem=args.c_dem)
    c_base = execute_open_loop(plan_b[1], plan_b[2], nl_act, price, c_dem=args.c_dem)
    print(f"as issued:         planned={plan_b[0]:.2f}, realised={c_base:.2f}, "
          f"regret={c_base - c_perf:.2f}")

    denom = c_base - c_perf
    print("\nband surgery (heal one load band, then execute against the actuals):")
    for band in BAND_LIST:
        f2 = band_surgery(fc, aL, band)
        r = solve_dispatch(f2 - G[i0].astype(np.float64), price,
                           b["p_kw"], b["e_kwh"], c_dem=args.c_dem)
        c = execute_open_loop(r[1], r[2], nl_act, price, c_dem=args.c_dem)
        share = (c_base - c) / denom if denom > 1e-9 else float("nan")
        print(f"  {band:8s} realised={c:9.2f}  regret recovered={share:7.1%}")


if __name__ == "__main__":
    main()
