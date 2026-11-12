"""
Diagnose the PV far-band heal anomaly (heal cost >> base cost).

For the worst samples, reconstruct base / healP_far plans and compare
per-band: planned vs realized imports, charge, peak, and cost split.
Hypotheses:
  H1 (splicing artifact): healed PV profile is discontinuous at band
      edges, misleading the LP near the boundary.
  H2 (residual load error): after healing PV, far-band net-load error
      equals the pure load error, which dominates and flips the sign.
  H3 (open-loop fragility): with more expected far-band PV the LP picks
      a schedule (charge shift) that is fragile to near-band errors.
"""


import numpy as np
import pandas as pd

from foresurgery.config import BANDS, battery_specs
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import (band_surgery, estimate_utc_offset_hours,
                                 shifted_price)

TARGETS = [(26, 57113)]


def band_costs(nl_act, ch, dis, price, c_dem, off4):
    """Cost split per 6-hour local-time block + global peak."""
    net = nl_act + ch - dis
    imp = np.maximum(net, 0.0)
    cost = price * imp
    # local hour of step k in UTC day grid: (k + off*4) % 96 / 4
    loc = (np.arange(96) + off4) % 96
    blocks = {"00-06": (0, 24), "06-12": (24, 48),
              "12-18": (48, 72), "18-24": (72, 96)}
    out = {}
    for name, (a, b) in blocks.items():
        m = (loc >= a) & (loc < b)
        out[name] = float(cost[m].sum())
    out["peak"] = float(imp.max())
    out["charge"] = float(ch.sum())
    out["discharge"] = float(dis.sum())
    out["curtail"] = float((nl_act - imp + ch - dis).__abs__().sum() * 0)  # unused
    return out


def main():
    b = None
    for sid, i0 in TARGETS:
        df, F, G = load_site(sid)
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        off = estimate_utc_offset_hours(act_l, ts)
        price = shifted_price(off)
        b = battery_specs(act_l)
        aL = act_l[i0:i0 + 96]
        aP = act_p[i0:i0 + 96]
        fL = F[i0].astype(np.float64)
        fP = G[i0].astype(np.float64)
        nl_act = aL - aP
        nl_base = fL - fP
        g2 = band_surgery(fP, aP, "h12_24", k=0.0)
        nl_heal = fL - g2

        print(f"=== site {sid} i0 {i0}  ts={ts[i0]}  tz_off={off}h ===")
        # band-wise forecast error of PV and LOAD (horizon bands)
        for band, (lo, hi) in BANDS.items():
            print(f"  band {band:6s}: PV fc {fP[lo:hi].mean():7.1f} act {aP[lo:hi].mean():7.1f} "
                  f"| LOAD fc {fL[lo:hi].mean():7.1f} act {aL[lo:hi].mean():7.1f} "
                  f"| net fc {nl_base[lo:hi].mean():7.1f} act {nl_act[lo:hi].mean():7.1f}")

        plans = {}
        for name, nl in (("perfect", nl_act), ("base", nl_base), ("healP_far", nl_heal)):
            r = solve_dispatch(nl, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
            ch, dis = r[1], r[2]
            c = execute_open_loop(ch, dis, nl_act, price, c_dem=0.5)
            split = band_costs(nl_act, ch, dis, price, 0.5, off * 4)
            plans[name] = (ch, dis)
            print(f"\n  [{name}] planned_cost={r[0]:.1f} realized={c:.1f}")
            print("    " + "  ".join(f"{k}={v:8.1f}" for k, v in split.items()))

        # realized import profile around the 12-24h band, base vs heal
        ch_b, dis_b = plans["base"]
        ch_h, dis_h = plans["healP_far"]
        imp_b = np.maximum(nl_act + ch_b - dis_b, 0.0)
        imp_h = np.maximum(nl_act + ch_h - dis_h, 0.0)
        print("\n  step  nl_act   imp_base  imp_heal   ch_base  ch_heal   (local h)")
        loc = (np.arange(96) + off * 4) % 96
        for k in range(0, 96, 2):
            d = imp_h[k] - imp_b[k]
            if abs(d) > 15 or abs(ch_h[k] - ch_b[k]) > 15:
                print(f"  {k:3d}  {nl_act[k]:8.1f} {imp_b[k]:9.1f} {imp_h[k]:9.1f} "
                      f"{ch_b[k]:9.1f} {ch_h[k]:9.1f}   h={loc[k]/4:.1f}")


if __name__ == "__main__":
    main()
