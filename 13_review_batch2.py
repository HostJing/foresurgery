"""
Batch 2 review experiments: B1 cycle counts, B5 resample stability,
C1 reserve-constrained scheduler, C3 real (Italian F1/F2/F3) tariff.

All variants run the same heal campaign at the heal60 scale (60 issues/site,
70 sites) and report the cost-weighted band attribution plus, for B1, the
battery cycle equivalent (sum(ch)/E_cap per day).

Run:  python review2_experiments.py
Out : results/batch2_{resampleS,reserve,realtariff}.parquet
      results/batch2_report.md
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
BAND_LIST = list(BANDS)

# Italian industrial three-period TOU (F1/F2/F3; ARERA structure), EUR/kWh.
# Day-of-week distinction is dropped (the LP has no dow dimension); the
# Mon-Fri F1 pattern is applied every day as a documented approximation.
_REAL_SEGMENTS = [
    ((7, 8), 0.23),    # F2 morning shoulder
    ((8, 20), 0.28),   # F1 peak  (08-19 inclusive boundary approximated)
    ((20, 23), 0.23),  # F2 evening shoulder
    ((23, 24), 0.18),  # F3 off-peak
    ((0, 7), 0.18),    # F3 off-peak
]


def real_price_vector():
    v = np.zeros(96)
    for (h0, h1), p in _REAL_SEGMENTS:
        v[h0 * 4:h1 * 4] = p
    return v


def shifted(vec, off_hours):
    return np.roll(vec, -off_hours * 4)


def campaign(sid_list, n_issues, mode, seed=None, tag=""):
    """mode: 'base' (standard tariff/scheduler, linspace sampling),
    'resample' (random sampling), 'reserve' (soc_floor=0.5E),
    'tariff' (real F1/F2/F3 prices)."""
    rows = []
    t0 = time.time()
    for sid in sid_list:
        df, F, G = load_site(sid)
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        seg = df["seg_id"].to_numpy()
        ok = seg == np.roll(seg, -95)
        ok[-95:] = False
        cand = np.where(ok)[0]
        if mode == "resample":
            rng = np.random.default_rng(seed * 100000 + sid)
            sel = (rng.choice(cand, min(n_issues, len(cand)), replace=False)
                   if len(cand) > n_issues else cand)
            sel = np.sort(sel)
        else:
            sel = (cand[np.linspace(0, len(cand) - 1, n_issues).astype(int)]
                   if len(cand) > n_issues else cand)

        off = estimate_utc_offset_hours(act_l, ts)
        price = (shifted(real_price_vector(), off) if mode == "tariff"
                 else shifted_price(off))
        b = battery_specs(act_l)
        sched_kw = {"soc_floor": 0.5 * b["e_kwh"]} if mode == "reserve" else {}

        for i0 in sel:
            aL = act_l[i0:i0 + 96]
            aP = act_p[i0:i0 + 96]
            if np.isnan(aL).any() or np.isnan(aP).any():
                continue
            fL = F[i0].astype(np.float64)
            fP = G[i0].astype(np.float64)
            if np.isnan(fL).any() or np.isnan(fP).any():
                continue

            def cost(nl_fc):
                r = solve_dispatch(nl_fc, price, b["p_kw"], b["e_kwh"],
                                   c_dem=0.5, **sched_kw)
                if r is None:
                    return None, None
                return (execute_open_loop(r[1], r[2], aL - aP, price,
                                          c_dem=0.5),
                        float(r[1].sum()) / b["e_kwh"])

            c_perf, cyc_p = cost(aL - aP)
            if c_perf is None:
                continue
            c_base, cyc_b = cost(fL - fP)
            if c_base is None:
                continue
            denom = c_base - c_perf
            rows.append({"site_id": sid, "i0": int(i0), "scenario": "perfect",
                         "cost": c_perf, "regret": 0.0, "denom": denom,
                         "cycles": cyc_p})
            rows.append({"site_id": sid, "i0": int(i0), "scenario": "base",
                         "cost": c_base, "regret": c_base - c_perf,
                         "denom": denom, "cycles": cyc_b})
            for axis, fc, ac, other in (("L", fL, aL, fP), ("P", fP, aP, fL)):
                for band, (lo, hi) in BANDS.items():
                    f2 = fc.copy()
                    f2[lo:hi] = ac[lo:hi]
                    # net load = load forecast - PV forecast, regardless of axis
                    nl_fc = (f2 - other) if axis == "L" else (other - f2)
                    c, cyc = cost(nl_fc)
                    if c is None:
                        continue
                    rows.append({"site_id": sid, "i0": int(i0),
                                 "scenario": f"heal{axis}_{band}", "cost": c,
                                 "regret": c - c_perf, "denom": denom,
                                 "cycles": cyc})
        print(f"  [{tag}] site {sid} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def attribution(df):
    med = float(df[df.scenario == "base"].cost.median())
    kd = df[df.scenario == "base"]
    kd = kd[kd.denom > 0.05 * med]
    keys = set(zip(kd.site_id, kd.i0))
    heal = df[df.scenario.str.startswith("heal")].copy()
    heal = heal[[tuple(k) in keys for k in zip(heal.site_id, heal.i0)]]
    heal["recovered"] = heal.denom - heal.regret
    heal["axis"] = np.where(heal.scenario.str.startswith("healL"), "load", "pv")
    heal["band"] = heal.scenario.str.split("_", n=1).str[1]
    out = {}
    for axis in ["load", "pv"]:
        for band in BAND_LIST:
            sub = heal[(heal.axis == axis) & (heal.band == band)]
            out[(axis, band)] = (100 * sub.recovered.sum() / sub.denom.sum()
                                 if len(sub) else np.nan)
    return out, len(kd)


def main():
    t0 = time.time()
    sites = list(range(1, 71))
    lines = ["# Batch-2 review experiments\n"]

    # ---------- B1: cycle counts (on the reference heal60 sampling) ----------
    print("B1 cycle counts...", flush=True)
    df = campaign(sites, 60, "base", tag="b1")
    med = float(df[df.scenario == "base"].cost.median())
    kd = df[df.scenario == "base"]
    kd = kd[kd.denom > 0.05 * med]
    keys = set(zip(kd.site_id, kd.i0))
    cyc = df[[tuple(k) in keys for k in zip(df.site_id, df.i0)]]
    cyc_tab = cyc.groupby("scenario").cycles.agg(["mean", "median"])
    lines.append("## B1 battery cycle equivalent [cycles/day]\n")
    lines.append(cyc_tab.round(3).to_string())
    lines.append("")
    spread = (cyc_tab["mean"].max() - cyc_tab["mean"].min()) / cyc_tab["mean"].mean()
    lines.append(f"\nmean-cycle spread across scenarios: {100 * spread:.1f}% "
                 "of the mean (ageing intensity comparable across compared "
                 "scenarios).\n")

    # ---------- B5: resample stability (2 seeds) ----------
    res_lines = []
    real = pd.read_parquet(RES / "surgery_heal60.parquet")
    b = real[real.scenario == "base"].set_index(["site_id", "i0"]).cost
    p = real[real.scenario == "perfect"].set_index(["site_id", "i0"]).cost
    dm = b - p
    real["denom"] = [dm.get((s, i), np.nan)
                     for s, i in zip(real.site_id, real.i0)]
    attr0, n0 = attribution(real)
    res_lines.append("| run | load far | pv far | valid |")
    res_lines.append("|---|---|---|---|")
    res_lines.append(f"| heal60 (linspace) | {attr0[('load','h12_24')]:.1f} | "
                     f"{attr0[('pv','h12_24')]:.1f} | {n0} |")
    for seed in [7, 42]:
        print(f"B5 resample seed {seed}...", flush=True)
        d = campaign(sites, 60, "resample", seed=seed, tag=f"b5s{seed}")
        d.to_parquet(RES / f"batch2_resample{seed}.parquet", index=False)
        a, nv = attribution(d)
        res_lines.append(f"| resample seed {seed} | {a[('load','h12_24')]:.1f} "
                         f"| {a[('pv','h12_24')]:.1f} | {nv} |")
    lines.append("\n## B5 sampling stability (far-band share, cost-weighted %)\n")
    lines += res_lines
    lines.append("")

    # ---------- C1: reserve-constrained scheduler ----------
    print("C1 reserve-constrained...", flush=True)
    d = campaign(sites, 60, "reserve", tag="c1")
    d.to_parquet(RES / "batch2_reserve.parquet", index=False)
    a, nv = attribution(d)
    lines.append("\n## C1 reserve-constrained scheduler (SOC floor 50%, "
                 "cost-weighted heal shares %)\n")
    lines.append("| axis | 0-2h | 2-6h | 6-12h | 12-24h | valid |")
    lines.append("|---|" + "---|" * 5)
    for axis in ["load", "pv"]:
        cells = " | ".join(f"{a[(axis, bn)]:.1f}" for bn in BAND_LIST)
        lines.append(f"| {axis} | {cells} | {nv} |")
    lines.append("")

    # ---------- C3: real Italian F1/F2/F3 tariff ----------
    print("C3 real tariff...", flush=True)
    d = campaign(sites, 60, "tariff", tag="c3")
    d.to_parquet(RES / "batch2_realtariff.parquet", index=False)
    a, nv = attribution(d)
    lines.append("\n## C3 real tariff (Italian F1/F2/F3 structure, "
                 "cost-weighted heal shares %)\n")
    lines.append("| axis | 0-2h | 2-6h | 6-12h | 12-24h | valid |")
    lines.append("|---|" + "---|" * 5)
    for axis in ["load", "pv"]:
        cells = " | ".join(f"{a[(axis, bn)]:.1f}" for bn in BAND_LIST)
        lines.append(f"| {axis} | {cells} | {nv} |")

    txt = "\n".join(lines)
    (RES / "batch2_report.md").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
