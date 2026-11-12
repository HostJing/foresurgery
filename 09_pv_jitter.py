"""
C1: PV scale-jitter sensitivity.

EMSx carries a single PV production profile rescaled across the fleet.
The PV-side attribution (23.3% far band) could in principle depend on the
specific rescaling. Test: draw a per-site PV capacity factor
f_s ~ U(0.7, 1.3) (seeded), scale BOTH actual PV and issued PV forecast by
f_s, rerun the heal-PV surgery at the heal60 issue times, compare the
attribution profile with the unscaled run and across jitter draws.

Run:  python pv_jitter.py            # 3 jitter draws
Out : results/pv_jitter_draw{1,2,3}.parquet, results/pv_jitter_report.md
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
DRAWS = [11, 22, 33]  # seeds; factor = rng.uniform(0.7, 1.3)


def run_draw(seed: int) -> pd.DataFrame:
    ref = pd.read_parquet(RES / "surgery_heal60.parquet")
    pairs = ref[ref.scenario == "base"][["site_id", "i0"]].drop_duplicates()
    rows = []
    t0 = time.time()
    for sid in sorted(pairs.site_id.unique()):
        i0s = pairs.loc[pairs.site_id == sid, "i0"].to_numpy()
        df, F, G = load_site(sid)
        rng = np.random.default_rng(seed * 1000 + sid)
        f_s = rng.uniform(0.7, 1.3)
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64) * f_s
        ts = pd.DatetimeIndex(df["ts"])
        price = shifted_price(estimate_utc_offset_hours(act_l, ts))
        b = battery_specs(act_l)
        for i0 in i0s:
            aL = act_l[i0:i0 + 96]
            aP = act_p[i0:i0 + 96]
            if np.isnan(aL).any() or np.isnan(aP).any():
                continue
            fL = F[i0].astype(np.float64)
            fP = G[i0].astype(np.float64) * f_s
            if np.isnan(fL).any() or np.isnan(fP).any():
                continue
            nl_act = aL - aP
            r = solve_dispatch(nl_act, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
            if r is None:
                continue
            c_perf = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
            r = solve_dispatch(fL - fP, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
            if r is None:
                continue
            c_base = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
            denom = c_base - c_perf
            rows.append({"site_id": sid, "i0": int(i0), "scenario": "perfect",
                         "cost": c_perf, "regret": 0.0, "denom": denom})
            rows.append({"site_id": sid, "i0": int(i0), "scenario": "base",
                         "cost": c_base, "regret": c_base - c_perf,
                         "denom": denom})
            for band, (lo, hi) in BANDS.items():
                f2 = fP.copy()
                f2[lo:hi] = aP[lo:hi]
                r = solve_dispatch(fL - f2, price, b["p_kw"], b["e_kwh"],
                                   c_dem=0.5)
                if r is None:
                    continue
                c = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
                rows.append({"site_id": sid, "i0": int(i0),
                             "scenario": f"healP_{band}", "cost": c,
                             "regret": c - c_perf, "denom": denom})
        print(f"  [draw {seed}] site {sid} f_s={f_s:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def attribution(df):
    med = float(df[df.scenario == "base"].cost.median())
    kd = df[df.scenario == "base"]
    kd = kd[kd.denom > 0.05 * med]
    keys = set(zip(kd.site_id, kd.i0))
    heal = df[df.scenario.str.startswith("heal")].copy()
    heal = heal[[tuple(k) in keys for k in zip(heal.site_id, heal.i0)]]
    heal["recovered"] = heal.denom - heal.regret
    heal["band"] = heal.scenario.str.split("_", n=1).str[1]
    return {b: (100 * heal[heal.band == b].recovered.sum()
                / heal[heal.band == b].denom.sum())
            for b in BANDS}, len(kd)


def main():
    real_df = pd.read_parquet(RES / "surgery_heal60.parquet")
    b = real_df[real_df.scenario == "base"].set_index(["site_id", "i0"]).cost
    p = real_df[real_df.scenario == "perfect"].set_index(["site_id", "i0"]).cost
    dm = b - p
    real_df["denom"] = [dm.get((s, i), np.nan)
                        for s, i in zip(real_df.site_id, real_df.i0)]
    real, n_real = attribution(real_df)
    lines = ["# C1 PV scale-jitter sensitivity (heal PV, cost-weighted %)\n",
             "| draw | 0-2h | 2-6h | 6-12h | 12-24h | valid |",
             "|---|---|---|---|---|---|"]
    cells = " | ".join(f"{real[b]:.1f}" for b in BANDS)
    lines.append(f"| no jitter (real) | {cells} | {n_real} |")
    for seed in DRAWS:
        df = run_draw(seed)
        df.to_parquet(RES / f"pv_jitter_draw{seed}.parquet", index=False)
        attr, n = attribution(df)
        cells = " | ".join(f"{attr[b]:.1f}" for b in BANDS)
        lines.append(f"| draw {seed} (U[0.7,1.3]) | {cells} | {n} |")
        print(f"draw {seed}: " + " ".join(f"{attr[b]:.1f}" for b in BANDS),
              flush=True)
    txt = "\n".join(lines)
    (RES / "pv_jitter_report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
