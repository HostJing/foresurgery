"""
A1: Synthetic-error control experiment.

Question: is the far-band value concentration a property of the REAL issued
error structure, or would ANY error of the same marginal magnitude produce it?

Design (paired with the real-archive heal60 run):
  - reuse the EXACT issue times of results/surgery_heal60.parquet (60/site)
  - build a synthetic forecast S = clip(actual + eps, 0, None) where
    eps ~ N(0, sigma[site, band, axis]) with sigma matched to the site's own
    per-band MAE (sigma = MAE * sqrt(pi/2), the Gaussian MAE-equivalent),
    estimated from the site's sampled issue times (deterministic seed).
  - scenarios: perfect | base(syn) | healL_b x4 | healP_b x4  (same operator)
  - same LP, tariff, battery, validity filter, cost-weighted attribution.
Comparison: synthetic attribution profile vs real heal60 profile.

Run:  python surgery_synthetic.py
Out : results/surgery_synth.parquet, results/synth_report.md
"""

import argparse
import time

import numpy as np
import pandas as pd

from foresurgery.config import BANDS, RESULTS_DIR, battery_specs
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import estimate_utc_offset_hours, shifted_price

RES = RESULTS_DIR
GAUSS_MAE = np.sqrt(np.pi / 2)  # sigma = MAE * this for N(0, sigma)
# The real issued errors are strongly autocorrelated within a band (level-type
# errors), so "block" noise is the closer control and "white" noise the
# adversarial extreme. Running both brackets the space of possible structures.
MODE = "block"  # set by --mode


def synth_noise(rng, sigma, band_steps, daylight, mode):
    """Return noise array for one band. daylight: boolean mask of steps."""
    if mode == "white":
        return rng.normal(0, sigma, band_steps)
    # block: single standard-normal draw scaled over the band
    return rng.normal(0, sigma) * np.ones(band_steps)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["block", "white"], default="block",
                    help="error structure of the synthetic archive")
    args = ap.parse_args(argv)
    global MODE
    MODE = args.mode
    ref = pd.read_parquet(RES / "surgery_heal60.parquet")
    pairs = ref[ref.scenario == "base"][["site_id", "i0"]].drop_duplicates()
    sites = sorted(pairs.site_id.unique())
    print(f"{len(pairs)} issue times across {len(sites)} sites", flush=True)

    rows = []
    t0 = time.time()
    for sid in sites:
        i0s = pairs.loc[pairs.site_id == sid, "i0"].to_numpy()
        df, F, G = load_site(sid)
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        price = shifted_price(estimate_utc_offset_hours(act_l, ts))
        b = battery_specs(act_l)

        # per-site, per-band error scale from the site's real issued errors:
        # load: additive MAE; PV: multiplicative relative RMS (real PV errors
        # are proportional level/cloud errors, not additive offsets)
        sig = {}
        sigP_rel = {}
        for band, (lo, hi) in BANDS.items():
            maes, rels = [], []
            for i0 in i0s:
                a = act_l[i0 + lo:i0 + hi]
                e = np.abs(F[i0, lo:hi].astype(np.float64) - a)
                if len(e):
                    maes.append(np.nanmean(e))
                ap = act_p[i0 + lo:i0 + hi]
                day = ap > 0
                if day.sum() >= 4:
                    m = ap[day].mean()
                    rels.append(np.sqrt(np.nanmean(
                        ((G[i0, lo:hi].astype(np.float64) - ap)[day] / m) ** 2)))
            sig[("L", band)] = float(np.nanmedian(maes)) * GAUSS_MAE
            sigP_rel[band] = float(np.nanmedian(rels)) if rels else 0.0

        for i0 in i0s:
            aL = act_l[i0:i0 + 96]
            aP = act_p[i0:i0 + 96]
            if np.isnan(aL).any() or np.isnan(aP).any():
                continue
            rng = np.random.default_rng(sid * 10_000_000 + i0)
            # load: additive noise (block = one draw per band, level-type)
            epsL = {bd: synth_noise(rng, sig[("L", bd)], hi - lo, None, MODE)
                    for bd, (lo, hi) in BANDS.items()}
            # PV: multiplicative level noise matched in relative RMS; zero
            # at night by construction (factor applies to actual profile)
            zP = {bd: rng.normal(0, 1) * (1 if MODE == "block" else
                  np.ones(hi - lo))
                  for bd, (lo, hi) in BANDS.items()}
            sL = aL.copy()
            sP = aP.copy()
            for bd, (lo, hi) in BANDS.items():
                sL[lo:hi] = np.maximum(aL[lo:hi] + epsL[bd], 0.0)
                z = zP[bd] * sigP_rel[bd]
                sP[lo:hi] = np.maximum(aP[lo:hi] * (1.0 + z), 0.0)

            nl_act = aL - aP
            nl_syn = sL - sP

            r = solve_dispatch(nl_act, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
            if r is None:
                continue
            c_perf = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
            r = solve_dispatch(nl_syn, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
            if r is None:
                continue
            c_base = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
            denom = c_base - c_perf

            rows.append({"site_id": sid, "i0": int(i0), "scenario": "perfect",
                         "cost": c_perf, "regret": 0.0, "denom": denom})
            rows.append({"site_id": sid, "i0": int(i0), "scenario": "base",
                         "cost": c_base, "regret": c_base - c_perf, "denom": denom})
            for axis, (s_arr, a_arr, other) in (("L", (sL, aL, sP)),
                                                ("P", (sP, aP, sL))):
                for band, (lo, hi) in BANDS.items():
                    f2 = s_arr.copy()
                    f2[lo:hi] = a_arr[lo:hi]
                    # net load = load forecast - PV forecast, regardless of axis
                    nl_fc = (f2 - other) if axis == "L" else (other - f2)
                    r = solve_dispatch(nl_fc, price, b["p_kw"], b["e_kwh"],
                                       c_dem=0.5)
                    if r is None:
                        continue
                    c = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
                    rows.append({"site_id": sid, "i0": int(i0),
                                 "scenario": f"heal{axis}_{band}", "cost": c,
                                 "regret": c - c_perf, "denom": denom})
        print(f"  site {sid} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    out_name = "surgery_synth.parquet" if MODE == "block" else "surgery_synth_white.parquet"
    res.to_parquet(RES / out_name, index=False)

    # ---- attribution comparison: synthetic vs real ----
    def attribution(df):
        med = float(df[df.scenario == "base"].cost.median())
        kd = df[df.scenario == "base"]
        kd = kd[kd.denom > 0.05 * med]
        keys = set(zip(kd.site_id, kd.i0))
        out = {}
        heal = df[df.scenario.str.startswith("heal")].copy()
        heal["axis"] = np.where(heal.scenario.str.startswith("healL"),
                                "load", "pv")
        heal["band"] = heal.scenario.str.split("_", n=1).str[1]
        heal = heal[[tuple(k) in keys for k in zip(heal.site_id, heal.i0)]]
        heal["recovered"] = heal.denom - heal.regret
        for axis in ["load", "pv"]:
            for band in BANDS:
                sub = heal[(heal.axis == axis) & (heal.band == band)]
                out[(axis, band)] = (100 * sub.recovered.sum() / sub.denom.sum()
                                     if len(sub) else np.nan)
        return out, len(keys)

    syn, n_syn = attribution(res)
    real_df = pd.read_parquet(RES / "surgery_heal60.parquet")
    bdf = (real_df[real_df.scenario == "base"]
           .set_index(["site_id", "i0"]).cost)
    pdf = (real_df[real_df.scenario == "perfect"]
           .set_index(["site_id", "i0"]).cost)
    denom_map = (bdf - pdf)
    real_df["denom"] = [denom_map.get((s, i), np.nan)
                        for s, i in zip(real_df.site_id, real_df.i0)]
    real, n_real = attribution(real_df)
    try:
        white_df = pd.read_parquet(RES / "surgery_synth_white.parquet")
        white, n_white = attribution(white_df)
    except Exception:
        white, n_white = None, 0

    lines = [f"# A1 synthetic ({MODE}) vs real attribution "
             "(heal, cost-weighted %)\n",
             f"valid issue times: synthetic {n_syn}, real {n_real}\n",
             "| axis | band | real [%] | synthetic-block [%] | synthetic-white [%] |",
             "|---|---|---|---|---|"]
    for axis in ["load", "pv"]:
        for band in BANDS:
            wtxt = f"{white[(axis, band)]:.1f}" if white else "—"
            lines.append(f"| {axis} | {band} | {real[(axis, band)]:.1f} | "
                         f"{syn[(axis, band)]:.1f} | {wtxt} |")
    txt = "\n".join(lines)
    (RES / "synth_report.md").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
