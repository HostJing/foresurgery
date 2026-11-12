"""
Layer 2 iteration 3: ENDOGENOUS (surgery-in-the-loop) band weights.

Motivation: fleet-level (modelW) and site-level (modelW2) weights derived
from Schneider's *own* error attribution were both falsified. Structural
diagnosis: the corrector's residual structure differs from Schneider's
error structure, so exogenous weights aim at the wrong distribution.

Fix: measure the value structure of the CORRECTOR'S OWN errors.
  Step 1 (calibrate): predict with modelU on calibration issue points
          (training period only - no holdout leakage); run band surgery
          on modelU's net-load forecasts (heal 4 bands, load side;
          PV kept at actual, same isolation as eval).
          -> V_modelU[site, band] = cost-weighted recovered share.
  Step 2 (train)    : retrain with sample weights w[site,band] from
          V_modelU (mean-1 per site, clip [0.2, 5]) -> modelE.
  Step 3 (eval)     : same closed loop, all five variants:
          base | perfect | modelU | modelW | modelW2 | modelE.

Run:  python scripts/16_retrain_endogenous.py calibrate
      python scripts/16_retrain_endogenous.py train
      python scripts/16_retrain_endogenous.py eval
"""

import time

import numpy as np
import pandas as pd

from foresurgery import corrector as R
import argparse
from foresurgery.config import MODELS_DIR
from foresurgery.config import BANDS, battery_specs
from foresurgery.config import RESULTS_DIR
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import estimate_utc_offset_hours, shifted_price

RES = RESULTS_DIR
MODELS = MODELS_DIR
ATTR_CSV = RES / "modelU_band_weights.csv"
CAL_ISSUES = 50


# ---------------------------------------------------------------- step 1
def cmd_calibrate():
    from joblib import load
    mu = load(MODELS / "gbm_uniform.joblib")

    rows = []
    t0 = time.time()
    for sid, df, F, tr, _ in R.collect():
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        price = shifted_price(estimate_utc_offset_hours(act_l, ts))
        b = battery_specs(act_l)
        # calibration points: first CAL_ISSUES of the train subset
        done = 0
        for i0 in tr:
            if done >= CAL_ISSUES:
                break
            aL = act_l[i0:i0 + 96]; aP = act_p[i0:i0 + 96]
            if np.isnan(aL).any() or np.isnan(aP).any():
                continue
            try:
                fU = R.predict_96(mu, sid, df, F, i0)
            except Exception:
                continue
            if len(fU) != 96 or np.isnan(fU).any():
                continue
            nl_fc = fU - aP
            nl_act = aL - aP
            c_perf = None
            r = solve_dispatch(nl_act, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
            if r is None:
                continue
            c_perf = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
            r = solve_dispatch(nl_fc, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
            if r is None:
                continue
            c_base = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
            denom = c_base - c_perf
            for band in BANDS:
                lo, hi = BANDS[band]
                f2 = fU.copy(); f2[lo:hi] = aL[lo:hi]
                r = solve_dispatch(f2 - aP, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
                if r is None:
                    continue
                c_h = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
                rows.append({"site_id": sid, "i0": int(i0), "band": band,
                             "denom": denom, "recovered": c_base - c_h})
            done += 1
        print(f"  site {sid} calibrated ({time.time() - t0:.0f}s)", flush=True)

    cal = pd.DataFrame(rows)
    cal.to_parquet(RES / "modelU_calibration.parquet", index=False)

    # cost-weighted attribution per site x band -> weights
    kd = cal[cal.denom > 0.05 * float(cal.denom.median())]
    piv = kd.groupby(["site_id", "band"]).apply(
        lambda g: g.recovered.sum() / max(g.denom.sum(), 1e-9),
        include_groups=False).unstack().reindex(columns=list(BANDS))
    w = piv.div(piv.mean(axis=1), axis=0).clip(0.2, 5.0)
    # sites with NaN (all-empty denominators) -> uniform
    w = w.fillna(1.0)
    w.to_csv(ATTR_CSV)
    print(f"wrote {ATTR_CSV} ({len(w)} sites)")
    print(w.round(2).to_string())


# ---------------------------------------------------------------- step 2
def cmd_train():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from joblib import dump
    w_df = pd.read_csv(ATTR_CSV, index_col=0)
    w_df.columns = list(BANDS)

    Xs, ys, ws = [], [], []
    for sid, df, F, tr, _ in R.collect():
        X, y = R.build_matrix(sid, df, F, tr)
        if len(X) == 0:
            continue
        Xs.append(X); ys.append(y)
        if sid in w_df.index:
            v = w_df.loc[sid].to_numpy(dtype=np.float64)
        else:
            v = np.ones(4)
        v = np.clip(v, 0.2, 5.0)
        ws.append(np.tile(np.repeat(v, [8, 16, 24, 48]), len(X) // 96))
    X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    print(f"train matrix {X.shape}; weight range [{w.min():.2f}, {w.max():.2f}]")
    m = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
        categorical_features=[0], l2_regularization=1.0, random_state=0)
    m.fit(X, y, sample_weight=w)
    dump(m, MODELS / "gbm_endogenous.joblib")
    print("saved gbm_endogenous.joblib")


# ---------------------------------------------------------------- step 3
def cmd_eval():
    from joblib import load
    mu = load(MODELS / "gbm_uniform.joblib")
    ms = load(MODELS / "gbm_surgery.joblib")
    m2 = load(MODELS / "gbm_surgery2.joblib")
    me = load(MODELS / "gbm_endogenous.joblib")
    models = {"modelU": mu, "modelW": ms, "modelW2": m2, "modelE": me}

    rows = []
    t0 = time.time()
    for sid, df, F, tr, ev in R.collect(R.EVAL_ISSUES_PER_SITE):
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        price = shifted_price(estimate_utc_offset_hours(act_l, ts))
        b = None
        for i0 in ev[:R.EVAL_ISSUES_PER_SITE]:
            aL = act_l[i0:i0 + 96]; aP = act_p[i0:i0 + 96]
            if np.isnan(aL).any() or np.isnan(aP).any():
                continue
            fS = F[i0].astype(np.float64)
            if np.isnan(fS).any():
                continue
            if b is None:
                b = R.battery_specs(act_l)
            try:
                preds = {n: R.predict_96(m, sid, df, F, i0)
                         for n, m in models.items()}
            except Exception:
                continue
            if any(len(p) != 96 or np.isnan(p).any() for p in preds.values()):
                continue
            nl_act = aL - aP
            variants = {"base": fS, "perfect": aL, **preds}
            costs = {}
            for name, f in variants.items():
                r = solve_dispatch(f - aP, price, b["p_kw"], b["e_kwh"], c_dem=0.5)
                if r is None:
                    costs = None
                    break
                costs[name] = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
            if costs is None:
                continue
            for name, f in variants.items():
                mae = 0.0 if name == "perfect" else float(np.abs(f - aL).mean())
                rows.append({"site_id": sid, "i0": int(i0), "scenario": name,
                             "cost": costs[name], "mae": mae})
        print(f"  site {sid} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(RES / "retrain_eval3.parquet", index=False)
    print(f"DONE {len(res)} rows -> retrain_eval3.parquet")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=['calibrate', 'train', 'eval'])
    args = ap.parse_args(argv)
    {"calibrate": cmd_calibrate, "train": cmd_train, "eval": cmd_eval}[args.stage]()


if __name__ == "__main__":
    main()
