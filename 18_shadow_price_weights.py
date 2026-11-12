"""
C2: shadow-price weighting (SPO-lite).

The paper shows that BAND-level decision-value weights fail. This variant
injects decision value at STEP resolution using the LP's own shadow prices:
for each training issue, solve the dispatch LP on the issued forecast and
read the energy-balance duals lambda[t] (EUR/kWh sensitivity of cost to net
load at step t). Sample weight w[i,h] = |lambda[h]|, normalized to mean 1
per issue and clipped to [0.2, 5]. This is the continuous,
optimizer-derived analogue of band weighting - the mechanism DFL argues
for, without differentiating through the LP.

Run:  python retrain_shadow.py
Out : models/gbm_shadow.joblib, results/retrain_eval_shadow.parquet,
      results/shadow_report.md
"""

import time

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingRegressor

from foresurgery import corrector as R
from foresurgery.config import MODELS_DIR
from foresurgery.config import battery_specs
from foresurgery.config import RESULTS_DIR
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import estimate_utc_offset_hours, shifted_price

RES = RESULTS_DIR
MODELS = MODELS_DIR


def cmd_train():
    t0 = time.time()
    Xs, ys, ws = [], [], []
    for sid, df, F, tr, _ in R.collect():
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        price = shifted_price(estimate_utc_offset_hours(act_l, ts))
        b = battery_specs(act_l)
        X, y = R.build_matrix(sid, df, F, tr)
        if len(X) == 0:
            continue
        # dual-based weights per issue row (rows kept by build_matrix are the
        # first r rows; recompute the kept row list identically)
        keep = []
        for i in tr:
            lag_ok = all(i - l >= 0 for l in R.LAGS)
            if not lag_ok or np.isnan(act_l[i - 672:i + 96]).any():
                continue
            if np.isnan(F[i]).any():
                continue
            keep.append(i)
            if len(keep) * 96 >= len(X):
                break
        w_rows = []
        for i in keep:
            r = solve_dispatch(F[i].astype(np.float64) - act_p[i:i + 96],
                               price, b["p_kw"], b["e_kwh"], c_dem=0.5,
                               return_duals=True)
            if r is None:
                w_rows.append(np.ones(96))
                continue
            lam = np.abs(r[4])
            lam = lam / lam.mean() if lam.mean() > 1e-9 else np.ones(96)
            w_rows.append(np.clip(lam, 0.2, 5.0))
        w = np.concatenate(w_rows)
        if len(w) < len(X):
            w = np.concatenate([w, np.ones(len(X) - len(w))])
        Xs.append(X)
        ys.append(y)
        ws.append(w[:len(X)])
        print(f"  site {sid}: {len(X)} samples ({time.time() - t0:.0f}s)",
              flush=True)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    print(f"train matrix {X.shape}, weight range [{w.min():.2f}, {w.max():.2f}]",
          flush=True)
    m = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
        categorical_features=[0], l2_regularization=1.0, random_state=0)
    m.fit(X, y, sample_weight=w)
    dump(m, MODELS / "gbm_shadow.joblib")
    print(f"saved gbm_shadow.joblib ({time.time() - t0:.0f}s)", flush=True)


def cmd_eval():
    mu = load(MODELS / "gbm_uniform.joblib")
    mp = load(MODELS / "gbm_shadow.joblib")
    models = {"modelU": mu, "modelP": mp}
    rows = []
    t0 = time.time()
    for sid, df, F, tr, ev in R.collect(R.EVAL_ISSUES_PER_SITE):
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        price = shifted_price(estimate_utc_offset_hours(act_l, ts))
        b = None
        for i0 in ev[:R.EVAL_ISSUES_PER_SITE]:
            aL = act_l[i0:i0 + 96]
            aP = act_p[i0:i0 + 96]
            if np.isnan(aL).any() or np.isnan(aP).any():
                continue
            fS = F[i0].astype(np.float64)
            if np.isnan(fS).any():
                continue
            if b is None:
                b = battery_specs(act_l)
            try:
                preds = {n: R.predict_96(m, sid, df, F, i0)
                         for n, m in models.items()}
            except Exception:
                continue
            if any(len(pp) != 96 or np.isnan(pp).any() for pp in preds.values()):
                continue
            nl_act = aL - aP
            variants = {"base": fS, "perfect": aL, **preds}
            costs = {}
            for name, f in variants.items():
                r = solve_dispatch(f - aP, price, b["p_kw"], b["e_kwh"],
                                   c_dem=0.5)
                if r is None:
                    costs = None
                    break
                costs[name] = execute_open_loop(r[1], r[2], nl_act, price,
                                                c_dem=0.5)
            if costs is None:
                continue
            for name, f in variants.items():
                mae = 0.0 if name == "perfect" else float(np.abs(f - aL).mean())
                rows.append({"site_id": sid, "i0": int(i0), "scenario": name,
                             "cost": costs[name], "mae": mae})
        print(f"  site {sid} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(RES / "retrain_eval_shadow.parquet", index=False)

    piv = res.pivot_table(index=["site_id", "i0"], columns="scenario",
                          values="cost").dropna()
    med = float(res[res.scenario == "base"].cost.median())
    valid = (piv["base"] - piv["perfect"]) > 0.05 * med
    pv = piv[valid]
    base_r = (pv["base"] - pv["perfect"]).mean()
    mae_df = res[res.scenario != "perfect"].merge(
        pv.reset_index()[["site_id", "i0"]], on=["site_id", "i0"])
    lines = [f"# C2 shadow-price weighting (valid {valid.sum()}/{len(piv)})\n",
             "| forecaster | regret [EUR/day] | reduction | MAE [kWh/step] |",
             "|---|---|---|---|"]
    for name in ["base", "modelU", "modelP"]:
        r = (pv[name] - pv["perfect"]).mean() if name != "base" else base_r
        red = 100 * (1 - r / base_r)
        m = mae_df[mae_df.scenario == name].mae.mean()
        lines.append(f"| {name} | {r:.1f} | {red:+.1f}% | {m:.2f} |")
    # paired U vs P
    x = (100 * ((pv["modelP"] - pv["perfect"]) - (pv["modelU"] - pv["perfect"]))
         / base_r).dropna()
    rng = np.random.default_rng(0)
    boots = [rng.choice(x.values, len(x), replace=True).mean()
             for _ in range(500)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lines.append(f"\nshadow vs uniform: {x.mean():+.2f}% CI [{lo:+.2f}, {hi:+.2f}]")
    txt = "\n".join(lines)
    (RES / "shadow_report.md").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd_train()
    cmd_eval()
