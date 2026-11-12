"""
A4: simple baseline forecasters in the same scheduling closed loop.

Motivation (review): is the +11% cost reduction a HistGBM-specific result or
would ANY sensible corrector achieve it? Two baselines on the same 800 valid
eval issue times, same LP/tariff/battery, PV held at actuals:
  - seasonal-naive persistence: f[i,h] = L[i-96+h]  (same time yesterday)
  - ridge regression on the corrector feature set (site one-hot + 9 numerics
    + issued forecast), trained on the same 80% temporal window

Run:  python retrain_baselines.py
Out : results/retrain_eval_baselines.parquet, results/baselines_report.md
"""

import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge

from foresurgery import corrector as R
from foresurgery.config import MODELS_DIR
from foresurgery.config import battery_specs
from foresurgery.config import RESULTS_DIR
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import estimate_utc_offset_hours, shifted_price

RES = RESULTS_DIR
MODELS = MODELS_DIR


def build_ridge_features(sid, df, F, rows):
    """Numeric features (cols 1..10 of build_matrix) + site one-hot."""
    X, y = R.build_matrix(sid, df, F, rows)
    n_num = X.shape[1] - 1  # drop numeric site id
    onehot = np.zeros((X.shape[0], 70), dtype=np.float32)
    onehot[np.arange(X.shape[0]), sid - 1] = 1.0
    return np.hstack([X[:, 1:].astype(np.float32), onehot]), y


def main():
    # ---------------- train ridge ----------------
    t0 = time.time()
    Xs, ys = [], []
    for sid, df, F, tr, _ in R.collect():
        X, y = build_ridge_features(sid, df, F, tr)
        if len(X):
            Xs.append(X)
            ys.append(y)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    print(f"ridge train matrix {X.shape} ({time.time() - t0:.0f}s)", flush=True)
    mu = X[:, :9].mean(0)
    sd = X[:, :9].std(0) + 1e-9
    X[:, :9] = (X[:, :9] - mu) / sd
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    dump({"ridge": ridge, "mu": mu, "sd": sd}, MODELS / "ridge.joblib")
    print(f"ridge fit done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- closed-loop eval ----------------
    rows = []
    for sid, df, F, tr, ev in R.collect(R.EVAL_ISSUES_PER_SITE):
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        price = shifted_price(estimate_utc_offset_hours(act_l, ts))
        b = None
        for i0 in ev[:R.EVAL_ISSUES_PER_SITE]:
            aL = act_l[i0:i0 + 96]
            aP = act_p[i0:i0 + 96]
            if np.isnan(aL).any() or np.isnan(aP).any() or i0 < 96:
                continue
            fS = F[i0].astype(np.float64)
            if np.isnan(fS).any():
                continue
            if b is None:
                b = battery_specs(act_l)
            f_pers = act_l[i0 - 96:i0]  # seasonal naive
            Xr, _ = build_ridge_features(sid, df, F, [i0])
            if len(Xr) != 96:
                continue
            f_ridge = ridge.predict(np.hstack([(Xr[:, :9] - mu) / sd,
                                               Xr[:, 9:]]))
            if np.isnan(f_ridge).any():
                continue
            nl_act = aL - aP
            variants = {"base": fS, "persistence": f_pers,
                        "ridge": f_ridge, "perfect": aL}
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
                mae = (0.0 if name == "perfect"
                       else float(np.abs(f - aL).mean()))
                rows.append({"site_id": sid, "i0": int(i0),
                             "scenario": name, "cost": costs[name],
                             "mae": mae})
        print(f"  site {sid} ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_parquet(RES / "retrain_eval_baselines.parquet", index=False)

    # ---------------- report on valid subset ----------------
    piv = res.pivot_table(index=["site_id", "i0"], columns="scenario",
                          values="cost").dropna()
    med = float(res[res.scenario == "base"].cost.median())
    valid = (piv["base"] - piv["perfect"]) > 0.05 * med
    pv = piv[valid]
    base_r = (pv["base"] - pv["perfect"]).mean()
    lines = [f"# A4 baselines (valid {valid.sum()}/{len(piv)} issue times)\n",
             "| forecaster | pooled regret [EUR/day] | reduction vs issued | "
             "MAE [kWh/step] (valid) |", "|---|---|---|---|"]
    mae_df = res[res.scenario != "perfect"]
    for name in ["base", "persistence", "ridge"]:
        r = (pv[name] - pv["perfect"]).mean() if name != "base" else base_r
        red = 100 * (1 - r / base_r)
        m = mae_df.merge(pv.reset_index()[["site_id", "i0"]],
                         on=["site_id", "i0"])
        m = m[m.scenario == name].mae.mean()
        lines.append(f"| {name} | {r:.1f} | {red:+.1f}% | {m:.1f} |")
    txt = "\n".join(lines)
    (RES / "baselines_report.md").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
