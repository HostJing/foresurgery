"""
Layer 2: horizon-weighted forecast retraining, evaluated in the same
scheduling closed loop.

Design
------
Model  : global HistGradientBoostingRegressor predicting actual load (kWh/step)
         at horizon h. Features (all available at issue time i):
           site_id (categorical), horizon h, hour sin/cos, dow,
           lag load: L[i], L[i-4], L[i-96], L[i-672],
           roll mean 1h / 24h, and the *issued Schneider forecast* F[i,h]
         -> model acts as a residual corrector of the operational forecast.
Variants:
   uniform : sample_weight = 1
   surgery : sample_weight w_h ∝ cost-weighted band attribution
             [h0_2, h2_6, h6_12, h12_24] = [1.3, 2.8, 10.5, 19.4], mean-1 norm
Split  : temporal 80/20 per site; holdout = last 20%.
Eval   : at holdout issue times (60/site): dispatch LP with
         base(Schneider) | modelU | modelW | perfect, open-loop execution,
         same tariff/battery as the main experiment. PV forecast = Schneider
         (unchanged across variants, isolates the load side + avoids the
         single-site PV rescaling caveat).

Run:  python scripts/14_retrain_uniform_fleet.py train
      python scripts/14_retrain_uniform_fleet.py eval
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import argparse
from foresurgery.config import battery_specs
from foresurgery.config import MODELS_DIR, RESULTS_DIR
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import estimate_utc_offset_hours, shifted_price

RES = RESULTS_DIR
MODELS = MODELS_DIR
MODELS.mkdir(parents=True, exist_ok=True)

ATTR_LOAD = np.array([1.3, 2.8, 10.5, 19.4])
W_BAND = np.repeat(ATTR_LOAD / ATTR_LOAD.mean(), [8, 16, 24, 48])
TRAIN_ISSUES_PER_SITE = 400
EVAL_ISSUES_PER_SITE = 60
LAGS = [0, 4, 96, 672]


def build_matrix(sid: int, df, F, rows, horizons=range(96)):
    """rows: issue row indices. Returns X, y."""
    L = df["actual_load"].to_numpy(dtype=np.float64)
    ts = pd.DatetimeIndex(df["ts"])
    hod = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    dow = ts.day_of_week.to_numpy()
    n = len(rows) * len(horizons)
    X = np.empty((n, 11), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    r = 0
    for i in rows:
        lag_ok = all(i - l >= 0 for l in LAGS)
        if not lag_ok or np.isnan(L[i - 672:i + 96]).any():
            continue
        f_row = F[i]
        if np.isnan(f_row).any():
            continue
        h24 = hod[i]
        base = [sid,
                np.sin(2 * np.pi * h24 / 24), np.cos(2 * np.pi * h24 / 24),
                dow[i],
                L[i], L[i - 4], L[i - 96], L[i - 672],
                L[max(0, i - 4):i + 1].mean(), L[i - 96:i].mean()]
        tgt = L[i:i + 96]
        for h in horizons:
            X[r, :10] = base
            X[r, 10] = f_row[h]
            y[r] = tgt[h]
            r += 1
    return X[:r], y[:r]


def w_for_rows(n_rows_effective):
    """band weights aligned with horizon-major ordering (h inner loop)."""
    return np.tile(W_BAND, n_rows_effective // 96 + 1)[:n_rows_effective]


def collect(rows_per_site=TRAIN_ISSUES_PER_SITE, frac=0.8):
    """Yield per-site train/eval issue-row splits."""
    for sid in range(1, 71):
        df, F, G = load_site(sid)
        n = len(df)
        cut = int(n * frac)
        seg = df["seg_id"].to_numpy()
        ok = (seg == np.roll(seg, -95))
        ok[-95:] = False
        ok &= np.arange(n) >= 672 + 1  # lags available
        cand = np.where(ok)[0]
        train_c = cand[cand < cut]
        eval_c = cand[cand >= cut]
        rng = np.random.default_rng(sid)
        if len(train_c) > rows_per_site:
            train_c = rng.choice(train_c, rows_per_site, replace=False)
        yield sid, df, F, np.sort(train_c), np.sort(eval_c)


def cmd_train():
    t0 = time.time()
    Xs, ys, ws_u, ws_s = [], [], [], []
    for sid, df, F, tr, _ in collect():
        X, y = build_matrix(sid, df, F, tr)
        if len(X) == 0:
            continue
        Xs.append(X); ys.append(y)
        ws_u.append(np.ones(len(X), dtype=np.float32))
        ws_s.append(w_for_rows(len(X)))
        print(f"  site {sid}: {len(X)} samples", flush=True)
    X = np.vstack(Xs); y = np.concatenate(ys)
    wu = np.concatenate(ws_u); wsv = np.concatenate(ws_s)
    print(f"train matrix: {X.shape}, {time.time() - t0:.0f}s", flush=True)

    for name, w in [("uniform", wu), ("surgery", wsv)]:
        t1 = time.time()
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
            categorical_features=[0], l2_regularization=1.0, random_state=0)
        m.fit(X, y, sample_weight=w)
        from joblib import dump
        dump(m, MODELS / f"gbm_{name}.joblib")
        print(f"model {name}: fit {time.time() - t1:.0f}s", flush=True)


def predict_96(model, sid: int, df, F, i):
    """Vectorized 96-horizon prediction for issue row i."""
    X, _ = build_matrix(sid, df, F, [i])
    return model.predict(X)


def cmd_eval():
    from joblib import load
    mu = load(MODELS / "gbm_uniform.joblib")
    ms = load(MODELS / "gbm_surgery.joblib")
    rows = []
    t0 = time.time()
    for sid, df, F, tr, ev in collect(EVAL_ISSUES_PER_SITE):
        act_l = df["actual_load"].to_numpy(dtype=np.float64)
        act_p = df["actual_pv"].to_numpy(dtype=np.float64)
        ts = pd.DatetimeIndex(df["ts"])
        off = estimate_utc_offset_hours(act_l, ts)
        price = shifted_price(off)
        L = act_l
        b_dict = None
        for i0 in ev[:EVAL_ISSUES_PER_SITE]:
            aL = L[i0:i0 + 96]; aP = act_p[i0:i0 + 96]
            if np.isnan(aL).any() or np.isnan(aP).any():
                continue
            fS = F[i0].astype(np.float64)
            if np.isnan(fS).any():
                continue
            if b_dict is None:
                b_dict = battery_specs(L)
            nl_act = aL - aP
            try:
                fU = predict_96(mu, sid, df, F, i0)
                fW = predict_96(ms, sid, df, F, i0)
            except Exception:
                continue
            if len(fU) != 96 or len(fW) != 96 or np.isnan(fU).any() or np.isnan(fW).any():
                continue
            variants = {"base": fS, "modelU": fU, "modelW": fW, "perfect": aL}
            costs = {}
            for name, f in variants.items():
                r = solve_dispatch(f - aP, price, b_dict["p_kw"],
                                   b_dict["e_kwh"], c_dem=0.5)
                if r is None:
                    costs = None
                    break
                costs[name] = execute_open_loop(r[1], r[2], nl_act, price, c_dem=0.5)
            if costs is None:
                continue
            for name, c in costs.items():
                rows.append({"site_id": sid, "i0": int(i0), "scenario": name,
                             "cost": c,
                             "mae": float(np.abs(fS - aL).mean()) if name == "base"
                             else (float(np.abs(fU - aL).mean()) if name == "modelU"
                                   else (float(np.abs(fW - aL).mean()) if name == "modelW"
                                         else 0.0))})
        print(f"  site {sid} done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(RES / "retrain_eval.parquet", index=False)
    print(f"DONE {len(res)} rows -> retrain_eval.parquet")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=['train', 'eval'])
    args = ap.parse_args(argv)
    {"train": cmd_train, "eval": cmd_eval}[args.stage]()


if __name__ == "__main__":
    main()
