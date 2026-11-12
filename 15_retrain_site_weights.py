"""
Layer 2 iteration 2: site-specific band weights.

w[site, band] ∝ cost-weighted attribution V_load(site, band) from the main
surgery run (surgery_full), normalized per site to mean 1, clipped to [0.2, 5].
Train a third model (surgery2) and evaluate in the same closed loop as
base / modelU / modelW(fleet weights).

Run:  python scripts/15_retrain_site_weights.py train
      python scripts/15_retrain_site_weights.py eval
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from foresurgery import corrector as R
from foresurgery.config import BAND_LIST, MODELS_DIR, RESULTS_DIR
from foresurgery.data import load_site
from foresurgery.dispatch import execute_open_loop, solve_dispatch
from foresurgery.surgery import estimate_utc_offset_hours, shifted_price

RES = RESULTS_DIR
MODELS = MODELS_DIR
BANDS = BAND_LIST


def build_site_weights() -> pd.DataFrame:
    """Per-site load-side band attribution -> normalized weights."""
    df = pd.read_parquet(RES / "surgery_full.parquet")
    base = df[df.scenario == "base"].set_index(["site_id", "i0"]).cost
    perf = df[df.scenario == "perfect"].set_index(["site_id", "i0"]).cost
    kd = (base - perf).rename("denom").reset_index()
    kd = kd[kd.denom > 0.05 * float(base.median())]
    h = df[df.scenario.str.startswith("healL")].merge(kd, on=["site_id", "i0"])
    h["band"] = h.scenario.str.split("_", n=1).str[1]
    h["recovered"] = h.denom - h.regret
    piv = h.groupby(["site_id", "band"]).apply(
        lambda g: g.recovered.sum() / g.denom.sum(), include_groups=False
    ).unstack().reindex(columns=BANDS)
    w = piv.div(piv.mean(axis=1), axis=0).clip(0.2, 5.0)
    w.to_csv(RES / "site_band_weights.csv")
    print(w.round(2).to_string())
    return w


class SiteWeighter:
    def __init__(self, w_csv: Path):
        w = pd.read_csv(w_csv, index_col=0)
        w.columns = BANDS
        self.w = w

    def for_site(self, sid: int) -> np.ndarray:
        if sid in self.w.index:
            v = self.w.loc[sid].to_numpy(dtype=np.float64)
        else:  # unseen site -> uniform
            v = np.ones(4)
        return np.repeat(v, [8, 16, 24, 48])


def cmd_train():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from joblib import dump
    build_site_weights()
    sw = SiteWeighter(RES / "site_band_weights.csv")

    Xs, ys, ws = [], [], []
    for sid, df, F, tr, _ in R.collect():
        X, y = R.build_matrix(sid, df, F, tr)
        if len(X) == 0:
            continue
        Xs.append(X); ys.append(y)
        n_i = len(X) // 96
        ws.append(np.tile(sw.for_site(sid), n_i))
    X = np.vstack(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    print(f"train matrix {X.shape}; weight range [{w.min():.2f}, {w.max():.2f}]")
    m = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
        categorical_features=[0], l2_regularization=1.0, random_state=0)
    m.fit(X, y, sample_weight=w)
    dump(m, MODELS / "gbm_surgery2.joblib")
    print("saved gbm_surgery2.joblib")


def cmd_eval():
    from joblib import load
    mu = load(MODELS / "gbm_uniform.joblib")
    ms = load(MODELS / "gbm_surgery.joblib")
    m2 = load(MODELS / "gbm_surgery2.joblib")
    rows = []
    import time
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
                preds = {"modelU": R.predict_96(mu, sid, df, F, i0),
                         "modelW": R.predict_96(ms, sid, df, F, i0),
                         "modelW2": R.predict_96(m2, sid, df, F, i0)}
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
            for name, c in costs.items():
                mae = (float(np.abs(fS - aL).mean()) if name == "base" else
                       float(np.abs(aL).mean()) * 0 + {
                           "modelU": np.abs(preds["modelU"] - aL).mean(),
                           "modelW": np.abs(preds["modelW"] - aL).mean(),
                           "modelW2": np.abs(preds["modelW2"] - aL).mean(),
                       }.get(name, 0.0))
                rows.append({"site_id": sid, "i0": int(i0), "scenario": name,
                             "cost": c, "mae": float(mae)})
        print(f"  site {sid} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(RES / "retrain_eval2.parquet", index=False)
    print(f"DONE {len(res)} rows -> retrain_eval2.parquet")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=['train', 'eval'])
    args = ap.parse_args(argv)
    {"train": cmd_train, "eval": cmd_eval}[args.stage]()


if __name__ == "__main__":
    main()
