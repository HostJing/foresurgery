"""Feature matrix and rolling-archive splits for the retraining experiments.

The corrector predicts the *realised* load 96 steps ahead and is given the
issued forecast as one of its inputs, so it learns a residual correction on top
of what the operator already had.  All features are available at issue time
``i``; the horizon ``h`` enters only through the issued forecast value
``F[i, h]``, which is why the matrix is horizon-major (96 consecutive rows per
issue time).

Splits are temporal per site: the last 20 % of the archive is the holdout, so
no observation is used to train a model that is later evaluated on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import N_SITES, load_site

__all__ = ["ATTR_LOAD", "W_BAND", "TRAIN_ISSUES_PER_SITE", "EVAL_ISSUES_PER_SITE",
           "LAGS", "build_matrix", "w_for_rows", "collect", "predict_96"]

# Fleet cost-weighted load attribution from the main surgery run, used as the
# "exogenous" band weights of the paper's model W.
ATTR_LOAD = np.array([1.3, 2.8, 10.5, 19.4])
W_BAND = np.repeat(ATTR_LOAD / ATTR_LOAD.mean(), [8, 16, 24, 48])

TRAIN_ISSUES_PER_SITE = 400
EVAL_ISSUES_PER_SITE = 60
LAGS = [0, 4, 96, 672]


def build_matrix(sid: int, df, F, rows, horizons=range(96)):
    """Feature matrix and targets for the issue rows in *rows*.

    Columns: site id (categorical), hour sin, hour cos, day of week, the four
    lags, two rolling means, and the issued forecast at the horizon.
    """
    L = df["actual_load"].to_numpy(dtype=np.float64)
    ts = pd.DatetimeIndex(df["ts"])
    hod = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    dow = ts.day_of_week.to_numpy()
    n = len(rows) * len(horizons)
    X = np.empty((n, 11), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    r = 0
    for i in rows:
        lag_ok = all(i - lag >= 0 for lag in LAGS)
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


def w_for_rows(n_rows_effective: int) -> np.ndarray:
    """Fleet band weights aligned with the horizon-major row order."""
    return np.tile(W_BAND, n_rows_effective // 96 + 1)[:n_rows_effective]


def collect(rows_per_site: int = TRAIN_ISSUES_PER_SITE, frac: float = 0.8,
            n_sites: int = N_SITES):
    """Yield ``(site_id, df, F, train_rows, eval_rows)`` per site."""
    for sid in range(1, n_sites + 1):
        df, F, G = load_site(sid)
        n = len(df)
        cut = int(n * frac)
        seg = df["seg_id"].to_numpy()
        ok = (seg == np.roll(seg, -95))
        ok[-95:] = False
        ok &= np.arange(n) >= 672 + 1  # longest lag available
        cand = np.where(ok)[0]
        train_c = cand[cand < cut]
        eval_c = cand[cand >= cut]
        rng = np.random.default_rng(sid)
        if len(train_c) > rows_per_site:
            train_c = rng.choice(train_c, rows_per_site, replace=False)
        yield sid, df, F, np.sort(train_c), np.sort(eval_c)


def predict_96(model, sid: int, df, F, i: int) -> np.ndarray:
    """Predict the 96-step load trajectory for issue row *i*."""
    X, _ = build_matrix(sid, df, F, [i])
    return model.predict(X)
