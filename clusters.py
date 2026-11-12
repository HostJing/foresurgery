"""Load-shape site typing.

Five dimensionless features are computed per site on the *local* clock (the
offset estimated in :func:`foresurgery.surgery.estimate_utc_offset_hours`):

=================  =====================================================
feature            definition
=================  =====================================================
``workday_share``  share of days whose mean load exceeds 0.55x the site median
``day_night``      mean load 07-19 h local / mean load 19-07 h local
``weekday_weekend`` mean weekday load / mean weekend load
``load_factor``    mean load / 99th percentile load
``night_day``      mean load 00-06 h / mean load 07-19 h
=================  =====================================================

Features are standardised and clustered with a numpy-only k-means (50 seeded
restarts).  ``k`` is chosen among {3, 4, 5} by the between/total variance ratio
subject to every cluster holding at least three sites: k=5 produced a singleton
on this fleet, which cannot be reported as a site *type*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BANDS, BAND_LIST, RESULTS_DIR
from .data import N_SITES, load_site
from .surgery import estimate_utc_offset_hours

__all__ = ["FEATURE_COLS", "site_features", "kmeans", "cluster_type",
           "fit_site_clusters", "save_site_clusters"]

FEATURE_COLS = ["workday_share", "day_night", "weekday_weekend",
                "load_factor", "night_day"]


def site_features(sid: int) -> dict:
    """Load-shape descriptors for one site."""
    df, F, G = load_site(sid)
    load = df["actual_load"].to_numpy(dtype=np.float64)
    ts = pd.DatetimeIndex(df["ts"])
    off = estimate_utc_offset_hours(load, ts)
    local = (ts + pd.Timedelta(hours=off)).round("15min")

    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    daily = pd.Series(load, index=local).resample("1D").mean()
    med_day = float(daily.median())
    work = float((daily > 0.55 * med_day).mean())

    day_mask = (hour >= 7) & (hour < 19)
    night_mask = ~day_mask
    day_night = float(np.nanmean(load[day_mask]) / (np.nanmean(load[night_mask]) + 1e-9))
    wk = dow < 5
    wknd_ratio = float(np.nanmean(load[wk]) / (np.nanmean(load[~wk]) + 1e-9))
    load_factor = float(np.nanmean(load) / (np.nanpercentile(load, 99) + 1e-9))
    night_ratio = float(np.nanmean(load[hour < 6]) / (np.nanmean(load[day_mask]) + 1e-9))

    return {
        "site_id": sid, "tz_off": off, "workday_share": work,
        "day_night": day_night, "weekday_weekend": wknd_ratio,
        "load_factor": load_factor, "night_day": night_ratio,
        "load_mean": float(np.nanmean(load)),
    }


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    """Seeded k-means with 50 restarts, best-inertia solution returned."""
    rng = np.random.default_rng(seed)
    n = len(X)
    best_labels, best_inertia = None, np.inf
    for _ in range(50):
        cent = X[rng.choice(n, k, replace=False)].copy()
        for _ in range(iters):
            d = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            newc = np.array([X[lab == j].mean(0) if (lab == j).any() else cent[j]
                             for j in range(k)])
            if np.allclose(newc, cent):
                cent = newc
                break
            cent = newc
        inertia = ((X - cent[lab]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, lab.copy()
    return best_labels


def cluster_type(row: pd.Series) -> str:
    """Heuristic name for a cluster centroid (reporting only, not used in the
    experiments)."""
    if row["night_day"] > 0.8 and row["workday_share"] > 0.9:
        return "continuous process"
    if row["weekday_weekend"] > 1.6:
        return "weekday batch"
    if row["day_night"] > 2.5:
        return "day-shift"
    return "two-shift / mixed"


def fit_site_clusters(n_sites: int = N_SITES, min_size: int = 3,
                      ks=(3, 4, 5), seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """Compute features and assign every site to a cluster type."""
    feats = pd.DataFrame([site_features(s) for s in range(1, n_sites + 1)])
    X = feats[FEATURE_COLS].to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)

    chosen = None
    for k in ks:
        lab = kmeans(X, k, seed=seed)
        sizes = np.bincount(lab)
        tot = ((X - X.mean(0)) ** 2).sum()
        between = sum(((X[lab == j].mean(0) - X.mean(0)) ** 2).sum() * sizes[j]
                      for j in range(k))
        score = between / tot
        ok = sizes.min() >= min_size
        if verbose:
            print(f"  k={k}: between/total variance={score:.3f} "
                  f"sizes={sizes.tolist()} {'' if ok else '(rejected)'}")
        if ok and (chosen is None or score > chosen[1]):
            chosen = (k, score, lab)

    k, score, lab = chosen
    feats["cluster"] = lab
    feats["cluster_type"] = (feats.groupby("cluster")[FEATURE_COLS].transform("mean")
                             .apply(cluster_type, axis=1))
    if verbose:
        print(f"chosen k={k} (between/total variance {score:.3f})")
        print(feats.groupby(["cluster", "cluster_type"]).size().to_string())
    return feats


def save_site_clusters(out_dir=None, **kwargs) -> pd.DataFrame:
    """Fit the clustering and write ``site_clusters.csv``."""
    out = RESULTS_DIR if out_dir is None else out_dir
    feats = fit_site_clusters(**kwargs)
    (out / "site_clusters.csv").parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out / "site_clusters.csv", index=False)
    print(f"saved -> {out / 'site_clusters.csv'}")
    return feats


def prescriptive_table(feats: pd.DataFrame, surgery: pd.DataFrame,
                       axis: str = "L") -> pd.DataFrame:
    """Cost-weighted band attribution per cluster (the Layer-4 table).

    ``axis='L'`` heals the load forecast, ``axis='P'`` the PV forecast.
    """
    base = surgery[surgery.scenario == "base"].set_index(["site_id", "i0"]).cost
    perf = surgery[surgery.scenario == "perfect"].set_index(["site_id", "i0"]).cost
    kd = (base - perf).rename("denom").reset_index()
    kd = kd[kd.denom > 0.05 * float(base.median())]
    h = surgery[surgery.scenario.str.startswith(f"heal{axis}")].merge(
        kd, on=["site_id", "i0"])
    h["band"] = h.scenario.str.split("_", n=1).str[1]
    h["recovered"] = h.denom - h.regret
    h = h.merge(feats[["site_id", "cluster"]], on="site_id")
    piv = h.groupby(["cluster", "band"]).apply(
        lambda g: 100 * g.recovered.sum() / g.denom.sum(),
        include_groups=False).unstack().reindex(columns=BAND_LIST).round(1)
    piv["n_sites"] = feats.groupby("cluster").size()
    return piv
