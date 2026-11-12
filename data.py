"""EMSx warehouse construction and site loading.

The raw release ships one ``<site_id>.csv.gz`` per site (semicolon separated)
with, per 15-min step: the realised load and PV, and the 96-step load/PV
forecast *that was issued at that step*.  Two properties of the raw files
motivate the warehouse:

1. rows are not time-sorted for many sites;
2. data outages leave gaps, so a site is not one continuous 15-min grid.

``build_warehouse`` therefore sorts, folds duplicate timestamps, cuts the site
into contiguous *segments* (``seg_id`` increments at any step that is not
900 s), and stores the 96-step forecasts as fixed-shape tensors.  Downstream
code only ever asks whether 96 consecutive steps share one ``seg_id``: that is
exactly the condition under which an issued forecast can be executed open-loop
to the end of its horizon.

Layout on disk::

    warehouse/site_001.parquet   ts, actual_load, actual_pv, seg_id, F[n,96], G[n,96]
    warehouse/_summary.csv       per-site QA statistics

``F`` is the issued *load* forecast, ``G`` the issued *PV* forecast, both in
kWh per 15-min step.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, WAREHOUSE_DIR

__all__ = ["N_SITES", "LOAD_COLS", "PV_COLS", "site_path", "build_warehouse",
           "process_site", "load_site", "issue_mask"]

N_SITES = 70
LOAD_COLS = [f"load_{i:02d}" for i in range(96)]
PV_COLS = [f"pv_{i:02d}" for i in range(96)]


def site_path(site_id: int, warehouse_dir: Path | None = None) -> Path:
    root = WAREHOUSE_DIR if warehouse_dir is None else Path(warehouse_dir)
    return root / f"site_{site_id:03d}.parquet"


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def process_site(site_id: int, raw_dir: Path | None = None,
                 out_dir: Path | None = None) -> dict:
    """Sort, de-duplicate, segment and store one site. Returns QA statistics."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw_root = RAW_DIR if raw_dir is None else Path(raw_dir)
    out_root = WAREHOUSE_DIR if out_dir is None else Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(raw_root / f"{site_id}.csv.gz", sep=";")

    # 1. sort on the UTC timestamp (raw rows are not ordered)
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    order = np.argsort(ts.values, kind="stable")
    ts = ts.iloc[order].reset_index(drop=True)
    actual_load = df["actual_consumption"].to_numpy(dtype=np.float32)[order]
    actual_pv = df["actual_pv"].to_numpy(dtype=np.float32)[order]
    F = df[LOAD_COLS].to_numpy(dtype=np.float32)[order]
    G = df[PV_COLS].to_numpy(dtype=np.float32)[order]

    # 2. fold duplicate timestamps into their mean
    dup_mask = np.diff(ts.values).astype("timedelta64[s]").astype(np.int64) == 0
    n_dup = int(dup_mask.sum())
    if n_dup:
        keep = np.ones(len(ts), dtype=bool)
        for i in np.where(dup_mask)[0]:
            actual_load[i] = np.nanmean([actual_load[i], actual_load[i + 1]])
            actual_pv[i] = np.nanmean([actual_pv[i], actual_pv[i + 1]])
            F[i] = np.nanmean(np.vstack([F[i], F[i + 1]]), axis=0)
            G[i] = np.nanmean(np.vstack([G[i], G[i + 1]]), axis=0)
            keep[i + 1] = False
        ts, actual_load, actual_pv, F, G = (
            ts[keep].reset_index(drop=True), actual_load[keep], actual_pv[keep],
            F[keep], G[keep],
        )

    # 3. segments: break at any step that is not 900 s
    d = np.diff(ts.values).astype("timedelta64[s]").astype(np.int64)
    seg = np.concatenate([[0], np.cumsum(d != 900)]).astype(np.int32)
    n_seg = int(seg.max()) + 1

    # 4. usable issue rows: 96 consecutive steps inside one segment
    seg_ok = seg == np.roll(seg, -95)
    seg_ok[-95:] = False
    n_issue = int(seg_ok.sum())

    stats = {
        "site_id": site_id,
        "n_rows": len(ts),
        "t_start": str(ts.iloc[0]),
        "t_end": str(ts.iloc[-1]),
        "n_segments": n_seg,
        "n_dup_removed": n_dup,
        "longest_seg_rows": int(np.bincount(seg).max()),
        "n_issue_rows_96h": n_issue,
        "nan_forecast_pct": float(np.isnan(F).mean() * 100),
        "nan_actual_pct": float(np.isnan(actual_load).mean() * 100),
        "neg_actual": int((actual_load < 0).sum()),
        "actual_load_mean_kwh": float(np.nanmean(actual_load)),
        "actual_load_p99_kwh": float(np.nanpercentile(actual_load, 99)),
        "corr_f00": float(np.corrcoef(F[:, 0], actual_load)[0, 1]),
        "corr_f95": float(np.corrcoef(F[:, 95], np.roll(actual_load, -95))[0, 1]),
        "elapsed_s": round(time.time() - t0, 1),
    }

    tbl = pa.Table.from_pydict(
        {
            "ts": pa.array(ts.values, type=pa.timestamp("ns", tz="UTC")),
            "actual_load": pa.array(actual_load, type=pa.float32()),
            "actual_pv": pa.array(actual_pv, type=pa.float32()),
            "seg_id": pa.array(seg, type=pa.int32()),
            "F": pa.FixedShapeTensorArray.from_numpy_ndarray(F),
            "G": pa.FixedShapeTensorArray.from_numpy_ndarray(G),
        }
    )
    pq.write_table(tbl, out_root / f"site_{site_id:03d}.parquet", compression="zstd")
    return stats


def build_warehouse(sites=range(1, N_SITES + 1), raw_dir: Path | None = None,
                    out_dir: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Build the warehouse for *sites* and write ``_summary.csv``."""
    t0 = time.time()
    stats = []
    for sid in sites:
        s = process_site(sid, raw_dir=raw_dir, out_dir=out_dir)
        stats.append(s)
        if verbose:
            print(
                f"[{sid:3d}] rows={s['n_rows']:6d} segs={s['n_segments']:3d} "
                f"longest={s['longest_seg_rows']:6d} "
                f"issue96h={s['n_issue_rows_96h']:6d} "
                f"corrF0={s['corr_f00']:.3f} corrF95={s['corr_f95']:.3f} "
                f"({s['elapsed_s']}s)",
                flush=True,
            )
    summary = pd.DataFrame(stats)
    out_root = WAREHOUSE_DIR if out_dir is None else Path(out_dir)
    summary.to_csv(out_root / "_summary.csv", index=False)
    if verbose:
        print(f"all sites done in {time.time() - t0:.0f}s -> {out_root}")
    return summary


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_site(site_id: int, warehouse_dir: Path | None = None):
    """Return ``(df, F, G)`` for one site.

    ``df``  columns ts / actual_load / actual_pv / seg_id
    ``F``   float32 ``[n, 96]`` issued load forecast (kWh per step)
    ``G``   float32 ``[n, 96]`` issued PV forecast   (kWh per step)
    """
    import pyarrow.parquet as pq

    tbl = pq.read_table(site_path(site_id, warehouse_dir))
    df = tbl.select(["ts", "actual_load", "actual_pv", "seg_id"]).to_pandas()
    F = tbl.column("F").combine_chunks().to_numpy_ndarray()
    G = tbl.column("G").combine_chunks().to_numpy_ndarray()
    return df, F, G


def issue_mask(seg: np.ndarray, horizon: int = 96) -> np.ndarray:
    """Rows whose next *horizon* steps lie inside the same segment.

    Such a row is an issue time at which the 24-h schedule can actually be
    executed open-loop, i.e. the executed trajectory never steps over a data
    outage.
    """
    ok = seg == np.roll(seg, -(horizon - 1))
    ok[-(horizon - 1):] = False
    return ok
