"""
A2 + A3 + A7: post-hoc analyses on existing surgery parquets.

A2  Peak-location stratification: attribute heals by where the ACTUAL daily
    net-load peak sits (band 0-6h / 6-12h / 12-24h). Prop 1(i) says the
    attribution peak should follow the peak location.
A3  Validity-filter sensitivity: attribution at thresholds 2.5/5/10% of the
    median base cost + profile of excluded issue times (local hour, site,
    PV share quartile).
A7  Value per unit error: heal attribution normalized by band MAE
    (marginal value per kWh of band error).

Outputs: results/review_a2a3_report.md (+ printed).
"""

import numpy as np
import pandas as pd

from foresurgery.config import BANDS
from foresurgery.config import RESULTS_DIR
from foresurgery.data import load_site

RES = RESULTS_DIR
BAND_LIST = list(BANDS)
BAND_LABELS = {"h0_2": "0-2h", "h2_6": "2-6h", "h6_12": "6-12h",
               "h12_24": "12-24h"}


def load_base():
    df = pd.read_parquet(RES / "surgery_full.parquet")
    base = df[df.scenario == "base"].set_index(["site_id", "i0"])
    perf = df[df.scenario == "perfect"].set_index(["site_id", "i0"])
    key = pd.DataFrame({"denom": base.cost - perf.cost}).reset_index()
    med_cost = float(base.cost.median())
    return df, key, med_cost


def attribution(df, key, thr_frac):
    """Cost-weighted heal attribution under threshold thr_frac * median cost."""
    med_cost = float(df[df.scenario == "base"].cost.median())
    kd = key[key.denom > thr_frac * med_cost]
    valid_keys = set(zip(kd.site_id, kd.i0))
    heal = df[df.scenario.str.startswith("heal")].copy()
    mask = [k in valid_keys for k in zip(heal.site_id, heal.i0)]
    heal = heal[mask]
    heal = heal.merge(kd[["site_id", "i0", "denom"]], on=["site_id", "i0"])
    heal["recovered"] = heal.denom - heal.regret
    heal["axis"] = np.where(heal.scenario.str.startswith("healL"), "load", "pv")
    heal["band"] = heal.scenario.str.split("_", n=1).str[1]
    out = {}
    for axis in ["load", "pv"]:
        for band in BAND_LIST:
            sub = heal[(heal.axis == axis) & (heal.band == band)]
            out[(axis, band)] = (100 * sub.recovered.sum() / sub.denom.sum()
                                 if len(sub) else np.nan)
    return out, len(kd), len(key)


def main():
    df, key, med_cost = load_base()

    # ---------------- A3: threshold sensitivity ----------------
    lines = ["# A3 validity-threshold sensitivity (cost-weighted heal shares %)\n"]
    rows = []
    for thr in [0.025, 0.05, 0.10]:
        attr, n_valid, n_tot = attribution(df, key, thr)
        rows.append((thr, n_valid, attr))
        lines.append(f"## threshold = {thr:.1%} of median base cost "
                     f"(valid {n_valid}/{n_tot} issue times)\n")
        lines.append("| axis | " + " | ".join(BAND_LABELS[b] for b in BAND_LIST) + " |")
        lines.append("|---|" + "---|" * 4)
        for axis in ["load", "pv"]:
            cells = " | ".join(f"{attr[(axis, b)]:.1f}" for b in BAND_LIST)
            lines.append(f"| {axis} | {cells} |")
        lines.append("")

    # excluded issue-time profile (threshold 5%)
    kd5 = key[key.denom > 0.05 * med_cost].copy()
    key["excluded"] = ~key.set_index(["site_id", "i0"]).index.isin(
        kd5.set_index(["site_id", "i0"]).index)
    exc = key[key.excluded]
    keep = key[~key.excluded]
    # local hour from ts
    ts_all = {}
    for sid in key.site_id.unique():
        d, _, _ = load_site(sid)
        ts_all[sid] = pd.DatetimeIndex(d["ts"])
    for name, sub in [("excluded", exc), ("kept", keep)]:
        hrs, pvshares = [], []
        for sid, i0 in zip(sub.site_id, sub.i0):
            hrs.append(ts_all[sid][i0].hour)
        lines.append(f"- {name}: n={len(sub)}, UTC-hour distribution "
                     f"{np.percentile(hrs, [25, 50, 75]).round(1).tolist()} (Q1/med/Q3)")
    # PV share quartiles of excluded vs kept
    first = df[df.scenario == "perfect"][["site_id", "i0", "load_mean", "pv_mean"]]
    key2 = key.merge(first, on=["site_id", "i0"], how="left")
    key2["pv_share"] = key2.pv_mean / (key2.load_mean + 1e-9)
    q = key2.pv_share.quantile([0.25, 0.5, 0.75])
    for name, sub in [("excluded", key2[key2.excluded]), ("kept", key2[~key2.excluded])]:
        lines.append(f"- {name}: mean PV share {sub.pv_share.mean():.3f}, "
                     f"mean UTC hour {pd.Series([ts_all[s][i].hour for s, i in zip(sub.site_id, sub.i0)]).mean():.1f}")

    # ---------------- A2: peak-location stratification ----------------
    lines.append("\n# A2 attribution stratified by ACTUAL peak location\n")
    peak_rows = []
    for sid in key.site_id.unique():
        d, _, _ = load_site(sid)
        nl = d["actual_load"].to_numpy(dtype=np.float64) - d["actual_pv"].to_numpy(dtype=np.float64)
        for _, r in kd5[kd5.site_id == sid].iterrows():
            i0 = int(r.i0)
            seg = nl[i0:i0 + 96]
            if np.isnan(seg).any():
                continue
            p = int(np.argmax(seg))
            if p < 24:
                pb = "0-6h"
            elif p < 48:
                pb = "6-12h"
            else:
                pb = "12-24h"
            peak_rows.append((sid, i0, pb))
    peak_df = pd.DataFrame(peak_rows, columns=["site_id", "i0", "peak_band"])
    counts = peak_df.peak_band.value_counts()
    lines.append(f"peak location counts: {counts.to_dict()}\n")

    heal = df[df.scenario.str.startswith("heal")].copy()
    heal = heal.merge(kd5[["site_id", "i0", "denom"]], on=["site_id", "i0"])
    heal["recovered"] = heal.denom - heal.regret
    heal["axis"] = np.where(heal.scenario.str.startswith("healL"), "load", "pv")
    heal["band"] = heal.scenario.str.split("_", n=1).str[1]
    heal = heal.merge(peak_df, on=["site_id", "i0"])

    lines.append("| peak band | heal axis | " + " | ".join(BAND_LABELS[b] for b in BAND_LIST) + " | n |")
    lines.append("|---|---|" + "---|" * 5)
    for pb in ["0-6h", "6-12h", "12-24h"]:
        for axis in ["load", "pv"]:
            cells, sub_all = [], heal[(heal.peak_band == pb) & (heal.axis == axis)]
            for band in BAND_LIST:
                sub = sub_all[sub_all.band == band]
                cells.append(f"{100 * sub.recovered.sum() / sub.denom.sum():.1f}"
                             if len(sub) else "--")
            lines.append(f"| {pb} | {axis} | " + " | ".join(cells) +
                         f" | {sub_all[['site_id','i0']].drop_duplicates().shape[0]} |")

    # ---------------- A7: value per unit band error ----------------
    lines.append("\n# A7 marginal value per kWh of band error (heal share / band MAE)\n")
    err_cols = ([f"err_{b}" for b in BAND_LIST] + [f"perr_{b}" for b in BAND_LIST])
    errs = df[["site_id", "i0"] + err_cols].dropna(subset=["err_h0_2"]).drop_duplicates(["site_id", "i0"])
    errs = errs.merge(kd5, on=["site_id", "i0"])
    attr, _, _ = attribution(df, key, 0.05)
    lines.append("| axis | band | share [%] | fleet band MAE [kWh] | share per kWh MAE |")
    lines.append("|---|---|---|---|---|")
    for axis, ecol in [("load", "err"), ("pv", "perr")]:
        for band in BAND_LIST:
            mae = float(errs[f"{ecol}_{band}"].mean())
            s = attr[(axis, band)]
            lines.append(f"| {axis} | {BAND_LABELS[band]} | {s:.1f} | {mae:.1f} | "
                         f"{s / mae:.3f} |")

    txt = "\n".join(lines)
    (RES / "review_a2a3_report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
