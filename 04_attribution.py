#!/usr/bin/env python
"""Stage 4 — cost-weighted band attribution, cluster breakdown, summary report.

This is the script behind the paper's main attribution table and figures 2-3.

Why cost-weighted
-----------------
Regret shares with near-zero denominators explode, so the fleet and cluster
aggregates are built from ratios of sums::

    share_h = sum_t (c_base - c_heal_h) / sum_t (c_base - c_perfect)

over issue times whose regret exceeds 5 % of the median base cost.

Inputs : work/results/surgery_full.parquet, work/results/site_clusters.csv
Outputs: work/results/fig_band_attribution.png
         work/results/fig_cluster_heatmap_{load,pv}.png
         work/results/fig_error_vs_value.png
         work/results/cluster_attribution_{load,pv}.csv
         work/results/summary_report.md

Usage
-----
    python scripts/04_attribution.py
    python scripts/04_attribution.py --tag heal60     # 60-issue campaign
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foresurgery.attribution import cw_stats, prepare_attribution, regret_profile
from foresurgery.config import BAND_LABELS, BAND_LIST, RESULTS_DIR

matplotlib.use("Agg")

BAND_PAIRS = list(zip(BAND_LIST, BAND_LABELS))


def table_by_axis_band(h: pd.DataFrame) -> pd.DataFrame:
    """Share [%] with CI and support, per axis and band."""
    rows = []
    for axis in ("load", "pv"):
        for band in BAND_LIST:
            s, lo, hi = cw_stats(h[(h.axis == axis) & (h.band == band)])
            rows.append({"axis": axis, "band": band, "share_pct": 100 * s,
                         "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                         "n_issue_times": h[(h.axis == axis)
                                            & (h.band == band)].key.nunique()})
    return pd.DataFrame(rows)


def figure_band_attribution(tab: pd.DataFrame, n_sites: int, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, axis in zip(axes, ("load", "pv")):
        sub = tab[tab.axis == axis].set_index("band").reindex(BAND_LIST)
        x = np.arange(len(BAND_LIST))
        means = sub.share_pct.to_numpy()
        ax.bar(x, means, color="#2b7bba" if axis == "load" else "#e1812c",
               width=0.62)
        ax.errorbar(x, means,
                    yerr=[means - sub.ci_lo.to_numpy(), sub.ci_hi.to_numpy() - means],
                    fmt="none", ecolor="k", capsize=3, lw=1)
        ax.set_xticks(x, BAND_LABELS)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"Heal {'load' if axis == 'load' else 'PV'} forecast band")
        ax.set_xlabel("Forecast horizon band healed (→ actual)")
    axes[0].set_ylabel("Cost-weighted share of regret recovered [%]")
    fig.suptitle("Counterfactual forecast surgery: which horizon band destroys "
                 f"scheduling value? ({n_sites} EMSx sites)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_band_attribution.png", bbox_inches="tight")
    plt.close(fig)


def figure_cluster_heatmaps(h: pd.DataFrame, out_dir):
    tables = {}
    for axis in ("load", "pv"):
        rows = []
        for c in sorted(h.cluster.dropna().unique()):
            for band in BAND_LIST:
                s, _, _ = cw_stats(h[(h.axis == axis) & (h.band == band)
                                     & (h.cluster == c)])
                rows.append((c, band, 100 * s))
        piv = (pd.DataFrame(rows, columns=["cluster", "band", "share"])
               .pivot(index="cluster", columns="band", values="share")
               .reindex(columns=BAND_LIST))
        sizes = h.groupby("cluster").site_id.nunique()
        fig, ax = plt.subplots(figsize=(5.6, 0.62 * len(piv) + 1.6))
        im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto",
                       vmin=-20, vmax=40)
        ax.set_xticks(range(len(BAND_LIST)), BAND_LABELS)
        ax.set_yticks(range(len(piv)),
                      [f"cluster {c} (n={sizes[c]})" for c in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center",
                        va="center", fontsize=9)
        ax.set_title(f"Cost-weighted regret share recovered [%] — "
                     f"heal {axis}, by site cluster")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_cluster_heatmap_{axis}.png",
                    bbox_inches="tight")
        plt.close(fig)
        piv.to_csv(out_dir / f"cluster_attribution_{axis}.csv")
        tables[axis] = piv
    return tables


def figure_error_vs_value(surgery: pd.DataFrame, h: pd.DataFrame, out_dir):
    err_cols = {b: f"err_{b}" for b in BAND_LIST}
    errs = (surgery[["site_id", "i0", *err_cols.values()]].dropna()
            .drop_duplicates(["site_id", "i0"]))
    errs = errs.merge(h[["site_id", "i0"]].drop_duplicates(),
                      on=["site_id", "i0"], how="inner")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, band in zip(axes, BAND_LIST):
        sub = h[(h.axis == "load") & (h.band == band)]
        site_r = sub.groupby("site_id").recovered.sum() / sub.groupby("site_id").denom.sum()
        e = errs.groupby("site_id")[err_cols[band]].mean()
        common = e.index.intersection(site_r.index)
        x = 100 * e[common] / e[common].mean()
        y = 100 * site_r[common]
        ax.scatter(x, y, s=18, alpha=0.7, color="#2b7bba")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{BAND_LABELS[BAND_LIST.index(band)]}  "
                     f"(r={np.corrcoef(x, y)[0, 1]:.2f})")
        ax.set_xlabel("site load-MAE / fleet mean [%]")
    axes[0].set_ylabel("cost-weighted share recovered [%]")
    fig.suptitle("Does a site's load-forecast error predict its "
                 "scheduling-value loss? (site level)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_error_vs_value.png", bbox_inches="tight")
    plt.close(fig)


def write_report(surgery, h, tab, cluster_tabs, out_dir) -> str:
    pct = regret_profile(surgery)
    neg = h[h.recovered < -0.05 * h.denom.abs()]
    lines = ["# Counterfactual forecast surgery — full-run summary", "",
             f"- sites: {surgery.site_id.nunique()}",
             f"- valid issue times: {h.key.nunique()}",
             f"- scenario rows: {len(surgery)}",
             f"- per-site regret as a share of the bill: median {pct.median():.1f} %, "
             f"IQR [{pct.quantile(.25):.1f} %, {pct.quantile(.75):.1f} %]", "",
             "## Cost-weighted attribution by horizon band", "",
             "| axis | band | share [%] | 95 % CI | site-issue times |",
             "|---|---|---|---|---|"]
    for r in tab.itertuples():
        lines.append(f"| {r.axis} | {r.band} | {r.share_pct:.1f} | "
                     f"[{r.ci_lo:.1f}, {r.ci_hi:.1f}] | {r.n_issue_times} |")
    for axis, piv in cluster_tabs.items():
        lines += ["", f"## Site cluster x horizon band (heal {axis}, %) ", "",
                  piv.round(1).to_string(), ""]
    lines += ["", "## Counter-intuitive cases", "",
              f"- issue-time/scenario pairs where healing made the cost worse by "
              f"more than 5 % of the regret: {100 * len(neg) / len(h):.1f} %", ""]
    text = "\n".join(lines)
    (out_dir / "summary_report.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="full", help="surgery table suffix")
    ap.add_argument("--clusters", default="site_clusters.csv")
    args = ap.parse_args(argv)

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
    out = RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    surgery = pd.read_parquet(out / f"surgery_{args.tag}.parquet")
    h, _ = prepare_attribution(surgery)
    h = h.merge(pd.read_csv(out / args.clusters)[["site_id", "cluster"]],
                on="site_id", how="left")
    print(f"valid issue times: {h.key.nunique()} across "
          f"{surgery.site_id.nunique()} sites")

    tab = table_by_axis_band(h)
    figure_band_attribution(tab, surgery.site_id.nunique(), out)
    cluster_tabs = figure_cluster_heatmaps(h, out)
    figure_error_vs_value(surgery, h, out)
    text = write_report(surgery, h, tab, cluster_tabs, out)
    print(text)
    print(f"\nfigures + summary -> {out}")


if __name__ == "__main__":
    main()
