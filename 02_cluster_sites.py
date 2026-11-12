#!/usr/bin/env python
"""Stage 2 — type the 70 sites by the shape of their load.

Computes five dimensionless load-shape features per site, standardises them
and runs a seeded k-means. ``k`` is chosen among {3, 4, 5} by the between/total
variance ratio, subject to every cluster holding at least three sites, because
k=5 isolates a single site and a singleton is not a reportable site type.

If ``--report`` is given and the main surgery table is present, the
cost-weighted load and PV attribution is also tabulated per cluster — the
prescriptive table of the paper's last results subsection.

Output: ``work/results/site_clusters.csv``

Usage
-----
    python scripts/02_cluster_sites.py
    python scripts/02_cluster_sites.py --report
"""
import argparse

import pandas as pd

from foresurgery.config import RESULTS_DIR
from foresurgery.clusters import fit_site_clusters, prescriptive_table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true",
                    help="also tabulate attribution per cluster (needs the "
                         "main surgery table)")
    ap.add_argument("--surgery-tag", default="full")
    args = ap.parse_args(argv)

    feats = fit_site_clusters()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_csv(RESULTS_DIR / "site_clusters.csv", index=False)
    print(f"saved -> {RESULTS_DIR / 'site_clusters.csv'}")
    print(feats.groupby("cluster")[["workday_share", "day_night",
                                    "weekday_weekend", "load_factor",
                                    "night_day"]].mean().round(2).to_string())

    fp = RESULTS_DIR / f"surgery_{args.surgery_tag}.parquet"
    if args.report and fp.exists():
        surgery = pd.read_parquet(fp)
        for axis, name in (("L", "load"), ("P", "PV")):
            piv = prescriptive_table(feats, surgery, axis=axis)
            print(f"\n=== cost-weighted {name} attribution by cluster [%] ===")
            print(piv.to_string())
            piv.to_csv(RESULTS_DIR / f"cluster_attribution_{name}.csv")
    elif args.report:
        print(f"[skip] {fp.name} not found; run 03_run_surgery.py first")


if __name__ == "__main__":
    main()
