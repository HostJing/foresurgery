#!/usr/bin/env python
"""Stage 3 — the main experiment: counterfactual forecast surgery.

For every sampled issue time of every site, ten scenarios are dispatched and
executed open-loop: perfect foresight, the forecast as issued, and the heal of
each of the four horizon bands on each of the two channels (load, PV). See
:mod:`foresurgery.surgery` for the operator and :mod:`foresurgery.dispatch`
for the LP.

Usage
-----
    python scripts/03_run_surgery.py --tag full                 # 180 issues/site
    python scripts/03_run_surgery.py --tag heal60 --max-issues 60
    python scripts/03_run_surgery.py --sites 1-5 --max-issues 4 --tag pilot

Output: ``work/results/surgery_<tag>.parquet`` (one row per issue time and
scenario). The paper's main table and figures 2-4 use ``full`` and ``heal60``.
"""
import argparse

from foresurgery.config import DEFAULT_C_DEM
from foresurgery.surgery import main as surgery_main


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", default="1-70")
    ap.add_argument("--max-issues", type=int, default=180,
                    help="issue times per site, spread evenly (60 and 180 are "
                         "the paper's two campaigns)")
    ap.add_argument("--c-dem", type=float, default=DEFAULT_C_DEM,
                    help="demand charge, EUR/kW/day")
    ap.add_argument("--e-share", type=float, default=None,
                    help="battery energy as a share of mean daily load")
    ap.add_argument("--degrade-k", type=float, default=None,
                    help="amplify the issued error instead of healing")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args(argv)

    surgery_main(["--sites", args.sites,
                  "--max-issues", str(args.max_issues),
                  "--c-dem", str(args.c_dem),
                  "--tag", args.tag]
                 + ([] if args.e_share is None else ["--e-share", str(args.e_share)])
                 + ([] if args.degrade_k is None else ["--degrade-k", str(args.degrade_k)]))


if __name__ == "__main__":
    main()
