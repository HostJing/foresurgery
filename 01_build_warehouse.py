#!/usr/bin/env python
"""Stage 1 — build the analysis warehouse from the raw EMSx release.

Reads ``<raw>/<site_id>.csv.gz`` for every site and writes one parquet per
site to the warehouse directory: sorted timestamps, de-duplicated rows,
``seg_id`` segmentation and the 96-step issued forecasts as fixed-shape
tensors. See :mod:`foresurgery.data` for the exact transformations.

Usage
-----
    python scripts/01_build_warehouse.py                       # all 70 sites
    python scripts/01_build_warehouse.py --sites 1-10          # a subset
    python scripts/01_build_warehouse.py --raw-dir /data/emsx  # explicit input

Cost: about 10 min for the full fleet (11 GB of CSV in, 1.8 GB parquet out).
"""
import argparse

from foresurgery.config import RAW_DIR, WAREHOUSE_DIR
from foresurgery.data import N_SITES, build_warehouse
from foresurgery.surgery import parse_sites


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", default=f"1-{N_SITES}",
                    help="site ids, e.g. 1-70 or 1,5,9-12")
    ap.add_argument("--raw-dir", default=str(RAW_DIR),
                    help=f"directory holding <site_id>.csv.gz (default {RAW_DIR})")
    ap.add_argument("--out-dir", default=str(WAREHOUSE_DIR),
                    help=f"warehouse directory (default {WAREHOUSE_DIR})")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    build_warehouse(parse_sites(args.sites), raw_dir=args.raw_dir,
                    out_dir=args.out_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
