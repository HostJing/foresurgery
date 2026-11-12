"""
Robustness batch (Layer 3): degrade dose-response + design-parameter sweep.
Run: python scripts/07_design_sweep.py   (sequential configs, one parquet per config)
"""

import time

import pandas as pd

from foresurgery.config import RESULTS_DIR
from foresurgery.surgery import parse_sites, run_site

OUT = RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)

SITES = list(range(1, 71))
N_ISSUES = 60

CONFIGS = []
# (a) degrade dose-response on default design
for k in (2.0, 4.0):
    CONFIGS.append(dict(tag=f"deg{k:g}", c_dem=0.5, e_share=0.10, degrade_k=k))
# (b) demand-charge sweep (battery 10%)
for cd in (0.2, 1.0, 2.0):
    CONFIGS.append(dict(tag=f"cdem{cd:g}", c_dem=cd, e_share=0.10, degrade_k=None))
# (c) battery-size sweep (c_dem 0.5)
for es in (0.05, 0.20, 0.40):
    CONFIGS.append(dict(tag=f"batt{es:g}", c_dem=0.5, e_share=es, degrade_k=None))


def main():
    t00 = time.time()
    for cfg in CONFIGS:
        t0 = time.time()
        frames = []
        for sid in SITES:
            try:
                frames.append(run_site(sid, 1, cfg["c_dem"], N_ISSUES,
                                       e_share=cfg["e_share"],
                                       degrade_k=cfg["degrade_k"]))
            except Exception as e:
                print(f"  site {sid} FAILED: {e}", flush=True)
        res = pd.concat(frames, ignore_index=True)
        fp = OUT / f"surgery_{cfg['tag']}.parquet"
        res.to_parquet(fp, index=False)
        print(f"[{cfg['tag']:8s}] {len(res):6d} rows, {res.site_id.nunique()} sites "
              f"in {time.time() - t0:.0f}s -> {fp.name}", flush=True)
    print(f"ALL CONFIGS DONE in {(time.time() - t00) / 60:.1f} min")


if __name__ == "__main__":
    main()
