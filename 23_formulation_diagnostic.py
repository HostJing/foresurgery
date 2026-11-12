#!/usr/bin/env python
"""Diagnostic — how much do the two formulation choices change the attribution?

Reruns a small campaign under three variants of the dispatch model and prints
the cost-weighted attribution side by side:

``published``
    the model behind the manuscript: peak variable sharing the terminal-SOC slot
    and the initial SOC bounded below only;
``fixed-soc``
    same peak slot, initial SOC pinned to ``BATT['soc_init']``;
``corrected``
    peak in its own slot and initial SOC pinned.

See ``docs/LIMITATIONS.md`` for what the two switches do and why the default is
the published formulation.

Usage
-----
    python scripts/23_formulation_diagnostic.py                 # 12 sites, 30 issues
    python scripts/23_formulation_diagnostic.py --sites 1-5 --issues 20

Output: printed table, and ``work/results/formulation_diagnostic.csv``.
"""
import argparse

import numpy as np
import pandas as pd

import foresurgery.surgery as S
from foresurgery.attribution import cw_share, prepare_attribution
from foresurgery.config import BAND_LIST, RESULTS_DIR
from foresurgery.dispatch import solve_dispatch as _solve
from foresurgery.surgery import parse_sites

VARIANTS = {
    "published": dict(peak_slot="legacy", fix_initial_soc=False),
    "fixed-soc": dict(peak_slot="legacy", fix_initial_soc=True),
    "corrected": dict(peak_slot="own", fix_initial_soc=True),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sites", default="1,3,5,8,12,17,23,31,44,57,63,70")
    ap.add_argument("--issues", type=int, default=30)
    ap.add_argument("--c-dem", type=float, default=0.5)
    args = ap.parse_args(argv)

    sites = parse_sites(args.sites)
    rows = []
    for name, kw in VARIANTS.items():
        S.solve_dispatch = lambda *a, _kw=kw, **k: _solve(*a, **_kw, **k)
        frames = [S.run_site(s, c_dem=args.c_dem, max_issues=args.issues)
                  for s in sites]
        df = pd.concat(frames, ignore_index=True)
        h, _ = prepare_attribution(df)

        base = df[df.scenario == "base"].groupby(["site_id", "i0"]).cost.first()
        perf = df[df.scenario == "perfect"].groupby(["site_id", "i0"]).cost.first()
        bill = perf[perf > 1e-9]
        rec = {"variant": name,
               "regret_pct_of_bill": round(float(((base - perf) / bill).mean() * 100), 2),
               "valid_issue_times": int(h.key.nunique())}
        for band in BAND_LIST:
            rec[f"L_{band}"] = round(cw_share(h, "load", band), 1)
            rec[f"P_{band}"] = round(cw_share(h, "pv", band), 1)
        rows.append(rec)

        print(f"\n[{name}] {rec['valid_issue_times']} valid issue times, "
              f"regret {rec['regret_pct_of_bill']:.2f} % of the perfect-foresight bill")
        for band in BAND_LIST:
            print(f"   {band:8s} load {rec['L_' + band]:6.1f} %   "
                  f"pv {rec['P_' + band]:6.1f} %")

    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "formulation_diagnostic.csv", index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nwritten -> {RESULTS_DIR / 'formulation_diagnostic.csv'}")


if __name__ == "__main__":
    main()
