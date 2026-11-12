"""
A6: site-level cluster bootstrap for the closed-loop forecaster comparison.

Issue times within a site are serially dependent; the per-issue-time paired
bootstrap may understate variance. Re-test the key pairwise claims resampling
SITES (clusters), on the SAME 800 valid issue times as paper Table 1.

Input : results/retrain_eval3.parquet
Output: results/bootstrap_cluster_report.md
"""

import numpy as np
import pandas as pd

from foresurgery.config import RESULTS_DIR

RES = RESULTS_DIR


def main():
    df = pd.read_parquet(RES / "retrain_eval3.parquet")
    piv = df.pivot_table(index=["site_id", "i0"], columns="scenario",
                         values="cost").dropna()
    med = float(df[df.scenario == "base"].cost.median())
    valid = (piv["base"] - piv["perfect"]) > 0.05 * med
    piv = piv[valid]
    piv["regret_base"] = piv["base"] - piv["perfect"]
    for m in ["modelU", "modelW", "modelW2", "modelE"]:
        piv[f"regret_{m}"] = piv[m] - piv["perfect"]
    n_sites = piv.index.get_level_values(0).nunique()

    ps = piv.reset_index().groupby("site_id")[
        ["regret_base", "regret_modelU", "regret_modelW",
         "regret_modelW2", "regret_modelE"]].mean()
    base_mean = ps["regret_base"].mean()

    rng = np.random.default_rng(0)
    B = 2000

    def ci(d):
        boots = [d.loc[rng.choice(ps.index, len(ps), replace=True)].mean()
                 for _ in range(B)]
        return np.percentile(boots, [2.5, 97.5])

    lines = [f"# A6 site-cluster bootstrap (B={B}, resample {n_sites} sites; "
             f"same {len(piv)} valid issue times as Table 1)\n",
             "## Pairwise differences (X vs uniform, % of base regret; "
             "positive = X worse)\n",
             "| comparison | mean diff | 95% CI (site-cluster) | sig? |",
             "|---|---|---|---|"]
    for b in ["modelW", "modelW2", "modelE"]:
        d = 100 * (ps[f"regret_{b}"] - ps["regret_modelU"]) / base_mean
        lo, hi = ci(d)
        sig = "yes" if (lo > 0 or hi < 0) else "n.s."
        lines.append(f"| {b} vs uniform | {d.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {sig} |")

    lines.append("\n## Cost reduction vs operational forecast (equal site weighting)\n")
    lines.append("| model | reduction [%] | 95% CI (site-cluster) |")
    lines.append("|---|---|---|")
    for m in ["modelU", "modelW", "modelW2", "modelE"]:
        d = 100 * (ps["regret_base"] - ps[f"regret_{m}"]) / base_mean
        lo, hi = ci(d)
        lines.append(f"| {m} | {d.mean():+.1f} | [{lo:+.1f}, {hi:+.1f}] |")

    txt = "\n".join(lines)
    (RES / "bootstrap_cluster_report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
