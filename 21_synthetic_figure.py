#!/usr/bin/env python
"""Figure 7 — synthetic-error control: is the far-band concentration an
artefact of the error *magnitude*, or of the error *structure*?

Compares the attribution measured on the real issued archive with two synthetic
archives built to carry the same per-band error magnitude but different
correlation structure: block-correlated noise (as the operator's forecast
models are) and white noise.

Inputs : work/results/surgery_heal60.parquet,
         work/results/surgery_synth.parquet,        (block, from 08)
         work/results/surgery_synth_white.parquet   (white, from 08 --mode white)
Outputs: work/figures/fig7_synthetic.{pdf,png}

Run:  python scripts/21_synthetic_figure.py
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from foresurgery.attribution import prepare_attribution
from foresurgery.config import BAND_LABELS, BAND_LIST, FIGURES_DIR, RESULTS_DIR
from foresurgery.plotting import NPG, panel, save, set_paper_style, style_axis


def attribution(df: pd.DataFrame) -> dict:
    """Cost-weighted share recovered for every (axis, band)."""
    h, _ = prepare_attribution(df)
    out = {}
    for axis in ("load", "pv"):
        for band in BAND_LIST:
            sub = h[(h.axis == axis) & (h.band == band)]
            out[(axis, band)] = (100 * sub.recovered.sum() / sub.denom.sum()
                                 if len(sub) else np.nan)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", default="surgery_heal60.parquet")
    ap.add_argument("--block", default="surgery_synth.parquet")
    ap.add_argument("--white", default="surgery_synth_white.parquet")
    args = ap.parse_args(argv)

    set_paper_style()
    real = attribution(pd.read_parquet(RESULTS_DIR / args.real))
    block = attribution(pd.read_parquet(RESULTS_DIR / args.block))
    white = attribution(pd.read_parquet(RESULTS_DIR / args.white))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05), sharey=True)
    x, w = np.arange(4), 0.26
    series = [("Issued (real)", NPG["blue"], real),
              ("Synthetic, block-correlated", NPG["teal"], block),
              ("Synthetic, white noise", NPG["red"], white)]

    for ax, (axis, tag, title) in zip(
            axes, [("load", "(a)", "Load forecast healed"),
                   ("pv", "(b)", "PV forecast healed")]):
        for j, (label, color, table) in enumerate(series):
            vals = [table[(axis, b)] for b in BAND_LIST]
            ax.bar(x + (j - 1) * w, vals, w, label=label, color=color,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0, color=NPG["ink"], lw=0.8, zorder=2)
        ax.set_xticks(x, BAND_LABELS)
        ax.set_xlabel("Forecast horizon band")
        style_axis(ax)
        panel(ax, tag, title)

    axes[0].set_ylabel("Share of regret recovered [%]")
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(left=False, which="both")
    handles = [Patch(facecolor=c, label=lbl) for lbl, c, _ in series]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=8.5, handlelength=1.6, columnspacing=2.0,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=2.0)
    save(fig, "fig7_synthetic", FIGURES_DIR)
    print(f"saved -> {FIGURES_DIR / 'fig7_synthetic.png'}")


if __name__ == "__main__":
    main()
