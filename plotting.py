"""Figure style used in the manuscript.

The palette is ggsci's ``npg`` (Nature Publishing Group) family plus a few
neutral tints, with vector PDF output and 300 dpi PNG previews.  Call
:func:`set_paper_style` once at the top of a figure script; every helper below
then shares one visual recipe.
"""
from __future__ import annotations

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["NPG", "BAND_COLORS", "HEAT_CMAP", "C_LOAD", "C_PV", "set_paper_style",
           "style_axis", "panel", "save"]

NPG = {
    "red": "#E64B35", "blue": "#3C5488", "teal": "#00A087", "cyan": "#4DBBD5",
    "amber": "#F39C12", "greyblue": "#8491B4", "mint": "#91D1C2",
    "sand": "#B09C85", "light": "#EEF2F6", "ink": "#242C36",
    "muted": "#5C6672", "grid": "#DDE3E8",
}

#: semantic colours: load is red, PV is blue (the two forecast channels)
C_LOAD, C_PV = NPG["red"], NPG["blue"]

#: sequential horizon ramp, near term pale -> day-ahead dark
BAND_COLORS = ["#6BBFAE", "#4DBBD5", "#3C7BA6", "#2F4576"]

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "npg_heat", ["#F4F7F9", "#D3E7E2", "#8CCBC0", "#00A087", "#3C5488"])


def set_paper_style() -> None:
    """Apply the manuscript's rc settings (idempotent)."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.linewidth": 0.8, "axes.edgecolor": NPG["ink"],
        "lines.linewidth": 1.6, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def style_axis(ax, grid: bool = True) -> None:
    """Light horizontal grid, outward ticks, ink-coloured left/bottom spines."""
    if grid:
        ax.grid(axis="y", lw=0.6, color=NPG["grid"], zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=0.8, colors=NPG["ink"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(NPG["ink"])


def panel(ax, tag: str, title: str) -> None:
    """Left-aligned bold panel label, Nature-style."""
    ax.set_title(f"{tag}  {title}", loc="left", fontsize=9.5,
                 fontweight="bold", color=NPG["ink"], pad=9)


def save(fig, stem, out_dir=None) -> None:
    """Write ``<stem>.pdf`` and a 300 dpi ``<stem>.png`` next to it."""
    from .config import FIGURES_DIR

    out = FIGURES_DIR if out_dir is None else out_dir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
