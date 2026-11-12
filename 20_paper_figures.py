"""
Publication figures for the Applied Energy manuscript (figures 1--6).

Nature/NPG styling: restrained sequential palette, Arial/Helvetica, 8--9.5 pt,
vector PDF plus 300 dpi PNG previews. All panels share one visual recipe.

Outputs -> work/figures/fig[1-6]_*.pdf and .png
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

from foresurgery.attribution import cw_stats, prepare_attribution
from foresurgery.config import BAND_LABELS, BAND_LIST, FIGURES_DIR, RESULTS_DIR
from foresurgery.plotting import (BAND_COLORS, C_LOAD, C_PV, HEAT_CMAP, NPG,
                                  panel, save, set_paper_style, style_axis)

set_paper_style()
RES, FIGS = RESULTS_DIR, FIGURES_DIR
BANDS = BAND_LIST
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- fig 1
def fig1_schematic():
    fig = plt.figure(figsize=(7.6, 3.3))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc, tc=None, ec=None, fs=8.2, lw=1.1):
        if tc is None:
            tc = NPG["ink"] if fc in (NPG["light"], "white") else "white"
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
            fc=fc, ec=ec or fc, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs, linespacing=1.45, zorder=3)

    def arrow(x0, y0, x1, y1, lw=1.3):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
            color=NPG["muted"], lw=lw, shrinkA=0, shrinkB=1.5, zorder=1))

    # Dark nodes mark the two experimental manipulations; process nodes are
    # deliberately pale so the eye follows the intervention, not the boxes.
    box(0.005, 0.48, 0.185, 0.42,
        "Issued forecast\n$\\hat{F}_t$ (96 steps)\nreal rolling archive",
        NPG["blue"], fs=8.0)
    box(0.265, 0.695, 0.175, 0.205,
        "Heal band $h$\nreplace by actual", NPG["teal"], fs=8.0)
    box(0.265, 0.480, 0.175, 0.205,
        "Degrade band $h$\nscale error by $k$", NPG["red"], fs=8.0)
    box(0.515, 0.48, 0.145, 0.42,
        "LP scheduler\nTOU + demand\ncharge, battery", NPG["light"], fs=8.0)
    box(0.700, 0.48, 0.135, 0.42,
        "Open-loop\nexecution on\nactuals", NPG["light"], fs=8.0)
    box(0.870, 0.48, 0.115, 0.42,
        "Attribution\n$V_h$ + CI", "white", tc=NPG["blue"],
        ec=NPG["blue"], lw=1.6, fs=8.0)

    arrow(0.190, 0.690, 0.265, 0.7975)
    arrow(0.190, 0.690, 0.265, 0.5825)
    arrow(0.440, 0.7975, 0.515, 0.690)
    arrow(0.440, 0.5825, 0.515, 0.690)
    arrow(0.660, 0.690, 0.700, 0.690)
    arrow(0.835, 0.690, 0.870, 0.690)

    # Horizon strip: segment widths are proportional to band duration, so the
    # 12-24 h band occupies half of the issued horizon.
    x0, x1, y0, sh = 0.035, 0.965, 0.185, 0.080
    ax.text(0.5, 0.345,
            "Issued 24-h horizon (96 $\\times$ 15-min steps): four surgery bands",
            ha="center", va="center", fontsize=8.5, color=NPG["muted"])
    xx = x0
    for j, (lab, dur, col) in enumerate(zip(BAND_LABELS, [2, 4, 6, 12], BAND_COLORS)):
        w = (x1 - x0) * dur / 24.0
        ax.add_patch(plt.Rectangle((xx, y0), w, sh, fc=col, ec="white",
                                   lw=1.4, zorder=2))
        ax.text(xx + w / 2, y0 + sh / 2, lab, ha="center", va="center",
                fontsize=8, color="white" if j >= 2 else NPG["ink"], zorder=3)
        xx += w
    ax.text(0.5, 0.075,
            "Surgery is applied to one band at a time; all remaining bands are kept as issued.",
            ha="center", va="center", fontsize=8.2, color=NPG["muted"])
    save(fig, "fig1_schematic")


# ---------------------------------------------------------------- fig 2
def fig2_attribution():
    df = pd.read_parquet(RES / "surgery_full.parquet")
    heal, _ = prepare_attribution(df)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05), sharey=True)
    for ax, axis, col, tag, title in zip(
            axes, ["load", "pv"], [C_LOAD, C_PV],
            ["(a)", "(b)"], ["Load forecast healed", "PV forecast healed"]):
        means, los, his = [], [], []
        for band in BANDS:
            s, lo, hi = cw_stats(heal[(heal.axis == axis) & (heal.band == band)])
            means.append(100 * s); los.append(100 * lo); his.append(100 * hi)
        means = np.asarray(means); los = np.asarray(los); his = np.asarray(his)
        x = np.arange(4)
        ax.bar(x, means, color=col, width=0.58, edgecolor="white",
               linewidth=0.8, zorder=3)
        ax.errorbar(x, means, yerr=[means - los, his - means], fmt="none",
                    ecolor=NPG["ink"], elinewidth=1.0, capsize=3.2,
                    capthick=1.0, zorder=4)
        ax.set_xticks(x, BAND_LABELS)
        ax.axhline(0, color=NPG["ink"], lw=0.8, zorder=2)
        ax.set_xlabel("Forecast horizon band")
        ax.set_ylim(0, max(np.nanmax(his) * 1.20, 27))
        style_axis(ax)
        panel(ax, tag, title)
        for xi, m, hi in zip(x, means, his):
            ax.text(xi, hi + 0.55, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=NPG["ink"])
    # Shared y scale: keep the grid but drop the unlabelled duplicate spine.
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(left=False, which="both")
    axes[0].set_ylabel("Regret recovered (% of baseline regret)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_attribution")


# ---------------------------------------------------------------- fig 3
def fig3_dose():
    srcs = {0.0: "surgery_heal60.parquet", 1.0: "surgery_heal60.parquet",
            2.0: "surgery_deg2.parquet", 4.0: "surgery_deg4.parquet"}
    cache = {}
    for a, f in srcs.items():
        d = pd.read_parquet(RES / f)
        lvl = d[d.scenario == "base"].groupby(["site_id", "i0"]).cost.first()
        per = d[d.scenario == "perfect"].groupby(["site_id", "i0"]).cost.first()
        kd = (lvl - per).rename("denom").reset_index()
        med = float(d[d.scenario == "base"].cost.median())
        kd = kd[kd.denom > 0.05 * med]
        tag = {0.0: "heal", 1.0: "base", 2.0: "deg2", 4.0: "deg4"}[a]
        for axis, p in (("load", "L"), ("pv", "P")):
            for band in BANDS:
                sc = ("base" if tag == "base" else f"{tag}{p}_{band}")
                sub = d[d.scenario == sc].merge(kd, on=["site_id", "i0"])
                if a == 0.0:
                    val = (sub.denom - sub.regret).sum() / sub.denom.sum() * 100
                else:
                    val = sub.regret.sum() / sub.denom.sum() * 100
                cache[(axis, band, a)] = val

    doses = [0.0, 1.0, 2.0, 4.0]
    x = np.arange(len(doses))  # categorical spacing; a is an ordinal label
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05), sharey=True)
    lines = []
    for ax, axis, tag, title in zip(
            axes, ["load", "pv"], ["(a)", "(b)"],
            ["Load forecast", "PV forecast"]):
        for bi, band in enumerate(BANDS):
            ys = [cache[(axis, band, a)] for a in doses]
            ln, = ax.plot(x, ys, "-o", color=BAND_COLORS[bi],
                          label=BAND_LABELS[bi], ms=5.5, lw=1.8,
                          markeredgecolor="white", markeredgewidth=0.8,
                          zorder=3)
            if axis == "load":
                lines.append(ln)
        ax.axhline(0, color=NPG["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(x, ["heal\n($a=0$)", "issued\n($a=1$)",
                          "$\\times$2\n($a=2$)", "$\\times$4\n($a=4$)"])
        ax.set_xlim(-0.25, 3.25)
        ax.set_xlabel("Surgery dose (error multiplier $a$)")
        style_axis(ax)
        panel(ax, tag, title)
    axes[0].set_ylabel("Cost change vs. issued forecast (% of regret)")
    fig.legend(handles=lines, labels=BAND_LABELS, loc="upper center",
               ncol=4, frameon=False, fontsize=8.5, handlelength=2.0,
               columnspacing=1.8, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=2.0)
    save(fig, "fig3_dose")


# ---------------------------------------------------------------- fig 4
def fig4_robust():
    def far_shares(fname):
        d = pd.read_parquet(RES / fname)
        h, _ = prepare_attribution(d)
        out = {}
        for axis in ["load", "pv"]:
            s, _, _ = cw_stats(h[(h.axis == axis) & (h.band == "h12_24")])
            out[axis] = 100 * s
        return out

    cdems = [0.2, 0.5, 1.0, 2.0]
    cdem_files = ["surgery_cdem0.2.parquet", "surgery_full.parquet",
                  "surgery_cdem1.parquet", "surgery_cdem2.parquet"]
    batts = [5, 10, 20, 40]
    batt_files = ["surgery_batt0.05.parquet", "surgery_heal60.parquet",
                  "surgery_batt0.2.parquet", "surgery_batt0.4.parquet"]

    L1 = [far_shares(f)["load"] for f in cdem_files]
    P1 = [far_shares(f)["pv"] for f in cdem_files]
    L2 = [far_shares(f)["load"] for f in batt_files]
    P2 = [far_shares(f)["pv"] for f in batt_files]

    rp = pd.read_parquet(RES / "surgery_replan80.parquet")
    hr, _ = prepare_attribution(rp)
    hf, _ = prepare_attribution(pd.read_parquet(RES / "surgery_heal60.parquet"))
    labels, ols, rps = [], [], []
    for axis in ["load", "pv"]:
        for band in BANDS:
            labels.append(("L" if axis == "load" else "P") + "\n"
                          + BAND_LABELS[BANDS.index(band)])
            s1, _, _ = cw_stats(hf[(hf.axis == axis) & (hf.band == band)])
            s2, _, _ = cw_stats(hr[(hr.axis == axis) & (hr.band == band)])
            ols.append(100 * s1); rps.append(100 * s2)

    jit_labels = ["no\njitter", "draw 1", "draw 2", "draw 3"]
    jit_files = ["surgery_heal60.parquet", "pv_jitter_draw11.parquet",
                 "pv_jitter_draw22.parquet", "pv_jitter_draw33.parquet"]
    jvals = []
    for f in jit_files:
        hj, _ = prepare_attribution(pd.read_parquet(RES / f))
        s, _, _ = cw_stats(hj[(hj.axis == "pv") & (hj.band == "h12_24")])
        jvals.append(100 * s)

    # 2x2 layout: the original single row of four panels compressed the
    # grouped-bar panel (c) to unreadable 5.6-pt tick labels.
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.9))
    a0, a1, a2, a3 = axes.ravel()

    a0.plot(cdems, L1, "-o", color=C_LOAD, ms=5, label="load",
            markeredgecolor="white", markeredgewidth=0.7)
    a0.plot(cdems, P1, "-s", color=C_PV, ms=5, label="PV",
            markeredgecolor="white", markeredgewidth=0.7)
    a0.set_xscale("log")
    a0.set_xticks(cdems, [str(c) for c in cdems])
    a0.xaxis.set_minor_locator(plt.NullLocator())
    a0.set_xlabel(r"Demand charge $c^{\rm dem}$ (EUR/kW/day)")
    a0.legend(frameon=False, loc="best")
    panel(a0, "(a)", "Tariff robustness")

    a1.plot(batts, L2, "-o", color=C_LOAD, ms=5, label="load",
            markeredgecolor="white", markeredgewidth=0.7)
    a1.plot(batts, P2, "-s", color=C_PV, ms=5, label="PV",
            markeredgecolor="white", markeredgewidth=0.7)
    a1.set_xticks(batts, [str(b) for b in batts])
    a1.set_xlabel("Battery size (% of daily energy)")
    a1.legend(frameon=False, loc="best")
    panel(a1, "(b)", "Battery-size scaling")

    xb = np.arange(len(labels))
    a2.bar(xb - 0.2, ols, width=0.38, color="#AAB8CE",
           edgecolor="white", linewidth=0.6, label="open-loop", zorder=3)
    a2.bar(xb + 0.2, rps, width=0.38, color=C_PV,
           edgecolor="white", linewidth=0.6, label="hourly replan", zorder=3)
    a2.set_xticks(xb, labels, fontsize=7.2)
    a2.set_xlim(-0.6, len(labels) - 0.4)
    a2.set_ylim(0, 31)
    a2.set_xlabel("Axis and horizon band")
    a2.legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.2)
    panel(a2, "(c)", "Replanning control")

    a3.bar(np.arange(4), jvals, width=0.58,
           color=[C_PV, NPG["cyan"], NPG["cyan"], NPG["cyan"]],
           edgecolor="white", linewidth=0.6, zorder=3)
    a3.axhline(jvals[0], color=C_LOAD, lw=1.1, ls=(0, (4, 3)), zorder=4)
    a3.set_xticks(np.arange(4), jit_labels, fontsize=8)
    a3.set_ylim(0, max(jvals) * 1.28)
    a3.set_xlabel("PV capacity jitter $\\pm$30%")
    for xi, v in enumerate(jvals):
        a3.text(xi, v + 0.9, f"{v:.1f}", ha="center", va="bottom",
                fontsize=8, color=NPG["ink"])
    panel(a3, "(d)", "PV-capacity sensitivity")

    for ax in axes.ravel():
        style_axis(ax)
        ax.set_ylim(bottom=0)
    a2.set_ylim(0, 31)
    for ax in (a0, a2):
        ax.set_ylabel("Far-band share (%)")
    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    save(fig, "fig4_robust")


# ---------------------------------------------------------------- fig 5
def fig5_clusters():
    df = pd.read_parquet(RES / "surgery_full.parquet")
    cl = pd.read_csv(RES / "site_clusters.csv")[["site_id", "cluster"]]
    df = df.merge(cl, on="site_id", how="left")
    heal, _ = prepare_attribution(df)
    names = {0: "day-shift", 1: "weekday batch",
             2: "continuous process", 3: "two-shift / mixed"}
    # A dedicated third axes holds the colour bar; a shared colour bar placed
    # with ax=axes would be drawn between the two heatmaps by tight_layout.
    fig, axes = plt.subplots(
        1, 3, figsize=(7.8, 2.95),
        gridspec_kw={"width_ratios": [1, 1, 0.045]})
    a1, a2, cax = axes
    for ax, axis, tag, title in zip(
            (a1, a2), ["load", "pv"], ["(a)", "(b)"],
            ["Load healed", "PV healed"]):
        rows = []
        for c in sorted(heal.cluster.dropna().unique()):
            for band in BANDS:
                s, _, _ = cw_stats(heal[(heal.axis == axis) & (heal.band == band)
                                        & (heal.cluster == c)])
                rows.append((c, band, 100 * s))
        piv = pd.DataFrame(rows, columns=["c", "b", "v"]).pivot(
            index="c", columns="b", values="v").reindex(columns=BANDS)
        sizes = df.groupby("cluster").site_id.nunique()
        im = ax.imshow(piv.values, cmap=HEAT_CMAP, aspect="auto",
                       vmin=0, vmax=35)
        ax.set_xticks(range(4), BAND_LABELS)
        # Both panels share the same four site types; long row labels are
        # printed once, on the left panel only, to avoid overlapping cells.
        if ax is a1:
            ax.set_yticks(range(len(piv)),
                          [f"{names[c]}  (n={sizes[c]})" for c in piv.index],
                          fontsize=8)
        else:
            ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(piv), 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", bottom=False, left=False, length=0)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 19 else NPG["ink"])
        ax.spines[:].set_visible(False)
        ax.set_title(f"{tag}  {title}", loc="left", fontsize=9.5,
                     fontweight="bold", color=NPG["ink"], pad=9)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 10, 20, 30])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2.5, colors=NPG["ink"])
    cb.set_label("Regret recovered (%)", fontsize=9, labelpad=8)
    fig.subplots_adjust(left=0.215, right=0.895, bottom=0.15, top=0.86,
                        wspace=0.30)
    pos = a2.get_position()
    cax.set_position([pos.x1 + 0.018, pos.y0 + 0.025,
                      0.016, pos.height - 0.05])
    save(fig, "fig5_clusters")


# ---------------------------------------------------------------- fig 6
def fig6_forecasters():
    df = pd.read_parquet(RES / "retrain_eval3.parquet")
    bl = pd.read_parquet(RES / "retrain_eval_baselines.parquet")
    both = pd.concat([df[["site_id", "i0", "scenario", "cost"]],
                      bl[["site_id", "i0", "scenario", "cost"]]])
    piv = both.pivot_table(index=["site_id", "i0"], columns="scenario",
                           values="cost").dropna()
    med = piv["base"].median()
    v = piv[(piv["base"] - piv["perfect"]) > 0.05 * med]
    scen = [("base", "Operational\nforecast"),
            ("ridge", "Ridge\ncorrector"),
            ("modelW2", "Site-weighted\n(exogenous)"),
            ("modelW", "Fleet-weighted\n(exogenous)"),
            ("modelE", "Self-weighted\n(endogenous)"),
            ("modelU", "Uniform\n(GBM)"),
            ("persistence", "Persistence\n(no learning)")]
    means, los, his = [], [], []
    rng = np.random.default_rng(1)
    for name, _ in scen:
        r = (v[name] - v["perfect"]).values
        means.append(r.mean())
        bs = [rng.choice(r, len(r), replace=True).mean() for _ in range(500)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        los.append(lo); his.append(hi)
    means = np.asarray(means); los = np.asarray(los); his = np.asarray(his)
    colors = ["#B7BEC8", NPG["greyblue"], NPG["mint"], NPG["cyan"],
              NPG["teal"], NPG["blue"], NPG["red"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.15))
    x = np.arange(len(scen))
    ax.bar(x, means, color=colors, width=0.64, edgecolor="white",
           linewidth=0.7, zorder=3)
    ax.errorbar(x, means, yerr=[means - los, his - means], fmt="none",
                ecolor=NPG["ink"], elinewidth=1.0, capsize=3.2,
                capthick=1.0, zorder=4)
    ax.set_xticks(x, [s[1] for s in scen], fontsize=8)
    ax.set_ylabel("Closed-loop regret (EUR/day)")
    ax.axhline(means[5], color=NPG["blue"], lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.legend(handles=[Line2D([0], [0], color=NPG["blue"], lw=1.1,
                              ls=(0, (4, 3)))],
              labels=["uniform GBM reference"], frameon=False,
              loc="upper right", fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    save(fig, "fig6_forecasters")
    # print numbers used in the text
    tot = (v["base"] - v["perfect"]).sum()
    for name, _ in scen:
        r = v[name] - v["perfect"]
        print(f"{name:12s} mean={r.mean():7.2f} med={r.median():6.2f} "
              f"red_vs_base={100 * (1 - r.sum() / tot):+5.1f}%")
    mae_all = pd.concat([df, bl])[["site_id", "i0", "scenario", "mae"]]
    mae = mae_all[mae_all.scenario != "perfect"].pivot_table(
        index=["site_id", "i0"], columns="scenario", values="mae")
    common = v.index.intersection(mae.index)
    print("MAE:", mae.loc[common, [s[0] for s in scen if s[0] != "perfect"]]
          .mean().round(2).to_dict())


if __name__ == "__main__":
    fig1_schematic()
    fig2_attribution()
    fig3_dose()
    fig4_robust()
    fig5_clusters()
    fig6_forecasters()
    print("figures ->", FIGS)
