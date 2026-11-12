"""
Layer 4: secondary findings on the main surgery results.
  (a) healing-can-hurt: rate & correlates
  (b) PV vs load asymmetry (site-level correlates)
  (c) prescriptive table by site cluster
Outputs: results/findings_report.md
"""

import numpy as np
import pandas as pd

from foresurgery.config import BAND_LIST, RESULTS_DIR

RES = RESULTS_DIR
BANDS = BAND_LIST


def main():
    df = pd.read_parquet(RES / "surgery_full.parquet")
    cl = pd.read_csv(RES / "site_clusters.csv")[["site_id", "cluster"]]
    df = df.merge(cl, on="site_id", how="left")

    base = df[df.scenario == "base"].set_index(["site_id", "i0"]).cost
    perf = df[df.scenario == "perfect"].set_index(["site_id", "i0"]).cost
    key_df = (base - perf).rename("denom").reset_index()
    med = float(base.median())
    key_df["valid"] = key_df.denom > 0.05 * med
    heal = df[df.scenario.str.startswith("heal")].copy()
    heal = heal.merge(key_df, on=["site_id", "i0"])
    heal["axis"] = np.where(heal.scenario.str.startswith("healL"), "load", "pv")
    heal["band"] = heal.scenario.str.split("_", n=1).str[1]
    heal["recovered"] = heal.denom - heal.regret
    v = heal[heal.valid].copy()

    lines = ["# 第四层：副发现分析\n"]

    # ---------- (a) healing can hurt ----------
    hurt = v[v.recovered < -0.05 * v.denom.abs()]
    lines.append("## (a) 治愈致害（信息价值非单调）\n")
    lines.append(f"- 有效时点-场景对 {len(v)} 个中，治愈后成本恶化 >5% 悔恨的占 "
                 f"{100 * len(hurt) / len(v):.1f}%（{len(hurt)} 个）\n")
    rate = (v.assign(hurt=v.recovered < -0.05 * v.denom.abs())
             .groupby(["axis", "band"]).hurt.mean().unstack() * 100).round(1)
    lines.append("按轴×带的致害率 [%]：\n")
    lines.append(rate[BANDS].to_string())
    rate_c = (v.assign(hurt=v.recovered < -0.05 * v.denom.abs())
              .groupby("cluster").hurt.mean() * 100).round(1)
    lines.append("\n按站点聚类的致害率 [%]：\n")
    lines.append(rate_c.to_string())
    # correlate with PV share at issue time
    meta = (df[(df.scenario == "perfect")]
            .assign(pv_share=lambda d: d.pv_mean / (d.load_mean + 1e-9))
            [["site_id", "i0", "pv_share", "load_mean"]])
    v2 = v.merge(meta, on=["site_id", "i0"], how="left")
    hurt_by_pv = v2.assign(q=pd.qcut(v2.pv_share.fillna(0), 4,
                                     labels=["Q1低", "Q2", "Q3", "Q4高"]))
    tab = (hurt_by_pv.assign(hurt=hurt_by_pv.recovered < -0.05 * hurt_by_pv.denom.abs())
           .groupby("q", observed=True).hurt.mean() * 100).round(1)
    lines.append("\n按 PV 占比四分位的致害率 [%]：\n")
    lines.append(tab.to_string())

    # ---------- (b) PV vs load asymmetry ----------
    lines.append("\n## (b) PV vs 负荷不对称\n")
    site_pv = (v[v.axis == "pv"].groupby("site_id")
               .apply(lambda g: g.recovered.sum() / g.denom.sum(), include_groups=False))
    site_ld = (v[v.axis == "load"].groupby("site_id")
               .apply(lambda g: g.recovered.sum() / g.denom.sum(), include_groups=False))
    share = meta.groupby("site_id").pv_share.mean()
    common = share.index.intersection(site_pv.index)
    r_pv = np.corrcoef(share[common], site_pv[common])[0, 1]
    r_ld = np.corrcoef(share[common], site_ld[common])[0, 1]
    lines.append(f"- PV 价值份额 vs 站点 PV 占比相关: r={r_pv:.2f}")
    lines.append(f"- 负荷价值份额 vs 站点 PV 占比相关: r={r_ld:.2f}")
    lines.append(f"- PV 总价值份额 {100*site_pv.mean():.1f}% vs 负荷 {100*site_ld.mean():.1f}%"
                 f"（站点均值口径）\n")

    # ---------- (c) prescriptive table ----------
    lines.append("## (c) 处方式表格（每类站点应优先改进的预报维度）\n")
    rows = []
    for c in sorted(v.cluster.dropna().unique()):
        best, best_v = None, -1e9
        tot = 0.0
        detail = {}
        for axis in ["load", "pv"]:
            for band in BANDS:
                s = v[(v.axis == axis) & (v.band == band) & (v.cluster == c)]
                val = s.recovered.sum() / s.denom.sum()
                detail[f"{axis[:1]}:{band}"] = 100 * val
                if val > best_v:
                    best, best_v = f"{axis} 预报 {band}", val
                tot += val
        rows.append({"cluster": int(c), "n_sites": int(v[v.cluster == c].site_id.nunique()),
                     "first_action": best, "first_value_%": round(100 * best_v, 1),
                     "second_action": max(detail, key=detail.get),
                     "total_%": round(100 * tot, 1)})
    lines.append(pd.DataFrame(rows).to_string(index=False))
    lines.append("\n（detail: L=load, P=pv；数值为成本加权恢复份额 %）")

    out = RES / "findings_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
