"""Attribution analysis for the replan80 v2 run, paired with heal60."""

import numpy as np
import pandas as pd

from foresurgery.config import BAND_LIST, BANDS, RESULTS_DIR

RES = RESULTS_DIR
BAND_LIST = list(BANDS)


def attribution(df):
    if "denom" not in df.columns:
        b = df[df.scenario == "base"].set_index(["site_id", "i0"]).cost
        p = df[df.scenario == "perfect"].set_index(["site_id", "i0"]).cost
        df = df.copy()
        df["denom"] = [b.get((s, i), np.nan) - p.get((s, i), np.nan)
                       for s, i in zip(df.site_id, df.i0)]
    med = float(df[df.scenario == "base"].cost.median())
    kd = df[df.scenario == "base"]
    kd = kd[kd.denom > 0.05 * med]
    keys = set(zip(kd.site_id, kd.i0))
    heal = df[df.scenario.str.startswith("heal")].copy()
    heal = heal[[tuple(k) in keys for k in zip(heal.site_id, heal.i0)]]
    cp = df[df.scenario == "perfect"].set_index(["site_id", "i0"]).cost
    heal["regret"] = [c - cp.get((s, i), np.nan)
                      for s, i, c in zip(heal.site_id, heal.i0, heal.cost)]
    heal["recovered"] = heal.denom - heal.regret
    heal["axis"] = np.where(heal.scenario.str.startswith("healL"), "load", "pv")
    heal["band"] = heal.scenario.str.split("_", n=1).str[1]
    out = {}
    for axis in ["load", "pv"]:
        for band in BAND_LIST:
            sub = heal[(heal.axis == axis) & (heal.band == band)]
            out[(axis, band)] = (100 * sub.recovered.sum() / sub.denom.sum()
                                 if len(sub) else np.nan)
    return out, len(kd)


rp = pd.read_parquet(RES / "surgery_replan80.parquet")
ol = pd.read_parquet(RES / "surgery_heal60.parquet")

a_rp, n_rp = attribution(rp)
a_ol, n_ol = attribution(ol)

print(f"replan80: {n_rp} valid issue times; open-loop arm on same keys: {n_ol}")
print(f"{'axis':5s} {'band':7s} {'open-loop %':>12s} {'replanned %':>12s}")
for axis in ["load", "pv"]:
    for band in BAND_LIST:
        print(f"{axis:5s} {band:7s} {a_ol[(axis, band)]:12.1f} "
              f"{a_rp[(axis, band)]:12.1f}")

# base-cost context
mb = rp[rp.scenario == "base"].cost.mean()
mp = rp[rp.scenario == "perfect"].cost.mean()
print(f"\nreplanned mean base cost {mb:.1f} EUR/day-equivalent "
      f"(24 hourly demand windows at 4x charge), perfect {mp:.1f}")
