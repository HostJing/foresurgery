# Counterfactual forecast surgery on issued rolling forecasts

[![tests](https://github.com/your-github/foresurgery/actions/workflows/tests.yml/badge.svg)](https://github.com/your-github/foresurgery/actions/workflows/tests.yml)
[![MIT licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/1455518527.svg)](https://doi.org/10.5281/zenodo.24167722)

Code and derived tables for the study *Forecast value concentrates day-ahead:
counterfactual surgery on issued rolling forecasts from 70 industrial
microgrids*.

The experiment takes the load and PV forecasts that Schneider Electric
**actually issued** to 70 industrial microgrids (the [EMSx
dataset](https://doi.org/10.5281/zenodo.5510400)), and, at each issue time,
replaces (“heals”) one horizon band of the issued forecast with the realised
values while leaving every other band untouched. Each surgery is dispatched by
the same battery LP and executed open-loop against the actuals, so the change
in cost is attributable to that band alone. The operator is

```
healed[t] = actual[t] + k * (issued[t] - actual[t])        for t in band
```

with `k = 0` healing, `k = 1` leaving the forecast as issued and `k > 1`
amplifying the issued error, which makes healing and degrading the *same*
intervention at different doses.

## Headline numbers

Over 70 sites, 12,600 issue times and 126,000 dispatched scenarios, of which
6,496 issue times pass the regret-validity filter (regret above 5 % of the
median bill):

| axis | 0-2 h | 2-6 h | 6-12 h | 12-24 h |
|---|---|---|---|---|
| load forecast healed | 1.2 % | 3.3 % | 9.2 % | **21.0 %** |
| PV forecast healed | 1.4 % | 3.6 % | 11.3 % | **19.2 %** |

Cost-weighted share of the baseline regret recovered by healing that band, with
paired bootstrap intervals in `work/results/summary_report.md`. Roughly a fifth
of what a perfect daily forecast would save sits in the last twelve hours of
the horizon, i.e. in the part of the day that is decided once and never revised.

These are version 1.1 numbers, computed with the demand charge actually priced
in the dispatch (`FORESURGERY_PEAK_SLOT=own`, the default). Version 1.0 shipped
a peak variable that shared its LP slot with the terminal state of charge, so
the demand charge never entered the plan; `docs/LIMITATIONS.md` tabulates what
moved and how to regenerate the earlier figures.

## Layout

```
src/foresurgery/          the library
  config.py               directories, tariff, battery, horizon bands
  data.py                 EMSx -> warehouse (sorting, segments, 96-step tensors)
  dispatch.py             battery dispatch LP + open-loop execution
  surgery.py              heal/degrade operator, issue sampling, campaign runner
  clusters.py             load-shape site typing (k-means)
  attribution.py          cost-weighted shares + bootstrap intervals
  corrector.py            feature matrix and rolling-archive splits
  plotting.py             Nature/NPG figure style
scripts/                  numbered pipeline, one script per experiment
tests/                    data-free tests of the LP, the operator, the arithmetic
docs/                     data, pipeline, results and limitations notes
data/emsx/                put the raw EMSx CSV files here (not tracked)
work/                     generated: warehouse/, results/, models/, figures/
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                                       # or: pip install -r requirements.txt

# 1. put the raw EMSx release in data/emsx/ : 1.csv.gz ... 70.csv.gz
#    (see docs/DATA.md)
python scripts/01_build_warehouse.py                   # ~10 min, 11 GB in -> 1.8 GB parquet

# 2. check that everything works
python scripts/00_smoke_test.py                        # one issue time, all scenarios

# 3. the main experiment
python scripts/03_run_surgery.py --tag full --max-issues 180     # ~2 h on a laptop
python scripts/02_cluster_sites.py --report
python scripts/04_attribution.py
```

Nothing is written outside `work/`. Every run is seeded: the site clustering
(seed 42), the synthetic archives (per-site seeds), the bootstrap resamples
(seed 0) and the gradient-boosting fits (seed 0).

## Pipeline

Scripts are numbered in run order. “Cost” is desktop CPU time for the full
70-site run.

| # | script | inputs | outputs | cost |
|---|---|---|---|---|
| 00 | `00_smoke_test.py` | warehouse | stdout only | s |
| 01 | `01_build_warehouse.py` | `data/emsx/*.csv.gz` | `warehouse/site_*.parquet` | 10 min |
| 02 | `02_cluster_sites.py` | warehouse | `site_clusters.csv`, cluster tables | 10 min |
| 03 | `03_run_surgery.py` | warehouse | `surgery_<tag>.parquet` | 20 min – 2 h |
| 04 | `04_attribution.py` | surgery, clusters | attribution table, figures 2-3, report | 1 min |
| 05 | `05_replan_mpc.py` | warehouse | `surgery_replan80.parquet` | 1 h |
| 06 | `06_replan_report.py` | 03, 05 | open-loop vs hourly replanning table | 1 min |
| 07 | `07_design_sweep.py` | warehouse | `surgery_{deg,cdem,batt}*.parquet` | 4 h |
| 08 | `08_synthetic_control.py` | 03 (`heal60`) | `surgery_synth{,_white}.parquet` | 1 h |
| 09 | `09_pv_jitter.py` | warehouse | `pv_jitter_draw*.parquet` | 1 h |
| 10 | `10_monthly_settlement.py` | warehouse | `monthly_settlement*.parquet` | 30 min |
| 11 | `11_findings.py` | 02, 03 | secondary findings, prescriptive table | 1 min |
| 12 | `12_review_stratification.py` | 03 | peak-location stratification, filter sensitivity | 2 min |
| 13 | `13_review_batch2.py` | warehouse | cycle counts, reserve, real tariff | 2 h |
| 14 | `14_retrain_uniform_fleet.py` | warehouse | `gbm_{uniform,surgery}.joblib` | 40 min |
| 15 | `15_retrain_site_weights.py` | 14 | `gbm_surgery2.joblib` | 40 min |
| 16 | `16_retrain_endogenous.py` | 14, 15 | `gbm_endogenous.joblib`, `retrain_eval3.parquet` | 1.5 h |
| 17 | `17_baseline_forecasters.py` | warehouse | `retrain_eval_baselines.parquet` | 20 min |
| 18 | `18_shadow_price_weights.py` | warehouse | `gbm_shadow.joblib`, `retrain_eval_shadow.parquet` | 40 min |
| 19 | `19_cluster_bootstrap.py` | 16 | site-level bootstrap of the forecaster ranking | 1 min |
| 20 | `20_paper_figures.py` | all of the above | `figures/fig[1-6]_*.{pdf,png}` | 1 min |
| 21 | `21_synthetic_figure.py` | 03, 08 | `figures/fig7_synthetic.{pdf,png}` | s |
| 22 | `22_diagnose_pv_anomaly.py` | warehouse | PV far-band case study | 1 min |
| 23 | `23_formulation_diagnostic.py` | warehouse | attribution under the published vs corrected dispatch model | 10 min |

`docs/RESULTS.md` maps every table and figure of the manuscript onto the
scripts and result files that produce it. `docs/PIPELINE.md` explains what each
experiment is for.

## What each stage is for

1. **Warehouse (01).** Raw rows are not time-sorted and long outages break the
   grid, so each site is sorted, de-duplicated and cut into contiguous
   segments. An issue time is usable only if its next 96 steps fall in one
   segment, which is exactly when a 24-h plan can be executed open-loop.
2. **Attribution (03, 04).** Ten scenarios per issue time — perfect, as issued,
   and one heal per band per channel. Shares are ratios of sums, not means of
   ratios, because near-zero regret makes per-issue ratios explode.
3. **Controls (05-10).** Hourly replanning instead of open-loop execution,
   amplified instead of healed errors, demand-charge and battery sweeps,
   synthetic errors with the same marginal magnitude but different correlation
   structure, PV scaling jitter, monthly instead of daily peak settlement.
4. **Learning (14-18).** Does decision-value weighting of a learned corrector
   beat uniform weighting? Short answer: no — the exogenous and site-level
   weights are falsified, and only the corrector's *own* value structure
   produces a small gain, which is reported as a negative-to-weak result.
5. **Figures (20, 21).** All panels are generated from the parquet tables, so a
   rerun of stage 03 propagates to the figures without manual editing.

## Configuration

Paths are environment variables with sensible defaults (see
`src/foresurgery/config.py`):

| variable | default | meaning |
|---|---|---|
| `EMSX_RAW_DIR` | `data/emsx` | folder with `1.csv.gz` … `70.csv.gz` |
| `FORESURGERY_WORK` | `work` | root for `warehouse/`, `results/`, `models/`, `figures/` |
| `FORESURGERY_RESULTS` | `$FORESURGERY_WORK/results` | where parquet tables and reports go |
| `FORESURGERY_PEAK_SLOT` | `own` | `own` is the model as published; `legacy` reproduces the first draft (see below) |
| `FORESURGERY_LP_RETRY` | `1` | retry failed LPs with another HiGHS algorithm; `0` restores the v1.0 thinning |

## Known limitations

`docs/LIMITATIONS.md` documents one open issue and two closed ones that the
reader must know about before reusing the numbers:

* **Open.** The initial state of charge is only bounded below, so the LP may
  start the day with a fuller battery than the stated 50 % and discharge the
  difference for free. `solve_dispatch(..., fix_initial_soc=True)` pins it, but
  the switch is off in the published runs.
* **Closed in 1.1.** The peak (epigraph) variable used to share its LP slot with
  the terminal state of charge, so in the formulation behind the first draft
  the demand charge did not enter the dispatch decision — the charging plan was
  bit-identical for `c_dem = 0`, `0.5` and `2 EUR/kW/day`. `own` is now the
  default and planned cost equals realised cost. This moved the numbers:
  far-band load 19.4 → 21.0 %, far-band PV 23.3 → 19.2 %.
* **Closed in 1.1.** HiGHS occasionally rejected an always-feasible LP, which
  dropped about 2 % of issue times at random and made reruns irreproducible.
  The retry is now on, the sample is complete, and a rerun reproduces itself.

One consequence of the fix is worth flagging: under the legacy slot the load
far-band share was flat across strata defined by where the daily peak falls
(18.8-20.3 %), and the study read that as evidence that peak *location* does
not matter. Under the corrected model it is not flat (14.9 % when the peak is
near, 26.1 % when it is far), because the plan now actually targets the peak.
The far band still dominates in every stratum, but part of its weight comes
from the peak being decided there.

## Data and licence

* Code: MIT (see `LICENSE`).
* EMSx: Open Database License, DOI [10.5281/zenodo.5510400](https://doi.org/10.5281/zenodo.5510400);
  derived tables in this repository are redistributed under the same licence.
* The PV channel of EMSx is a single production profile rescaled across the
  fleet; `09_pv_jitter.py` tests how much the PV-side conclusions depend on
  that. Load and PV conclusions should be read with this in mind.

## Citation

See `CITATION.cff`. If you use the code, please cite the manuscript and the
EMSx dataset (Le Franc & Ittel), which is what makes the study possible.

## Community

* `CONTRIBUTING.md` — how to run a three-site pilot, what is easy to
  contribute, and the two rules that matter: do not move a published number
  silently, and keep the pipeline runnable end to end.
* `CODE_OF_CONDUCT.md` — Contributor Covenant 2.1.
* `SECURITY.md` — private reporting for input-handling problems.
* `CHANGELOG.md` — release notes, including the open issues in the current
  formulation.
* `AUTHORS.md` — who did what.
* Issues: bug reports, usage questions and reproducibility reports each have
  their own form. Blank issues are disabled so that a report arrives with the
  version, the pipeline stage and the size of the warehouse attached.

## Contact

Jingjie Li — questions about the code, the data pipeline, or a rerun with the
corrected dispatch model.
