# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) with the caveat that
a *minor* bump can move a published number.

## [1.1.0] — 2026-11-12

Corrected dispatch model. The demand charge now enters the plan, the sample is
complete, and the numbers in `docs/RESULTS.md` are the ones the revised
manuscript reports.

### Changed

- `FORESURGERY_PEAK_SLOT` now defaults to `own`: the peak epigraph variable
  gets its own LP slot (`5T + 1`) instead of sharing the terminal state of
  charge (`5T`), so `imp_t <= D` is enforced and planned cost equals realised
  cost. `legacy` restores the v1.0 behaviour for bit-exact reproduction.
- `FORESURGERY_LP_RETRY` now defaults to `1`: failed HiGHS solves are retried
  with dual simplex and then interior point, so the pipeline is reproducible
  and no issue times are lost.
- `docs/LIMITATIONS.md` rewritten around the two switches and the measured
  delta between them; `docs/RESULTS.md` carries the new expected numbers.
- New test `test_defaults_use_the_corrected_formulation` fails if either
  default silently reverts.

### Numbers that moved (cost-weighted heal share, 70 sites)

| band | load 1.0 → 1.1 | PV 1.0 → 1.1 |
|---|---|---|
| 0-2 h | 1.3 → 1.2 % | 1.1 → 1.4 % |
| 2-6 h | 2.8 → 3.3 % | 3.9 → 3.6 % |
| 6-12 h | 10.5 → 9.2 % | 13.5 → 11.3 % |
| 12-24 h | 19.4 → 21.0 % | 23.3 → 19.2 % |

Scenario rows 121,992 → 126,000; issue times passing the validity filter
4,446 → 6,496. The band ordering is unchanged and monotone on both axes.

### Findings that changed

- The demand charge is a load-side mechanism: pricing the peak raises the load
  far-band share and lowers the PV one, so the two channels are now close
  (site-mean 7.3 % load versus 10.4 % PV, was 6.5 % versus 13.2 %).
- Peak location is no longer irrelevant: the load far-band share varies from
  14.9 % (peak in the first six hours) to 26.1 % (peak in 12-24 h), instead of
  being flat at 18.8-20.3 %. The far band still dominates in every stratum.

## [1.0.0] — 2026-11-12

First public release, matching the manuscript submission.

### Added

- `src/foresurgery/` — the library behind the study: warehouse construction,
  battery dispatch LP, the forecast-surgery operator, site clustering,
  cost-weighted attribution, the corrector feature matrix and the figure style.
- `scripts/00`–`23` — the numbered pipeline, one script per experiment,
  configurable through environment variables instead of hard-coded paths.
- `tests/test_core.py` — fifteen data-free tests covering the LP, the operator
  and the attribution arithmetic; run in CI on Python 3.10 and 3.12.
- `docs/DATA.md`, `docs/PIPELINE.md`, `docs/RESULTS.md`,
  `docs/LIMITATIONS.md` — data dictionary, experiment map, figure-to-script map
  with expected numbers, and the three known formulation issues.
- `CITATION.cff`, `LICENSE` (MIT for code, ODbL for EMSx-derived tables),
  issue forms, pull-request template, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`.

### Known issues

- `FORESURGERY_PEAK_SLOT=legacy` (the default, and the formulation behind the
  manuscript) puts the peak variable in the same LP slot as the terminal state
  of charge, so the demand charge does not enter the dispatch decision. Set
  `FORESURGERY_PEAK_SLOT=own` for the model as described in the paper. This
  moves numbers; see `docs/LIMITATIONS.md`.
- The initial state of charge is only bounded below in the default
  formulation, which allows free energy at the start of the day.
- HiGHS occasionally reports an always-feasible LP as infeasible, dropping a
  small share of issue times at random. `FORESURGERY_LP_RETRY=1` removes this
  and makes reruns bit-reproducible.

### Notes

- The 11 GB EMSx archive is not redistributed here; see `docs/DATA.md`.
- Runtime for the full 70-site pipeline is roughly one working day on a
  laptop CPU, dominated by `07_design_sweep.py`.
