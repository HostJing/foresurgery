# Contributing

This repository accompanies a manuscript, so the bar for changing the
*experiments* is higher than for fixing code. Two rules matter more than any
style guide:

1. **Do not change a published number silently.** The default configuration
   reproduces the manuscript exactly. If your change touches `dispatch.py`,
   `surgery.py`, `attribution.py` or the battery/tariff defaults, say in the pull
   request which numbers move and by how much, and update `docs/RESULTS.md`.
2. **Keep the pipeline runnable end to end.** Every script must still work from a
   clean clone with `pip install -e .` and a warehouse in `work/warehouse`.

## Setup

```bash
git clone https://github.com/your-github/foresurgery.git
cd foresurgery
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
python tests/test_core.py                            # data-free, ~5 s
```

The 11 GB raw archive is not in the repository; see `docs/DATA.md`. Most
contributions can be developed against a 2-3 site subset:

```bash
python scripts/01_build_warehouse.py --sites 1-3
python scripts/00_smoke_test.py --site 1
python scripts/03_run_surgery.py --sites 1-3 --max-issues 4 --tag pilot
```

## What is easy to contribute

| area | notes |
|---|---|
| another tariff | `TOU_SEGMENTS` in `src/foresurgery/config.py`, or follow `13_review_batch2.py` for a per-site tariff |
| another dispatch model | add a `solve_*` next to `solve_dispatch` and a scenario branch in `surgery.py`; keep the return signature |
| a new control experiment | a new numbered script, and a row in `docs/PIPELINE.md` |
| docs and typos | always welcome, no discussion needed |
| a different dataset | `foresurgery.data.process_site` is the only function that touches raw files |

## Style

- Python 3.10+, 4-space indent, ~90 column lines, no line longer than 100.
- Type hints on public functions in `src/foresurgery`; docstrings in the
  imperative, describing *why* rather than restating the code.
- `scripts/` are linear: read top to bottom, one experiment per file, no hidden
  global state beyond the paths and constants at the top.
- Figures go through `foresurgery.plotting` so that a rerun looks like the paper.
- New numeric constants belong in `config.py`, not inline.

## Tests

`tests/test_core.py` needs no data and is the minimum bar:

```bash
python tests/test_core.py     # or: make test
```

If you change the LP, add a test that fails on the old behaviour — the existing
tests for the two peak-variable slots are the model to follow.

## Reporting a number that differs from the paper

Open an issue with the *reproducibility* template. Include the warehouse
`_summary.csv`, the exact commands, and the environment variables — the three
known causes (peak slot, initial SOC, LP retries) are documented in
`docs/LIMITATIONS.md`.

## Licence

By contributing you agree that your code is released under the MIT licence and
that any table derived from EMSx is redistributed under ODbL.
