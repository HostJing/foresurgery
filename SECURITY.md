# Security policy

This is a research code repository: it reads local CSV files, solves linear
programs and writes parquet tables. It has no network service, no credentials
and no database, so the realistic attack surface is small — but it is not zero,
because the pipeline parses files that a user supplies.

## Supported versions

| version | supported |
|---|---|
| 1.0.x | yes |
| < 1.0 | no (pre-release snapshots, not distributed) |

Only the latest tagged release on `main` receives fixes.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private reporting:

1. Go to the repository's **Security** tab.
2. Choose **Report a vulnerability** (GitHub Security Advisories).
3. Describe the problem, the version or commit, and — if you can — a minimal
   input file that triggers it.

You will get an acknowledgement within a week. If the report is valid we will
fix it on `main`, credit you in the advisory unless you prefer otherwise, and
tag a patch release. If it is not a security issue we will say so and point you
at the ordinary issue tracker.

## In scope

- Crafted `.csv.gz` input causing code execution, path traversal outside
  `EMSX_RAW_DIR` / `FORESURGERY_WORK`, or unbounded memory growth (a
  decompression bomb in `foresurgery.data.process_site`).
- Pickle or joblib files in `work/models/` being loaded from an untrusted
  source. `scripts/14`–`18` load their own artifacts; do not load artifacts
  from someone else.
- A dependency in `pyproject.toml` pinned to a version with a known CVE that
  actually affects the code paths used here.

## Out of scope

- A number that differs from the manuscript. That is a reproducibility
  question — use the *reproducibility* issue template and read
  `docs/LIMITATIONS.md` first.
- Resource exhaustion from running the full 70-site pipeline on a small
  machine. It is expected to take about a day and to want ~16 GB of RAM.
- Anything in the EMSx dataset itself. Report that upstream
  ([10.5281/zenodo.5510400](https://doi.org/10.5281/zenodo.5510400)).

## Handling data

The repository ships no personal data. Everything it produces is derived from
the public EMSx release, which is already anonymised and redistributed under
ODbL. If you add a dataset of your own, check its licence before committing the
derived tables — `docs/DATA.md` explains which part of the pipeline counts as
"derived".
