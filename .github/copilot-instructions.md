## Repository overview

This repo is a small Python project for customer transaction data cleaning and feature engineering. The main executable script is `src/Main.py` (capitalized filename). It is a single-process, imperative pipeline that: loads CSV data, performs EDA (plots), fills missing values, caps outliers, creates example features, and scales numeric columns.

Key files to reference when making changes
- `src/Main.py` — the primary script. Functions to look for: `load_data`, `perform_eda`, `handle_missing_values`, `detect_outliers`, `feature_engineering`, `scale_features`.
- `requirements.txt` — pinned dependencies (pandas, numpy, matplotlib, scikit-learn).
- `.github/workflows/*.yml` — CI runs `flake8` and `pytest`; CI expects tests and linting even if tests are not present yet.

Big-picture architecture and intent
- Single-module data-processing pipeline. The code is intended to be run locally against a CSV dataset (default path `'your_dataset.csv'` in `src/Main.py`).
- Designed for interactive EDA (the script opens Matplotlib windows). This means running in headless CI will fail on plotting unless modified or guarded.

Developer workflows (how to run / test / lint)
- Install deps:
  - PowerShell: `python -m pip install --upgrade pip; pip install -r requirements.txt`
  - POSIX / GitHub Actions: `python -m pip install --upgrade pip && pip install -r requirements.txt`
- Run main script locally (update dataset path first): `python src/Main.py`
- Lint locally: `pip install flake8` then `flake8 .` (CI uses `--max-line-length=127` and `--max-complexity=10`).
- Tests: CI runs `pytest`. There are currently no tests; add unit tests under `tests/` to match CI expectations.

Project-specific conventions and important notes for code edits
- Filename casing: `Main.py` is capitalized. New modules should follow snake_case for filenames but maintain existing casing if modifying `Main.py`.
- The code mixes CLI-style printing and plotting. For automated runs (CI, headless servers), avoid showing plots or add a `--no-plot` flag.
- Data path is hard-coded in `main()` — code changes that introduce CLI args (argparse) will be welcomed and make the repo easier to test.

Integration points & CI
- GitHub Actions in `.github/workflows/python-app.yml` install `flake8` and `pytest` and will fail if flake8 finds syntax/undefined-name errors. Keep code importable and avoid top-level GUI plotting in imports.
- `.github/workflows/python-publish.yml` contains release/publish steps; packaging is not currently implemented. If adding publishing, provide `pyproject.toml` or setup scripts.

Quick examples to reference
- To find the data-loading code: open `src/Main.py` and inspect `load_data(filepath)`.
- To add tests: create `tests/test_main.py` and import functions from `src.Main` (example: `from src.Main import handle_missing_values`).

What to avoid / watchouts
- Do not rely on interactive plotting in CI. Wrap plotting with `if __name__ == '__main__'` or add a CLI flag.
- Avoid adding heavy runtime dependencies without updating `requirements.txt` and workflows.

If unsure, prefer small, testable changes
- Break changes into: (1) add unit tests, (2) make code importable (no side effects), (3) run CI. This project’s CI will catch flake8/pytest regressions.

If you want me to iterate: tell me whether you want stricter rules (type hints, tests scaffolded), or a small PR that converts `src/Main.py` into an importable module with a CLI.

---
Last updated: auto-generated for this repo
