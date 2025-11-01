# Customer Transactions Data Cleaning and Feature Engineering

# Customer Transactions Data Cleaning and Feature Engineering

This small project demonstrates a single-file data-processing pipeline that loads a CSV, performs simple EDA, fills missing values, caps outliers, creates example features, and scales numeric columns.

Quick start
1. Install dependencies (PowerShell):

```powershell
python -m pip install --upgrade pip; pip install -r requirements.txt
```

2. Run the pipeline against the example fixture (no plotting):

```powershell
python src/Main.py --data-path tests/fixtures/sample.csv --no-plot --output out.csv
```

3. Run tests:

```powershell
python -m pytest
```

Notes
- The main script filename is `src/Main.py` (capitalized). The CLI supports `--data-path`, `--no-plot` and `--output`.
- CI (GitHub Actions) installs dependencies and runs `flake8` + `pytest`. On Windows, installing `pandas` may require Visual Studio build tools for some Python versions; the test suite includes fast mock-based tests so CI can validate logic even in minimal environments.
