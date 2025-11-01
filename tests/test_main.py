import os
import pytest

# Skip the tests early if pandas isn't available in this environment.
# In CI this repo installs the pinned dependencies (pandas, numpy, ...) so
# the tests will run there. Locally, environments that can't install pandas
# (for example missing build tools on Windows) will skip these tests.
pd = pytest.importorskip("pandas")
import numpy as np
import importlib

# Import the module after ensuring pandas is available
Main = importlib.import_module("src.Main")
handle_missing_values = Main.handle_missing_values
detect_outliers = Main.detect_outliers
load_data = Main.load_data


def test_handle_missing_values_fills_numeric_and_categorical():
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0],
        "cat": ["a", None, "b"],
    })

    out = handle_missing_values(df.copy())

    # No missing values remain
    assert out["num"].isnull().sum() == 0
    assert out["cat"].isnull().sum() == 0

    # Numeric filled with median of [1,3] -> 2.0
    assert out.loc[1, "num"] == 2.0

    # Categorical filled with mode (one of 'a' or 'b'); verify non-null
    assert out.loc[1, "cat"] in {"a", "b"}


def test_detect_outliers_caps_extreme_values():
    df = pd.DataFrame({"val": [1, 2, 3, 4, 1000]})
    out = detect_outliers(df.copy())

    Q1 = df["val"].quantile(0.25)
    Q3 = df["val"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    # The maximum should be capped to upper_bound
    assert out["val"].max() <= upper_bound + 1e-8


def test_load_data_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("this_file_does_not_exist_hopefully.csv")


def test_parse_args_defaults():
    # parse_args does not require pandas; test default values
    import importlib
    pa = importlib.import_module("src.Main").parse_args
    args = pa([])
    assert args.data_path == "your_dataset.csv"
    assert args.no_plot is False
    assert args.output is None


def test_integration_cli_writes_output(tmp_path):
    # Integration test: run main against a small fixture CSV and verify output CSV is written
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")
    out_file = tmp_path / "out.csv"

    # Run the CLI main with --no-plot to avoid GUI during tests
    import importlib
    Main = importlib.import_module("src.Main")
    Main.main(["--data-path", fixture, "--no-plot", "--output", str(out_file)])

    # Output should exist and be a CSV
    assert out_file.exists()
    # Basic sanity: has header and at least one data row
    text = out_file.read_text()
    assert "total_purchase" in text
    assert len([line for line in text.splitlines() if line.strip()]) >= 2
