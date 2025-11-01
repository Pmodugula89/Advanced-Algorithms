import argparse
from typing import Any

# Heavy data science libraries are imported inside functions so this module
# can be imported in lightweight test environments without pandas/numpy/matplotlib
# or scikit-learn installed. Tests can monkeypatch the pipeline functions.


def load_data(filepath: str) -> Any:
    """Load CSV into a DataFrame.

    Raises FileNotFoundError if the path does not exist so callers/tests can handle it.
    """
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover - defensive
        raise RuntimeError("pandas is required to load data") from e

    try:
        df = pd.read_csv(filepath)
        print(f"Loaded data from {filepath}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{filepath}' not found.")

def perform_eda(df: Any, show_plots: bool = True) -> None:
    """Print basic EDA info and optionally show plots.

    show_plots=False is required for headless/CI runs.
    """
    print("Data Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

    if not show_plots:
        return

    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting optional
        print("Plotting libraries not available; skipping plots")
        return

    # Visualize distribution of numeric features
    df.hist(bins=20, figsize=(12, 10))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.show()

    # Boxplots for outlier detection
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure()
        df.boxplot(column=col)
        plt.title(f"Boxplot of {col}")
        plt.show()

def handle_missing_values(df):
    # Try to use pandas type checks if available; otherwise fall back to
    # a simple heuristic based on Python types so tests can run without pandas.
    try:
        import pandas as pd
        have_pd = True
    except Exception:
        pd = None
        have_pd = False

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            is_numeric = False
            if have_pd:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
            else:
                # Fallback: look for first non-null value and check its Python type
                first_vals = [v for v in list(df[col]) if v is not None]
                if first_vals:
                    is_numeric = isinstance(first_vals[0], (int, float))

            if is_numeric:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Filled missing values in {col} with median: {median_value}")
            else:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_value}")
    return df

def detect_outliers(df: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numeric ops require numpy
        raise RuntimeError("numpy is required for outlier detection")

    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers at bounds
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        print(f"Outliers in {col} capped at {lower_bound:.2f} and {upper_bound:.2f}")
    return df

def feature_engineering(df: Any) -> Any:
    # Example 1: Ratio of total_purchase to num_transactions
    if {'total_purchase', 'num_transactions'}.issubset(df.columns):
        df['avg_purchase_per_txn'] = df['total_purchase'] / (df['num_transactions'] + 1)
        print("Created feature: avg_purchase_per_txn")

    # Example 2: Days since last transaction
    if 'last_transaction_date' in df.columns:
        try:
            import pandas as pd
            df['last_transaction_date'] = pd.to_datetime(df['last_transaction_date'], errors='coerce')
            df['days_since_last_txn'] = (pd.Timestamp.today() - df['last_transaction_date']).dt.days
            print("Created feature: days_since_last_txn")
        except Exception as e:
            print(f"Error processing 'last_transaction_date': {e}")
    return df

def scale_features(df: Any) -> Any:
    try:
        from sklearn.preprocessing import StandardScaler
        import numpy as np
    except Exception:  # pragma: no cover - requires scikit-learn and numpy
        raise RuntimeError("scikit-learn and numpy are required for scaling")

    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Scaled numeric features using StandardScaler")
    return df

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Data cleaning & feature engineering pipeline")
    parser.add_argument("--data-path", dest="data_path", default="your_dataset.csv",
                        help="Path to input CSV file")
    parser.add_argument("--no-plot", dest="no_plot", action="store_true",
                        help="Disable showing plots (useful for CI/headless runs)")
    parser.add_argument("--output", dest="output", default=None,
                        help="Optional path to write processed CSV (will overwrite)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = load_data(args.data_path)

    perform_eda(df, show_plots=not args.no_plot)
    df = handle_missing_values(df)
    df = detect_outliers(df)
    df = feature_engineering(df)
    df = scale_features(df)

    print("Final Data Sample:\n", df.head())

    if args.output:
        try:
            df.to_csv(args.output, index=False)
            print(f"Wrote processed data to {args.output}")
        except Exception as e:
            print(f"Failed to write output to {args.output}: {e}")

if __name__ == "__main__":
    main()
