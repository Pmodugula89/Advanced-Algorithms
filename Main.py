import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df
def perform_eda(df):
    print("Data Info:\n", df.info())
    print("\nSummary Statistics:\n", df.describe())
    # Visualize distribution of numeric features
    df.hist(bins=20, figsize=(12, 10))
    plt.suptitle("Feature Distributions")
    plt.show()
        # Boxplots for outlier detection
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure()
        df.boxplot(column=col)
        plt.title(f"Boxplot of {col}")
        plt.show()
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Filled missing values in {col} with median: {median_value}")
            else:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_value}")
    return df
def detect_outliers(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap outliers at bounds
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        print(f"Outliers in {col} capped at {lower_bound} and {upper_bound}")
    return df
def feature_engineering(df):
    # Example 1: Ratio of total_purchase to num_transactions
    if {'total_purchase', 'num_transactions'}.issubset(df.columns):
        df['avg_purchase_per_txn'] = df['total_purchase'] / (df['num_transactions'] + 1)
        print("Created feature: avg_purchase_per_txn")
            # Example 2: Days since last transaction
    if 'last_transaction_date' in df.columns:
        df['days_since_last_txn'] = (pd.Timestamp.today() - pd.to_datetime(df['last_transaction_date'])).dt.days
        print("Created feature: days_since_last_txn")
    return df
def scale_features(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Scaled numeric features using StandardScaler")
    return df
def main():
    # Change the path to your dataset location
    filepath = 'your_dataset.csv'
    df = load_data(filepath)
        perform_eda(df)
    df = handle_missing_values(df)
    df = detect_outliers(df)
    df = feature_engineering(df)
    df = scale_features(df)
    
    print("Final Data Sample:\n", df.head())
if __name__ == "__main__":
    main()
