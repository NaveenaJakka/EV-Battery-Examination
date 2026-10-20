"""Data exploration script for EV battery dataset.
Run: python data_exploration.py --csv ev_battery.csv
Saves basic plots to reports/ and prints summary stats.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Place your dataset there or pass --csv.")
    return pd.read_csv(path)


def run_exploration(df, out_dir="reports"):
    os.makedirs(out_dir, exist_ok=True)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:\n", df.head())
    print("\nDescribe:\n", df.describe(include='all').transpose())

    # Missing values
    missing = df.isnull().sum().sort_values(ascending=False)
    print("\nMissing values:\n", missing[missing>0])

    # Histograms for numeric columns
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if num_cols:
        df[num_cols].hist(figsize=(12, 8))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'histograms.png'))
        plt.clf()

    # Correlation heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'))
        plt.clf()

    # Pairplot sample (limit size)
    if len(num_cols) <= 6 and len(df) <= 2000:
        sns.pairplot(df[num_cols].dropna())
        plt.savefig(os.path.join(out_dir, 'pairplot.png'))
        plt.clf()
    else:
        print('Skipping pairplot due to dimensionality or size.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='ev_battery.csv', help='Path to dataset CSV')
    parser.add_argument('--out', default='reports', help='Output folder for plots')
    args = parser.parse_args()

    df = load_df(args.csv)
    run_exploration(df, args.out)
    print('Exploration complete. Plots and stats saved to', args.out)
