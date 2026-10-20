"""Basic data cleaning pipeline.
- imputes numeric missing values with median
- fills categorical missing with mode
- encodes categorical columns using one-hot
- saves cleaned CSV to data/cleaned.csv

Run: python data_cleaning.py --csv ev_battery.csv
"""
import argparse
import os
import pandas as pd


def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Place your dataset there or pass --csv.")
    return pd.read_csv(path)


def clean_df(df):
    df = df.copy()
    # Drop columns with all NA
    df.dropna(axis=1, how='all', inplace=True)

    # Impute numeric with median
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for c in num_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].median(), inplace=True)

    # Fill categorical with mode
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for c in cat_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else '', inplace=True)

    # One-hot encode small-cardinality categorical features
    for c in cat_cols:
        if df[c].nunique() > 1 and df[c].nunique() <= 20:
            df = pd.get_dummies(df, columns=[c], prefix=c)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='ev_battery.csv', help='Path to raw CSV')
    parser.add_argument('--out', default='data/cleaned.csv', help='Output cleaned CSV')
    args = parser.parse_args()

    df = load_df(args.csv)
    cleaned = clean_df(df)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cleaned.to_csv(args.out, index=False)
    print('Cleaned data saved to', args.out)
