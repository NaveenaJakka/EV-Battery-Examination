"""Data visualization utilities for EV battery dataset.
Saves plots to `reports/visualizations`.
Usage:
  python data_visualization.py --cleaned data/cleaned.csv --model models/model.pkl
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set(style='whitegrid')


def load_df(cleaned_path=None, raw_path='ev_battery.csv'):
    if cleaned_path and os.path.exists(cleaned_path):
        return pd.read_csv(cleaned_path)
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    raise FileNotFoundError('No cleaned or raw CSV found. Provide --cleaned or ensure ev_battery.csv exists.')


def choose_target(df):
    for candidate in ['failure', 'target', 'is_failed', 'label']:
        if candidate in df.columns:
            return candidate
    return None


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def plot_distributions(df, num_cols, out_dir):
    if not num_cols:
        print('No numeric columns to plot distributions.')
        return
    fig, axes = plt.subplots(nrows=int(np.ceil(len(num_cols)/3)), ncols=3, figsize=(15, 4*int(np.ceil(len(num_cols)/3))))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    save_fig(fig, os.path.join(out_dir, 'distributions.png'))


def plot_box_by_target(df, num_cols, target, out_dir):
    if target is None:
        print('No target column found; skipping boxplots by target.')
        return
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6,4))
        try:
            sns.boxplot(x=target, y=col, data=df, ax=ax)
            ax.set_title(f'{col} by {target}')
            save_fig(fig, os.path.join(out_dir, f'box_{col}_by_{target}.png'))
        except Exception:
            plt.close(fig)


def plot_correlation(df, num_cols, out_dir):
    if len(num_cols) < 2:
        print('Not enough numeric columns for correlation plot.')
        return
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Correlation heatmap')
    save_fig(fig, os.path.join(out_dir, 'correlation_heatmap.png'))
    try:
        cg = sns.clustermap(corr, cmap='vlag', figsize=(10,10), annot=False)
        cg.fig.suptitle('Correlation clustermap')
        cg.savefig(os.path.join(out_dir, 'correlation_clustermap.png'))
        plt.close(cg.fig)
    except Exception:
        print('Clustermap failed; continuing.')


def plot_pairplot_sample(df, num_cols, out_dir, limit=5000):
    sample = df[num_cols].dropna()
    if len(num_cols) <= 6 and len(sample) <= limit:
        fig = sns.pairplot(sample)
        fig.savefig(os.path.join(out_dir, 'pairplot.png'))
        plt.close()
    else:
        print('Skipping pairplot due to dimensionality or size.')


def plot_feature_importances(df, num_cols, model_path, out_dir):
    if not os.path.exists(model_path):
        print('Model not found; skipping feature importances.')
        return
    model = joblib.load(model_path)
    if not hasattr(model, 'feature_importances_'):
        print('Model has no feature_importances_; skipping.')
        return
    importances = model.feature_importances_
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(num_cols):
        print('Model expects', model.n_features_in_, 'features but found', len(num_cols), "— can't align importances reliably.")
        return
    feat_imp = pd.Series(importances, index=num_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.2*len(feat_imp))))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax)
    ax.set_title('Feature importances')
    save_fig(fig, os.path.join(out_dir, 'feature_importances.png'))
    feat_imp.to_csv(os.path.join(out_dir, 'feature_importances.csv'))


def run_all(cleaned_path=None, raw_path='ev_battery.csv', model_path='models/model.pkl', out_root='reports/visualizations'):
    df = load_df(cleaned_path, raw_path)
    os.makedirs(out_root, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target = choose_target(df)

    print('Numeric columns:', num_cols)
    print('Target column:', target)

    plot_distributions(df, num_cols, out_root)
    plot_correlation(df, num_cols, out_root)
    plot_pairplot_sample(df, num_cols, out_root)
    plot_box_by_target(df, num_cols, target, out_root)
    plot_feature_importances(df, num_cols, model_path, out_root)
    print('Visualizations saved to', out_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaned', default='data/cleaned.csv')
    parser.add_argument('--raw', default='ev_battery.csv')
    parser.add_argument('--model', default='models/model.pkl')
    parser.add_argument('--out', default='reports/visualizations')
    args = parser.parse_args()
    run_all(cleaned_path=args.cleaned, raw_path=args.raw, model_path=args.model, out_root=args.out)
