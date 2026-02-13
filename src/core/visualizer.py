"""
Visualization Tools

Generates analysis plots for model performance and prediction accuracy.

Output:
    output/nba/analysis_plots/individual_model_accuracy.png
    output/nba/analysis_plots/feature_importance.png
    output/nba/analysis_plots/win_rate_trend.png

Usage:
    $ python3 -m src.core.visualizer
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import xgboost as xgb
import os

# Resolve paths relative to the project root (two levels up from this file)
# src/core/visualizer.py -> src/ -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NBA_MODELS_DIR  = os.path.join(BASE_DIR, 'models', 'nba')
NBA_OUTPUT_DIR  = os.path.join(BASE_DIR, 'output', 'nba', 'analysis_plots')
NBA_SCANS_DIR   = os.path.join(BASE_DIR, 'output', 'nba', 'scans')

if not os.path.exists(NBA_OUTPUT_DIR):
    os.makedirs(NBA_OUTPUT_DIR)


def plot_individual_model_accuracy():
    metrics_file = os.path.join(NBA_MODELS_DIR, 'model_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"⚠️  Metrics file not found at '{metrics_file}'. Run train.py first.")
        return

    df = pd.read_csv(metrics_file)
    df = df.sort_values('Directional_Accuracy', ascending=False)

    plt.figure(figsize=(14, 7))
    colors = ['#2ecc71' if x >= 85 else '#f1c40f' if x >= 75 else '#e74c3c' for x in df['Directional_Accuracy']]
    bars = plt.bar(df['Target'], df['Directional_Accuracy'], color=colors)
    plt.axhline(y=54.1, color='black', linestyle='--', linewidth=1.5, label='PP Breakeven (54.1%)')
    plt.title('Individual Model Accuracy (Directional Win %)', fontsize=16, fontweight='bold')
    plt.ylabel('Win Rate %', fontsize=12)
    plt.ylim(40, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1, f'{height}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.legend(loc='lower right')
    plt.tight_layout()

    save_path = os.path.join(NBA_OUTPUT_DIR, 'individual_model_accuracy.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")


def plot_feature_importance():
    model_path = os.path.join(NBA_MODELS_DIR, 'PTS_model.json')
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find {model_path}")
        return

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    importance = model.get_booster().get_score(importance_type='weight')
    importance = pd.Series(importance).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    importance.plot(kind='barh', color='skyblue')
    plt.title('Feature Importance: What is the Model Weighting?')
    plt.xlabel('Weight (F-Score)')
    plt.tight_layout()

    save_path = os.path.join(NBA_OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")


def plot_win_rate():
    history_file = os.path.join(NBA_SCANS_DIR, 'win_rate_history.csv')
    if not os.path.exists(history_file):
        print("⚠️ No history file found yet.")
        return

    df = pd.read_csv(history_file)
    df['Win_Rate'] = pd.to_numeric(df['Win_Rate'].astype(str).str.replace('%', ''), errors='coerce')
    df = df.dropna(subset=['Win_Rate'])

    if df['Win_Rate'].mean() < 1.0:
        df['Win_Rate'] = df['Win_Rate'] * 100

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Win_Rate'], marker='o', linestyle='-', color='green', linewidth=3, label='Model Accuracy')
    plt.axhline(y=54.1, color='red', linestyle='--', linewidth=2, label='PrizePicks Breakeven (54.1%)')
    plt.ylim(40, 60)
    plt.title('NBA Bot Accuracy Tracker', fontsize=16, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(NBA_OUTPUT_DIR, 'win_rate_trend.png')
    plt.savefig(save_path)
    print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    print("📊 Generating Visualizations...")
    plot_individual_model_accuracy()
    plot_feature_importance()
    plot_win_rate()
    print(f"🚀 Done! Check '{NBA_OUTPUT_DIR}' for your plots.")
