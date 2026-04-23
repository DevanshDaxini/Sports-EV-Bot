"""
Walk-Forward Backtester for NBA Player Props

Evaluates the trained XGBoost models against the test portion of the training
dataset using player-specific L20 medians as proxy betting lines (much more
realistic than the global-median metric used during training).

Why this matters:
  - Training reports 74-79% directional accuracy vs test-set global median.
  - That metric is misleading: a model predicting LeBron > 18 PTS doesn't prove
    edge when the line is 26.5 PTS.
  - This backtester compares predictions vs each player's own L20 median at the
    time of the game — the closest proxy to a well-set sportsbook line.

Output:
    output/nba/backtests/backtest_results.csv   (row-level)
    output/nba/backtests/backtest_summary.csv   (per-stat summary)

Usage:
    python3 -m src.sports.nba.backtester
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, BASE_DIR)

DATA_FILE  = os.path.join(BASE_DIR, 'data',   'nba', 'processed', 'training_dataset.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models', 'nba')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'nba', 'backtests')

from src.sports.nba.train import get_features_for_target, ensure_combo_stats, TARGETS, LOG_TRANSFORM_TARGETS

# PrizePicks pays $1 profit per $1.15 risked (roughly -115).
# Break-even win rate = 1.15 / (1 + 1.15) = 53.5%
PP_PAYOUT = 1.0 / 1.15      # $0.87 profit per $1 bet on a win
PP_BREAK_EVEN = 1.15 / 2.15 # ~53.5% needed to break even

# Minimum edge % required to flag a bet as "high-confidence"
HIGH_CONF_EDGE_PCT = 5.0

# Minimum proxy line value to include a row in backtest.
# Rows below this threshold don't represent real sportsbook markets
# (e.g. a player whose L10_Median BLK is 0 would never have a BLK line offered).
MIN_PROXY_LINE = {
    'BLK': 0.5, 'STL': 0.5, 'SB': 0.5, 'FG3M': 0.5, 'TOV': 0.5,
    'FTM': 1.0, 'FTA': 1.0,
}
DEFAULT_MIN_PROXY_LINE = 1.0


def load_test_data():
    df = pd.read_csv(DATA_FILE)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = ensure_combo_stats(df)
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    split_idx = int(len(df) * 0.70)
    test_df   = df.iloc[split_idx:].copy().reset_index(drop=True)
    train_end = df.iloc[:split_idx]['GAME_DATE'].max()
    test_start = test_df['GAME_DATE'].min()
    test_end   = test_df['GAME_DATE'].max()

    print(f"  Training ended:  {train_end.date()}")
    print(f"  Test period:     {test_start.date()} → {test_end.date()}")
    print(f"  Test rows:       {len(test_df):,}")
    return test_df


def backtest_target(target, test_df):
    model_path = os.path.join(MODEL_DIR, f"{target}_model.json")
    if not os.path.exists(model_path):
        return []
    if target not in test_df.columns:
        return []

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # Use the exact features the saved model was trained on (avoids mismatch)
    model_features = model.get_booster().feature_names
    if model_features is None:
        raw_features = get_features_for_target(target)
        model_features = [f for f in raw_features if f in test_df.columns]

    # Keep only features present in our dataset; fill the rest with 0
    available = [f for f in model_features if f in test_df.columns]
    X_test = test_df[available].fillna(0)

    # If some model features are missing from the dataset, pad with zeros
    missing_cols = [f for f in model_features if f not in test_df.columns]
    if missing_cols:
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[model_features]  # restore original order
    y_test    = test_df[target].values
    raw_preds = model.predict(X_test)

    # Inverse log-transform if training used it
    if target in LOG_TRANSFORM_TARGETS:
        predictions = np.expm1(np.clip(raw_preds, 0, 10))
    else:
        predictions = np.clip(raw_preds, 0, None)

    # Proxy line = player's rolling median at prediction time.
    # Prefer L10_Median (always present), fall back to L20, then global median.
    for candidate in [f'{target}_L10_Median', f'{target}_L20_Median', f'{target}_L10']:
        if candidate in test_df.columns and test_df[candidate].notna().mean() > 0.4:
            proxy_lines = test_df[candidate].fillna(test_df[target].median()).values
            break
    else:
        proxy_lines = np.full(len(y_test), float(np.median(y_test)))

    min_line = MIN_PROXY_LINE.get(target, DEFAULT_MIN_PROXY_LINE)
    rows = []
    for i, (pred, actual, line) in enumerate(zip(predictions, y_test, proxy_lines)):
        if pd.isna(actual) or pd.isna(line) or line < min_line:
            continue

        ai_side = 'Over' if pred > line else 'Under'

        if actual > line:
            actual_side = 'Over'
        elif actual < line:
            actual_side = 'Under'
        else:
            actual_side = 'Push'

        if actual_side == 'Push':
            result = 'Push'
        elif ai_side == actual_side:
            result = 'Win'
        else:
            result = 'Loss'

        edge_pct = abs(pred - line) / (line + 0.01) * 100

        rows.append({
            'Target':     target,
            'Date':       test_df['GAME_DATE'].iloc[i].date(),
            'AI_Pred':    round(float(pred), 2),
            'Line':       round(float(line), 2),
            'Actual':     round(float(actual), 2),
            'AI_Side':    ai_side,
            'Result':     result,
            'Edge_Pct':   round(edge_pct, 2),
        })
    return rows


def compute_stats(rows_df):
    settled = rows_df[rows_df['Result'].isin(['Win', 'Loss'])].copy()
    if len(settled) == 0:
        return {}

    wins  = (settled['Result'] == 'Win').sum()
    total = len(settled)
    wr    = wins / total

    roi = (wins * PP_PAYOUT - (total - wins)) / total * 100

    hc    = settled[settled['Edge_Pct'] >= HIGH_CONF_EDGE_PCT]
    hc_wr = (hc['Result'] == 'Win').mean() * 100 if len(hc) > 0 else 0.0
    hc_n  = len(hc)

    return {
        'Win_Rate':    round(wr * 100, 2),
        'Wins':        int(wins),
        'Total':       int(total),
        'ROI_Pct':     round(roi, 2),
        'HC_Win_Rate': round(hc_wr, 2),
        'HC_Bets':     int(hc_n),
    }


def run_backtest():
    print("\n" + "=" * 60)
    print("   NBA WALK-FORWARD BACKTESTER")
    print("   Line Proxy: Player L20 Median")
    print("=" * 60 + "\n")

    test_df = load_test_data()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    for target in TARGETS:
        print(f"  Testing {target}...", end='', flush=True)
        rows = backtest_target(target, test_df)
        all_rows.extend(rows)
        if rows:
            settled = [r for r in rows if r['Result'] in ('Win', 'Loss')]
            wins    = sum(1 for r in settled if r['Result'] == 'Win')
            total   = len(settled)
            wr      = wins / total * 100 if total else 0
            print(f"  {wr:.1f}% ({wins}/{total})")
        else:
            print("  no data")

    results_df = pd.DataFrame(all_rows)
    results_path = os.path.join(OUTPUT_DIR, 'backtest_results.csv')
    results_df.to_csv(results_path, index=False)

    # Per-stat summary
    print("\n" + "=" * 60)
    print(f"   RESULTS  (Break-even = {PP_BREAK_EVEN*100:.1f}%)")
    print("=" * 60)

    header = f"{'Stat':<10} {'WinRate':>8} {'N':>6} {'ROI':>8} {'HC%':>7} {'HC_N':>6}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for target in TARGETS:
        sub = results_df[results_df['Target'] == target]
        if sub.empty:
            continue
        s = compute_stats(sub)
        if not s:
            continue
        profitable = "✓" if s['Win_Rate'] >= PP_BREAK_EVEN * 100 else " "
        print(
            f"{target:<10} {s['Win_Rate']:>7.1f}%  {s['Total']:>5,}  "
            f"{s['ROI_Pct']:>+7.1f}%  {s['HC_Win_Rate']:>6.1f}%  {s['HC_Bets']:>5,}  {profitable}"
        )
        summary_rows.append({'Stat': target, **s})

    # Overall
    overall = compute_stats(results_df)
    print("-" * len(header))
    print(
        f"{'OVERALL':<10} {overall.get('Win_Rate', 0):>7.1f}%  "
        f"{overall.get('Total', 0):>5,}  {overall.get('ROI_Pct', 0):>+7.1f}%  "
        f"{overall.get('HC_Win_Rate', 0):>6.1f}%  {overall.get('HC_Bets', 0):>5,}"
    )

    hc_all = results_df[
        results_df['Edge_Pct'] >= HIGH_CONF_EDGE_PCT
    ]
    hc_stats = compute_stats(hc_all)

    print(f"\n  Break-even line: {PP_BREAK_EVEN*100:.1f}%  (PrizePicks -115)")
    print(f"  High-confidence filter (>{HIGH_CONF_EDGE_PCT}% edge): "
          f"{hc_stats.get('Win_Rate', 0):.1f}% "
          f"({hc_stats.get('Wins', 0)}/{hc_stats.get('Total', 0)} bets)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'backtest_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  Row-level results:  {results_path}")
    print(f"  Summary:            {summary_path}")
    return results_df, summary_df


if __name__ == '__main__':
    run_backtest()
