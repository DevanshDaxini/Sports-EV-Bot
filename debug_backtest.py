#!/usr/bin/env python3
"""
Backtest projections vs actuals for past week.
Compare scan projections to actual game results.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'nba', 'processed', 'training_dataset.csv')
PROJECTIONS_DIR = os.path.join(BASE_DIR, 'data', 'nba', 'projections')

# Load full game history
print("Loading game history...")
df_history = pd.read_csv(DATA_FILE)
df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])

# Get past week scans
today = datetime.now()
past_week_files = []
for i in range(1, 8):
    check_date = today - timedelta(days=i)
    filename = f"scan_{check_date.strftime('%Y-%m-%d')}.csv"
    filepath = os.path.join(PROJECTIONS_DIR, filename)
    if os.path.exists(filepath):
        past_week_files.append((check_date, filepath))
        print(f"Found: {filename}")

if not past_week_files:
    print("No projection files found for past week")
    sys.exit(1)

# Analyze each scan
all_misses = []

for scan_date, filepath in sorted(past_week_files):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {scan_date.strftime('%Y-%m-%d')}")
    print(f"{'='*80}")

    df_scan = pd.read_csv(filepath)

    # Scan files use columns: NAME, TARGET, AI, PP
    print(f"Projection file columns: {list(df_scan.columns)}")
    print(f"Total projections: {len(df_scan)}")

    # Get games for the scan date AND next 1-2 days (scan usually made day before/morning-of)
    game_dates = df_history[
        (df_history['GAME_DATE'] >= scan_date) &
        (df_history['GAME_DATE'] <= scan_date + timedelta(days=2))
    ]['GAME_DATE'].unique()

    if len(game_dates) == 0:
        print("No games found on scan date or next 2 days")
        continue

    game_dates = sorted(game_dates)
    print(f"Games found for dates: {[d.strftime('%Y-%m-%d') for d in game_dates]}")

    # Match projections to actual results
    for game_date in game_dates:
        df_game = df_history[df_history['GAME_DATE'] == game_date].copy()
        if df_game.empty:
            continue

        print(f"\n  📅 {game_date.strftime('%Y-%m-%d')} ({len(df_game)} player-games)")

        # Normalize names for matching (handle "FirstName LastName" format)
        df_game['PLAYER_NAME_NORMALIZED'] = df_game['PLAYER_NAME'].str.strip().str.lower()
        df_scan_copy = df_scan.copy()
        df_scan_copy['NAME_NORMALIZED'] = df_scan_copy['NAME'].str.strip().str.lower()

        # Check critical stats: PTS, AST, REB, PRA, RA
        key_stats = ['PTS', 'AST', 'REB', 'PRA', 'RA']

        misses = []
        for _, proj_row in df_scan_copy.iterrows():
            if 'NAME' not in proj_row or 'TARGET' not in proj_row or 'AI' not in proj_row:
                continue

            player_name = str(proj_row['NAME_NORMALIZED']).strip()
            stat = str(proj_row['TARGET']).strip().upper()
            proj_val = float(proj_row['AI']) if pd.notna(proj_row['AI']) else 0

            # Skip if not a key stat
            if stat not in key_stats:
                continue

            # Find actual value
            actual_row = df_game[df_game['PLAYER_NAME_NORMALIZED'] == player_name]
            if actual_row.empty:
                continue

            actual_val = float(actual_row[stat].iloc[0]) if stat in actual_row.columns else None
            if actual_val is None or pd.isna(actual_val):
                continue

            error = actual_val - proj_val
            error_pct = (error / proj_val * 100) if proj_val > 0 else 0

            # Flag significant misses (>1.5 points or >15% error)
            if abs(error) > 1.5 or abs(error_pct) > 15:
                misses.append({
                    'date': game_date.strftime('%Y-%m-%d'),
                    'player': proj_row['NAME'],
                    'stat': stat,
                    'proj': proj_val,
                    'actual': actual_val,
                    'error': error,
                    'error_pct': error_pct
                })

        if misses:
            print(f"    ⚠️  MISSES ({len(misses)}):")
            misses_df = pd.DataFrame(misses)
            misses_df = misses_df.sort_values('error', key=abs, ascending=False).head(10)
            for _, m in misses_df.iterrows():
                direction = "OVER" if m['error'] > 0 else "UNDER"
                print(f"       {m['player']:20s} {m['stat']:5s}: proj={m['proj']:5.1f} actual={m['actual']:5.1f} ({direction} {abs(m['error']):+.1f})")
            all_misses.extend(misses)

print(f"\n{'='*80}")
print(f"SUMMARY - PAST WEEK")
print(f"{'='*80}")
print(f"Total significant misses: {len(all_misses)}")

if all_misses:
    summary_df = pd.DataFrame(all_misses)

    # By stat
    print("\nMisses by stat:")
    stat_summary = summary_df.groupby('stat').agg({
        'error': ['count', 'mean', 'std'],
        'error_pct': 'mean'
    }).round(2)
    print(stat_summary)

    # By direction
    print("\nMisses by direction:")
    summary_df['direction'] = summary_df['error'].apply(lambda x: 'OVER' if x > 0 else 'UNDER')
    dir_summary = summary_df['direction'].value_counts()
    print(dir_summary)

    # Most common misses
    print("\nWorst misses (absolute error):")
    summary_df['abs_error'] = summary_df['error'].abs()
    worst = summary_df.nlargest(15, 'abs_error')
    for _, row in worst.iterrows():
        print(f"  {row['date']} | {row['player']:20s} | {row['stat']:5s} | proj={row['proj']:5.1f} actual={row['actual']:5.1f} | error={row['error']:+5.1f}")
