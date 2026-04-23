"""
NBA CLI - Main Entry Point for the NBA EV Bot

Provides the interactive menu system connecting all tools:
    1. Super Scanner (Math + AI correlated plays)
    2. Odds Scanner (FanDuel vs PrizePicks arbitrage)
    3. NBA AI Scanner (Standalone AI predictions)

All NBA-specific configuration lives in src/sports/nba/.
Shared tools (FanDuel, PrizePicks, Analyzer) live in src/core/.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

from src.core.odds_providers.prizepicks import PrizePicksClient
from src.core.odds_providers.fanduel    import FanDuelClient
from src.core.analyzers.analyzer        import PropsAnalyzer
from src.sports.nba.config import (
    ODDS_API_KEY, SPORT_MAP, REGIONS, ODDS_FORMAT, STAT_MAP,
    MODEL_QUALITY, ACTIVE_TARGETS, print_mode_info
)
from src.sports.nba.train import LOG_TRANSFORM_TARGETS
from src.sports.nba.scanner import LOG_CALIBRATION
from src.sports.nba.mappings import PP_NORMALIZATION_MAP, STAT_MAPPING, VOLATILITY_MAP
import src.sports.nba.scanner as ai_scanner_module
from src.sports.nba.scanner import (
    load_data, load_models, get_games, prepare_features, normalize_name,
    refresh_injuries, get_player_status, auto_refresh_data
)

warnings.filterwarnings('ignore')

# Project root is 3 levels up from src/cli/nba_cli.py
_BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(_BASE, 'output', 'nba', 'scans')


# --- HELPER: RUN AI PREDICTIONS ---
def get_ai_predictions():
    refresh_injuries()  # Fresh injury data for accurate projections
    df_history = load_data()
    if df_history is not None:
        df_history = auto_refresh_data(df_history)
    models     = load_models()

    if df_history is None or not models:
        return pd.DataFrame()

    # --- Fetch game schedule ---
    # get_games() searches forward when a date has no games, so offset=0 and
    # offset=1 often both resolve to the same future date (e.g. next Thursday).
    # When that happens we skip the second call entirely to avoid printing the
    # same "Found N games on …" block twice.
    first_teams, first_date = get_games(date_offset=0, require_scheduled=True)

    all_teams = dict(first_teams) if first_teams else {}

    # Only make the second call if the first result was actually today
    # (meaning tomorrow might have different games).  If offset=0 already
    # jumped forward, offset=1 will land on the same date — skip it.
    today_str = datetime.now().strftime('%Y-%m-%d')
    if first_date != today_str:
        # first_date is already a future date; offset=1 would find the same slate
        pass
    else:
        second_teams, second_date = get_games(date_offset=1, require_scheduled=True)
        if second_teams and second_date != first_date:
            for team_id, info in second_teams.items():
                if team_id not in all_teams:
                    all_teams[team_id] = info

    if not all_teams:
        return pd.DataFrame()

    ai_results = []

    for team_id, info in all_teams.items():
        team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()

        # Calculate missing usage + OUT count for USAGE_VACUUM fix
        missing_usage_today = 0.0
        team_out_count = 0
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty:
                continue
            last_row = p_rows.iloc[-1]
            if get_player_status(last_row['PLAYER_NAME'], ai_scanner_module.INJURY_DATA) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15:
                    missing_usage_today += usage
                    team_out_count += 1

        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty:
                continue
            last_row    = p_rows.iloc[-1]
            player_name = last_row['PLAYER_NAME']

            # Skip injured (OUT) players — no projection for them
            if get_player_status(player_name, ai_scanner_module.INJURY_DATA) == 'OUT':
                continue

            input_row = prepare_features(
                last_row,
                is_home=info['is_home'],
                missing_usage=missing_usage_today
            )

            # Fix stale USAGE_VACUUM: each OUT teammate reduces STAR_COUNT by 1,
            # so USAGE_VACUUM (= TEAM_AVG_STARS - STAR_COUNT) increases by out count.
            if team_out_count > 0 and 'USAGE_VACUUM' in input_row.columns:
                input_row['USAGE_VACUUM'] = float(input_row['USAGE_VACUUM'].iloc[0]) + team_out_count

            for target, model in models.items():
                if target not in ACTIVE_TARGETS:
                    continue
                feats       = model.feature_names_in_
                valid_input = input_row.reindex(columns=feats, fill_value=0)

                # 1H backfill: rolling stats are NaN/0 when auto-refresh lacks 1H data
                if target in ('PTS_1H', 'PRA_1H', 'FPTS_1H'):
                    for _stat in ['PTS_1H', 'PRA_1H', 'FPTS_1H', 'MIN_1H', 'PRA', 'PTS']:
                        _sc = f'{_stat}_Season'
                        if _sc not in valid_input.columns:
                            continue
                        _sv = valid_input[_sc].iloc[0]
                        if pd.isna(_sv) or _sv <= 0:
                            continue
                        _sv = float(_sv)
                        for _suf in ['_L5', '_L10', '_L20', '_L5_Median', '_L10_Median']:
                            _col = f'{_stat}{_suf}'
                            if _col not in valid_input.columns:
                                continue
                            _cur = valid_input[_col].iloc[0]
                            if pd.isna(_cur) or float(_cur) == 0:
                                valid_input[_col] = _sv

                raw = float(model.predict(valid_input)[0])
                if target in LOG_TRANSFORM_TARGETS:
                    proj = float(np.expm1(max(raw, 0))) * LOG_CALIBRATION.get(target, 1.0)
                else:
                    proj = max(raw, 0.0)
                ai_results.append({'Player': player_name, 'Stat': target, 'AI_Proj': round(proj, 2)})

    return pd.DataFrame(ai_results)


# --- TOOL 1: SUPER SCANNER ---
def run_correlated_scanner():
    print("")
    print("\n" + "="*50)
    print("   SUPER SCANNER (Math + AI Correlation)")
    print("="*50)

    # 1. Fetch market odds
    print("\n--- 1. Fetching Market Odds ---")
    try:
        import time

        # --- PrizePicks: retry up to 3 times (403s are usually temporary rate limits) ---
        pp    = PrizePicksClient(stat_map=STAT_MAP)
        pp_df = pd.DataFrame()
        for attempt in range(1, 4):
            pp_df = pp.fetch_board(league_filter='NBA')
            if not pp_df.empty:
                break
            if attempt < 3:
                print(f"   PrizePicks attempt {attempt}/3 failed. Retrying in 10s...")
                time.sleep(10)

        if pp_df.empty:
            print("PrizePicks unavailable after 3 attempts.")
            input("Press Enter...")
            return

        # --- Preprocessing: 1H Namespacing ---
        # If the row is from NBA1H, prefix the stat so mappings.py can find it as '1H Points' etc.
        def _namespace_stat(row):
            stat = str(row['Stat'])
            league = str(row.get('League', '')).upper()
            if league == 'NBA1H' and not stat.startswith('1H '):
                return f"1H {stat}"
            return stat

        pp_df['Stat'] = pp_df.apply(_namespace_stat, axis=1)
        pp_df['Stat'] = pp_df['Stat'].replace(PP_NORMALIZATION_MAP)

        # --- FanDuel ---
        fd    = FanDuelClient(
            api_key=ODDS_API_KEY, sport_map=SPORT_MAP,
            regions=REGIONS, odds_format=ODDS_FORMAT, stat_map=STAT_MAP
        )
        # Determine the active slate date (handles late-night rollover)
        _, target_date = get_games(date_offset=0, require_scheduled=True)
        print(f"   Active slate: {target_date}")
        fd_df = fd.get_all_odds(target_date=target_date)

        if fd_df.empty:
            print(f"FanDuel odds unavailable for {target_date}.")
            input("Press Enter...")
            return

        analyzer  = PropsAnalyzer(pp_df, fd_df, league='NBA')
        math_bets = analyzer.calculate_edges()

        if math_bets.empty:
            print("No math-based edges found.")
            input("Press Enter...")
            return

        print(f"Found {len(math_bets)} math-based plays.")
        unique_stats = math_bets['Stat'].unique()
        print(f"   Markets: {', '.join(unique_stats)}")

    except Exception as e:
        print(f"Error in Odds Scanner: {e}")
        return

    # 2. AI Projections
    try:
        ai_df = get_ai_predictions()
        if ai_df.empty:
            print("Could not generate AI projections.")
            return
    except Exception as e:
        print(f"Error in AI Scanner: {e}")
        return

    # 3. Correlate
    print("\n--- 3. Correlating Results ---")
    math_bets['Stat']      = math_bets['Stat'].map(STAT_MAPPING).fillna(math_bets['Stat'])
    math_bets['CleanName'] = math_bets['Player'].apply(normalize_name)
    ai_df['CleanName']     = ai_df['Player'].apply(normalize_name)

    merged = pd.merge(math_bets, ai_df, on=['CleanName', 'Stat'], how='inner')
    
    before = len(merged)
    merged = merged.drop_duplicates(
        subset=['CleanName', 'Stat', 'Line', 'Side'],
        keep='first'
    )
    if before > len(merged):
        print(f"   Removed {before - len(merged)} duplicate entries")

    correlated_plays = []

    for _, row in merged.iterrows():
        math_side = row['Side']
        line      = row['Line']
        ai_proj   = row['AI_Proj']
        win_pct   = row['Implied_Win_%']

        ai_diff_raw = abs(ai_proj - line)
        ai_edge_pct = min((ai_diff_raw / line) * 100, 25) if line != 0 else 0

        ai_side = "Over" if ai_proj > line else "Under"
        if math_side == ai_side:
            math_rank    = max(0, min(10, (win_pct - 51) / 5 * 10))
            ai_rank      = max(0, min(10, (ai_edge_pct / 20) * 10))
            stat_weight  = VOLATILITY_MAP.get(row['Stat'], 1.0)
            combined_score = ((math_rank * 0.5) + (ai_rank * 0.5)) * 10 * stat_weight

            tier_info  = MODEL_QUALITY.get(row['Stat'], {})
            tier_text  = tier_info.get('tier', '-')

            correlated_plays.append({
                'Tier': tier_text, 'Player': row['Player_x'], 'Stat': row['Stat'],
                'Line': line, 'Side': math_side, 'Win%': win_pct,
                'AI_Proj': ai_proj, 'Score': round(combined_score, 1),
                'FD_Line': row.get('FD_Line', line),
                'Line_Diff': row.get('Line_Diff', 0.0)
            })

    # 4. Display results
    if not correlated_plays:
        print("No correlated plays found.")
    else:
        import unicodedata

        def vw(s):
            """Visual (terminal) width — wide chars like ⭐ count as 2."""
            return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in str(s))

        def pad(s, width, align='left'):
            """Pad to visual width so emoji-containing cells stay aligned."""
            s = str(s)
            spaces = max(0, width - vw(s))
            return (' ' * spaces + s) if align == 'right' else (s + ' ' * spaces)

        # Column widths
        W_RANK=3; W_TIER=10; W_PLAYER=24; W_STAT=7; W_LINE=6
        W_SIDE=8; W_WIN=7; W_AI=7; W_SCORE=6; W_LDIFF=5
        SEP = " │ "
        total_w = W_RANK+W_TIER+W_PLAYER+W_STAT+W_LINE+W_SIDE+W_WIN+W_AI+W_SCORE+W_LDIFF + len(SEP)*9

        def print_table(df, title, limit=None):
            """Print a formatted table of correlated plays."""
            rows = df.head(limit) if limit else df
            if rows.empty:
                return
            print(f"\n{'─'*total_w}")
            print(f"  {title}")
            print(f"{'─'*total_w}")
            header = (
                pad('#',       W_RANK,  'right') + SEP +
                pad('TIER',    W_TIER)            + SEP +
                pad('PLAYER',  W_PLAYER)           + SEP +
                pad('STAT',    W_STAT)             + SEP +
                pad('LINE',    W_LINE,  'right')   + SEP +
                pad('SIDE',    W_SIDE)             + SEP +
                pad('WIN %',   W_WIN,   'right')   + SEP +
                pad('AI PROJ', W_AI,    'right')   + SEP +
                pad('SCORE',   W_SCORE, 'right')   + SEP +
                pad('ΔLINE',   W_LDIFF, 'right')
            )
            print(header)
            print(f"{'─'*total_w}")
            for i, row in rows.reset_index(drop=True).iterrows():
                tier   = str(row['Tier'])
                player = str(row['Player'])
                while vw(player) > W_PLAYER:
                    player = player[:-1]
                side      = str(row['Side'])
                side_cell = f"{'▲' if side == 'Over' else '▼'} {side}"
                ld = float(row.get('Line_Diff', 0))
                ld_cell = f"{ld:+.1f}" if ld != 0 else "  ="
                if abs(ld) >= 2.0:
                    ld_cell = f"⚡{ld_cell}"
                print(
                    pad(str(i+1),                    W_RANK,  'right') + SEP +
                    pad(tier,                         W_TIER)           + SEP +
                    pad(player,                       W_PLAYER)         + SEP +
                    pad(str(row['Stat']).replace('_1H', ' 1H').replace('FPTS', 'FSCR'), W_STAT) + SEP +
                    pad(f"{float(row['Line']):.1f}",  W_LINE,  'right') + SEP +
                    pad(side_cell,                    W_SIDE)           + SEP +
                    pad(f"{float(row['Win%']):.2f}%", W_WIN,   'right') + SEP +
                    pad(f"{float(row['AI_Proj']):.2f}",W_AI,   'right') + SEP +
                    pad(f"{float(row['Score']):.1f}", W_SCORE, 'right') + SEP +
                    pad(ld_cell,                      W_LDIFF, 'right')
                )
            print(f"{'─'*total_w}")

        # ── Build the full sorted frame ────────────────────────────────────
        final_df = pd.DataFrame(correlated_plays)
        final_df = final_df.sort_values(by='Score', ascending=False)
        final_df = final_df.drop_duplicates(subset=['Player', 'Stat', 'Line', 'Side'], keep='first')

        # ── Main table: overall top 20 ─────────────────────────────────────
        print_table(final_df, "TOP 20 CORRELATED PLAYS  --  Math + AI Confidence", limit=30)

        # ── Bonus sections: best play(s) for every market NOT in the top 20 ─
        top20_stats = set(final_df.head(30)['Stat'].unique())
        all_stats   = set(final_df['Stat'].unique())
        missing_stats = all_stats - top20_stats

        # Friendly display names for stat codes
        STAT_LABELS = {
            'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists',
            'PRA': 'Pts+Rebs+Asts', 'PR': 'Pts+Rebs', 'PA': 'Pts+Asts',
            'RA': 'Rebs+Asts', 'FG3M': '3-Pt Made',
            'BLK': 'Blocks', 'STL': 'Steals', 'SB': 'Blks+Stls',
            'TOV': 'Turnovers', 'FGM': 'FG Made', 'FGA': 'FG Attempted',
            'FTM': 'Free Throws Made', 'FTA': 'Free Throws Attempted',
        }

        if missing_stats:
            print(f"\n  BEST PLAYS BY MARKET  --  markets not in top 20")
            for stat in sorted(missing_stats):
                stat_df = final_df[final_df['Stat'] == stat]
                if stat_df.empty:
                    continue
                label = STAT_LABELS.get(stat, stat).replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                print_table(stat_df, f"  {label} ({stat.replace('_1H', ' 1H').replace('FPTS', 'FSCR')})  —  Top 3", limit=3)

        # ── Save ───────────────────────────────────────────────────────────
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, 'correlated_plays.csv')
        final_df.to_csv(path, index=False)
        print(f"\nSaved to {path}")

    input("\nPress Enter to return to menu...")


# --- TOOL 2: ODDS SCANNER ---
def run_odds_scanner():
    print("")
    print("\n" + "="*40)
    print("   ODDS SCANNER")
    print("="*40)

    try:
        print("--- 1. Fetching PrizePicks Lines ---")
        pp    = PrizePicksClient(stat_map=STAT_MAP)
        pp_df = pp.fetch_board(league_filter='NBA')
        if not pp_df.empty:
            # --- Preprocessing: 1H Namespacing ---
            def _namespace_stat(row):
                stat = str(row['Stat'])
                league = str(row.get('League', '')).upper()
                if league == 'NBA1H' and not stat.startswith('1H '):
                    return f"1H {stat}"
                return stat
            
            pp_df['Stat'] = pp_df.apply(_namespace_stat, axis=1)
            pp_df['Stat'] = pp_df['Stat'].replace(PP_NORMALIZATION_MAP)
        print(f"Got {len(pp_df)} PrizePicks props.")

        print("\n--- 2. Fetching FanDuel Odds ---")
        fd    = FanDuelClient(
            api_key=ODDS_API_KEY, sport_map=SPORT_MAP,
            regions=REGIONS, odds_format=ODDS_FORMAT, stat_map=STAT_MAP
        )
        # Determine the active slate date (handles late-night rollover)
        _, target_date = get_games(date_offset=0, require_scheduled=True)
        print(f"   Active slate: {target_date}")
        fd_df = fd.get_all_odds(target_date=target_date)
        print(f"Got {len(fd_df)} FanDuel props.")

        if pp_df.empty or fd_df.empty:
            print("\nStopping: one of the data sources is empty.")
            input("\nPress Enter to return to menu...")
            return

        print("\n--- 3. Analyzing All Lines ---")
        analyzer = PropsAnalyzer(pp_df, fd_df, league='NBA')
        all_bets = analyzer.calculate_edges()

        if not all_bets.empty:
            sorted_bets = all_bets.sort_values(by='Implied_Win_%', ascending=False)
            print("\nTOP 15 HIGHEST PROBABILITY PLAYS:")
            display_cols = ['Date', 'Player', 'Stat', 'Side', 'Line']
            if 'FD_Line' in sorted_bets.columns:
                display_cols.append('FD_Line')
            if 'Line_Diff' in sorted_bets.columns:
                display_cols.append('Line_Diff')
            display_cols.append('Implied_Win_%')
            print(sorted_bets[display_cols].head(20).to_string(index=False))

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            for game_date in sorted_bets['Date'].unique():
                day_data = sorted_bets[sorted_bets['Date'] == game_date]
                day_data.to_csv(os.path.join(OUTPUT_DIR, f"scan_{game_date}.csv"), index=False)
        else:
            print("No profitable matches found.")

    except Exception as e:
        print(f"\nError: {e}")

    input("\nPress Enter to return to menu...")


# --- TOOL 3: AI SCANNER ---
def run_ai_scanner():
    try:
        ai_scanner_module.main()
    except Exception as e:
        print(f"Error running AI Scanner: {e}")
        input("Press Enter...")


# --- SETUP: BUILD DATA ---
def run_builder():
    print("\n" + "=" * 55)
    print("   BUILD NBA DATA")
    print("=" * 55)
    print("Downloads NBA game logs and player history via nba_api")
    print("")
    confirm = input("This may take 2-5 minutes. Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    try:
        from src.sports.nba.builder import fetch_all_game_logs, fetch_player_positions
        fetch_all_game_logs()
        fetch_player_positions()
        print("\nData build complete!")
        print("   Next: Run 'Engineer Features' then 'Train Models'.")
    except ImportError as e:
        print(f"Builder import error: {e}")
    except Exception as e:
        print(f"\nBuilder error: {e}")
    input("\nPress Enter to continue...")


def run_feature_engineering():
    print("\n" + "=" * 55)
    print("   FEATURE ENGINEERING")
    print("=" * 55)
    print("Building features from raw game logs...")
    print("")
    try:
        from src.sports.nba.features import main as features_main
        features_main()
        print("\nFeatures built!")
        print("   Next: Run 'Train Models'.")
    except ImportError as e:
        print(f"Features import error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()
    input("\nPress Enter to continue...")


def run_training():
    print("\n" + "=" * 55)
    print("   TRAIN NBA MODELS")
    print("=" * 55)
    print(f"Training {len(ACTIVE_TARGETS)} XGBoost models...")
    print("")
    try:
        from src.sports.nba.train import train_and_evaluate
        train_and_evaluate()
    except ImportError as e:
        print(f"Train import error: {e}")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback; traceback.print_exc()
    input("\nPress Enter to continue...")


# --- REPORTING ---
def view_metrics():
    import pandas as pd

    metrics_path = os.path.join(_BASE, 'models', 'nba', 'model_metrics.csv')

    print("\n" + "=" * 55)
    print("   NBA MODEL METRICS")
    print("=" * 55)

    if not os.path.exists(metrics_path):
        print("No metrics found. Run 'Train Models' first.")
        input("\nPress Enter to continue...")
        return

    df = pd.read_csv(metrics_path)

    has_realistic = 'Directional_Accuracy' in df.columns and 'Legacy_Global_Accuracy' in df.columns

    if has_realistic:
        print(f"\n{'TARGET':<10} {'MAE':>6} {'R²':>6} {'RealDir%':>9} {'GlobalDir%':>11}")
        print("-" * 48)
        for _, row in df.iterrows():
            print(f"{row['Target']:<10} {row['MAE']:>6.3f} {row['R2']:>6.3f} "
                  f"{row['Directional_Accuracy']:>8.1f}%  {row.get('Legacy_Global_Accuracy', 0):>9.1f}%")
        print("\nRealDir%   = Directional accuracy vs player L10-median (realistic line proxy)")
        print("GlobalDir% = Legacy accuracy vs global test median (inflated/misleading)")
    else:
        print(f"\n{'TARGET':<8} {'MAE':>6} {'R²':>6} {'DIR%':>7}")
        print("-" * 35)
        for _, row in df.iterrows():
            print(f"{row['Target']:<8} {row['MAE']:>6.3f} {row['R2']:>6.3f} "
                  f"{row['Directional_Accuracy']:>6.1f}%")
        print("\nMAE  = Mean Absolute Error (lower is better)")
        print("DIR% = Directional accuracy — did we predict Over/Under correctly")

    if 'Last_Updated' in df.columns:
        print(f"\nLast trained: {df['Last_Updated'].iloc[-1]}")

    input("\nPress Enter to continue...")


def run_injury_debug():
    """Test injury report lookups."""
    print("\n" + "=" * 55)
    print("   INJURY REPORT")
    print("=" * 55)

    try:
        refresh_injuries()
        from src.sports.nba.injuries import get_injury_report
        report = get_injury_report()
        if not report:
            print("No injuries reported.")
        else:
            out_players = [p for p in report if report[p] == 'OUT']
            gtd_players = [p for p in report if report[p] != 'OUT']
            print(f"\nOUT ({len(out_players)}):")
            for p in sorted(out_players)[:20]:
                print(f"   {p}")
            if len(out_players) > 20:
                print(f"   ... and {len(out_players) - 20} more")
            if gtd_players:
                print(f"\nGTD / Other ({len(gtd_players)}):")
                for p in sorted(gtd_players)[:10]:
                    print(f"   {p}: {report[p]}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()

    input("\nPress Enter to continue...")


def run_backtester():
    print("\n" + "=" * 55)
    print("   WALK-FORWARD BACKTESTER")
    print("=" * 55)
    print("Evaluates model accuracy on the held-out 30% test set")
    print("using player L10-median as the proxy betting line.")
    print("")
    try:
        from src.sports.nba.backtester import run_backtest
        run_backtest()
    except Exception as e:
        print(f"Backtester error: {e}")
        import traceback; traceback.print_exc()
    input("\nPress Enter to continue...")


def search_by_market():
    import glob as _glob

    PROJ_DIR = os.path.join(_BASE, 'data', 'nba', 'projections')

    # Find most recent scan file
    dated = sorted(_glob.glob(os.path.join(PROJ_DIR, 'scan_20*.csv')), reverse=True)
    fallback = os.path.join(PROJ_DIR, 'todays_automated_analysis.csv')
    scan_path = dated[0] if dated else (fallback if os.path.exists(fallback) else None)

    if not scan_path:
        print("\nNo scan data found. Run the AI Scanner first (Option 3).")
        input("\nPress Enter to continue...")
        return

    df = pd.read_csv(scan_path)
    if df.empty or 'TARGET' not in df.columns:
        print("\nScan file is empty or missing expected columns.")
        input("\nPress Enter to continue...")
        return

    # Only keep rows that actually have a PrizePicks line
    df = df[df['PP'].notna() & (df['PP'] > 0)].copy()
    if df.empty:
        print("\nNo lines found in scan file.")
        input("\nPress Enter to continue...")
        return

    # Match the scanner's safe_denominator logic: max(line, 2.0) prevents
    # low lines (e.g. BLK 0.5) from inflating edge% vs what the scanner reports.
    df['PCT_EDGE'] = (df['EDGE'] / df['PP'].clip(lower=2.0)) * 100

    available = sorted(df['TARGET'].unique())
    labels = {
        'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists',
        'FG3M': '3-PT Made', 'FG3A': '3-PT Attempted',
        'BLK': 'Blocks', 'STL': 'Steals', 'TOV': 'Turnovers',
        'FGM': 'FG Made', 'FGA': 'FG Attempted',
        'FTM': 'FT Made', 'FTA': 'FT Attempted',
        'PRA': 'Pts+Rebs+Asts', 'PR': 'Pts+Rebs', 'PA': 'Pts+Asts',
        'RA': 'Rebs+Asts', 'SB': 'Blks+Stls', 'FPTS': 'Fantasy Score',
        'PTS_1H': '1H Points', 'PRA_1H': '1H PRA', 'FPTS_1H': '1H Fantasy Score',
    }

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "=" * 55)
        print("   SEARCH BY MARKET")
        print(f"   Scan: {os.path.basename(scan_path)}")
        print("=" * 55)
        print("\nAvailable markets:")
        for i, t in enumerate(available, 1):
            n = len(df[df['TARGET'] == t])
            print(f"  {i:>2}. {t:<10} {labels.get(t, ''):<25} ({n} players)")

        print("\n  0. Back")
        choice = input("\nEnter market code or number: ").strip().upper()

        if choice == '0' or choice == '':
            break

        # Allow selection by number or by code
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                target = available[idx]
            else:
                print("Invalid number."); input("Press Enter..."); continue
        elif choice in available:
            target = choice
        else:
            print(f"'{choice}' not found. Try one of: {', '.join(available)}")
            input("Press Enter..."); continue

        sub = df[df['TARGET'] == target].copy()
        sub = sub.sort_values('PCT_EDGE', key=abs, ascending=False)

        overs  = sub[sub['EDGE'] > 0].sort_values('PCT_EDGE', ascending=False)
        unders = sub[sub['EDGE'] < 0].sort_values('PCT_EDGE', ascending=True)

        def _print_market_table(rows, side_label):
            if rows.empty:
                return
            print(f"\n{'─' * 100}")
            print(f"   {side_label}  ({len(rows)} plays)")
            print(f"{'─' * 100}")
            print(f"   {'#':>3}  {'PLAYER':<25} {'PROJ':>6} {'LINE':>6} {'EDGE':>8} {'L5':>4} {'L10':>4} {'L20':>4} {'H2H':>8}")
            print(f"{'─' * 100}")
            is_over = side_label.startswith('OVER')
            for i, (_, row) in enumerate(rows.iterrows(), 1):
                l5  = f"{row['L5_HIT']*100:.0f}%"  if is_over else f"{(1-row['L5_HIT'])*100:.0f}%"
                l10 = f"{row['L10_HIT']*100:.0f}%" if is_over else f"{(1-row['L10_HIT'])*100:.0f}%"
                l20 = f"{row['L20_HIT']*100:.0f}%" if is_over else f"{(1-row['L20_HIT'])*100:.0f}%"
                h2h_n = int(row.get('H2H_N', 0))
                if h2h_n > 0:
                    rate = row['H2H_HIT'] if is_over else 1 - row['H2H_HIT']
                    h2h = f"{rate*100:.0f}%({h2h_n})"
                else:
                    h2h = '--'
                print(f"   {i:>3}  {str(row['NAME'])[:25]:<25} {row['AI']:>6.1f} {row['PP']:>6.1f} "
                      f"{row['PCT_EDGE']:>+8.1f}% {l5:>4} {l10:>4} {l20:>4} {h2h:>8}")
            print(f"{'─' * 100}")

        os.system('cls' if os.name == 'nt' else 'clear')
        mkt_label = labels.get(target, target)
        print(f"\n{'═' * 100}")
        print(f"   {mkt_label} ({target})  —  {len(sub)} players on the board")
        print(f"{'═' * 100}")
        _print_market_table(overs,  f"OVERS  — players projected above the line")
        _print_market_table(unders, f"UNDERS — players projected below the line")

        input("\nPress Enter to search another market or go back...")


def run_grade_all():
    print("\n" + "=" * 55)
    print("   GRADE ALL UNGRADED SCAN FILES")
    print("=" * 55)
    print("Fetches actual NBA results and scores every ungraded scan.")
    print("")
    try:
        from src.sports.nba.grader import grade_all_ungraded
        grade_all_ungraded()
    except Exception as e:
        print(f"Grader error: {e}")
        import traceback; traceback.print_exc()
    input("\nPress Enter to continue...")


# --- MAIN MENU ---
def main_menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("\n" + "=" * 55)
        print("   NBA EV BOT")
        print("=" * 55)
        print(f"   {datetime.now().strftime('%A, %B %d, %Y')}")
        print("=" * 55)
        print_mode_info()

        print("\nANALYSIS")
        print("1. Super Scanner         -- Math + AI correlated plays")
        print("2. Odds Scanner          -- FanDuel vs PrizePicks")
        print("3. AI Scanner            -- Scan / Scout / Grade")

        print("\nSETUP  (run once in order)")
        print("4. Build Data            -- Download NBA game history")
        print("5. Engineer Features     -- Build training features")
        print("6. Train Models          -- Train all 13 XGBoost models")

        print("\nREPORTING")
        print("7. Model Metrics         -- Accuracy by market")
        print("8. Injury Report         -- Current injury status")
        print("9. Run Backtester        -- Historical win rate & ROI simulation")
        print("A. Grade All Results     -- Grade every ungraded scan file")
        print("B. Search by Market      -- Browse all plays for a specific stat")

        print("\n" + "=" * 55)
        print("0. Back")
        print("=" * 55)

        choice = input("\nSelect: ").strip().upper()

        if   choice == '1': run_correlated_scanner()
        elif choice == '2': run_odds_scanner()
        elif choice == '3': run_ai_scanner()
        elif choice == '4': run_builder()
        elif choice == '5': run_feature_engineering()
        elif choice == '6': run_training()
        elif choice == '7': view_metrics()
        elif choice == '8': run_injury_debug()
        elif choice == '9': run_backtester()
        elif choice == 'A': run_grade_all()
        elif choice == 'B': search_by_market()
        elif choice == '0': break
        else:
            print("\nInvalid selection.")
            input("Press Enter to try again...")


if __name__ == "__main__":
    main_menu()