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
import warnings
from datetime import datetime

from src.core.odds_providers.prizepicks import PrizePicksClient
from src.core.odds_providers.fanduel    import FanDuelClient
from src.core.analyzers.analyzer        import PropsAnalyzer
from src.sports.nba.config import (
    ODDS_API_KEY, SPORT_MAP, REGIONS, ODDS_FORMAT, STAT_MAP,
    MODEL_QUALITY, ACTIVE_TARGETS
)
from src.sports.nba.mappings import PP_NORMALIZATION_MAP, STAT_MAPPING, VOLATILITY_MAP
import src.sports.nba.scanner as ai_scanner_module
from src.sports.nba.scanner import (
    load_data, load_models, get_games, prepare_features, normalize_name,
    refresh_injuries, get_player_status
)

warnings.filterwarnings('ignore')

# Project root is 3 levels up from src/cli/nba_cli.py
_BASE      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(_BASE, 'output', 'nba', 'scans')


# --- HELPER: RUN AI PREDICTIONS ---
def get_ai_predictions():
    print("Loading AI models & data...")
    refresh_injuries()  # Fresh injury data for accurate projections
    df_history = load_data()
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

    print("Generating AI projections...")
    ai_results = []

    for team_id, info in all_teams.items():
        team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()

        # Calculate missing usage from injured (OUT) teammates — boosts projections for healthy players
        missing_usage_today = 0.0
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty:
                continue
            last_row = p_rows.iloc[-1]
            if get_player_status(last_row['PLAYER_NAME'], ai_scanner_module.INJURY_DATA) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15:
                    missing_usage_today += usage

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

            for target, model in models.items():
                if target not in ACTIVE_TARGETS:
                    continue
                feats       = model.feature_names_in_
                valid_input = input_row.reindex(columns=feats, fill_value=0)
                proj        = float(model.predict(valid_input)[0])
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
    print("\n--- 2. Generating AI Projections ---")
    try:
        ai_df = get_ai_predictions()
        if ai_df.empty:
            print("Could not generate AI projections.")
            return
        print(f"Generated {len(ai_df)} AI projections.")
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
            tier_emoji = tier_info.get('emoji', '?')

            correlated_plays.append({
                'Tier': tier_emoji, 'Player': row['Player_x'], 'Stat': row['Stat'],
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
        W_RANK=3; W_TIER=4; W_PLAYER=24; W_STAT=5; W_LINE=6
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
                    pad(str(row['Stat']),             W_STAT)           + SEP +
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
        final_df['Tier'] = final_df['Tier'].replace({'?': '–', '~': '–'})

        # ── Main table: overall top 20 ─────────────────────────────────────
        print_table(final_df, "TOP 20 CORRELATED PLAYS  --  Math + AI Confidence", limit=20)

        # ── Bonus sections: best play(s) for every market NOT in the top 20 ─
        top20_stats = set(final_df.head(20)['Stat'].unique())
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
                label = STAT_LABELS.get(stat, stat)
                print_table(stat_df, f"  {label} ({stat})  —  Top 3", limit=3)

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
            print(sorted_bets[display_cols].head(15).to_string(index=False))

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


# --- MAIN MENU ---
def main_menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("\n" + "=" * 55)
        print("   NBA EV BOT")
        print("=" * 55)
        print(f"   {datetime.now().strftime('%A, %B %d, %Y')}")
        print("=" * 55)

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

        print("\n" + "=" * 55)
        print("0. Back")
        print("=" * 55)

        choice = input("\nSelect: ").strip()

        if   choice == '1': run_correlated_scanner()
        elif choice == '2': run_odds_scanner()
        elif choice == '3': run_ai_scanner()
        elif choice == '4': run_builder()
        elif choice == '5': run_feature_engineering()
        elif choice == '6': run_training()
        elif choice == '7': view_metrics()
        elif choice == '8': run_injury_debug()
        elif choice == '0': break
        else:
            print("\nInvalid selection.")
            input("Press Enter to try again...")


if __name__ == "__main__":
    main_menu()