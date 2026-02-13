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

# --- IMPORTS: Core shared tools ---
from src.core.odds_providers.prizepicks import PrizePicksClient
from src.core.odds_providers.fanduel    import FanDuelClient
from src.core.analyzers.analyzer        import PropsAnalyzer

# --- IMPORTS: NBA-specific config and mappings ---
from src.sports.nba.config import (
    ODDS_API_KEY, SPORT_MAP, REGIONS, ODDS_FORMAT, STAT_MAP,
    MODEL_QUALITY, ACTIVE_TARGETS
)
from src.sports.nba.mappings import PP_NORMALIZATION_MAP, STAT_MAPPING, VOLATILITY_MAP

# --- IMPORTS: NBA scanner (load_data, load_models, get_games, etc.) ---
import src.sports.nba.scanner as ai_scanner_module
from src.sports.nba.scanner import load_data, load_models, get_games, prepare_features, normalize_name

warnings.filterwarnings('ignore')

# Output directory for scan results
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'output', 'nba', 'scans'
)


# --- HELPER: RUN AI PREDICTIONS ---
def get_ai_predictions():
    print("...Loading AI Models & Data")
    df_history = load_data()
    models     = load_models()

    if df_history is None or not models:
        return pd.DataFrame()

    todays_teams    = get_games(date_offset=0, require_scheduled=True)
    tomorrows_teams = get_games(date_offset=1, require_scheduled=True)
    all_teams       = {**todays_teams, **tomorrows_teams}

    if not all_teams:
        return pd.DataFrame()

    print("...Generating AI Projections")
    ai_results = []

    for team_id, info in all_teams.items():
        team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row    = p_rows.iloc[-1]
            player_name = last_row['PLAYER_NAME']

            input_row = prepare_features(last_row, is_home=info['is_home'])

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
    print("   🚀 SUPER SCANNER (Math + AI Correlation)")
    print("="*50)

    # 1. Fetch market odds
    print("\n--- 1. Fetching Market Odds (FanDuel vs PrizePicks) ---")
    try:
        pp     = PrizePicksClient(stat_map=STAT_MAP)
        pp_df  = pp.fetch_board(league_filter='NBA')
        if not pp_df.empty:
            pp_df['Stat'] = pp_df['Stat'].replace(PP_NORMALIZATION_MAP)

        fd    = FanDuelClient(
            api_key=ODDS_API_KEY, sport_map=SPORT_MAP,
            regions=REGIONS, odds_format=ODDS_FORMAT, stat_map=STAT_MAP
        )
        fd_df = fd.get_all_odds()

        if pp_df.empty or fd_df.empty:
            print("❌ Error: Missing market data. Cannot run correlation.")
            input("Press Enter...")
            return

        analyzer  = PropsAnalyzer(pp_df, fd_df, league='NBA')
        math_bets = analyzer.calculate_edges()

        if math_bets.empty:
            print("❌ No math-based edges found.")
            input("Press Enter...")
            return

        print(f"✅ Found {len(math_bets)} math-based plays.")
        unique_stats = math_bets['Stat'].unique()
        print(f"   ℹ️  Markets found: {', '.join(unique_stats)}")

    except Exception as e:
        print(f"❌ Error in Odds Scanner: {e}")
        return

    # 2. AI Projections
    print("\n--- 2. Generating AI Projections ---")
    try:
        ai_df = get_ai_predictions()
        if ai_df.empty:
            print("❌ Could not generate AI projections.")
            return
        print(f"✅ Generated {len(ai_df)} AI projections.")
    except Exception as e:
        print(f"❌ Error in AI Scanner: {e}")
        return

    # 3. Correlate
    print("\n--- 3. Correlating Results ---")
    math_bets['Stat']      = math_bets['Stat'].map(STAT_MAPPING).fillna(math_bets['Stat'])
    math_bets['CleanName'] = math_bets['Player'].apply(normalize_name)
    ai_df['CleanName']     = ai_df['Player'].apply(normalize_name)

    merged = pd.merge(math_bets, ai_df, on=['CleanName', 'Stat'], how='inner')
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
                'AI_Proj': ai_proj, 'Score': round(combined_score, 1)
            })

    # 4. Display results
    if not correlated_plays:
        print("❌ No correlated plays found.")
    else:
        final_df = pd.DataFrame(correlated_plays)
        final_df = final_df.sort_values(by='Score', ascending=False).head(20)

        print("\n💎 TOP 20 CORRELATED PLAYS (Math + AI Confidence)")
        print(f"{'TIER':<6} | {'PLAYER':<18} | {'STAT':<5} | {'LINE':<5} | {'SIDE':<5} | {'WIN%':<6} | {'AI PROJ':<7} | SCORE")
        print("-" * 85)

        for _, row in final_df.iterrows():
            print(f"{row['Tier']:<6} | {row['Player']:<18} | {row['Stat']:<5} | {row['Line']:<5} | {row['Side']:<5} | {row['Win%']:<5}% | {row['AI_Proj']:<7} | {row['Score']}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, 'correlated_plays.csv')
        final_df.to_csv(path, index=False)
        print(f"\n💾 Saved list to {path}")

    input("\nPress Enter to return to menu...")


# --- TOOL 2: ODDS SCANNER ---
def run_odds_scanner():
    print("")
    print("\n" + "="*40)
    print("   💰 ODDS ARBITRAGE SCANNER")
    print("="*40)

    try:
        print("--- 1. Fetching PrizePicks Lines ---")
        pp    = PrizePicksClient(stat_map=STAT_MAP)
        pp_df = pp.fetch_board(league_filter='NBA')
        if not pp_df.empty:
            pp_df['Stat'] = pp_df['Stat'].replace(PP_NORMALIZATION_MAP)
        print(f"✅ Got {len(pp_df)} PrizePicks props.")

        print("\n--- 2. Fetching FanDuel Odds ---")
        fd    = FanDuelClient(
            api_key=ODDS_API_KEY, sport_map=SPORT_MAP,
            regions=REGIONS, odds_format=ODDS_FORMAT, stat_map=STAT_MAP
        )
        fd_df = fd.get_all_odds()
        print(f"✅ Got {len(fd_df)} FanDuel props.")

        if pp_df.empty or fd_df.empty:
            print("\n⚠️  Stopping: One of the data sources is empty.")
            input("\nPress Enter to return to menu...")
            return

        print("\n--- 3. Analyzing All Lines ---")
        analyzer = PropsAnalyzer(pp_df, fd_df, league='NBA')
        all_bets = analyzer.calculate_edges()

        if not all_bets.empty:
            sorted_bets = all_bets.sort_values(by='Implied_Win_%', ascending=False)
            print("\n🔥 TOP 15 HIGHEST PROBABILITY PLAYS:")
            print(sorted_bets[['Date', 'Player', 'Stat', 'Side', 'Line', 'Implied_Win_%']].head(15).to_string(index=False))

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            for game_date in sorted_bets['Date'].unique():
                day_data = sorted_bets[sorted_bets['Date'] == game_date]
                day_data.to_csv(os.path.join(OUTPUT_DIR, f"scan_{game_date}.csv"), index=False)
        else:
            print("❌ No profitable matches found!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

    input("\nPress Enter to return to menu...")


# --- TOOL 3: AI SCANNER ---
def run_ai_scanner():
    try:
        ai_scanner_module.main()
    except Exception as e:
        print(f"❌ Error running AI Scanner: {e}")
        input("Press Enter...")


# --- MAIN MENU ---
def main_menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("")
        print("\n" + "🏀"*12 + "  SPORTS ANALYTICS HUB  " + "🏀"*12)
        print("-" * 72)
        print("\nSelect a Tool:")
        print("1. 🚀 Super Scanner (Correlated Plays)")
        print("   -> COMBINES the Odds Scanner and AI Scanner.")
        print("   -> Shows plays where BOTH the Math and AI agree.")
        print("\n2. 💰 Odds Scanner (Arbitrage)")
        print("   -> Compares FanDuel vs PrizePicks for math-based edges.")
        print("\n3. 🤖 NBA AI Scanner (Predictive Model)")
        print("   -> Uses your XGBoost models to predict Over/Under.")
        print("\n0. 🚪 Exit")

        choice = input("\nSelect Option: ").strip()
        if choice == '1':   run_correlated_scanner()
        elif choice == '2': run_odds_scanner()
        elif choice == '3': run_ai_scanner()
        elif choice == '0':
            print("\nGoodbye! 👋\n")
            break
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main_menu()
