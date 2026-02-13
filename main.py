import os
import sys
import pandas as pd
import warnings
from datetime import datetime

# --- SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# --- IMPORT TOOLS ---
try:
    from src.prizepicks import PrizePicksClient
    from src.fanduel import FanDuelClient
    from src.analyzer import PropsAnalyzer
    from src.scanner import load_data, load_models, get_games, prepare_features, normalize_name
except ImportError as e:
    print(f"⚠️ Warning: Could not import core modules: {e}")

ai_scanner_module = None
try:
    import src.scanner as ai_scanner_module
except ImportError:
    pass

warnings.filterwarnings('ignore')

# --- 1. PP NORMALIZATION MAP (PrizePicks -> Standard/FanDuel Name) ---
# This ensures "Blocked Shots" matches "Blocks", etc.
PP_NORMALIZATION_MAP = {
    'Blocked Shots': 'Blocks',
    '3-PT Made': '3-Pt Made',
    'Three Point Field Goals': '3-Pt Made',
    'Free Throws Made': 'Free Throws Made',
    'Turnovers': 'Turnovers',
    'Steals': 'Steals',
    'Fantasy Score': 'Fantasy', 
    
    # CRITICAL: MAPPING THE "MISSING" MARKETS
    'FG Made': 'Field Goals Made',
    'FG Attempted': 'Field Goals Attempted', 
    'Free Throws Attempted': 'Free Throws Attempted',
    'Blks+Stls': 'Blks+Stls',
    'Pts+Rebs+Asts': 'Pts+Rebs+Asts',
    'Pts+Rebs': 'Pts+Rebs',
    'Pts+Asts': 'Pts+Asts',
    'Rebs+Asts': 'Rebs+Asts'
}

# --- 2. STAT MAPPING (Standard Name -> AI Target Code) ---
STAT_MAPPING = {
    # Core
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Assists': 'AST',
    'Pts+Rebs+Asts': 'PRA',
    'Pts+Rebs': 'PR',
    'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA',
    'Blks+Stls': 'SB',
    
    # Alternative Markets
    '3-Pt Made': 'FG3M',
    'Blocks': 'BLK',
    'Steals': 'STL',
    'Turnovers': 'TOV',
    'Free Throws Made': 'FTM',
    'Field Goals Made': 'FGM',
    'Free Throws Attempted': 'FTA',
    'Field Goals Attempted': 'FGA'
}

# --- SCORING CONFIGURATION ---
# Higher weight = More predictable/reliable stat
VOLATILITY_MAP = {
    'PTS': 1.0,  
    'REB': 1.15,  
    'AST': 1.1,
    'PRA': 1.05,  
    'PR': 1.05,
    'PA': 1.05,
    'RA': 1.1,
    'FG3M': 0.90, 
    'BLK': 0.75, 
    'STL': 0.75,
    'TOV': 0.9,
    
    # NEW WEIGHTS FOR EFFICIENCY PROPS
    'SB': 0.8,
    'FGM': 1.0,
    'FGA': 1.0,
    'FTM': 0.9,
    'FTA': 0.9
}

# --- HELPER: RUN AI PREDICTIONS ---
def get_ai_predictions():
    print("...Loading AI Models & Data")
    df_history = load_data()
    models = load_models()
    
    if df_history is None or not models:
        return pd.DataFrame()

    todays_teams = get_games(date_offset=0, require_scheduled=True)
    tomorrows_teams = get_games(date_offset=1, require_scheduled=True)
    all_teams = {**todays_teams, **tomorrows_teams}
    
    if not all_teams:
        return pd.DataFrame()

    print("...Generating AI Projections")
    ai_results = []

    for team_id, info in all_teams.items():
        team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            player_name = last_row['PLAYER_NAME']
            
            input_row = prepare_features(last_row, is_home=info['is_home'])
            
            for target, model in models.items():
                feats = model.feature_names_in_
                valid_input = input_row.reindex(columns=feats, fill_value=0)
                proj = float(model.predict(valid_input)[0])
                
                ai_results.append({
                    'Player': player_name,
                    'Stat': target, 
                    'AI_Proj': round(proj, 2)
                })
    
    return pd.DataFrame(ai_results)

# --- NEW TOOL: CORRELATED SCANNER ---
def run_correlated_scanner():
    print("")
    print("\n" + "="*50)
    print("   🚀 SUPER SCANNER (Math + AI Correlation)")
    print("="*50)
    
    # 1. Run Odds Scanner (Math)
    print("\n--- 1. Fetching Market Odds (FanDuel vs PrizePicks) ---")
    try:
        pp = PrizePicksClient()
        pp_df = pp.fetch_board()
        
        if not pp_df.empty:
            # FIX: Normalize PrizePicks names to match FanDuel
            pp_df['Stat'] = pp_df['Stat'].replace(PP_NORMALIZATION_MAP)

        fd = FanDuelClient()
        fd_df = fd.get_all_odds()
        
        if pp_df.empty or fd_df.empty:
            print("❌ Error: Missing market data. Cannot run correlation.")
            input("Press Enter...")
            return

        analyzer = PropsAnalyzer(pp_df, fd_df)
        math_bets = analyzer.calculate_edges()
        
        if math_bets.empty:
            print("❌ No math-based edges found.")
            input("Press Enter...")
            return
            
        print(f"✅ Found {len(math_bets)} math-based plays.")
        
        # --- DEBUG: SHOW FOUND STATS ---
        unique_stats = math_bets['Stat'].unique()
        print(f"   ℹ️  Markets found in math scan: {', '.join(unique_stats)}")
        # -------------------------------
        
    except Exception as e:
        print(f"❌ Error in Odds Scanner: {e}")
        return

    # 2. Run AI Scanner (Data)
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

    # 3. Correlate Results
    print("\n--- 3. Correlating Results ---")
    
    # Map Math Stats to AI Target Codes (e.g., 'Points' -> 'PTS')
    math_bets['Stat'] = math_bets['Stat'].map(STAT_MAPPING).fillna(math_bets['Stat'])
    
    # Normalize names for merging
    math_bets['CleanName'] = math_bets['Player'].apply(normalize_name)
    ai_df['CleanName'] = ai_df['Player'].apply(normalize_name)
    
    merged = pd.merge(math_bets, ai_df, on=['CleanName', 'Stat'], how='inner')
    
    correlated_plays = []
    
    for _, row in merged.iterrows():
        math_side = row['Side'] 
        line = row['Line']
        ai_proj = row['AI_Proj']
        win_pct = row['Implied_Win_%']
        
        # 1. AI Edge Percentage (Capped at 25% to ignore outliers/errors)
        ai_diff_raw = abs(ai_proj - line)
        ai_edge_pct = min((ai_diff_raw / line) * 100, 25) if line != 0 else 0
        
        # 2. Agreement Check
        ai_side = "Over" if ai_proj > line else "Under"
        if math_side == ai_side:
            
            # --- NORMALIZED SCORING (0-10 Scale for each) ---
            
            # MATH: Scales 51% (Weak) to 56% (Elite) onto a 0-10 scale
            # (win_pct - 51) / (56 - 51) * 10
            math_rank = max(0, min(10, (win_pct - 51) / 5 * 10))
                
            # 2. AI Rank (0-10)
            ai_rank = max(0, min(10, (ai_edge_pct / 20) * 10))
            
            # 3. Apply Volatility Weight
            stat_weight = VOLATILITY_MAP.get(row['Stat'], 1.0)
            
            # 4. Final Balanced & Weighted Score
            # We average the ranks, scale to 100, then apply the stat reliability weight
            combined_score = ((math_rank * 0.5) + (ai_rank * 0.5)) * 10 * stat_weight
            
            correlated_plays.append({
                'Player': row['Player_x'], 
                'Stat': row['Stat'],
                'Line': line,
                'Side': math_side,
                'Win%': win_pct, 
                'AI_Proj': ai_proj,
                'Score': round(combined_score, 1)
            })
            
    # 4. Display Results
    if not correlated_plays:
        print("❌ No correlated plays found.")
    else:
        # Sort by the new Weighted Score instead of just Win%
        final_df = pd.DataFrame(correlated_plays)
        final_df = final_df.sort_values(by='Score', ascending=False).head(20)
        
        print("\n💎 TOP 20 CORRELATED PLAYS (Math + AI Confidence)")
        print(f"{'PLAYER':<18} | {'STAT':<5} | {'LINE':<5} | {'SIDE':<5} | {'WIN%':<6} | {'AI PROJ':<7} | {'SCORE'}")
        print("-" * 75)
        
        for _, row in final_df.iterrows():
            print(f"{row['Player']:<18} | {row['Stat']:<5} | {row['Line']:<5} | {row['Side']:<5} | {row['Win%']:<5}% | {row['AI_Proj']:<7} | {row['Score']}")
            
        # Save
        path = "program_runs/correlated_plays.csv"
        if not os.path.exists("program_runs"): os.makedirs("program_runs")
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
        pp = PrizePicksClient()
        pp_df = pp.fetch_board()
        
        # --- FIX: NORMALIZE NAMES HERE TOO ---
        if not pp_df.empty:
             pp_df['Stat'] = pp_df['Stat'].replace(PP_NORMALIZATION_MAP)
        # -------------------------------------
        
        print(f"✅ Got {len(pp_df)} PrizePicks props.")

        print("\n--- 2. Fetching FanDuel Odds ---")
        fd = FanDuelClient()
        fd_df = fd.get_all_odds() 
        print(f"✅ Got {len(fd_df)} FanDuel props.")

        if pp_df.empty or fd_df.empty:
            print("\n⚠️  Stopping: One of the data sources is empty.")
            input("\nPress Enter to return to menu...")
            return

        print("\n--- 3. Analyzing All Lines ---")
        analyzer = PropsAnalyzer(pp_df, fd_df)
        all_bets = analyzer.calculate_edges()

        if not all_bets.empty:
            sorted_bets = all_bets.sort_values(by='Implied_Win_%', ascending=False)
            print("\n🔥 TOP 15 HIGHEST PROBABILITY PLAYS:")
            print(sorted_bets[['Date', 'Player', 'Stat', 'Side', 'Line', 'Implied_Win_%']].head(15).to_string(index=False))
            
            output_folder = "program_runs"
            if not os.path.exists(output_folder): os.makedirs(output_folder)
            
            for game_date in sorted_bets['Date'].unique():
                day_data = sorted_bets[sorted_bets['Date'] == game_date]
                day_data.to_csv(f"{output_folder}/scan_{game_date}.csv", index=False)
        else:
            print("❌ No profitable matches found!")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    
    input("\nPress Enter to return to menu...")

# --- TOOL 3: AI SCANNER ---
def run_ai_scanner():
    if ai_scanner_module:
        try:
            ai_scanner_module.main()
        except Exception as e:
            print(f"❌ Error running AI Scanner: {e}")
            input("Press Enter...")
    else:
        print("\n❌ Error: AI Scanner module not loaded.")
        input("Press Enter...")

# --- MAIN MENU UI ---
def main_menu():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("")
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
        
        if choice == '1':
            run_correlated_scanner()
        elif choice == '2':
            run_odds_scanner()
        elif choice == '3':
            run_ai_scanner()
        elif choice == '0':
            print("\nGoodbye! 👋\n")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main_menu()