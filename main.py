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
    # Import specific functions from scanner to run them programmatically
    from src.scanner import load_data, load_models, get_games, prepare_features, normalize_name
except ImportError as e:
    print(f"⚠️ Warning: Could not import core modules: {e}")

# Try importing the AI Scanner module for the menu
ai_scanner_module = None
try:
    import src.scanner as ai_scanner_module
except ImportError:
    pass

warnings.filterwarnings('ignore')

# --- HELPER: RUN AI PREDICTIONS ---
def get_ai_predictions():
    """
    Runs the AI model logic silently and returns a DataFrame of projections.
    """
    print("...Loading AI Models & Data")
    df_history = load_data()
    models = load_models()
    
    if df_history is None or not models:
        return pd.DataFrame()

    # Get games for Today (0) and Tomorrow (1) to cover all bases
    todays_teams = get_games(date_offset=0, require_scheduled=True)
    tomorrows_teams = get_games(date_offset=1, require_scheduled=True)
    
    # Merge dictionaries
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
            
            # Predict for all targets
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
        
    except Exception as e:
        print(f"❌ Error in Odds Scanner: {e}")
        return

    # 2. Run AI Scanner (Data)
    print("\n--- 2. generating AI Projections ---")
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
    
    # Normalize names for merging
    math_bets['CleanName'] = math_bets['Player'].apply(normalize_name)
    ai_df['CleanName'] = ai_df['Player'].apply(normalize_name)
    
    # Merge Math Bets with AI Projections
    merged = pd.merge(math_bets, ai_df, on=['CleanName', 'Stat'], how='inner')
    
    correlated_plays = []
    
    for _, row in merged.iterrows():
        math_side = row['Side'] # 'Over' or 'Under'
        line = row['Line']
        ai_proj = row['AI_Proj']
        
        # Check for AGREEMENT
        ai_side = "None"
        if ai_proj > line: ai_side = "Over"
        elif ai_proj < line: ai_side = "Under"
        
        # Only keep plays where Math and AI agree
        if math_side == ai_side:
            # Calculate AI Edge
            ai_diff = ai_proj - line
            
            correlated_plays.append({
                'Date': row['Date'],
                'Player': row['Player_x'], # Name from math df
                'Stat': row['Stat'],
                'Line': line,
                'Side': math_side,
                'Win%': row['Implied_Win_%'], # From FanDuel
                'AI_Proj': ai_proj,
                'AI_Diff': f"{ai_diff:+.1f}" # e.g. +2.5 or -1.2
            })
            
    # 4. Display Results
    if not correlated_plays:
        print("❌ No correlated plays found (Math and AI disagreed on everything).")
    else:
        # Create DataFrame and Sort by Win%
        final_df = pd.DataFrame(correlated_plays)
        final_df = final_df.sort_values(by='Win%', ascending=False).head(20)
        
        print("\n💎 TOP 20 CORRELATED PLAYS (Math + AI Agree)")
        print(f"{'PLAYER':<20} | {'STAT':<5} | {'LINE':<5} | {'SIDE':<5} | {'WIN%':<6} | {'AI PROJ':<8} | {'DIFF'}")
        print("-" * 85)
        
        for _, row in final_df.iterrows():
            print(f"{row['Player']:<20} | {row['Stat']:<5} | {row['Line']:<5} | {row['Side']:<5} | {row['Win%']:<6}% | {row['AI_Proj']:<8} | {row['AI_Diff']}")
            
        # Save
        path = "program_runs/correlated_plays.csv"
        if not os.path.exists("program_runs"): os.makedirs("program_runs")
        final_df.to_csv(path, index=False)
        print(f"\n💾 Saved list to {path}")

    input("\nPress Enter to return to menu...")

# --- TOOL 2: ODDS SCANNER (Original) ---
def run_odds_scanner():
    print("")
    print("\n" + "="*40)
    print("   💰 ODDS ARBITRAGE SCANNER")
    print("="*40)
    
    try:
        print("--- 1. Fetching PrizePicks Lines ---")
        pp = PrizePicksClient()
        pp_df = pp.fetch_board()
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
            
            # Save by date
            for game_date in sorted_bets['Date'].unique():
                day_data = sorted_bets[sorted_bets['Date'] == game_date]
                day_data.to_csv(f"{output_folder}/scan_{game_date}.csv", index=False)

        else:
            print("❌ No profitable matches found!")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    
    input("\nPress Enter to return to menu...")

# --- TOOL 3: AI SCANNER (Original) ---
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
        # Soft Clear
        print("\n" * 50)
        
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