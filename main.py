import os
import sys
from datetime import datetime

# --- SYSTEM PATH SETUP ---
# This ensures that when we import scanner from 'src', it can still find 
# its own dependencies (like prizepicks) inside that same folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# --- IMPORT TOOLS ---
try:
    from src.prizepicks import PrizePicksClient
    from src.fanduel import FanDuelClient
    from src.analyzer import PropsAnalyzer
except ImportError as e:
    print(f"⚠️ Warning: Could not import core modules: {e}")

# Try importing the AI Scanner from the src folder
ai_scanner_module = None
try:
    import src.scanner as ai_scanner_module
except ImportError:
    print("⚠️ Warning: 'scanner.py' not found in 'src' folder.")

# --- TOOL 1: ODDS SCANNER (Your Original Code) ---
def run_odds_scanner():
    os.system('cls' if os.name == 'nt' else 'clear')
    
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

        # --- FIX: Stop if either side is empty ---
        if pp_df.empty or fd_df.empty:
            print("\n⚠️  Stopping: One of the data sources is empty.")
            print("   (This usually happens late at night when odds are pulled).")
            input("\nPress Enter to return to menu...")
            return
        # ----------------------------------------

        print("\n--- 3. Analyzing All Lines ---")
        analyzer = PropsAnalyzer(pp_df, fd_df)
        all_bets = analyzer.calculate_edges()

        if not all_bets.empty:
            # Sort by Win %
            sorted_bets = all_bets.sort_values(by='Implied_Win_%', ascending=False)
            
            print("\n🔥 TOP 15 HIGHEST PROBABILITY PLAYS:")
            print(sorted_bets[['Date', 'Player', 'Stat', 'Side', 'Line', 'Implied_Win_%']].head(15).to_string(index=False))
            
            # Save Data
            output_folder = "program_runs"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            unique_dates = sorted_bets['Date'].unique()
            print(f"\n💾 Saving Data for {len(unique_dates)} Game Days...")

            for game_date in unique_dates:
                day_data = sorted_bets[sorted_bets['Date'] == game_date]
                filename = f"{output_folder}/scan_{game_date}.csv"
                day_data.to_csv(filename, index=False)
                print(f" -> Saved {len(day_data)} lines to {filename}")

        else:
            print("❌ No profitable matches found!")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    
    input("\nPress Enter to return to menu...")

# --- TOOL 2: AI SCANNER ---
def run_ai_scanner():
    if ai_scanner_module:
        try:
            # Transfer control to the scanner's internal menu
            ai_scanner_module.main()
        except Exception as e:
            print(f"❌ Error running AI Scanner: {e}")
            input("Press Enter...")
    else:
        print("\n❌ Error: AI Scanner module not loaded.")
        print("Make sure 'scanner.py' is inside the 'src' folder.")
        input("Press Enter...")

# --- MAIN MENU UI ---
def main_menu():
    while True:
        # Clear screen for a clean UI
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("")
        print("")
        print("\n" + "🏀"*12 + "  SPORTS ANALYTICS HUB  " + "🏀"*12)
        print("-" * 72)
        print("\nSelect a Tool:")
        print("1. 💰 Odds Scanner (Arbitrage)")
        print("   -> Compares FanDuel vs PrizePicks for math-based edges.")
        print("\n2. 🤖 NBA AI Scanner (Predictive Model)")
        print("   -> Uses your XGBoost models to predict Over/Under.")
        print("\n0. 🚪 Exit")
        
        choice = input("\nSelect Option: ").strip()
        
        if choice == '1':
            run_odds_scanner()
        elif choice == '2':
            run_ai_scanner()
        elif choice == '0':
            print("")
            print("Goodbye! 👋")
            print("")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main_menu()