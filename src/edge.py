import pandas as pd
import os
import sys

# --- CONFIGURATION ---
PROJECTIONS_FILE = 'data/todays_projections.csv'

def load_projections():
    if not os.path.exists(PROJECTIONS_FILE):
        print("ERROR: Projections not found. Run scanner.py first.")
        return None
    return pd.read_csv(PROJECTIONS_FILE)

def find_edge():
    df = load_projections()
    if df is None: return

    print("\n🏀 --- NBA AI EDGE FINDER --- 🏀")
    print("Type 'exit' to quit at any time.")
    
    while True:
        # 1. Ask for Player
        query = input("\n>> Enter Player Name (e.g. 'LeBron'): ").strip().lower()
        if query == 'exit': break
            
        # Fuzzy Search
        matches = df[df['NAME'].str.lower().str.contains(query)]
        
        if matches.empty:
            print("❌ Player not found.")
            continue
            
        for _, player in matches.iterrows():
            print(f"\n🔎 FOUND: {player['NAME']} ({player['TEAM']})")
            
            # Identify all stat columns (excluding metadata)
            ignore_cols = ['PLAYER_ID', 'NAME', 'TEAM', 'IS_HOME', 'MATCHUP', 'GAME_DATE']
            available_stats = [col for col in player.index if col not in ignore_cols]
            
            print(f"   📊 Markets: {', '.join(available_stats)}")

            while True:
                # 2. Ask for Stat
                print("-" * 20)
                choice = input("   Select Stat (or type 'ALL' to see everything): ").strip().upper()
                
                if choice == 'EXIT': sys.exit()
                if choice == 'NEXT': break # Go to next player search
                
                # OPTION A: Show All Projections
                if choice == 'ALL':
                    print(f"\n   🤖 FULL MODEL REPORT: {player['NAME']}")
                    print("   " + "="*30)
                    # Group them nicely if possible, otherwise list all
                    for stat in available_stats:
                        val = player[stat]
                        print(f"   • {stat:<5} : {val}")
                    print("   " + "="*30)
                    continue

                # OPTION B: Check Specific Edge
                if choice not in available_stats:
                    print(f"   ❌ '{choice}' not found. Try 'ALL' to see options.")
                    continue
                
                try:
                    line = float(input(f"   Enter PrizePicks Line for {choice}: "))
                    proj = player[choice]
                    diff = proj - line
                    
                    print(f"\n   🔮 Model: {proj} | 🏛️ Line: {line}")
                    
                    if diff > 1.5:
                        print(f"   🔥 HAMMER OVER (Edge: +{diff:.1f})")
                    elif diff < -1.5:
                        print(f"   ❄️ HAMMER UNDER (Edge: {diff:.1f})")
                    else:
                        print(f"   ⚠️ NO PLAY (Edge: {diff:.1f})")
                        
                except ValueError:
                    print("   ❌ Invalid line.")

if __name__ == "__main__":
    find_edge()