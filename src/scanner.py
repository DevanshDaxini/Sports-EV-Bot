import pandas as pd
import xgboost as xgb
import os
import warnings
import unicodedata
import re
from datetime import datetime
from nba_api.stats.endpoints import ScoreboardV2
from prizepicks import fetch_current_lines_dict

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) 
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'training_dataset.csv')
PROJ_DIR = os.path.join(ROOT_DIR, 'data', 'projections')

warnings.filterwarnings('ignore')

TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'SB',
    'FGM', 'FTM', 'FTA'
]

FEATURES = [
    'PTS_L5', 'PTS_L20', 'PTS_SEASON',
    'REB_L5', 'REB_L20', 'REB_SEASON',
    'AST_L5', 'AST_L20', 'AST_SEASON',
    'FG3M_L5', 'FG3M_L20', 'FG3M_SEASON',
    'STL_L5', 'STL_L20', 'STL_SEASON',
    'BLK_L5', 'BLK_L20', 'BLK_SEASON',
    'TOV_L5', 'TOV_L20', 'TOV_SEASON',
    'MIN_L5', 'MIN_L20', 'MIN_SEASON',
    'GAME_SCORE_L5', 'GAME_SCORE_L20', 'GAME_SCORE_SEASON',
    'TS_PCT', 'DAYS_REST', 'IS_HOME', 'IS_B2B',
    'IMPLIED_BLOWOUT_RISK', 'TEAM_ACTIVE_QUALITY', 
    'PTS_HEAT_L5', 'REB_HEAT_L5', 'AST_HEAT_L5',
    'PTS_MOMENTUM', 'REB_MOMENTUM', 'AST_MOMENTUM'
]
for stat in ['PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')

# --- NORMALIZATION HELPERS ---
def normalize_name(name):
    """Strips accents and suffixes to ensure matching."""
    if not name: return ""
    n = unicodedata.normalize('NFKD', name)
    clean = "".join([c for c in n if not unicodedata.combining(c)])
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)
    suffixes = ['Jr', 'Sr', 'III', 'II', 'IV']
    for s in suffixes:
        clean = clean.replace(f" {s}", "")
    return " ".join(clean.lower().split())

# --- BETTING LOGIC ---
def get_betting_indicator(proj, line, target):
    """
    Calculates edge with dynamic thresholds based on model confidence.
    - High-Variance Stats (STL, BLK, SB): 15% Edge required.
    - Standard Stats (PTS, REB, AST, etc.): 8% Edge required.
    """
    if line is None or line <= 0: return "⚪ NO LINE"
    
    diff_pct = (proj - line) / line
    
    # Define high-variance/low-confidence markets
    high_variance_stats = ['STL', 'BLK', 'SB']
    
    # Set threshold based on the target
    threshold = 0.15 if target in high_variance_stats else 0.08
    
    if diff_pct > threshold: return f"🟢 OVER ({diff_pct:+.1%})"
    if diff_pct < -threshold: return f"🔴 UNDER ({diff_pct:+.1%})"
    
    return "⚪ NO BET"

# --- CORE FUNCTIONS ---
def load_data():
    if not os.path.exists(DATA_FILE): return None
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().upper() for c in df.columns]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def load_models():
    models = {}
    for target in TARGETS:
        path = os.path.join(MODEL_DIR, f"{target}_model.json")
        if os.path.exists(path):
            m = xgb.XGBRegressor()
            m.load_model(path)
            models[target] = m
    return models

def get_todays_games():
    """Fetches today's NBA schedule."""
    today = datetime.now().strftime('%Y-%m-%d')
    try:
        board = ScoreboardV2(game_date=today, league_id='00', day_offset=0)
        games = board.game_header.get_data_frame()
        if 'GAME_STATUS_ID' in games.columns:
            games = games[games['GAME_STATUS_ID'] == 1]
        team_map = {} 
        for _, g in games.iterrows():
            team_map[g['HOME_TEAM_ID']] = {'is_home': True, 'opp': g['VISITOR_TEAM_ID']}
            team_map[g['VISITOR_TEAM_ID']] = {'is_home': False, 'opp': g['HOME_TEAM_ID']}
        return team_map
    except Exception as e:
        print(f"Error fetching games: {e}")
        return {}

def prepare_features(player_row, is_home=0, days_rest=2):
    features = {col: player_row.get(col, 0) for col in FEATURES}
    features['IS_HOME'] = 1 if is_home else 0
    features['DAYS_REST'] = days_rest
    features['IS_B2B'] = 1 if days_rest == 1 else 0
    return pd.DataFrame([features])

def scout_player(df_history, models, todays_teams):
    print("\n🔎 --- PLAYER SCOUT ---")
    query = input("Enter player name: ").strip().lower()
    matches = df_history[df_history['PLAYER_NAME'].str.lower().str.contains(query)]
    
    if matches.empty:
        print("❌ Player not found.")
        return
    
    unique_players = matches[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
    if len(unique_players) > 1:
        print(f"\nFound {len(unique_players)} players:")
        print(unique_players.to_string(index=False))
        pid_input = input("Enter PLAYER_ID from above: ")
        try:
            matches = matches[matches['PLAYER_ID'] == int(pid_input)]
        except: return

    print("...Fetching live PrizePicks lines")
    live_lines = fetch_current_lines_dict()
    norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

    player_data = matches.sort_values('GAME_DATE').iloc[-1]
    name = player_data['PLAYER_NAME']
    team_id = player_data['TEAM_ID']
    is_home = todays_teams.get(team_id, {'is_home': 0})['is_home']
    
    print(f"\n📊 SCOUTING REPORT: {name}")
    print(f"{'MARKET':<8} | {'PROJ':<8} | {'LINE':<8} | {'BET REC':<12}")
    print("-" * 50)
    
    input_row = prepare_features(player_data, is_home=is_home)
    
    for target in TARGETS:
        if target in models:
            feats = models[target].feature_names_in_
            valid_input = input_row.reindex(columns=feats, fill_value=0)
            pred = float(models[target].predict(valid_input)[0])
            
            line = norm_lines.get(normalize_name(name), {}).get(target)
            indicator = get_betting_indicator(pred, line)
            line_str = f"{line:.2f}" if line else "N/A"
            print(f"{target:<8} : {pred:<8.2f} | {line_str:<8} | {indicator}")
    
    input("\nPress Enter to continue...")

def scan_all(df_history, models, todays_teams):
    if not todays_teams:
        print("❌ No games found.")
        return

    print("\n🚀 Fetching Live PrizePicks Lines...")
    live_lines = fetch_current_lines_dict() 

    print(f"\n🚀 Scanning All Markets...")
    all_projections = []
    best_bets = [] # Stores potential plays with mathematical edge
    
    for team_id, info in todays_teams.items():
        team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            player_name = last_row['PLAYER_NAME']
            
            input_row = prepare_features(last_row, is_home=info['is_home'])
            preds = {'NAME': player_name, 'TEAM': last_row['TEAM_ABBREVIATION']}
            
            for target, model in models.items():
                feats = model.feature_names_in_
                valid_input = input_row.reindex(columns=feats, fill_value=0)
                proj = float(model.predict(valid_input)[0])
                
                # Rounding for cleanliness
                preds[f"{target}_PROJ"] = round(proj, 2)
                line = live_lines.get(player_name, {}).get(target)
                preds[f"{target}_LINE"] = round(line, 2) if line else None
                
                rec = get_betting_indicator(proj, line, target)
                preds[f"{target}_REC"] = rec
                
                # Capture the raw edge for sorting
                if line and line > 0:
                    edge = (proj - line) / line
                    if abs(edge) > 0.08: # Only track if edge meets 8% threshold
                        best_bets.append({
                            'REC': rec,
                            'NAME': player_name,
                            'TARGET': target,
                            'AI': round(proj, 2),
                            'PP': round(line, 2),
                            'EDGE': edge # Positive for Over, Negative for Under
                        })
            
            all_projections.append(preds)
            
    res_df = pd.DataFrame(all_projections)
    if not res_df.empty:
        # Separate and sort the top 5 Overs and top 5 Unders
        top_overs = sorted([b for b in best_bets if b['EDGE'] > 0], key=lambda x: x['EDGE'], reverse=True)[:5]
        top_unders = sorted([b for b in best_bets if b['EDGE'] < 0], key=lambda x: x['EDGE'])[:5]
        
        print("\n🔥 TOP AI ADVANTAGES FOUND:")
        print(f" {'REC':<18} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15}")
        print("-" * 70)
        
        # Print Top Overs
        for bet in top_overs:
            print(f" {bet['REC']:<18} | {bet['NAME']:<20} | {bet['TARGET']:<5} | {bet['AI']:>6.2f} vs {bet['PP']:>6.2f}")
            
        print("-" * 70)
        
        # Print Top Unders
        for bet in top_unders:
            print(f" {bet['REC']:<18} | {bet['NAME']:<20} | {bet['TARGET']:<5} | {bet['AI']:>6.2f} vs {bet['PP']:>6.2f}")
        
        # Save full data to CSV for record keeping
        if not os.path.exists(PROJ_DIR): os.makedirs(PROJ_DIR)
        path = os.path.join(PROJ_DIR, 'todays_automated_analysis.csv')
        res_df.to_csv(path, index=False)
        print(f"\n✅ Full analysis saved to {path}")
    input("\nPress Enter to continue...")

def main():
    print("...Initializing System")
    df = load_data()
    models = load_models()
    if df is None or not models:
        print("❌ Setup failed. Check your data and models.")
        return

    while True:
        todays_teams = get_todays_games() # This was the missing function
        print("\n" + "="*30 + "\n   🤖 NBA AI SCANNER v2.3\n" + "="*30)
        print(f"   📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"   🏀 Games: {len(todays_teams) // 2}")
        print("-" * 30)
        print("1. 🚀 Run Automated Market Scan\n2. 🔎 Scout Player (with live PP comparison)\n0. 🚪 Exit")
        
        choice = input("\nSelect: ").strip()
        if choice == '1': scan_all(df, models, todays_teams)
        elif choice == '2': scout_player(df, models, todays_teams)
        elif choice == '0': break

if __name__ == "__main__":
    main()