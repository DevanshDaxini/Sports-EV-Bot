import pandas as pd
import xgboost as xgb
import os
import warnings
import unicodedata
import re
from datetime import datetime, timedelta
from nba_api.stats.endpoints import ScoreboardV2, BoxScoreTraditionalV2
from prizepicks import fetch_current_lines_dict

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) 
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'training_dataset.csv')
PROJ_DIR = os.path.join(ROOT_DIR, 'data', 'projections')

# File paths
TODAY_SCAN_FILE = os.path.join(PROJ_DIR, 'todays_automated_analysis.csv')
TOMORROW_SCAN_FILE = os.path.join(PROJ_DIR, 'tomorrows_automated_analysis.csv')

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
def get_betting_indicator(proj, line):
    """Returns simplified Over/Under signal with raw difference."""
    if line is None or line <= 0: return "⚪ NO LINE"
    diff = proj - line
    if diff > 0: return f"🟢 OVER (+{diff:.2f})"
    else: return f"🔴 UNDER ({diff:.2f})"

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

def get_games(date_offset=0, require_scheduled=True):
    """
    Fetches NBA schedule.
    require_scheduled=True -> Only returns games that haven't started (Status 1).
    """
    target_date = (datetime.now() + timedelta(days=date_offset)).strftime('%Y-%m-%d')
    print(f"...Fetching games for {target_date}")
    try:
        board = ScoreboardV2(game_date=target_date, league_id='00', day_offset=0)
        games = board.game_header.get_data_frame()
        
        if games.empty:
            return {}

        # KEY FIX: Filter specifically for Scheduled games (Status 1)
        if require_scheduled:
            available_games = games[games['GAME_STATUS_ID'] == 1]
            if available_games.empty:
                return {} # Return empty if all games are done
            games = available_games

        team_map = {} 
        for _, g in games.iterrows():
            team_map[g['HOME_TEAM_ID']] = {'is_home': True, 'opp': g['VISITOR_TEAM_ID']}
            team_map[g['VISITOR_TEAM_ID']] = {'is_home': False, 'opp': g['HOME_TEAM_ID']}
        return team_map
    except Exception as e:
        print(f"Error fetching games: {e}")
        return {}

def get_actual_stats_for_grading():
    """Fetches finalized box scores for TODAY only."""
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"...Fetching actual results for {today} from NBA API...")
    try:
        board = ScoreboardV2(game_date=today, league_id='00', day_offset=0)
        games = board.game_header.get_data_frame()
        
        if games.empty:
            print("❌ No games scheduled for today.")
            return {}

        completed_games = games[games['GAME_STATUS_ID'] == 3]
        
        if completed_games.empty:
            print("⚠️ No games have finished yet today. Cannot grade.")
            return {}

        player_stats = {}
        game_ids = completed_games['GAME_ID'].tolist()
        print(f"...Found {len(game_ids)} completed games. Downloading box scores...")
        
        for gid in game_ids:
            try:
                box = BoxScoreTraditionalV2(game_id=gid, timeout=10)
                stats = box.player_stats.get_data_frame()
                if stats is None or stats.empty: continue
                
                for _, row in stats.iterrows():
                    name = normalize_name(row['PLAYER_NAME'])
                    # Basic stats
                    pts = row['PTS'] if row['PTS'] else 0
                    reb = row['REB'] if row['REB'] else 0
                    ast = row['AST'] if row['AST'] else 0
                    blk = row['BLK'] if row['BLK'] else 0
                    stl = row['STL'] if row['STL'] else 0
                    tov = row['TOV'] if row['TOV'] else 0
                    fg3m = row['FG3M'] if row['FG3M'] else 0
                    
                    # Compound stats
                    player_stats[name] = {
                        'PTS': pts, 'REB': reb, 'AST': ast,
                        'FG3M': fg3m, 'BLK': blk, 'STL': stl, 'TOV': tov,
                        'PRA': pts + reb + ast,
                        'PR': pts + reb,
                        'PA': pts + ast,
                        'RA': reb + ast,
                        'SB': stl + blk
                    }
            except: continue
                
        return player_stats
    except Exception as e:
        print(f"❌ Error fetching box scores: {e}")
        return {}

def grade_results():
    """Grades the 'todays_automated_analysis.csv' file."""
    if not os.path.exists(TODAY_SCAN_FILE):
        print("❌ No scan file found for today. Run 'Scan TODAY's Games' first.")
        input("Press Enter to continue...")
        return

    actuals = get_actual_stats_for_grading()
    if not actuals:
        input("Press Enter to continue...")
        return

    print("\n📝 --- GRADING REPORT ---")
    df_preds = pd.read_csv(TODAY_SCAN_FILE)
    
    total_graded = 0
    correct_picks = 0
    stat_tracker = {} 

    print(f"\n{'PLAYER':<20} | {'STAT':<5} | {'PICK':<5} | {'LINE':<5} | {'ACTUAL':<6} | {'RESULT'}")
    print("-" * 75)
    
    for _, row in df_preds.iterrows():
        name = normalize_name(row['NAME'])
        target = row['TARGET']
        line = row['PP']
        rec = row['REC'] 
        
        if name not in actuals: continue 
        
        actual_val = actuals[name].get(target)
        if actual_val is None: continue
        
        pick_type = "NONE"
        if "OVER" in rec: pick_type = "OVER"
        elif "UNDER" in rec: pick_type = "UNDER"
        else: continue 
        
        is_win = False
        result_icon = "❌"
        
        if pick_type == "OVER":
            if actual_val > line: is_win = True
            elif actual_val == line: is_win = "PUSH"
        elif pick_type == "UNDER":
            if actual_val < line: is_win = True
            elif actual_val == line: is_win = "PUSH"
            
        if is_win == True: result_icon = "✅"
        elif is_win == "PUSH": result_icon = "➖"
        
        print(f"{row['NAME']:<20} | {target:<5} | {pick_type:<5} | {line:<5.1f} | {actual_val:<6.0f} | {result_icon}")

        if is_win != "PUSH":
            total_graded += 1
            if target not in stat_tracker: stat_tracker[target] = {'wins': 0, 'total': 0}
            stat_tracker[target]['total'] += 1
            
            if is_win:
                correct_picks += 1
                stat_tracker[target]['wins'] += 1

    if total_graded > 0:
        print("\n" + "="*40)
        print("📊 PERFORMANCE SUMMARY")
        print("="*40)
        
        total_acc = (correct_picks / total_graded) * 100
        print(f"\n🏆 TOTAL ACCURACY: {total_acc:.1f}% ({correct_picks}/{total_graded})")
        
        print("\n📌 ACCURACY BY STAT CATEGORY:")
        print(f"{'CATEGORY':<10} | {'WIN %':<10} | {'RECORD':<10}")
        print("-" * 35)
        
        for stat, data in sorted(stat_tracker.items()):
            wins = data['wins']
            tot = data['total']
            pct = (wins / tot) * 100 if tot > 0 else 0
            print(f"{stat:<10} | {pct:<9.1f}% | {wins}/{tot}")
            
    else:
        print("\n⚠️ No bets could be graded.")
    
    input("\nPress Enter to continue...")

def prepare_features(player_row, is_home=0, days_rest=2):
    features = {col: player_row.get(col, 0) for col in FEATURES}
    features['IS_HOME'] = 1 if is_home else 0
    features['DAYS_REST'] = days_rest
    features['IS_B2B'] = 1 if days_rest == 1 else 0
    return pd.DataFrame([features])

def scout_player(df_history, models):
    print("\n🔎 --- PLAYER SCOUT ---")
    
    print("Which game date?")
    print("1. Today")
    print("2. Tomorrow")
    d_choice = input("Select (1/2): ").strip()
    
    offset = 1 if d_choice == '2' else 0
    
    todays_teams = get_games(date_offset=offset, require_scheduled=True)
    
    if not todays_teams:
        print("\n❌ No upcoming/scheduled games found for this date.")
        input("Press Enter to return to menu...")
        return

    scouting = True
    while scouting:
        query = input("\nEnter player name: ").strip().lower()
        matches = df_history[df_history['PLAYER_NAME'].str.lower().str.contains(query)]
        
        if matches.empty:
            print("❌ Player not found in database.")
        
        else:
            unique_players = matches[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
            if len(unique_players) > 1:
                print(f"\nFound {len(unique_players)} players:")
                print(unique_players.to_string(index=False))
                pid_input = input("Enter PLAYER_ID from above: ")
                try:
                    matches = matches[matches['PLAYER_ID'] == int(pid_input)]
                except: continue

        print("...Fetching live PrizePicks lines")
        live_lines = fetch_current_lines_dict()
        norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

        player_data = matches.sort_values('GAME_DATE').iloc[-1]
        name = player_data['PLAYER_NAME']
        team_id = player_data['TEAM_ID']
        
        if team_id not in todays_teams:
            print(f"\n⚠️ {name}'s team is not playing on the selected date (or game is over).")
            input("Press Enter to continue...")
            return

        is_home = todays_teams.get(team_id, {'is_home': 0})['is_home']
        
        print(f"\n📊 SCOUTING REPORT: {name}")
        print(f"{'MARKET':<8} | {'PROJ':<8} | {'LINE':<8} | {'RECOMMENDATION':<20}")
        print("-" * 60)
        
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
        
        user_input = input("\nType 'next' to scout another player OR Press Enter for Main Menu: ").lower()
        
        if user_input != 'next':
            scouting = False

def scan_all(df_history, models, is_tomorrow=False):
    offset = 1 if is_tomorrow else 0
    todays_teams = get_games(date_offset=offset, require_scheduled=True)
    
    if not todays_teams:
        date_str = "Tomorrow" if is_tomorrow else "Today"
        print(f"\n❌ No scheduled games found for {date_str}.")
        print("   (Games may be over or none are scheduled).")
        input("Press Enter to return to menu...")
        return

    print("\n🚀 Fetching Live PrizePicks Lines...")
    live_lines = fetch_current_lines_dict() 
    norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

    print(f"🚀 Scanning Markets...")
    best_bets = []
    
    for team_id, info in todays_teams.items():
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
                
                line = norm_lines.get(normalize_name(player_name), {}).get(target)
                rec = get_betting_indicator(proj, line)
                
                if line is not None and line > 0:
                    edge = proj - line
                    best_bets.append({
                        'REC': rec, 'NAME': player_name, 'TARGET': target,
                        'AI': round(proj, 2), 'PP': round(line, 2), 'EDGE': edge
                    })
            
    if best_bets:
        # Sort and separate Top 10 Overs and Unders
        top_overs = sorted([b for b in best_bets if b['EDGE'] > 0], key=lambda x: x['EDGE'], reverse=True)[:10]
        top_unders = sorted([b for b in best_bets if b['EDGE'] < 0], key=lambda x: x['EDGE'])[:10] # Most negative first
        
        print("\n🔥 TOP 10 OVERS (Highest Value)")
        print(f" {'REC':<20} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15}")
        print("-" * 75)
        for bet in top_overs:
            print(f" {bet['REC']:<20} | {bet['NAME']:<20} | {bet['TARGET']:<5} | {bet['AI']:>6.2f} vs {bet['PP']:>6.2f}")
        
        print("\n❄️ TOP 10 UNDERS (Lowest Value)")
        print(f" {'REC':<20} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15}")
        print("-" * 75)
        for bet in top_unders:
            print(f" {bet['REC']:<20} | {bet['NAME']:<20} | {bet['TARGET']:<5} | {bet['AI']:>6.2f} vs {bet['PP']:>6.2f}")
        
        save_path = TOMORROW_SCAN_FILE if is_tomorrow else TODAY_SCAN_FILE
        res_df = pd.DataFrame(best_bets)
        res_df.to_csv(save_path, index=False)
        print(f"\n✅ Full analysis saved to {save_path}")
    else:
        print("\n⚠️ No active lines found matching the scheduled players.")
    
    input("\nPress Enter to continue...")

def main():
    print("...Initializing System")
    df = load_data()
    models = load_models()
    if df is None or not models:
        print("❌ Setup failed. Check your data and models.")
        return

    while True:
        print("")
        print("")
        print("\n" + "="*30 + "\n   🤖 NBA AI SCANNER v2.7\n" + "="*30)
        print("1. 🚀 Scan TODAY'S Games")
        print("2. 🔮 Scan TOMORROW'S Games")
        print("3. 📝 Grade TODAY'S Results")
        print("4. 🔎 Scout Specific Player")
        print("0. 🚪 Exit")
        
        choice = input("\nSelect: ").strip()
        
        if choice == '1': 
            scan_all(df, models, is_tomorrow=False)
        elif choice == '2': 
            scan_all(df, models, is_tomorrow=True)
        elif choice == '3':
            grade_results()
        elif choice == '4': 
            scout_player(df, models)
        elif choice == '0': break

if __name__ == "__main__":
    main()