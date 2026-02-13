"""
NBA Props Scanner - AI-Powered Prediction System

This module scans upcoming NBA games, generates player performance 
predictions using trained XGBoost models, and identifies profitable 
betting opportunities by comparing predictions against PrizePicks lines.

Key Features:
    - Real-time injury data integration
    - Correlation-constrained predictions (ensures logical consistency)
    - Model confidence tiers (Elite/Strong/Decent/Weak)
    - Feature leakage prevention during prediction
    - Historical accuracy tracking and grading

Main Components:
    - scout_player(): Individual player deep-dive analysis
    - scan_all(): Batch analysis of all games
    - grade_results(): Post-game accuracy validation
    - prepare_features(): Feature vector construction for models

Model Tiers (Based on Training Directional Accuracy):
    - ⭐ ELITE (>85%): PTS, FGM, PA, PR, PRA
    - ✓ STRONG (80-85%): FG3A, FGA
    - ~ DECENT (75-80%): FG3M, FTA
    - ⚠ WEAK (<75%): BLK, STL, SB, TOV (filtered out)

Usage:
    $ python3 -m src.scanner
    Select from menu:
        1. Scan TODAY'S Games
        2. Scan TOMORROW'S Games
        3. Grade Results
        4. Scout Specific Player
"""

import pandas as pd
import xgboost as xgb
import os
import warnings
import unicodedata
import re
import time
from datetime import datetime, timedelta
from nba_api.stats.endpoints import ScoreboardV2, LeagueGameLog
from prizepicks import fetch_current_lines_dict
from config import STAT_MAP 
from injuries import get_injury_report

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) 
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'training_dataset.csv')
PROJ_DIR = os.path.join(ROOT_DIR, 'data', 'projections')

TODAY_SCAN_FILE = os.path.join(PROJ_DIR, 'todays_automated_analysis.csv')
TOMORROW_SCAN_FILE = os.path.join(PROJ_DIR, 'tomorrows_automated_analysis.csv')
ACCURACY_LOG_FILE = os.path.join(PROJ_DIR, 'accuracy_log.csv')

warnings.filterwarnings('ignore')

# LOAD INJURIES ONCE WHEN SCRIPT STARTS
print("...Loading Injury Report")
INJURY_DATA = get_injury_report() 

def get_player_status(name):
    """
    Check if a player is injured or available.
    
    Args:
        name (str): Player's full name
        
    Returns:
        str: Status code - 'OUT', 'QUESTIONABLE', 'GTD', or 'Active'
        
    Note:
        Uses fuzzy name matching via normalize_name() to handle Jr/Sr suffixes
    """

    norm_name = normalize_name(name)
    for injured_name, status in INJURY_DATA.items():
        if normalize_name(injured_name) == norm_name:
            return status
    return "Active"

TARGETS = list(STAT_MAP.values())

# FIX #16: Match train.py FEATURES exactly (with proper 
# column names from training data)
FEATURES = [
    'PTS_L5', 'PTS_L20', 'PTS_Season',
    'REB_L5', 'REB_L20', 'REB_Season',
    'AST_L5', 'AST_L20', 'AST_Season',
    'FG3M_L5', 'FG3M_L20', 'FG3M_Season',
    'STL_L5', 'STL_L20', 'STL_Season',
    'BLK_L5', 'BLK_L20', 'BLK_Season',
    'TOV_L5', 'TOV_L20', 'TOV_Season',
    'FGM_L5', 'FGM_L20', 'FGM_Season',
    'FTM_L5', 'FTM_L20', 'FTM_Season',
    'MIN_L5', 'MIN_L20', 'MIN_Season',
    'GAME_SCORE_L5', 'GAME_SCORE_L20', 'GAME_SCORE_Season',
    'USAGE_RATE_L5', 'USAGE_RATE_L20', 'USAGE_RATE_Season',
    'MISSING_USAGE',
    'TS_PCT', 'DAYS_REST', 'IS_HOME',
    'GAMES_7D', 'IS_4_IN_6', 'IS_B2B', 'IS_FRESH',
    'PACE_ROLLING', 'FGA_PER_MIN', 'TOV_PER_USAGE',
    'USAGE_VACUUM', 'STAR_COUNT'
]

# Add combo rolling features
combo_features = []
for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    combo_features.extend([f'{combo}_L5', f'{combo}_L20', f'{combo}_Season'])
FEATURES.extend(combo_features)

# Add Defense Columns
for stat in ['PTS', 'REB', 'AST', 'FG3M','FGA', 'BLK', 'STL', 'TOV',
             'FGM', 'FTM', 'FTA']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')

# Add combo defensive features
for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.append(f'OPP_{combo}_ALLOWED')

def normalize_name(name):
    """
    Normalize player names for fuzzy matching.
    
    Removes unicode characters, punctuation, and suffixes (Jr, Sr, II, III, IV)
    to improve name matching between different data sources.
    
    Args:
        name (str): Raw player name
        
    Returns:
        str: Normalized lowercase name with suffixes removed
        
    Examples:
        >>> normalize_name("LeBron James Jr.")
        'lebron james'
        >>> normalize_name("Tim Hardaway Jr.")
        'tim hardaway'
    """

    if not name: return ""
    n = unicodedata.normalize('NFKD', name)
    clean = "".join([c for c in n if not unicodedata.combining(c)])
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)
    suffixes = ['Jr', 'Sr', 'III', 'II', 'IV']
    for s in suffixes:
        clean = clean.replace(f" {s}", "")
    return " ".join(clean.lower().split())

def get_betting_indicator(proj, line):
    """
    Generate recommendation text based on projection vs line.
    
    Args:
        proj (float): Model's predicted value
        line (float or None): PrizePicks line value
        
    Returns:
        str: Formatted recommendation string
        
    Examples:
        >>> get_betting_indicator(28.5, 25.5)
        '🟢 OVER (+3.00)'
        >>> get_betting_indicator(22.3, 25.5)
        '🔴 UNDER (-3.20)'
    """

    if line is None or line <= 0: return "⚪ NO LINE"
    diff = proj - line
    if diff > 0: return f"🟢 OVER (+{diff:.2f})"
    else: return f"🔴 UNDER ({diff:.2f})"

def load_data():
    """
    Load historical training dataset with player statistics.
    
    Returns:
        pandas.DataFrame: Historical game logs with engineered features,
                         or None if file doesn't exist
                         
    Note:
        Converts GAME_DATE to datetime and normalizes column names to uppercase
    """

    if not os.path.exists(DATA_FILE): return None
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().upper() for c in df.columns]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def load_models():
    """
    Load all trained XGBoost models from disk.
    
    Returns:
        dict: Mapping of stat name (str) to XGBRegressor model
              Example: {'PTS': <XGBRegressor>, 'REB': <XGBRegressor>, ...}
              
    Note:
        Only loads models that exist in the models/ directory.
        Missing models are skipped silently.
    """

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
    Fetch games scheduled for a specific date using NBA API.
    
    Args:
        date_offset (int): Days from today (0=today, 1=tomorrow, -1=yesterday)
        require_scheduled (bool): If True, only return games not yet started
        
    Returns:
        dict: Mapping of team_id to game info
              Format: {team_id: {'is_home': bool, 'opp': opponent_team_id}}
              Returns empty dict if no games found or API error
              
    Example:
        >>> games = get_games(date_offset=1)  # Tomorrow's games
        >>> print(games[1610612747])  # Lakers
        {'is_home': True, 'opp': 1610612744}  # vs Warriors
    """

    target_date = (datetime.now() + timedelta(days=date_offset)).strftime('%Y-%m-%d')
    print(f"...Fetching games for {target_date}")
    try:
        board = ScoreboardV2(game_date=target_date, league_id='00', day_offset=0)
        games = board.game_header.get_data_frame()
        if games.empty: return {}
        if require_scheduled:
            available_games = games[games['GAME_STATUS_ID'] == 1]
            if available_games.empty: return {}
            games = available_games
        team_map = {} 
        for _, g in games.iterrows():
            team_map[g['HOME_TEAM_ID']] = {'is_home': True, 'opp': g['VISITOR_TEAM_ID']}
            team_map[g['VISITOR_TEAM_ID']] = {'is_home': False, 'opp': g['HOME_TEAM_ID']}
        return team_map
    except Exception as e:
        print(f"Error fetching games: {e}")
        return {}

def get_actual_stats_for_grading(target_date_obj):
    """
    Fetch completed game results for accuracy grading.
    
    Args:
        target_date_obj (datetime): Date of games to fetch
        
    Returns:
        dict: Mapping of normalized player names to their actual stats
              Format: {player_name: {'PTS': 28, 'REB': 8, 'AST': 7, 'PRA': 43, ...}}
              Returns empty dict if no games completed or API error
              
    Note:
        - Automatically detects correct NBA season based on date
        - Includes combo stats (PRA, PR, PA, RA, SB) calculated from base stats
        - Uses normalized names for matching (handles Jr/Sr suffixes)
    """
    
    date_str = target_date_obj.strftime('%m/%d/%Y')
    
    # Logic to find season string (e.g., '2025-26')
    if target_date_obj.month >= 10:
        season_str = f"{target_date_obj.year}-{str(target_date_obj.year + 1)[-2:]}"
    else:
        season_str = f"{target_date_obj.year - 1}-{str(target_date_obj.year)[-2:]}"

    print(f"...Fetching results for {date_str} (Season {season_str})...")
    
    try:
        log = LeagueGameLog(
            season=season_str, 
            date_from_nullable=date_str, 
            date_to_nullable=date_str, 
            player_or_team_abbreviation='P'
        )
        stats_frames = log.get_data_frames()
        
        if not stats_frames or stats_frames[0].empty:
            print("⚠️ No completed games found for this date.")
            return {}

        stats = stats_frames[0]
        print(f"   ✅ Successfully loaded stats for {len(stats)} players.")
        
        player_stats = {}
        
        for _, row in stats.iterrows():
            name = normalize_name(row['PLAYER_NAME'])
            # Extract
            pts, reb, ast = row['PTS'], row['REB'], row['AST']
            blk, stl, tov = row['BLK'], row['STL'], row['TOV']
            fg3m, fgm, fga = row['FG3M'], row['FGM'], row['FGA']
            ftm, fta = row['FTM'], row['FTA']
            
            player_stats[name] = {
                'PTS': pts, 'REB': reb, 'AST': ast,
                'FG3M': fg3m, 'BLK': blk, 'STL': stl, 'TOV': tov,
                'FGM': fgm, 'FGA': fga, 'FTM': ftm, 'FTA': fta,
                'PRA': pts + reb + ast, 
                'PR': pts + reb, 'PA': pts + ast, 'RA': reb + ast, 
                'SB': stl + blk
            }
        return player_stats

    except Exception as e:
        print(f"API Error: {e}")
        return {}

def grade_results():
    """
    Grade previous predictions against actual game results.
    
    Interactive function that:
        1. Prompts user to select TODAY or YESTERDAY
        2. Loads predictions from CSV
        3. Fetches actual stats from NBA API
        4. Compares predictions to reality
        5. Calculates win rate and logs results
        
    Output:
        - Prints top winning bets and biggest misses
        - Updates accuracy_log.csv with results
        - Modifies prediction CSV to add 'Result' and 'Actual' columns
        
    Note:
        Requires predictions to exist in TODAY_SCAN_FILE
        Pushes (ties) are excluded from win rate calculation
    """

    # 1. Ask User for Date
    print("\n📅 GRADING OPTIONS:")
    print("1. Grade TODAY'S Games")
    print("2. Grade YESTERDAY'S Games")
    choice = input("Select (1/2): ").strip()
    
    if choice == '2':
        target_date = datetime.now() - timedelta(days=1)
        print(f"\n🔙 Grading YESTERDAY ({target_date.strftime('%Y-%m-%d')})")
    else:
        target_date = datetime.now()
        print(f"\n📅 Grading TODAY ({target_date.strftime('%Y-%m-%d')})")

    # 2. Load Predictions
    if not os.path.exists(TODAY_SCAN_FILE):
        print("❌ No scan file found. Run Option 1 first to generate predictions.")
        return

    try:
        df_preds = pd.read_csv(TODAY_SCAN_FILE)
    except:
        print("❌ Error reading prediction file.")
        return

    # 3. Get Actuals
    actuals = get_actual_stats_for_grading(target_date)
    if not actuals:
        print("\n❌ Stats unavailable. Games may not be finished.")
        return

    print("\n" + "="*65)
    print("📝 RESULTS ANALYSIS")
    print("="*65)
    
    results = [] # Store detailed results for sorting
    
    total_graded = 0
    correct_picks = 0
    
    for _, row in df_preds.iterrows():
        if row['PP'] == 0: continue
        
        name = normalize_name(row['NAME'])
        target = row['TARGET']
        line = float(row['PP'])
        rec_text = row['REC']
        
        if name not in actuals: continue 
        actual_val = actuals[name].get(target)
        if actual_val is None: continue
        
        pick_type = "NONE"
        if "OVER" in rec_text: pick_type = "OVER"
        elif "UNDER" in rec_text: pick_type = "UNDER"
        else: continue 
        
        is_win = False
        margin = 0
        
        # Margin Calculation: Positive = Good, Negative = Bad
        if pick_type == "OVER":
            margin = actual_val - line # Over 10, Actual 15 => +5 (Win)
            if actual_val > line: is_win = True
            elif actual_val == line: is_win = "PUSH"
                
        elif pick_type == "UNDER":
            margin = line - actual_val # Under 10, Actual 5 => +5 (Win)
            if actual_val < line: is_win = True
            elif actual_val == line: is_win = "PUSH"
            
        if is_win != "PUSH":
            total_graded += 1
            if is_win: correct_picks += 1
            
            results.append({
                'Player': row['NAME'],
                'Stat': target,
                'Pick': pick_type,
                'Line': line,
                'Actual': actual_val,
                'Margin': margin,
                'Win': is_win
            })

    if total_graded == 0:
        print("⚠️ Predictions found, but no matching player stats (check date?).")
        return

    # 4. Sort Top Wins and Losses
    # Sort by Margin (Descending)
    sorted_results = sorted(results, key=lambda x: x['Margin'], reverse=True)
    
    top_wins = [r for r in sorted_results if r['Win']][:5]
    worst_losses = [r for r in sorted_results if not r['Win']][-5:]
    # Reverse losses so biggest negative is first
    worst_losses = sorted(worst_losses, key=lambda x: x['Margin'])[:5] 

    # 5. Display Report
    print("\n🏆 TOP 5 BEST WINS (Biggest Margin)")
    print(f"{'PLAYER':<20} | {'STAT':<5} | {'PICK':<5} | {'LINE':<5} | {'ACTUAL':<6} | {'MARGIN'}")
    print("-" * 70)
    for r in top_wins:
        print(f"{r['Player']:<20} | {r['Stat']:<5} | {r['Pick']:<5} | {r['Line']:<5} | {r['Actual']:<6} | 🟢 +{r['Margin']:.1f}")

    print("\n💀 TOP 5 WORST LOSSES (Biggest Miss)")
    print(f"{'PLAYER':<20} | {'STAT':<5} | {'PICK':<5} | {'LINE':<5} | {'ACTUAL':<6} | {'MARGIN'}")
    print("-" * 70)
    for r in worst_losses:
        print(f"{r['Player']:<20} | {r['Stat']:<5} | {r['Pick']:<5} | {r['Line']:<5} | {r['Actual']:<6} | 🔴 {r['Margin']:.1f}")

    # 6. Summary & Logging
    accuracy = (correct_picks / total_graded) * 100
    print("-" * 70)
    print(f"📊 FINAL ACCURACY: {accuracy:.1f}% ({correct_picks}/{total_graded})")
    
    # Save to Log
    log_exists = os.path.exists(ACCURACY_LOG_FILE)
    with open(ACCURACY_LOG_FILE, 'a') as f:
        if not log_exists:
            f.write("Date,Graded,Correct,Accuracy,Best_Win_Margin\n")
        f.write(f"{target_date.strftime('%Y-%m-%d')},{total_graded},{correct_picks},{accuracy:.2f},{top_wins[0]['Margin'] if top_wins else 0}\n")
    print(f"✅ Results logged to {ACCURACY_LOG_FILE}")
    
    input("\nPress Enter to continue...")

def prepare_features(player_row, is_home=0, days_rest=2, missing_usage=0):
    """
    Construct feature vector for model prediction.
    
    Args:
        player_row (pandas.Series): Player's most recent game data with rolling stats
        is_home (int): 1 if home game, 0 if away
        days_rest (int): Days since player's last game
        missing_usage (float): Percentage of team usage from injured teammates
        
    Returns:
        pandas.DataFrame: Single-row DataFrame with all required features
        
    Note:
        - Fills missing features with 0 (safe default for most features)
        - Must include all features from FEATURES list
        - Feature order doesn't matter (XGBoost uses column names)
    """

    features = {col: player_row.get(col, 0) for col in FEATURES}
    features['IS_HOME'] = 1 if is_home else 0
    features['DAYS_REST'] = days_rest
    features['IS_B2B'] = 1 if days_rest == 1 else 0
    features['MISSING_USAGE'] = missing_usage
    return pd.DataFrame([features])

def scout_player(df_history, models):
    """
    Interactive deep-dive analysis of a specific player.
    
    Args:
        df_history (pandas.DataFrame): Historical training data
        models (dict): Loaded prediction models
        
    User Flow:
        1. Select date (today/tomorrow)
        2. Enter player name (fuzzy search)
        3. View predictions for all stat markets
        4. See injury impact on team
        5. Compare to PrizePicks lines
        
    Output:
        Formatted table showing:
            - Market (PTS, REB, AST, etc.)
            - AI Projection
            - PrizePicks Line
            - Recommendation (OVER/UNDER)
    """

    print("\n🔎 --- PLAYER SCOUT ---")
    d_choice = input("Select Date (1=Today, 2=Tomorrow): ").strip()
    offset = 1 if d_choice == '2' else 0
    todays_teams = get_games(date_offset=offset, require_scheduled=True)
    if not todays_teams:
        print("❌ No scheduled games found."); return

    scouting = True
    while scouting:
        print("\n(Type '0' to return to Main Menu)")
        query = input("Enter player name: ").strip().lower()
        if query == '0':
            print("Returning to menu...")
            break

        matches = df_history[df_history['PLAYER_NAME'].str.lower().str.contains(query)]
        if matches.empty:
            print("❌ Player not found in database.")
        else:
            unique_players = matches[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
            if len(unique_players) > 1:
                print(unique_players.to_string(index=False))
                try:
                    pid = int(input("Enter PLAYER_ID: "))
                    matches = matches[matches['PLAYER_ID'] == pid]
                except: continue

            print("...Fetching live PrizePicks lines")
            live_lines = fetch_current_lines_dict()
            norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

            player_data = matches.sort_values('GAME_DATE').iloc[-1]
            name = player_data['PLAYER_NAME']
            team_id = player_data['TEAM_ID']
            if team_id not in todays_teams:
                print(f"⚠️ {name} is not playing today/tomorrow."); continue

            is_home = todays_teams.get(team_id, {'is_home': 0})['is_home']
            
            team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()
            missing_usage_today = 0.0
            for teammate_id in team_players:
                t_rows = df_history[df_history['PLAYER_ID'] == teammate_id].sort_values('GAME_DATE')
                if t_rows.empty: continue
                last_t_row = t_rows.iloc[-1]
                t_name = last_t_row['PLAYER_NAME']
                if get_player_status(t_name) == 'OUT':
                    usage = last_t_row.get('USAGE_RATE_Season', 0)
                    if usage > 15: missing_usage_today += usage
            
            print(f"\n📊 SCOUTING REPORT: {name}")
            print(f"🚑 Team Injury Impact: {missing_usage_today:.1f}% Missing Usage")
            print(f"{'MARKET':<8} | {'PROJ':<8} | {'LINE':<8} | {'RECOMMENDATION':<20}")
            print("-" * 60)
            
            input_row = prepare_features(player_data, is_home=is_home, missing_usage=missing_usage_today)
            
            for target in TARGETS:
                if target in models:
                    # FIX #17: Filter features to prevent leakage (same as train.py)
                    model_features = [f for f in models[target].feature_names_in_ if target not in f]
                    valid_input = input_row.reindex(columns=model_features, fill_value=0)
                    pred = float(models[target].predict(valid_input)[0])
                    
                    line = norm_lines.get(normalize_name(name), {}).get(target)
                    indicator = get_betting_indicator(pred, line)
                    line_str = f"{line:.2f}" if line else "N/A"
                    print(f"{target:<8} : {pred:<8.2f} | {line_str:<8} | {indicator}")
        
        if input("\nScout another? (y/n): ").lower() != 'y': scouting = False

def scan_all(df_history, models, is_tomorrow=False):
    """
    Batch analysis of all games and players for a given date.
    
    This is the main prediction engine that:
        1. Fetches today's/tomorrow's game schedule
        2. Gets current PrizePicks lines
        3. Calculates missing usage for each team (injuries)
        4. Makes predictions for every active player
        5. Applies correlation constraints (ensures PRA ≥ PTS, etc.)
        6. Filters predictions by model quality
        7. Ranks opportunities by edge percentage
        
    Args:
        df_history (pandas.DataFrame): Historical training data
        models (dict): Loaded prediction models
        is_tomorrow (bool): True to scan tomorrow's games, False for today
        
    Output:
        - Prints top 10 OVERS (highest edge %)
        - Prints top 10 UNDERS (highest edge %)
        - Saves full analysis to CSV (all players, all stats)
        
    Key Features:
        - **Real-time injury integration**: Calculates team's missing usage
        - **Correlation constraints**: Prevents illogical predictions
        - **Model quality filtering**: Hides weak models (BLK, STL, SB, TOV)
        - **Confidence tiers**: Labels bets as ELITE/STRONG/DECENT
        
    Example Output:
        🔥 TOP 10 OVERS (Highest Value)
         TIER         | PLAYER              | STAT  | AI vs PP       | EDGE %
         ⭐ ELITE     | LeBron James        | PTS   |  28.50 vs 25.5 |   11.8%
         ⭐ ELITE     | Luka Doncic         | PA    |  42.30 vs 38.5 |    9.9%
    """

    offset = 1 if is_tomorrow else 0
    todays_teams = get_games(date_offset=offset, require_scheduled=True)
    if not todays_teams:
        print("❌ No scheduled games found."); return

    print("\n🚀 Fetching Live PrizePicks Lines...")
    live_lines = fetch_current_lines_dict() 
    norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

    print(f"🚀 Scanning Markets...")
    
    best_bets = [] 
    all_projections = []

    for team_id, info in todays_teams.items():
        team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()
        
        missing_usage_today = 0.0
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            p_name = last_row['PLAYER_NAME']
            
            status = get_player_status(p_name)
            if status == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15:
                    missing_usage_today += usage

        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            player_name = last_row['PLAYER_NAME']
            
            if get_player_status(player_name) == 'OUT':
                continue

            input_row = prepare_features(last_row, is_home=info['is_home'], missing_usage=missing_usage_today)
            
            # Store all predictions for this player to apply correlation constraints
            player_predictions = {}
            
            for target, model in models.items():
                # FIX #17: Filter features to prevent leakage
                model_features = [f for f in model.feature_names_in_ if target not in f]
                valid_input = input_row.reindex(columns=model_features, fill_value=0)
                proj = float(model.predict(valid_input)[0])
                player_predictions[target] = proj
            
            # FIX #18: Apply correlation constraints to ensure logical consistency
            # PRA must be >= PTS, REB, AST individually
            # PR must be >= PTS and REB
            # PA must be >= PTS and AST
            # RA must be >= REB and AST
            # SB must be >= STL and BLK
            
            if 'PRA' in player_predictions and 'PTS' in player_predictions:
                pts, reb, ast = player_predictions.get('PTS', 0), player_predictions.get('REB', 0), player_predictions.get('AST', 0)
                calculated_pra = pts + reb + ast
                # Use the higher of model prediction vs calculated sum
                player_predictions['PRA'] = max(player_predictions['PRA'], calculated_pra)
            
            if 'PR' in player_predictions and 'PTS' in player_predictions:
                calculated_pr = player_predictions.get('PTS', 0) + player_predictions.get('REB', 0)
                player_predictions['PR'] = max(player_predictions['PR'], calculated_pr)
            
            if 'PA' in player_predictions and 'PTS' in player_predictions:
                calculated_pa = player_predictions.get('PTS', 0) + player_predictions.get('AST', 0)
                player_predictions['PA'] = max(player_predictions['PA'], calculated_pa)
            
            if 'RA' in player_predictions and 'REB' in player_predictions:
                calculated_ra = player_predictions.get('REB', 0) + player_predictions.get('AST', 0)
                player_predictions['RA'] = max(player_predictions['RA'], calculated_ra)
            
            if 'SB' in player_predictions and 'STL' in player_predictions:
                calculated_sb = player_predictions.get('STL', 0) + player_predictions.get('BLK', 0)
                player_predictions['SB'] = max(player_predictions['SB'], calculated_sb)
            
            # Now create recommendations with constrained predictions
            for target, proj in player_predictions.items():
                
                line = norm_lines.get(normalize_name(player_name), {}).get(target)
                rec = get_betting_indicator(proj, line)
                
                all_projections.append({
                    'REC': rec, 'NAME': player_name, 'TARGET': target,
                    'AI': round(proj, 2), 'PP': round(line, 2) if line else 0, 
                    'EDGE': round(proj - line, 2) if line else 0
                })

                if line is not None and line > 0:
                    edge = proj - line
                    pct_edge = (edge / line) * 100
                    
                    best_bets.append({
                        'REC': rec, 'NAME': player_name, 'TARGET': target,
                        'AI': round(proj, 2), 'PP': round(line, 2), 
                        'EDGE': edge, 'PCT_EDGE': pct_edge
                    })
            
    if best_bets:
        # FIX #19: Filter bets by model quality (only show predictions from reliable models)
        # Tier 1: Elite models (>85% directional accuracy)
        elite_models = {'PTS', 'FGM', 'PA', 'PR', 'PRA'}
        # Tier 2: Strong models (80-85%)
        strong_models = {'FG3A', 'FGA'}
        # Tier 3: Decent models (75-80%)
        decent_models = {'FG3M', 'FTA'}
        
        # Mark bet quality
        for bet in best_bets:
            if bet['TARGET'] in elite_models:
                bet['TIER'] = '⭐ ELITE'
                bet['CONFIDENCE'] = 'HIGH'
            elif bet['TARGET'] in strong_models:
                bet['TIER'] = '✓ STRONG'
                bet['CONFIDENCE'] = 'MEDIUM'
            elif bet['TARGET'] in decent_models:
                bet['TIER'] = '~ DECENT'
                bet['CONFIDENCE'] = 'LOW'
            else:
                bet['TIER'] = '⚠ WEAK'
                bet['CONFIDENCE'] = 'AVOID'
        
        # Filter out WEAK model bets (optional - can remove this line to see all)
        quality_bets = [b for b in best_bets if b['CONFIDENCE'] != 'AVOID']
        
        top_overs = sorted([b for b in quality_bets if b['EDGE'] > 0], key=lambda x: x['PCT_EDGE'], reverse=True)[:10]
        top_unders = sorted([b for b in quality_bets if b['EDGE'] < 0], key=lambda x: x['PCT_EDGE'])[:10]
        
        print("\n🔥 TOP 10 OVERS (Highest Value)")
        print(f" {'TIER':<12} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15} | {'EDGE %':<8}")
        print("-" * 85)
        for bet in top_overs:
            print(f" {bet['TIER']:<12} | {bet['NAME']:<20} | {bet['TARGET']:<5} | {bet['AI']:>6.2f} vs {bet['PP']:>6.2f} | {bet['PCT_EDGE']:>6.1f}%")
        
        print("\n❄️ TOP 10 UNDERS (Lowest Value)")
        print(f" {'TIER':<12} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15} | {'EDGE %':<8}")
        print("-" * 85)
        for bet in top_unders:
            print(f" {bet['TIER']:<12} | {bet['NAME']:<20} | {bet['TARGET']:<5} | {bet['AI']:>6.2f} vs {bet['PP']:>6.2f} | {bet['PCT_EDGE']:>6.1f}%")
        
        save_path = TOMORROW_SCAN_FILE if is_tomorrow else TODAY_SCAN_FILE
        pd.DataFrame(all_projections).to_csv(save_path, index=False)
        print(f"\n✅ Full analysis ({len(all_projections)} rows) saved to {save_path}")
    else:
        print("\n⚠️ No active lines found.")
    input("\nPress Enter to continue...")

def main():
    """
    Main entry point - Interactive menu system.
    
    Menu Options:
        1. Scan TODAY'S Games - Immediate betting opportunities
        2. Scan TOMORROW'S Games - Prepare for next day
        3. Grade Results - Check accuracy of past predictions
        4. Scout Specific Player - Deep dive on one player
        0. Exit
        
    Workflow:
        1. Load historical data and trained models on startup
        2. Display interactive menu
        3. Execute selected function
        4. Loop until user exits
        
    Note:
        Injury data is loaded once at startup and cached for session
    """

    print("...Initializing System")
    df = load_data()
    models = load_models()
    if df is None or not models:
        print("❌ Setup failed."); return

    while True:
        print("\n" + "="*30 + "\n   🤖 NBA AI SCANNER v2.7\n" + "="*30)
        print("1. 🚀 Scan TODAY'S Games")
        print("2. 🔮 Scan TOMORROW'S Games")
        print("3. 📝 Grade Results")
        print("4. 🔎 Scout Specific Player")
        print("0. 🚪 Exit")
        choice = input("\nSelect: ").strip()
        if choice == '1': scan_all(df, models, is_tomorrow=False)
        elif choice == '2': scan_all(df, models, is_tomorrow=True)
        elif choice == '3': grade_results() 
        elif choice == '4': scout_player(df, models)
        elif choice == '0': break

if __name__ == "__main__":
    main()