"""
NBA Props Scanner - AI-Powered Prediction System

Scans upcoming NBA games, generates player performance predictions using
trained XGBoost models, and identifies profitable betting opportunities
by comparing predictions against PrizePicks lines.

Usage:
    $ python3 -m src.sports.nba.scanner
"""

import pandas as pd
import xgboost as xgb
import os
import time
import warnings
import unicodedata
import re
from datetime import datetime, timedelta
import requests

from nba_api.stats.endpoints import ScoreboardV2, LeagueGameLog

from src.core.odds_providers.prizepicks import PrizePicksClient
from src.sports.nba.config   import STAT_MAP, MODEL_QUALITY, ACTIVE_TARGETS
from src.sports.nba.injuries import get_injury_report

# --- CONFIGURATION ---
# scanner.py lives at src/sports/nba/scanner.py → root is 4 levels up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'nba')
DATA_FILE = os.path.join(BASE_DIR, 'data',   'nba', 'processed', 'training_dataset.csv')
PROJ_DIR  = os.path.join(BASE_DIR, 'data',   'nba', 'projections')

TODAY_SCAN_FILE    = os.path.join(PROJ_DIR, 'todays_automated_analysis.csv')
TOMORROW_SCAN_FILE = os.path.join(PROJ_DIR, 'tomorrows_automated_analysis.csv')
ACCURACY_LOG_FILE  = os.path.join(PROJ_DIR, 'accuracy_log.csv')

warnings.filterwarnings('ignore')

# Load injuries once at startup
print("...Loading Injury Report")
INJURY_DATA = get_injury_report()

TARGETS = ACTIVE_TARGETS

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

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.extend([f'{combo}_L5', f'{combo}_L20', f'{combo}_Season'])

for stat in ['PTS', 'REB', 'AST', 'FG3M', 'FGA', 'BLK', 'STL', 'TOV', 'FGM', 'FTM', 'FTA']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.append(f'OPP_{combo}_ALLOWED')


def normalize_name(name):
    if not name: return ""
    n = unicodedata.normalize('NFKD', name)
    clean = "".join([c for c in n if not unicodedata.combining(c)])
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)
    for s in ['Jr', 'Sr', 'III', 'II', 'IV']:
        clean = clean.replace(f" {s}", "")
    return " ".join(clean.lower().split())


def get_player_status(name):
    norm_name = normalize_name(name)
    for injured_name, status in INJURY_DATA.items():
        if normalize_name(injured_name) == norm_name:
            return status
    return "Active"


def get_betting_indicator(proj, line):
    if line is None or line <= 0: return "⚪ NO LINE"
    diff = proj - line
    if diff > 0: return f"🟢 OVER (+{diff:.2f})"
    else:        return f"🔴 UNDER ({diff:.2f})"


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


# stats.nba.com is often slow/unreliable - use longer timeout + retries
NBA_API_TIMEOUT = 90
NBA_API_RETRIES = 3
NBA_API_RETRY_DELAY = 8


def _fetch_scoreboard(game_date, retries=NBA_API_RETRIES):
    """Fetch scoreboard with retries and extended timeout (stats.nba.com is flaky)."""
    for attempt in range(retries):
        try:
            board = ScoreboardV2(
                game_date=game_date, league_id='00', day_offset=0, timeout=NBA_API_TIMEOUT
            )
            return board.game_header.get_data_frame()
        except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout, ConnectionError, OSError) as e:
            if attempt < retries - 1:
                print(f"   ⏳ Timeout (attempt {attempt + 1}/{retries}), retrying in {NBA_API_RETRY_DELAY}s...")
                time.sleep(NBA_API_RETRY_DELAY)
            else:
                raise
    return None


def get_games(date_offset=0, require_scheduled=True, max_days_forward=7):
    """
    Fetch games for a specific date, with fallback to search forward.
    
    Args:
        date_offset (int): Days from today (0=today, 1=tomorrow, etc.)
        require_scheduled (bool): Only return games not yet started
        max_days_forward (int): Maximum days to search forward if no games found
        
    Returns:
        tuple: (team_map, actual_date_used)
            team_map: dict of {team_id: {'is_home': bool, 'opp': opponent_id}}
            actual_date_used: str of date where games were found
            
    Workflow:
        1. Try the requested date (today, tomorrow, etc.)
        2. If no games found, search forward day-by-day
        3. Stop at first date with games (up to max_days_forward)
        4. Return games + the date they were found on
        
    Example:
        # Today is Monday, no games today/tomorrow
        # Thursday has games
        team_map, date = get_games(date_offset=0)
        # Returns: (thursday_games, '2026-02-20')
        # Prints: "No games today. Found games on 2026-02-20 (Thursday)"
    """
    # Try the initially requested date
    initial_date = datetime.now() + timedelta(days=date_offset)
    target_date = initial_date.strftime('%Y-%m-%d')
    
    print(f"...Checking for games on {target_date}")
    
    try:
        games = _fetch_scoreboard(target_date)
        
        if not games.empty:
            if require_scheduled:
                scheduled_games = games[games['GAME_STATUS_ID'] == 1]
                if not scheduled_games.empty:
                    print(f"✅ Found {len(scheduled_games)} scheduled games on {target_date}")
                    return _build_team_map(scheduled_games), target_date
            else:
                print(f"✅ Found {len(games)} games on {target_date}")
                return _build_team_map(games), target_date
        
        # No games on requested date - search forward
        print(f"   No games on {target_date}. Searching forward...")
        
        for days_ahead in range(1, max_days_forward + 1):
            search_date = (initial_date + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            # Show progress every 2 days
            if days_ahead % 2 == 0 or days_ahead == 1:
                print(f"   Checking {search_date}...", end='\r')
            
            try:
                games = _fetch_scoreboard(search_date)
                
                if not games.empty:
                    if require_scheduled:
                        scheduled_games = games[games['GAME_STATUS_ID'] == 1]
                        if not scheduled_games.empty:
                            # Found games!
                            day_name = (initial_date + timedelta(days=days_ahead)).strftime('%A')
                            print(f"\n✅ Found {len(scheduled_games)} games on {search_date} ({day_name})")
                            print(f"   📅 That's {days_ahead} day{'s' if days_ahead > 1 else ''} from now")
                            return _build_team_map(scheduled_games), search_date
                    else:
                        if not games.empty:
                            day_name = (initial_date + timedelta(days=days_ahead)).strftime('%A')
                            print(f"\n✅ Found {len(games)} games on {search_date} ({day_name})")
                            return _build_team_map(games), search_date
            
            except Exception as e:
                # Skip this date if error
                continue
        
        # No games found in entire search window
        print(f"\n❌ No scheduled games found in the next {max_days_forward} days")
        return {}, None
        
    except Exception as e:
        print(f"Error fetching games: {e}")
        return {}, None


def _build_team_map(games_df):
    """
    Helper function to build team mapping from games DataFrame.
    
    Args:
        games_df: DataFrame with HOME_TEAM_ID and VISITOR_TEAM_ID columns
        
    Returns:
        dict: {team_id: {'is_home': bool, 'opp': opponent_id}}
    """
    team_map = {}
    for _, g in games_df.iterrows():
        team_map[g['HOME_TEAM_ID']] = {
            'is_home': True,
            'opp': g['VISITOR_TEAM_ID']
        }
        team_map[g['VISITOR_TEAM_ID']] = {
            'is_home': False,
            'opp': g['HOME_TEAM_ID']
        }
    return team_map


# ============================================================================
# UPDATED scan_all FUNCTION (to use the new return format)
# ============================================================================

def scan_all(df_history, models, is_tomorrow=False):
    """
    Batch analysis of all games, with automatic forward search.
    
    Changes:
        - Now handles the (team_map, date) tuple from get_games
        - Shows which date was actually used for scanning
        - Updates save filename if using future date
    """
    offset = 1 if is_tomorrow else 0
    
    # NEW: get_games now returns (team_map, actual_date)
    todays_teams, actual_date = get_games(
        date_offset=offset,
        require_scheduled=True,
        max_days_forward=7
    )
    
    if not todays_teams:
        print("❌ No scheduled games found in the next 7 days.")
        input("\nPress Enter to continue...")
        return
    
    # Show what date we're actually scanning
    if actual_date:
        scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
        day_name = scan_date_obj.strftime('%A, %B %d, %Y')
        print(f"\n📅 Scanning games for: {day_name}")
    
    print("\n🚀 Fetching Live PrizePicks Lines...")
    pp_client = PrizePicksClient(stat_map=STAT_MAP)
    live_lines = pp_client.fetch_lines_dict(league_filter='NBA')

    # ✅ DEBUG: Show what we got
    if live_lines:
        print(f"   ✅ Loaded {len(live_lines)} players from PrizePicks")
        sample_player = list(live_lines.keys())[0]
        sample_stats = live_lines[sample_player]
        print(f"   Example: {sample_player} - {list(sample_stats.keys())[:5]}")
    else:
        print("   ⚠️  Got empty response from PrizePicks")

    norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

    print("🚀 Scanning Markets...")
    best_bets = []
    all_projections = []

    for team_id, info in todays_teams.items():
        team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()

        # Calculate missing usage (injured players)
        missing_usage_today = 0.0
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            if get_player_status(last_row['PLAYER_NAME']) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15:
                    missing_usage_today += usage

        # Generate predictions for each player
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            player_name = last_row['PLAYER_NAME']

            if get_player_status(player_name) == 'OUT':
                continue

            input_row = prepare_features(
                last_row,
                is_home=info['is_home'],
                missing_usage=missing_usage_today
            )

            player_predictions = {}

            for target, model in models.items():
                model_features = [f for f in model.feature_names_in_ if target not in f]
                valid_input = input_row.reindex(columns=model_features, fill_value=0)
                proj = float(model.predict(valid_input)[0])
                player_predictions[target] = proj

            # Apply correlation constraints
            if 'PRA' in player_predictions:
                pts = player_predictions.get('PTS', 0)
                reb = player_predictions.get('REB', 0)
                ast = player_predictions.get('AST', 0)
                player_predictions['PRA'] = max(player_predictions['PRA'], pts + reb + ast)

            if 'PR' in player_predictions:
                player_predictions['PR'] = max(
                    player_predictions['PR'],
                    player_predictions.get('PTS', 0) + player_predictions.get('REB', 0)
                )

            if 'PA' in player_predictions:
                player_predictions['PA'] = max(
                    player_predictions['PA'],
                    player_predictions.get('PTS', 0) + player_predictions.get('AST', 0)
                )

            if 'RA' in player_predictions:
                player_predictions['RA'] = max(
                    player_predictions['RA'],
                    player_predictions.get('REB', 0) + player_predictions.get('AST', 0)
                )

            if 'SB' in player_predictions:
                player_predictions['SB'] = max(
                    player_predictions['SB'],
                    player_predictions.get('STL', 0) + player_predictions.get('BLK', 0)
                )

            # Create recommendations
            for target, proj in player_predictions.items():
                line = norm_lines.get(normalize_name(player_name), {}).get(target)
                rec = get_betting_indicator(proj, line)

                all_projections.append({
                    'REC': rec,
                    'NAME': player_name,
                    'TARGET': target,
                    'AI': round(proj, 2),
                    'PP': round(line, 2) if line else 0,
                    'EDGE': round(proj - line, 2) if line else 0
                })

                if line is not None and line > 0:
                    edge = proj - line
                    pct_edge = (edge / line) * 100

                    tier_info = MODEL_QUALITY.get(target, {})

                    best_bets.append({
                        'REC': rec,
                        'NAME': player_name,
                        'TARGET': target,
                        'AI': round(proj, 2),
                        'PP': round(line, 2),
                        'EDGE': edge,
                        'PCT_EDGE': pct_edge,
                        'TIER': tier_info.get('emoji', '~') + ' ' + tier_info.get('tier', 'UNKNOWN'),
                        'THRESHOLD': tier_info.get('threshold', 2.5)
                    })

    # Display results
    if best_bets:
        # ✅ DEDUPLICATE: Remove duplicate player+stat+line combinations
        seen = set()
        deduped_bets = []
        
        for bet in best_bets:
            # Create unique key: player + stat + line
            key = (bet['NAME'], bet['TARGET'], bet['PP'])
            if key not in seen:
                seen.add(key)
                deduped_bets.append(bet)
        
        print(f"   🧹 Removed {len(best_bets) - len(deduped_bets)} duplicate entries")
        
        # Sort by tier and edge
        tier_order = {'ELITE': 0, 'STRONG': 1, 'DECENT': 2, 'RISKY': 3, 'AVOID': 4}
        deduped_bets.sort(key=lambda x: (tier_order.get(x['TIER'], 99), -abs(x['PCT_EDGE'])))
        
        # Take top 10 after deduplication
        top_overs  = [b for b in deduped_bets if b['EDGE'] > 0][:10]
        top_unders = [b for b in deduped_bets if b['EDGE'] < 0][:10]

        print("\n🔥 TOP 10 OVERS (Highest Value)")
        print(f" {'TIER':<12} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15} | {'EDGE %':<8}")
        print("-" * 85)
        for bet in top_overs:
            print(f" {bet['TIER']:<12} | {bet['NAME']:<20} | {bet['TARGET']:<5} | "
                  f"{bet['AI']:>6.2f} vs {bet['PP']:>6.2f} | {bet['PCT_EDGE']:>6.1f}%")

        print("\n❄️ TOP 10 UNDERS (Lowest Value)")
        print(f" {'TIER':<12} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15} | {'EDGE %':<8}")
        print("-" * 85)
        for bet in top_unders:
            print(f" {bet['TIER']:<12} | {bet['NAME']:<20} | {bet['TARGET']:<5} | "
                  f"{bet['AI']:>6.2f} vs {bet['PP']:>6.2f} | {bet['PCT_EDGE']:>6.1f}%")

        # Determine save filename based on actual date used
        if actual_date:
            save_path = os.path.join(PROJ_DIR, f"scan_{actual_date}.csv")
        else:
            save_path = TOMORROW_SCAN_FILE if is_tomorrow else TODAY_SCAN_FILE

        pd.DataFrame(all_projections).to_csv(save_path, index=False)
        print(f"\n✅ Full analysis ({len(all_projections)} rows) saved to {save_path}")
    else:
        print("\n⚠️ No active lines found.")

    input("\nPress Enter to continue...")


def get_actual_stats_for_grading(target_date_obj):
    date_str = target_date_obj.strftime('%m/%d/%Y')
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
            player_or_team_abbreviation='P',
            timeout=NBA_API_TIMEOUT
        )
        stats_frames = log.get_data_frames()
        if not stats_frames or stats_frames[0].empty:
            print("⚠️ No completed games found for this date.")
            return {}
        stats = stats_frames[0]
        print(f"   ✅ Loaded stats for {len(stats)} players.")
        player_stats = {}
        for _, row in stats.iterrows():
            name = normalize_name(row['PLAYER_NAME'])
            pts, reb, ast = row['PTS'], row['REB'], row['AST']
            blk, stl, tov = row['BLK'], row['STL'], row['TOV']
            fg3m, fgm, fga = row['FG3M'], row['FGM'], row['FGA']
            ftm, fta = row['FTM'], row['FTA']
            player_stats[name] = {
                'PTS': pts, 'REB': reb, 'AST': ast,
                'FG3M': fg3m, 'BLK': blk, 'STL': stl, 'TOV': tov,
                'FGM': fgm, 'FGA': fga, 'FTM': ftm, 'FTA': fta,
                'PRA': pts + reb + ast, 'PR': pts + reb, 'PA': pts + ast,
                'RA': reb + ast, 'SB': stl + blk
            }
        return player_stats
    except Exception as e:
        print(f"API Error: {e}")
        return {}


def prepare_features(player_row, is_home=0, days_rest=2, missing_usage=0):
    features = {col: player_row.get(col, 0) for col in FEATURES}
    features['IS_HOME']       = 1 if is_home else 0
    features['DAYS_REST']     = days_rest
    features['IS_B2B']        = 1 if days_rest == 1 else 0
    features['MISSING_USAGE'] = missing_usage
    return pd.DataFrame([features])


def grade_results():
    print("\n📅 GRADING OPTIONS:")
    print("1. Grade TODAY'S Games")
    print("2. Grade YESTERDAY'S Games")
    choice = input("Select (1/2): ").strip()
    target_date = datetime.now() - timedelta(days=1) if choice == '2' else datetime.now()
    print(f"\n📅 Grading {target_date.strftime('%Y-%m-%d')}")

    if not os.path.exists(TODAY_SCAN_FILE):
        print("❌ No scan file found. Run Option 1 first.")
        return

    try:
        df_preds = pd.read_csv(TODAY_SCAN_FILE)
    except:
        print("❌ Error reading prediction file.")
        return

    actuals = get_actual_stats_for_grading(target_date)
    if not actuals:
        print("\n❌ Stats unavailable. Games may not be finished.")
        return

    print("\n" + "="*65)
    print("📝 RESULTS ANALYSIS")
    print("="*65)

    results      = []
    total_graded = 0
    correct_picks = 0

    for _, row in df_preds.iterrows():
        if row['PP'] == 0: continue
        name      = normalize_name(row['NAME'])
        target    = row['TARGET']
        line      = float(row['PP'])
        rec_text  = row['REC']
        if name not in actuals: continue
        actual_val = actuals[name].get(target)
        if actual_val is None: continue
        pick_type = "NONE"
        if "OVER"  in rec_text: pick_type = "OVER"
        elif "UNDER" in rec_text: pick_type = "UNDER"
        else: continue
        is_win = False
        margin = 0
        if pick_type == "OVER":
            margin = actual_val - line
            if actual_val > line:  is_win = True
            elif actual_val == line: is_win = "PUSH"
        elif pick_type == "UNDER":
            margin = line - actual_val
            if actual_val < line:  is_win = True
            elif actual_val == line: is_win = "PUSH"
        if is_win != "PUSH":
            total_graded += 1
            if is_win: correct_picks += 1
            results.append({
                'Player': row['NAME'], 'Stat': target, 'Pick': pick_type,
                'Line': line, 'Actual': actual_val, 'Margin': margin, 'Win': is_win
            })

    if total_graded == 0:
        print("⚠️ Predictions found, but no matching player stats (check date?).")
        return

    sorted_results = sorted(results, key=lambda x: x['Margin'], reverse=True)
    top_wins     = [r for r in sorted_results if r['Win']][:5]
    worst_losses = sorted([r for r in sorted_results if not r['Win']], key=lambda x: x['Margin'])[:5]

    print("\n🏆 TOP 5 BEST WINS")
    print(f"{'PLAYER':<20} | {'STAT':<5} | {'PICK':<5} | {'LINE':<5} | {'ACTUAL':<6} | MARGIN")
    print("-" * 70)
    for r in top_wins:
        print(f"{r['Player']:<20} | {r['Stat']:<5} | {r['Pick']:<5} | {r['Line']:<5} | {r['Actual']:<6} | 🟢 +{r['Margin']:.1f}")

    print("\n💀 TOP 5 WORST LOSSES")
    print(f"{'PLAYER':<20} | {'STAT':<5} | {'PICK':<5} | {'LINE':<5} | {'ACTUAL':<6} | MARGIN")
    print("-" * 70)
    for r in worst_losses:
        print(f"{r['Player']:<20} | {r['Stat']:<5} | {r['Pick']:<5} | {r['Line']:<5} | {r['Actual']:<6} | 🔴 {r['Margin']:.1f}")

    accuracy = (correct_picks / total_graded) * 100
    print("-" * 70)
    print(f"📊 FINAL ACCURACY: {accuracy:.1f}% ({correct_picks}/{total_graded})")

    os.makedirs(PROJ_DIR, exist_ok=True)
    log_exists = os.path.exists(ACCURACY_LOG_FILE)
    with open(ACCURACY_LOG_FILE, 'a') as f:
        if not log_exists:
            f.write("Date,Graded,Correct,Accuracy,Best_Win_Margin\n")
        f.write(f"{target_date.strftime('%Y-%m-%d')},{total_graded},{correct_picks},{accuracy:.2f},{top_wins[0]['Margin'] if top_wins else 0}\n")
    print(f"✅ Results logged to {ACCURACY_LOG_FILE}")
    input("\nPress Enter to continue...")


def scout_player(df_history, models):
    print("\n🔎 --- PLAYER SCOUT ---")
    d_choice = input("Select Start Date (1=Today, 2=Tomorrow): ").strip()
    offset = 1 if d_choice == '2' else 0
    
    # Use the improved get_games logic to find the next available games
    todays_teams, actual_date = get_games(
        date_offset=offset, 
        require_scheduled=True, 
        max_days_forward=7
    )
    
    if not todays_teams:
        print("❌ No scheduled games found in the next 7 days.")
        return

    # Display the date being scouted
    scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
    print(f"\n📅 Scouting for games on: {scan_date_obj.strftime('%A, %B %d, %Y')}")

    pp_client  = PrizePicksClient(stat_map=STAT_MAP)
    scouting   = True

    while scouting:
        print("\n(Type '0' to return to Main Menu)")
        query = input("Enter player name: ").strip().lower()
        if query == '0':
            break

        try:
            matches = df_history[df_history['PLAYER_NAME'].str.lower().str.contains(query)]
        except Exception as e:
            print(f"❌ Search error: {e}")
            continue

        if matches.empty:
            print(f"❌ No players found matching '{query}'.")
            continue

        unique_players = matches[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
        if len(unique_players) > 1:
            print(unique_players.to_string(index=False))
            try:
                pid = int(input("Enter PLAYER_ID: "))
                matches = matches[matches['PLAYER_ID'] == pid]
                if matches.empty:
                    print(f"❌ No data found for PLAYER_ID {pid}.")
                    continue
            except ValueError:
                print("❌ Invalid PLAYER_ID.")
                continue

        # Fetch lines for the identified date
        print(f"...Fetching PrizePicks lines")
        live_lines = pp_client.fetch_lines_dict(league_filter='NBA')
        norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

        try:
            player_data = matches.sort_values('GAME_DATE').iloc[-1]
        except IndexError:
            print("❌ No recent history found for this player.")
            continue

        name    = player_data['PLAYER_NAME']
        team_id = player_data['TEAM_ID']
        
        # Check if the player's team is in the team_map for the 'actual_date'
        if team_id not in todays_teams:
            print(f"⚠️ {name} is not scheduled to play on {actual_date}.")
            continue

        is_home = todays_teams[team_id]['is_home']

        # Calculate injury impact for the target date
        team_players = df_history[df_history['TEAM_ID'] == team_id]['PLAYER_ID'].unique()
        missing_usage_today = 0.0
        for pid in team_players:
            p_rows = df_history[df_history['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            if get_player_status(last_row['PLAYER_NAME']) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15: missing_usage_today += usage

        print(f"\n📊 SCOUTING REPORT: {name} ({actual_date})")
        print(f"🚑 Team Injury Impact: {missing_usage_today:.1f}% Missing Usage")
        print(f"{'TIER':<6} | {'MARKET':<8} | {'PROJ':<8} | {'LINE':<8} | RECOMMENDATION")
        print("-" * 75)

        input_row = prepare_features(player_data, is_home=is_home, missing_usage=missing_usage_today)

        for target in TARGETS:
            if target in models:
                tier_emoji    = MODEL_QUALITY.get(target, {}).get('emoji', '?')
                model_features = [f for f in models[target].feature_names_in_ if target not in f]
                valid_input   = input_row.reindex(columns=model_features, fill_value=0)
                pred          = float(models[target].predict(valid_input)[0])
                line          = norm_lines.get(normalize_name(name), {}).get(target)
                indicator     = get_betting_indicator(pred, line)
                line_str      = f"{line:.2f}" if line else "N/A"
                print(f"{tier_emoji:<6} | {target:<8} : {pred:<8.2f} | {line_str:<8} | {indicator}")

        if input("\nScout another player? (y/n): ").lower() != 'y':
            scouting = False


def main():
    print("...Initializing System")
    df     = load_data()
    models = load_models()
    if df is None or not models:
        print("❌ Setup failed.")
        return

    while True:
        print("\n" + "="*30 + "\n   🤖 NBA AI SCANNER\n" + "="*30)
        print("1. 🚀 Scan TODAY'S Games")
        print("2. 🔮 Scan TOMORROW'S Games")
        print("3. 📝 Grade Results")
        print("4. 🔎 Scout Specific Player")
        print("0. 🚪 Exit")
        choice = input("\nSelect: ").strip()
        if choice == '1':   scan_all(df, models, is_tomorrow=False)
        elif choice == '2': scan_all(df, models, is_tomorrow=True)
        elif choice == '3': grade_results()
        elif choice == '4': scout_player(df, models)
        elif choice == '0': break


if __name__ == "__main__":
    main()