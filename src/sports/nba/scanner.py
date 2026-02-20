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
import sys
import time
import warnings
import unicodedata
import re
from datetime import datetime, timedelta
import requests

from nba_api.stats.endpoints import ScoreboardV2, LeagueGameLog

from src.core.odds_providers.prizepicks import PrizePicksClient
from src.sports.nba.config   import STAT_MAP, MODEL_QUALITY, ACTIVE_TARGETS, ABSORPTION_RATES
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

# Injury cache — refreshed before each scan (see refresh_injuries)
INJURY_DATA = {}


def refresh_injuries():
    """Fetch fresh injury report and update global cache. Call before each scan."""
    global INJURY_DATA
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
    'USAGE_VACUUM', 'STAR_COUNT',
    # Location splits + opponent context (synced with train.py)
    'PTS_LOC_MEAN', 'REB_LOC_MEAN', 'AST_LOC_MEAN', 'FG3M_LOC_MEAN', 'PRA_LOC_MEAN',
    'OPP_WIN_PCT', 'IS_VS_ELITE_TEAM'
]

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.extend([f'{combo}_L5', f'{combo}_L20', f'{combo}_Season'])

for stat in ['PTS', 'REB', 'AST', 'FG3M', 'FGA', 'BLK', 'STL', 'TOV', 'FGM', 'FTM', 'FTA']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')
    FEATURES.append(f'OPP_{stat}_ALLOWED_DIFF')  # DvP Diff (synced with train.py)

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.append(f'OPP_{combo}_ALLOWED')
    FEATURES.append(f'OPP_{combo}_ALLOWED_DIFF')  # DvP Diff (synced with train.py)


def normalize_name(name):
    if not name: return ""
    n = unicodedata.normalize('NFKD', name)
    clean = "".join([c for c in n if not unicodedata.combining(c)])
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)
    for s in ['Jr', 'Sr', 'III', 'II', 'IV']:
        clean = clean.replace(f" {s}", "")
    return " ".join(clean.lower().split())


def get_player_status(name):
    """
    Check if player is on injury report. Uses exact match + last-name fallback
    (ESPN may use 'Patrick Williams' while our data has 'Patrick Williams II').
    """
    norm_name = normalize_name(name)
    for injured_name, status in INJURY_DATA.items():
        if normalize_name(injured_name) == norm_name:
            return status
    # Fallback: last-name match when exactly one injured player shares that last name
    parts = norm_name.split()
    if len(parts) >= 2:
        last_name = parts[-1]
        matches = [
            s for inj_name, s in INJURY_DATA.items()
            if normalize_name(inj_name).split()[-1] == last_name
        ]
        if len(matches) == 1:
            return matches[0]
    return "Active"


def get_betting_indicator(proj, line):
    if line is None or line <= 0: return "⚪ NO LINE"
    diff = proj - line
    if diff > 0: return f"🟢 OVER (+{diff:.2f})"
    else:        return f"🔴 UNDER ({diff:.2f})"


def load_data():
    if not os.path.exists(DATA_FILE): return None
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df


# ---------------------------------------------------------------------------
# DATA CACHING (Optimization)
# ---------------------------------------------------------------------------

def build_data_cache(df_history):
    """
    Pre-indexes the dataframe for O(1) lookups.
    Returns:
        latest_rows_map: {pid: row_dict}
        team_rosters_map: {team_id: [pid, pid, ...]}
    """
    print("...Building data cache for fast lookups")
    
    # Filter to current season only — prevents traded players from inflating
    # team rosters (e.g. Deandre Ayton still mapping to the Suns)
    season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
    current_season = df_history[season_col].max()
    df_current = df_history[df_history[season_col] == current_season]
    print(f"   Using season {current_season} ({df_current['PLAYER_ID'].nunique()} players)")
    
    # Sort by date, take latest row per player
    df_sorted = df_current.sort_values(['PLAYER_ID', 'GAME_DATE'])
    df_latest = df_sorted.drop_duplicates(subset=['PLAYER_ID'], keep='last')
    
    # 1. Latest Rows Map
    latest_rows_map = df_latest.set_index('PLAYER_ID').to_dict('index')
    
    # 2. Team Rosters Map (current season only)
    team_rosters_map = df_latest.groupby('TEAM_ID')['PLAYER_ID'].apply(list).to_dict()
    
    return latest_rows_map, team_rosters_map


def load_models():
    models = {}
    for target in TARGETS:
        path = os.path.join(MODEL_DIR, f"{target}_model.json")
        if os.path.exists(path):
            m = xgb.XGBRegressor()
            m.load_model(path)
            models[target] = m
    return models


# stats.nba.com is often slow/unreliable - use timeout + retries
NBA_API_TIMEOUT = 45   # Fail faster, rely on retries
NBA_API_RETRIES = 3
NBA_API_RETRY_DELAY = 5

# cdn.nba.com endpoints — fast and reliable
CDN_SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
CDN_SCHEDULE_URL   = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
CDN_TIMEOUT = 15

CDN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# Catch all request-related errors (timeout, connection, etc.)
_REQUEST_ERRORS = (
    requests.exceptions.Timeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectionError,
    requests.exceptions.RequestException,
    ConnectionError,
    OSError,
)


def _cdn_scoreboard_to_df(games_list):
    """Convert cdn.nba.com games list to a DataFrame matching ScoreboardV2 format."""
    rows = []
    for g in games_list:
        rows.append({
            'GAME_ID':         g.get('gameId', ''),
            'GAME_STATUS_ID':  g.get('gameStatus', 1),
            'HOME_TEAM_ID':    g.get('homeTeam', {}).get('teamId', 0),
            'VISITOR_TEAM_ID': g.get('awayTeam', {}).get('teamId', 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _fetch_scoreboard_cdn(game_date):
    """
    Fetch scoreboard via cdn.nba.com (primary, fast).

    - For today's games: uses the live scoreboard endpoint.
    - For any date: uses the full-season schedule endpoint.
    """
    target = datetime.strptime(game_date, '%Y-%m-%d').date()
    today  = datetime.now().date()

    # Fast path: today's games via the live scoreboard
    if target == today:
        try:
            print(f"   Fetching today's scoreboard...", flush=True)
            resp = requests.get(CDN_SCOREBOARD_URL, headers=CDN_HEADERS, timeout=CDN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            games = data.get('scoreboard', {}).get('games', [])
            df = _cdn_scoreboard_to_df(games)
            if not df.empty:
                return df
        except Exception as e:
            print(f"   CDN scoreboard failed, trying schedule...", flush=True)

    # Fallback / future dates: use the full schedule
    return _fetch_schedule_cdn(game_date)


def _fetch_schedule_cdn(game_date):
    """Fetch games for a specific date from the cdn.nba.com season schedule."""
    from datetime import datetime as _dt
    target = _dt.strptime(game_date, '%Y-%m-%d').date()
    try:
        print(f"   Fetching schedule...", flush=True)
        resp = requests.get(CDN_SCHEDULE_URL, headers=CDN_HEADERS, timeout=CDN_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        for gd in data.get('leagueSchedule', {}).get('gameDates', []):
            date_str = gd.get('gameDate', '')
            try:
                gd_date = _dt.strptime(date_str, '%m/%d/%Y %H:%M:%S').date()
            except ValueError:
                continue
            if gd_date == target:
                return _cdn_scoreboard_to_df(gd.get('games', []))
        # Date exists but no games scheduled
        return pd.DataFrame()
    except Exception as e:
        print(f"   CDN schedule failed", flush=True)
        return None


def _fetch_scoreboard_statsapi(game_date, retries=NBA_API_RETRIES):
    """Fetch scoreboard from stats.nba.com (fallback — often slow/unreliable)."""
    for attempt in range(retries):
        try:
            if attempt > 0:
                print(f"   Retry {attempt + 1}/{retries}...", flush=True)
            else:
                print(f"   Trying stats.nba.com fallback...", flush=True)
            sys.stdout.flush()
            board = ScoreboardV2(
                game_date=game_date, league_id='00', day_offset=0, timeout=NBA_API_TIMEOUT
            )
            return board.game_header.get_data_frame()
        except _REQUEST_ERRORS as e:
            if attempt < retries - 1:
                print(f"   Request failed, retrying in {NBA_API_RETRY_DELAY}s...", flush=True)
                time.sleep(NBA_API_RETRY_DELAY)
            else:
                raise
    return None


def _fetch_scoreboard(game_date, retries=NBA_API_RETRIES):
    """
    Fetch scoreboard with automatic failover:
      1. cdn.nba.com  (fast, reliable)
      2. stats.nba.com (legacy fallback)
    """
    # --- Primary: cdn.nba.com ---
    try:
        df = _fetch_scoreboard_cdn(game_date)
        if df is not None and not df.empty:
            print(f"   Got {len(df)} games from cdn.nba.com", flush=True)
            return df
        elif df is not None:
            # CDN responded but no games on this date
            return df
    except Exception as e:
        print(f"   cdn.nba.com failed, falling back...", flush=True)

    # --- Fallback: stats.nba.com ---
    try:
        return _fetch_scoreboard_statsapi(game_date, retries=retries)
    except Exception as e:
        print(f"   Both cdn.nba.com and stats.nba.com failed", flush=True)
        raise


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
                    print(f"Found {len(scheduled_games)} scheduled games on {target_date}")
                    return _build_team_map(scheduled_games), target_date
            else:
                print(f"Found {len(games)} games on {target_date}")
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
                            print(f"\nFound {len(scheduled_games)} games on {search_date} ({day_name})")
                            print(f"   📅 That's {days_ahead} day{'s' if days_ahead > 1 else ''} from now")
                            return _build_team_map(scheduled_games), search_date
                    else:
                        if not games.empty:
                            day_name = (initial_date + timedelta(days=days_ahead)).strftime('%A')
                            print(f"\nFound {len(games)} games on {search_date} ({day_name})")
                            return _build_team_map(games), search_date
            
            except Exception as e:
                # Skip this date if error
                continue
        
        # No games found in entire search window
        print(f"\nNo scheduled games found in the next {max_days_forward} days")
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

def scan_all(df_history, models, is_tomorrow=False, max_days_forward=7):
    """
    Batch analysis of all games, with automatic forward search (optional).
    
    OPTIMIZED: Uses pre-built dictionaries for O(1) lookups instead of O(N) DataFrame filtering.
    """
    refresh_injuries()
    offset = 1 if is_tomorrow else 0
    
    # NEW: get_games now returns (team_map, actual_date)
    todays_teams, actual_date = get_games(
        date_offset=offset,
        require_scheduled=True,
        max_days_forward=max_days_forward
    )
    
    if not todays_teams:
        print("No scheduled games found in the next 7 days.")
        input("\nPress Enter to continue...")
        return
    
    # Show what date we're actually scanning
    if actual_date:
        scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
        day_name = scan_date_obj.strftime('%A, %B %d, %Y')
        print(f"\n📅 Scanning games for: {day_name}")
    
    print("\nFetching PrizePicks lines...")
    pp_client = PrizePicksClient(stat_map=STAT_MAP)
    live_lines = pp_client.fetch_lines_dict(league_filter='NBA')

    if live_lines:
        print(f"   Loaded {len(live_lines)} players from PrizePicks")
    else:
        print("   Warning: empty response from PrizePicks")

    norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

    print("Building data cache & scanning markets...")
    # --- PRE-BUILD CACHE (O(N) once) ---
    latest_rows_map, team_rosters_map = build_data_cache(df_history)

    best_bets = []
    all_projections = []

    for team_id, info in todays_teams.items():
        # O(1) Lookup for roster
        team_players = team_rosters_map.get(team_id, [])

        # Calculate missing usage (injured players)
        missing_usage_today = 0.0
        for pid in team_players:
            # O(1) Lookup for player data
            last_row = latest_rows_map.get(pid)
            if not last_row: continue
            
            pname = last_row['PLAYER_NAME']
            if get_player_status(pname) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15:
                    missing_usage_today += usage

        # Generate predictions for each player
        for pid in team_players:
            last_row = latest_rows_map.get(pid)
            if not last_row: continue
            
            player_name = last_row['PLAYER_NAME']

            if get_player_status(player_name) == 'OUT':
                continue
                
            input_row = prepare_features(
                last_row,
                is_home=info['is_home'],
                missing_usage=missing_usage_today
            )

            player_predictions = {}
            features_ok = True

            for target, model in models.items():
                if not features_ok: break
                try:
                    # Filter for features actually used by THIS model
                    model_features = [f for f in model.feature_names_in_ if target not in f]
                    valid_input = input_row.reindex(columns=model_features, fill_value=0)
                    proj = float(model.predict(valid_input)[0])
                    player_predictions[target] = proj
                except Exception:
                    # Model mismatch or missing features
                    features_ok = False
            
            if not features_ok: continue

            # --- Injury redistribution adjustments ---
            # Pass the cache to avoid re-filtering inside
            inj_adj = _calculate_injury_adjustments_fast(
                latest_rows_map, team_players, pid
            )
            for stat, adj_val in inj_adj.items():
                if stat in player_predictions:
                    player_predictions[stat] += adj_val

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
                
                # Default empty pp/edge if no line
                pp_val   = round(line, 2) if line else 0
                edge_val = round(proj - line, 2) if line else 0

                all_projections.append({
                    'REC': rec,
                    'NAME': player_name,
                    'TARGET': target,
                    'AI': round(proj, 2),
                    'PP': pp_val,
                    'EDGE': edge_val
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
        
        print(f"   Removed {len(best_bets) - len(deduped_bets)} duplicate entries")
        
        # Sort by tier and edge
        tier_order = {'ELITE': 0, 'STRONG': 1, 'DECENT': 2, 'RISKY': 3, 'AVOID': 4}
        # Extract tier name for sorting (TIER is "⭐ ELITE", we need "ELITE")
        def _tier_key(b):
            for name in tier_order:
                if name in b.get('TIER', ''):
                    return tier_order[name]
            return 99
        deduped_bets.sort(key=lambda x: (_tier_key(x), -abs(x['PCT_EDGE'])))
        
        # Take top 10 with market diversity — ensure every stat type is covered
        def _diverse_top(bets, n=10):
            """Pick best bet per market first, then fill to n."""
            if len(bets) <= n:
                return bets[:n]
            by_market = {}
            for b in bets:
                t = b['TARGET']
                if t not in by_market or abs(b['PCT_EDGE']) > abs(by_market[t]['PCT_EDGE']):
                    by_market[t] = b
            result = list(by_market.values())
            seen = {(x['NAME'], x['TARGET'], x['PP']) for x in result}
            for b in bets:
                key = (b['NAME'], b['TARGET'], b['PP'])
                if key not in seen and len(result) < n:
                    result.append(b)
                    seen.add(key)
            result.sort(key=lambda x: (_tier_key(x), -abs(x['PCT_EDGE'])))
            return result[:n]

        top_overs  = _diverse_top([b for b in deduped_bets if b['EDGE'] > 0])
        top_unders = _diverse_top([b for b in deduped_bets if b['EDGE'] < 0])

        # Table format — consistent column widths, spacing, separator
        sep = "-" * 73
        print("\n  TOP 10 OVERS (Highest Value)")
        print()
        print(f" {'TIER':<12} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15} | {'EDGE %':<8}")
        print(sep)
        for bet in top_overs:
            print(f" {bet['TIER']:<12} | {bet['NAME'][:20]:<20} | {bet['TARGET']:<5} | "
                  f"{bet['AI']:>6.2f} vs {bet['PP']:>6.2f} | {bet['PCT_EDGE']:>6.1f}%")

        print("\n  TOP 10 UNDERS (Lowest Value)")
        print()
        print(f" {'TIER':<12} | {'PLAYER':<20} | {'STAT':<5} | {'AI vs PP':<15} | {'EDGE %':<8}")
        print(sep)
        for bet in top_unders:
            print(f" {bet['TIER']:<12} | {bet['NAME'][:20]:<20} | {bet['TARGET']:<5} | "
                  f"{bet['AI']:>6.2f} vs {bet['PP']:>6.2f} | {bet['PCT_EDGE']:>6.1f}%")

        # Determine save filename based on actual date used
        if actual_date:
            save_path = os.path.join(PROJ_DIR, f"scan_{actual_date}.csv")
        else:
            save_path = TOMORROW_SCAN_FILE if is_tomorrow else TODAY_SCAN_FILE

        pd.DataFrame(all_projections).to_csv(save_path, index=False)
        print(f"\nFull analysis ({len(all_projections)} rows) saved to {save_path}")
    else:
        print("\nNo active lines found.")

    input("\nPress Enter to continue...")


def prepare_features(player_row, is_home=0, days_rest=2, missing_usage=0):
    features = player_row.to_dict()
    features['IS_HOME']       = 1 if is_home else 0
    features['DAYS_REST']     = days_rest
    features['IS_B2B']        = 1 if days_rest == 1 else 0
    features['MISSING_USAGE'] = missing_usage
    return pd.DataFrame([features])


# --- STAT COLUMNS used for injury redistribution ---
# NOTE: Column names match the training CSV (mixed case from features.py)
_STAT_SEASON_COLS = {
    'PTS': 'PTS_Season', 'REB': 'REB_Season', 'AST': 'AST_Season',
    'FG3M': 'FG3M_Season', 'FGM': 'FGM_Season', 'FGA': 'FGA_Season',
    'FG3A': 'FG3A_Season', 'FTM': 'FTM_Season', 'FTA': 'FTA_Season',
    'STL': 'STL_Season', 'BLK': 'BLK_Season', 'TOV': 'TOV_Season',
}


def _calculate_injury_adjustments_fast(latest_rows_map, team_players, active_pid):
    """
    Optimized version of injury adjustments using pre-computed cache.
    """
    # --- Gather last-row data for every teammate ---
    out_production = {}       # stat -> total missing production
    active_usage   = {}       # pid  -> usage rate (for active players)

    for pid in team_players:
        last = latest_rows_map.get(pid)
        if not last: continue
        
        pname = last['PLAYER_NAME']
        usage = last.get('USAGE_RATE_Season', 0)

        if get_player_status(pname) == 'OUT':
            # Accumulate missing production
            for stat, col in _STAT_SEASON_COLS.items():
                val = last.get(col, 0)
                # Check for nan/None
                if val and val > 0:
                     out_production[stat] = out_production.get(stat, 0) + val
        else:
            if usage > 0:
                active_usage[pid] = usage

    if not out_production or active_pid not in active_usage:
        return {}  # nothing to adjust

    # --- Redistribute proportionally by usage share ---
    total_active_usage = sum(active_usage.values())
    if total_active_usage == 0: return {}
    
    player_share = active_usage[active_pid] / total_active_usage

    adjustments = {}
    for stat, missing_total in out_production.items():
        rate = ABSORPTION_RATES.get(stat, 0.40)
        adj  = missing_total * player_share * rate
        if abs(adj) > 0.01:
            adjustments[stat] = round(adj, 2)

    # Derive combo-stat adjustments from components
    pts_adj = adjustments.get('PTS', 0)
    reb_adj = adjustments.get('REB', 0)
    ast_adj = adjustments.get('AST', 0)
    stl_adj = adjustments.get('STL', 0)
    blk_adj = adjustments.get('BLK', 0)

    if pts_adj or reb_adj or ast_adj:
        adjustments['PRA'] = round(pts_adj + reb_adj + ast_adj, 2)
        adjustments['PR']  = round(pts_adj + reb_adj, 2)
        adjustments['PA']  = round(pts_adj + ast_adj, 2)
        adjustments['RA']  = round(reb_adj + ast_adj, 2)
    if stl_adj or blk_adj:
        adjustments['SB'] = round(stl_adj + blk_adj, 2)

    return adjustments


def _calculate_injury_adjustments(df_history, team_id, active_pid):
    """
    Calculate per-stat injury adjustments for a single active player.

    For each OUT teammate, sum their season-average production per stat.
    Distribute that missing production to active players proportional to
    each active player's usage share.  Scale by ABSORPTION_RATE per stat.

    Args:
        df_history: full historical DataFrame
        team_id:    team to analyse
        active_pid: PLAYER_ID of the player getting the adjustment

    Returns:
        dict  {stat: adjustment_value, ...}
              e.g. {'PTS': +2.31, 'REB': +1.80, 'AST': +0.42, ...}
    """
    # Filter to current season to avoid counting traded players
    season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
    current_season = df_history[season_col].max()
    team_df = df_history[(df_history[season_col] == current_season) & (df_history['TEAM_ID'] == team_id)]
    all_pids = team_df['PLAYER_ID'].unique()

    # --- Gather last-row data for every teammate ---
    out_production = {}       # stat -> total missing production
    active_usage   = {}       # pid  -> usage rate (for active players)

    for pid in all_pids:
        p_rows = team_df[team_df['PLAYER_ID'] == pid].sort_values('GAME_DATE')
        if p_rows.empty:
            continue
        last = p_rows.iloc[-1]
        pname = last['PLAYER_NAME']
        usage = last.get('USAGE_RATE_Season', 0)

        if get_player_status(pname) == 'OUT':
            # Accumulate missing production
            for stat, col in _STAT_SEASON_COLS.items():
                val = last.get(col, 0)
                if pd.notna(val) and val > 0:
                    out_production[stat] = out_production.get(stat, 0) + val
        else:
            if usage > 0:
                active_usage[pid] = usage

    if not out_production or active_pid not in active_usage:
        return {}  # nothing to adjust

    # --- Redistribute proportionally by usage share ---
    total_active_usage = sum(active_usage.values())
    player_share = active_usage[active_pid] / total_active_usage

    adjustments = {}
    for stat, missing_total in out_production.items():
        rate = ABSORPTION_RATES.get(stat, 0.40)
        adj  = missing_total * player_share * rate
        if abs(adj) > 0.01:
            adjustments[stat] = round(adj, 2)

    # Derive combo-stat adjustments from components
    pts_adj = adjustments.get('PTS', 0)
    reb_adj = adjustments.get('REB', 0)
    ast_adj = adjustments.get('AST', 0)
    stl_adj = adjustments.get('STL', 0)
    blk_adj = adjustments.get('BLK', 0)

    if pts_adj or reb_adj or ast_adj:
        adjustments['PRA'] = round(pts_adj + reb_adj + ast_adj, 2)
        adjustments['PR']  = round(pts_adj + reb_adj, 2)
        adjustments['PA']  = round(pts_adj + ast_adj, 2)
        adjustments['RA']  = round(reb_adj + ast_adj, 2)
    if stl_adj or blk_adj:
        adjustments['SB'] = round(stl_adj + blk_adj, 2)

    return adjustments


def scout_player(df_history, models):
    print("\n--- PLAYER SCOUT ---")
    refresh_injuries()
    d_choice = input("Select Start Date (1=Today, 2=Tomorrow): ").strip()
    offset = 1 if d_choice == '2' else 0
    
    # Use the improved get_games logic to find the next available games
    todays_teams, actual_date = get_games(
        date_offset=offset, 
        require_scheduled=True, 
        max_days_forward=7
    )
    
    if not todays_teams:
        print("No scheduled games found in the next 7 days.")
        return

    # Display the date being scouted
    scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
    print(f"\n📅 Scouting for games on: {scan_date_obj.strftime('%A, %B %d, %Y')}")

    pp_client  = PrizePicksClient(stat_map=STAT_MAP)

    # --- Load FanDuel cache from disk (NO API call) ---
    import json, math
    fd_cache_file = os.path.join('fanduel_cache', 'fanduel_cache_nba.json')
    fd_odds_by_player = {}   # {normalized_name: {stat_code: {'over': odds, 'under': odds, 'line': float}}}
    fd_cache_age_str = "N/A"

    # Reverse map: FanDuel display name -> scanner target code
    _FD_TO_CODE = {
        'Points': 'PTS', 'Rebounds': 'REB', 'Assists': 'AST',
        '3-Pt Made': 'FG3M', 'Pts+Rebs+Asts': 'PRA', 'Pts+Rebs': 'PR',
        'Pts+Asts': 'PA', 'Rebs+Asts': 'RA', 'Blks+Stls': 'SB',
        'Blocks': 'BLK', 'Steals': 'STL', 'Turnovers': 'TOV',
        'Field Goals Made': 'FGM', 'Free Throws Made': 'FTM',
        'Free Throws Attempted': 'FTA', 'Field Goals Attempted': 'FGA',
    }

    if os.path.exists(fd_cache_file):
        try:
            cache_age_mins = (time.time() - os.path.getmtime(fd_cache_file)) / 60
            if cache_age_mins < 60:
                fd_cache_age_str = f"{int(cache_age_mins)} min(s) ago"
            else:
                fd_cache_age_str = f"{cache_age_mins / 60:.1f} hr(s) ago"
            with open(fd_cache_file, 'r') as f:
                fd_raw = json.load(f)
            for entry in fd_raw:
                p_name = normalize_name(entry.get('Player', ''))
                raw_stat = entry.get('Stat', '')
                stat_code = _FD_TO_CODE.get(raw_stat, raw_stat)
                side = entry.get('Side', '')
                odds = entry.get('Odds', 0)
                line = entry.get('Line', 0)
                if not p_name or not stat_code:
                    continue
                if p_name not in fd_odds_by_player:
                    fd_odds_by_player[p_name] = {}
                if stat_code not in fd_odds_by_player[p_name]:
                    fd_odds_by_player[p_name][stat_code] = {'over': 0, 'under': 0, 'line': line}
                if side == 'Over':
                    fd_odds_by_player[p_name][stat_code]['over'] = odds
                elif side == 'Under':
                    fd_odds_by_player[p_name][stat_code]['under'] = odds
            print(f"   FanDuel cache loaded ({len(fd_odds_by_player)} players, {fd_cache_age_str})")
        except Exception as e:
            print(f"   Warning: could not load FanDuel cache: {e}")
    else:
        print("   No FanDuel cache found -- run Odds Scanner first to populate it")

    # Helper: calculate vig-removed true probability from American odds
    def _odds_to_prob(odds):
        if odds < 0:
            return (-odds) / ((-odds) + 100)
        else:
            return 100 / (odds + 100)

    # Line adjustment factors (same as analyzer.py)
    _LINE_ADJ = {
        'PTS': 0.035, 'PRA': 0.025, 'PR': 0.030, 'PA': 0.030,
        'REB': 0.040, 'AST': 0.045, 'RA': 0.035,
        'FG3M': 0.055, 'STL': 0.060, 'BLK': 0.060, 'SB': 0.055,
        'TOV': 0.045, 'FGM': 0.040, 'FGA': 0.040, 'FTM': 0.050, 'FTA': 0.050,
    }

    scouting   = True

    while scouting:
        print("\n(Type '0' to return to Main Menu)")
        query = input("Enter player name: ").strip().lower()
        if query == '0':
            break

        try:
            matches = df_history[df_history['PLAYER_NAME'].str.lower().str.contains(query)]
        except Exception as e:
            print(f"Search error: {e}")
            continue

        if matches.empty:
            print(f"No players found matching '{query}'.")
            continue

        unique_players = matches[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
        if len(unique_players) > 1:
            print(unique_players.to_string(index=False))
            try:
                pid = int(input("Enter PLAYER_ID: "))
                matches = matches[matches['PLAYER_ID'] == pid]
                if matches.empty:
                    print(f"No data found for PLAYER_ID {pid}.")
                    continue
            except ValueError:
                print("Invalid PLAYER_ID.")
                continue

        # Fetch lines for the identified date (includes goblin/demon alt lines)
        print("Fetching PrizePicks lines...")
        live_lines_full = pp_client.fetch_lines_with_type(league_filter='NBA')
        norm_lines_full = {normalize_name(k): v for k, v in live_lines_full.items()}

        try:
            player_data = matches.sort_values('GAME_DATE').iloc[-1]
        except IndexError:
            print("No recent history found for this player.")
            continue

        name    = player_data['PLAYER_NAME']
        team_id = player_data['TEAM_ID']
        
        # Check if the player's team is in the team_map for the 'actual_date'
        if team_id not in todays_teams:
            print(f"{name} is not scheduled to play on {actual_date}.")
            continue

        is_home = todays_teams[team_id]['is_home']

        # Calculate injury impact for the target date (current season only)
        season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
        current_season = df_history[season_col].max()
        df_current = df_history[(df_history[season_col] == current_season) & (df_history['TEAM_ID'] == team_id)]
        team_players = df_current['PLAYER_ID'].unique()
        missing_usage_today = 0.0
        for pid in team_players:
            p_rows = df_current[df_current['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            if get_player_status(last_row['PLAYER_NAME']) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15: missing_usage_today += usage

        print(f"\nSCOUTING REPORT: {name} ({actual_date})")
        print(f"Injury Impact: {missing_usage_today:.1f}% Missing Usage")

        # Calculate injury adjustments for this player
        pid_for_adj = matches.sort_values('GAME_DATE').iloc[-1]['PLAYER_ID']
        inj_adj = _calculate_injury_adjustments(df_history, team_id, pid_for_adj)
        if inj_adj:
            print(f"Injury Boost: adjustments for {len(inj_adj)} stats")

        # Show FanDuel cache age
        print(f"FanDuel odds: cached {fd_cache_age_str}")

        import unicodedata
        def _term_width(s):
            """Terminal display width of a string (handles emojis correctly)."""
            w = 0
            for ch in s:
                cat = unicodedata.category(ch)
                if cat in ('Mn', 'Me', 'Cf'):   # zero-width: marks, variation selectors
                    continue
                eaw = unicodedata.east_asian_width(ch)
                w += 2 if eaw in ('W', 'F') else 1
            return w

        TIER_WIDTH = 4  # desired terminal cols for tier column

        # (G) = goblin alt line, (D) = demon alt line
        print()
        print(f" TIER | STAT | {'PROJ':<14} | {'LINE':^11} | {'WIN%':>6} | SIDE")
        print("-" * 78)

        input_row = prepare_features(player_data, is_home=is_home, missing_usage=missing_usage_today)

        # Lookup FanDuel odds and PrizePicks lines for this player
        player_fd = fd_odds_by_player.get(normalize_name(name), {})
        player_pp = norm_lines_full.get(normalize_name(name), {})

        for target in TARGETS:
            if target in models:
                tier_emoji    = MODEL_QUALITY.get(target, {}).get('emoji', '?')
                model_features = [f for f in models[target].feature_names_in_ if target not in f]
                valid_input   = input_row.reindex(columns=model_features, fill_value=0)
                base_pred     = float(models[target].predict(valid_input)[0])
                adj           = inj_adj.get(target, 0)
                pred          = base_pred + adj

                # Get PP line with type info
                pp_info = player_pp.get(target)
                line      = pp_info['line'] if pp_info else None
                line_type = pp_info['type'] if pp_info else None

                # Format line string — fixed 11 chars wide
                if line is not None:
                    num_str = f"{line:.2f}"
                    if line_type == 'goblin':
                        line_str = f"{num_str:>6} (G) "
                    elif line_type == 'demon':
                        line_str = f"{num_str:>6} (D) "
                    else:
                        line_str = f"{num_str:>6}     "
                else:
                    line_str = "     --    "

                # --- Calculate WIN% from FanDuel odds — fixed 6 chars ---
                win_pct_str = "    --"
                if line is not None and target in player_fd:
                    fd_info = player_fd[target]
                    fd_over_odds = fd_info['over']
                    fd_under_odds = fd_info['under']
                    fd_line = fd_info['line']
                    if fd_over_odds != 0 and fd_under_odds != 0:
                        prob_o = _odds_to_prob(fd_over_odds)
                        prob_u = _odds_to_prob(fd_under_odds)
                        total = prob_o + prob_u
                        true_over = prob_o / total
                        true_under = prob_u / total
                        # Adjust for PP vs FD line difference
                        line_diff = line - fd_line
                        if line_diff != 0:
                            factor = _LINE_ADJ.get(target, 0.035)
                            adjustment = factor * math.log(1 + abs(line_diff)) / math.log(2)
                            if line_diff < 0:
                                true_over = min(true_over + adjustment, 0.90)
                                true_under = max(true_under - adjustment, 0.10)
                            else:
                                true_over = max(true_over - adjustment, 0.10)
                                true_under = min(true_under + adjustment, 0.90)
                            norm = true_over + true_under
                            true_over /= norm
                            true_under /= norm
                        # Show the win% for the projected side
                        if pred > line:
                            win_pct_str = f"{true_over * 100:5.1f}%"
                        else:
                            win_pct_str = f"{true_under * 100:5.1f}%"

                # --- ▲/▼ indicator ---
                if line is None or line <= 0:
                    side_str = "--"
                else:
                    diff = pred - line
                    if diff > 0:
                        side_str = f"▲ Over (+{diff:.2f})"
                    else:
                        side_str = f"▼ Under ({diff:.2f})"

                # Pad tier to fixed terminal width
                pad           = " " * max(0, TIER_WIDTH - _term_width(tier_emoji))
                tier_col      = tier_emoji + pad
                proj_str = f"{pred:>6.2f}       "
                print(f" {tier_col}| {target:<4} | {proj_str:<14} | {line_str} | {win_pct_str} | {side_str}")

        if input("\nScout another player? (y/n): ").lower() != 'y':
            scouting = False


def main():
    print("...Initializing System")
    refresh_injuries()
    df     = load_data()
    models = load_models()
    if df is None or not models:
        print("Setup failed.")
        return

    while True:
        print("\n" + "="*30 + "\n   NBA AI SCANNER\n" + "="*30)
        print("1. Scan TODAY's Games")
        print("2. Scan NEXT Match")
        print("3. Scout Specific Player")
        print("0. Exit")
        choice = input("\nSelect: ").strip()
        if choice == '1':   scan_all(df, models, is_tomorrow=False, max_days_forward=0)
        elif choice == '2': scan_all(df, models, is_tomorrow=True, max_days_forward=7)
        elif choice == '3': scout_player(df, models)
        elif choice == '0': break


if __name__ == "__main__":
    main()