"""
NBA Props Scanner - AI-Powered Prediction System

Scans upcoming NBA games, generates player performance predictions using
trained XGBoost models, and identifies profitable betting opportunities
by comparing predictions against PrizePicks lines.

Usage:
    $ python3 -m src.sports.nba.scanner
"""

import pandas as pd
import numpy as np
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
from src.sports.nba.train   import LOG_TRANSFORM_TARGETS

# Empirical calibration factors for log-transformed targets.
# Log-transform regression produces E[log1p(y)] which, after expm1, is
# systematically below E[y] (Jensen's inequality). Factors derived from the
# held-out 30% test set using only players with meaningful baselines.
LOG_CALIBRATION = {
    'BLK': 1.2835,
    'STL': 1.3315,
    'TOV': 1.2397,
    'FG3M': 1.2265,
    'SB': 1.2401,
}

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

# Suppress verbose internal prints during batch loading (set True in scan_all)
_QUIET = False

def _log(*args, **kwargs):
    if not _QUIET:
        print(*args, **kwargs)

# Injury cache — refreshed before each scan (see refresh_injuries)
INJURY_DATA = {}


def refresh_injuries():
    """Fetch fresh injury report and update global cache. Call before each scan."""
    global INJURY_DATA
    INJURY_DATA = get_injury_report()

TARGETS = ACTIVE_TARGETS

# --- FEATURES LIST ---
# MUST stay in sync with train.py get_features_for_target().
# The models use L5, L10, L20, Season, L5_Median, L10_Median per stat.
# Missing features get fill_value=0 at inference → corrupted predictions.

_BASE_STATS = ['PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'STL', 'BLK', 'TOV',
               'FGM', 'FGA', 'FTM', 'FTA', 'MIN', 'GAME_SCORE', 'USAGE_RATE', 'FPTS']
_COMBO_STATS = ['PRA', 'PR', 'PA', 'RA', 'SB']
_1H_STATS = ['PTS_1H', 'MIN_1H', 'FPTS_1H', 'PRA_1H']

FEATURES = [
    # Core context features (used by every model)
    'MISSING_USAGE',
    'TS_PCT', 'DAYS_REST', 'IS_HOME',
    'GAMES_7D', 'IS_4_IN_6', 'IS_B2B', 'IS_FRESH',
    'PACE_ROLLING', 'FGA_PER_MIN', 'TOV_PER_USAGE',
    'USAGE_VACUUM', 'STAR_COUNT',
    # Location splits + opponent context
    'PTS_LOC_MEAN', 'REB_LOC_MEAN', 'AST_LOC_MEAN', 'FG3M_LOC_MEAN', 'PRA_LOC_MEAN',
    'OPP_WIN_PCT', 'IS_VS_ELITE_TEAM'
]

# Rolling variants for ALL stats (L5, L10, L20, Season, L5_Median, L10_Median)
for stat in _BASE_STATS + _COMBO_STATS + _1H_STATS:
    for suffix in ['_Season', '_L5', '_L10', '_L20', '_L5_Median', '_L10_Median']:
        FEATURES.append(f'{stat}{suffix}')

# Defensive matchup features (DvP)
for stat in ['PTS', 'REB', 'AST', 'FG3M', 'FGA', 'BLK', 'STL', 'TOV', 'FGM', 'FTM', 'FTA']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')
    FEATURES.append(f'OPP_{stat}_ALLOWED_DIFF')

for combo in _COMBO_STATS:
    FEATURES.append(f'OPP_{combo}_ALLOWED')
    FEATURES.append(f'OPP_{combo}_ALLOWED_DIFF')


def normalize_name(name):
    if not name: return ""
    n = unicodedata.normalize('NFKD', name)
    clean = "".join([c for c in n if not unicodedata.combining(c)])
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)
    for s in ['Jr', 'Sr', 'III', 'II', 'IV']:
        clean = clean.replace(f" {s}", "")
    return " ".join(clean.lower().split())


def get_player_status(name, injury_data=None):
    """
    Check if player is on injury report. Uses exact match, then initial+last fallback,
    then last-name-only fallback (only when exactly one injured player shares that last name).
    Handles injury reports that abbreviate first names (e.g. 'D. Murray' vs 'Dejounte Murray').
    """
    if injury_data is None:
        injury_data = INJURY_DATA

    norm_name = normalize_name(name)
    for injured_name, status in injury_data.items():
        if normalize_name(injured_name) == norm_name:
            return status

    parts = norm_name.split()
    if len(parts) < 2:
        return "Active"

    first_initial = parts[0][0]   # e.g. 'd' from 'dejounte'
    last_name     = parts[-1]     # e.g. 'murray'

    # Initial+last match: 'D. Murray' → initial='d', last='murray'
    initial_matches = [
        (inj_name, s) for inj_name, s in injury_data.items()
        if (lambda p: len(p) >= 2 and p[0][0] == first_initial and p[-1] == last_name)(
            normalize_name(inj_name).split()
        )
    ]
    if len(initial_matches) == 1:
        return initial_matches[0][1]

    # Last-name-only fallback: only when exactly one injured player shares that last name
    last_matches = [
        s for inj_name, s in injury_data.items()
        if normalize_name(inj_name).split()[-1] == last_name
    ]
    if len(last_matches) == 1:
        return last_matches[0]

    return "Active"


def prepare_features(player_row, is_home=0, days_rest=2, missing_usage=0):
    if isinstance(player_row, dict):
        features = player_row.copy()
    else:
        features = player_row.to_dict()
    
    features['IS_HOME']       = 1 if is_home else 0
    features['DAYS_REST']     = days_rest
    features['IS_B2B']        = 1 if days_rest == 1 else 0
    features['IS_FRESH']      = 1 if days_rest >= 3 else 0
    features['MISSING_USAGE'] = missing_usage
    return pd.DataFrame([features])


def get_betting_indicator(proj, line):
    if line is None or line <= 0: return "⚪ NO LINE"
    diff = proj - line
    if diff > 0: return f"🟢 OVER (+{diff:.2f})"
    else:        return f"🔴 UNDER ({diff:.2f})"


def _derive_stat_column(player_logs, stat):
    """
    Derive a stat column if it doesn't exist in the DataFrame.
    Handles full-game combos, FPTS, and 1H stats.
    Returns the column name if derivable, else None.
    """
    if stat in player_logs.columns and player_logs[stat].notna().any():
        return stat  # Already exists with data

    # Full-game combo stats
    if stat == 'PRA' and all(c in player_logs.columns for c in ['PTS', 'REB', 'AST']):
        player_logs['PRA'] = player_logs['PTS'] + player_logs['REB'] + player_logs['AST']
    elif stat == 'PR' and all(c in player_logs.columns for c in ['PTS', 'REB']):
        player_logs['PR'] = player_logs['PTS'] + player_logs['REB']
    elif stat == 'PA' and all(c in player_logs.columns for c in ['PTS', 'AST']):
        player_logs['PA'] = player_logs['PTS'] + player_logs['AST']
    elif stat == 'RA' and all(c in player_logs.columns for c in ['REB', 'AST']):
        player_logs['RA'] = player_logs['REB'] + player_logs['AST']
    elif stat == 'SB' and all(c in player_logs.columns for c in ['STL', 'BLK']):
        player_logs['SB'] = player_logs['STL'] + player_logs['BLK']

    # Fantasy Score (PrizePicks formula)
    elif stat == 'FPTS' and all(c in player_logs.columns for c in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV']):
        player_logs['FPTS'] = (player_logs['PTS'] * 1 + player_logs['REB'] * 1.2
                               + player_logs['AST'] * 1.5 + player_logs['BLK'] * 3
                               + player_logs['STL'] * 3 - player_logs['TOV'])

    # 1H combo / derived stats
    elif stat == 'PRA_1H' and all(c in player_logs.columns for c in ['PTS_1H', 'REB_1H', 'AST_1H']):
        player_logs['PRA_1H'] = player_logs['PTS_1H'] + player_logs['REB_1H'] + player_logs['AST_1H']
    elif stat == 'FPTS_1H' and all(c in player_logs.columns for c in ['PTS_1H', 'REB_1H', 'AST_1H', 'BLK_1H', 'STL_1H', 'TOV_1H']):
        player_logs['FPTS_1H'] = (player_logs['PTS_1H'] * 1 + player_logs['REB_1H'] * 1.2
                                  + player_logs['AST_1H'] * 1.5 + player_logs['BLK_1H'] * 3
                                  + player_logs['STL_1H'] * 3 - player_logs['TOV_1H'])
    else:
        return None

    return stat


def calculate_hit_rates(df_history, player_id, stat, line, player_logs_index=None):
    """
    Calculate L5, L10, L20 hit rates against a specific line for a player.
    Returns: (l5_rate, l10_rate, l20_rate) as floats between 0.0 and 1.0.
    """
    if line is None or line <= 0:
        return 0.0, 0.0, 0.0

    # Use pre-built index for O(1) lookup; fall back to O(N) filter if not provided
    if player_logs_index is not None:
        player_logs = player_logs_index.get(player_id, pd.DataFrame()).copy()
    else:
        player_logs = df_history[df_history['PLAYER_ID'] == player_id].sort_values('GAME_DATE').copy()

    # Derive stat column if needed (handles FPTS, 1H stats, combos)
    if _derive_stat_column(player_logs, stat) is None:
        return 0.0, 0.0, 0.0

    # Drop rows where the stat is NaN (e.g. auto-refreshed rows missing 1H data)
    player_logs = player_logs.dropna(subset=[stat])

    recent_20 = player_logs.tail(20)
    if recent_20.empty:
        return 0.0, 0.0, 0.0

    # How many times did they hit OVER the line?
    hits_20 = (recent_20[stat] > line).sum()
    hits_10 = (recent_20.tail(10)[stat] > line).sum()
    hits_5  = (recent_20.tail(5)[stat] > line).sum()

    count_20 = len(recent_20)
    count_10 = len(recent_20.tail(10))
    count_5  = len(recent_20.tail(5))

    l20_rate = hits_20 / count_20 if count_20 > 0 else 0.0
    l10_rate = hits_10 / count_10 if count_10 > 0 else 0.0
    l5_rate  = hits_5 / count_5 if count_5 > 0 else 0.0

    return l5_rate, l10_rate, l20_rate


def calculate_h2h_hit_rate(df_history, player_id, stat, line, opp_abbr, player_logs_index=None):
    """
    Calculate hit rate against a specific opponent (head-to-head).
    Limited to the last 2 seasons to keep data relevant.

    Args:
        df_history:        Full game log DataFrame
        player_id:         Player's NBA ID
        stat:              Stat column name (e.g. 'PTS', 'PRA_1H', 'FPTS')
        line:              PrizePicks line value
        opp_abbr:          Opponent team abbreviation (e.g. 'BOS')
        player_logs_index: Optional pre-built {pid: sorted_df} for O(1) lookup

    Returns:
        (h2h_rate, h2h_count): over-rate and number of H2H games found
    """
    if line is None or line <= 0 or not opp_abbr:
        return 0.0, 0

    if player_logs_index is not None:
        player_logs = player_logs_index.get(player_id, pd.DataFrame()).copy()
    else:
        player_logs = df_history[df_history['PLAYER_ID'] == player_id].sort_values('GAME_DATE').copy()

    if 'MATCHUP' not in player_logs.columns or player_logs.empty:
        return 0.0, 0

    # Limit to last 2 seasons — older matchups are irrelevant
    cutoff = pd.to_datetime(datetime.now()) - timedelta(days=730)
    player_logs = player_logs[player_logs['GAME_DATE'] >= cutoff]

    # Derive stat column if needed
    if _derive_stat_column(player_logs, stat) is None:
        return 0.0, 0

    # Filter to games vs this opponent
    # MATCHUP format: "LAL vs. NOP" (home) or "LAL @ NOP" (away)
    opp_upper = str(opp_abbr).upper() if opp_abbr else ''
    h2h_mask = player_logs['MATCHUP'].str.contains(opp_upper, case=False, na=False)
    h2h_logs = player_logs.loc[h2h_mask].dropna(subset=[stat])

    if h2h_logs.empty:
        return 0.0, 0

    count = len(h2h_logs)
    hits = (h2h_logs[stat] > line).sum()
    rate = hits / count if count > 0 else 0.0

    return rate, count


def analyze_player_availability(df_history, player_id, scan_date_str):
    """
    Dynamically analyze a player's recent game log to detect:
    - Extended absence (missed recent games)
    - Minute restriction (ramping back up after injury)
    - Recent return from multi-game absence
    
    Returns dict with:
        penalty      (float): quality score penalty to apply (0 to -35)
        scale_factor (float): projection multiplier (0.0 to 1.0, lower = more reduction)
        flag         (str):   display flag ('⚠INJ', '⚠MIN', '⚠RTN', or '')
        reason       (str):   human-readable reason
    """
    result = {'penalty': 0.0, 'scale_factor': 1.0, 'flag': '', 'reason': ''}
    
    player_logs = df_history[df_history['PLAYER_ID'] == player_id].sort_values('GAME_DATE')
    if len(player_logs) < 3:
        return result  # Not enough history to analyze
    
    scan_date = pd.to_datetime(scan_date_str) if scan_date_str else pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    
    # --- 1. Identify Team Schedule ---
    # Use all teams the player appeared for in the current season to handle trades correctly.
    season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in player_logs.columns else 'SEASON_ID'
    current_season = player_logs[season_col].max()
    current_season_logs = player_logs[player_logs[season_col] == current_season]
    current_team_ids = current_season_logs['TEAM_ID'].unique() if not current_season_logs.empty else [player_logs['TEAM_ID'].iloc[-1]]
    team_dates = df_history[df_history['TEAM_ID'].isin(current_team_ids)]['GAME_DATE'].drop_duplicates().sort_values()
    
    # --- 2. Current missed team games ---
    last_game_date = player_logs['GAME_DATE'].iloc[-1]
    # Games the team played after the player's last game (up to but not including scan_date)
    team_games_missed_now = len(team_dates[(team_dates > last_game_date) & (team_dates < scan_date)])
    
    # --- 3. Historic gaps (Detect missed games recently) ---
    recent_dates = player_logs['GAME_DATE'].tail(10).tolist()
    max_missed_games_in_gap = 0
    
    for i in range(1, len(recent_dates)):
        d1 = recent_dates[i-1]
        d2 = recent_dates[i]
        # Count team games strictly between d1 and d2
        missed = len(team_dates[(team_dates > d1) & (team_dates < d2)])
        if missed > max_missed_games_in_gap:
            max_missed_games_in_gap = missed
    
    # --- 4. Minute restriction detection ---
    # Compare last 3 games' minutes to season average
    if 'MIN' in player_logs.columns:
        season_min_avg = player_logs['MIN'].mean()
        last_3_mins = player_logs['MIN'].tail(3)
        last_3_avg = last_3_mins.mean()
        last_game_min = player_logs['MIN'].iloc[-1]
        
        # Minute ratio: how do recent minutes compare to season?
        min_ratio = last_3_avg / season_min_avg if season_min_avg > 5 else 1.0
        last_min_ratio = last_game_min / season_min_avg if season_min_avg > 5 else 1.0
        
        # Check for ramp-up pattern: minutes increasing game-over-game
        is_ramping = False
        if len(last_3_mins) == 3:
            mins_list = last_3_mins.tolist()
            # If the last game was back up to full minutes, we are fully ramped, no need to penalize.
            is_ramping = mins_list[0] < mins_list[1] < mins_list[2] and min_ratio < 0.85 and last_min_ratio < 0.95
    else:
        min_ratio = 1.0
        last_min_ratio = 1.0
        is_ramping = False
        season_min_avg = 0
    
    # --- 5. Apply penalties based on dynamic analysis ---
    
    # Extended absence: missed 4+ team games
    if team_games_missed_now >= 4:
        result['penalty'] = -30.0
        result['scale_factor'] = 0.70  # Expect ~30% production loss from rust + minutes
        result['flag'] = '⚠INJ'
        result['reason'] = f'Out {team_games_missed_now} games — extended absence'
    
    # Moderate absence: missed 2-3 team games
    elif team_games_missed_now >= 2:
        result['penalty'] = -20.0
        result['scale_factor'] = 0.80  # Expect ~20% production loss
        result['flag'] = '⚠INJ'
        result['reason'] = f'Out {team_games_missed_now} games — recent absence'
    
    # Recent multi-game absence detected AND returned
    elif max_missed_games_in_gap >= 2 and team_games_missed_now == 0:
        # Player missed games but has returned. How many games back?
        games_back = 0
        for i in range(1, len(recent_dates)):
            d1, d2 = recent_dates[i-1], recent_dates[i]
            if len(team_dates[(team_dates > d1) & (team_dates < d2)]) == max_missed_games_in_gap:
                games_back = len(recent_dates) - i
                break
                
        if games_back <= 3:
            result['penalty'] = -15.0
            # Scale based on how many games back: 1 game = more rust, 3 games = almost normal
            result['scale_factor'] = 0.75 + (games_back * 0.05)  # 0.80, 0.85, 0.90
            result['flag'] = '⚠RTN'
            result['reason'] = f'Just returned — {games_back} games back after missing {max_missed_games_in_gap} games'
    
    # Minute restriction: recent minutes well below season average
    if last_min_ratio < 0.65 and season_min_avg > 10:
        # Severe minute restriction - scale production to match minutes
        extra_penalty = -20.0 if not result['flag'] else -10.0
        result['penalty'] += extra_penalty
        # Production scales with minutes but slightly worse (rust/conditioning)
        result['scale_factor'] = min(result['scale_factor'], last_min_ratio * 0.90)
        if not result['flag']:
            result['flag'] = '⚠MIN'
            result['reason'] = f'Min restriction: {last_game_min:.0f}min vs {season_min_avg:.0f}avg'
        else:
            result['reason'] += f' + min restricted ({last_game_min:.0f}/{season_min_avg:.0f})'
    
    elif min_ratio < 0.75 and season_min_avg > 10 and not result['flag']:
        # Moderate minute restriction (L3 average below 75% of season)
        result['penalty'] = -12.0
        result['scale_factor'] = min(result['scale_factor'], min_ratio * 0.95)
        result['flag'] = '⚠MIN'
        result['reason'] = f'Reduced mins: L3 avg {last_3_avg:.0f} vs {season_min_avg:.0f} season'
    
    # Ramp-up pattern: minutes increasing but still below season avg
    elif is_ramping and not result['flag']:
        result['penalty'] = -10.0
        result['scale_factor'] = min(result['scale_factor'], min_ratio)
        result['flag'] = '⚠RTN'
        result['reason'] = f'Ramping up: {mins_list[0]:.0f}→{mins_list[1]:.0f}→{mins_list[2]:.0f}min (avg {season_min_avg:.0f})'
    
    # Clamp scale_factor
    result['scale_factor'] = max(0.50, min(1.0, result['scale_factor']))
    
    return result


def calculate_confidence_score(edge_pct, l10_hit, opponent_win_pct=None, is_role_expansion=False):
    """
    Calibrated confidence score (0-100) grounded in backtested win rates.

    Calibration source: walk-forward backtest on 28k test rows using
    player L10-median as the proxy betting line (backtester.py).

    Edge % → estimated win rate (profitable stats):
        0-8%   → ~50-52%  (below break-even at 53.5%)
        8-15%  → ~53%     (marginal)
        15-20% → ~55.7%   (profitable)
        >20%   → ~56.8%   (profitable, capped ~60%)

    Score maps estimated win % to 0-100:
        50% = 0   (break-even / no edge)
        58% = 100 (elite edge)
    """
    abs_edge = abs(edge_pct)

    # Base win probability from edge calibration curve
    if abs_edge < 8:
        est_win_pct = 50.0 + abs_edge * 0.25          # 50.0 → 52.0%
    elif abs_edge < 15:
        est_win_pct = 52.0 + (abs_edge - 8) * (1.0 / 7)   # 52.0 → 53.0%
    elif abs_edge < 20:
        est_win_pct = 53.0 + (abs_edge - 15) * (2.7 / 5)  # 53.0 → 55.7%
    else:
        est_win_pct = min(55.7 + (abs_edge - 20) * 0.1, 60.0)  # 55.7 → 60%

    # L10 hit rate: confirmation or contradiction
    if l10_hit >= 0.70:
        est_win_pct += 1.0   # Strong confirmation
    elif l10_hit <= 0.30:
        est_win_pct -= 2.0   # Historical contradiction — significant red flag

    # Matchup quality
    if opponent_win_pct is not None:
        if opponent_win_pct < 0.40:
            est_win_pct += 0.5   # Weak opponent
        elif opponent_win_pct > 0.60:
            est_win_pct -= 0.5   # Strong opponent

    # Role expansion penalty: UNDER on newly expanded-role player is trap
    if is_role_expansion and edge_pct < 0:
        est_win_pct -= 3.0

    # Map to 0-100 (50% win rate = 0 score, 58% win rate = 100 score)
    score = max(0.0, (est_win_pct - 50.0) / 8.0 * 100.0)
    return min(score, 100.0)


def load_data():
    if not os.path.exists(DATA_FILE): return None
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df


def auto_refresh_data(df_history):
    """
    Fetch recent game logs from NBA API and merge into df_history.
    On success, saves updated data to disk so future runs skip the API call.
    
    Returns: updated df_history with new rows merged in.
    """
    from nba_api.stats.endpoints import playergamelogs

    latest_date = df_history['GAME_DATE'].max()
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    days_stale = (today - latest_date).days

    if days_stale <= 1:
        _log(f"   ✅ Data is fresh (last game: {latest_date.date()})")
        return df_history

    _log(f"   ⚠️  Data is {days_stale} days stale (last: {latest_date.date()}, today: {today.date()})")
    _log(f"   Fetching recent game logs from NBA API...")

    # Retry up to 3 times with increasing timeout
    api_df = None
    for attempt in range(3):
        try:
            timeout = 60 + attempt * 30  # 60, 90, 120
            if attempt > 0:
                _log(f"   Retry {attempt + 1}/3 (timeout={timeout}s)...")
                time.sleep(3)
            date_from = (latest_date + pd.Timedelta(days=1)).strftime('%m/%d/%Y')
            logs = playergamelogs.PlayerGameLogs(
                season_nullable='2025-26',
                league_id_nullable='00',
                date_from_nullable=date_from,
                timeout=timeout
            )
            api_df = logs.get_data_frames()[0]
            break  # Success
        except Exception as e:
            if attempt == 2:
                _log(f"   ⚠️  Auto-refresh failed ({e}), using existing dataset")
                return df_history

    if api_df is None or api_df.empty:
        _log("   ⚠️  No new data from API, using existing dataset")
        return df_history

    # --- Fetch 1st Half box scores ---
    api_1h_df = None
    _log(f"   Fetching 1st Half box scores...")
    for attempt in range(3):
        try:
            timeout = 60 + attempt * 30
            if attempt > 0:
                _log(f"   Retry {attempt + 1}/3 for 1H (timeout={timeout}s)...")
                time.sleep(3)
            logs_1h = playergamelogs.PlayerGameLogs(
                season_nullable='2025-26',
                league_id_nullable='00',
                game_segment_nullable='First Half',
                date_from_nullable=date_from,
                timeout=timeout
            )
            api_1h_df = logs_1h.get_data_frames()[0]
            break
        except Exception as e:
            if attempt == 2:
                _log(f"   ⚠️  1H fetch failed ({e}), 1H stats will use last known values")

    try:
        # Parse dates and filter to only new rows
        api_df['GAME_DATE'] = pd.to_datetime(api_df['GAME_DATE'])
        new_rows = api_df[api_df['GAME_DATE'] > latest_date].copy()

        if new_rows.empty:
            _log(f"   ✅ No new games since {latest_date.date()}")
            return df_history

        _log(f"   📥 Found {len(new_rows)} new game rows ({new_rows['GAME_DATE'].min().date()} → {new_rows['GAME_DATE'].max().date()})")

        # Standardize column names to match training dataset
        # The API uses SEASON_YEAR, training dataset uses SEASON_ID
        if 'SEASON_YEAR' in new_rows.columns and 'SEASON_ID' not in new_rows.columns:
            new_rows['SEASON_ID'] = new_rows['SEASON_YEAR']
        if 'SEASON_ID' not in new_rows.columns:
            new_rows['SEASON_ID'] = '2025-26'

        # Keep only columns that exist in training dataset (raw stats)
        raw_stat_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
                         'GAME_ID', 'GAME_DATE',
                         'MATCHUP', 'WL', 'SEASON_ID',
                         'MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'STL', 'BLK', 'TOV',
                         'FGM', 'FGA', 'FTM', 'FTA', 'OREB', 'DREB', 'PF',
                         'NBA_FANTASY_PTS', 'PLUS_MINUS']
        
        # Also keep SEASON_YEAR if present in the training dataset
        if 'SEASON_YEAR' in df_history.columns:
            raw_stat_cols.append('SEASON_YEAR')

        available_cols = [c for c in raw_stat_cols if c in new_rows.columns]
        new_rows = new_rows[available_cols].copy()

        # Add combo stats
        if all(c in new_rows.columns for c in ['PTS', 'REB', 'AST']):
            new_rows['PRA'] = new_rows['PTS'] + new_rows['REB'] + new_rows['AST']
            new_rows['PR'] = new_rows['PTS'] + new_rows['REB']
            new_rows['PA'] = new_rows['PTS'] + new_rows['AST']
            new_rows['RA'] = new_rows['REB'] + new_rows['AST']
        if all(c in new_rows.columns for c in ['STL', 'BLK']):
            new_rows['SB'] = new_rows['STL'] + new_rows['BLK']
        if 'NBA_FANTASY_PTS' in new_rows.columns:
            new_rows['FPTS'] = new_rows['NBA_FANTASY_PTS']

        # --- Merge 1H data into new rows ---
        if api_1h_df is not None and not api_1h_df.empty:
            api_1h_df['GAME_DATE'] = pd.to_datetime(api_1h_df['GAME_DATE'])
            new_1h = api_1h_df[api_1h_df['GAME_DATE'] > latest_date].copy()
            if not new_1h.empty:
                rename_cols = {
                    'MIN': 'MIN_1H', 'PTS': 'PTS_1H', 'REB': 'REB_1H', 'AST': 'AST_1H',
                    'FG3M': 'FG3M_1H', 'STL': 'STL_1H', 'BLK': 'BLK_1H', 'TOV': 'TOV_1H',
                    'FGM': 'FGM_1H', 'FGA': 'FGA_1H', 'FTM': 'FTM_1H', 'FTA': 'FTA_1H',
                    'NBA_FANTASY_PTS': 'NBA_FANTASY_PTS_1H', 'FG3A': 'FG3A_1H'
                }
                new_1h.rename(columns=rename_cols, inplace=True)
                cols_to_keep = ['PLAYER_ID', 'GAME_ID'] + list(rename_cols.values())
                new_1h = new_1h[[c for c in cols_to_keep if c in new_1h.columns]]
                new_rows = new_rows.merge(new_1h, on=['PLAYER_ID', 'GAME_ID'], how='left')
                # Compute 1H combo stats
                if all(c in new_rows.columns for c in ['PTS_1H', 'REB_1H', 'AST_1H']):
                    new_rows['PRA_1H'] = new_rows['PTS_1H'] + new_rows['REB_1H'] + new_rows['AST_1H']
                if all(c in new_rows.columns for c in ['PTS_1H', 'REB_1H', 'AST_1H', 'BLK_1H', 'STL_1H', 'TOV_1H']):
                    new_rows['FPTS_1H'] = (new_rows['PTS_1H'] + new_rows['REB_1H'] * 1.2
                                           + new_rows['AST_1H'] * 1.5 + new_rows['BLK_1H'] * 3
                                           + new_rows['STL_1H'] * 3 - new_rows['TOV_1H'])
                _log(f"   📥 Merged {len(new_1h)} 1H box scores")

        # Merge new rows into history (dedup by PLAYER_ID + GAME_ID)
        combined = pd.concat([df_history, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=['PLAYER_ID', 'GAME_ID'], keep='first')
        combined = combined.sort_values(['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)

        # Recompute rolling stats with vectorized groupby.transform — eliminates O(P×S) Python loops
        updated_pids = new_rows['PLAYER_ID'].unique()
        _log(f"   🔄 Recomputing rolling stats (vectorized, {len(updated_pids)} updated players)...")

        base_stats = ['PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'STL', 'BLK', 'TOV',
                      'FGM', 'FGA', 'FTM', 'FTA', 'MIN', 'PRA', 'PR', 'PA', 'RA', 'SB', 'FPTS',
                      'PTS_1H', 'REB_1H', 'AST_1H', 'FG3M_1H', 'STL_1H', 'BLK_1H', 'TOV_1H',
                      'FGM_1H', 'FGA_1H', 'FTM_1H', 'FTA_1H', 'MIN_1H', 'PRA_1H', 'FPTS_1H',
                      'NBA_FANTASY_PTS_1H', 'FG3A_1H']
        base_stats = [s for s in base_stats if s in combined.columns]

        # combined is already sorted by [PLAYER_ID, GAME_DATE] — required for correct rolling
        for stat in base_stats:
            _tmp = '__tmp'
            combined[_tmp] = combined[stat].copy()
            if '_1H' in stat:
                # 0s in 1H cols mean "no 1H box score available" — treat as NaN to avoid dilution
                combined[_tmp] = combined[_tmp].replace(0, float('nan'))
            grp = combined.groupby('PLAYER_ID')[_tmp]
            combined[f'{stat}_L5']         = grp.transform(lambda x: x.rolling(5,  min_periods=1).mean())
            combined[f'{stat}_L10']        = grp.transform(lambda x: x.rolling(10, min_periods=1).mean())
            combined[f'{stat}_L20']        = grp.transform(lambda x: x.rolling(20, min_periods=1).mean())
            combined[f'{stat}_Season']     = grp.transform(lambda x: x.expanding().mean())
            combined[f'{stat}_L5_Median']  = grp.transform(lambda x: x.rolling(5,  min_periods=1).median())
            combined[f'{stat}_L10_Median'] = grp.transform(lambda x: x.rolling(10, min_periods=1).median())
            combined.drop(columns=[_tmp], inplace=True)

        # GAME_SCORE and USAGE_RATE are derived stats — recompute only for updated players
        gs_cols = ['PTS', 'FGM', 'FGA', 'FTM', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        for pid in updated_pids:
            mask = combined['PLAYER_ID'] == pid
            player_df = combined.loc[mask]

            if all(c in player_df.columns for c in gs_cols):
                gs = (player_df['PTS'] + 0.4 * player_df['FGM'] - 0.7 * player_df['FGA']
                      - 0.4 * (player_df['FTA'] - player_df['FTM']) + 0.7 * player_df['OREB'].fillna(0)
                      + 0.3 * player_df['DREB'].fillna(0) + player_df['STL']
                      + 0.7 * player_df['AST'] + 0.7 * player_df['BLK']
                      - 0.4 * player_df['PF'] - player_df['TOV'])
                combined.loc[mask, 'GAME_SCORE']        = gs.values
                combined.loc[mask, 'GAME_SCORE_L5']     = gs.rolling(5,  min_periods=1).mean().values
                combined.loc[mask, 'GAME_SCORE_L10']    = gs.rolling(10, min_periods=1).mean().values
                combined.loc[mask, 'GAME_SCORE_Season'] = gs.expanding().mean().values

            if all(c in player_df.columns for c in ['FGA', 'FTA', 'TOV', 'MIN']):
                mins  = player_df['MIN'].replace(0, 1)
                usage = ((player_df['FGA'] + 0.44 * player_df['FTA'] + player_df['TOV']) / mins * 48 / 5).clip(0, 50)
                combined.loc[mask, 'USAGE_RATE']        = usage.values
                combined.loc[mask, 'USAGE_RATE_L5']     = usage.rolling(5,  min_periods=1).mean().values
                combined.loc[mask, 'USAGE_RATE_L10']    = usage.rolling(10, min_periods=1).mean().values
                combined.loc[mask, 'USAGE_RATE_Season'] = usage.expanding().mean().values

        # ── Recompute team-level features for new game rows ──────────────────
        # Auto-refresh adds raw stats only; STAR_COUNT, PACE_ROLLING, OPP_PTS_ALLOWED,
        # USAGE_VACUUM are NaN for new rows. NaN feeds XGBoost's missing-value path
        # (trained on zeros), which suppresses projections. Fix: compute what we can,
        # forward-fill the rest per player.

        # 1. STAR_COUNT: per game+team, count players with USAGE_RATE_Season > 28%
        if 'USAGE_RATE_Season' in combined.columns:
            new_gids = new_rows['GAME_ID'].unique()
            new_mask = combined['GAME_ID'].isin(new_gids)
            star_flags = (combined.loc[new_mask, 'USAGE_RATE_Season'] > 28).astype(float)
            combined.loc[new_mask, '_star_flag'] = star_flags
            star_counts = combined[new_mask].groupby(['GAME_ID', 'TEAM_ID'])['_star_flag'].sum()
            star_counts.name = 'STAR_COUNT_NEW'
            combined = combined.merge(
                star_counts.reset_index().rename(columns={'STAR_COUNT_NEW': '_sc_new'}),
                on=['GAME_ID', 'TEAM_ID'], how='left'
            )
            combined.loc[new_mask & combined['STAR_COUNT'].isna(), 'STAR_COUNT'] = \
                combined.loc[new_mask & combined['STAR_COUNT'].isna(), '_sc_new']
            combined.drop(columns=['_star_flag', '_sc_new'], inplace=True, errors='ignore')

        # 2. TEAM_AVG_STARS → USAGE_VACUUM for new rows
        if 'STAR_COUNT' in combined.columns:
            # Per player, update TEAM_AVG_STARS as expanding mean of STAR_COUNT
            for pid in updated_pids:
                mask = combined['PLAYER_ID'] == pid
                sc = combined.loc[mask, 'STAR_COUNT'].sort_index()
                team_avg = sc.shift(1).expanding().mean()
                combined.loc[mask, 'TEAM_AVG_STARS'] = team_avg.values
                uv = (team_avg - sc).clip(lower=0).fillna(0)
                combined.loc[mask, 'USAGE_VACUUM'] = uv.values

        # 3. PACE_ROLLING and slow-changing team/opp features: forward-fill per player
        slow_features = ['PACE_ROLLING', 'OPP_PTS_ALLOWED', 'OPP_PTS_ALLOWED_DIFF',
                         'OPP_REB_ALLOWED', 'OPP_AST_ALLOWED', 'OPP_FGA_ALLOWED',
                         'OPP_WIN_PCT', 'DAYS_REST', 'IS_4_IN_6', 'IS_FRESH']
        slow_present = [f for f in slow_features if f in combined.columns]
        if slow_present:
            for pid in updated_pids:
                mask = combined['PLAYER_ID'] == pid
                combined.loc[mask, slow_present] = (
                    combined.loc[mask, slow_present].ffill()
                )
            _log(f"   ↪ Forward-filled {len(slow_present)} team/opp features for {len(updated_pids)} players")

        # 4. Recompute PACE_ROLLING for teams with new games
        #    Formula: team's rolling 10-game mean of PACE_PER_48 (shifted by 1 game)
        if all(c in combined.columns for c in ['FGA', 'FTA', 'TOV', 'MIN']):
            updated_tids = new_rows['TEAM_ID'].unique()
            for _tid in updated_tids:
                _tmask = combined['TEAM_ID'] == _tid
                _tdf   = combined[_tmask].sort_values('GAME_DATE').copy()
                _mins  = _tdf['MIN'].replace(0, 0.1)
                _oreb  = _tdf['OREB'] if 'OREB' in _tdf.columns else 0
                _poss  = _tdf['FGA'] + 0.44 * _tdf['FTA'] - _oreb + _tdf['TOV']
                _tdf['_p48'] = (_poss / _mins * 48).clip(0, 200)
                _game_pace = _tdf.groupby('GAME_ID')['_p48'].transform('mean')
                _tdf['_game_pace'] = _game_pace
                _tdf_by_game = _tdf.drop_duplicates('GAME_ID').sort_values('GAME_DATE')
                _rolling = _tdf_by_game['_game_pace'].shift(1).rolling(10, min_periods=3).mean()
                _gid_to_pace = dict(zip(_tdf_by_game['GAME_ID'], _rolling))
                pace_vals = _tdf['GAME_ID'].map(_gid_to_pace)
                combined.loc[_tmask, 'PACE_ROLLING'] = pace_vals.values
            combined['PACE_ROLLING'] = combined['PACE_ROLLING'].fillna(combined['PACE_ROLLING'].median())
            _log(f"   ↪ Recomputed PACE_ROLLING for {len(updated_tids)} teams")

        new_latest = combined['GAME_DATE'].max()
        _log(f"   ✅ Data refreshed: now {len(combined):,} rows up to {new_latest.date()}")

        # If 1H data wasn't fetched, forward-fill so latest rows inherit last known values
        if api_1h_df is None or api_1h_df.empty:
            _1h_rolling_cols = [c for c in combined.columns
                               if ('_1H' in c and any(c.endswith(s) for s in
                                   ['_L5', '_L10', '_L20', '_Season', '_L5_Median', '_L10_Median']))]
            if _1h_rolling_cols:
                for pid in updated_pids:
                    mask = combined['PLAYER_ID'] == pid
                    combined.loc[mask, _1h_rolling_cols] = (
                        combined.loc[mask, _1h_rolling_cols].ffill()
                    )
                _log(f"   ↪ Forward-filled {len(_1h_rolling_cols)} 1H rolling columns (1H API unavailable)")

        # Persist to disk so next run doesn't re-fetch
        combined.to_csv(DATA_FILE, index=False)
        _log(f"   💾 Saved to {os.path.basename(DATA_FILE)}")

        return combined

    except Exception as e:
        print(f"   ⚠️  Auto-refresh failed ({e}), using existing dataset")
        return df_history


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
    _log("...Building data cache for fast lookups")

    # Filter to current season only — prevents traded players from inflating
    # team rosters (e.g. Deandre Ayton still mapping to the Suns)
    season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
    current_season = df_history[season_col].max()
    df_current = df_history[df_history[season_col] == current_season]
    _log(f"   Using season {current_season} ({df_current['PLAYER_ID'].nunique()} players)")

    # Sort by date; forward-fill slow-changing features before taking the latest row.
    # Auto-refresh adds raw stats but leaves team-level features (STAR_COUNT, USAGE_VACUUM,
    # PACE_ROLLING, OPP_*) as NaN for the newest game rows. Forward-fill ensures the
    # latest row always has a real value rather than feeding XGBoost's missing-value path
    # (which was trained on zeros and suppresses projections).
    SLOW_FEATURES = [
        'STAR_COUNT', 'TEAM_AVG_STARS', 'USAGE_VACUUM', 'PACE_ROLLING',
        'OPP_PTS_ALLOWED', 'OPP_PTS_ALLOWED_DIFF',
        'OPP_REB_ALLOWED', 'OPP_AST_ALLOWED', 'OPP_FGA_ALLOWED',
        'OPP_FG3M_ALLOWED', 'OPP_STL_ALLOWED', 'OPP_BLK_ALLOWED',
        'OPP_TOV_ALLOWED', 'OPP_FGM_ALLOWED', 'OPP_FTM_ALLOWED', 'OPP_FTA_ALLOWED',
        'OPP_SB_ALLOWED', 'OPP_WIN_PCT',
        'OPP_PRA_ALLOWED', 'OPP_PR_ALLOWED', 'OPP_PA_ALLOWED',
        'OPP_PTS_ALLOWED_DIFF', 'OPP_REB_ALLOWED_DIFF', 'OPP_AST_ALLOWED_DIFF',
        'OPP_FGA_ALLOWED_DIFF',
    ]
    slow_present = [f for f in SLOW_FEATURES if f in df_current.columns]

    df_sorted = df_current.sort_values(['PLAYER_ID', 'GAME_DATE']).copy()
    if slow_present:
        for _f in slow_present:
            df_sorted[_f] = df_sorted.groupby('PLAYER_ID')[_f].ffill()

    df_latest = df_sorted.drop_duplicates(subset=['PLAYER_ID'], keep='last')

    n_fixed = df_latest[slow_present].isna().sum().sum() if slow_present else 0
    if n_fixed > 0:
        _log(f"   ⚠️  {n_fixed} slow-feature NaNs remain after ffill (new players with no history)")

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
            _log(f"   Fetching today's scoreboard...", flush=True)
            resp = requests.get(CDN_SCOREBOARD_URL, headers=CDN_HEADERS, timeout=CDN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            games = data.get('scoreboard', {}).get('games', [])
            df = _cdn_scoreboard_to_df(games)
            if not df.empty:
                return df
        except Exception as e:
            _log(f"   CDN scoreboard failed, trying schedule...", flush=True)

    # Fallback / future dates: use the full schedule
    return _fetch_schedule_cdn(game_date)


def _fetch_schedule_cdn(game_date):
    """Fetch games for a specific date from the cdn.nba.com season schedule."""
    from datetime import datetime as _dt
    target = _dt.strptime(game_date, '%Y-%m-%d').date()
    try:
        _log(f"   Fetching schedule...", flush=True)
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
        _log(f"   CDN schedule failed", flush=True)
        return None


def _fetch_scoreboard_statsapi(game_date, retries=NBA_API_RETRIES):
    """Fetch scoreboard from stats.nba.com (fallback — often slow/unreliable)."""
    for attempt in range(retries):
        try:
            if attempt > 0:
                _log(f"   Retry {attempt + 1}/{retries}...", flush=True)
            else:
                _log(f"   Trying stats.nba.com fallback...", flush=True)
            sys.stdout.flush()
            board = ScoreboardV2(
                game_date=game_date, league_id='00', day_offset=0, timeout=NBA_API_TIMEOUT
            )
            return board.game_header.get_data_frame()
        except _REQUEST_ERRORS as e:
            if attempt < retries - 1:
                _log(f"   Request failed, retrying in {NBA_API_RETRY_DELAY}s...", flush=True)
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
            _log(f"   Got {len(df)} games from cdn.nba.com", flush=True)
            return df
        elif df is not None:
            # CDN responded but no games on this date
            return df
    except Exception as e:
        _log(f"   cdn.nba.com failed, falling back...", flush=True)

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
    
    _log(f"...Checking for games on {target_date}")

    try:
        games = _fetch_scoreboard(target_date)

        if not games.empty:
            if require_scheduled:
                scheduled_games = games[games['GAME_STATUS_ID'] == 1]
                if not scheduled_games.empty:
                    _log(f"Found {len(scheduled_games)} scheduled games on {target_date}")
                    return _build_team_map(scheduled_games), target_date
            else:
                _log(f"Found {len(games)} games on {target_date}")
                return _build_team_map(games), target_date

        # No games on requested date - search forward
        _log(f"   No games on {target_date}. Searching forward...")

        for days_ahead in range(1, max_days_forward + 1):
            search_date = (initial_date + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

            if days_ahead % 2 == 0 or days_ahead == 1:
                _log(f"   Checking {search_date}...", end='\r')

            try:
                games = _fetch_scoreboard(search_date)

                if not games.empty:
                    if require_scheduled:
                        scheduled_games = games[games['GAME_STATUS_ID'] == 1]
                        if not scheduled_games.empty:
                            day_name = (initial_date + timedelta(days=days_ahead)).strftime('%A')
                            _log(f"\nFound {len(scheduled_games)} games on {search_date} ({day_name})")
                            return _build_team_map(scheduled_games), search_date
                    else:
                        if not games.empty:
                            day_name = (initial_date + timedelta(days=days_ahead)).strftime('%A')
                            _log(f"\nFound {len(games)} games on {search_date} ({day_name})")
                            return _build_team_map(games), search_date

            except Exception as e:
                continue

        _log(f"\nNo scheduled games found in the next {max_days_forward} days")
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
    global _QUIET
    _QUIET = True  # Suppress internal loading noise
    try:
        return _scan_all_impl(df_history, models, is_tomorrow, max_days_forward)
    finally:
        _QUIET = False


def _scan_all_impl(df_history, models, is_tomorrow=False, max_days_forward=7):
    global _QUIET
    refresh_injuries()
    n_out = sum(1 for s in INJURY_DATA.values() if s == 'OUT')
    n_gtd = sum(1 for s in INJURY_DATA.values() if s != 'OUT')

    offset = 1 if is_tomorrow else 0

    todays_teams, actual_date = get_games(
        date_offset=offset,
        require_scheduled=True,
        max_days_forward=max_days_forward
    )

    if not todays_teams:
        _QUIET = False
        print("No scheduled games found in the next 7 days.")
        input("\nPress Enter to continue...")
        return

    pp_client = PrizePicksClient(stat_map=STAT_MAP)
    live_lines = pp_client.fetch_lines_dict(league_filter='NBA')
    n_pp_lines = len(live_lines)

    norm_lines = {normalize_name(k): v for k, v in live_lines.items()}

    # --- Load FanDuel cache to cross-reference lines ---
    import json, math as _math
    fd_cache_file = os.path.join(BASE_DIR, 'fanduel_cache', 'fanduel_cache_nba.json')
    fd_lines_by_player = {}  # {normalized_name: {stat_code: fd_line}}

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
            with open(fd_cache_file, 'r') as f:
                fd_raw = json.load(f)
            for entry in fd_raw:
                p_name = normalize_name(entry.get('Player', ''))
                raw_stat = entry.get('Stat', '')
                stat_code = _FD_TO_CODE.get(raw_stat, raw_stat)
                line = entry.get('Line', 0)
                if not p_name or not stat_code or not line:
                    continue
                if p_name not in fd_lines_by_player:
                    fd_lines_by_player[p_name] = {}
                fd_lines_by_player[p_name][stat_code] = line
        except Exception:
            pass

    # --- PRE-BUILD CACHE (O(N) once) ---
    latest_rows_map, team_rosters_map = build_data_cache(df_history)

    # Pre-index player logs so hit-rate lookups are O(1) instead of O(N) per call
    player_logs_index = {
        pid: grp.sort_values('GAME_DATE')
        for pid, grp in df_history.groupby('PLAYER_ID')
    }

    _QUIET = False  # Re-enable output — status block goes here

    # ── CLEAN STATUS BLOCK ────────────────────────────────────────────────────
    if actual_date:
        scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
        day_name = scan_date_obj.strftime('%A, %B %d, %Y')
    else:
        day_name = datetime.now().strftime('%A, %B %d, %Y')

    n_games  = len(todays_teams) // 2
    n_models = len(models)
    n_fd     = len(fd_lines_by_player)
    data_date = df_history['GAME_DATE'].max().strftime('%b %d, %Y')
    today_str = datetime.now().strftime('%b %d')
    freshness = 'refreshed today' if data_date == today_str else f'last game {data_date}'

    W = 70
    print(f"\n{'━' * W}")
    print(f"  NBA SCANNER  ·  {day_name}  ·  {n_games} games")
    print(f"{'━' * W}")
    print(f"  ● Models    {n_models} loaded")
    print(f"  ● Data      {freshness}")
    print(f"  ● Injuries  {n_out} OUT  ·  {n_gtd} GTD")
    print(f"  ● Lines     {n_pp_lines} PrizePicks  ·  {n_fd} FanDuel")
    print(f"{'━' * W}")

    # ── GTD WARNING SECTION ───────────────────────────────────────────────────
    gtd_players_today = []
    for inj_name, status in INJURY_DATA.items():
        if status == 'OUT':
            continue  # OUT players are already excluded from predictions

        norm_gtd = normalize_name(inj_name)
        matched_pid, matched_row = None, None
        for pid, row in latest_rows_map.items():
            if normalize_name(row.get('PLAYER_NAME', '')) == norm_gtd:
                matched_pid, matched_row = pid, row
                break

        if not matched_pid or not matched_row:
            continue

        tid = matched_row.get('TEAM_ID')
        if tid not in todays_teams:
            continue  # Not playing today

        gtd_pts   = matched_row.get('PTS_Season') or 0
        gtd_reb   = matched_row.get('REB_Season') or 0
        gtd_ast   = matched_row.get('AST_Season') or 0
        gtd_usage = matched_row.get('USAGE_RATE_Season') or 0

        # Compute which teammates benefit most if this player sits
        teammates = team_rosters_map.get(tid, [])
        active_mates, total_usage = [], 0.0
        for tpid in teammates:
            if tpid == matched_pid:
                continue
            trow = latest_rows_map.get(tpid)
            if not trow:
                continue
            if get_player_status(trow.get('PLAYER_NAME', '')) == 'OUT':
                continue
            tusage = trow.get('USAGE_RATE_Season') or 0
            if tusage > 5:
                active_mates.append((trow.get('PLAYER_NAME', ''), tusage))
                total_usage += tusage

        beneficiaries = []
        for tname, tusage in active_mates:
            share = tusage / total_usage if total_usage > 0 else 0
            pts_boost = gtd_pts * share * 0.40  # 40% absorption rate
            if pts_boost >= 0.3:
                beneficiaries.append((tname, pts_boost))
        beneficiaries.sort(key=lambda x: -x[1])

        gtd_players_today.append({
            'name': inj_name, 'status': status,
            'pts': gtd_pts, 'reb': gtd_reb, 'ast': gtd_ast, 'usage': gtd_usage,
            'beneficiaries': beneficiaries[:3],
        })

    if gtd_players_today:
        print(f"\n{'─' * W}")
        print(f"  ⚠  GTD WATCH  —  {len(gtd_players_today)} player(s) questionable today")
        print(f"  Check lineup news before locking bets — projections assume they play.")
        print(f"{'─' * W}")
        for w in gtd_players_today:
            print(f"\n  {w['name']}  [{w['status']}]  —  "
                  f"{w['pts']:.1f} pts · {w['reb']:.1f} reb · {w['ast']:.1f} ast/g  "
                  f"({w['usage']:.0f}% usage)")
            if w['beneficiaries']:
                print(f"  If OUT, estimated pts boost for teammates:")
                for bname, pts_b in w['beneficiaries']:
                    print(f"    +{pts_b:.1f} pts  →  {bname}")
        print(f"\n{'─' * W}\n")

    # Build team_id → abbreviation lookup for H2H
    team_id_to_abbr = {}
    for pid, row in latest_rows_map.items():
        tid = row.get('TEAM_ID')
        abbr = row.get('TEAM_ABBREVIATION', '')
        if tid and isinstance(abbr, str) and abbr:
            team_id_to_abbr[tid] = abbr

    best_bets = []
    all_projections = []
    avail_cache = {}  # Cache per-player availability analysis

    # Pre-compute fresh opponent defensive stats for each team in today's games.
    # At inference, input_row['OPP_PTS_ALLOWED'] comes from the player's last game
    # row (stale — different opponent). We override it with stats from today's actual
    # opponent's last 10 games so the matchup context is always current.
    _OPP_STAT_COLS = {
        'OPP_PTS_ALLOWED': 'PTS', 'OPP_REB_ALLOWED': 'REB', 'OPP_AST_ALLOWED': 'AST',
        'OPP_FG3M_ALLOWED': 'FG3M', 'OPP_FGA_ALLOWED': 'FGA', 'OPP_FGM_ALLOWED': 'FGM',
        'OPP_FTM_ALLOWED': 'FTM', 'OPP_FTA_ALLOWED': 'FTA',
        'OPP_STL_ALLOWED': 'STL', 'OPP_BLK_ALLOWED': 'BLK', 'OPP_TOV_ALLOWED': 'TOV',
    }
    _season_col = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
    _cur_season = df_history[_season_col].max()
    _df_cur = df_history[df_history[_season_col] == _cur_season]

    # Pre-compute fresh PACE_ROLLING for each team playing today.
    # Each team's pace = rolling 10-game mean of their POSS_EST/48 from recent history.
    fresh_pace_cache = {}  # team_id → float
    for _tid in todays_teams:
        _tgames = _df_cur[_df_cur['TEAM_ID'] == _tid].copy()
        if _tgames.empty or not all(c in _tgames.columns for c in ['FGA', 'FTA', 'TOV', 'MIN']):
            continue
        _mins  = _tgames['MIN'].replace(0, 0.1)
        _oreb  = _tgames['OREB'] if 'OREB' in _tgames.columns else 0
        _poss  = _tgames['FGA'] + 0.44 * _tgames['FTA'] - _oreb + _tgames['TOV']
        _tgames['_p48'] = (_poss / _mins * 48).clip(0, 200)
        _by_game = _tgames.groupby('GAME_ID').agg({'_p48': 'mean', 'GAME_DATE': 'first'})
        _by_game = _by_game.sort_values('GAME_DATE')
        _last10_mean = _by_game['_p48'].iloc[-10:].mean()
        if not pd.isna(_last10_mean):
            fresh_pace_cache[_tid] = float(_last10_mean)

    fresh_opp_stats_cache = {}  # team_id → {OPP_PTS_ALLOWED: val, ...}
    for _tid in todays_teams:
        _opp_id = todays_teams[_tid].get('opp')
        if not _opp_id or _opp_id in fresh_opp_stats_cache:
            continue
        # Find games played BY the opponent team and compute their per-player-game averages
        _opp_games = _df_cur[_df_cur['TEAM_ID'] == _opp_id].sort_values('GAME_DATE')
        _last10_gids = _opp_games['GAME_ID'].unique()[-10:]
        _opp_recent = _opp_games[_opp_games['GAME_ID'].isin(_last10_gids)]
        # Get opposing players in those games (what the opponent "allowed")
        _opp_facing = _df_cur[
            _df_cur['GAME_ID'].isin(_last10_gids) & (_df_cur['TEAM_ID'] != _opp_id)
        ]
        if _opp_facing.empty:
            continue
        _stats = {}
        for _opp_col, _raw_col in _OPP_STAT_COLS.items():
            if _raw_col in _opp_facing.columns:
                _stats[_opp_col] = float(_opp_facing[_raw_col].mean())
        # Diff = how much they allow vs league average
        for _opp_col, _raw_col in _OPP_STAT_COLS.items():
            _diff_col = _opp_col + '_DIFF'
            if _raw_col in _opp_facing.columns and _raw_col in _df_cur.columns:
                _league_avg = float(_df_cur[_raw_col].mean())
                _stats[_diff_col] = _stats.get(_opp_col, _league_avg) - _league_avg
        fresh_opp_stats_cache[_opp_id] = _stats

    for team_id, info in todays_teams.items():
        # O(1) Lookup for roster
        team_players = team_rosters_map.get(team_id, [])

        # Calculate missing usage + count of OUT teammates (for USAGE_VACUUM fix)
        missing_usage_today = 0.0
        team_out_count = 0
        for pid in team_players:
            # O(1) Lookup for player data
            last_row = latest_rows_map.get(pid)
            if not last_row: continue

            pname = last_row['PLAYER_NAME']
            if get_player_status(pname) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15:
                    missing_usage_today += usage
                    team_out_count += 1

        # Generate predictions for each player
        for pid in team_players:
            last_row = latest_rows_map.get(pid)
            if not last_row: continue

            player_name = last_row['PLAYER_NAME']

            if get_player_status(player_name) == 'OUT':
                continue

            # Compute actual DAYS_REST from last game date
            last_game_date = last_row.get('GAME_DATE')
            if last_game_date and actual_date:
                try:
                    last_dt = pd.to_datetime(last_game_date)
                    scan_dt = pd.to_datetime(actual_date)
                    days_rest = max(1, (scan_dt - last_dt).days)
                    days_rest = min(days_rest, 7)  # Cap at 7 like features.py
                except Exception:
                    days_rest = 2
            else:
                days_rest = 2

            # Cache availability analysis per player (not per stat)
            if pid not in avail_cache:
                avail_cache[pid] = analyze_player_availability(df_history, pid, actual_date)

            input_row = prepare_features(
                last_row,
                is_home=info['is_home'],
                days_rest=days_rest,
                missing_usage=missing_usage_today
            )

            # Fix stale USAGE_VACUUM: each OUT teammate reduces STAR_COUNT by 1
            if team_out_count > 0:
                old_uv = float(input_row['USAGE_VACUUM'].iloc[0]) if 'USAGE_VACUUM' in input_row.columns else 0.0
                input_row['USAGE_VACUUM'] = old_uv + team_out_count

            # Inject fresh PACE_ROLLING for this team's last 10 games
            if team_id in fresh_pace_cache and 'PACE_ROLLING' in input_row.columns:
                input_row['PACE_ROLLING'] = fresh_pace_cache[team_id]

            # Inject fresh opponent stats for today's actual matchup (overrides stale last-game opp)
            _opp_id = info.get('opp')
            if _opp_id and _opp_id in fresh_opp_stats_cache:
                for _col, _val in fresh_opp_stats_cache[_opp_id].items():
                    if _col in input_row.columns:
                        input_row[_col] = _val

            player_predictions = {}
            features_ok = True

            for target, model in models.items():
                if not features_ok: break
                try:
                    # Filter for features actually used by THIS model
                    model_features = [f for f in model.feature_names_in_]
                    valid_input = input_row.reindex(columns=model_features, fill_value=0)

                    # 1H rolling stats are NaN or 0 in auto-refresh rows (no 1H
                    # box scores available for recently-added games). Backfill
                    # L5/L10/L20 from Season so the model gets a real signal.
                    if target in ('PTS_1H', 'PRA_1H', 'FPTS_1H'):
                        for _stat in ['PTS_1H', 'PRA_1H', 'FPTS_1H', 'MIN_1H', 'PRA', 'PTS']:
                            _season_col = f'{_stat}_Season'
                            if _season_col not in valid_input.columns:
                                continue
                            _season_val = valid_input[_season_col].iloc[0]
                            if pd.isna(_season_val) or _season_val <= 0:
                                continue
                            _season_val = float(_season_val)
                            for _suf in ['_L5', '_L10', '_L20', '_L5_Median', '_L10_Median']:
                                _col = f'{_stat}{_suf}'
                                if _col not in valid_input.columns:
                                    continue
                                _cur = valid_input[_col].iloc[0]
                                if pd.isna(_cur) or float(_cur) == 0:
                                    valid_input[_col] = _season_val

                    raw = float(model.predict(valid_input)[0])
                    if target in LOG_TRANSFORM_TARGETS:
                        proj = float(np.expm1(max(raw, 0))) * LOG_CALIBRATION.get(target, 1.0)
                    else:
                        proj = max(raw, 0.0)
                    player_predictions[target] = proj
                except Exception:
                    # Model mismatch or missing features
                    features_ok = False
            
            if not features_ok: continue

            # Scale projections for injury-return / minute restrictions
            # This affects ALL stats — a player on a minute restriction or
            # returning from injury produces less across the board.
            avail = avail_cache.get(pid, {'penalty': 0, 'scale_factor': 1.0, 'flag': '', 'reason': ''})
            if avail['scale_factor'] < 1.0:
                for tgt in player_predictions:
                    player_predictions[tgt] *= avail['scale_factor']

            # NOTE: Injury redistribution (Layer 2) intentionally removed.
            # MISSING_USAGE is fed into XGBoost as a feature (Layer 1), and the
            # model already learned the injury boost from historical training data.
            # Adding a second manual bump on top caused double-counting and
            # inflated projections when teammates were out.

            # Apply correlation constraints (average model + components to avoid bias)
            pts = player_predictions.get('PTS', 0)
            reb = player_predictions.get('REB', 0)
            ast = player_predictions.get('AST', 0)
            stl = player_predictions.get('STL', 0)
            blk = player_predictions.get('BLK', 0)

            if 'PRA' in player_predictions:
                component_sum = pts + reb + ast
                player_predictions['PRA'] = (player_predictions['PRA'] + component_sum) / 2

            if 'PR' in player_predictions:
                player_predictions['PR'] = (player_predictions['PR'] + pts + reb) / 2

            if 'PA' in player_predictions:
                player_predictions['PA'] = (player_predictions['PA'] + pts + ast) / 2

            if 'RA' in player_predictions:
                player_predictions['RA'] = (player_predictions['RA'] + reb + ast) / 2

            if 'SB' in player_predictions:
                player_predictions['SB'] = (player_predictions['SB'] + stl + blk) / 2

            # Create recommendations
            for target, proj in player_predictions.items():
                line = norm_lines.get(normalize_name(player_name), {}).get(target)
                rec = get_betting_indicator(proj, line)
                
                # Default empty pp/edge if no line
                pp_val   = round(line, 2) if line else 0
                edge_val = round(proj - line, 2) if line else 0

                # Calculate Hit Rates (O(1) via pre-built index)
                l5_hit, l10_hit, l20_hit = calculate_hit_rates(df_history, pid, target, line, player_logs_index)

                # Calculate H2H Hit Rate
                opp_id = info.get('opp')
                opp_abbr = team_id_to_abbr.get(opp_id, '')
                h2h_hit, h2h_n = calculate_h2h_hit_rate(df_history, pid, target, line, opp_abbr, player_logs_index)

                all_projections.append({
                    'REC': rec,
                    'NAME': player_name,
                    'TARGET': target,
                    'AI': round(proj, 2),
                    'PP': pp_val,
                    'EDGE': edge_val,
                    'L5_HIT': l5_hit,
                    'L10_HIT': l10_hit,
                    'L20_HIT': l20_hit,
                    'H2H_HIT': h2h_hit,
                    'H2H_N': h2h_n
                })

                # Calculate Confidence Score
                line_diff_for_hit = l10_hit if edge_val > 0 else (1 - l10_hit)
                
                # Detect Role Expansion (High variance traps for Unders)
                is_role_expansion = False
                min_l5 = last_row.get('MIN_L5', 0)
                min_season = last_row.get('MIN_Season', 0)
                
                # If they are stepping into huge minutes relative to season avg, OR massive team injuries
                if (min_season > 5 and min_l5 / min_season > 1.4) or missing_usage_today > 25.0:
                    is_role_expansion = True
                
                # Fetch opponent context for Matchup Score from last_row (avoids stale valid_input scope)
                opp_win_pct = last_row.get('OPP_WIN_PCT')

                # Low-Line Mathematical Variance Fix:
                # Dividing by 0.5 can inflate a 0.38 block edge into a 76% edge.
                # We enforce a baseline statistical denominator of 2.0.
                safe_denominator = max(line, 2.0) if line else 2.0
                
                pct_edge_safe = (edge_val / safe_denominator) * 100
                conf_score = calculate_confidence_score(pct_edge_safe, line_diff_for_hit, opp_win_pct, is_role_expansion)

                if line is not None and line > 0:
                    # ── Line Diff validation ──
                    # Skip plays where PP line differs from FD line by more than 0.5
                    norm_name = normalize_name(player_name)
                    fd_line = fd_lines_by_player.get(norm_name, {}).get(target)
                    if fd_line is not None:
                        line_diff = abs(line - fd_line)
                        if line_diff > 0.5:
                            continue  # Lines don't match — skip this play

                    edge = proj - line
                    pct_edge = (edge / safe_denominator) * 100

                    tier_info = MODEL_QUALITY.get(target, {})
                    edge_threshold = tier_info.get('threshold', 2.5)

                    # Only surface plays with meaningful edge above model tier threshold
                    if abs(pct_edge) >= edge_threshold:
                        # ── MULTI-SIGNAL CONFIRMATION FILTER ──────────────────────
                        # Skip bets where historical data strongly contradicts the AI.
                        # Derived from backtest analysis: when L10 hit-rate strongly
                        # opposes the model's direction, win rates drop below 48%.
                        ai_says_over = edge > 0

                        # L10 veto: player almost never goes the direction AI predicts
                        l10_vetoes = (
                            (ai_says_over and l10_hit < 0.25) or
                            (not ai_says_over and l10_hit > 0.75)
                        )

                        # H2H veto: in head-to-head matchups AI direction is very rare
                        h2h_vetoes = (
                            h2h_n >= 4 and (
                                (ai_says_over and h2h_hit < 0.20) or
                                (not ai_says_over and h2h_hit > 0.80)
                            )
                        )

                        if l10_vetoes or h2h_vetoes:
                            continue  # Historical data overrules the model
                        # ──────────────────────────────────────────────────────────

                        # Use cached availability analysis
                        avail = avail_cache.get(pid, {'penalty': 0, 'flag': '', 'reason': ''})

                        best_bets.append({
                            'REC': rec,
                            'NAME': player_name,
                            'TARGET': target,
                            'AI': round(proj, 2),
                            'PP': round(line, 2),
                            'EDGE': edge,
                            'PCT_EDGE': pct_edge,
                            'TIER': tier_info.get('tier', 'UNKNOWN'),
                            'THRESHOLD': edge_threshold,
                            'L5_HIT': l5_hit,
                            'L10_HIT': l10_hit,
                            'L20_HIT': l20_hit,
                            'H2H_HIT': h2h_hit,
                            'H2H_N': h2h_n,
                            'CONFIDENCE': conf_score,
                            'AVAIL': avail
                        })

    # ══════════════════════════════════════════════════════════════════════
    # PLAYS OF THE DAY — Quality-over-quantity display
    # ══════════════════════════════════════════════════════════════════════

    # Save full projections regardless of display filter
    if actual_date:
        save_path = os.path.join(PROJ_DIR, f"scan_{actual_date}.csv")
    else:
        save_path = TOMORROW_SCAN_FILE if is_tomorrow else TODAY_SCAN_FILE
    if all_projections:
        pd.DataFrame(all_projections).to_csv(save_path, index=False)

    if best_bets:
        # ✅ DEDUPLICATE
        seen = set()
        deduped_bets = []
        for bet in best_bets:
            key = (bet['NAME'], bet['TARGET'], bet['PP'])
            if key not in seen:
                seen.add(key)
                deduped_bets.append(bet)

        # ── STRICT QUALITY FILTER ──────────────────────────────────────────
        # Only keep plays from reliable model tiers
        TRUSTED_TIERS = {'ELITE', 'STRONG', 'DECENT'}
        MAX_PER_PLAYER = 2   # Diversity: don't load up on one player

        qualified = []
        for bet in deduped_bets:
            if bet['TIER'] not in TRUSTED_TIERS:
                continue

            is_over = bet['EDGE'] > 0
            # L10 hit rate alignment: for overs, need high over-hit %;
            # for unders, need high miss % (= 1 - L10_HIT)
            aligned_hit = bet['L10_HIT'] if is_over else (1 - bet['L10_HIT'])

            # ── Composite Quality Score (0-100) ──
            import math

            # 1. Edge magnitude score (0-30 pts)
            #    Logarithmic scaling: first 5% edge is most valuable,
            #    diminishing returns after that. Caps at ~30 pts.
            abs_edge = abs(bet['PCT_EDGE'])
            edge_score = min(30.0, 30.0 * math.log1p(abs_edge) / math.log1p(15.0))

            # Penalty for suspiciously large edges (>30% = likely model misfire)
            if abs_edge > 30.0:
                edge_score *= 0.7  # Discount — something is probably wrong

            # 2. Hit rate alignment score (0-25 pts)
            #    Scales across full range (0-100%), not gated
            hit_score = aligned_hit * 25.0

            # 3. Model tier score (0-20 pts)
            tier_scores = {'ELITE': 20.0, 'STRONG': 14.0, 'DECENT': 8.0}
            tier_score = tier_scores.get(bet['TIER'], 0)

            # 4. Cross-window consistency bonus (0-15 pts)
            #    If L5, L10, AND L20 all agree, that's a strong signal
            l5_aligned  = bet['L5_HIT']  if is_over else (1 - bet['L5_HIT'])
            l20_aligned = bet['L20_HIT'] if is_over else (1 - bet['L20_HIT'])

            consistency_count = sum(1 for r in [l5_aligned, aligned_hit, l20_aligned] if r >= 0.60)
            consistency_score = (consistency_count / 3.0) * 15.0

            # 5. Edge realism bonus (0-10 pts)
            #    Sweet spot is 3-15% edge — not too small, not suspiciously huge
            if 3.0 <= abs_edge <= 15.0:
                realism_bonus = 10.0
            elif abs_edge < 3.0:
                realism_bonus = (abs_edge / 3.0) * 5.0  # Small edge = less confident
            else:
                realism_bonus = max(0, 10.0 - (abs_edge - 15.0) * 0.3)  # Penalize extremes

            quality = edge_score + hit_score + tier_score + consistency_score + realism_bonus

            # 6. Dynamic injury-return / minute restriction penalty
            avail = bet.get('AVAIL', {'penalty': 0, 'flag': '', 'reason': ''})
            quality += avail['penalty']  # penalty is negative
            inj_return_flag = avail['flag']

            quality = max(0, quality)  # Floor at 0

            # Assign letter grade (calibrated for full-range hit scoring)
            if quality >= 92:
                grade = 'A+'
            elif quality >= 80:
                grade = 'A '
            elif quality >= 68:
                grade = 'B+'
            elif quality >= 55:
                grade = 'B '
            else:
                grade = 'C '

            bet['QUALITY'] = round(quality, 1)
            bet['GRADE'] = grade
            bet['ALIGNED_HIT'] = aligned_hit
            bet['CONSISTENCY'] = consistency_count
            bet['INJ_FLAG'] = inj_return_flag
            qualified.append(bet)

        # Sort by quality score (highest first)
        qualified.sort(key=lambda b: -b['QUALITY'])

        # ── PER-PLAYER CAP: max 2 plays per player for diversity ──
        player_counts = {}
        top_plays = []
        MAX_PLAYS = 15
        for bet in qualified:
            name = bet['NAME']
            player_counts[name] = player_counts.get(name, 0) + 1
            if player_counts[name] <= MAX_PER_PLAYER:
                top_plays.append(bet)
            if len(top_plays) >= MAX_PLAYS:
                break

        # ── DISPLAY ────────────────────────────────────────────────────────
        scan_date_str = actual_date or datetime.now().strftime('%Y-%m-%d')

        if top_plays:
            print(f"\n{'═' * 110}")
            print(f"   🏆 PLAYS OF THE DAY — {scan_date_str}   ({len(top_plays)} plays from {len(deduped_bets)} scanned)")
            print(f"{'═' * 110}")
            print(f"   Filters: Trusted tiers · Edge above threshold · Max 2 per player")
            print(f"{'─' * 110}")
            print(f" {'#':>2} | {'GRD':^3} | {'PLAYER':<22} | {'STAT':<8} | {'PROJ':>6} {'LINE':>6} | {'EDGE':>7} | {'SIDE':<5} | {'L5':>4} {'L10':>4} {'L20':>4} | {'H2H':>7} | {'SCORE':>5} | {'FLAG':<4}")
            print(f"{'─' * 110}")

            for i, bet in enumerate(top_plays, 1):
                is_over = bet['EDGE'] > 0
                side_str = 'OVER' if is_over else 'UNDER'
                edge_str = f"{bet['PCT_EDGE']:+.1f}%"
                target_str = bet['TARGET'].replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                flag_str = bet.get('INJ_FLAG', '')

                # Show hit rates from the correct perspective
                l5_str  = f"{bet['L5_HIT']*100:.0f}%" if is_over else f"{(1-bet['L5_HIT'])*100:.0f}%"
                l10_str = f"{bet['ALIGNED_HIT']*100:.0f}%"
                l20_str = f"{bet['L20_HIT']*100:.0f}%" if is_over else f"{(1-bet['L20_HIT'])*100:.0f}%"

                # H2H display: perspective-adjusted rate + sample size
                h2h_n = bet.get('H2H_N', 0)
                if h2h_n > 0:
                    h2h_rate = bet['H2H_HIT'] if is_over else (1 - bet['H2H_HIT'])
                    h2h_str = f"{h2h_rate*100:.0f}%({h2h_n})"
                else:
                    h2h_str = '  --  '

                print(f" {i:>2} | {bet['GRADE']:^3} | {bet['NAME'][:22]:<22} | {target_str:<8} | "
                      f"{bet['AI']:>6.1f} {bet['PP']:>6.1f} | {edge_str:>7} | {side_str:<5} | "
                      f"{l5_str:>4} {l10_str:>4} {l20_str:>4} | {h2h_str:>7} | {bet['QUALITY']:>5.1f} | {flag_str:<4}")

            print(f"{'─' * 110}")

            # Summary stats
            overs  = sum(1 for b in top_plays if b['EDGE'] > 0)
            unders = len(top_plays) - overs
            avg_quality = sum(b['QUALITY'] for b in top_plays) / len(top_plays)
            a_plus_count = sum(1 for b in top_plays if b['GRADE'] == 'A+')

            print(f"\n   📈 {overs} Overs · {unders} Unders · Avg Score: {avg_quality:.1f} · A+ Plays: {a_plus_count}")

            if a_plus_count > 0:
                print(f"\n   💎 TOP CONVICTION:")
                for bet in top_plays:
                    if bet['GRADE'] == 'A+':
                        side = 'OVER' if bet['EDGE'] > 0 else 'UNDER'
                        target_str = bet['TARGET'].replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                        print(f"      {bet['NAME']} {target_str} {side} {bet['PP']} "
                              f"(AI: {bet['AI']:.1f}, Edge: {bet['PCT_EDGE']:+.1f}%, "
                              f"L10: {bet['ALIGNED_HIT']*100:.0f}%)")

            print()

        # ── BEST BETS — All Three Signals Agree ───────────────────────────
        best_bet_candidates = []
        for bet in deduped_bets:
            is_over = bet['EDGE'] > 0
            abs_edge = abs(bet['PCT_EDGE'])
            aligned_l10 = bet['L10_HIT'] if is_over else (1 - bet['L10_HIT'])
            h2h_n = bet.get('H2H_N', 0)
            aligned_h2h = (bet['H2H_HIT'] if is_over else (1 - bet['H2H_HIT'])) if h2h_n > 0 else 0

            if (abs_edge >= 15.0 and aligned_l10 >= 0.60 and h2h_n >= 5 and aligned_h2h >= 0.60
                    and bet.get('TIER') in {'ELITE', 'STRONG', 'DECENT'}):
                bet['_BB_SCORE'] = abs_edge * aligned_l10 * aligned_h2h
                best_bet_candidates.append(bet)

        best_bet_candidates.sort(key=lambda b: -b['_BB_SCORE'])

        W = 115
        if best_bet_candidates:
            print(f"\n{'═' * W}")
            print(f"   🎯 BEST BETS — {scan_date_str}   (Edge ≥15% · L10 ≥60% · H2H ≥60% with 5+ games · all same direction)")
            print(f"{'═' * W}")
            print(f"   {'#':>3}  {'STAT':<10} {'PLAYER':<23} {'PROJ':>6} {'LINE':>6} {'EDGE':>8} {'L5':>4} {'L10':>4} {'H2H':>8}  {'TIER'}")
            print(f"{'─' * W}")
            for i, bet in enumerate(best_bet_candidates, 1):
                is_over = bet['EDGE'] > 0
                edge_str   = f"{bet['PCT_EDGE']:+.1f}%"
                target_str = bet['TARGET'].replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                l5_str  = f"{bet['L5_HIT']*100:.0f}%"  if is_over else f"{(1-bet['L5_HIT'])*100:.0f}%"
                l10_str = f"{bet['L10_HIT']*100:.0f}%" if is_over else f"{(1-bet['L10_HIT'])*100:.0f}%"
                h2h_n   = bet.get('H2H_N', 0)
                h2h_str = f"{((bet['H2H_HIT'] if is_over else 1-bet['H2H_HIT'])*100):.0f}%({h2h_n})" if h2h_n > 0 else '--'
                print(f"   {i:>3}  {target_str:<10} {bet['NAME'][:23]:<23} {bet['AI']:>6.1f} {bet['PP']:>6.1f} "
                      f"{edge_str:>8} {l5_str:>4} {l10_str:>4} {h2h_str:>8}  {bet['TIER']}")
            print(f"{'─' * W}")
            print(f"\n   {len(best_bet_candidates)} plays where model edge, recent form, AND opponent history all point the same way.\n")
        else:
            print(f"\n   🎯 BEST BETS: No plays today meet all three criteria (Edge ≥15%, L10 ≥60%, H2H ≥60%/5+).\n")

        # ── ALL MARKETS BREAKDOWN ──────────────────────────────────────────
        from collections import defaultdict
        market_groups = defaultdict(list)
        for bet in deduped_bets:
            market_groups[bet['TARGET']].append(bet)

        overs_list  = sorted([b for b in deduped_bets if b['EDGE'] > 0], key=lambda b: -b['PCT_EDGE'])[:20]
        unders_list = sorted([b for b in deduped_bets if b['EDGE'] < 0], key=lambda b:  b['PCT_EDGE'])[:20]

        def _print_side_table(plays, label):
            if not plays:
                return
            print(f"\n{'═' * 115}")
            print(f"   {label} — {scan_date_str}")
            print(f"{'═' * 115}")
            print(f"   {'#':>3}  {'STAT':<10} {'PLAYER':<23} {'PROJ':>6} {'LINE':>6} {'EDGE':>8} {'L5':>4} {'L10':>4} {'H2H':>8}  {'TIER'}")
            print(f"{'─' * 115}")
            for i, bet in enumerate(plays, 1):
                is_over = bet['EDGE'] > 0
                edge_str   = f"{bet['PCT_EDGE']:+.1f}%"
                target_str = bet['TARGET'].replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                l5_str  = f"{bet['L5_HIT']*100:.0f}%"  if is_over else f"{(1-bet['L5_HIT'])*100:.0f}%"
                l10_str = f"{bet['L10_HIT']*100:.0f}%" if is_over else f"{(1-bet['L10_HIT'])*100:.0f}%"
                h2h_n   = bet.get('H2H_N', 0)
                h2h_str = f"{((bet['H2H_HIT'] if is_over else 1-bet['H2H_HIT'])*100):.0f}%({h2h_n})" if h2h_n > 0 else '--'
                print(f"   {i:>3}  {target_str:<10} {bet['NAME'][:23]:<23} {bet['AI']:>6.1f} {bet['PP']:>6.1f} "
                      f"{edge_str:>8} {l5_str:>4} {l10_str:>4} {h2h_str:>8}  {bet['TIER']}")
            print(f"{'─' * 115}")

        if overs_list or unders_list:
            _print_side_table(overs_list,  "📈 TOP 20 OVERS")
            _print_side_table(unders_list, "📉 TOP 20 UNDERS")

        else:
            print(f"\n⚠️  No plays passed quality filters today ({len(deduped_bets)} scanned, 0 qualified)")
            print(f"   This means no strong edges + hit rate alignment found. Sit today out.\n")

        print(f"Full raw analysis ({len(all_projections)} rows) saved to {save_path}")
    else:
        print("\nNo active lines found.")

    input("\nPress Enter to continue...")


def prepare_features(player_row, is_home=0, days_rest=2, missing_usage=0):
    if isinstance(player_row, dict):
        features = player_row.copy()
    else:
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


def _get_position_category(pos):
    """Map raw NBA positions to Guard, Wing, Big for injury tracking."""
    pos = str(pos).upper()
    if 'C' in pos: return 'Big'
    if 'F' in pos: return 'Wing'
    return 'Guard'


def _calculate_injury_adjustments_fast(latest_rows_map, team_players, active_pid):
    """
    Optimized version of injury adjustments using pre-computed cache.
    Distributes 50% globally across usage, 50% positionally across usage.
    """
    out_production_global = {}
    out_production_pos = {'Guard': {}, 'Wing': {}, 'Big': {}}
    
    active_usage_global = {}
    active_usage_pos = {'Guard': {}, 'Wing': {}, 'Big': {}}

    active_last = latest_rows_map.get(active_pid, {})
    active_cat = _get_position_category(active_last.get('POSITION', 'G'))

    for pid in team_players:
        last = latest_rows_map.get(pid)
        if not last: continue
        
        pname = last['PLAYER_NAME']
        usage = last.get('USAGE_RATE_Season', 0)
        pos = last.get('POSITION', 'G')
        cat = _get_position_category(pos)

        if get_player_status(pname) == 'OUT':
            # Accumulate missing production
            for stat, col in _STAT_SEASON_COLS.items():
                val = last.get(col, 0)
                if pd.notna(val) and val > 0:
                     out_production_global[stat] = out_production_global.get(stat, 0) + val
                     out_production_pos[cat][stat] = out_production_pos[cat].get(stat, 0) + val
        else:
            if usage > 0:
                active_usage_global[pid] = usage
                active_usage_pos[cat][pid] = usage

    if not out_production_global or active_pid not in active_usage_global:
        return {}  # nothing to adjust

    # BLEND: 50% Positional, 50% Global
    total_active_usage = sum(active_usage_global.values())
    global_share = active_usage_global[active_pid] / total_active_usage if total_active_usage > 0 else 0

    cat_usage = sum(active_usage_pos[active_cat].values())
    pos_share = active_usage_pos[active_cat][active_pid] / cat_usage if cat_usage > 0 else 0

    adjustments = {}
    for stat in _STAT_SEASON_COLS.keys():
        rate = ABSORPTION_RATES.get(stat, 0.40)
        
        missing_global = out_production_global.get(stat, 0) * 0.5
        adj_global = missing_global * global_share * rate
        
        missing_pos = out_production_pos[active_cat].get(stat, 0) * 0.5
        adj_pos = missing_pos * pos_share * rate

        adj = adj_global + adj_pos
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

    # Get active player's category
    active_cat = 'Guard'
    active_rows = team_df[team_df['PLAYER_ID'] == active_pid].sort_values('GAME_DATE')
    if not active_rows.empty:
        active_cat = _get_position_category(active_rows.iloc[-1].get('POSITION', 'G'))

    # --- Gather last-row data for every teammate ---
    out_production_global = {}
    out_production_pos = {'Guard': {}, 'Wing': {}, 'Big': {}}
    
    active_usage_global = {}
    active_usage_pos = {'Guard': {}, 'Wing': {}, 'Big': {}}

    for pid in all_pids:
        p_rows = team_df[team_df['PLAYER_ID'] == pid].sort_values('GAME_DATE')
        if p_rows.empty:
            continue
        last = p_rows.iloc[-1]
        pname = last['PLAYER_NAME']
        usage = last.get('USAGE_RATE_Season', 0)
        pos = last.get('POSITION', 'G')
        cat = _get_position_category(pos)

        if get_player_status(pname) == 'OUT':
            # Accumulate missing production
            for stat, col in _STAT_SEASON_COLS.items():
                val = last.get(col, 0)
                if pd.notna(val) and val > 0:
                    out_production_global[stat] = out_production_global.get(stat, 0) + val
                    out_production_pos[cat][stat] = out_production_pos[cat].get(stat, 0) + val
        else:
            if usage > 0:
                active_usage_global[pid] = usage
                active_usage_pos[cat][pid] = usage

    if not out_production_global or active_pid not in active_usage_global:
        return {}  # nothing to adjust

    # BLEND: 50% Positional, 50% Global
    total_active_usage = sum(active_usage_global.values())
    global_share = active_usage_global[active_pid] / total_active_usage if total_active_usage > 0 else 0

    cat_usage = sum(active_usage_pos[active_cat].values())
    pos_share = active_usage_pos[active_cat][active_pid] / cat_usage if cat_usage > 0 else 0

    adjustments = {}
    for stat in _STAT_SEASON_COLS.keys():
        rate = ABSORPTION_RATES.get(stat, 0.40)
        
        missing_global = out_production_global.get(stat, 0) * 0.5
        adj_global = missing_global * global_share * rate
        
        missing_pos = out_production_pos[active_cat].get(stat, 0) * 0.5
        adj_pos = missing_pos * pos_share * rate

        adj = adj_global + adj_pos
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
        team_out_count = 0
        for pid in team_players:
            p_rows = df_current[df_current['PLAYER_ID'] == pid].sort_values('GAME_DATE')
            if p_rows.empty: continue
            last_row = p_rows.iloc[-1]
            if get_player_status(last_row['PLAYER_NAME']) == 'OUT':
                usage = last_row.get('USAGE_RATE_Season', 0)
                if usage > 15:
                    missing_usage_today += usage
                    team_out_count += 1

        print(f"\nSCOUTING REPORT: {name} ({actual_date})")
        print(f"Injury Impact: {team_out_count} teammate(s) OUT ({missing_usage_today:.1f}% usage missing)")

        inj_adj = {}

        # Analyze player availability (injury return / minute restriction)
        player_id = player_data['PLAYER_ID']
        avail = analyze_player_availability(df_history, player_id, actual_date)
        if avail['flag']:
            print(f"Availability: {avail['flag']} {avail['reason']}")

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
        print(f" {'TIER':<10} | {'STAT':<8} | {'PROJ':<14} | {'LINE':^11} | {'WIN%':>6} | SIDE")
        print("-" * 82)

        # Compute actual DAYS_REST for scout_player
        last_game_date = player_data.get('GAME_DATE')
        if last_game_date and actual_date:
            try:
                last_dt = pd.to_datetime(last_game_date)
                scan_dt = pd.to_datetime(actual_date)
                days_rest = max(1, (scan_dt - last_dt).days)
                days_rest = min(days_rest, 7)
            except Exception:
                days_rest = 2
        else:
            days_rest = 2

        input_row = prepare_features(player_data, is_home=is_home, days_rest=days_rest, missing_usage=missing_usage_today)

        # Fix stale USAGE_VACUUM (same logic as scan_all)
        if team_out_count > 0:
            old_uv = float(input_row['USAGE_VACUUM'].iloc[0]) if 'USAGE_VACUUM' in input_row.columns else 0.0
            input_row['USAGE_VACUUM'] = old_uv + team_out_count

        # Fresh PACE_ROLLING from this team's last 10 games
        _sc = 'SEASON_YEAR' if 'SEASON_YEAR' in df_history.columns else 'SEASON_ID'
        _df_c = df_history[df_history[_sc] == df_history[_sc].max()]
        _tgames = _df_c[_df_c['TEAM_ID'] == team_id].copy()
        if not _tgames.empty and all(c in _tgames.columns for c in ['FGA', 'FTA', 'TOV', 'MIN']):
            _mins = _tgames['MIN'].replace(0, 0.1)
            _oreb = _tgames['OREB'] if 'OREB' in _tgames.columns else 0
            _poss = _tgames['FGA'] + 0.44 * _tgames['FTA'] - _oreb + _tgames['TOV']
            _tgames['_p48'] = (_poss / _mins * 48).clip(0, 200)
            _by_game = _tgames.groupby('GAME_ID').agg({'_p48': 'mean', 'GAME_DATE': 'first'}).sort_values('GAME_DATE')
            _pace = _by_game['_p48'].iloc[-10:].mean()
            if not pd.isna(_pace) and 'PACE_ROLLING' in input_row.columns:
                input_row['PACE_ROLLING'] = float(_pace)

        # Fresh opponent stats for today's actual matchup
        _opp_team_id = todays_teams[team_id].get('opp')
        if _opp_team_id:
            _opp_games = _df_c[_df_c['TEAM_ID'] == _opp_team_id]
            _last10_gids = _opp_games['GAME_ID'].unique()[-10:]
            _opp_facing = _df_c[_df_c['GAME_ID'].isin(_last10_gids) & (_df_c['TEAM_ID'] != _opp_team_id)]
            if not _opp_facing.empty:
                for _opp_col, _raw_col in [
                    ('OPP_PTS_ALLOWED','PTS'), ('OPP_REB_ALLOWED','REB'), ('OPP_AST_ALLOWED','AST'),
                    ('OPP_FGA_ALLOWED','FGA'), ('OPP_FGM_ALLOWED','FGM'), ('OPP_FG3M_ALLOWED','FG3M'),
                    ('OPP_STL_ALLOWED','STL'), ('OPP_BLK_ALLOWED','BLK'), ('OPP_TOV_ALLOWED','TOV'),
                ]:
                    if _raw_col in _opp_facing.columns and _opp_col in input_row.columns:
                        input_row[_opp_col] = float(_opp_facing[_raw_col].mean())

        # Lookup FanDuel odds and PrizePicks lines for this player
        player_fd = fd_odds_by_player.get(normalize_name(name), {})
        player_pp = norm_lines_full.get(normalize_name(name), {})

        # --- Generate all predictions first, then post-process ---
        player_predictions = {}
        target_tiers = {}

        for target in TARGETS:
            if target in models:
                target_tiers[target] = MODEL_QUALITY.get(target, {}).get('tier', 'UNKNOWN')
                model_features = [f for f in models[target].feature_names_in_]
                valid_input   = input_row.reindex(columns=model_features, fill_value=0)

                # 1H backfill: rolling stats are NaN/0 when auto-refresh lacks 1H data
                if target in ('PTS_1H', 'PRA_1H', 'FPTS_1H'):
                    for _stat in ['PTS_1H', 'PRA_1H', 'FPTS_1H', 'MIN_1H', 'PRA', 'PTS']:
                        _season_col = f'{_stat}_Season'
                        if _season_col not in valid_input.columns:
                            continue
                        _season_val = valid_input[_season_col].iloc[0]
                        if pd.isna(_season_val) or _season_val <= 0:
                            continue
                        _season_val = float(_season_val)
                        for _suf in ['_L5', '_L10', '_L20', '_L5_Median', '_L10_Median']:
                            _col = f'{_stat}{_suf}'
                            if _col not in valid_input.columns:
                                continue
                            _cur = valid_input[_col].iloc[0]
                            if pd.isna(_cur) or float(_cur) == 0:
                                valid_input[_col] = _season_val

                raw = float(models[target].predict(valid_input)[0])

                # Apply LOG_CALIBRATION (same as scan_all)
                if target in LOG_TRANSFORM_TARGETS:
                    pred = float(np.expm1(max(raw, 0))) * LOG_CALIBRATION.get(target, 1.0)
                else:
                    pred = max(raw, 0.0)

                player_predictions[target] = pred

        # Apply injury scale_factor (same as scan_all)
        if avail['scale_factor'] < 1.0:
            for tgt in player_predictions:
                player_predictions[tgt] *= avail['scale_factor']

        # Apply correlation constraints (same as scan_all)
        pts = player_predictions.get('PTS', 0)
        reb = player_predictions.get('REB', 0)
        ast = player_predictions.get('AST', 0)
        stl = player_predictions.get('STL', 0)
        blk = player_predictions.get('BLK', 0)

        if 'PRA' in player_predictions:
            player_predictions['PRA'] = (player_predictions['PRA'] + pts + reb + ast) / 2
        if 'PR' in player_predictions:
            player_predictions['PR'] = (player_predictions['PR'] + pts + reb) / 2
        if 'PA' in player_predictions:
            player_predictions['PA'] = (player_predictions['PA'] + pts + ast) / 2
        if 'RA' in player_predictions:
            player_predictions['RA'] = (player_predictions['RA'] + reb + ast) / 2
        if 'SB' in player_predictions:
            player_predictions['SB'] = (player_predictions['SB'] + stl + blk) / 2

        # --- Display each target ---
        for target in TARGETS:
            if target in player_predictions:
                tier_text = target_tiers[target]
                pred = player_predictions[target]

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

                # Format Tier and Target natively without emojis
                tier_col = f"{tier_text:<10}"
                target_str = target.replace('_1H', ' 1H').replace('FPTS', 'FSCR')
                proj_str = f"{pred:>6.2f}       "
                print(f" {tier_col} | {target_str:<8} | {proj_str:<14} | {line_str} | {win_pct_str} | {side_str}")

        if input("\nScout another player? (y/n): ").lower() != 'y':
            scouting = False


def main():
    refresh_injuries()
    df     = load_data()
    models = load_models()
    if df is None or not models:
        print("Setup failed.")
        return

    df = auto_refresh_data(df)

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