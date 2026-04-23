"""
Historical NBA Data Collection

Downloads raw game logs and player positions from NBA's official API.
Creates the foundational dataset for feature engineering and model training.

Data Sources:
    - nba_api.stats.endpoints.playergamelogs - Box scores for all players
    - nba_api.stats.endpoints.commonteamroster - Player positions
    
Output Files:
    data/nba/raw/raw_game_logs.csv      - ~200K rows (all players, all seasons)
    data/nba/processed/player_positions.csv - ~500 rows (active players + positions)
    
Configuration:
    SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
    
Usage:
    $ python3 -m src.sports.nba.builder
    
Performance:
    Takes ~3-5 minutes for 4 seasons (includes API rate limiting)
"""

import pandas as pd
import time
import os
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

# --- CONFIGURATION ---
SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']

# Resolve project root so this works no matter where it's run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_FOLDER       = os.path.join(BASE_DIR, 'data', 'nba', 'raw')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'data', 'nba', 'processed')
OUTPUT_FILE      = os.path.join(RAW_FOLDER, 'raw_game_logs.csv')
OUTPUT_1H_FILE   = os.path.join(RAW_FOLDER, 'raw_game_logs_1h.csv')
POSITION_FILE    = os.path.join(PROCESSED_FOLDER, 'player_positions.csv')


def fetch_all_game_logs():
    """
    Download box scores for all players across multiple seasons.
    
    Output:
        data/nba/raw/raw_game_logs.csv
    """
    os.makedirs(RAW_FOLDER, exist_ok=True)

    all_logs = []
    print(f"--- STARTING HISTORICAL DOWNLOAD ({len(SEASONS)} Seasons) ---")
    print("This may take a few minutes...")

    for season in SEASONS:
        max_retries = 5
        timeout_sec = 30
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"Fetching logs for season: {season}...")
                else:
                    print(f"   --> Retry {attempt}/{max_retries} for {season} (after timeout)...")
                    
                logs = playergamelogs.PlayerGameLogs(
                    season_nullable=season,
                    league_id_nullable='00',
                    timeout=timeout_sec
                )
                df = logs.get_data_frames()[0]
                df['SEASON_ID'] = season
                all_logs.append(df)
                print(f" -> Found {len(df)} game rows for {season}")
                time.sleep(1)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"❌ Error fetching {season} after {max_retries} attempts: {e}")

    if all_logs:
        master_df = pd.concat(all_logs, ignore_index=True)
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS: Saved {len(master_df)} total game rows to {OUTPUT_FILE}")
    else:
        print("FAILED: No data found.")

def fetch_1h_game_logs():
    """
    Download 1st Half box scores for all players across multiple seasons.
    
    Output:
        data/nba/raw/raw_game_logs_1h.csv
    """
    os.makedirs(RAW_FOLDER, exist_ok=True)

    all_logs = []
    print(f"--- STARTING 1H HISTORICAL DOWNLOAD ({len(SEASONS)} Seasons) ---")

    for season in SEASONS:
        max_retries = 5
        timeout_sec = 30
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"Fetching 1H logs for season: {season}...")
                else:
                    print(f"   --> Retry {attempt}/{max_retries} for 1H {season}...")
                    
                logs = playergamelogs.PlayerGameLogs(
                    season_nullable=season,
                    league_id_nullable='00',
                    game_segment_nullable='First Half',
                    timeout=timeout_sec
                )
                df = logs.get_data_frames()[0]
                df['SEASON_ID'] = season
                all_logs.append(df)
                print(f" -> Found {len(df)} 1H game rows for {season}")
                time.sleep(1)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"❌ Error fetching 1H {season} after {max_retries} attempts: {e}")

    if all_logs:
        master_df = pd.concat(all_logs, ignore_index=True)
        # Rename stats column to include _1H to prevent collision during merge
        rename_cols = {
            'MIN': 'MIN_1H', 'PTS': 'PTS_1H', 'REB': 'REB_1H', 'AST': 'AST_1H', 
            'FG3M': 'FG3M_1H', 'STL': 'STL_1H', 'BLK': 'BLK_1H', 'TOV': 'TOV_1H',
            'FGM': 'FGM_1H', 'FGA': 'FGA_1H', 'FTM': 'FTM_1H', 'FTA': 'FTA_1H',
            'NBA_FANTASY_PTS': 'NBA_FANTASY_PTS_1H', 'FG3A': 'FG3A_1H'
        }
        master_df.rename(columns=rename_cols, inplace=True)
        # Drop columns we don't need from 1H
        cols_to_keep = ['PLAYER_ID', 'GAME_ID'] + list(rename_cols.values())
        master_df = master_df[[c for c in cols_to_keep if c in master_df.columns]]
        master_df.to_csv(OUTPUT_1H_FILE, index=False)
        print(f"\nSUCCESS: Saved {len(master_df)} total 1H game rows to {OUTPUT_1H_FILE}")
    else:
        print("FAILED: No 1H data found.")


def fetch_player_positions(force_refresh=False):
    """
    Download current player positions (G, F, C) for all 30 teams.

    Args:
        force_refresh: If True, re-download even if file exists (use after trades).

    Output:
        data/nba/processed/player_positions.csv
    """
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    if os.path.exists(POSITION_FILE) and not force_refresh:
        print(f"Positions file found at {POSITION_FILE}. Skipping download.")
        return

    print("\n--- FETCHING PLAYER POSITIONS (30 Teams) ---")
    nba_teams = teams.get_teams()
    all_rosters = []

    for team in nba_teams:
        t_id = team['id']
        t_name = team['full_name']
        
        max_retries = 4
        timeout_sec = 20
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"Fetching roster for: {t_name}...", end=" ")
                
                roster = commonteamroster.CommonTeamRoster(
                    team_id=t_id, 
                    season='2025-26',
                    timeout=timeout_sec
                )
                df = roster.get_data_frames()[0]
                df = df[['PLAYER', 'PLAYER_ID', 'POSITION']]
                all_rosters.append(df)
                print("✓")
                time.sleep(0.6)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"❌ Failed after {max_retries} attempts: {e}")

    if all_rosters:
        master_roster = pd.concat(all_rosters, ignore_index=True)
        master_roster.to_csv(POSITION_FILE, index=False)
        print(f"SUCCESS: Saved {len(master_roster)} player positions to {POSITION_FILE}")
    else:
        print("FAILED: No roster data found.")


if __name__ == "__main__":
    fetch_all_game_logs()
    fetch_1h_game_logs()
    fetch_player_positions()
