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
        print(f"Fetching logs for season: {season}...")
        try:
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                league_id_nullable='00'
            )
            df = logs.get_data_frames()[0]
            df['SEASON_ID'] = season
            all_logs.append(df)
            print(f" -> Found {len(df)} game rows for {season}")
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching {season}: {e}")

    if all_logs:
        master_df = pd.concat(all_logs, ignore_index=True)
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS: Saved {len(master_df)} total game rows to {OUTPUT_FILE}")
    else:
        print("FAILED: No data found.")


def fetch_player_positions():
    """
    Download current player positions (G, F, C) for all 30 teams.
    
    Output:
        data/nba/processed/player_positions.csv
    """
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    if os.path.exists(POSITION_FILE):
        print(f"Positions file found at {POSITION_FILE}. Skipping download.")
        return

    print("\n--- FETCHING PLAYER POSITIONS (30 Teams) ---")
    nba_teams = teams.get_teams()
    all_rosters = []

    for team in nba_teams:
        t_id = team['id']
        t_name = team['full_name']
        print(f"Fetching roster for: {t_name}...")
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=t_id, season='2025-26')
            df = roster.get_data_frames()[0]
            df = df[['PLAYER', 'PLAYER_ID', 'POSITION']]
            all_rosters.append(df)
            time.sleep(0.6)
        except Exception as e:
            print(f"Error fetching {t_name}: {e}")

    if all_rosters:
        master_roster = pd.concat(all_rosters, ignore_index=True)
        master_roster.to_csv(POSITION_FILE, index=False)
        print(f"SUCCESS: Saved {len(master_roster)} player positions to {POSITION_FILE}")
    else:
        print("FAILED: No roster data found.")


if __name__ == "__main__":
    fetch_all_game_logs()
    fetch_player_positions()
