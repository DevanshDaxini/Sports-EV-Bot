"""
Historical NBA Data Collection

Downloads raw game logs and player positions from NBA's official API.
Creates the foundational dataset for feature engineering and model training.

Data Sources:
    - nba_api.stats.endpoints.playergamelogs - Box scores for all players
    - nba_api.stats.endpoints.commonteamroster - Player positions
    
Output Files:
    data/raw_game_logs.csv - ~200K rows (all players, all seasons)
    data/player_positions.csv - ~500 rows (active players + positions)
    
Configuration:
    SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26']
    
Usage:
    $ python3 -m src.builder
    
Performance:
    Takes ~3-5 minutes for 4 seasons (includes API rate limiting)
    
Note:
    Respects NBA API rate limits with time.sleep() calls
    Safe to run multiple times (positions file skips if exists)
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
DATA_FOLDER = 'data'
OUTPUT_FILE = f'{DATA_FOLDER}/raw_game_logs.csv'

def fetch_all_game_logs():
    """
    Download box scores for all players across multiple seasons.
    
    Steps:
        1. Create data/ folder if doesn't exist
        2. For each season in SEASONS:
            a. Call playergamelogs API (bulk endpoint)
            b. Tag rows with SEASON_ID
            c. Append to list
            d. Sleep 1 second (rate limiting)
        3. Concatenate all seasons
        4. Save to data/raw_game_logs.csv
        
    Output:
        CSV with columns: PLAYER_ID, PLAYER_NAME, TEAM_ID, GAME_ID, 
                         GAME_DATE, PTS, REB, AST, FGM, FGA, etc.
                         
    Note:
        Uses playergamelogs.PlayerGameLogs() instead of looping through
        individual players (100x faster)
        
    Example Output:
        --- STARTING HISTORICAL DOWNLOAD (4 Seasons) ---
        Fetching logs for season: 2022-23...
         -> Found 48532 game rows for 2022-23
        ...
        SUCCESS: Saved 194128 total game rows to data/raw_game_logs.csv
    """

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Created folder: {DATA_FOLDER}")

    all_logs = []
    
    print(f"--- STARTING HISTORICAL DOWNLOAD ({len(SEASONS)} Seasons) ---")
    print("This may take a few minutes...")

    for season in SEASONS:
        print(f"Fetching logs for season: {season}...")
        try:
            # We use the bulk endpoint to get everyone at once (Much faster)
            logs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                league_id_nullable='00' # NBA
            )
            df = logs.get_data_frames()[0]
            df['SEASON_ID'] = season # Tag the season
            all_logs.append(df)
            print(f" -> Found {len(df)} game rows for {season}")
            
            # Sleep to be nice to the API
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error fetching {season}: {e}")

    # Combine all seasons
    if all_logs:
        master_df = pd.concat(all_logs, ignore_index=True)
        
        # Save to CSV
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS: Saved {len(master_df)} total game rows to {OUTPUT_FILE}")
        print("You can now proceed to Feature Engineering.")
    else:
        print("FAILED: No data found.")

def fetch_player_positions():
    """
    Download current player positions (G, F, C) for all 30 teams.
    
    Steps:
        1. Check if data/player_positions.csv exists (skip if yes)
        2. Get all 30 NBA team IDs from nba_api.stats.static.teams
        3. For each team:
            a. Call commonteamroster API
            b. Extract PLAYER, PLAYER_ID, POSITION columns
            c. Sleep 0.6 seconds (rate limiting)
        4. Concatenate all rosters
        5. Save to data/player_positions.csv
        
    Output:
        CSV with columns: PLAYER, PLAYER_ID, POSITION
        Example: LeBron James, 2544, F
        
    Note:
        Skips download if file exists (avoids redundant API calls)
        Position data is used for defensive matchup features
        
    Positions:
        G (Guard), F (Forward), C (Center), G-F, F-C, etc.
    """

    POSITION_FILE = f'{DATA_FOLDER}/player_positions.csv'
    
    if os.path.exists(POSITION_FILE):
        print(f"Positions file found at {POSITION_FILE}. Skipping download.")
        return

    print("\n--- FETCHING PLAYER POSITIONS (30 Teams) ---")
    
    # 1. Get all team IDs (using static.teams)
    nba_teams = teams.get_teams()
    all_rosters = []

    # 2. Loop through every team
    for team in nba_teams:
        t_id = team['id']
        t_name = team['full_name']
        print(f"Fetching roster for: {t_name}...")
        
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=t_id, 
                                                       season='2025-26')

            df = roster.get_data_frames()[0]
            
            df = df[['PLAYER', 'PLAYER_ID', 'POSITION']]
            
            all_rosters.append(df)
            
            time.sleep(0.6)
            
        except Exception as e:
            print(f"Error fetching {t_name}: {e}")

    # 3. Save to CSV
    if all_rosters:
        master_roster = pd.concat(all_rosters, ignore_index=True)
        master_roster.to_csv(POSITION_FILE, index=False)
        print(f"SUCCESS: Saved {len(master_roster)} player positions to {POSITION_FILE}")
    else:
        print("FAILED: No roster data found.")

if __name__ == "__main__":
    fetch_all_game_logs()
    fetch_player_positions()