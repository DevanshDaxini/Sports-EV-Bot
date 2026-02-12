import requests
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from src.config import ODDS_API_KEY, REGIONS, ODDS_FORMAT, SPORT_MAP, STAT_MAP
from src.utils import SimpleCache

# --- CONFIGURATION ---
CACHE_DIR = 'fanduel_cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'fanduel_cache.json')
CACHE_DURATION_MINUTES = 60  # <--- Adjust this (e.g., 30, 60, 120 mins)

# Expanded Market List
SAFE_MARKETS = [
    'player_points',
    'player_rebounds',
    'player_assists',
    'player_threes',
    'player_points_rebounds_assists',
    'player_points_rebounds',
    'player_points_assists',
    'player_rebounds_assists',
    'player_blocks_steals',
    'player_blocks',          
    'player_steals',          
    'player_turnovers',       
    'player_field_goals',     
    'player_frees_made',      
    'player_frees_attempts'
]

# Ensure new markets map to readable names for the Analyzer
# (This extends the imported STAT_MAP locally)
LOCAL_STAT_MAP = {
    'player_points': 'Points',
    'player_rebounds': 'Rebounds',
    'player_assists': 'Assists',
    'player_threes': '3-Pt Made',
    'player_points_rebounds_assists': 'Pts+Rebs+Asts',
    'player_points_rebounds': 'Pts+Rebs',
    'player_points_assists': 'Pts+Asts',
    'player_rebounds_assists': 'Rebs+Asts',
    'player_blocks_steals': 'Blks+Stls',
    'player_blocks': 'Blocks',
    'player_steals': 'Steals',
    'player_turnovers': 'Turnovers',
    'player_field_goals': 'Field Goals Made',
    'player_frees_made': 'Free Throws Made',
    'player_frees_attempts': 'Free Throws Attempted'
}

class FanDuelClient:
    def __init__(self):
        self.api_key = ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.cache = SimpleCache(duration=300) # In-memory cache

    def get_all_odds(self, limit_games=None):
        """
        Fetches odds with DISK CACHING to save API credits.
        """
        # 1. Try Loading from Disk First (Saves $$$)
        cached_df = self._load_from_disk_cache()
        if cached_df is not None:
            return cached_df

        print(f"   💸 Cache expired or missing. Fetching fresh odds (Costs Credits)...")
        all_data = []

        # 2. Loop through Sports (NBA)
        for league_name, sport_key in SPORT_MAP.items():
            print(f"   -> Scanning {league_name}...")
            
            # Get Schedule
            games_url = f"{self.base_url}/{sport_key}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': REGIONS,
                'markets': 'h2h', 
                'oddsFormat': ODDS_FORMAT,
                'bookmakers': 'fanduel'
            }

            try:
                response = requests.get(games_url, params=params)
                response.raise_for_status()
                games = response.json()
            except Exception as e:
                print(f"      Error fetching schedule: {e}")
                continue

            print(f"      Found {len(games)} games.")
            games_to_check = games[:limit_games] if limit_games else games

            for i, game in enumerate(games_to_check):
                # Calculate Game Date (Approx EST)
                game_date_str = "Unknown"
                try:
                    commence_time = game.get('commence_time')
                    if commence_time:
                        dt_utc = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ")
                        dt_est = dt_utc - timedelta(hours=5)
                        game_date_str = dt_est.strftime('%Y-%m-%d')
                except:
                    game_date_str = datetime.now().strftime('%Y-%m-%d')

                print(f"      [{game_date_str}] Fetching props for Game {i+1}/{len(games_to_check)}...", end='\r')
                props = self._fetch_props_for_game(sport_key, game['id'], game_date_str)
                all_data.extend(props)
                time.sleep(0.5)
            print("") 

        # 3. Save to Disk & Return
        if not all_data:
            return pd.DataFrame(columns=['Player', 'Stat', 'Line', 'Odds', 'Side', 'Date'])
            
        final_df = pd.DataFrame(all_data)
        
        # Save to JSON for next time
        self._save_to_disk_cache(all_data)
        
        return final_df

    def _load_from_disk_cache(self):
        """Checks if a valid local cache file exists."""
        if not os.path.exists(CACHE_FILE):
            return None
            
        try:
            # Check file age
            file_mod_time = os.path.getmtime(CACHE_FILE)
            file_age_minutes = (time.time() - file_mod_time) / 60
            
            if file_age_minutes < CACHE_DURATION_MINUTES:
                print(f"   ♻️  Using Saved FanDuel Data from {int(file_age_minutes)} mins ago.")
                print(f"       (0 API Credits Used) - Expires in {int(CACHE_DURATION_MINUTES - file_age_minutes)} mins")
                
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            else:
                print(f"   ⚠️  Saved data is too old ({int(file_age_minutes)} mins). Need refresh.")
                return None
        except Exception as e:
            print(f"   Warning: Could not load cache: {e}")
            return None

    def _save_to_disk_cache(self, data_list):
        """Saves the fresh data to a JSON file inside the cache folder."""
        try:
            # Create the directory if it doesn't exist
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
                print(f"   📂 Created directory: {CACHE_DIR}")

            with open(CACHE_FILE, 'w') as f:
                json.dump(data_list, f)
            print(f"   💾 Saved fresh odds to '{CACHE_FILE}' for future use.")
        except Exception as e:
            print(f"   Warning: Could not save cache: {e}")


    def _fetch_props_for_game(self, sport_key, game_id, game_date):
        markets_string = ",".join(SAFE_MARKETS)
        
        url = f"{self.base_url}/{sport_key}/events/{game_id}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': REGIONS,
            'markets': markets_string, 
            'oddsFormat': ODDS_FORMAT,
            'bookmakers': 'fanduel'
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code != 200: return []
            data = response.json()
        except: return []

        clean_odds = []
        bookmakers = data.get('bookmakers', [])
        if not bookmakers: return []
        book = bookmakers[0] 
        
        for market in book['markets']:
            raw_stat = market['key']
            
            # Use Local Map first, fallback to Config Map, then Raw
            stat_name = LOCAL_STAT_MAP.get(raw_stat, STAT_MAP.get(raw_stat, raw_stat))
            
            for outcome in market['outcomes']:
                clean_odds.append({
                    'Player': outcome['description'],
                    'Stat': stat_name,
                    'Line': outcome.get('point', 0),
                    'Odds': outcome.get('price', 0),
                    'Side': outcome['name'], 
                    'Date': game_date
                })
        return clean_odds

if __name__ == "__main__":
    client = FanDuelClient()
    df = client.get_all_odds(limit_games=1)
    if not df.empty:
        print(df.head())
    else:
        print("No props found.")