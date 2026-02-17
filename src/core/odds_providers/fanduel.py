"""
FanDuel Odds Client (via The Odds API)

Fetches player prop odds from FanDuel sportsbook using The Odds API.
Implements disk caching to minimize API credit usage.

API Credits:
    Each game's props costs 1 credit
    500 requests/month on free tier
    
Caching Strategy:
    - Saves odds to fanduel_cache/fanduel_cache.json
    - Cache valid for 30 minutes
    - Reuses cache if <30 mins old (saves credits)
    
Markets Fetched:
    player_points, player_rebounds, player_assists, player_threes,
    player_points_rebounds_assists, player_blocks, player_steals,
    player_turnovers, player_field_goals, player_frees_made, etc.
    
Usage:
    from src.core.odds_providers.fanduel import FanDuelClient
    client = FanDuelClient()
    odds_df = client.get_all_odds(limit_games=5)
    
Output Format:
    DataFrame with columns: Player, Stat, Line, Odds, Side, Date
    Example: LeBron James, Points, 25.5, -120, Over, 2026-02-12
"""

import requests
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from src.core.utils import SimpleCache

# --- CONFIGURATION ---
CACHE_DIR = 'fanduel_cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'fanduel_cache.json')
CACHE_DURATION_MINUTES = 30

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
    def __init__(self, api_key, sport_map, regions, odds_format, stat_map):
        """
        Args:
            api_key (str):      From config (ODDS_API_KEY)
            sport_map (dict):   From sport config e.g. {'NBA': 'basketball_nba'}
            regions (str):      From config e.g. 'us'
            odds_format (str):  From config e.g. 'american'
            stat_map (dict):    From sport config (fallback name mapping)
        """
        self.api_key = api_key
        self.sport_map = sport_map
        self.regions = regions
        self.odds_format = odds_format
        self.stat_map = stat_map
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.cache = SimpleCache(duration=300)

    def get_all_odds(self, limit_games=None):
        """
        Fetch player prop odds with intelligent caching.
        
        Args:
            limit_games (int or None): Max games to fetch (None = all games)
                                       Use for testing: limit_games=1
                                       
        Returns:
            pandas.DataFrame: Player props with Over/Under odds
        """
        cached_df = self._load_from_disk_cache()
        if cached_df is not None:
            return cached_df

        print(f"   💸 Cache expired or missing. Fetching fresh odds (Costs Credits)...")
        all_data = []

        for league_name, sport_key in self.sport_map.items():
            print(f"   -> Scanning {league_name}...")

            games_url = f"{self.base_url}/{sport_key}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': self.regions,
                'markets': 'h2h',
                'oddsFormat': self.odds_format,
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

        if not all_data:
            return pd.DataFrame(columns=['Player', 'Stat', 'Line', 'Odds', 'Side', 'Date'])

        final_df = pd.DataFrame(all_data)
        self._save_to_disk_cache(all_data)
        return final_df

    def _load_from_disk_cache(self):
        if not os.path.exists(CACHE_FILE):
            return None
        try:
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
        try:
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
            'regions': self.regions,
            'markets': markets_string,
            'oddsFormat': self.odds_format,
            'bookmakers': 'fanduel'
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                return []
            data = response.json()
        except:
            return []

        clean_odds = []
        bookmakers = data.get('bookmakers', [])
        if not bookmakers:
            return []
        book = bookmakers[0]

        for market in book['markets']:
            raw_stat = market['key']
            stat_name = LOCAL_STAT_MAP.get(raw_stat, self.stat_map.get(raw_stat, raw_stat))
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
    # Run from project root: python -m src.core.odds_providers.fanduel
    from src.sports.nba.config import ODDS_API_KEY, SPORT_MAP, REGIONS, ODDS_FORMAT, STAT_MAP
    client = FanDuelClient(ODDS_API_KEY, SPORT_MAP, REGIONS, ODDS_FORMAT, STAT_MAP)
    df = client.get_all_odds(limit_games=1)
    if not df.empty:
        print(df.head())
    else:
        print("No props found.")
