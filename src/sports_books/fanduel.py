import requests
import pandas as pd
import time
from datetime import datetime
from src.config import ODDS_API_KEY, REGIONS, ODDS_FORMAT, SPORT_MAP, STAT_MAP
from src.utils import SimpleCache

# We override the config MARKETS to prevent 422 Errors (Too many requests)
SAFE_MARKETS = [
    'player_points',
    'player_rebounds',
    'player_assists',
    'player_threes',
    'player_points_rebounds_assists',
    'player_points_rebounds',
    'player_points_assists',
    'player_rebounds_assists',
    'player_blocks_steals'
]

class FanDuelClient:
    def __init__(self):
        self.api_key = ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.cache = SimpleCache(duration=300)

    def get_all_odds(self, limit_games=None):
        """
        Fetches odds for all active NBA games.
        """
        # 1. Check Cache
        cache_key = f"fd_odds_{limit_games}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            print(f"   [Cache] Using saved FanDuel data ({len(cached_data)} rows).")
            return cached_data

        all_data = []

        # 2. Loop through Sports (NBA)
        for league_name, sport_key in SPORT_MAP.items():
            print(f"   -> Scanning {league_name}...")
            
            # Get Game IDs
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
            
            # Loop through specific games
            games_to_check = games[:limit_games] if limit_games else games

            for i, game in enumerate(games_to_check):
                print(f"      Fetching props for Game {i+1}/{len(games_to_check)}...", end='\r')
                props = self._fetch_props_for_game(sport_key, game['id'])
                all_data.extend(props)
                time.sleep(0.5) # Respect API limits
            print("") # New line after loop

        # 3. Save & Return
        final_df = pd.DataFrame(all_data)

        if not final_df.empty:
            self.cache.set(cache_key, final_df)
        
        return final_df

    def _fetch_props_for_game(self, sport_key, game_id):
        """
        Fetches specific player props for ONE game ID.
        """
        # We join our SAFE list into a comma-separated string
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
            if response.status_code != 200:
                return [] # Skip if error (usually means no props yet)
            data = response.json()
        except:
            return []

        clean_odds = []
        bookmakers = data.get('bookmakers', [])
        
        if not bookmakers:
            return []
            
        # We assume FanDuel is the only bookmaker requested
        book = bookmakers[0] 
        
        for market in book['markets']:
            raw_stat = market['key']
            # Map "player_points" -> "PTS"
            stat_name = STAT_MAP.get(raw_stat, raw_stat)
            
            for outcome in market['outcomes']:
                # Flatten the data: One row per Outcome
                clean_odds.append({
                    'Player': outcome['description'],
                    'Stat': stat_name,
                    'Line': outcome.get('point', 0),
                    'Odds': outcome.get('price', 0),
                    'Side': outcome['name'], # 'Over' or 'Under'
                    'Date': datetime.now().strftime('%Y-%m-%d')
                })

        return clean_odds

if __name__ == "__main__":
    client = FanDuelClient()
    df = client.get_all_odds(limit_games=1)
    if not df.empty:
        print(df.head())
    else:
        print("No props found.")