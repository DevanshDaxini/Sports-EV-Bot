import requests
import pandas as pd
import time
from src.config import ODDS_API_KEY, MARKETS, REGIONS, ODDS_FORMAT, SPORT_MAP

class FanDuelClient:
    def __init__(self):
        self.api_key = ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4/sports"

    def get_all_odds(self):
        """
        Loops through every sport in SPORT_MAP, fetches odds, and combines them.
        """
        all_data = []

        # Loop through our supported sports (NBA, NHL, NFL...)
        for league_name, sport_key in SPORT_MAP.items():
            print(f"Fetching FanDuel odds for: {league_name} ({sport_key})...")
            
            # Fetch data for this specific sport
            sport_data = self._fetch_sport_odds(sport_key)
            all_data.extend(sport_data)
            
            # Sleep briefly to be nice to the API
            time.sleep(0.5)

        return pd.DataFrame(all_data)

    def _fetch_sport_odds(self, sport_key):
        """
        Private helper: Calls the API for ONE specific sport.
        """
        url = f"{self.base_url}/{sport_key}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'bookmakers': 'fanduel' # Strict filter
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching {sport_key}: {e}")
            return []

        clean_odds = []

        # --- YOUR HOMEWORK STARTS HERE ---
        # The JSON structure is nested 4 layers deep:
        # Games (List) -> Bookmakers (List) -> Markets (List) -> Outcomes (List)

        for game in data:
            game_date = game.get('commence_time')
            
            # Loop through bookmakers (should only be FanDuel because we filtered)
            for bookmaker in game['bookmakers']:
                
                # Loop through markets (Points, Rebounds, Pass TDs...)
                for market in bookmaker['markets']:
                    stat_type = market['key'] # e.g. "player_points"
                    
                    # The Outcomes are a list:
                    # [ {'name': 'Over', 'point': 20.5, 'price': -110}, 
                    #   {'name': 'Under', 'point': 20.5, 'price': -110} ]
                    outcomes = market['outcomes']
                    
                    # TODO: Group these outcomes by Player Name + Line
                    # You need to turn those 2 separate items into 1 dictionary:
                    # { 'player': 'LeBron', 'line': 20.5, 'over': -110, 'under': -110 }
                    
                    pass # <--- Delete this and write your logic

        return clean_odds

# --- TEST BLOCK ---
if __name__ == "__main__":
    client = FanDuelClient()
    df = client.get_all_odds()
    
    if not df.empty:
        print(f"\nSuccess! Found {len(df)} lines.")
        print(df.head())
        print(df['stat'].unique()) # Check what stats we found
    else:
        print("DataFrame is empty. Did you write the parsing logic?")