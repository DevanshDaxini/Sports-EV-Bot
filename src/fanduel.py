import requests
import pandas as pd
import time
from src.config import ODDS_API_KEY, MARKETS, REGIONS, ODDS_FORMAT, SPORT_MAP

class FanDuelClient:
    def __init__(self):
        self.api_key = ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4/sports"

    # Change the variable limit_games when you want to get data
    # from all available nba games
    def get_all_odds(self, limit_games=3):
        """
        1. Loops through sports in SPORT_MAP.
        2. Fetches Game IDs.
        3. Loops through the first 'limit_games' to get player props.
        """
        all_data = []

        for league_name, sport_key in SPORT_MAP.items():
            print(f"\n--- Scanning {league_name} ({sport_key}) ---")
            
            # STEP 1: Get the Game IDs (Cheap Call)
            # We use 'h2h' just to see the schedule
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
                print(f"Error fetching schedule for {league_name}: {e}")
                continue

            print(f"Found {len(games)} games active.")
            
            # STEP 2: Loop through specific games to get Props (Expensive Call)
            # We limit to the first few games to save your quota
            games_to_check = games[:limit_games]

            for game in games_to_check:
                game_id = game['id']
                home_team = game['home_team']
                away_team = game['away_team']
                print(f"  > Fetching props for: {away_team} vs {home_team}...")

                # Fetch props for this specific game
                props = self._fetch_props_for_game(sport_key, game_id)
                all_data.extend(props)
                
                # Sleep to be nice to the API
                time.sleep(0.5)

        return pd.DataFrame(all_data)

    def _fetch_props_for_game(self, sport_key, game_id):
        """
        Fetches specific player props for ONE game ID.
        """
        # CORRECT ENDPOINT: /events/{id}/odds
        url = f"{self.base_url}/{sport_key}/events/{game_id}/odds"
        
        params = {
            'apiKey': self.api_key,
            'regions': REGIONS,
            'markets': MARKETS, 
            'oddsFormat': ODDS_FORMAT,
            'bookmakers': 'fanduel'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"    Error on game {game_id}: {e}")
            return []

        clean_odds = []
        
        # --- YOUR HOMEWORK STARTS HERE ---
        # The JSON structure is:
        # data (Dict) -> bookmakers (List) -> markets (List) -> outcomes (List)
        
        # Note: 'data' is a Dictionary here, not a list of games!
        bookmakers = data.get('bookmakers', [])
        
        for book in bookmakers:
            # Loop through markets (Points, Rebounds, etc.)
            for market in book['markets']:
                stat_type = market['key']
                outcomes = market['outcomes']

                temp_players = {}

                for outcome in outcomes:
                    player_name = outcome['description']
                    side = outcome['name']
                    line = outcome['point']
                    price = outcome['price']
                    
                    if player_name not in temp_players:
                        temp_players[player_name] = {
                            'Player': player_name,
                            'Stat': stat_type,
                            'Line': line
                        }

                    if side == 'Over':
                        temp_players[player_name]['over_price'] = price
                    elif side == 'Under':
                        temp_players[player_name]['under_price'] = price

                for p_data in temp_players.values():
                    if 'over_price' in p_data and 'under_price' in p_data:
                        clean_odds.append(p_data)

        return clean_odds

# --- TEST BLOCK ---
if __name__ == "__main__":
    client = FanDuelClient()
    df = client.get_all_odds(limit_games=3)
    
    if not df.empty:
        print(f"\nSuccess! Gathered {len(df)} player props.")
        
        # SAVE TO FILE
        filename = "fanduel_test_data.csv"
        df.to_csv(filename, index=False)
        print(f"Saved all data to {filename}. Go open it!")
        
        # Still show the preview
        print(df.head())
    else:
        print("\nNo props found (or maybe no games are scheduled today).")