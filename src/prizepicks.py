import requests
import pandas as pd
import json

class PrizePicksClient:
    def __init__(self):
        self.url = "https://api.prizepicks.com/projections"
        # Bypassing the 403 error
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://app.prizepicks.com/",
            "Origin": "https://app.prizepicks.com"
        }

    def fetch_board(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error connecting to PrizePicks: {e}")
            return pd.DataFrame()

        projections_list = data['data']
        included_list = data['included']
        
        # Finding the player and league name
        player_map = {}
        league_map = {}
        
        for item in included_list:
            if item['type'] == 'new_player':
                p_id = str(item['id'])
                player_name = item['attributes']['name']
                player_map[p_id] = player_name

            if item['type'] == 'league':
                l_id = str(item['id'])
                league_name = item['attributes']['name']
                league_map[l_id] = league_name


        print(f"DEBUG: I learned {len(player_map)} player names.")
        
        # Parse through projections to get back ID, Stats, and Lines.
        clean_lines = []
        
        for proj in projections_list:
            if proj['attributes'].get('is_promo') is True:
                continue
            
            if proj['attributes'].get('odds_type') != 'standard':
                continue
            
            p_id = str(proj['relationships']['new_player']['data']['id'])
            current_name = player_map.get(p_id)
            
            l_id = str(proj['relationships']['league']['data']['id'])
            current_league = league_map.get(l_id)

            p_line = proj['attributes']['line_score']
            p_stat = proj['attributes']['stat_type']

            my_map = {}
            my_map['ID'] = p_id
            my_map['Player'] = current_name
            my_map['League'] = current_league
            my_map['Stat'] = p_stat
            my_map['Line'] = p_line

            clean_lines.append(my_map)

        return pd.DataFrame(clean_lines)

# --- TEST BLOCK ---
if __name__ == "__main__":
    client = PrizePicksClient()
    df = client.fetch_board()
    
    if not df.empty:
        print(f"Success! Found {len(df)} lines.")
        print(df.head())
    else:
        print("DataFrame is empty. Did you fill in the logic?")