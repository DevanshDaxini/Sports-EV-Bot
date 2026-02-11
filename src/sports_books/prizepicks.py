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

        projections_list = data.get('data', [])
        included_list = data.get('included', [])
        
        # 1. Build Lookup Maps
        player_map = {}
        league_map = {}
        
        for item in included_list:
            if item['type'] == 'new_player':
                p_id = str(item['id'])
                player_map[p_id] = item['attributes']['name']
            if item['type'] == 'league':
                l_id = str(item['id'])
                league_map[l_id] = item['attributes']['name']

        print(f"DEBUG: Learned {len(player_map)} players.")
        
        # 2. Parse Projections
        clean_lines = []
        
        for proj in projections_list:
            attrs = proj['attributes']
            
            # Filter bad lines
            if attrs.get('is_promo') is True: continue
            if attrs.get('odds_type') != 'standard': continue

            # IDs
            p_id = str(proj['relationships']['new_player']['data']['id'])
            l_id = str(proj['relationships']['league']['data']['id'])
            
            current_league = league_map.get(l_id)
            if current_league != 'NBA': continue # Strictly NBA

            # --- NEW: Get Game Date ---
            # Format is usually: "2026-02-07T19:00:00-05:00"
            raw_start = attrs.get('start_time')
            game_date = "Unknown"
            if raw_start:
                # Take just the first 10 chars (YYYY-MM-DD)
                game_date = raw_start[:10]

            clean_lines.append({
                'ID': p_id,
                'Player': player_map.get(p_id),
                'League': current_league,
                'Stat': attrs['stat_type'],
                'Line': attrs['line_score'],
                'Date': game_date
            })

        return pd.DataFrame(clean_lines)

# --- TEST BLOCK ---
if __name__ == "__main__":
    client = PrizePicksClient()
    df = client.fetch_board()
    
    if not df.empty:
        print(f"Success! Found {len(df)} lines.")
        
        # SAVE TO FILE
        filename = "prizepicks_test_data.csv"
        df.to_csv(f'csvFiles/{filename}', index=False)
        print(f"Saved all data to {filename}. Go open it!")

        print(df.head())
    else:
        print("DataFrame is empty. Did you fill in the logic?")