"""
PrizePicks API Client

Fetches current prop lines from PrizePicks' public API.
Converts stat names to abbreviations using STAT_MAP for consistency.

API Endpoint:
    https://api.prizepicks.com/projections
    
Returns:
    Player props with lines (e.g., LeBron James, Points, 25.5)
    
Usage:
    from src.core.odds_providers.prizepicks import PrizePicksClient
    client = PrizePicksClient(stat_map=STAT_MAP)
    df = client.fetch_board(league_filter='NBA')
    
    # Or as a dict:
    lines = client.fetch_lines_dict(league_filter='NBA')
    # {'LeBron James': {'PTS': 25.5, 'REB': 7.5}}
    
Note:
    - Filters to 'standard' odds_type only (excludes promos)
    - Pass league_filter='NBA' for NBA, 'CBB' for college basketball, etc.
    - Uses STAT_MAP to convert 'Points' → 'PTS'
"""

import requests
import pandas as pd


class PrizePicksClient:
    def __init__(self, stat_map):
        """
        Args:
            stat_map (dict): Stat name mapping from sport config
                             e.g. {'Points': 'PTS', 'Rebounds': 'REB'}
        """
        self.url = "https://api.prizepicks.com/projections"
        self.stat_map = stat_map
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://app.prizepicks.com/",
            "Origin": "https://app.prizepicks.com"
        }

    def fetch_board(self, league_filter='NBA'):
        """
        Fetch current PrizePicks prop offerings.
        
        Args:
            league_filter (str): League to filter by. e.g. 'NBA', 'CBB'.
                                 Pass None to return all leagues.
        
        Returns:
            pandas.DataFrame: Columns: Player, League, Stat, Line
            
        Filtering:
            - is_promo = False (no promotional lines)
            - odds_type = 'standard' (no demon/goblin lines)
            - League = league_filter value
        """
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error connecting to PrizePicks: {e}")
            return pd.DataFrame()

        projections_list = data.get('data', [])
        included_list = data.get('included', [])

        player_map = {}
        league_map = {}

        for item in included_list:
            if item['type'] == 'new_player':
                player_map[str(item['id'])] = item['attributes']['name']
            if item['type'] == 'league':
                league_map[str(item['id'])] = item['attributes']['name']

        clean_lines = []

        for proj in projections_list:
            attrs = proj['attributes']
            if attrs.get('is_promo') is True:
                continue
            if attrs.get('odds_type') != 'standard':
                continue
            if 'new_player' not in proj['relationships'] or 'league' not in proj['relationships']:
                continue

            p_id = str(proj['relationships']['new_player']['data']['id'])
            l_id = str(proj['relationships']['league']['data']['id'])

            current_league = league_map.get(l_id)

            # Sport-agnostic filter: only skip if a filter is set and doesn't match
            if league_filter and current_league != league_filter:
                continue

            clean_lines.append({
                'Player': player_map.get(p_id),
                'League': current_league,
                'Stat': attrs['stat_type'],
                'Line': attrs['line_score']
            })

        return pd.DataFrame(clean_lines)

    def fetch_lines_dict(self, league_filter='NBA'):
        """
        Fetch PrizePicks lines as nested dictionary.
        
        Args:
            league_filter (str): League to filter. e.g. 'NBA', 'CBB'.
        
        Returns:
            dict: {player_name: {stat_abbr: line_value}}
                  Example: {'LeBron James': {'PTS': 25.5, 'REB': 7.5}}
        """
        df = self.fetch_board(league_filter=league_filter)
        if df.empty:
            return {}

        lines_dict = {}
        for _, row in df.iterrows():
            player = row['Player']
            stat = row['Stat']
            line = row['Line']

            if player not in lines_dict:
                lines_dict[player] = {}

            clean_stat = self.stat_map.get(stat, stat)
            lines_dict[player][clean_stat] = float(line)

        return lines_dict


if __name__ == "__main__":
    from core.config import STAT_MAP
    client = PrizePicksClient(stat_map=STAT_MAP)
    lines = client.fetch_lines_dict(league_filter='NBA')
    print(f"Fetched {len(lines)} players.")
    for p, stats in list(lines.items())[:2]:
        print(f"{p}: {stats}")
