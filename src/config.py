# src/config.py
import os
from dotenv import load_dotenv

# Load secrets from the .env file
load_dotenv()

# 1. API Configuration
# We now get the key safely from the environment variable
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

# If the key is missing, warn the user (helpful for debugging)
if not ODDS_API_KEY:
    raise ValueError("API Key not found! Make sure you have a .env file with ODDS_API_KEY inside.")

# ... (Rest of your file stays the same: SPORT_MAP, MARKETS, etc.)

# 2. Sport Constants

SPORT_MAP = {
    'NBA': 'basketball_nba',
}

# Looking for all of these markets will make lose API keys fast
"""
SPORT_MAP = {
    'NBA': 'basketball_nba',
    'NHL': 'icehockey_nhl',
    'NFL': 'americanfootball_nfl',
}
"""

REGIONS = 'us'

# Updated and conservative market
MARKETS = (
        'player_points,player_rebounds,'
        'player_assists,player_threes,player_blocks,'
        'player_steals,player_blocks_steals,player_turnovers,'
        'player_points_rebounds_assists,player_points_rebounds,'
        'player_points_assists,player_rebounds_assists,'
        'player_field_goals,player_frees_made,player_frees_attempts,'
)

# Looking for all of these markets will make lose API keys fast
"""
MARKETS = (
    # NBA (Basketball)


    # NHL (Hockey)
    'player_power_play_points,player_blocked_shots,'
    'player_goals,player_total_saves,player_shots_on_goal,'
    
    # NFL (Football)
    'player_pass_attempts,player_pass_completions,player_pass_interceptions,'
    'player_pass_longest_completion,player_pass_rush_yds,player_pass_tds,'
    'player_pass_yds,player_receptions,player_reception_longest,'
    'player_reception_tds,player_reception_yds,player_rush_attempts,'
    'player_rush_longest,player_rush_reception_yds,player_rush_tds,'
    'player_rush_yds,player_sacks,player_solo_tackles,player_tackles_assists,'
    'player_pass_rush_reception_tds,player_pass_rush_reception_yds'
)
"""


ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# 3. PrizePicks Mathematical Hurdles (The "Breakeven" Win Rates)
# This tells the bot: "If FanDuel gives us a 56% chance, is that good enough?"
SLIP_CONFIG = {
    '2_man_power': {'hurdle': 57.74, 'min_odds': -137},
    '3_man_power': {'hurdle': 58.48, 'min_odds': -141},
    '3_man_flex':  {'hurdle': 59.80, 'min_odds': -149},
    '4_man_power': {'hurdle': 56.23, 'min_odds': -128},
    '4_man_flex':  {'hurdle': 56.90, 'min_odds': -132},
    '5_man_power': {'hurdle': 61.00, 'min_odds': -157},
    '5_man_flex':  {'hurdle': 54.25, 'min_odds': -119}, # Best Value usually
    '6_man_power': {'hurdle': 65.00, 'min_odds': -186},
    '6_man_flex':  {'hurdle': 54.21, 'min_odds': -118}, # Best Value usually
}


STAT_MAP = {
    'player_points': 'Points',
    'player_rebounds': 'Rebounds',
    'player_assists': 'Assists',
    'player_threes': '3-PT Made',
    'player_points_rebounds_assists': 'Pts+Rebs+Asts',
    'player_blocks': 'Blocks',
    'player_steals': 'Steals',
    'player_turnovers': 'Turnovers'
}