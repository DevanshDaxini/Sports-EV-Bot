"""
CBB (College Basketball) Configuration

Future sport configuration. Mirror the structure of src/sports/nba/config.py
and fill in CBB-specific values when you're ready to add this sport.

Steps to activate CBB:
    1. Fill in SPORT_MAP with the correct Odds API key for CBB
    2. Define CBB-specific STAT_MAP (college stats differ from NBA)
    3. Define MODEL_TIERS based on CBB model training results
    4. Create src/sports/cbb/mappings.py (PP normalization for CBB)
    5. Create src/sports/cbb/builder.py (uses NCAA data source, not nba_api)
    6. Create src/sports/cbb/features.py (CBB-specific feature engineering)
    7. Create src/sports/cbb/train.py (same XGBoost approach)
    8. Add 'CBB' option to src/cli/nba_cli.py menu
"""

import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv('ODDS_API_KEY')

# TODO: Replace with correct CBB Odds API sport key
SPORT_MAP = {
    'CBB': 'basketball_ncaab',  # verify this key at https://api.the-odds-api.com/v4/sports
}

REGIONS    = 'us'
ODDS_FORMAT = 'american'

# TODO: Map CBB PrizePicks stat names to internal codes
STAT_MAP = {
    'Points':   'PTS',
    'Rebounds': 'REB',
    'Assists':  'AST',
    # Add CBB-specific stats here
}

# TODO: Populate after training CBB models
MODEL_TIERS   = {}
MODEL_QUALITY = {}
ACTIVE_TARGETS = []
