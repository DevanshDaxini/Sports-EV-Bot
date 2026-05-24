"""
NBA Configuration Constants

All NBA-specific settings: API keys, sport mappings, stat names,
model quality tiers, and scanning modes.

For shared cross-sport settings (SLIP_CONFIG) see:
    src/core/config.py

Environment Variables:
    .env file must contain: ODDS_API_KEY=your_key_here

Usage:
    from src.sports.nba.config import STAT_MAP, MODEL_QUALITY, ACTIVE_TARGETS
"""

import os
from dotenv import load_dotenv

load_dotenv()

# 1. API Configuration
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
if not ODDS_API_KEY:
    raise ValueError("API Key not found! Make sure you have a .env file with ODDS_API_KEY inside.")

# 2. Sport Constants
SPORT_MAP = {
    'NBA': 'basketball_nba',
}

REGIONS = 'us'

MARKETS = (
    'player_points,player_rebounds,'
    'player_assists,player_threes,player_blocks,'
    'player_steals,player_blocks_steals,player_turnovers,'
    'player_points_rebounds_assists,player_points_rebounds,'
    'player_points_assists,player_rebounds_assists,'
    'player_field_goals,player_frees_made,player_frees_attempts,'
)

ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# 3. Stat Name Map  (PrizePicks display name -> internal abbreviation)
STAT_MAP = {
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Assists': 'AST',
    '3-PT Made': 'FG3M',
    '3-PT Attempted': 'FG3A',
    'Blocked Shots': 'BLK',
    'Steals': 'STL',
    'Turnovers': 'TOV',
    'FG Made': 'FGM',
    'FG Attempted': 'FGA',
    'Free Throws Made': 'FTM',
    'Free Throws Attempted': 'FTA',
    'Pts+Rebs+Asts': 'PRA',
    'Pts+Rebs': 'PR',
    'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA',
    'Blks+Stls': 'SB',
    'Fantasy Score': 'FPTS',
    # 1st Half (1H) Markets
    '1H Pts+Rebs+Asts': 'PRA_1H',
    '1H Points': 'PTS_1H',
    '1H Fantasy Score': 'FPTS_1H'
}

# 4. Model Quality Tiers
# Updated: 2026-04-21 — per-stat edge thresholds derived from walk-forward
# backtest on 28k held-out rows, using player L10-median as the proxy line.
#
# Per-stat minimum edge that produces >53.5% win rate (break-even at -115):
#   FGA    >=10% → 55.6%+   FG3A  >=10% → 53.5%+   AST   >=10% → 54.3%+
#   FGM    >=10% → 54.4%+   PRA   >=10% → 53.8%+   PTS   >=10% → 53.5%+
#   RA     >=15% → 56.1%+   REB   >=15% → 54.5%+   PA    >=15% → 54.0%+
#   FPTS   >=15% → 55.5%+   PR    >=20% → 54.4%+   FTM   >=20% → 54.3%+
#   FTA    >=20% → 54.0%+   PRA_1H>=20% → 54.1%+
#   PTS_1H >=25% → 53.9%+   FPTS_1H>=30%→ 55.0%+
#   BLK    >=5%  → 56.6% (5-10% bucket ONLY — inconsistent, use caution)
#   FG3M / SB / STL / TOV — NEVER profitable at any tested edge level.
#     These are included at 40% threshold so rare extreme cases surface, but
#     they are labeled SPECULATIVE and should not be core of any strategy.
MODEL_TIERS = {
    'ELITE': {
        'models': ['FGA', 'FG3A', 'AST', 'FGM', 'PRA', 'PTS'],
        'accuracy_range': '53.5-55.6%',
        'edge_threshold': 10.0,
        'description': '⭐ Core plays — profitable at >=10% edge',
        'emoji': '⭐'
    },
    'STRONG': {
        'models': ['RA', 'REB', 'PA', 'FPTS'],
        'accuracy_range': '54.0-56.1%',
        'edge_threshold': 15.0,
        'description': '✔ Strong plays — profitable at >=15% edge',
        'emoji': '✔'
    },
    'DECENT': {
        'models': ['PR', 'FTM', 'FTA', 'PRA_1H'],
        'accuracy_range': '54.0-54.4%',
        'edge_threshold': 20.0,
        'description': '~ Good plays — profitable at >=20% edge',
        'emoji': '~'
    },
    'RISKY': {
        'models': ['PTS_1H', 'FPTS_1H', 'BLK'],
        'accuracy_range': '53.9-56.6%',
        'edge_threshold': 25.0,
        'description': '⚠️ High-edge only — 25%+ needed for +EV',
        'emoji': '⚠️'
    },
    'SPECULATIVE': {
        'models': ['FG3M', 'SB', 'STL', 'TOV'],
        'accuracy_range': '~47-50%',
        'edge_threshold': 40.0,
        'description': '🎲 Lottery plays — model has no reliable edge',
        'emoji': '🎲'
    }
}

# Quick lookup dict: model name -> tier info
MODEL_QUALITY = {}
for tier, data in MODEL_TIERS.items():
    for model in data['models']:
        MODEL_QUALITY[model] = {
            'tier': tier,
            'threshold': data['edge_threshold'],
            'emoji': data['emoji']
        }

# 5. Scanning Mode — ALL stats always active; thresholds above do the filtering.
# Each stat's edge_threshold is the minimum % gap (AI proj vs line) needed to show
# a bet. This was derived from per-stat backtest calibration, so changing the
# mode doesn't help — the threshold per stat is what matters.
SCANNING_MODE = 'ALL'

SCANNING_MODES = {
    'ELITE_ONLY': MODEL_TIERS['ELITE']['models'],
    'SAFE':       MODEL_TIERS['ELITE']['models'] + MODEL_TIERS['STRONG']['models'],
    'BALANCED':   MODEL_TIERS['ELITE']['models'] + MODEL_TIERS['STRONG']['models'] + MODEL_TIERS['DECENT']['models'],
    'AGGRESSIVE': MODEL_TIERS['ELITE']['models'] + MODEL_TIERS['STRONG']['models'] + MODEL_TIERS['DECENT']['models'] + MODEL_TIERS['RISKY']['models'],
    'ALL':        [m for tier in MODEL_TIERS.values() for m in tier['models']],
}

ACTIVE_TARGETS = SCANNING_MODES.get(SCANNING_MODE, SCANNING_MODES['ALL'])

mode_descriptions = {
    'ELITE_ONLY': "⭐ ELITE — 6 core stats, >=10% edge",
    'SAFE':       "✔ SAFE — 10 profitable stats, >=10-15% edge",
    'BALANCED':   "📊 BALANCED — 14 stats, >=10-20% edge",
    'AGGRESSIVE': "⚡ AGGRESSIVE — 17 stats, >=10-25% edge",
    'ALL':        "🎲 ALL — every stat incl. speculative (calibrated per-stat threshold)",
}

def print_mode_info():
    print(f"⚙️  {mode_descriptions.get(SCANNING_MODE, 'UNKNOWN MODE')}")
    print(f"   Scanning: {', '.join(ACTIVE_TARGETS)}")

# 6. Injury Adjustment — Absorption Rates
# What fraction of a missing player's production redistributes to active teammates.
# Lower = more player-specific skill (hard to replace),  Higher = more opportunity-based.
ABSORPTION_RATES = {
    'PTS':  0.50,   # Scoring redistributes moderately
    'FGM':  0.45,   # Shot-making is partially skill-dependent
    'FGA':  0.55,   # Shot attempts redistribute well
    'FG3M': 0.30,   # 3PT shooting is very skill-dependent
    'FG3A': 0.45,   # 3PT attempts redistribute somewhat
    'REB':  0.65,   # Rebounds strongly redistribute (someone grabs them)
    'AST':  0.35,   # Assists are playmaker-specific
    'STL':  0.20,   # Steals are position/skill-dependent
    'BLK':  0.20,   # Blocks are heavily position-dependent
    'TOV':  0.25,   # Turnovers don't "redistribute" meaningfully
    'FTM':  0.35,   # Free throws depend on who drives to rim
    'FTA':  0.40,   # FT attempts redistribute a bit
    'PRA':  0.50,   # Combo stat — average of components
    'PR':   0.55,   # PTS + REB — rebounds help
    'PA':   0.42,   # PTS + AST — assists drag it down
    'RA':   0.48,   # REB + AST — mixed
    'SB':   0.20,   # Steals + Blocks — very position-dependent
}
