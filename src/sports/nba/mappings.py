"""
NBA-Specific Mappings

Contains all NBA name normalization, stat mappings, and volatility weights
that were previously embedded in main.py / nba_cli.py.

Keeping these here means:
  - nba_cli.py stays clean (just workflow logic)
  - Adding CBB just means creating src/sports/cbb/mappings.py

Usage:
    from src.sports.nba.mappings import PP_NORMALIZATION_MAP, STAT_MAPPING, VOLATILITY_MAP
"""

# PrizePicks display name -> Standard/FanDuel name
# Used to normalize PP stat names before merging with FanDuel data
PP_NORMALIZATION_MAP = {
    'Blocked Shots': 'Blocks',
    '3-PT Made': '3-Pt Made',
    'Three Point Field Goals': '3-Pt Made',
    'Free Throws Made': 'Free Throws Made',
    'Turnovers': 'Turnovers',
    'Steals': 'Steals',
    'Fantasy Score': 'Fantasy',
    # Combo / efficiency markets
    'FG Made': 'Field Goals Made',
    'FG Attempted': 'Field Goals Attempted',
    'Free Throws Attempted': 'Free Throws Attempted',
    'Blks+Stls': 'Blks+Stls',
    'Pts+Rebs+Asts': 'Pts+Rebs+Asts',
    'Pts+Rebs': 'Pts+Rebs',
    'Pts+Asts': 'Pts+Asts',
    'Rebs+Asts': 'Rebs+Asts'
}

# Standard display name -> AI model target code
# Used to map math-scan stat names to the codes the ML models use
STAT_MAPPING = {
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Assists': 'AST',
    'Pts+Rebs+Asts': 'PRA',
    'Pts+Rebs': 'PR',
    'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA',
    'Blks+Stls': 'SB',
    '3-Pt Made': 'FG3M',
    'Blocks': 'BLK',
    'Steals': 'STL',
    'Turnovers': 'TOV',
    'Free Throws Made': 'FTM',
    'Field Goals Made': 'FGM',
    'Free Throws Attempted': 'FTA',
    'Field Goals Attempted': 'FGA'
}

# Stat reliability weight for combined scoring.
# Higher = more predictable stat = score weighted upward.
# Penalties are intentionally mild — enough to rank high-confidence stats
# above volatile ones when edge is equal, but not so severe that BLK/STL/SB
# can never appear. At 0.90 a BLK play only needs ~54% win rate to match a
# PTS play at 53.5%, which is achievable.
VOLATILITY_MAP = {
    'PTS': 1.00,
    'REB': 1.15,
    'AST': 1.10,
    'PRA': 1.05,
    'PR':  1.05,
    'PA':  1.05,
    'RA':  1.10,
    'FG3M': 0.95,
    'BLK':  0.90,  # was 0.75 — softened so BLK can compete
    'STL':  0.90,  # was 0.75 — softened so STL can compete
    'SB':   0.95,  # was 0.80 — softened so Blks+Stls can compete
    'TOV':  0.92,  # was 0.90
    'FGM':  1.00,
    'FGA':  1.00,
    'FTM':  0.92,  # was 0.90
    'FTA':  0.92,  # was 0.90
}