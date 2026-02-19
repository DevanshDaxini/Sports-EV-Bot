"""
Tennis Name & Stat Mappings

Handles name normalization between PrizePicks display names and
Sackmann dataset names. Tennis names frequently differ in:
    - Accented characters (Djokovic vs Đoković)
    - Abbreviated first names (C. Alcaraz vs Carlos Alcaraz)
    - Hyphenated surnames (Ruud vs Ruud)

Usage:
    from src.sports.tennis.mappings import normalize_name, PP_NAME_MAP
"""

import unicodedata
import re

# ---------------------------------------------------------------------------
# MANUAL OVERRIDES
# Known mismatches between PrizePicks display names and Sackmann names.
# Format: 'prizepicks_name_lowercased' -> 'sackmann_name_lowercased'
# Add to this as you discover mismatches during scanning.
# ---------------------------------------------------------------------------
PP_NAME_MAP = {
    # ATP examples
    'novak djokovic':       'novak djokovic',
    'carlos alcaraz':       'carlos alcaraz',
    'jannik sinner':        'jannik sinner',
    'daniil medvedev':      'daniil medvedev',
    'alexander zverev':     'alexander zverev',
    'andrey rublev':        'andrey rublev',
    'casper ruud':          'casper ruud',
    'taylor fritz':         'taylor fritz',
    'tommy paul':           'tommy paul',
    'ben shelton':          'ben shelton',
    'frances tiafoe':       'frances tiafoe',
    'sebastian korda':      'sebastian korda',
    'alex de minaur':       'alex de minaur',
    'stefanos tsitsipas':   'stefanos tsitsipas',
    'grigor dimitrov':      'grigor dimitrov',
    'hubert hurkacz':       'hubert hurkacz',
    'felix auger aliassime': 'felix auger-aliassime',

    # WTA examples
    'iga swiatek':           'iga swiatek',
    'aryna sabalenka':       'aryna sabalenka',
    'coco gauff':            'coco gauff',
    'elena rybakina':        'elena rybakina',
    'jessica pegula':        'jessica pegula',
    'caroline wozniacki':    'caroline wozniacki',
    'madison keys':          'madison keys',
    'mirra andreeva':        'mirra andreeva',
    'emma navarro':          'emma navarro',
    'danielle collins':      'danielle collins',
}

# ---------------------------------------------------------------------------
# STAT MAPPING (PrizePicks label -> internal column)
# Duplicated here for convenience in scanner without importing config
# ---------------------------------------------------------------------------
STAT_MAPPING = {
    'Total Games':      'total_games',
    'Total Games Won':  'games_won',
    'Total Sets':       'total_sets',
    'Aces':             'aces',
    'Break Points Won': 'bp_won',
    'Total Tie Breaks': 'total_tiebreaks',
    'Double Faults':    'double_faults',
}


# ---------------------------------------------------------------------------
# NAME NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Normalize a player name for fuzzy matching.
    Steps:
        1. Lowercase
        2. Strip accents (é -> e, ñ -> n)
        3. Remove punctuation (hyphens, apostrophes)
        4. Collapse whitespace
        5. Apply manual override if present
    """
    if not isinstance(name, str):
        return ''

    # Lowercase
    name = name.lower().strip()

    # Strip accents
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')

    # Remove hyphens, apostrophes, periods
    name = re.sub(r"[-'.]+", ' ', name)

    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()

    # Apply manual override
    return PP_NAME_MAP.get(name, name)


def match_player_name(pp_name: str, history_names: list) -> str | None:
    """
    Find the best matching name in the training history.
    Returns matched name or None.
    """
    norm_pp = normalize_name(pp_name)

    # Exact match
    norm_history = {normalize_name(n): n for n in history_names}
    if norm_pp in norm_history:
        return norm_history[norm_pp]

    # Partial match on last name
    pp_last = norm_pp.split()[-1] if norm_pp else ''
    candidates = [n for norm, n in norm_history.items() if pp_last in norm.split()]
    if len(candidates) == 1:
        return candidates[0]

    return None
