"""
PrizePicks Props Client

Fetches player prop lines from PrizePicks using the partner API endpoint,
which is not Cloudflare-protected unlike the public api.prizepicks.com endpoint.

Key fixes vs. original:
    - URL: partner-api.prizepicks.com (no Cloudflare WAF)
    - per_page=1000 to avoid pagination
    - Updated Chrome 131 User-Agent and matching sec-ch-ua headers
    - requests.Session() for persistent cookies across calls
    - 30-minute disk cache (same pattern as fanduel.py)
    - Handles both old and new PrizePicks JSON response shapes

Usage:
    from src.core.odds_providers.prizepicks import PrizePicksClient
    client = PrizePicksClient(stat_map=STAT_MAP)
    df     = client.fetch_board(league_filter='NBA')
    lines  = client.fetch_lines_dict(league_filter='NBA')
"""

import requests
import pandas as pd
import time
import random
import json
import os

# ---------------------------------------------------------------------------
# Disk cache config  (mirrors fanduel.py pattern)
# ---------------------------------------------------------------------------
CACHE_DIR              = 'prizepicks_cache'
CACHE_DURATION_MINUTES = 30


def _cache_file_for_league(league_filter):
    """Return per-league cache path to prevent cross-sport contamination."""
    tag = (league_filter or 'all').upper().replace(' ', '_')
    return os.path.join(CACHE_DIR, f'prizepicks_cache_{tag}.json')


class PrizePicksClient:

    # -----------------------------------------------------------------------
    # partner-api endpoint — no Cloudflare WAF, returns full board
    # per_page=1000  → get everything in one shot (avoids pagination)
    # single_stat=true → only standard single-stat lines (no internal combos)
    # -----------------------------------------------------------------------
    BASE_URL = "https://partner-api.prizepicks.com/projections?per_page=1000&single_stat=true"

    LEAGUE_ID_MAP = {
        'NBA':    7,
        'NFL':    1,
        'MLB':    3,
        'NHL':    8,
        'WNBA':   9,
        'CBB':    4,
        'TENNIS': 29,   # ATP + WTA combined on PrizePicks
    }

    # Stat name normalisation: PrizePicks display name → model target code
    STAT_NORMALIZATION = {
        # --- NBA ---
        'Points':                'PTS',
        'Rebounds':              'REB',
        'Assists':               'AST',
        'Pts+Rebs+Asts':         'PRA',
        'Pts+Rebs':              'PR',
        'Pts+Asts':              'PA',
        'Rebs+Asts':             'RA',
        'Blks+Stls':             'SB',
        '3-PT Made':             'FG3M',
        'Blocked Shots':         'BLK',
        'Blocks':                'BLK',
        'Steals':                'STL',
        'Turnovers':             'TOV',
        'Free Throws Made':      'FTM',
        'Field Goals Made':      'FGM',
        'Free Throws Attempted': 'FTA',
        'Field Goals Attempted': 'FGA',

        # --- Tennis ---
        'Total Games':           'total_games',
        'Total Games Won':       'games_won',
        'Total Sets':            'total_sets',
        'Aces':                  'aces',
        'Break Points Won':      'bp_won',
        'Total Tie Breaks':      'total_tiebreaks',
        'Double Faults':         'double_faults',
    }

    def __init__(self, stat_map=None):
        """
        Args:
            stat_map (dict): Optional extra stat-name overrides (merged on top
                             of STAT_NORMALIZATION at runtime).
        """
        self.stat_map = stat_map or {}

        # Persistent session — carries cookies automatically between calls,
        # which is important if PrizePicks sets any session cookies on first hit.
        self.session = requests.Session()

        # Chrome 131 headers (current as of early 2026).
        # Keeping the UA version current matters — WAFs cross-check the
        # Chrome version in the UA against the TLS/HTTP2 fingerprint.
        self.session.headers.update({
            "User-Agent":      (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept":          "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer":         "https://app.prizepicks.com/",
            "Origin":          "https://app.prizepicks.com",
            "Connection":      "keep-alive",
            "Sec-Fetch-Dest":  "empty",
            "Sec-Fetch-Mode":  "cors",
            "Sec-Fetch-Site":  "same-site",
            # sec-ch-ua brand list must match the Chrome version above
            "sec-ch-ua":          '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile":   "?0",
            "sec-ch-ua-platform": '"macOS"',
        })
        
    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def fetch_board(self, league_filter=None, date_filter=None, include_alts=False, league_id=None):
        """
        Fetch current PrizePicks prop board.

        Args:
            league_filter (str): e.g. 'NBA' or 'TENNIS' — filters by league name.
            date_filter   (str): e.g. '2026-02-19' — filters by game date.
            include_alts (bool): If True, include goblin/demon alt lines.
            league_id     (int): Direct league ID filter for the API.

        Returns:
            pd.DataFrame with columns: Player, League, Stat, Line, Date, OddsType
            Returns empty DataFrame on any failure.
        """
        # --- 1. Try disk cache first ---
        cached = self._load_cache(league_filter)
        if cached is not None:
            df = pd.DataFrame(cached)
            if not include_alts and 'OddsType' in df.columns:
                df = df[df['OddsType'].isin(['standard', ''])]
            return self._apply_filters(df, league_filter, date_filter)

        # --- 2. Build URL ---
        # League-specific fetch to avoid board truncation (1000 line limit)
        active_ids = []
        if league_id:
            active_ids = [league_id] if isinstance(league_id, (int, str)) else league_id
        elif league_filter:
            # Expand 'NBA' to include its sub-leagues (1H, 1Q)
            if league_filter.upper() == 'NBA':
                active_ids = [7, 84, 192]
            else:
                l_id = self.LEAGUE_ID_MAP.get(league_filter.upper())
                if l_id: active_ids = [l_id]

        url = self.BASE_URL
        if active_ids:
            # Append each league_id to the query
            for lid in active_ids:
                url += f"&league_id[]={lid}"

        # Small random delay — be polite
        time.sleep(random.uniform(0.5, 1.5))

        def _stale_cache_fallback(reason):
            """Return stale cache data rather than an empty result."""
            stale = self._load_stale_cache(league_filter)
            if stale is not None:
                print(f"   PrizePicks: {reason} — using stale cache")
                df = pd.DataFrame(stale)
                if not include_alts and 'OddsType' in df.columns:
                    df = df[df['OddsType'].isin(['standard', ''])]
                return self._apply_filters(df, league_filter, date_filter)
            print(f"   PrizePicks: {reason} — no cache available")
            return pd.DataFrame()

        # --- 3. Fetch ---
        try:
            response = self.session.get(url, timeout=15)
        except requests.exceptions.ConnectionError as e:
            return _stale_cache_fallback("connection error")
        except requests.exceptions.Timeout:
            return _stale_cache_fallback("request timed out")
        except Exception as e:
            return _stale_cache_fallback(f"unexpected error — {e}")

        # --- 4. Handle bad status ---
        if response.status_code == 403:
            return _stale_cache_fallback("403 Forbidden")

        if response.status_code == 429:
            return _stale_cache_fallback("rate limited (429)")

        if response.status_code != 200:
            return _stale_cache_fallback(f"unexpected status {response.status_code}")

        # --- 5. Parse JSON ---
        try:
            data = response.json()
        except Exception as e:
            return _stale_cache_fallback(f"failed to parse JSON — {e}")

        # --- 6. Build clean rows (always includes all line types) ---
        clean_lines = self._parse_response(data)

        if not clean_lines:
            return _stale_cache_fallback("0 lines parsed")

        # --- 7. Save ALL lines to disk cache (including alts) ---
        self._save_cache(clean_lines, league_filter)

        df = pd.DataFrame(clean_lines)
        if not include_alts and 'OddsType' in df.columns:
            df = df[df['OddsType'].isin(['standard', ''])]
        return self._apply_filters(df, league_filter, date_filter)

    def fetch_lines_dict(self, league_filter='NBA', date_filter=None):
        """
        Fetch PrizePicks lines as a nested dict with normalised stat names.

        Returns:
            dict: {player_name: {stat_code: line_value}}
            e.g.  {'Carlos Alcaraz': {'aces': 6.5, 'total_games': 22.5}}
        """
        df = self.fetch_board(league_filter=league_filter, date_filter=date_filter)
        if df.empty:
            return {}

        lines_dict = {}
        for _, row in df.iterrows():
            player   = row['Player']
            raw_stat = row['Stat']
            line     = row['Line']
            league   = row.get('League', '')
            if not player:
                continue
            if player not in lines_dict:
                lines_dict[player] = {}
                
            norm_stat = self.STAT_NORMALIZATION.get(raw_stat, self.stat_map.get(raw_stat, raw_stat))
            
            # CRITICAL FIX: PrizePicks uses the same raw stat names ("Points") for both
            # the full game 'NBA' tab and the 1st half 'NBA1H' tab.
            # If the league is NBA1H, we MUST append '_1H' so it targets the right model.
            if league.upper() == 'NBA1H' and not norm_stat.endswith('_1H'):
                norm_stat = f"{norm_stat}_1H"
                
            lines_dict[player][norm_stat] = float(line)

        return lines_dict

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _parse_response(self, data):
        """
        Parse the PrizePicks JSON response into a list of clean dicts.

        Handles both the old api.prizepicks.com shape and the newer
        partner-api shape, where player name can appear in different places.
        """
        projections_list = data.get('data', [])
        included_list    = data.get('included', [])

        # Build lookup maps from the 'included' array
        player_map = {}
        league_map = {}
        for item in included_list:
            item_type = item.get('type', '')
            item_id   = str(item.get('id', ''))
            attrs     = item.get('attributes', {})
            if item_type == 'new_player':
                player_map[item_id] = attrs.get('name', attrs.get('display_name', ''))
            elif item_type == 'league':
                league_map[item_id] = attrs.get('name', '')

        clean_lines = []

        for proj in projections_list:
            attrs = proj.get('attributes', {})
            rels  = proj.get('relationships', {})

            # --- Skip promos only (keep goblin/demon for cache) ---
            if attrs.get('is_promo') is True:
                continue

            odds_type = attrs.get('odds_type', 'standard') or 'standard'

            # --- Resolve player name ---
            player_name = None

            if 'new_player' in rels:
                try:
                    p_id        = str(rels['new_player']['data']['id'])
                    player_name = player_map.get(p_id)
                except (KeyError, TypeError):
                    pass

            if not player_name:
                player_name = attrs.get('player_name') or attrs.get('name')

            if not player_name:
                continue

            # --- Resolve league ---
            league_name = None
            if 'league' in rels:
                try:
                    l_id        = str(rels['league']['data']['id'])
                    league_name = league_map.get(l_id)
                except (KeyError, TypeError):
                    pass

            if not league_name:
                league_name = attrs.get('league', '')

            # --- Parse date ---
            start_time = attrs.get('start_time', '')
            game_date  = start_time.split('T')[0] if 'T' in start_time else 'Unknown'

            clean_lines.append({
                'Player': player_name,
                'League': league_name,
                'Stat':   attrs.get('stat_type', ''),
                'Line':   attrs.get('line_score', 0),
                'Date':   game_date,
                'OddsType': odds_type,
            })

        return clean_lines

    def fetch_lines_with_type(self, league_filter='NBA', date_filter=None):
        """
        Fetch PrizePicks lines with line type info (standard/goblin/demon).

        Prefers standard lines; falls back to goblin/demon if no standard exists.

        Returns:
            dict: {player_name: {stat_code: {'line': float, 'type': str}}}
        """
        df = self.fetch_board(league_filter=league_filter, date_filter=date_filter, include_alts=True)
        if df.empty:
            return {}

        lines_dict = {}
        for _, row in df.iterrows():
            player   = row['Player']
            raw_stat = row['Stat']
            line     = row['Line']
            league   = row.get('League', '')
            odds_type = row.get('OddsType', 'standard') or 'standard'
            if not player:
                continue
                
            norm_stat = self.STAT_NORMALIZATION.get(raw_stat, self.stat_map.get(raw_stat, raw_stat))
            
            if league.upper() == 'NBA1H' and not norm_stat.endswith('_1H'):
                norm_stat = f"{norm_stat}_1H"
                
            if player not in lines_dict:
                lines_dict[player] = {}
            # Prefer standard over alt — only overwrite if upgrading to standard
            if norm_stat in lines_dict[player]:
                existing_type = lines_dict[player][norm_stat]['type']
                if existing_type == 'standard':
                    continue  # keep standard, skip this alt
            lines_dict[player][norm_stat] = {'line': float(line), 'type': odds_type}

        return lines_dict

    def _apply_filters(self, df, league_filter, date_filter):
        """Apply league and date filters to a board DataFrame."""
        if df.empty:
            return df
        df_all = df  # keep reference to full board for diagnostics
        if league_filter:
            # Expand 'NBA' filter to automatically include 'NBA1H'
            allowed_leagues = [league_filter.upper()]
            if league_filter.upper() == 'NBA':
                allowed_leagues.append('NBA1H')
                
            df = df[df['League'].str.upper().isin(allowed_leagues)]
            
        if date_filter:
            df = df[df['Date'] == date_filter]
            
        if df.empty:
            # Show available leagues to help diagnose
            avail = df_all['League'].unique().tolist()
            print(f"   No {league_filter or 'matching'} lines on PrizePicks right now.")
            if avail:
                print(f"   Available leagues: {avail}")
            if league_filter and league_filter.upper() == 'TENNIS':
                print(f"   Tennis lines are posted 1-2 days before matches.")
        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Disk cache (30-minute TTL)
    # -----------------------------------------------------------------------

    def _load_cache(self, league_filter=None):
        """Return cached data list if fresh, else None."""
        cache_file = _cache_file_for_league(league_filter)
        if not os.path.exists(cache_file):
            return None
        try:
            age_mins = (time.time() - os.path.getmtime(cache_file)) / 60
            if age_mins < CACHE_DURATION_MINUTES:
                print(f"   Using cached PP data ({int(age_mins)}m ago, expires in {int(CACHE_DURATION_MINUTES - age_mins)}m)")
                with open(cache_file, 'r') as f:
                    return json.load(f)
            else:
                print(f"   PP cache expired ({int(age_mins)}m old) — fetching fresh")
                return None
        except Exception as e:
            print(f"   Warning: could not read PP cache: {e}")
            return None

    def _load_stale_cache(self, league_filter=None):
        """Return stale cache data regardless of age (fallback when live fetch fails)."""
        cache_file = _cache_file_for_league(league_filter)
        if not os.path.exists(cache_file):
            return None
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, data_list, league_filter=None):
        """Save raw lines list to disk under a per-league cache file."""
        cache_file = _cache_file_for_league(league_filter)
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(data_list, f)
            print(f"   PP data cached to '{cache_file}'")
        except Exception as e:
            print(f"   Warning: could not save PP cache: {e}")


# ---------------------------------------------------------------------------
# Standalone backwards-compat function
# ---------------------------------------------------------------------------

def fetch_current_lines_dict(league_filter='NBA', date_filter=None):
    """Standalone wrapper — maintains compatibility with old call sites."""
    client = PrizePicksClient()
    return client.fetch_lines_dict(league_filter=league_filter, date_filter=date_filter)


# ---------------------------------------------------------------------------
# Test block — run as:  python -m src.core.odds_providers.prizepicks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- TESTING PRIZEPICKS CLIENT ---")
    client = PrizePicksClient()

    print("\n[NBA]")
    nba_lines = client.fetch_lines_dict(league_filter='NBA')
    if nba_lines:
        print(f"✅ Fetched lines for {len(nba_lines)} NBA players")
        sample = list(nba_lines.keys())[0]
        print(f"   Example → {sample}: {nba_lines[sample]}")
    else:
        print("❌ Failed to fetch NBA data")

    print("\n[TENNIS]")
    # Force fresh fetch (bypass cache) for tennis test
    tennis_client = PrizePicksClient()
    tennis_df = tennis_client.fetch_board(league_filter='TENNIS')
    if not tennis_df.empty:
        print(f"✅ Fetched {len(tennis_df)} tennis lines")
        print(f"   Players: {tennis_df['Player'].nunique()}")
        print(f"   Stats: {tennis_df['Stat'].unique().tolist()}")
        print(f"\n   Sample:\n{tennis_df.head(5).to_string(index=False)}")
    else:
        print("❌ No tennis lines found")
        print("   Note: Tennis may not be active on PrizePicks today.")
        print("   If league_id=29 is wrong, run the debug below to find the correct ID:")
        print("   client.fetch_board()  # no filter → prints all leagues")
