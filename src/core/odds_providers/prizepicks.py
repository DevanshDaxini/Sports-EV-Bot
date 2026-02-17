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
CACHE_FILE             = os.path.join(CACHE_DIR, 'prizepicks_cache.json')
CACHE_DURATION_MINUTES = 30


class PrizePicksClient:

    # -----------------------------------------------------------------------
    # partner-api endpoint — no Cloudflare WAF, returns full board
    # per_page=1000  → get everything in one shot (avoids pagination)
    # single_stat=true → only standard single-stat lines (no internal combos)
    # -----------------------------------------------------------------------
    BASE_URL = "https://partner-api.prizepicks.com/projections?per_page=1000&single_stat=true"

    # NBA league_id on PrizePicks
    LEAGUE_ID_MAP = {
        'NBA':  7,
        'NFL':  1,
        'MLB':  3,
        'NHL':  8,
        'WNBA': 9,
        'CBB':  4,
    }

    # Stat name normalisation: PrizePicks display name → model target code
    STAT_NORMALIZATION = {
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

    def fetch_board(self, league_filter=None, date_filter=None):
        """
        Fetch current PrizePicks prop board.

        Args:
            league_filter (str): e.g. 'NBA'  — filters by league name.
            date_filter   (str): e.g. '2026-02-19' — filters by game date.

        Returns:
            pd.DataFrame with columns: Player, League, Stat, Line, Date
            Returns empty DataFrame on any failure.
        """
        # --- 1. Try disk cache first ---
        cached = self._load_cache()
        if cached is not None:
            df = pd.DataFrame(cached)
            return self._apply_filters(df, league_filter, date_filter)

        # --- 2. Build URL (add league_id filter if we know it) ---
        url = self.BASE_URL
        if league_filter and league_filter in self.LEAGUE_ID_MAP:
            url += f"&league_id={self.LEAGUE_ID_MAP[league_filter]}"

        # Small random delay — be polite, avoid hammering the server
        time.sleep(random.uniform(0.5, 1.5))

        # --- 3. Fetch ---
        try:
            response = self.session.get(url, timeout=15)
        except requests.exceptions.ConnectionError as e:
            print(f"❌ PrizePicks: connection error — {e}")
            return pd.DataFrame()
        except requests.exceptions.Timeout:
            print("❌ PrizePicks: request timed out (15s)")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ PrizePicks: unexpected error — {e}")
            return pd.DataFrame()

        # --- 4. Handle bad status ---
        if response.status_code == 403:
            print("⚠️  PrizePicks returned 403 Forbidden")
            print("   The partner endpoint is normally unprotected — check your network / IP.")
            return pd.DataFrame()

        if response.status_code == 429:
            print("⚠️  PrizePicks returned 429 Too Many Requests — rate limited")
            return pd.DataFrame()

        if response.status_code != 200:
            print(f"⚠️  PrizePicks returned unexpected status {response.status_code}")
            return pd.DataFrame()

        # --- 5. Parse JSON ---
        try:
            data = response.json()
        except Exception as e:
            print(f"❌ PrizePicks: failed to parse JSON — {e}")
            return pd.DataFrame()

        # --- 6. Build clean rows ---
        clean_lines = self._parse_response(data)

        if not clean_lines:
            print("⚠️  PrizePicks: parsed 0 lines from response")
            return pd.DataFrame()

        # --- 7. Save to disk cache ---
        self._save_cache(clean_lines)

        df = pd.DataFrame(clean_lines)
        return self._apply_filters(df, league_filter, date_filter)

    def fetch_lines_dict(self, league_filter='NBA', date_filter=None):
        """
        Fetch PrizePicks lines as a nested dict with normalised stat names.

        Returns:
            dict: {player_name: {stat_code: line_value}}
            e.g.  {'LeBron James': {'PTS': 25.5, 'PRA': 43.5}}
        """
        df = self.fetch_board(league_filter=league_filter, date_filter=date_filter)
        if df.empty:
            return {}

        lines_dict = {}
        for _, row in df.iterrows():
            player   = row['Player']
            raw_stat = row['Stat']
            line     = row['Line']
            if not player:
                continue
            if player not in lines_dict:
                lines_dict[player] = {}
            norm_stat = self.STAT_NORMALIZATION.get(raw_stat, self.stat_map.get(raw_stat, raw_stat))
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

            # --- Skip promos and non-standard lines ---
            if attrs.get('is_promo') is True:
                continue
            if attrs.get('odds_type') not in ('standard', None, ''):
                continue

            # --- Resolve player name ---
            # Shape A (old): relationships.new_player.data.id → player_map
            # Shape B (new): attributes.name or attributes.player_name directly
            player_name = None

            if 'new_player' in rels:
                try:
                    p_id        = str(rels['new_player']['data']['id'])
                    player_name = player_map.get(p_id)
                except (KeyError, TypeError):
                    pass

            if not player_name:
                # Fallback: some partner-api responses embed name in attributes
                player_name = attrs.get('player_name') or attrs.get('name')

            if not player_name:
                continue  # Can't identify player — skip

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
            })

        return clean_lines

    def _apply_filters(self, df, league_filter, date_filter):
        """Apply league and date filters to a board DataFrame."""
        if df.empty:
            return df
        if league_filter:
            df = df[df['League'] == league_filter]
        if date_filter:
            df = df[df['Date'] == date_filter]
        if df.empty:
            print(f"⚠️  PrizePicks: no lines after filtering"
                  + (f" (league={league_filter})" if league_filter else "")
                  + (f" (date={date_filter})"   if date_filter   else ""))
        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Disk cache (30-minute TTL, same pattern as fanduel.py)
    # -----------------------------------------------------------------------

    def _load_cache(self):
        """Return cached data list if fresh, else None."""
        if not os.path.exists(CACHE_FILE):
            return None
        try:
            age_mins = (time.time() - os.path.getmtime(CACHE_FILE)) / 60
            if age_mins < CACHE_DURATION_MINUTES:
                print(f"   ♻️  Using saved PrizePicks data from {int(age_mins)} min(s) ago."
                      f"  (Expires in {int(CACHE_DURATION_MINUTES - age_mins)} min(s))")
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            else:
                print(f"   ⚠️  PrizePicks cache is {int(age_mins)} min(s) old — fetching fresh data.")
                return None
        except Exception as e:
            print(f"   ⚠️  Could not read PrizePicks cache: {e}")
            return None

    def _save_cache(self, data_list):
        """Save raw lines list to disk."""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump(data_list, f)
            print(f"   💾 PrizePicks data cached to '{CACHE_FILE}'.")
        except Exception as e:
            print(f"   ⚠️  Could not save PrizePicks cache: {e}")


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
    lines  = client.fetch_lines_dict(league_filter='NBA')

    if lines:
        print(f"\n✅ Success! Fetched lines for {len(lines)} players")
        sample = list(lines.keys())[0]
        print(f"\nExample → {sample}: {lines[sample]}")
    else:
        print("\n❌ Failed to fetch data")