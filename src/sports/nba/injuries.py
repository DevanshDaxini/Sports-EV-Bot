"""
Live NBA Injury Report Scraper

Scrapes ESPN's injury page to get real-time player availability status.
Falls back to CBS Sports for additional coverage (long-term IR players).
Used by scanner.py to calculate MISSING_USAGE (team impact of injured players).

Data Sources:
    1. https://www.espn.com/nba/injuries         (primary)
    2. https://www.cbssports.com/nba/injuries/    (supplemental)

Usage:
    from src.sports.nba.injuries import get_injury_report
    injuries = get_injury_report()
    # {'LeBron James': 'OUT', 'Anthony Davis': 'QUESTIONABLE'}
"""

import requests
from bs4 import BeautifulSoup

INJURY_URL = "https://www.espn.com/nba/injuries"
CBS_INJURY_URL = "https://www.cbssports.com/nba/injuries/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
}


def _classify_status(raw_status):
    """Normalise a raw status string into a clean category."""
    s = raw_status.lower().strip()
    if "out" in s or "sidelined" in s:
        return "OUT"
    if "doubtful" in s:
        return "OUT"         # treat doubtful as effectively OUT
    if "questionable" in s:
        return "QUESTIONABLE"
    if "day-to-day" in s or "day to day" in s:
        return "GTD"
    if "probable" in s:
        return "GTD"
    return "Active"


def _scrape_espn():
    """
    Scrape ESPN injury report.

    ESPN table columns (Feb 2026 layout):
        [0] Name  [1] Pos  [2] Est. Return  [3] Status  [4] Comment
    """
    try:
        response = requests.get(INJURY_URL, headers=_HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"   ⚠️  ESPN returned status {response.status_code}")
            return {}

        soup = BeautifulSoup(response.content, 'html.parser')
        injury_data = {}

        tables = soup.find_all('table', class_='Table')
        for table in tables:
            # Detect status column index from header row (defensive)
            status_col = 3  # default for current layout
            header_row = table.find('tr')
            if header_row:
                ths = header_row.find_all('th')
                for idx, th in enumerate(ths):
                    if th.text.strip().lower() in ('status', 'stat'):
                        status_col = idx
                        break

            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) > status_col:
                    try:
                        name_el = cols[0].find('a')
                        if name_el:
                            name = name_el.text.strip()
                            raw_status = cols[status_col].text.strip()
                            injury_data[name] = _classify_status(raw_status)
                    except Exception:
                        continue

        return injury_data

    except Exception as e:
        print(f"   ⚠️  ESPN scrape failed: {e}")
        return {}


def _scrape_cbs():
    """
    Scrape CBS Sports injury report for supplemental coverage.
    CBS often lists long-term IR players that ESPN omits.
    """
    try:
        response = requests.get(CBS_INJURY_URL, headers=_HEADERS, timeout=15)
        if response.status_code != 200:
            return {}

        soup = BeautifulSoup(response.content, 'html.parser')
        injury_data = {}

        # CBS uses TableBase class
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    try:
                        name_el = cols[0].find('a')
                        if not name_el:
                            # CBS sometimes has name as plain text in a span
                            name_el = cols[0].find('span', class_='CellPlayerName--long')
                        if name_el:
                            name = name_el.text.strip()
                            # CBS status is usually in the last meaningful column
                            raw_status = cols[-1].text.strip()
                            if not raw_status:
                                raw_status = cols[-2].text.strip()
                            status = _classify_status(raw_status)
                            if status != "Active":
                                injury_data[name] = status
                    except Exception:
                        continue

        return injury_data
    except Exception:
        return {}


def _is_abbrev_match(name_a, name_b):
    """
    Check if two names refer to the same player, accounting for abbreviations.
    e.g. "A. Davis" ↔ "Anthony Davis",  "J. Jackson Jr." ↔ "Jaren Jackson Jr."
    """
    _SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv'}

    def _split(n):
        parts = n.replace('.', '').split()
        core = [p for p in parts if p.lower().rstrip('.') not in _SUFFIXES]
        suffix = [p for p in parts if p.lower().rstrip('.') in _SUFFIXES]
        return core, suffix

    core_a, suf_a = _split(name_a)
    core_b, suf_b = _split(name_b)

    if len(core_a) < 2 or len(core_b) < 2:
        return False
    # Last names must match exactly
    if core_a[-1].lower() != core_b[-1].lower():
        return False
    # One of the first parts must be an abbreviation (1-2 chars)
    fa, fb = core_a[0], core_b[0]
    is_a_abbrev = len(fa) <= 2
    is_b_abbrev = len(fb) <= 2
    if not (is_a_abbrev or is_b_abbrev):
        return False  # both are full names — not an abbreviation match
    # First initials must match
    if fa[0].lower() != fb[0].lower():
        return False
    return True


def _already_tracked(name, existing_names):
    """Check if a name (possibly abbreviated) already exists in the injury dict."""
    if name in existing_names:
        return True
    for existing in existing_names:
        if _is_abbrev_match(name, existing) or _is_abbrev_match(existing, name):
            return True
    return False


def get_injury_report():
    """
    Fetch combined injury report from multiple sources.

    Returns:
        dict: {player_name: status}  e.g. {'LeBron James': 'OUT'}
              Returns {} if all scrapes fail (safe fallback).
    """
    print("...Fetching Live Injury Report from ESPN")

    # Primary: ESPN
    injury_data = _scrape_espn()

    # Supplemental: CBS (fills gaps — only adds truly new players)
    try:
        cbs_data = _scrape_cbs()
        existing_names = set(injury_data.keys())
        added = 0
        for name, status in cbs_data.items():
            if not _already_tracked(name, existing_names):
                injury_data[name] = status
                existing_names.add(name)
                added += 1
        if added:
            print(f"   📋 +{added} additional players from CBS Sports")
    except Exception:
        pass  # CBS is best-effort

    out_count = sum(1 for s in injury_data.values() if s == "OUT")
    print(f"✅ Loaded {len(injury_data)} injury reports ({out_count} OUT).")
    return injury_data
