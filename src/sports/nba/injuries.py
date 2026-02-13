"""
Live NBA Injury Report Scraper

Scrapes ESPN's injury page to get real-time player availability status.
Used by scanner.py to calculate MISSING_USAGE (team impact of injured players).

Data Source:
    https://www.espn.com/nba/injuries
    
Usage:
    from src.sports.nba.injuries import get_injury_report
    injuries = get_injury_report()
    # {'LeBron James': 'OUT', 'Anthony Davis': 'QUESTIONABLE'}
"""

import requests
from bs4 import BeautifulSoup

INJURY_URL = "https://www.espn.com/nba/injuries"


def get_injury_report():
    """
    Scrape current NBA injury status from ESPN.
    
    Returns:
        dict: {player_name: status}  e.g. {'LeBron James': 'OUT'}
              Returns {} if scrape fails (safe fallback).
    """
    print("...Fetching Live Injury Report from ESPN")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(INJURY_URL, headers=headers)
        if response.status_code != 200:
            return {}

        soup = BeautifulSoup(response.content, 'html.parser')
        injury_data = {}

        tables = soup.find_all('table', class_='Table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    try:
                        name_div = cols[0].find('a')
                        if name_div:
                            name   = name_div.text.strip()
                            status = cols[1].text.strip()

                            clean_status = "Active"
                            if "Out" in status or "out" in status:
                                clean_status = "OUT"
                            elif "Doubtful" in status:
                                clean_status = "OUT"
                            elif "Questionable" in status:
                                clean_status = "QUESTIONABLE"
                            elif "Day-To-Day" in status:
                                clean_status = "GTD"

                            injury_data[name] = clean_status
                    except:
                        continue

        print(f"✅ Loaded {len(injury_data)} injury reports.")
        return injury_data

    except Exception as e:
        print(f"❌ Injury Scrape Failed: {e}")
        return {}
