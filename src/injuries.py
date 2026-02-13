"""
Live NBA Injury Report Scraper

Scrapes ESPN's injury page to get real-time player availability status.
Used by scanner.py to calculate MISSING_USAGE (team impact of injured players).

Data Source:
    https://www.espn.com/nba/injuries
    
Status Codes:
    'OUT' - Player will not play
    'QUESTIONABLE' - Uncertain (50/50)
    'GTD' - Game-time decision
    'Active' - Available to play
    
Note:
    'Doubtful' is treated as 'OUT' (rarely plays)
    
Usage:
    from src.injuries import get_injury_report
    injuries = get_injury_report()
    # {'LeBron James': 'OUT', 'Anthony Davis': 'QUESTIONABLE'}
    
Limitations:
    - Requires web scraping (can break if ESPN changes HTML)
    - May be delayed vs official team reports
    - Returns empty dict {} if scrape fails (safe fallback)
"""

import requests
from bs4 import BeautifulSoup

# URL for ESPN NBA Injuries
INJURY_URL = "https://www.espn.com/nba/injuries"

def get_injury_report():
    """
    Scrape current NBA injury status from ESPN.
    
    Returns:
        dict: Mapping of player name (str) to status (str)
              Example: {'LeBron James': 'OUT', 'Luka Doncic': 'GTD'}
              Returns {} if scrape fails
              
    Status Values:
        - 'OUT': Confirmed out
        - 'QUESTIONABLE': 50/50 to play
        - 'GTD': Game-time decision
        - 'Active': Not on injury report
        
    Process:
        1. Send GET request with realistic User-Agent header
        2. Parse HTML with BeautifulSoup
        3. Find all injury tables
        4. Extract player name and status from each row
        5. Normalize status codes (e.g., "Out for Season" → "OUT")
        
    Note:
        Prints "✅ Loaded X injury reports" on success
        Prints "❌ Injury Scrape Failed" on error (but doesn't crash)
        
    Example:
        injuries = get_injury_report()
        if injuries.get('LeBron James') == 'OUT':
            print("LeBron is out tonight")
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

        # Scan all tables on the ESPN page
        tables = soup.find_all('table', class_='Table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    try:
                        name_div = cols[0].find('a')
                        if name_div:
                            name = name_div.text.strip()
                            status = cols[1].text.strip()
                            
                            # Normalize Status for our model
                            clean_status = "Active"
                            if "Out" in status or "out" in status: clean_status = "OUT"
                            elif "Doubtful" in status: clean_status = "OUT" # Treat doubtful as out
                            elif "Questionable" in status: clean_status = "QUESTIONABLE"
                            elif "Day-To-Day" in status: clean_status = "GTD"
                            
                            injury_data[name] = clean_status
                    except:
                        continue
                        
        print(f"✅ Loaded {len(injury_data)} injury reports.")
        return injury_data

    except Exception as e:
        print(f"❌ Injury Scrape Failed: {e}")
        return {}