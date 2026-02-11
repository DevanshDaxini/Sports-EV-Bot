import pandas as pd
import os
from datetime import datetime
from nba_api.stats.endpoints import playergamelogs
from src.config import STAT_MAP

# PrizePicks Name -> NBA API Column Name
NBA_STAT_MAP = {
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Assists': 'AST',
    '3-PT Made': 'FG3M',
    'Blocks': 'BLK',
    'Steals': 'STL',
    'Turnovers': 'TOV',
    'Field Goals Made': 'FGM',
    'Free Throws Made': 'FTM',
    'Free Throws Attempted': 'FTA',
    'Pts+Rebs+Asts': 'PRA',
    'Pts+Rebs': 'PR',
    'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA',
    'Blks+Stls': 'SB'
}

def get_user_date():
    """Asks the user for a date and validates it."""
    while True:
        date_str = input("\nEnter the date to grade (YYYY-MM-DD) or press Enter for Yesterday: ")
        if not date_str.strip():
            from datetime import timedelta
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            print("Invalid format! Please use YYYY-MM-DD")

def update_history_file(date_str, wins, losses, total_graded, win_rate):
    """Updates the master CSV file using the STRICT 6-column format."""
    history_file = "program_runs/win_rate_history.csv"
    new_row_data = {
        "Date": date_str,
        "Total_Bets": total_graded,
        "Wins": wins,
        "Losses": losses,
        "Win_Rate": f"{win_rate:.2f}%",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if os.path.exists(history_file):
        try:
            df_history = pd.read_csv(history_file)
            df_history = df_history[df_history['Date'] != date_str]
            df_new = pd.DataFrame([new_row_data])
            df_final = pd.concat([df_history, df_new], ignore_index=True)
        except Exception:
            df_final = pd.DataFrame([new_row_data])
    else:
        df_final = pd.DataFrame([new_row_data])

    df_final = df_final.sort_values(by='Date', ascending=True)
    df_final.to_csv(history_file, index=False)
    print(f"Updated history log: {history_file}")

def normalize_name(name):
    """
    Removes common suffixes to help match PrizePicks names to NBA API names.
    Ex: 'Tim Hardaway Jr.' -> 'Tim Hardaway'
    """
    name = name.lower().replace('.', '') # Remove periods
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)] # Cut off the suffix
    return name.strip()

def grade_bets():
    target_date = get_user_date()
    filename = f"program_runs/scan_{target_date}.csv"
    
    print(f"\n--- GRADING BETS FOR {target_date} ---")
    
    if not os.path.exists(filename):
        print(f"ERROR: No file found at {filename}")
        return

    df = pd.read_csv(filename)

    print("Fetching actual game results from NBA API...")
    # NOTE: Ensure season is correct (2025-26)
    logs = playergamelogs.PlayerGameLogs(season_nullable='2025-26', 
                                         date_from_nullable=target_date, 
                                         date_to_nullable=target_date)
    
    frames = logs.get_data_frames()
    if not frames:
        print("NBA API returned no data. (Are games finished? Is the date correct?)")
        return
        
    box_scores = frames[0]
    
    # Create look-up dict with NORMALIZED keys
    # We store BOTH the exact name and the normalized name to catch everything
    player_stats = {}
    for _, row in box_scores.iterrows():
        real_name = row['PLAYER_NAME']
        stats = row.to_dict()
        stats['PRA'] = row['PTS'] + row['REB'] + row['AST']
        stats['PR']  = row['PTS'] + row['REB']
        stats['PA']  = row['PTS'] + row['AST']
        stats['RA']  = row['REB'] + row['AST']
        stats['SB']  = row['STL'] + row['BLK']
        
        # Key 1: Exact Match (e.g. "LeBron James")
        player_stats[real_name] = stats
        
        # Key 2: Normalized Match (e.g. "Tim Hardaway" from "Tim Hardaway Jr.")
        norm_name = normalize_name(real_name)
        if norm_name != real_name.lower():
            player_stats[norm_name] = stats

    print(f"Found stats for {len(box_scores)} players.")

    wins = 0
    losses = 0
    pushes = 0
    total_graded = 0 
    
    results = []
    actuals = []

    for index, row in df.iterrows():
        pp_name = row['Player']
        prop = row['Stat']
        line = row['Line']
        side = row['Side']
        
        # 1. Try Exact Match
        stats = player_stats.get(pp_name)
        
        # 2. If fail, try Normalized Match
        if not stats:
            stats = player_stats.get(normalize_name(pp_name))
            
        if not stats:
            results.append("DNP/Unknown")
            actuals.append(0)
            continue
            
        nba_col = NBA_STAT_MAP.get(prop)
        if not nba_col:
            results.append("Unsupported Stat")
            actuals.append(0)
            continue
            
        actual_val = stats.get(nba_col, 0)
        actuals.append(actual_val)
        
        # Determine Outcome
        if (side == 'Over' and actual_val > line) or \
           (side == 'Under' and actual_val < line):
            results.append("WIN")
            wins += 1
            total_graded += 1
        elif actual_val == line:
            results.append("Push")
            pushes += 1
        else:
            results.append("LOSS")
            losses += 1
            total_graded += 1

    df['Result'] = results
    df['Actual'] = actuals
    df.to_csv(filename, index=False)
    print(f"Updated daily file with results: {filename}")
    
    if total_graded > 0:
        win_rate = (wins / total_graded) * 100
        print(f"\n--- REPORT CARD ({target_date}) ---")
        print(f"Wins:   {wins}")
        print(f"Losses: {losses}")
        print(f"Pushes: {pushes} (Excluded)")
        print(f"WIN RATE: {win_rate:.2f}%")
        
        update_history_file(target_date, wins, losses, total_graded, win_rate)
    else:
        print("No settled bets found.")

if __name__ == "__main__":
    grade_bets()

# Run this by doing: python3 -m src.grader