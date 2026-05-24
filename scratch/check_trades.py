import pandas as pd

RAW_FILE = 'data/nba/raw/raw_game_logs.csv'
df = pd.read_csv(RAW_FILE)

harden = df[df['PLAYER_NAME'] == 'James Harden'].sort_values('GAME_DATE')
garland = df[df['PLAYER_NAME'] == 'Darius Garland'].sort_values('GAME_DATE')

print("--- JAMES HARDEN TEAMS ---")
print(harden[['GAME_DATE', 'TEAM_ABBREVIATION']].drop_duplicates(subset=['TEAM_ABBREVIATION'], keep='first'))

print("\n--- DARIUS GARLAND TEAMS ---")
print(garland[['GAME_DATE', 'TEAM_ABBREVIATION']].drop_duplicates(subset=['TEAM_ABBREVIATION'], keep='first'))
