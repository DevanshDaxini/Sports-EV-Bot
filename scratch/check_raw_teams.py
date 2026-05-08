import pandas as pd

RAW_FILE = 'data/nba/raw/raw_game_logs.csv'
df = pd.read_csv(RAW_FILE)

harden = df[df['PLAYER_NAME'] == 'James Harden'].sort_values('GAME_DATE').tail(5)
print("--- RAW: JAMES HARDEN ---")
print(harden[['GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'PLAYER_ID', 'PTS']])

garland = df[df['PLAYER_NAME'] == 'Darius Garland'].sort_values('GAME_DATE').tail(5)
print("\n--- RAW: DARIUS GARLAND ---")
print(garland[['GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'PLAYER_ID', 'PTS']])
