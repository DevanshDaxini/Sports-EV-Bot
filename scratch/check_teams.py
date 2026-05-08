import pandas as pd

DATA_FILE = 'data/nba/processed/training_dataset.csv'
df = pd.read_csv(DATA_FILE)

# Print out James Harden's and Darius Garland's latest games to see the anomaly
harden = df[df['PLAYER_NAME'] == 'James Harden'].sort_values('GAME_DATE').tail(5)
garland = df[df['PLAYER_NAME'] == 'Darius Garland'].sort_values('GAME_DATE').tail(5)

print("--- JAMES HARDEN ---")
print(harden[['GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'PLAYER_ID', 'PTS']])

print("\n--- DARIUS GARLAND ---")
print(garland[['GAME_DATE', 'TEAM_ABBREVIATION', 'MATCHUP', 'PLAYER_ID', 'PTS']])
