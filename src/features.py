import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
LOGS_FILE = 'data/raw_game_logs.csv'
POS_FILE = 'data/player_positions.csv'
OUTPUT_FILE = 'data/training_dataset.csv'

# The specific stats we want to predict (Target Variables)
TARGET_STATS = ['PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV']

def load_and_merge_data():
    """
    Step 1: Load the raw data and merge it with player positions.
    """
    print("...Loading and Merging Data")
    
    if not os.path.exists(POS_FILE) or not os.path.exists(LOGS_FILE):
        print("Error: Data files not found please run builder.py first")
        return None
    
    df_logs = pd.read_csv(LOGS_FILE)
    df_pos = pd.read_csv(POS_FILE)

    # FIX: Only merge the columns we need. 
    # If we merge the whole file, we get duplicate columns like 'PLAYER_NAME_x' and 'PLAYER_NAME_y'
    df = pd.merge(df_logs, df_pos[['PLAYER_ID', 'POSITION']], on='PLAYER_ID', how='left')

    # Fill missing values
    df['POSITION'] = df['POSITION'].fillna('Unknown')
    
    # Drop rows where vital data (like MATCHUP) is missing
    df = df.dropna(subset=['MATCHUP', 'GAME_DATE'])

    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'], ascending=True)
    
    return df

def add_advanced_stats(df):
    """
    Step 2: Feature Engineering - Efficiency Metrics
    """
    print("...Calculating Advanced Stats")
    
    # Calculate True Shooting Percentage
    df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    df['TS_PCT'] = df['TS_PCT'].fillna(0)

    # Calculate Game Score
    df['GAME_SCORE'] = (df['PTS'] + 
                        (0.4 * df['FGM']) - 
                        (0.7 * df['FGA']) - 
                        (0.4 * (df['FTA'] - df['FTM'])) + 
                        (0.7 * df['OREB']) + 
                        (0.3 * df['DREB']) + 
                        df['STL'] + 
                        (0.7 * df['AST']) + 
                        (0.7 * df['BLK']) - 
                        (0.4 * df['PF']) - 
                        df['TOV'])
    
    df['GAME_SCORE'] = df['GAME_SCORE'].fillna(0)

    return df

def add_rolling_features(df):
    """
    Step 3: Feature Engineering - Recent Form (The "Hot/Cold" Factor)
    """
    print("...Calculating Rolling Averages")
    
    grouped = df.groupby('PLAYER_ID')

    # We roll the Targets + Context stats
    stats_to_roll = TARGET_STATS + ['MIN', 'GAME_SCORE']

    for stat in stats_to_roll:
        # Last 5 Games
        df[f'{stat}_L5'] = grouped[stat].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        # Last 20 Games
        df[f'{stat}_L20'] = grouped[stat].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
        # Season Average
        df[f'{stat}_Season'] = grouped[stat].transform(lambda x: x.shift(1).expanding().mean())

    return df

def add_context_features(df):
    """
    Step 4: Feature Engineering - Context (Rest, Home/Away, Opponent)
    """
    print("...Adding Context Features")

    # 1. Home/Away
    # Ensure it's a string before checking 'vs.'
    df['IS_HOME'] = df['MATCHUP'].astype(str).apply(lambda x: 1 if 'vs.' in x else 0)
    
    # 2. Opponent Code
    # FIX: Convert to string first so 'NaN' doesn't crash the .split() method
    df['OPPONENT'] = df['MATCHUP'].astype(str).apply(lambda x: x.split(' ')[-1])
    
    # 3. Rest Days
    # Group by player to diff THEIR dates
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3)
    df['DAYS_REST'] = df['DAYS_REST'].clip(upper=7)
    
    return df

def add_defense_vs_position(df):
    """
    Step 5: Feature Engineering - Defense vs. Position
    """
    print("...Calculating Defense vs. Position")
    
    defense_group = df.groupby(['OPPONENT', 'POSITION'])
    
    for stat in TARGET_STATS:
        col_name = f'OPP_{stat}_ALLOWED'
        
        # Calculate expanding mean (history only)
        df[col_name] = defense_group[stat].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        
    # Fill gaps (first time matchup) with global position average
    for stat in TARGET_STATS:
        col_name = f'OPP_{stat}_ALLOWED'
        global_pos_avg = df.groupby('POSITION')[stat].transform('mean')
        df[col_name] = df[col_name].fillna(global_pos_avg)
    
    return df

def main():
    # 1. Load
    df = load_and_merge_data()
    if df is None: return

    # 2. Engineer
    df = add_advanced_stats(df)
    df = add_context_features(df) # Must run BEFORE rolling features
    df = add_rolling_features(df)
    df = add_defense_vs_position(df)
    
    # 3. Clean
    # Drop rows that have NaNs (usually the first few games of a season where L5 isn't ready)
    df = df.dropna()

    # 4. Save
    print(f"\nSuccess! Saving {len(df)} training rows to {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()