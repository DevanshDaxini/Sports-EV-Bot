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
    
    Tasks:
    1. Load 'raw_game_logs.csv' and 'player_positions.csv' into DataFrames.
    2. Merge them on 'PLAYER_ID' so every row has a 'POSITION'.
    3. Fill missing positions with 'Unknown'.
    4. Convert 'GAME_DATE' to datetime objects.
    5. Sort by 'PLAYER_ID' and 'GAME_DATE' (Oldest to Newest).
    
    Returns:
        pd.DataFrame: The merged and sorted DataFrame.
    """
    print("...Loading and Merging Data")
    
    # TODO: Write your loading and merging logic here
    df = None 
    
    return df

def add_advanced_stats(df):
    """
    Step 2: Feature Engineering - Efficiency Metrics
    
    Tasks:
    1. Calculate True Shooting % (TS_PCT).
       Formula: PTS / (2 * (FGA + 0.44 * FTA))
    2. Calculate Game Score (A measure of productivity).
       Formula: PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
       
    Returns:
        pd.DataFrame: The DataFrame with new columns 'TS_PCT' and 'GAME_SCORE'.
    """
    print("...Calculating Advanced Stats")
    
    # TODO: Write your efficiency formulas here
    
    return df

def add_rolling_features(df):
    """
    Step 3: Feature Engineering - Recent Form (The "Hot/Cold" Factor)
    
    Tasks:
    1. Group the data by 'PLAYER_ID'.
    2. For each stat in TARGET_STATS (plus 'MIN' and 'GAME_SCORE'):
       a. Calculate the Rolling Average for the LAST 5 games.
       b. Calculate the Rolling Average for the LAST 20 games.
       c. Calculate the Expanding Mean (Season Average).
       
    CRITICAL: You MUST use .shift(1) before rolling. 
    (We cannot use tonight's points to predict tonight's points).
    
    Returns:
        pd.DataFrame: The DataFrame with columns like 'PTS_L5', 'REB_L20', etc.
    """
    print("...Calculating Rolling Averages")
    
    # TODO: Write your rolling window logic here
    # Hint: df.groupby('PLAYER_ID')[stat].transform(lambda x: ...)
    
    return df

def add_context_features(df):
    """
    Step 4: Feature Engineering - Context (Rest, Home/Away, Opponent)
    
    Tasks:
    1. Create 'IS_HOME': 1 if the 'MATCHUP' column contains 'vs.', 0 if '@'.
    2. Create 'DAYS_REST': Days since the previous game for this player.
       (Hint: Group by player, diff the date, fill NA with 3, clip at 7).
    3. Create 'OPPONENT': Extract the team code from 'MATCHUP' (e.g. 'GSW').
    
    Returns:
        pd.DataFrame: The DataFrame with 'IS_HOME', 'DAYS_REST', and 'OPPONENT'.
    """
    print("...Adding Context Features")
    
    # TODO: Write your context extraction logic here
    
    return df

def add_defense_vs_position(df):
    """
    Step 5: Feature Engineering - Defense vs. Position
    
    Tasks:
    1. Calculate the average stats allowed by each 'OPPONENT' against each 'POSITION'.
       (e.g., How many PTS does BOS allow to PGs on average?)
    2. Create a lookup table/mapping for this.
    3. Merge or Map these averages back into the main DataFrame.
    4. Rename the columns to something like 'OPP_POS_PTS'.
    
    Returns:
        pd.DataFrame: The DataFrame with new defensive metrics.
    """
    print("...Calculating Defense vs. Position")
    
    # TODO: Write your groupby and merge logic here
    
    return df

def main():
    # 1. Load
    df = load_and_merge_data()
    if df is None: return

    # 2. Engineer
    df = add_advanced_stats(df)
    df = add_context_features(df) # (Rest, Home/Away needs to happen before Rolling/Defense)
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