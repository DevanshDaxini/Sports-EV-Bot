import pandas as pd
import xgboost as xgb
import os
from nba_api.stats.endpoints import ScoreboardV2, CommonTeamRoster
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_DIR = 'models'
DATA_FILE = 'data/training_dataset.csv'
today_str = datetime.now().strftime('%Y-%m-%d')

# The stats we want to predict
TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'SB',
    'FGM', 'FTM', 'FTA'
]

# The exact features the model expects (Must match train.py!)
FEATURES = [
    'PTS_L5', 'PTS_L20', 'PTS_Season',
    'REB_L5', 'REB_L20', 'REB_Season',
    'AST_L5', 'AST_L20', 'AST_Season',
    'FG3M_L5', 'FG3M_L20', 'FG3M_Season',
    'STL_L5', 'STL_L20', 'STL_Season',
    'BLK_L5', 'BLK_L20', 'BLK_Season',
    'TOV_L5', 'TOV_L20', 'TOV_Season',
    'MIN_L5', 'MIN_L20', 'MIN_Season',
    'GAME_SCORE_L5', 'GAME_SCORE_L20', 'GAME_SCORE_Season',
    'TS_PCT', 'DAYS_REST', 'IS_HOME'
]

# Add Defense Columns
for stat in ['PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')

def load_data():
    """Loads the historical data to calculate 'Last 5', 'Season Avg', etc."""
    if not os.path.exists(DATA_FILE):
        print("ERROR: Training data not found.")
        return None
    df = pd.read_csv(DATA_FILE)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df

def get_todays_games():
    """Fetches today's NBA schedule using ScoreboardV2."""
    print(f"--- FETCHING GAMES FOR {today_str} ---")
    
    # UPDATED: Use ScoreboardV2 instead of scoreboard
    board = ScoreboardV2(game_date=today_str, league_id='00', day_offset=0)
    games = board.game_header.get_data_frame()
    
    if games.empty:
        print("No games scheduled for today.")
        return []
    
    game_list = []
    # ScoreboardV2 column names are often ALL CAPS
    for _, game in games.iterrows():
        # Matchup isn't always in V2, so we construct it manually
        # Note: Some versions use 'HOME_TEAM_ID', others 'TEAM_ID_HOME'
        # We will try standard V2 columns
        try:
            home_id = game['HOME_TEAM_ID']
            visitor_id = game['VISITOR_TEAM_ID']
            # We can't easily get City names from this endpoint alone, 
            # so we just store IDs and fetch names later if needed.
            game_list.append({
                'GAME_ID': game['GAME_ID'],
                'HOME_TEAM_ID': home_id,
                'VISITOR_TEAM_ID': visitor_id,
                'MATCHUP': f"Game {game['GAME_ID']}" # Placeholder name
            })
        except KeyError:
            print(f"Skipping malformed game row: {game}")
            
    return game_list

def get_roster(team_id):
    """Fetches the active roster for a team."""
    # CommonTeamRoster is still valid
    roster = CommonTeamRoster(team_id=team_id, season='2025-26')
    return roster.common_team_roster.get_data_frame()['PLAYER_ID'].tolist()

def prepare_player_features(player_id, opponent_id, is_home, df_history):
    """
    Constructs the feature row for a player based on their MOST RECENT game.
    Essentially: "What are his stats entering tonight?"
    """
    # 1. Get player's history
    player_games = df_history[df_history['PLAYER_ID'] == player_id].sort_values('GAME_DATE')
    
    if player_games.empty:
        return None  # Player not in our database
        
    # 2. Get the most recent row
    last_game = player_games.iloc[-1]
    
    # 3. Build the Feature Row
    features = {}
    
    # Copy the Rolling Stats (L5, L20)
    for col in FEATURES:
        if col in last_game:
            features[col] = last_game[col]
            
    # 4. Update Context Features (The stuff that changes tonight)
    features['IS_HOME'] = 1 if is_home else 0
    
    # Calculate Real Rest Days
    last_date = last_game['GAME_DATE']
    today = pd.to_datetime(today_str)
    days_rest = (today - last_date).days
    features['DAYS_REST'] = min(days_rest, 7)
    
    return pd.DataFrame([features])

def scan_market():
    # 1. Load History & Models
    df_history = load_data()
    if df_history is None: return
    
    games = get_todays_games()
    print(f"Found {len(games)} games.")
    
    if not games:
        return

    # Load all models once
    models = {}
    for target in TARGETS:
        model_path = f"{MODEL_DIR}/{target}_model.json"
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            models[target] = model
            
    print("--- STARTING SCAN ---")
    
    predictions = []
    
    for game in games:
        print(f"Scanning Game {game['GAME_ID']}...")
        
        # Get Players for both teams
        try:
            home_players = get_roster(game['HOME_TEAM_ID'])
            away_players = get_roster(game['VISITOR_TEAM_ID'])
        except Exception as e:
            print(f"Error fetching roster for game: {e}")
            continue
        
        all_players = [(p, True) for p in home_players] + [(p, False) for p in away_players]
        
        for player_id, is_home in all_players:
            # Prepare Input Data
            input_row = prepare_player_features(player_id, 0, is_home, df_history)
            
            if input_row is None: continue 
            
            # Predict for ALL targets
            player_preds = {'PLAYER_ID': player_id}
            
            # Get Name
            try:
                name_row = df_history[df_history['PLAYER_ID'] == player_id].iloc[0]
                player_preds['NAME'] = name_row['PLAYER_NAME']
                player_preds['TEAM'] = name_row['TEAM_ABBREVIATION'] # Helpful context
            except:
                player_preds['NAME'] = f"ID_{player_id}"
                player_preds['TEAM'] = "N/A"
                
            for target, model in models.items():
                # Ensure columns match exactly
                # If a feature is missing (rare), fill with 0 to prevent crash
                missing_cols = set(model.feature_names_in_) - set(input_row.columns)
                for c in missing_cols:
                    input_row[c] = 0
                    
                input_row = input_row[model.feature_names_in_]
                pred = model.predict(input_row)[0]
                player_preds[target] = round(pred, 1)
                
            predictions.append(player_preds)
            
    # Convert to DataFrame and Show Top Picks
    results = pd.DataFrame(predictions)
    
    if results.empty:
        print("No predictions generated.")
        return

    # Filter for significant players (PTS > 10)
    key_players = results[results['PTS'] > 10].sort_values('PTS', ascending=False)
    
    print("\n--- TOP SCORING PROJECTIONS FOR TONIGHT ---")
    # Show clean columns
    cols_to_show = ['NAME', 'TEAM', 'PTS', 'REB', 'AST', 'PRA']
    print(key_players[cols_to_show].head(15).to_string(index=False))
    
    # Save all projections
    results.to_csv('data/todays_projections.csv', index=False)
    print(f"\nFull projections saved to data/todays_projections.csv")

if __name__ == "__main__":
    scan_market()