import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
DATA_FILE = 'data/training_dataset.csv'
MODEL_DIR = 'models'
TEST_START_DATE = '2025-02-01'  # We test on games after this date

# 1. DEFINE TARGETS
# These are the columns the model will try to predict.
TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV',  # Base Stats
    'PRA', 'PR', 'PA', 'RA', 'SB',                     # Combo Stats (SB = Steals+Blocks)
    'FGM', 'FTM', 'FTA'                                # Efficiency Stats
]

# 2. DEFINE FEATURES
# The input data the model learns from.
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

# Add Defense Columns dynamically
# (We assume features.py created columns like 'OPP_PTS_ALLOWED')
for stat in ['PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')

def ensure_combo_stats(df):
    """
    Safety Check: Calculates combo stats if they are missing from the CSV.
    """
    print("...Verifying Combo Stats (PRA, SB, etc.)")
    
    # Points + Rebounds + Assists
    if 'PRA' not in df.columns:
        df['PRA'] = df['PTS'] + df['REB'] + df['AST']
        
    # Points + Rebounds
    if 'PR' not in df.columns:
        df['PR'] = df['PTS'] + df['REB']
        
    # Points + Assists
    if 'PA' not in df.columns:
        df['PA'] = df['PTS'] + df['AST']
        
    # Rebounds + Assists
    if 'RA' not in df.columns:
        df['RA'] = df['REB'] + df['AST']
        
    # Steals + Blocks (Stocks)
    if 'SB' not in df.columns:
        df['SB'] = df['STL'] + df['BLK']
        
    return df

def train_and_evaluate():
    print("--- STARTING TRAINING PIPELINE ---")
    
    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print("ERROR: Training data not found. Run features.py first.")
        return
        
    df = pd.read_csv(DATA_FILE)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # 2. Calculate Combos (Safety Step)
    df = ensure_combo_stats(df)
    
    # 3. Split Data (Time Series Split)
    train_df = df[df['GAME_DATE'] < TEST_START_DATE]
    test_df = df[df['GAME_DATE'] >= TEST_START_DATE]
    
    print(f"Training Set: {len(train_df)} games")
    print(f"Testing Set:  {len(test_df)} games")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 4. Train Loop
    for target in TARGETS:
        print(f"\nTraining Model for: {target}...")
        
        # Check if target exists (e.g., if raw logs didn't have FTA)
        if target not in df.columns:
            print(f" -> SKIPPING {target} (Column not found in data)")
            continue

        X_train = train_df[FEATURES]
        y_train = train_df[target]
        X_test = test_df[FEATURES]
        y_test = test_df[target]
        
        # Configure XGBoost
        model = xgb.XGBRegressor(
            n_estimators=1000, 
            learning_rate=0.05, 
            max_depth=5, 
            early_stopping_rounds=50,
            n_jobs=-1
        )
        
        # Train
        model.fit(
            X_train, y_train, 
            eval_set=[(X_test, y_test)], 
            verbose=False
        )
        
        # Evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f" -> MAE: {mae:.2f} (On average, off by {mae:.2f} {target})")
        print(f" -> R2 Score: {r2:.3f} (Predictive Power)")
        
        # Save
        model_path = f"{MODEL_DIR}/{target}_model.json"
        model.save_model(model_path)
        print(f" -> Saved to {model_path}")

    print("\n--- ALL MODELS TRAINED ---")

if __name__ == "__main__":
    train_and_evaluate()