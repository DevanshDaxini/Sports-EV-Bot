"""
XGBoost Model Training Pipeline

Trains separate regression models for 17 NBA statistics using time-series split
validation. Implements feature leakage prevention to ensure models don't "peek"
at the stat they're predicting.

Models Trained:
    Base Stats: PTS, REB, AST, FG3M, FG3A, BLK, STL, TOV, FGM, FGA, FTM, FTA
    Combo Stats: PRA, PR, PA, RA, SB
    
Evaluation Metrics:
    - MAE (Mean Absolute Error): Average prediction error in stat units
    - R² Score: Explained variance (0 to 1, higher is better)
    - Directional Accuracy: % of times predicting correct over/under side
    
Key Features:
    - Time-series split (train on past, test on future)
    - Feature leakage prevention (TOV model doesn't see TOV_L5)
    - Early stopping (prevents overfitting)
    - Saves models as JSON (portable, human-readable)
    
Configuration:
    TEST_START_DATE = '2025-02-01' - Games after this are test set
    n_estimators = 1000 - Number of decision trees
    learning_rate = 0.05 - Step size for gradient descent
    max_depth = 5 - Maximum tree depth
    early_stopping_rounds = 50 - Stop if no improvement
    
Usage:
    $ python3 -m src.train
    
Output:
    models/PTS_model.json
    models/REB_model.json
    ... (17 model files total)
"""

import pandas as pd
import xgboost as xgb
import os
import joblib
import csv
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
DATA_FILE = 'data/training_dataset.csv'
MODEL_DIR = 'models'
TEST_START_DATE = '2025-02-01'  # We test on games after this date

# 1. DEFINE TARGETS
# These are the columns the model will try to predict.
TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'BLK', 'STL', 'TOV',  # Base Stats
    'PRA', 'PR', 'PA', 'RA', 'SB',       # Combo Stats (SB = Steals+Blocks)
    'FGM','FGA', 'FTM', 'FTA'                            # Efficiency Stats
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
    'FGM_L5', 'FGM_L20', 'FGM_Season',
    'FTM_L5', 'FTM_L20', 'FTM_Season',
    'MIN_L5', 'MIN_L20', 'MIN_Season',
    'GAME_SCORE_L5', 'GAME_SCORE_L20', 'GAME_SCORE_Season',
    'USAGE_RATE_L5', 'USAGE_RATE_L20', 'USAGE_RATE_Season',
    'MISSING_USAGE',
    'TS_PCT', 'DAYS_REST', 'IS_HOME',
    # FIX #5: Add missing engineered features
    'GAMES_7D', 'IS_4_IN_6', 'IS_B2B', 'IS_FRESH',
    'PACE_ROLLING', 'FGA_PER_MIN', 'TOV_PER_USAGE',
    'USAGE_VACUUM', 'STAR_COUNT'
]

# Add combo rolling features if they exist
combo_features = []
for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    combo_features.extend([f'{combo}_L5', f'{combo}_L20', f'{combo}_Season'])
FEATURES.extend(combo_features)

# Add Defense Columns dynamically
# (We assume features.py created columns like 'OPP_PTS_ALLOWED')
for stat in ['PTS', 'REB', 'AST', 'FG3M','FGA', 'BLK', 'STL', 'TOV',
             'FGM', 'FTM', 'FTA']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')

# Add combo defensive features
for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.append(f'OPP_{combo}_ALLOWED')

def ensure_combo_stats(df):
    """
    Create combo stat columns if they don't exist.
    
    Combo Stats:
        PRA = Points + Rebounds + Assists
        PR  = Points + Rebounds
        PA  = Points + Assists
        RA  = Rebounds + Assists
        SB  = Steals + Blocks
        
    Args:
        df (pandas.DataFrame): Dataset with PTS, REB, AST, STL, BLK columns
        
    Returns:
        pandas.DataFrame: Input df with combo columns added
        
    Note:
        Uses .copy() to prevent DataFrame fragmentation warning
        Only creates columns that don't already exist (safe to call 
            multiple times)
    """
    
    # Create a copy to de-fragment the frame immediately
    df = df.copy() 
    
    # Use a dictionary to store new columns for a single concatenation if needed, 
    # but for just 5 columns, .copy() is usually enough to stop the warning.
    if 'PRA' not in df.columns: df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    if 'PR' not in df.columns: df['PR'] = df['PTS'] + df['REB']
    if 'PA' not in df.columns: df['PA'] = df['PTS'] + df['AST']
    if 'RA' not in df.columns: df['RA'] = df['REB'] + df['AST']
    if 'SB' not in df.columns: df['SB'] = df['STL'] + df['BLK']
    
    return df

def train_and_evaluate():
    """
    Main training loop - trains all 17 models and evaluates performance.
    
    Workflow:
        1. Load data/training_dataset.csv
        2. Calculate combo stats (safety check)
        3. Split into train (before TEST_START_DATE) and test (after)
        4. For each target stat:
            a. Filter features to prevent leakage (e.g., TOV model doesn't see TOV_L5)
            b. Train XGBoost model with early stopping
            c. Evaluate on test set (MAE, R², Directional Accuracy)
            d. Save model to models/{TARGET}_model.json
            
    Output:
        Prints to console:
            Training Model for: PTS...
             -> MAE: 1.99 (On average, off by 1.99 PTS)
             -> R2 Score: 0.886 (Predictive Power)
             -> Directional Accuracy: 89.7% (Predicting Right Side)
             -> Saved to models/PTS_model.json
             
    Raises:
        FileNotFoundError: If data/training_dataset.csv doesn't exist
        
    Note:
        Models with <10 features after filtering print a warning
        (indicates possible configuration error)
    """

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

    all_metrics = []

    # 4. Train Loop
    for target in TARGETS:
        print(f"\nTraining Model for: {target}...")
        
        # Check if target exists (e.g., if raw logs didn't have FTA)
        if target not in df.columns:
            print(f" -> SKIPPING {target} (Column not found in data)")
            continue

        # FIX #15: Filter out features that contain the target stat to prevent leakage
        # Example: TOV model shouldn't see TOV_L5, TOV_Season, OPP_TOV_ALLOWED, TOV_PER_USAGE
        features_to_use = [f for f in FEATURES if target not in f]
        
        # Verify we still have features left after filtering
        if len(features_to_use) < 10:
            print(f" -> WARNING: Only {len(features_to_use)} features after filtering for {target}")
        
        X_train = train_df[features_to_use]
        y_train = train_df[target]
        X_test = test_df[features_to_use]
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
        
        # FIX #11: Add directional accuracy (how often we predict the right side)
        # Use the median of the test set as the "line"
        test_median = y_test.median()
        actual_over = (y_test > test_median).astype(int)
        predicted_over = (predictions > test_median).astype(int)
        directional_accuracy = (actual_over == predicted_over).mean()

        all_metrics.append({
            'Target': target,
            'MAE': round(mae, 4),
            'R2': round(r2, 4),
            'Directional_Accuracy': round(directional_accuracy * 100, 2),
            'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        print(f" -> MAE: {mae:.2f} (On average, off by {mae:.2f} {target})")
        print(f" -> R2 Score: {r2:.3f} (Predictive Power)")
        print(f" -> Directional Accuracy: {directional_accuracy:.1%} (Predicting Right Side)")
        
        # Save
        model_path = f"{MODEL_DIR}/{target}_model.json"
        model.save_model(model_path)
        print(f" -> Saved to {model_path}")

    # Save all metrics to a CSV for tracking over time
    metrics_file = os.path.join(MODEL_DIR, 'model_metrics.csv')
    keys = all_metrics[0].keys()
    with open(metrics_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_metrics)
    print(f"\n✅ Performance metrics saved to {metrics_file}")

    print("\n--- ALL MODELS TRAINED ---")

if __name__ == "__main__":
    train_and_evaluate()