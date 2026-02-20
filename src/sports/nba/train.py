"""
XGBoost Model Training Pipeline

Trains separate regression models for 17 NBA statistics using time-series split
validation. Implements feature leakage prevention.

Output:
    models/nba/{TARGET}_model.json
    models/nba/model_metrics.csv

Usage:
    $ python3 -m src.sports.nba.train
"""

import pandas as pd
import xgboost as xgb
import os
import csv
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FILE   = os.path.join(BASE_DIR, 'data',   'nba', 'processed', 'training_dataset.csv')
MODEL_DIR   = os.path.join(BASE_DIR, 'models', 'nba')
# TEST_START_DATE removed in favor of dynamic split

TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'BLK', 'STL', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'SB',
    'FGM', 'FGA', 'FTM', 'FTA'
]

FEATURES = [
    'PTS_L5', 'PTS_L10', 'PTS_L20', 'PTS_Season', 'PTS_L5_Median', 'PTS_L10_Median',
    'REB_L5', 'REB_L10', 'REB_L20', 'REB_Season', 'REB_L5_Median', 'REB_L10_Median',
    'AST_L5', 'AST_L10', 'AST_L20', 'AST_Season', 'AST_L5_Median', 'AST_L10_Median',
    'FG3M_L5', 'FG3M_L10', 'FG3M_L20', 'FG3M_Season', 'FG3M_L5_Median', 'FG3M_L10_Median',
    'FG3A_L5', 'FG3A_L10', 'FG3A_L20', 'FG3A_Season', 'FG3A_L5_Median', 'FG3A_L10_Median',
    'STL_L5', 'STL_L10', 'STL_L20', 'STL_Season', 'STL_L5_Median', 'STL_L10_Median',
    'BLK_L5', 'BLK_L10', 'BLK_L20', 'BLK_Season', 'BLK_L5_Median', 'BLK_L10_Median',
    'TOV_L5', 'TOV_L10', 'TOV_L20', 'TOV_Season', 'TOV_L5_Median', 'TOV_L10_Median',
    'FGM_L5', 'FGM_L10', 'FGM_L20', 'FGM_Season', 'FGM_L5_Median', 'FGM_L10_Median',
    'FGA_L5', 'FGA_L10', 'FGA_L20', 'FGA_Season', 'FGA_L5_Median', 'FGA_L10_Median',
    'FTM_L5', 'FTM_L10', 'FTM_L20', 'FTM_Season', 'FTM_L5_Median', 'FTM_L10_Median',
    'FTA_L5', 'FTA_L10', 'FTA_L20', 'FTA_Season', 'FTA_L5_Median', 'FTA_L10_Median',
    'MIN_L5', 'MIN_L10', 'MIN_L20', 'MIN_Season', 'MIN_L5_Median', 'MIN_L10_Median',
    'GAME_SCORE_L5', 'GAME_SCORE_L10', 'GAME_SCORE_L20', 'GAME_SCORE_Season', 'GAME_SCORE_L5_Median', 'GAME_SCORE_L10_Median',
    'USAGE_RATE_L5', 'USAGE_RATE_L10', 'USAGE_RATE_L20', 'USAGE_RATE_Season', 'USAGE_RATE_L5_Median', 'USAGE_RATE_L10_Median',
    'MISSING_USAGE',
    'DAYS_REST', 'IS_HOME',
    'GAMES_7D', 'IS_4_IN_6', 'IS_B2B', 'IS_FRESH',
    'PACE_ROLLING',
    'USAGE_VACUUM', 'STAR_COUNT',
    # NEW FEATURES
    'PTS_LOC_MEAN', 'REB_LOC_MEAN', 'AST_LOC_MEAN', 'FG3M_LOC_MEAN', 'PRA_LOC_MEAN',
    'OPP_WIN_PCT', 'IS_VS_ELITE_TEAM'
]

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.extend([f'{combo}_L5', f'{combo}_L10', f'{combo}_L20', f'{combo}_Season', f'{combo}_L5_Median', f'{combo}_L10_Median'])

for stat in ['PTS', 'REB', 'AST', 'FG3M', 'FGA', 'BLK', 'STL', 'TOV', 'FGM', 'FTM', 'FTA']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')
    FEATURES.append(f'OPP_{stat}_ALLOWED_DIFF')  # New DvP Diff

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.append(f'OPP_{combo}_ALLOWED')
    FEATURES.append(f'OPP_{combo}_ALLOWED_DIFF')  # New DvP Diff


def ensure_combo_stats(df):
    df = df.copy()
    if 'PRA' not in df.columns: df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    if 'PR'  not in df.columns: df['PR']  = df['PTS'] + df['REB']
    if 'PA'  not in df.columns: df['PA']  = df['PTS'] + df['AST']
    if 'RA'  not in df.columns: df['RA']  = df['REB'] + df['AST']
    if 'SB'  not in df.columns: df['SB']  = df['STL'] + df['BLK']
    return df


def train_and_evaluate():
    print("--- STARTING TRAINING PIPELINE ---")

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Training data not found at {DATA_FILE}. Run features.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = ensure_combo_stats(df)
    
    # Sort by date for time-series split
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    
    # Dynamic 70/30 Split
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
    
    # Print date ranges for verification
    print(f"Train Date Range: {train_df['GAME_DATE'].min().date()} -> {train_df['GAME_DATE'].max().date()}")
    print(f"Test Date Range:  {test_df['GAME_DATE'].min().date()} -> {test_df['GAME_DATE'].max().date()}")

    print(f"Training Set: {len(train_df)} games")
    print(f"Testing Set:  {len(test_df)} games")

    os.makedirs(MODEL_DIR, exist_ok=True)

    all_metrics = []

    for target in TARGETS:
        print(f"\nTraining Model for: {target}...")

        if target not in df.columns:
            print(f" -> SKIPPING {target} (Column not found in data)")
            continue

        features_to_use = FEATURES

        X_train = train_df[features_to_use]
        y_train = train_df[target]
        X_test  = test_df[features_to_use]
        y_test  = test_df[target]

        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            early_stopping_rounds=50,
            n_jobs=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2  = r2_score(y_test, predictions)

        test_median       = y_test.median()
        actual_over       = (y_test > test_median).astype(int)
        predicted_over    = (predictions > test_median).astype(int)
        directional_accuracy = (actual_over == predicted_over).mean()

        all_metrics.append({
            'Target': target,
            'MAE': round(mae, 4),
            'R2': round(r2, 4),
            'Directional_Accuracy': round(directional_accuracy * 100, 2),
            'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        print(f" -> MAE: {mae:.2f}")
        print(f" -> R2 Score: {r2:.3f}")
        print(f" -> Directional Accuracy: {directional_accuracy:.1%}")

        model_path = os.path.join(MODEL_DIR, f"{target}_model.json")
        model.save_model(model_path)
        print(f" -> Saved to {model_path}")

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
