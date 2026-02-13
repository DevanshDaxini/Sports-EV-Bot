"""
XGBoost Hyperparameter Tuning Pipeline

Uses RandomizedSearchCV with TimeSeriesSplit to find optimal model parameters.

Output:
    models/nba/{TARGET}_model.json  (overwrites with optimized versions)

Usage:
    $ python3 -m src.sports.nba.tune_train
"""

import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIGURATION ---
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FILE = os.path.join(BASE_DIR, 'data',   'nba', 'processed', 'training_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'nba')
TEST_START_DATE = '2025-02-01'

TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'SB',
    'FGM', 'FTM', 'FTA'
]

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
    'GAMES_7D', 'IS_4_IN_6', 'IS_B2B', 'IS_FRESH',
    'PACE_ROLLING', 'FGA_PER_MIN', 'TOV_PER_USAGE',
    'USAGE_VACUUM', 'STAR_COUNT'
]

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.extend([f'{combo}_L5', f'{combo}_L20', f'{combo}_Season'])

for stat in ['PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV']:
    FEATURES.append(f'OPP_{stat}_ALLOWED')

for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
    FEATURES.append(f'OPP_{combo}_ALLOWED')


def ensure_combo_stats(df):
    if 'PRA' not in df.columns: df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    if 'PR'  not in df.columns: df['PR']  = df['PTS'] + df['REB']
    if 'PA'  not in df.columns: df['PA']  = df['PTS'] + df['AST']
    if 'RA'  not in df.columns: df['RA']  = df['REB'] + df['AST']
    if 'SB'  not in df.columns: df['SB']  = df['STL'] + df['BLK']
    return df


def tune_and_train():
    print("--- STARTING HYPERPARAMETER TUNING ---")

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data not found at {DATA_FILE}.")
        return

    df = pd.read_csv(DATA_FILE)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = ensure_combo_stats(df)

    train_df = df[df['GAME_DATE'] <  TEST_START_DATE]
    test_df  = df[df['GAME_DATE'] >= TEST_START_DATE]

    os.makedirs(MODEL_DIR, exist_ok=True)

    param_grid = {
        'n_estimators':    [500, 1000, 1500],
        'learning_rate':   [0.01, 0.03, 0.05, 0.1],
        'max_depth':       [3, 4, 5, 6],
        'subsample':       [0.7, 0.8, 0.9],
        'colsample_bytree':[0.7, 0.8, 0.9],
        'gamma':           [0, 0.1, 0.2]
    }

    for target in TARGETS:
        print(f"\nOptimization for: {target}...")

        if target not in df.columns:
            continue

        features_to_use = [f for f in FEATURES if target not in f]

        X_train = train_df[features_to_use]
        y_train = train_df[target]
        X_test  = test_df[features_to_use]
        y_test  = test_df[target]

        xgb_model    = xgb.XGBRegressor(n_jobs=-1, random_state=42)
        tscv         = TimeSeriesSplit(n_splits=3)
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=15,
            scoring='neg_mean_absolute_error',
            cv=tscv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        print(f" -> Best Params: {random_search.best_params_}")

        predictions = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2  = r2_score(y_test, predictions)
        print(f" -> OPTIMIZED MAE: {mae:.2f}")
        print(f" -> OPTIMIZED R2:  {r2:.3f}")

        model_path = os.path.join(MODEL_DIR, f"{target}_model.json")
        best_model.save_model(model_path)
        print(f" -> Saved to {model_path}")


if __name__ == "__main__":
    tune_and_train()
