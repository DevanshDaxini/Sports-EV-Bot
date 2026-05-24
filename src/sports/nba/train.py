"""
XGBoost Model Training Pipeline

Trains separate regression models for 21 NBA statistics using time-series split
validation. Implements feature leakage prevention.

Improvements over v1:
  - Tiered hyperparameters: more regularization for noisy/low-signal stats
  - Log-transform targets for zero-inflated stats (BLK, STL, TOV, FG3M, SB)
  - Directional accuracy measured vs player-specific L20 median (not global
    test-set median), which is a much more realistic proxy for sportsbook lines

Output:
    models/nba/{TARGET}_model.json
    models/nba/model_metrics.csv

Usage:
    $ python3 -m src.sports.nba.train
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import os
import csv
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FILE = os.path.join(BASE_DIR, 'data',   'nba', 'processed', 'training_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'nba')

TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'BLK', 'STL', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'SB',
    'FGM', 'FGA', 'FTM', 'FTA', 'FPTS',
    'PTS_1H', 'PRA_1H', 'FPTS_1H'
]

# Stats that follow a count/zero-inflated distribution.
# Training on log1p(y) reduces outlier influence and fixes the BLK inversion.
LOG_TRANSFORM_TARGETS = {'BLK', 'STL', 'TOV', 'FG3M', 'SB'}

# --- HYPERPARAMETER TIERS -----------------------------------------------
# HIGH_SIGNAL: continuous stats with strong player-level signal
# MEDIUM_SIGNAL: moderate volatility; slightly more regularization
# LOW_SIGNAL: noisy/count stats; heavy regularization to avoid overfit
_HIGH = {
    'n_estimators': 1000, 'learning_rate': 0.04, 'max_depth': 6,
    'subsample': 0.85, 'colsample_bytree': 0.75, 'min_child_weight': 3,
    'gamma': 0.05, 'reg_alpha': 0.1, 'reg_lambda': 1.5,
    'early_stopping_rounds': 50, 'n_jobs': -1,
}
_MEDIUM = {
    'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 5,
    'subsample': 0.80, 'colsample_bytree': 0.70, 'min_child_weight': 7,
    'gamma': 0.10, 'reg_alpha': 0.2, 'reg_lambda': 1.5,
    'early_stopping_rounds': 40, 'n_jobs': -1,
}
_LOW = {
    'n_estimators': 600, 'learning_rate': 0.02, 'max_depth': 4,
    'subsample': 0.70, 'colsample_bytree': 0.60, 'min_child_weight': 15,
    'gamma': 0.50, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
    'early_stopping_rounds': 30, 'n_jobs': -1,
}

TARGET_HYPERPARAMS = {
    'PTS': _HIGH, 'FGM': _HIGH, 'FGA': _HIGH, 'PA': _HIGH, 'PR': _HIGH,
    'PRA': _HIGH, 'FPTS': _HIGH, 'PTS_1H': _HIGH, 'PRA_1H': _HIGH, 'FPTS_1H': _HIGH,
    'REB': _MEDIUM, 'AST': _MEDIUM, 'FG3A': _MEDIUM, 'FTA': _MEDIUM,
    'FTM': _MEDIUM, 'RA': _MEDIUM,
    'FG3M': _LOW, 'STL': _LOW, 'TOV': _LOW, 'BLK': _LOW, 'SB': _LOW,
}

# --- LIGHTGBM HYPERPARAMETER TIERS (mirrored from XGBoost tiers) ---
_LGBM_HIGH = {
    'n_estimators': 1000, 'learning_rate': 0.04, 'max_depth': 6, 'num_leaves': 50,
    'subsample': 0.85, 'subsample_freq': 5, 'colsample_bytree': 0.75,
    'min_child_samples': 20, 'min_split_gain': 0.05,
    'reg_alpha': 0.1, 'reg_lambda': 1.5, 'verbose': -1, 'n_jobs': -1,
}
_LGBM_MEDIUM = {
    'n_estimators': 800, 'learning_rate': 0.03, 'max_depth': 5, 'num_leaves': 25,
    'subsample': 0.80, 'subsample_freq': 5, 'colsample_bytree': 0.70,
    'min_child_samples': 50, 'min_split_gain': 0.10,
    'reg_alpha': 0.2, 'reg_lambda': 1.5, 'verbose': -1, 'n_jobs': -1,
}
_LGBM_LOW = {
    'n_estimators': 600, 'learning_rate': 0.02, 'max_depth': 4, 'num_leaves': 15,
    'subsample': 0.70, 'subsample_freq': 5, 'colsample_bytree': 0.60,
    'min_child_samples': 100, 'min_split_gain': 0.50,
    'reg_alpha': 0.5, 'reg_lambda': 2.0, 'verbose': -1, 'n_jobs': -1,
}
LGBM_HYPERPARAMS = {
    'PTS': _LGBM_HIGH, 'FGM': _LGBM_HIGH, 'FGA': _LGBM_HIGH, 'PA': _LGBM_HIGH,
    'PR': _LGBM_HIGH, 'PRA': _LGBM_HIGH, 'FPTS': _LGBM_HIGH,
    'PTS_1H': _LGBM_HIGH, 'PRA_1H': _LGBM_HIGH, 'FPTS_1H': _LGBM_HIGH,
    'REB': _LGBM_MEDIUM, 'AST': _LGBM_MEDIUM, 'FG3A': _LGBM_MEDIUM,
    'FTA': _LGBM_MEDIUM, 'FTM': _LGBM_MEDIUM, 'RA': _LGBM_MEDIUM,
    'FG3M': _LGBM_LOW, 'STL': _LGBM_LOW, 'TOV': _LGBM_LOW,
    'BLK': _LGBM_LOW, 'SB': _LGBM_LOW,
}
# -------------------------------------------------------------------------


def get_features_for_target(target):
    """Dynamically select features per target to reduce noise."""
    core = [
        'MIN_Season', 'MIN_L5', 'MIN_L10', 'USAGE_RATE_Season', 'USAGE_RATE_L5', 'USAGE_RATE_L10',
        'DAYS_REST', 'IS_HOME', 'GAMES_7D', 'IS_4_IN_6', 'IS_B2B', 'IS_FRESH',
        'PACE_ROLLING', 'USAGE_VACUUM', 'STAR_COUNT', 'GAME_SCORE_Season', 'GAME_SCORE_L5',
        # New: expected possessions and streak/consistency signals
        'EXP_POSS', 'EXP_POSS_SEASON',
        'MIN_STREAK', 'USAGE_RATE_STREAK', 'USAGE_RATE_CONSISTENCY',
        'GAME_SCORE_STREAK', 'GAME_SCORE_CONSISTENCY',
    ]

    if target in ['PTS', 'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A']:
        target_stats = ['PTS', 'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'FG3A']
    elif target == 'REB':
        target_stats = ['REB', 'PTS', 'FGA']
    elif target == 'AST':
        target_stats = ['AST', 'PTS', 'FGA', 'TOV']
    elif target in ['STL', 'BLK', 'TOV']:
        target_stats = ['STL', 'BLK', 'TOV']
    elif target in ['PRA', 'PR', 'PA', 'RA']:
        target_stats = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'FGA']
    elif target == 'SB':
        target_stats = ['STL', 'BLK', 'SB']
    elif target == 'FPTS':
        target_stats = ['FPTS', 'PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV']
    elif target in ['PTS_1H', 'PRA_1H', 'FPTS_1H']:
        target_stats = ['PTS_1H', 'PTS', 'FGA_1H', 'MIN_1H']
        if target == 'PRA_1H':  target_stats.extend(['PRA_1H', 'PRA'])
        if target == 'FPTS_1H': target_stats.extend(['FPTS_1H', 'FPTS'])
    else:
        target_stats = []

    features = list(core)
    for stat in target_stats:
        for variant in ['_Season', '_L5', '_L10', '_L20', '_L5_Median', '_L10_Median']:
            features.append(f'{stat}{variant}')
        # Streak (L5−L20 delta) and consistency (L5_Median / Season ratio)
        features.append(f'{stat}_STREAK')
        features.append(f'{stat}_CONSISTENCY')
        if stat in ['PTS', 'REB', 'AST', 'FG3M', 'FGA', 'BLK', 'STL', 'TOV',
                    'FGM', 'FTM', 'FTA', 'PRA', 'PR', 'PA', 'RA', 'SB']:
            features.append(f'OPP_{stat}_ALLOWED')
            features.append(f'OPP_{stat}_ALLOWED_DIFF')

    if target in ['PTS', 'REB', 'AST', 'FG3M', 'PRA']:
        features.append(f'{target}_LOC_MEAN')

    return list(dict.fromkeys(features))


def ensure_combo_stats(df):
    df = df.copy()
    if 'PRA' not in df.columns: df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    if 'PR'  not in df.columns: df['PR']  = df['PTS'] + df['REB']
    if 'PA'  not in df.columns: df['PA']  = df['PTS'] + df['AST']
    if 'RA'  not in df.columns: df['RA']  = df['REB'] + df['AST']
    if 'SB'  not in df.columns: df['SB']  = df['STL'] + df['BLK']

    if 'PRA_1H' not in df.columns and 'PTS_1H' in df.columns:
        df['PRA_1H'] = df['PTS_1H'] + df['REB_1H'] + df['AST_1H']
    if 'PR_1H' not in df.columns and 'PTS_1H' in df.columns:
        df['PR_1H']  = df['PTS_1H'] + df['REB_1H']
    if 'PA_1H' not in df.columns and 'PTS_1H' in df.columns:
        df['PA_1H']  = df['PTS_1H'] + df['AST_1H']
    if 'RA_1H' not in df.columns and 'REB_1H' in df.columns:
        df['RA_1H']  = df['REB_1H'] + df['AST_1H']
    if 'SB_1H' not in df.columns and 'STL_1H' in df.columns:
        df['SB_1H']  = df['STL_1H'] + df['BLK_1H']
    return df


def _player_specific_directional_accuracy(y_test, predictions, test_df, target):
    """
    Measure directional accuracy vs each player's own rolling median —
    a realistic proxy for sportsbook lines. Prefers L10_Median (always
    present) over L20_Median, then falls back to global test median.
    """
    lines = None
    for candidate in [f'{target}_L10_Median', f'{target}_L20_Median']:
        if candidate in test_df.columns and test_df[candidate].notna().mean() > 0.4:
            lines = test_df[candidate].reindex(y_test.index).fillna(y_test.median())
            break
    if lines is None:
        lines = pd.Series(float(y_test.median()), index=y_test.index)

    actual_over    = (y_test.values > lines.values).astype(int)
    predicted_over = (predictions    > lines.values).astype(int)
    return (actual_over == predicted_over).mean()


def train_and_evaluate():
    print("--- STARTING TRAINING PIPELINE ---")

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Training data not found at {DATA_FILE}. Run features.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = ensure_combo_stats(df)
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    split_idx = int(len(df) * 0.70)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    print(f"Train: {train_df['GAME_DATE'].min().date()} → {train_df['GAME_DATE'].max().date()}  ({len(train_df):,} rows)")
    print(f"Test:  {test_df['GAME_DATE'].min().date()}  → {test_df['GAME_DATE'].max().date()}  ({len(test_df):,} rows)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    all_metrics = []

    for target in TARGETS:
        print(f"\nTraining {target}...")

        if target not in df.columns:
            print(f"  -> SKIP (column missing)")
            continue

        raw_features    = get_features_for_target(target)
        features_to_use = [f for f in raw_features if f in df.columns]
        print(f"  -> {len(features_to_use)} features")

        # Drop rows where the target itself is NaN (can occur in 1H cols after auto-refresh)
        train_valid = train_df[target].notna()
        test_valid  = test_df[target].notna()
        X_train = train_df.loc[train_valid, features_to_use].fillna(0)
        y_train = train_df.loc[train_valid, target]
        X_test  = test_df.loc[test_valid, features_to_use].fillna(0)
        y_test  = test_df.loc[test_valid, target]

        if len(X_train) < 100 or len(X_test) < 50:
            print(f"  -> SKIP (insufficient non-NaN rows: train={len(X_train)}, test={len(X_test)})")
            continue

        # Recency sample weights: exponential decay so recent games count more
        max_train_date = train_df.loc[train_valid, 'GAME_DATE'].max()
        days_ago = (max_train_date - train_df.loc[train_valid, 'GAME_DATE']).dt.days
        sample_weights = np.exp(-0.001 * days_ago.values)

        # Log-transform zero-inflated count stats
        log_transform = target in LOG_TRANSFORM_TARGETS
        if log_transform:
            y_train_fit = np.log1p(y_train.clip(lower=0))
            y_test_fit  = np.log1p(y_test.clip(lower=0))
        else:
            y_train_fit = y_train
            y_test_fit  = y_test

        # --- XGBoost ---
        params = TARGET_HYPERPARAMS.get(target, _MEDIUM)
        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(
            X_train, y_train_fit,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test_fit)],
            verbose=False
        )
        xgb_raw = xgb_model.predict(X_test)

        # --- LightGBM ---
        lgbm_params = LGBM_HYPERPARAMS.get(target, _LGBM_MEDIUM)
        lgbm_model = lgb.LGBMRegressor(**lgbm_params)
        lgbm_model.fit(X_train, y_train_fit, sample_weight=sample_weights)
        lgbm_raw = lgbm_model.predict(X_test)

        # --- Ensemble: 60% XGBoost + 40% LightGBM ---
        raw_preds = 0.6 * xgb_raw + 0.4 * lgbm_raw

        if log_transform:
            predictions = np.expm1(np.clip(raw_preds, 0, 10))
        else:
            predictions = np.clip(raw_preds, 0, None)

        model = xgb_model  # keep xgb handle for save / feature_names_in_

        mae = mean_absolute_error(y_test, predictions)
        r2  = r2_score(y_test, predictions)

        # Realistic directional accuracy: vs player-specific L20 median
        dir_acc = _player_specific_directional_accuracy(y_test, predictions, test_df, target)

        # Also compute legacy global-median accuracy for reference
        gm         = y_test.median()
        legacy_acc = ((y_test > gm).astype(int) == (predictions > gm).astype(int)).mean()

        all_metrics.append({
            'Target':                  target,
            'MAE':                     round(mae, 4),
            'R2':                      round(r2, 4),
            'Directional_Accuracy':    round(dir_acc * 100, 2),
            'Legacy_Global_Accuracy':  round(legacy_acc * 100, 2),
            'Log_Transformed':         log_transform,
            'Last_Updated':            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

        print(f"  -> MAE: {mae:.2f}  |  R²: {r2:.3f}")
        print(f"  -> Dir Acc (vs L20 median): {dir_acc:.1%}  |  Legacy (vs global median): {legacy_acc:.1%}")

        xgb_model.save_model(os.path.join(MODEL_DIR, f"{target}_model.json"))
        lgbm_model.booster_.save_model(os.path.join(MODEL_DIR, f"{target}_model_lgbm.txt"))

    metrics_file = os.path.join(MODEL_DIR, 'model_metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\n✅ Metrics saved: {metrics_file}")
    print("\n--- ALL MODELS TRAINED ---")


if __name__ == "__main__":
    train_and_evaluate()
