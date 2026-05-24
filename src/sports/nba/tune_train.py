"""
Optuna Hyperparameter Tuning Pipeline

Uses Bayesian optimization (Optuna) with TimeSeriesSplit and recency sample
weights to find optimal XGBoost parameters per target.

Output:
    models/nba/{TARGET}_model.json  (overwrites with optimized versions)

Usage:
    $ python3 -m src.sports.nba.tune_train
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from src.sports.nba.train import (
    get_features_for_target, ensure_combo_stats,
    LOG_TRANSFORM_TARGETS, LGBM_HYPERPARAMS
)
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_FILE = os.path.join(BASE_DIR, 'data',   'nba', 'processed', 'training_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'nba')

N_TRIALS  = 30   # Optuna trials per target
N_CV_SPLITS = 3  # TimeSeriesSplit folds

TARGETS = [
    'PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'SB',
    'FGM', 'FTM', 'FTA', 'FPTS',
    'PTS_1H', 'PRA_1H', 'FPTS_1H',
]


def _make_objective(X, y, sample_weights, tscv, log_transform):
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'max_depth':         trial.suggest_int('max_depth', 3, 7),
            'subsample':         trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'min_child_weight':  trial.suggest_int('min_child_weight', 1, 20),
            'gamma':             trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 3.0),
            'n_jobs': -1,
        }
        cv_maes = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            sw_tr = sample_weights[train_idx]
            y_tr_fit = np.log1p(y_tr.clip(lower=0)) if log_transform else y_tr
            y_val_fit = np.log1p(y_val.clip(lower=0)) if log_transform else y_val
            m = xgb.XGBRegressor(**params)
            m.fit(X_tr, y_tr_fit, sample_weight=sw_tr,
                  eval_set=[(X_val, y_val_fit)], verbose=False)
            raw = m.predict(X_val)
            preds = np.expm1(np.clip(raw, 0, 10)) if log_transform else np.clip(raw, 0, None)
            cv_maes.append(mean_absolute_error(y_val, preds))
        return float(np.mean(cv_maes))
    return objective


def tune_and_train():
    print("--- STARTING OPTUNA HYPERPARAMETER TUNING ---")

    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data not found at {DATA_FILE}.")
        return

    df = pd.read_csv(DATA_FILE)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = ensure_combo_stats(df)
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    split_idx = int(len(df) * 0.70)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]
    os.makedirs(MODEL_DIR, exist_ok=True)

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    for target in TARGETS:
        print(f"\nTuning {target}  ({N_TRIALS} trials)...")

        if target not in df.columns:
            print(f"  -> SKIP (column missing)")
            continue

        all_features    = get_features_for_target(target)
        features_to_use = [f for f in all_features if f in df.columns]

        train_valid = train_df[target].notna()
        test_valid  = test_df[target].notna()
        X_train = train_df.loc[train_valid, features_to_use].fillna(0)
        y_train = train_df.loc[train_valid, target]
        X_test  = test_df.loc[test_valid, features_to_use].fillna(0)
        y_test  = test_df.loc[test_valid, target]

        if len(X_train) < 100 or len(X_test) < 50:
            print(f"  -> SKIP (insufficient rows)")
            continue

        # Recency sample weights
        max_date   = train_df.loc[train_valid, 'GAME_DATE'].max()
        days_ago   = (max_date - train_df.loc[train_valid, 'GAME_DATE']).dt.days
        sw         = np.exp(-0.001 * days_ago.values)

        log_transform = target in LOG_TRANSFORM_TARGETS

        study = optuna.create_study(direction='minimize')
        study.optimize(_make_objective(X_train, y_train, sw, tscv, log_transform),
                       n_trials=N_TRIALS, show_progress_bar=False)

        best_params = {**study.best_params, 'n_jobs': -1}
        print(f"  -> Best CV MAE: {study.best_value:.3f}  |  params: {best_params}")

        # Retrain on full train set with best params
        y_train_fit = np.log1p(y_train.clip(lower=0)) if log_transform else y_train
        y_test_fit  = np.log1p(y_test.clip(lower=0))  if log_transform else y_test

        xgb_model = xgb.XGBRegressor(**best_params)
        xgb_model.fit(X_train, y_train_fit, sample_weight=sw,
                      eval_set=[(X_test, y_test_fit)], verbose=False)

        lgbm_params = LGBM_HYPERPARAMS.get(target, {})
        lgbm_model  = lgb.LGBMRegressor(**lgbm_params)
        lgbm_model.fit(X_train, y_train_fit, sample_weight=sw)

        xgb_raw  = xgb_model.predict(X_test)
        lgbm_raw = lgbm_model.predict(X_test)
        raw_ens  = 0.6 * xgb_raw + 0.4 * lgbm_raw

        if log_transform:
            predictions = np.expm1(np.clip(raw_ens, 0, 10))
        else:
            predictions = np.clip(raw_ens, 0, None)

        mae = mean_absolute_error(y_test, predictions)
        r2  = r2_score(y_test, predictions)
        print(f"  -> Test MAE: {mae:.2f}  |  R²: {r2:.3f}")

        xgb_model.save_model(os.path.join(MODEL_DIR, f"{target}_model.json"))
        lgbm_model.booster_.save_model(os.path.join(MODEL_DIR, f"{target}_model_lgbm.txt"))
        print(f"  -> Saved {target} XGB + LGBM models")

    print("\n--- TUNING COMPLETE ---")


if __name__ == "__main__":
    tune_and_train()
