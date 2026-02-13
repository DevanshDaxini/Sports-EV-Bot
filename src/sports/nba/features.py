"""
NBA Statistical Feature Engineering Pipeline - PRODUCTION VERSION v2.2

Transforms raw game logs into a rich feature set for machine learning models.
Creates 220+ predictive features including rolling averages, defensive matchups,
fatigue indicators, pace adjustments, team context, momentum signals, AND
specialized features for weak models (BLK, STL, TOV, REB, AST).

Output:
    data/nba/processed/training_dataset.csv - Ready for XGBoost training
    
Usage:
    $ python3 -m src.sports.nba.features
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
# Resolve project root: src/sports/nba/features.py -> src/sports/nba -> src/sports -> src -> root
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOGS_FILE   = os.path.join(BASE_DIR, 'data', 'nba', 'raw', 'raw_game_logs.csv')
POS_FILE    = os.path.join(BASE_DIR, 'data', 'nba', 'processed', 'player_positions.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'nba', 'processed', 'training_dataset.csv')

TARGET_STATS = ['PTS', 'REB', 'AST', 'FG3M', 'FG3A', 'STL',
                'BLK', 'TOV', 'FGM', 'FGA', 'FTM', 'FTA']


def load_and_merge_data():
    print("...Loading and Merging Data")
    if not os.path.exists(POS_FILE) or not os.path.exists(LOGS_FILE):
        print("Error: Data files not found. Run builder.py first.")
        return None
    df_logs = pd.read_csv(LOGS_FILE)
    df_pos  = pd.read_csv(POS_FILE)
    df = pd.merge(df_logs, df_pos[['PLAYER_ID', 'POSITION']], on='PLAYER_ID', how='left')
    df['POSITION'] = df['POSITION'].fillna('Unknown')
    df = df.dropna(subset=['MATCHUP', 'GAME_DATE'])
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'], ascending=True)
    print(f"   Loaded {len(df):,} game logs for {df['PLAYER_ID'].nunique():,} players")
    return df


def add_advanced_stats(df):
    print("...Calculating Advanced Stats")
    df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    df['TS_PCT'] = df['TS_PCT'].fillna(0).clip(upper=1.0)
    df['USAGE_RATE'] = 100 * ((df['FGA'] + 0.44 * df['FTA'] + df['TOV'])) / (df['MIN'] + 0.1)
    df['USAGE_RATE'] = df['USAGE_RATE'].fillna(0).clip(upper=50)
    df['GAME_SCORE'] = (df['PTS'] + (0.4 * df['FGM']) - (0.7 * df['FGA']) -
                        (0.4 * (df['FTA'] - df['FTM'])) + (0.7 * df['OREB']) +
                        (0.3 * df['DREB']) + df['STL'] + (0.7 * df['AST']) +
                        (0.7 * df['BLK']) - (0.4 * df['PF']) - df['TOV'])
    df['GAME_SCORE'] = df['GAME_SCORE'].fillna(0)
    return df


def add_rolling_features(df):
    print("...Calculating Rolling Averages (Rookie-Friendly)")
    df = df.copy()
    df['CAREER_GAMES'] = df.groupby('PLAYER_ID').cumcount() + 1
    grouped = df.groupby('PLAYER_ID')
    base_stats = ['PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV', 'FGM', 'FTM']
    stats_to_roll = base_stats + ['MIN', 'GAME_SCORE', 'USAGE_RATE']
    for combo in ['PRA', 'PR', 'PA', 'RA', 'SB']:
        if combo in df.columns:
            stats_to_roll.append(combo)
    rolling_data = {}
    for stat in stats_to_roll:
        rolling_data[f'{stat}_L5']     = grouped[stat].transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
        rolling_data[f'{stat}_L20']    = grouped[stat].transform(lambda x: x.shift(1).rolling(20, min_periods=10).mean())
        rolling_data[f'{stat}_Season'] = grouped[stat].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    df = pd.concat([df, pd.DataFrame(rolling_data, index=df.index)], axis=1)
    return df


def add_context_features(df):
    print("...Adding Context Features")
    df['IS_HOME']   = df['MATCHUP'].astype(str).apply(lambda x: 1 if 'vs.' in x else 0)
    df['OPPONENT']  = df['MATCHUP'].astype(str).apply(lambda x: x.split(' ')[-1])
    df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3).clip(upper=7)
    df['IS_B2B']    = (df['DAYS_REST'] == 1).astype(int)
    df['IS_FRESH']  = (df['DAYS_REST'] >= 3).astype(int)
    return df


def add_team_performance_context(df):
    print("...Adding Team Performance Context")
    df = df.copy()
    if 'WL' not in df.columns:
        print("   WARNING: No WL column found, skipping team performance features")
        return df
    df['TEAM_WIN'] = (df['WL'] == 'W').astype(int)
    df['TEAM_WIN_PCT'] = df.groupby(['TEAM_ID', 'SEASON_ID'])['TEAM_WIN'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()).fillna(0.5)
    df['TEAM_L5_WIN_PCT'] = df.groupby(['TEAM_ID', 'SEASON_ID'])['TEAM_WIN'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=5).mean()).fillna(df['TEAM_WIN_PCT'])
    df['AVG_POINT_DIFF'] = 0
    return df


def add_defense_vs_position(df):
    print("...Calculating Defense vs. Position (L10 Window)")
    df = df.copy()
    defense_group = df.groupby(['OPPONENT', 'POSITION'])
    new_def_cols = {}
    for stat in TARGET_STATS:
        col_name = f'OPP_{stat}_ALLOWED'
        new_def_cols[col_name] = defense_group[stat].transform(
            lambda x: x.shift(1).rolling(10, min_periods=10).mean())
    df = pd.concat([df, pd.DataFrame(new_def_cols, index=df.index)], axis=1)
    for stat in TARGET_STATS:
        col_name = f'OPP_{stat}_ALLOWED'
        league_pos_avg = df.groupby(['POSITION', 'SEASON_ID'])[stat].transform('median')
        df[col_name] = df[col_name].fillna(league_pos_avg)
    if 'OPP_PTS_ALLOWED' in df.columns and 'OPP_REB_ALLOWED' in df.columns:
        df['OPP_PRA_ALLOWED'] = df['OPP_PTS_ALLOWED'] + df['OPP_REB_ALLOWED'] + df['OPP_AST_ALLOWED']
        df['OPP_PR_ALLOWED']  = df['OPP_PTS_ALLOWED'] + df['OPP_REB_ALLOWED']
        df['OPP_PA_ALLOWED']  = df['OPP_PTS_ALLOWED'] + df['OPP_AST_ALLOWED']
        df['OPP_RA_ALLOWED']  = df['OPP_REB_ALLOWED'] + df['OPP_AST_ALLOWED']
        df['OPP_SB_ALLOWED']  = df['OPP_STL_ALLOWED'] + df['OPP_BLK_ALLOWED']
    return df


def add_usage_vacuum_features(df):
    print("...Calculating Usage Vacuum")
    df = df.copy()
    usage_col = 'USAGE_RATE_Season' if 'USAGE_RATE_Season' in df.columns else 'USAGE_RATE'
    stars = df[df[usage_col] > 28][['PLAYER_ID', 'GAME_ID', 'TEAM_ID']].copy()
    star_games = stars.groupby(['GAME_ID', 'TEAM_ID'])['PLAYER_ID'].count().reset_index()
    star_games.columns = ['GAME_ID', 'TEAM_ID', 'STAR_COUNT']
    df = df.merge(star_games, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['STAR_COUNT'] = df['STAR_COUNT'].fillna(0)
    team_avg_stars = df.groupby('TEAM_ID')['STAR_COUNT'].transform('mean')
    df['USAGE_VACUUM'] = (team_avg_stars - df['STAR_COUNT']).clip(lower=0)
    return df


def add_missing_player_context(df):
    print("...Calculating Missing Player Impact (Injury Simulation)")
    df = df.copy()
    season_stats = df.groupby(['SEASON_ID', 'TEAM_ID', 'PLAYER_ID'])['USAGE_RATE'].mean().reset_index()
    key_players  = season_stats[season_stats['USAGE_RATE'] > 18.0]
    team_games   = df[['SEASON_ID', 'TEAM_ID', 'GAME_ID']].drop_duplicates()
    expected     = team_games.merge(key_players, on=['SEASON_ID', 'TEAM_ID'], how='left')
    actual       = df[['GAME_ID', 'PLAYER_ID']].drop_duplicates()
    actual['PLAYED'] = True
    merged  = expected.merge(actual, on=['GAME_ID', 'PLAYER_ID'], how='left')
    missing = merged[merged['PLAYED'].isna()]
    missing_usage = missing.groupby(['GAME_ID', 'TEAM_ID'])['USAGE_RATE'].sum().reset_index()
    missing_usage.rename(columns={'USAGE_RATE': 'MISSING_USAGE'}, inplace=True)
    df = df.merge(missing_usage, on=['GAME_ID', 'TEAM_ID'], how='left')
    df['MISSING_USAGE'] = df['MISSING_USAGE'].fillna(0)
    return df


def add_schedule_density(df):
    print("...Calculating Schedule Density")
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
    def get_rolling_count(group):
        temp_series = pd.Series(1, index=group['GAME_DATE'])
        return temp_series.rolling('7D').count().values
    games_7d_list = []
    for player_id, group in df.groupby('PLAYER_ID'):
        counts = get_rolling_count(group)
        games_7d_list.extend(counts)
    df['GAMES_7D']  = games_7d_list
    df['GAMES_7D']  = df['GAMES_7D'].astype(float)
    df['IS_4_IN_6'] = (df['GAMES_7D'] >= 4).astype(int)
    return df


def add_pace_features(df):
    print("...Calculating Team Pace (Per-48 Standard)")
    df = df.copy()
    df['POSS_EST']   = df['FGA'] + (0.44 * df['FTA']) - df['OREB'] + df['TOV']
    df['PACE_PER_48'] = (df['POSS_EST'] / (df['MIN'] + 0.1)) * 48
    df['PACE_PER_48'] = df['PACE_PER_48'].clip(lower=0, upper=200)
    team_pace = df.groupby(['TEAM_ID', 'GAME_ID']).agg(
        {'PACE_PER_48': 'mean', 'GAME_DATE': 'first'}).reset_index()
    team_pace = team_pace.sort_values(['TEAM_ID', 'GAME_DATE'])
    team_pace['PACE_ROLLING'] = team_pace.groupby('TEAM_ID')['PACE_PER_48'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=10).mean())
    df = df.merge(team_pace[['GAME_ID', 'TEAM_ID', 'PACE_ROLLING']], on=['GAME_ID', 'TEAM_ID'], how='left')
    df['PACE_ROLLING'] = df['PACE_ROLLING'].fillna(df['PACE_ROLLING'].median())
    df = df.drop(columns=['POSS_EST', 'PACE_PER_48'], errors='ignore')
    return df


def add_efficiency_signals(df):
    print("...Calculating Efficiency Signals")
    df['FGA_PER_MIN'] = df['FGA'] / (df['MIN'] + 0.1)
    if 'TS_PCT_Season' in df.columns:
        df['TS_EFFICIENCY_GAP'] = (df['TS_PCT'] - df['TS_PCT_Season']).fillna(0)
    df['TOV_PER_USAGE'] = df['TOV'] / (df['USAGE_RATE'] + 0.1)
    return df


def add_role_features(df):
    print("...Adding Role Features")
    df = df.copy()
    team_mins = df.groupby(['GAME_ID', 'TEAM_ID'])['MIN'].transform('sum')
    df['MIN_SHARE']         = df['MIN'] / (team_mins + 0.1)
    df['ROLE_CONSISTENCY']  = df.groupby('PLAYER_ID')['MIN_SHARE'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=10).std()).fillna(0)
    df['IS_STARTER'] = df.groupby(['GAME_ID', 'TEAM_ID'])['MIN'].transform(
        lambda x: (x >= x.nlargest(5).min()).astype(int))
    return df


def add_rookie_features(df):
    print("...Adding Rookie Detection Features")
    df = df.copy()
    df['GAMES_THIS_SEASON'] = df.groupby(['PLAYER_ID', 'SEASON_ID']).cumcount() + 1
    df['IS_EARLY_SEASON']   = (df['GAMES_THIS_SEASON'] <= 10).astype(int)
    if 'CAREER_GAMES' in df.columns:
        df['IS_ROOKIE']         = (df['CAREER_GAMES'] <= 82).astype(int)
        df['ROOKIE_VOLATILITY'] = 1.0 + (1.5 * np.exp(-df['CAREER_GAMES'] / 50))
    else:
        df['IS_ROOKIE']         = 0
        df['ROOKIE_VOLATILITY'] = 1.0
    return df


def add_momentum_features(df):
    print("...Adding Momentum Features")
    df = df.copy()
    for stat in ['PTS', 'REB', 'AST', 'FG3M']:
        df[f'{stat}_L3_AVG'] = df.groupby('PLAYER_ID')[stat].transform(
            lambda x: x.shift(1).rolling(3, min_periods=3).mean())
        season_col = f'{stat}_Season'
        if season_col in df.columns:
            df[f'{stat}_HOT_STREAK'] = (df[f'{stat}_L3_AVG'] - df[season_col]).fillna(0)
        df = df.drop(columns=[f'{stat}_L3_AVG'], errors='ignore')
    return df


def add_head_to_head_stats(df):
    print("...Adding Head-to-Head Stats")
    df = df.copy()
    for stat in ['PTS', 'REB', 'AST']:
        df[f'{stat}_VS_OPP'] = df.groupby(['PLAYER_ID', 'OPPONENT'])[stat].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean())
        season_col = f'{stat}_Season'
        if season_col in df.columns:
            df[f'{stat}_VS_OPP'] = df[f'{stat}_VS_OPP'].fillna(df[season_col])
    return df


def add_blocks_specific_features(df):
    print("...Adding Block-Specific Features")
    df = df.copy()
    df['OPP_RIM_ATTEMPTS'] = df.groupby(['OPPONENT', 'SEASON_ID']).apply(
        lambda x: (x['FGA'] - x['FG3A']).shift(1).rolling(10, min_periods=5).mean()
    ).reset_index(level=[0, 1], drop=True)
    df['OPP_RIM_ATTEMPTS'] = df['OPP_RIM_ATTEMPTS'].fillna(
        df.groupby('SEASON_ID')['FGA'].transform('median') * 0.6)
    if 'PACE_ROLLING' in df.columns:
        df['OPP_RIM_ATTEMPT_RATE'] = df['OPP_RIM_ATTEMPTS'] / (df['PACE_ROLLING'] + 0.1)
    else:
        df['OPP_RIM_ATTEMPT_RATE'] = df['OPP_RIM_ATTEMPTS'] / 100
    df['IN_FOUL_TROUBLE']   = (df['PF'] >= 4).astype(int)
    df['FOUL_TROUBLE_RATE'] = df.groupby('PLAYER_ID')['IN_FOUL_TROUBLE'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()).fillna(0)
    position_block_avg = df.groupby(['POSITION', 'SEASON_ID'])['BLK'].transform('median')
    df['POSITION_BLOCK_BASELINE'] = position_block_avg
    if 'BLK_Season' in df.columns:
        df['BLOCK_SKILL_ADVANTAGE'] = df['BLK_Season'] - df['POSITION_BLOCK_BASELINE']
    else:
        df['BLOCK_SKILL_ADVANTAGE'] = 0
    return df


def add_steals_specific_features(df):
    print("...Adding Steal-Specific Features")
    df = df.copy()
    df['OPP_TOV_RATE'] = df.groupby(['OPPONENT', 'SEASON_ID'])['TOV'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean()).fillna(df['TOV'].median())
    if 'PACE_ROLLING' in df.columns:
        df['OPP_TOV_PER_100'] = (df['OPP_TOV_RATE'] / df['PACE_ROLLING']) * 100
    else:
        df['OPP_TOV_PER_100'] = df['OPP_TOV_RATE']
    df['STEAL_ATTEMPT_RATE']  = df['STL'] / (df['MIN'] + 0.1)
    df['STEAL_CONSISTENCY']   = df.groupby('PLAYER_ID')['STL'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).std()).fillna(1.0)
    df['POSITION_STEAL_BASELINE'] = df.groupby(['POSITION', 'SEASON_ID'])['STL'].transform('median')
    return df


def add_turnover_specific_features(df):
    print("...Adding Turnover-Specific Features")
    df = df.copy()
    if 'OPP_STL_ALLOWED' in df.columns:
        df['OPP_PRESSURE_RATE'] = df['OPP_STL_ALLOWED']
    else:
        df['OPP_PRESSURE_RATE'] = df.groupby(['OPPONENT', 'SEASON_ID'])['STL'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=5).mean()).fillna(df['STL'].median())
    if 'USAGE_RATE_L5' in df.columns and 'USAGE_RATE_Season' in df.columns:
        df['USAGE_SPIKE'] = (df['USAGE_RATE_L5'] - df['USAGE_RATE_Season']).clip(lower=0)
    else:
        df['USAGE_SPIKE'] = 0
    df['AST_TO_TOV_RATIO'] = df['AST'] / (df['TOV'] + 0.1)
    df['AST_TO_TOV_SKILL']  = df.groupby('PLAYER_ID')['AST_TO_TOV_RATIO'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()).fillna(2.0)
    if 'TEAM_WIN_PCT' in df.columns:
        df['GAME_SCRIPT_RISK'] = (0.5 - df['TEAM_WIN_PCT']).clip(lower=0)
    else:
        df['GAME_SCRIPT_RISK'] = 0
    return df


def add_rebound_specific_features(df):
    print("...Adding Rebound-Specific Features")
    df = df.copy()
    df['TEAM_OREB_EMPHASIS'] = df.groupby(['TEAM_ID', 'SEASON_ID']).apply(
        lambda x: x['OREB'].shift(1).rolling(10, min_periods=5).sum() /
                  (x['FGA'].shift(1).rolling(10, min_periods=5).sum() + 0.1)
    ).reset_index(level=[0, 1], drop=True).fillna(0.25)
    df['OPP_REB_WEAKNESS'] = df.groupby(['OPPONENT', 'SEASON_ID']).apply(
        lambda x: (x['OREB'] + x['DREB']).shift(1).rolling(10, min_periods=5).mean()
    ).reset_index(level=[0, 1], drop=True)
    df['OPP_REB_WEAKNESS']    = df['OPP_REB_WEAKNESS'].fillna(df.groupby('SEASON_ID')['REB'].transform('median'))
    df['MISSED_SHOTS_PROXY']  = df['FGA'] - df['FGM']
    df['REBOUND_OPPORTUNITY'] = df.groupby(['GAME_ID', 'TEAM_ID'])['MISSED_SHOTS_PROXY'].transform('sum')
    df['POSITION_REB_BASELINE'] = df.groupby(['POSITION', 'SEASON_ID'])['REB'].transform('median')
    return df


def add_assist_specific_features(df):
    print("...Adding Assist-Specific Features")
    df = df.copy()
    team_fgm = df.groupby(['GAME_ID', 'TEAM_ID'])['FGM'].transform('sum')
    team_fga = df.groupby(['GAME_ID', 'TEAM_ID'])['FGA'].transform('sum')
    df['TEAMMATE_FGM']    = team_fgm - df['FGM']
    df['TEAMMATE_FGA']    = team_fga - df['FGA']
    df['TEAMMATE_FG_PCT'] = df['TEAMMATE_FGM'] / (df['TEAMMATE_FGA'] + 0.1)
    df['TEAMMATE_SHOOTING_L10'] = df.groupby(['PLAYER_ID', 'SEASON_ID'])['TEAMMATE_FG_PCT'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()).fillna(0.45)
    if 'USAGE_RATE_Season' in df.columns and 'PTS_Season' in df.columns:
        df['PLAYMAKER_ROLE'] = (df['USAGE_RATE_Season'] / (df['PTS_Season'] + 0.1)).fillna(0).clip(upper=2.0)
    else:
        df['PLAYMAKER_ROLE'] = 0
    if 'PACE_ROLLING' in df.columns and 'USAGE_RATE_Season' in df.columns:
        df['ASSIST_OPPORTUNITY'] = (df['PACE_ROLLING'] / 100) * (df['USAGE_RATE_Season'] / 20)
    else:
        df['ASSIST_OPPORTUNITY'] = 1.0
    df['POSITION_AST_BASELINE'] = df.groupby(['POSITION', 'SEASON_ID'])['AST'].transform('median')
    return df


def ensure_combo_stats(df):
    df = df.copy()
    if 'PRA' not in df.columns: df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    if 'PR'  not in df.columns: df['PR']  = df['PTS'] + df['REB']
    if 'PA'  not in df.columns: df['PA']  = df['PTS'] + df['AST']
    if 'RA'  not in df.columns: df['RA']  = df['REB'] + df['AST']
    if 'SB'  not in df.columns: df['SB']  = df['STL'] + df['BLK']
    return df


def validate_data_quality(df):
    print("...Running Data Quality Checks")
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_pct = df.isna().mean()
    problematic_cols = nan_pct[nan_pct > 0.5].index.tolist()
    if problematic_cols:
        print(f"   ⚠️  WARNING: High NaN % in columns: {problematic_cols}")
    player_games = df.groupby('PLAYER_ID').size()
    low_sample = player_games[player_games < 10].count()
    if low_sample > 0:
        print(f"   ℹ️  Info: {low_sample} players have <10 games (will be filtered)")
    if 'PTS' in df.columns:
        max_pts = df['PTS'].max()
        if max_pts > 100:
            print(f"   ⚠️  WARNING: Max PTS = {max_pts} (seems high)")
    return df


def main():
    start_time = datetime.now()
    print("\n" + "="*60)
    print("   NBA FEATURE ENGINEERING PIPELINE v2.2")
    print("="*60 + "\n")

    df = load_and_merge_data()
    if df is None:
        print("❌ Pipeline failed: Could not load data")
        return

    print("\n--- STAGE 1: BASE FEATURES ---")
    df = add_advanced_stats(df)
    df = add_context_features(df)
    df = add_team_performance_context(df)

    print("\n--- STAGE 2: OPPORTUNITY FEATURES ---")
    df = add_missing_player_context(df)
    df = add_schedule_density(df)
    df = add_pace_features(df)

    df = ensure_combo_stats(df)
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    print("\n--- STAGE 3: HISTORICAL FEATURES ---")
    df = add_rolling_features(df)

    print("\n--- STAGE 4: ADVANCED FEATURES ---")
    df = add_role_features(df)
    df = add_rookie_features(df)
    df = add_momentum_features(df)
    df = add_efficiency_signals(df)

    print("\n--- STAGE 5: MATCHUP FEATURES ---")
    df = add_defense_vs_position(df)
    df = add_head_to_head_stats(df)
    df = add_usage_vacuum_features(df)

    print("\n--- STAGE 6: WEAK MODEL ENHANCEMENTS ---")
    df = add_blocks_specific_features(df)
    df = add_steals_specific_features(df)
    df = add_turnover_specific_features(df)
    df = add_rebound_specific_features(df)
    df = add_assist_specific_features(df)

    print("\n--- STAGE 7: QUALITY CHECKS ---")
    df = validate_data_quality(df)

    print("\n--- STAGE 8: FINAL CLEANING ---")
    initial_rows = len(df)
    df = df[df['MIN'] >= 10]
    print(f"   Filtered {initial_rows - len(df):,} low-minute games (MIN < 10)")
    df = df.dropna()
    print(f"   Dropped {initial_rows - len(df):,} rows with missing values")

    print("\n--- STAGE 9: SAVING ---")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*60)
    print("   ✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"   Output:   {OUTPUT_FILE}")
    print(f"   Rows:     {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Players:  {df['PLAYER_ID'].nunique():,}")
    print(f"   Runtime:  {elapsed:.1f} seconds")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
