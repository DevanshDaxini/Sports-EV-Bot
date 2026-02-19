"""
Tennis Statistical Feature Engineering Pipeline

Transforms raw player-match rows into a rich feature set for XGBoost models.
Creates 150+ predictive features including rolling averages, surface splits,
opponent rank adjustments, fatigue indicators, and H2H signals.

Targets (7 markets, Fantasy Score deferred):
    total_games, games_won, total_sets, aces,
    bp_won, total_tiebreaks, double_faults

Output:
    data/tennis/processed/training_dataset.csv

Usage:
    $ python3 -m src.sports.tennis.features
"""

import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ATP_FILE    = os.path.join(BASE_DIR, 'data', 'tennis', 'raw', 'atp_raw_matches.csv')
WTA_FILE    = os.path.join(BASE_DIR, 'data', 'tennis', 'raw', 'wta_raw_matches.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'tennis', 'processed', 'training_dataset.csv')

# These are the columns our XGBoost models will predict (one model per target)
TARGET_STATS = [
    'total_games',
    'games_won',
    'total_sets',
    'aces',
    'bp_won',
    'total_tiebreaks',
    'double_faults',
]

# Surface encoding (one-hot)
SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet']

# Round encoding (ordinal — later rounds = higher stakes/longer matches)
ROUND_ORDER = {
    'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
    'QF': 5, 'SF': 6, 'F': 7, 'RR': 3,
}


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_data():
    print("...Loading raw match data")
    frames = []
    for path, tour in [(ATP_FILE, 'atp'), (WTA_FILE, 'wta')]:
        if not os.path.exists(path):
            print(f"   ⚠️  Missing: {path}  — run builder.py first")
            continue
        df = pd.read_csv(path, low_memory=False)
        df['tour'] = tour
        frames.append(df)

    if not frames:
        print("ERROR: No data found.")
        return None

    df = pd.concat(frames, ignore_index=True)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    df = df.dropna(subset=['tourney_date', 'player_id'])
    df = df.sort_values(['player_id', 'tourney_date']).reset_index(drop=True)

    # Drop retired matches from TARGETS — partial stats skew the model
    df = df[df['retired'] == 0].copy()

    print(f"   Loaded {len(df):,} completed matches | {df['player_name'].nunique():,} players")
    return df


# ---------------------------------------------------------------------------
# SURFACE & CONTEXT FEATURES
# ---------------------------------------------------------------------------

def add_context_features(df):
    print("...Adding context features")

    # One-hot surface
    for s in SURFACES:
        df[f'surface_{s.lower()}'] = (df['surface'] == s).astype(int)

    # Round ordinal
    df['round_ordinal'] = df['round'].map(ROUND_ORDER).fillna(3)

    # Best-of indicator (best_of=5 for Slams)
    df['is_best_of_5'] = (df['best_of'] == 5).astype(int)

    # Tour indicator
    df['is_atp'] = (df['tour'] == 'atp').astype(int)

    # Rank features
    df['player_rank']   = pd.to_numeric(df['player_rank'],   errors='coerce').fillna(200)
    df['opp_rank']      = pd.to_numeric(df['opp_rank'],      errors='coerce').fillna(200)
    df['rank_delta']    = df['player_rank'] - df['opp_rank']   # negative = you're higher ranked
    df['rank_ratio']    = df['player_rank'] / (df['opp_rank'] + 1)
    df['log_rank']      = np.log1p(df['player_rank'])
    df['log_opp_rank']  = np.log1p(df['opp_rank'])

    return df


# ---------------------------------------------------------------------------
# ROLLING AVERAGES (player's own recent form)
# ---------------------------------------------------------------------------

def add_rolling_features(df):
    print("...Calculating rolling averages (L5, L20, Season)")
    df = df.copy()

    stats_to_roll = TARGET_STATS + ['won_match', 'sets_won', 'bp_faced', 'svpt', 'svc_games']

    grouped = df.groupby('player_id')
    rolling_data = {}

    for stat in stats_to_roll:
        if stat not in df.columns:
            continue
        # Shift(1) prevents leakage — current match not included
        s = grouped[stat].transform(lambda x: x.shift(1))
        rolling_data[f'{stat}_L5']     = s.groupby(df['player_id']).transform(lambda x: x.rolling(5,  min_periods=3).mean())
        rolling_data[f'{stat}_L20']    = s.groupby(df['player_id']).transform(lambda x: x.rolling(20, min_periods=10).mean())
        rolling_data[f'{stat}_Season'] = grouped[stat].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

    df = pd.concat([df, pd.DataFrame(rolling_data, index=df.index)], axis=1)
    return df


# ---------------------------------------------------------------------------
# SURFACE-SPECIFIC ROLLING AVERAGES
# ---------------------------------------------------------------------------

def add_surface_rolling(df):
    """
    Rolling averages split by surface.
    E.g. aces_hard_L10 = player's avg aces on hard courts, last 10 matches.
    Critical because clay/hard/grass produce very different stats.
    """
    print("...Calculating surface-split rolling averages")
    df = df.copy()
    key_stats = ['aces', 'double_faults', 'total_games', 'games_won', 'bp_won']

    surface_rolling = {}
    for surf in ['Hard', 'Clay', 'Grass']:
        mask = df['surface'] == surf
        surf_key = surf.lower()

        for stat in key_stats:
            if stat not in df.columns:
                continue
            col_name = f'{stat}_{surf_key}_L10'
            # Fill non-matching surface rows with NaN, then forward-fill within player
            stat_on_surf = df[stat].where(mask)
            surface_rolling[col_name] = (
                df.groupby('player_id')[stat]
                  .transform(lambda x: x.where(df.loc[x.index, 'surface'] == surf)
                                        .shift(1).rolling(10, min_periods=3).mean())
            )

    df = pd.concat([df, pd.DataFrame(surface_rolling, index=df.index)], axis=1)
    return df


# ---------------------------------------------------------------------------
# OPPONENT ROLLING AVERAGES (how does the opponent typically play?)
# ---------------------------------------------------------------------------

def add_opponent_features(df):
    """
    For each match, attach the opponent's recent rolling stats.
    E.g. opp_aces_L10 = how many aces the opponent typically hits.
    Helps predict total_games, aces allowed, etc.
    """
    print("...Attaching opponent rolling stats")

    # Build a lookup of player's rolling stats by player_id + tourney_date
    opp_cols = {}
    for stat in ['total_games', 'games_won', 'aces', 'double_faults', 'bp_won', 'bp_faced']:
        for window in ['L5', 'L20']:
            col = f'{stat}_{window}'
            if col in df.columns:
                opp_cols[f'opp_{col}'] = col

    if not opp_cols:
        return df

    # Create a slim lookup table: player_id, tourney_date -> rolling stats
    lookup_df = df[['player_id', 'tourney_date'] + list(set(opp_cols.values()))].copy()
    lookup_df = lookup_df.rename(columns={'player_id': 'opp_id', 'tourney_date': 'tourney_date'})
    rename_map = {v: k for k, v in opp_cols.items()}
    lookup_df = lookup_df.rename(columns=rename_map)

    df = df.merge(lookup_df, on=['opp_id', 'tourney_date'], how='left')
    return df


def add_opponent_surface_context(df):
    """
    Merge opponent's SURFACE-SPECIFIC rolling stats.
    e.g. opp_aces_clay_L10 = how many aces the opponent hits ON CLAY.
    """
    print("...Attaching opponent surface-specific stats")
    
    # Identify surface-specific rolling cols from add_surface_rolling
    # Format: {stat}_{surface}_L10
    surf_cols = [c for c in df.columns if '_L10' in c and ('hard' in c or 'clay' in c or 'grass' in c)]
    
    if not surf_cols:
        return df
        
    lookup_df = df[['player_id', 'tourney_date'] + surf_cols].copy()
    lookup_df = lookup_df.rename(columns={'player_id': 'opp_id'})
    
    # Rename cols to opp_{stat}_{surface}_L10
    rename_map = {c: f'opp_{c}' for c in surf_cols}
    lookup_df = lookup_df.rename(columns=rename_map)
    
    df = df.merge(lookup_df, on=['opp_id', 'tourney_date'], how='left')
    return df


# ---------------------------------------------------------------------------
# FATIGUE / SCHEDULE FEATURES
# ---------------------------------------------------------------------------

def add_fatigue_features(df):
    print("...Adding fatigue & schedule features")

    df = df.sort_values(['player_id', 'tourney_date'])

    # Days since last match
    df['prev_match_date'] = df.groupby('player_id')['tourney_date'].shift(1)
    df['days_rest'] = (df['tourney_date'] - df['prev_match_date']).dt.days.clip(upper=30).fillna(14)

    # Matches in last 14 days
    def matches_in_window(group, days=14):
        dates = group['tourney_date'].values
        result = []
        for i, d in enumerate(dates):
            cutoff = d - pd.Timedelta(days=days)
            count  = np.sum((dates[:i] > cutoff) & (dates[:i] < d))
            result.append(count)
        return pd.Series(result, index=group.index)

    df['matches_L14D'] = df.groupby('player_id', group_keys=False).apply(matches_in_window)

    # Is this a back-to-back day (within tournament)?
    df['is_b2b'] = (df['days_rest'] <= 1).astype(int)

    df.drop(columns=['prev_match_date'], inplace=True)
    return df


# ---------------------------------------------------------------------------
# H2H FEATURES
# ---------------------------------------------------------------------------

def add_h2h_features(df):
    """
    Head-to-head history between player and specific opponent.
    Uses only PRIOR matches to avoid leakage.
    """
    print("...Calculating H2H features")
    df = df.copy()
    df = df.sort_values(['player_id', 'opp_id', 'tourney_date'])

    df['h2h_key'] = df.apply(
        lambda r: '_'.join(sorted([str(r['player_id']), str(r['opp_id'])])), axis=1
    )

    # Win rate in prior H2H
    df['h2h_win_rate'] = (
        df.groupby(['player_id', 'opp_id'])['won_match']
          .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
          .fillna(0.5)
    )

    # Avg games in prior H2H (match length predictor)
    df['h2h_avg_games'] = (
        df.groupby(['h2h_key'])['total_games']
          .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
          .fillna(df['total_games'].mean())
    )

    df.drop(columns=['h2h_key'], inplace=True)
    return df


def add_rank_bracket_performance(df):
    """
    Calculate rolling win rate and stats against specific rank brackets:
    - Vs Top 20
    - Vs Top 50
    - Vs Top 100
    Addresses: "How they have performed against a similar player in that ranking"
    """
    print("...Calculating performance vs Rank Brackets (Top 20/50/100)")
    df = df.copy()
    
    # Pre-calculate opponent rank buckets for every match
    # active_opp_rank is the rank of the OPPONENT in that match
    # We want to know: when I played a Top 20 guy, did I win?
    
    brackets = [20, 50, 100]
    
    # We need to shift(1) to avoid leakage (prior matches only)
    # Group by player
    grouped = df.sort_values(['player_id', 'tourney_date']).groupby('player_id')
    
    for b in brackets:
        # 1. Flag matches where opponent was in this bracket
        is_in_bracket = (df['opp_rank'] <= b)
        
        # 2. Calculate rolling win rate in these specific matches
        # We only care about rows where the opponent WAS in the bracket
        # But we need to map it back to the current row index
        
        # Create a series where we have values only for relevant matches
        # won_match = 1 or 0. If opp not in bracket, set to NaN so it's ignored by rolling?
        # No, rolling ignores NaNs? Yes.
        
        relevant_wins = df['won_match'].where(is_in_bracket)
        
        # Rolling mean of wins against this bracket
        # We need to shift FIRST, then roll.
        # But rolling on a series with NaNs (skipped matches) works if we use min_periods?
        # Actually, we want the last N matches *against this bracket*.
        
        # Method: 
        # 1. Filter to just matches vs Top X
        # 2. Calculate rolling stats on that filtered df
        # 3. Merge back to main df by (player_id, date)
        
        # Filter:
        vs_bracket = df[df['opp_rank'] <= b].copy()
        if vs_bracket.empty:
            continue
            
        vs_bracket = vs_bracket.sort_values(['player_id', 'tourney_date'])
        
        # Calculate rolling win rate (L10 matches vs this bracket)
        # Shift 1 to exclude current match
        vs_bracket[f'win_rate_vs_top{b}'] = (
            vs_bracket.groupby('player_id')['won_match']
            .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        )
        
        # Merge back
        # We need the most recent value for the player as of this date
        # Forward fill? 
        # Actually, standard merge on player/date will only populate rows where the CURRENT opp is top X.
        # But we want to know my history vs Top X *regardless* of who I'm playing today?
        # "How does he perform vs Top 20?" is a feature of the PLAYER, valid for ANY match.
        
        # Correct approach:
        # 1. Calculate the running 'Vs Top X' stats for every match the player played
        #    (carrying forward the last known value)
        
        # Let's iterate players? Slow.
        # Vectorized:
        # Create a 'win_vs_topX' column (1/0/NaN).
        col_name = f'win_vs_top{b}_L10'
        
        # Mask: 1 if Won & Opp<=X, 0 if Lost & Opp<=X, NaN otherwise
        wins_vs_bracket = df['won_match'].where(df['opp_rank'] <= b)
        
        # We want the rolling mean of the valid values (ignoring NaNs), 
        # but we want the result available on EVERY row.
        # Pandas rolling() counts NaNs in the window size (step-wise), unless using time-based?
        # No, we want "Last 10 matches against Top 20".
        # This is hard to vectorize perfectly without complex indirection.
        
        # Approximation: 
        # Rolling Global L50, filtered? No.
        
        # Alternative: Cumulative stats (easier). 
        # expanding().mean() of wins_vs_bracket?
        
        # Let's implement Expanding Mean (Career Win % vs Top 20)
        # This is a strong signal. "Career win rate vs Top 20".
        
        # Group by player, transforming the wins_vs_bracket series
        # We use a custom expanding mean that skips NaNs
        # shift(1) is crucial
        
        df[f'career_win_pct_vs_top{b}'] = (
            grouped['won_match']
            .apply(lambda x: x.where(df.loc[x.index, 'opp_rank'] <= b).shift(1).expanding().mean())
            .reset_index(level=0, drop=True)
        ).fillna(0)  # Default to 0 if no history
        
        # Also Last 5 matches vs Top X? Harder to vectorize. 
        # Let's stick to Career Win % vs Bracket for now - robust and significant.
    
    return df


# ---------------------------------------------------------------------------
# CAREER / EXPERIENCE FEATURES
# ---------------------------------------------------------------------------

def add_career_features(df):
    print("...Adding career/experience features")
    df['career_matches'] = df.groupby('player_id').cumcount() + 1
    df['is_early_career'] = (df['career_matches'] < 30).astype(int)
    return df


# ---------------------------------------------------------------------------
# FINALIZE
# ---------------------------------------------------------------------------

def finalize_dataset(df):
    print("...Finalizing dataset")

    # Drop rows where all targets are NaN
    df = df.dropna(subset=TARGET_STATS, how='all')

    # Fill remaining NaNs in feature cols with 0 (safe default for tree models)
    feature_cols = [c for c in df.columns if c not in TARGET_STATS + [
        'player_id', 'opp_id', 'player_name', 'opp_name',
        'tourney_id', 'tourney_name', 'score', 'tourney_date',
    ]]
    df[feature_cols] = df[feature_cols].fillna(0)

    print(f"   Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Players: {df['player_name'].nunique():,}")
    print(f"   Date range: {df['tourney_date'].min().date()} → {df['tourney_date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def build_features():
    print("=" * 55)
    print("   🎾 TENNIS FEATURE ENGINEERING")
    print("=" * 55)

    df = load_data()
    if df is None:
        return

    df = add_context_features(df)
    df = add_rolling_features(df)
    df = add_surface_rolling(df)
    df = add_opponent_features(df)
    df = add_opponent_surface_context(df)  # NEW
    df = add_fatigue_features(df)
    df = add_h2h_features(df)
    df = add_rank_bracket_performance(df)  # NEW
    df = add_career_features(df)
    df = finalize_dataset(df)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅  Saved training dataset → {OUTPUT_FILE}")
    print("   Next step: Run train.py to build models.")
    print("=" * 55)


if __name__ == "__main__":
    build_features()
