"""
Tennis Props Scanner - AI-Powered Prediction System

Scans upcoming ATP/WTA matches, generates predictions using trained XGBoost
models, and identifies profitable PrizePicks opportunities.

Zero external API dependencies beyond PrizePicks:
    - Rankings:  Sackmann GitHub (free, same source as training data)
    - Surface:   Derived from tournament name via built-in lookup table
    - Schedule:  Driven entirely by PrizePicks lines (if PP has a line, there's a match)

Features:
    1. Scan today's matches
    2. Scan tomorrow's matches (with auto forward-search)
    3. Grade past results
    4. Scout specific player

Usage:
    $ python3 -m src.sports.tennis.scanner
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
import unicodedata
import re
from datetime import datetime, timedelta

from src.core.odds_providers.prizepicks import PrizePicksClient
from src.sports.tennis.config   import STAT_MAP, STAT_MAP_REVERSE, MODEL_QUALITY, ACTIVE_TARGETS
from src.sports.tennis.mappings import STAT_MAPPING
from src.sports.tennis.train    import FEATURES
from src.sports.tennis.rankings import TennisRankings

warnings.filterwarnings('ignore')

# --- PATHS ---
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR  = os.path.join(BASE_DIR, 'models', 'tennis')
DATA_FILE  = os.path.join(BASE_DIR, 'data',   'tennis', 'processed', 'training_dataset.csv')
PROJ_DIR   = os.path.join(BASE_DIR, 'data',   'tennis', 'projections')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'tennis', 'scans')
ACCURACY_LOG_FILE = os.path.join(PROJ_DIR, 'accuracy_log.csv')

os.makedirs(PROJ_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalize_name(name):
    if not name:
        return ""
    n = unicodedata.normalize('NFD', str(name))
    n = ''.join(c for c in n if unicodedata.category(c) != 'Mn')
    n = re.sub(r"[^a-zA-Z\s]", '', n)
    return ' '.join(n.lower().split())


def get_betting_indicator(proj, line):
    if line is None or line <= 0:
        return "⚪ NO LINE"
    diff = proj - line
    if diff > 0:
        return f"🟢 OVER  (+{diff:.2f})"
    return f"🔴 UNDER ({diff:.2f})"


def _map_pp_stat_to_target(pp_stat):
    """Map PrizePicks stat name to internal target (with flexible matching)."""
    if not pp_stat:
        return None
    target = STAT_MAPPING.get(pp_stat)
    if target:
        return target
    # Already a target code (e.g. PP returns 'total_games')
    if pp_stat in STAT_MAPPING.values():
        return pp_stat
    pp_lower = str(pp_stat).strip().lower()
    for key, val in STAT_MAPPING.items():
        if key.lower() == pp_lower:
            return val
    return None


def _target_display_name(target):
    """Convert internal target (e.g. total_games) to display name (e.g. Total Games)."""
    return STAT_MAP_REVERSE.get(target, target.replace('_', ' ').title())


# ---------------------------------------------------------------------------
# LOAD DATA & MODELS
# ---------------------------------------------------------------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌  Training data not found: {DATA_FILE}")
        print("    Run builder.py → features.py first.")
        return None
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    df = df.sort_values(['player_name', 'tourney_date']).reset_index(drop=True)
    print(f"   ✅  Loaded history: {len(df):,} matches | {df['player_name'].nunique():,} players")
    return df


def load_models():
    models = {}
    for target in ACTIVE_TARGETS:
        path = os.path.join(MODEL_DIR, f'{target}_model.json')
        if os.path.exists(path):
            m = xgb.XGBRegressor()
            m.load_model(path)
            models[target] = m
        else:
            print(f"   ⚠️  Model not found: {target}")
    print(f"   ✅  Loaded {len(models)}/{len(ACTIVE_TARGETS)} models")
    return models


# ---------------------------------------------------------------------------
# PRIZEPICKS FETCH WITH DATE FORWARD-SEARCH
# ---------------------------------------------------------------------------

def get_pp_lines(date_offset=0, max_days_forward=7):
    """
    Fetch PrizePicks tennis lines for the next available date.

    Strategy:
        1. Fetch the FULL board once (uses 30-min disk cache — no repeated API calls)
        2. Filter for tennis using case-insensitive league name match
        3. Search dates in the filtered results

    Returns:
        (pp_board DataFrame, actual_date_str)
    """
    pp_client    = PrizePicksClient(stat_map=STAT_MAP)
    initial_date = datetime.now() + timedelta(days=date_offset)

    print("...Fetching PrizePicks board (full, single request)...")

    # --- Fetch full board once — cache handles repeat calls ---
    full_board = pp_client.fetch_board()   # no league filter = everything

    if full_board.empty:
        print("❌  PrizePicks board unavailable (rate limited or down).")
        print("    Wait 30 seconds and try again.")
        return pd.DataFrame(), None

    # --- Filter for tennis with case-insensitive match ---
    tennis_board = full_board[
        full_board['League'].str.lower().str.contains('tennis', na=False)
    ].copy()

    if tennis_board.empty:
        # Show what leagues ARE available so user knows what's on the board
        available = full_board['League'].dropna().unique().tolist()
        print(f"   ⚠️  No tennis lines on PrizePicks right now.")
        print(f"   Available leagues: {available[:10]}")
        print(f"   Tennis lines are posted 1-2 days before matches.")
        return pd.DataFrame(), None

    print(f"   ✅  Found {len(tennis_board)} tennis lines for {tennis_board['Player'].nunique()} players")

    # --- Find lines for the target date (or nearest future date) ---
    tennis_board['Date'] = pd.to_datetime(tennis_board['Date'], errors='coerce')
    target_start = initial_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Group by date, find closest date >= target
    available_dates = sorted(tennis_board['Date'].dropna().unique())
    chosen_date_ts  = None

    for d in available_dates:
        if pd.Timestamp(d) >= pd.Timestamp(target_start):
            chosen_date_ts = d
            break

    if chosen_date_ts is None:
        # All dates are in the past — just take the most recent
        if available_dates:
            chosen_date_ts = available_dates[-1]
        else:
            print("   ⚠️  Could not determine match date from PP lines.")
            chosen_date_str = initial_date.strftime('%Y-%m-%d')
            return tennis_board.reset_index(drop=True), chosen_date_str

    chosen_date_str = pd.Timestamp(chosen_date_ts).strftime('%Y-%m-%d')
    day_name        = pd.Timestamp(chosen_date_ts).strftime('%A')

    # Filter to just that date
    date_board = tennis_board[tennis_board['Date'].dt.strftime('%Y-%m-%d') == chosen_date_str]

    if date_board.empty:
        # PP lines don't have dates — return all tennis lines
        date_board = tennis_board
        chosen_date_str = initial_date.strftime('%Y-%m-%d')

    # Count unique markets (stats) in the board
    stats_in_board = date_board['Stat'].dropna().unique().tolist()
    print(f"   📅 Lines found for: {day_name}, {chosen_date_str} ({len(date_board)} lines, {date_board['Player'].nunique()} players)")
    if stats_in_board:
        print(f"   📊 Markets: {', '.join(str(s) for s in stats_in_board)}")
    return date_board.reset_index(drop=True), chosen_date_str


# ---------------------------------------------------------------------------
# FEATURE PREPARATION
# ---------------------------------------------------------------------------

def get_player_latest_row(player_name, df_history):
    """Find most recent history row for a player."""
    norm = normalize_name(player_name)

    # Exact normalized match
    mask = df_history['player_name'].apply(normalize_name) == norm
    rows = df_history[mask]

    if rows.empty:
        # Last-name partial match
        last = norm.split()[-1] if norm else ''
        if len(last) > 3:
            mask = df_history['player_name'].apply(
                lambda n: normalize_name(n).split()[-1] == last
            )
            rows = df_history[mask]

    return rows.iloc[-1] if not rows.empty else None


def build_feature_row(player_row, surface='Hard', opp_rank=50.0, player_rank=50.0,
                       is_best_of_5=0, round_ordinal=3, is_atp=1, days_rest=2):
    latest = player_row.to_dict()

    # Clean rank values
    if not player_rank or player_rank <= 0 or (isinstance(player_rank, float) and np.isnan(player_rank)):
        player_rank = 50.0
    if not opp_rank or opp_rank <= 0 or (isinstance(opp_rank, float) and np.isnan(opp_rank)):
        opp_rank = 50.0

    latest.update({
        'surface_hard':   1 if surface == 'Hard'   else 0,
        'surface_clay':   1 if surface == 'Clay'   else 0,
        'surface_grass':  1 if surface == 'Grass'  else 0,
        'surface_carpet': 1 if surface == 'Carpet' else 0,
        'is_best_of_5':   is_best_of_5,
        'round_ordinal':  round_ordinal,
        'is_atp':         is_atp,
        'player_rank':    player_rank,
        'opp_rank':       opp_rank,
        'rank_delta':     player_rank - opp_rank,
        'rank_ratio':     player_rank / (opp_rank + 1),
        'log_rank':       np.log1p(player_rank),
        'log_opp_rank':   np.log1p(opp_rank),
        'days_rest':      days_rest,
        'is_b2b':         1 if days_rest <= 1 else 0,
    })
    return pd.DataFrame([latest])


def predict(player_row, models, surface, opp_rank, player_rank,
            is_best_of_5, round_ordinal, is_atp):
    """Run all models for one player. Returns dict of {target: prediction}."""
    feat_row = build_feature_row(
        player_row, surface=surface, opp_rank=opp_rank, player_rank=player_rank,
        is_best_of_5=is_best_of_5, round_ordinal=round_ordinal, is_atp=is_atp,
    )
    preds = {}
    for target, model in models.items():
        model_features = [f for f in model.feature_names_in_ if f in feat_row.columns]
        X = feat_row.reindex(columns=model_features, fill_value=0)
        try:
            preds[target] = float(model.predict(X)[0])
        except Exception:
            pass
    return preds


# ---------------------------------------------------------------------------
# SCAN ALL
# ---------------------------------------------------------------------------

def scan_all(df_history, models, rankings: TennisRankings, is_tomorrow=False):
    offset       = 1 if is_tomorrow else 0
    pp_board, actual_date = get_pp_lines(date_offset=offset, max_days_forward=7)

    if pp_board.empty or actual_date is None:
        input("\nPress Enter to continue...")
        return

    scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
    print(f"\n📅 Scanning tennis for: {scan_date_obj.strftime('%A, %B %d, %Y')}")
    print(f"   {pp_board['Player'].nunique()} players | {len(pp_board)} total lines\n")

    # --- Extract tournament context from PrizePicks board if available ---
    # PP sometimes includes a league/tournament column
    tournament_name = ''
    if 'League' in pp_board.columns:
        leagues = pp_board['League'].dropna().unique()
        if len(leagues) > 0:
            tournament_name = str(leagues[0])

    surface      = rankings.get_surface(tournament_name)
    is_slam      = rankings.is_slam(tournament_name)
    round_ord    = 3   # default to R32 equivalent

    if tournament_name:
        print(f"   🏆 Tournament: {tournament_name}")
    print(f"   🎾 Surface:    {surface}")
    if is_slam:
        print(f"   📌 Grand Slam — men play best-of-5")

    # --- Build player-centric lines dict (like NBA: player -> {target: line}) ---
    norm_lines     = {}   # norm_name -> {target: line}
    display_names  = {}   # norm_name -> display name for output
    for _, pp_row in pp_board.iterrows():
        pp_name = pp_row.get('Player', '')
        pp_stat = pp_row.get('Stat',   '')
        line    = pp_row.get('Line')
        target  = _map_pp_stat_to_target(pp_stat)
        if not target or target not in models:
            continue
        if line is None or float(line) <= 0:
            continue
        norm = normalize_name(pp_name)
        display_names[norm] = pp_name
        norm_lines.setdefault(norm, {})[target] = float(line)

    # --- Generate predictions (player-centric, like NBA: one predict per player, all markets) ---
    print("🚀 Generating Predictions...")
    best_bets       = []
    all_projections = []
    skipped         = []

    for norm_name, lines_for_player in norm_lines.items():
        pp_name = display_names.get(norm_name, norm_name)

        player_row = get_player_latest_row(pp_name, df_history)
        if player_row is None:
            skipped.append(pp_name)
            continue

        # Rankings from Sackmann GitHub
        player_rank = rankings.get_rank(pp_name)
        tour        = rankings.get_tour(pp_name)
        is_atp      = 1 if tour == 'atp' else 0
        bo5         = 1 if (is_slam and is_atp) else 0

        preds = predict(player_row, models,
                        surface=surface, opp_rank=50.0,
                        player_rank=player_rank, is_best_of_5=bo5,
                        round_ordinal=round_ord, is_atp=is_atp)

        # For each market PrizePicks has for this player, create a bet
        for target, line in lines_for_player.items():
            proj = preds.get(target)
            if proj is None:
                continue

            rec       = get_betting_indicator(proj, line)
            edge      = proj - line
            pct_edge  = (edge / line) * 100
            tier_info = MODEL_QUALITY.get(target, {})

            all_projections.append({
                'REC':     rec,
                'NAME':    pp_name,
                'TARGET':  target,
                'SURFACE': surface,
                'AI':      round(proj, 2),
                'PP':      round(float(line), 2),
                'EDGE':    round(edge, 2),
                'RANK':    int(player_rank) if player_rank != 50.0 else '?',
            })

            best_bets.append({
                'REC':           rec,
                'NAME':          pp_name,
                'TARGET':        target,
                'TARGET_DISPLAY': _target_display_name(target),
                'SURFACE':       surface,
                'AI':            round(proj, 2),
                'PP':            round(float(line), 2),
                'EDGE':          edge,
                'PCT_EDGE':      pct_edge,
                'TIER_KEY':      tier_info.get('tier', 'UNKNOWN'),
                'TIER':          tier_info.get('emoji', '~') + ' ' + tier_info.get('tier', 'UNKNOWN'),
                'THRESHOLD':     tier_info.get('threshold', 2.5),
            })

    # --- Warn about missing players ---
    if skipped:
        unique_skipped = list(set(skipped))
        print(f"\n   ⚠️  {len(unique_skipped)} player(s) not in training history:")
        for n in unique_skipped[:5]:
            print(f"      - {n}")
        if len(unique_skipped) > 5:
            print(f"      ... and {len(unique_skipped)-5} more")

    if not best_bets:
        print("\n⚠️  No predictions generated.")
        input("\nPress Enter to continue...")
        return

    # --- Deduplicate ---
    seen, deduped = set(), []
    for bet in best_bets:
        key = (bet['NAME'], bet['TARGET'], bet['PP'])
        if key not in seen:
            seen.add(key)
            deduped.append(bet)

    removed = len(best_bets) - len(deduped)
    if removed:
        print(f"   🧹 Removed {removed} duplicate entries")

    # --- Sort: tier first, then edge % ---
    tier_order = {'ELITE': 0, 'STRONG': 1, 'DECENT': 2, 'RISKY': 3}
    deduped.sort(key=lambda x: (tier_order.get(x['TIER_KEY'], 9), -abs(x['PCT_EDGE'])))

    overs_raw  = [b for b in deduped if b['EDGE'] > 0]
    unders_raw = [b for b in deduped if b['EDGE'] < 0]

    # Ensure market variety: take best per market first, then fill to 10 (like NBA shows diverse stats)
    def _diverse_top(bets, n=10):
        if len(bets) <= n:
            return bets[:n]
        by_market = {}
        for b in bets:
            t = b['TARGET']
            if t not in by_market or abs(b['PCT_EDGE']) > abs(by_market[t]['PCT_EDGE']):
                by_market[t] = b
        result = list(by_market.values())
        seen = {(x['NAME'], x['TARGET'], x['PP']) for x in result}
        for b in bets:
            key = (b['NAME'], b['TARGET'], b['PP'])
            if key not in seen and len(result) < n:
                result.append(b)
                seen.add(key)
        result.sort(key=lambda x: (tier_order.get(x['TIER_KEY'], 9), -abs(x['PCT_EDGE'])))
        return result[:n]

    top_overs  = _diverse_top(overs_raw)
    top_unders = _diverse_top(unders_raw)

    # Table format — match NBA scanner (see nba/scanner.py lines 430-441)
    print("\n🔥 TOP 10 OVERS (Highest Value)")
    print()
    print(f" {'TIER':<12} | {'PLAYER':<20} | {'MARKET':<18} | {'AI vs PP':<15} | {'EDGE %':<8}")
    print("-" * 86)
    for b in top_overs:
        mkt = b.get('TARGET_DISPLAY', _target_display_name(b['TARGET']))
        print(f" {b['TIER']:<12} | {b['NAME'][:20]:<20} | {mkt:<18} | "
              f"{b['AI']:>6.2f} vs {b['PP']:>6.2f} | {b['PCT_EDGE']:>6.1f}%")

    print("\n❄️ TOP 10 UNDERS (Lowest Value)")
    print()
    print(f" {'TIER':<12} | {'PLAYER':<20} | {'MARKET':<18} | {'AI vs PP':<15} | {'EDGE %':<8}")
    print("-" * 86)
    for b in top_unders:
        mkt = b.get('TARGET_DISPLAY', _target_display_name(b['TARGET']))
        print(f" {b['TIER']:<12} | {b['NAME'][:20]:<20} | {mkt:<18} | "
              f"{b['AI']:>6.2f} vs {b['PP']:>6.2f} | {b['PCT_EDGE']:>6.1f}%")

    # --- Save ---
    save_path = os.path.join(PROJ_DIR, f"scan_{actual_date}.csv")
    pd.DataFrame(all_projections).to_csv(save_path, index=False)
    print(f"\n✅  Full analysis ({len(all_projections)} rows) saved to {save_path}")

    input("\nPress Enter to continue...")


# ---------------------------------------------------------------------------
# SCOUT SPECIFIC PLAYER
# ---------------------------------------------------------------------------

def scout_player(df_history, models, rankings: TennisRankings):
    print("\n🔎 --- TENNIS PLAYER SCOUT ---")

    d_choice = input("Scan date (1=Today, 2=Tomorrow): ").strip()
    offset   = 1 if d_choice == '2' else 0

    pp_board, actual_date = get_pp_lines(date_offset=offset, max_days_forward=7)

    if actual_date is None:
        print("❌  No tennis lines found.")
        return

    scan_date_obj = datetime.strptime(actual_date, '%Y-%m-%d')
    print(f"\n📅 Scouting for: {scan_date_obj.strftime('%A, %B %d, %Y')}")

    # Derive surface from PP board tournament
    tournament_name = ''
    if not pp_board.empty and 'League' in pp_board.columns:
        leagues = pp_board['League'].dropna().unique()
        if len(leagues) > 0:
            tournament_name = str(leagues[0])

    surface  = rankings.get_surface(tournament_name)
    is_slam  = rankings.is_slam(tournament_name)

    # Build PP line lookup: normalized_name → {target: line} (all markets per player)
    pp_lines_lookup = {}
    if not pp_board.empty:
        for _, row in pp_board.iterrows():
            norm   = normalize_name(row.get('Player', ''))
            target = _map_pp_stat_to_target(row.get('Stat', ''))
            line   = row.get('Line')
            if norm and target and line:
                pp_lines_lookup.setdefault(norm, {})[target] = float(line)

    scouting = True
    while scouting:
        print("\n(Type '0' to return to menu)")
        query = input("Enter player name: ").strip().lower()
        if query == '0':
            break
        if not query:
            continue

        # Search history
        mask       = df_history['player_name'].apply(lambda n: query in normalize_name(n))
        matches_df = df_history[mask]

        if matches_df.empty:
            print(f"❌  No player found matching '{query}'.")
            continue

        unique = matches_df['player_name'].drop_duplicates().tolist()
        if len(unique) > 1:
            print("\nMultiple matches:")
            for i, name in enumerate(unique[:10], 1):
                print(f"  {i}. {name}")
            try:
                chosen_name = unique[int(input("Select number: ")) - 1]
            except (ValueError, IndexError):
                print("❌  Invalid.")
                continue
        else:
            chosen_name = unique[0]

        player_row  = df_history[df_history['player_name'] == chosen_name].iloc[-1]
        player_rank = rankings.get_rank(chosen_name)
        tour        = rankings.get_tour(chosen_name)
        is_atp      = 1 if tour == 'atp' else 0
        bo5         = 1 if (is_slam and is_atp) else 0
        pp_lines    = pp_lines_lookup.get(normalize_name(chosen_name), {})

        print(f"\n{'='*62}")
        print(f"📊 SCOUTING: {chosen_name}  ({tour.upper()})")
        print(f"   Date:       {actual_date}")
        print(f"   Surface:    {surface}")
        print(f"   World Rank: #{int(player_rank) if player_rank != 50.0 else 'Unknown'}")
        if tournament_name:
            print(f"   Tournament: {tournament_name}")
        print(f"{'='*62}")
        print(f"{'TIER':<6} | {'MARKET':<22} | {'AI PROJ':>8} | {'PP LINE':>8} | CALL")
        print("-" * 62)

        preds = predict(player_row, models, surface=surface, opp_rank=50.0,
                        player_rank=player_rank, is_best_of_5=bo5,
                        round_ordinal=3, is_atp=is_atp)

        for target, proj in preds.items():
            tier_emoji = MODEL_QUALITY.get(target, {}).get('emoji', '?')
            line       = pp_lines.get(target)
            rec        = get_betting_indicator(proj, line)
            line_str   = f"{line:.2f}" if line else "N/A"
            mkt        = _target_display_name(target)
            print(f"{tier_emoji:<6} | {mkt:<22} | {proj:>8.2f} | {line_str:>8} | {rec}")

        print(f"{'='*62}")

        if input("\nScout another player? (y/n): ").strip().lower() != 'y':
            scouting = False


# ---------------------------------------------------------------------------
# GRADE RESULTS
# ---------------------------------------------------------------------------

def grade_results():
    print("\n📅 GRADING OPTIONS:")
    print("1. Grade TODAY's results")
    print("2. Grade YESTERDAY's results")
    print("3. Grade a specific date")
    choice = input("Select (1/2/3): ").strip()

    if choice == '1':
        target_date = datetime.now()
    elif choice == '2':
        target_date = datetime.now() - timedelta(days=1)
    elif choice == '3':
        try:
            target_date = datetime.strptime(input("Date (YYYY-MM-DD): ").strip(), '%Y-%m-%d')
        except ValueError:
            print("❌  Invalid date."); return
    else:
        print("❌  Invalid choice."); return

    date_str  = target_date.strftime('%Y-%m-%d')
    scan_file = os.path.join(PROJ_DIR, f"scan_{date_str}.csv")

    print(f"\n📅 Grading: {date_str}")

    if not os.path.exists(scan_file):
        print(f"❌  No scan file: {scan_file}")
        print("    Run a scan for this date first.")
        input("\nPress Enter to continue..."); return

    try:
        df_preds = pd.read_csv(scan_file)
    except Exception as e:
        print(f"❌  {e}"); return

    # --- Fetch actuals from PrizePicks results (parse score from PP board) ---
    # Since we don't have a results API, we prompt the user to enter actuals
    # for grading, OR we use the PP board's settled lines if available.
    print("\n⚠️  Grading tennis results requires actual match stats.")
    print("   Automatic grading is available for: total_games, games_won, total_sets, total_tiebreaks")
    print("   Aces/double faults/break points require manual entry.\n")

    actuals = _fetch_actuals_from_sackmann(date_str)

    if not actuals:
        print("   Could not automatically fetch results.")
        print("   Tip: Re-run builder.py after matches complete to update data,")
        print("        then retry grading.")
        input("\nPress Enter to continue..."); return

    # --- Grade ---
    results      = []
    total_graded = 0
    correct      = 0

    for _, row in df_preds.iterrows():
        if not row.get('PP') or float(row.get('PP', 0)) <= 0:
            continue
        name   = normalize_name(row['NAME'])
        target = row['TARGET']
        line   = float(row['PP'])
        rec    = row.get('REC', '')

        if name not in actuals:
            continue
        actual_val = actuals[name].get(target)
        if actual_val is None:
            continue

        pick = 'OVER' if 'OVER' in rec else ('UNDER' if 'UNDER' in rec else None)
        if not pick:
            continue

        margin  = (actual_val - line) if pick == 'OVER' else (line - actual_val)
        is_win  = (actual_val > line and pick == 'OVER') or (actual_val < line and pick == 'UNDER')
        is_push = actual_val == line

        if not is_push:
            total_graded += 1
            if is_win: correct += 1
            results.append({'Player': row['NAME'], 'Stat': target, 'StatDisplay': _target_display_name(target),
                            'Pick': pick, 'Line': line, 'Actual': actual_val, 'Margin': margin, 'Win': is_win})

    if total_graded == 0:
        print("⚠️  No settled predictions found.")
        input("\nPress Enter to continue..."); return

    sorted_r   = sorted(results, key=lambda x: x['Margin'], reverse=True)
    top_wins   = [r for r in sorted_r if r['Win']][:5]
    top_losses = sorted([r for r in sorted_r if not r['Win']], key=lambda x: x['Margin'])[:5]

    print(f"\n{'='*65}")
    print(f"📋 RESULTS: {date_str}")
    print(f"{'='*65}")
    print(f"\n🏆 TOP WINS")
    print()
    print(f"{'PLAYER':<22} | {'MARKET':<18} | {'LINE':>5} | {'ACTUAL':>6} | MARGIN")
    print("-" * 65)
    for r in top_wins:
        mkt = r.get('StatDisplay', _target_display_name(r['Stat']))
        print(f"{r['Player'][:21]:<22} | {mkt:<18} | {r['Line']:>5} | {r['Actual']:>6} | 🟢 +{r['Margin']:.2f}")

    print(f"\n💀 TOP LOSSES")
    print()
    for r in top_losses:
        mkt = r.get('StatDisplay', _target_display_name(r['Stat']))
        print(f"{r['Player'][:21]:<22} | {mkt:<18} | {r['Line']:>5} | {r['Actual']:>6} | 🔴 -{abs(r['Margin']):.2f}")

    accuracy = (correct / total_graded) * 100
    print(f"\n{'='*65}")
    print(f"📊 ACCURACY: {accuracy:.1f}% ({correct}/{total_graded})")

    log_exists = os.path.exists(ACCURACY_LOG_FILE)
    with open(ACCURACY_LOG_FILE, 'a') as f:
        if not log_exists:
            f.write("Date,Graded,Correct,Accuracy,Best_Win\n")
        best = top_wins[0]['Margin'] if top_wins else 0
        f.write(f"{date_str},{total_graded},{correct},{accuracy:.2f},{best:.2f}\n")
    print(f"✅  Logged to {ACCURACY_LOG_FILE}")

    input("\nPress Enter to continue...")


def _fetch_actuals_from_sackmann(date_str):
    """
    Try to load match results from already-downloaded Sackmann raw data.
    Works for dates within the downloaded range.
    Returns: {normalized_player: {target: value}}
    """
    import glob

    actuals = {}
    raw_dir = os.path.join(BASE_DIR, 'data', 'tennis', 'raw')
    files   = glob.glob(os.path.join(raw_dir, '*_raw_matches.csv'))

    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
            day_matches = df[df['tourney_date'].dt.strftime('%Y-%m-%d') == date_str]

            for _, row in day_matches.iterrows():
                name = normalize_name(str(row.get('player_name', '')))
                if not name:
                    continue
                actuals[name] = {
                    'total_games':     int(row.get('total_games',     0) or 0),
                    'games_won':       int(row.get('games_won',       0) or 0),
                    'total_sets':      int(row.get('total_sets',      0) or 0),
                    'total_tiebreaks': int(row.get('total_tiebreaks', 0) or 0),
                    'aces':            int(row.get('aces',            0) or 0),
                    'double_faults':   int(row.get('double_faults',   0) or 0),
                    'bp_won':          int(row.get('bp_won',          0) or 0),
                }
        except Exception:
            continue

    return actuals


# ---------------------------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------------------------

def main():
    print("...Initializing Tennis Scanner")
    df_history = load_data()
    models     = load_models()

    if df_history is None or not models:
        print("❌  Setup failed. Run builder → features → train first.")
        return

    # Load rankings once at startup
    print("...Loading Rankings")
    rankings = TennisRankings()
    rankings.load()

    while True:
        print("\n" + "="*35)
        print("   🎾 TENNIS AI SCANNER")
        print("="*35)
        print("1. 🚀 Scan TODAY's Matches")
        print("2. 🔮 Scan TOMORROW's Matches")
        print("3. 📋 Grade Results")
        print("4. 🔎 Scout Specific Player")
        print("0. 🚪 Exit")

        choice = input("\nSelect: ").strip()

        if   choice == '1': scan_all(df_history, models, rankings, is_tomorrow=False)
        elif choice == '2': scan_all(df_history, models, rankings, is_tomorrow=True)
        elif choice == '3': grade_results()
        elif choice == '4': scout_player(df_history, models, rankings)
        elif choice == '0': break
        else:
            print("❌  Invalid selection.")


if __name__ == "__main__":
    main()