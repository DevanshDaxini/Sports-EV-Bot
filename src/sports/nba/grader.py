"""
Prediction Accuracy Grader

Compares scanner predictions to actual game results and tracks win rate.
Supports single-date grading and retroactive batch grading of all ungraded files.

PrizePicks economics:
  Standard juice ≈ -115 on each side.
  Break-even win rate = 1.15 / (1 + 1.15) = 53.5%
  ROI per bet at -115: win → +$0.87, loss → -$1.00

Output:
    output/nba/scans/scan_YYYY-MM-DD.csv  (Result + Actual columns added)
    output/nba/scans/win_rate_history.csv (aggregate history)

Usage:
    # Single date:
    python3 -m src.sports.nba.grader

    # Grade every ungraded file automatically:
    python3 -m src.sports.nba.grader --all
"""

import os
import sys
import glob
import argparse
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelogs

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCANS_DIR = os.path.join(BASE_DIR, 'output', 'nba', 'scans')
PROJ_DIR  = os.path.join(BASE_DIR, 'data',   'nba', 'projections')

PP_PAYOUT   = 1.0 / 1.15        # $0.87 profit per $1 wagered on a win
BREAK_EVEN  = 1.15 / 2.15 * 100 # ~53.5%

NBA_STAT_MAP = {
    'Points': 'PTS', 'Rebounds': 'REB', 'Assists': 'AST',
    '3-PT Made': 'FG3M', '3-PT Attempted': 'FG3A',
    'Blocked Shots': 'BLK', 'Steals': 'STL', 'Turnovers': 'TOV',
    'FG Made': 'FGM', 'FG Attempted': 'FGA',
    'Free Throws Made': 'FTM', 'Free Throws Attempted': 'FTA',
    'Pts+Rebs+Asts': 'PRA', 'Pts+Rebs': 'PR', 'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA', 'Blks+Stls': 'SB',
}


def normalize_name(name: str) -> str:
    name = str(name).lower().replace('.', '')
    for suffix in [' jr', ' sr', ' ii', ' iii', ' iv']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name.strip()


def _season_for_date(date_str: str) -> str:
    year = int(date_str[:4])
    month = int(date_str[5:7])
    season_year = year - 1 if month < 10 else year
    return f"{season_year}-{str(season_year + 1)[2:]}"


def fetch_box_scores(date_str: str) -> dict:
    """Fetch actual player stats from NBA API for a given date."""
    season = _season_for_date(date_str)
    logs   = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        date_from_nullable=date_str,
        date_to_nullable=date_str
    )
    frames = logs.get_data_frames()
    if not frames or frames[0].empty:
        return {}

    box = frames[0]
    player_stats = {}
    for _, row in box.iterrows():
        d = row.to_dict()
        d['PRA'] = d['PTS'] + d['REB'] + d['AST']
        d['PR']  = d['PTS'] + d['REB']
        d['PA']  = d['PTS'] + d['AST']
        d['RA']  = d['REB'] + d['AST']
        d['SB']  = d['STL'] + d['BLK']
        player_stats[row['PLAYER_NAME']] = d
        norm = normalize_name(row['PLAYER_NAME'])
        if norm not in player_stats:
            player_stats[norm] = d
    return player_stats


def grade_file(filename: str, player_stats: dict) -> dict:
    """Grade one scan file in-place. Returns summary dict."""
    df = pd.read_csv(filename)

    # Remap AI Scanner columns (NAME/TARGET/PP/EDGE) to grader schema
    ai_cols = {'NAME', 'TARGET', 'PP', 'EDGE'}
    if ai_cols.issubset(df.columns) and 'Player' not in df.columns:
        df = df.rename(columns={'NAME': 'Player', 'TARGET': 'Stat', 'PP': 'Line'})
        df['Side'] = df['EDGE'].apply(lambda e: 'Over' if float(e) >= 0 else 'Under')

    required = {'Player', 'Stat', 'Line', 'Side'}
    if not required.issubset(df.columns):
        return {}

    wins = losses = pushes = 0
    results, actuals = [], []

    for _, row in df.iterrows():
        name = str(row['Player'])
        stat = str(row['Stat'])
        side = str(row['Side'])
        try:
            line = float(row['Line'])
        except (ValueError, TypeError):
            results.append('Bad Line'); actuals.append(0); continue

        player = player_stats.get(name) or player_stats.get(normalize_name(name))
        if not player:
            results.append('DNP/Unknown'); actuals.append(0); continue

        col = NBA_STAT_MAP.get(stat)
        if not col:
            results.append('Unsupported Stat'); actuals.append(0); continue

        actual = float(player.get(col, 0))
        actuals.append(actual)

        if actual == line:
            results.append('Push'); pushes += 1
        elif (side == 'Over' and actual > line) or (side == 'Under' and actual < line):
            results.append('WIN'); wins += 1
        else:
            results.append('LOSS'); losses += 1

    df['Result'] = results
    df['Actual'] = actuals
    df.to_csv(filename, index=False)

    total = wins + losses
    wr    = wins / total * 100 if total else 0
    roi   = (wins * PP_PAYOUT - losses) / total * 100 if total else 0

    return {'wins': wins, 'losses': losses, 'pushes': pushes, 'total': total, 'win_rate': wr, 'roi': roi}


def update_history(date_str: str, s: dict):
    history_path = os.path.join(SCANS_DIR, 'win_rate_history.csv')
    row = {
        'Date': date_str, 'Total_Bets': s['total'],
        'Wins': s['wins'], 'Losses': s['losses'],
        'Win_Rate': f"{s['win_rate']:.2f}%",
        'ROI_Pct': f"{s['roi']:+.2f}%",
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    os.makedirs(SCANS_DIR, exist_ok=True)
    if os.path.exists(history_path):
        try:
            hist = pd.read_csv(history_path)
            hist = hist[hist['Date'] != date_str]
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            hist = pd.DataFrame([row])
    else:
        hist = pd.DataFrame([row])
    hist.sort_values('Date').to_csv(history_path, index=False)


def print_report(date_str: str, s: dict, filename: str):
    print(f"\n  {date_str}: {s['wins']}W / {s['losses']}L / {s['pushes']} Push  "
          f"→ {s['win_rate']:.1f}% win rate  |  ROI {s['roi']:+.1f}%")
    if s['total'] > 0:
        arrow = "✓ PROFITABLE" if s['win_rate'] >= BREAK_EVEN else "✗ below break-even"
        print(f"    {arrow} (break-even = {BREAK_EVEN:.1f}%)")


def grade_single():
    while True:
        date_str = input("\nDate to grade (YYYY-MM-DD) or Enter for yesterday: ").strip()
        if not date_str:
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            break
        except ValueError:
            print("Invalid format.")

    filename = os.path.join(SCANS_DIR, f"scan_{date_str}.csv")
    if not os.path.exists(filename):
        # Fall back to AI Scanner projections directory
        filename = os.path.join(PROJ_DIR, f"scan_{date_str}.csv")
    if not os.path.exists(filename):
        print(f"ERROR: scan_{date_str}.csv not found in {SCANS_DIR} or {PROJ_DIR}")
        return

    print(f"Fetching NBA results for {date_str}...")
    player_stats = fetch_box_scores(date_str)
    if not player_stats:
        print("No NBA data returned.")
        return

    s = grade_file(filename, player_stats)
    if s:
        print_report(date_str, s, filename)
        update_history(date_str, s)


def grade_all_ungraded():
    """Grade every scan file that has no Result column yet."""
    files = sorted(set(
        glob.glob(os.path.join(SCANS_DIR, 'scan_20*.csv')) +
        glob.glob(os.path.join(PROJ_DIR,  'scan_20*.csv'))
    ))
    ungraded = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'Result' not in df.columns or df['Result'].isna().all():
                date_str = os.path.basename(f).replace('scan_', '').replace('.csv', '')
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    ungraded.append((date_str, f))
                except ValueError:
                    pass
        except Exception:
            pass

    if not ungraded:
        print("No ungraded files found.")
        return

    print(f"\nGrading {len(ungraded)} ungraded file(s):\n")

    all_wins = all_losses = all_pushes = 0
    for date_str, filename in ungraded:
        print(f"  Fetching {date_str}...", end='', flush=True)
        try:
            player_stats = fetch_box_scores(date_str)
            if not player_stats:
                print("  (no NBA data)")
                continue
            s = grade_file(filename, player_stats)
            if s:
                update_history(date_str, s)
                print_report(date_str, s, filename)
                all_wins   += s['wins']
                all_losses += s['losses']
                all_pushes += s['pushes']
        except Exception as e:
            print(f"  ERROR: {e}")

    total = all_wins + all_losses
    if total > 0:
        wr  = all_wins / total * 100
        roi = (all_wins * PP_PAYOUT - all_losses) / total * 100
        print(f"\n{'='*50}")
        print(f"  CUMULATIVE  {all_wins}W / {all_losses}L / {all_pushes} Push")
        print(f"  Win Rate:   {wr:.1f}%  (break-even = {BREAK_EVEN:.1f}%)")
        print(f"  ROI:        {roi:+.1f}% per $1 bet")

        # Per-stat breakdown
        all_files = sorted(glob.glob(os.path.join(SCANS_DIR, 'scan_20*.csv')))
        dfs = []
        for f in all_files:
            try:
                d = pd.read_csv(f)
                if 'Result' in d.columns and 'Stat' in d.columns:
                    dfs.append(d)
            except Exception:
                pass
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            settled  = combined[combined['Result'].isin(['WIN', 'LOSS'])]
            if not settled.empty:
                print(f"\n  Per-stat breakdown:")
                for stat, grp in settled.groupby('Stat'):
                    w = (grp['Result'] == 'WIN').sum()
                    t = len(grp)
                    print(f"    {stat:<25} {w}/{t}  ({w/t*100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Grade all ungraded scan files')
    args = parser.parse_args()

    if args.all:
        grade_all_ungraded()
    else:
        grade_single()
