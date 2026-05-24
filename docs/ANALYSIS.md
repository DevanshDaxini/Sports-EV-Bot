# Codebase Analysis — sports_ev_bot

**Date:** 2026-04-23  
**Branch:** `ruflo-improvements`  
**Scope:** All Python source files in `src/`

---

## 1. Architecture Overview

### Data Flow

```
builder.py          → Raw game logs from nba_api + 1H box scores
features.py         → Feature engineering (220+ features)
train.py            → XGBoost training (21 targets, tiered hyperparams)
scanner.py          → Inference + display
backtester.py       → Walk-forward validation against L10/L20 medians
grader.py           → Post-hoc accuracy grading from actual results
```

### Entry Points

```
main.py → nba_cli.py → Tool 1: Super Scanner  (FD math + AI combined)
                     → Tool 2: Odds Scanner   (FanDuel vs PrizePicks devig only)
                     → Tool 3: AI Scanner     (XGBoost-only, calls scanner.main())
                     → Setup: Build / Engineer / Train / Backtest / Grade
       → tennis_cli.py → analogous flow
```

### External Dependencies

| Source | Used For | Caching |
|--------|----------|---------|
| The Odds API (FanDuel) | Reference market odds + devig | 30-min disk cache (per-sport file) |
| PrizePicks partner API | Lines to bet against | 2-min disk cache (single shared file) |
| ESPN + CBS Sports | Injury report scraping | None (live per scan) |
| nba_api | Historical + live game logs | training_dataset.csv on disk |
| Sackmann GitHub (tennis) | Rankings + match history | training_dataset.csv on disk |

### Model Architecture

- 21 separate XGBoost regressors (one per stat target)
- 3-tier hyperparameter set (HIGH/MEDIUM/LOW signal stats)
- Log-transform for zero-inflated targets: `{BLK, STL, TOV, FG3M, SB}`
- Empirical calibration factors correct Jensen's inequality bias post-expm1
- Directional accuracy measured vs player-specific L10 median (not global)
- 70/30 chronological train/test split

### Prediction Pipeline (scan_all)

1. Fetch injury report (ESPN + CBS)
2. Fetch PrizePicks live lines
3. Load FanDuel cache for line-diff validation
4. `build_data_cache()` → O(1) per-player latest-row and roster maps
5. Pre-compute fresh opponent defensive stats + pace for today's matchups
6. Per player: inject features → model inference → log-inverse transform
7. Apply injury/minute-restriction scale factors
8. Apply correlation constraints (average combo-stat model with component sum)
9. Filter by edge threshold, L10 veto, H2H veto, FD line-diff cap
10. Score and rank with composite quality formula → display top 15

---

## 2. Code Quality Issues

### 2.1 Duplicate `ensure_combo_stats` Function

`src/sports/nba/features.py` defines `ensure_combo_stats` **twice**:
- Lines 121–127 (used in `main()`)
- Lines 668–675 (identical definition, unused directly)

One copy should be deleted. If the second is needed by something external it should be the only definition.

### 2.2 Stage 3 Header Printed Twice

`features.py` `main()` lines 795–796:
```python
print("\n--- STAGE 3: HISTORICAL FEATURES ---")
print("\n--- STAGE 3: HISTORICAL FEATURES ---")  # duplicate
```

### 2.3 Identical `drop_duplicates` Call in nba_cli.py

`nba_cli.py` lines 351–352 calls the same `drop_duplicates` twice with identical arguments. Dead code.

### 2.4 `config.py` Prints at Import Time

`src/sports/nba/config.py` lines 159–160 print to stdout unconditionally when the module is imported:
```python
print(f"⚙️  {mode_descriptions.get(SCANNING_MODE, 'UNKNOWN MODE')}")
print(f"   Scanning: {', '.join(ACTIVE_TARGETS)}")
```
Every `from src.sports.nba.config import ...` statement causes terminal output. This breaks any library usage and makes test output noisy.

### 2.5 `LOG_TRANSFORM_TARGETS` Duplicated

Defined identically in both `train.py` (line 42) and `backtester.py` (line 47). `backtester.py` already imports `get_features_for_target` and `TARGETS` from `train.py`; `LOG_TRANSFORM_TARGETS` should be added to that import. If a new log-transform target is added to `train.py` without updating `backtester.py`, backtester silently uses wrong inverse transforms.

### 2.6 `tune_train.py` Feature List Is Stale

`tune_train.py` defines its own `FEATURES` list that is missing:
- `_L5_Median`, `_L10_Median` rolling variants (added in v2.2)
- `OPP_*_ALLOWED_DIFF` columns
- Most contextual features added by `features.py` stages 4–6

Hyperparameter tuning on this reduced feature space produces params that were not validated on the production feature set.

### 2.7 Bare `except` Clauses Swallow Errors

`builder.py` retry loops catch all exceptions and suppress them after the final attempt without logging the error reason:
```python
except Exception as e:
    if attempt < max_retries - 1:
        time.sleep(2)
    else:
        print(f"❌ Error fetching {season} after {max_retries} attempts.")  # e never printed
```
The actual exception `e` is available but not printed. Debugging silent failures requires adding print statements.

### 2.8 Global Mutable State in scanner.py

`INJURY_DATA = {}` and `_QUIET = False` are module-level globals mutated by `refresh_injuries()` and `scan_all()`. This is thread-unsafe and makes the module hard to test. `_QUIET` is set inside `scan_all` but never reset if the function raises, permanently silencing output.

### 2.9 `valid_input` Scope Leak in scan_all

`scanner.py` ~line 1530:
```python
if 'OPP_WIN_PCT' in valid_input.columns:
    opp_win_pct = valid_input['OPP_WIN_PCT'].iloc[0]
```
`valid_input` is defined inside `for target, model in models.items()` (line 1414). After the loop, this references the **last model's** input DataFrame from the previous loop iteration. If `models` is empty for a player (all features OK but no models loaded), `valid_input` could reference a different player's stale data. Works only because all models share the same player input row.

### 2.10 Fragile Index Manipulation in features.py

`add_missing_player_context` and `add_usage_vacuum_features` both do:
```python
sorted_idx = df.index
df = df.reset_index(drop=True)
# ... operations including merge ...
df.index = sorted_idx   # Only safe if row count hasn't changed
```
The index restoration after a merge is only safe because `how='left'` with unique right keys preserves row count. There's no assertion to enforce this invariant, making refactoring this code fragile.

---

## 3. Potential Bugs

### 3.1 Grader Cannot Grade AI Scanner Output (High Severity)

The AI Scanner (`scanner.scan_all`) saves projections to:
```
data/nba/projections/scan_{date}.csv
```
With columns: `REC, NAME, TARGET, AI, PP, EDGE, L5_HIT, ...`

The Grader (`grader.py`) looks for files in:
```
output/nba/scans/scan_{date}.csv
```
And requires columns: `Player, Stat, Line, Side`

These paths and column names are **completely incompatible**. The grader can only grade output from the Odds Scanner (Tool 2), which saves to `output/nba/scans/` with the right schema. This is undocumented. Any user running Tool 3 (AI Scanner) and trying to grade results will silently get `{}` from `grade_file` and no grades written.

### 3.2 PrizePicks Cross-League Cache Contamination (High Severity)

`prizepicks.py` uses a single hardcoded cache file for all leagues:
```python
CACHE_FILE = os.path.join(CACHE_DIR, 'prizepicks_cache.json')
```
If an NBA scan runs first (fetching with `?league_id[]=7&league_id[]=84&league_id[]=192`), the cache contains only NBA lines. A Tennis scan within the 2-minute TTL loads this NBA-only cache, passes it through `_apply_filters` with `league_filter='TENNIS'`, and returns an empty DataFrame — producing zero tennis projections without any clear error.

FanDuel's client correctly uses per-sport cache files (`fanduel_cache_{sport_tag}.json`). PrizePicks should follow the same pattern.

### 3.3 DST Bug in fanduel.py (Medium Severity)

```python
dt_est = dt_utc - timedelta(hours=5)  # hardcoded EST offset
```
This is correct for EST (UTC-5, Nov–Mar) but wrong for EDT (UTC-4, Mar–Nov). During daylight saving time, an 8pm EDT game broadcasts at midnight UTC, which `-5` maps to 7pm (same date, wrong timezone), and an unusual early-evening UTC game could map to the wrong calendar date. Use `zoneinfo.ZoneInfo('America/New_York')` or `pytz.timezone('US/Eastern')` instead.

### 3.4 Super Scanner Uses Stale Data (Medium Severity)

`nba_cli.py` `get_ai_predictions()` calls `load_data()` which reads `training_dataset.csv` directly. It does **not** call `auto_refresh_data()`. Tool 1 (Super Scanner) therefore uses potentially stale historical features for AI projections even when fresh game logs are available. Tool 3 (`ai_scanner_module.main()`) correctly calls `auto_refresh_data` before inference.

### 3.5 `analyze_player_availability` Can Miss Missed Games

`analyze_player_availability` computes missed team games as:
```python
team_dates = df_history[df_history['TEAM_ID'] == team_id]['GAME_DATE'].drop_duplicates()
team_games_missed_now = len(team_dates[(team_dates > last_game_date) & (team_dates < scan_date)])
```
It uses `TEAM_ID` from the **last row of the player's history**. If a player was recently traded, `TEAM_ID` on their last row is the new team. The function will then compare their absence against the new team's schedule, producing incorrect missed-game counts for the trade window.

### 3.6 `add_blocks_specific_features` vs. `add_defense_vs_position` Double-Counting

`add_defense_vs_position` computes `OPP_BLK_ALLOWED` (opponent's block allowance). `add_blocks_specific_features` computes `OPP_PAINT_SHOTS` and `OPP_RIM_ATTACK_RATE` separately. Both are fed into the BLK model. The DvP feature and the rim-attack features capture overlapping information (opponent's rim-attack tendency), which can confuse the model. This may contribute to BLK remaining at ~35% directional accuracy despite specialized features.

### 3.7 `auto_refresh_data` Always Fetches Full Season

When data is stale, `auto_refresh_data` always calls:
```python
logs = playergamelogs.PlayerGameLogs(season_nullable='2025-26', ...)
```
There is no `date_from_nullable` filter. This fetches all 2025-26 season logs for all players regardless of how many games are actually missing (could be 1 day or 50 days stale). The API rate limit and timeout are the only guards.

---

## 4. Performance Bottlenecks

### 4.1 O(N) Hit-Rate Computation Per Player × Stat (Highest Impact)

Inside `scan_all`, for every player × stat combination:
```python
l5_hit, l10_hit, l20_hit = calculate_hit_rates(df_history, pid, target, line)
h2h_hit, h2h_n = calculate_h2h_hit_rate(df_history, pid, target, line, opp_abbr)
```
Each call does `df_history[df_history['PLAYER_ID'] == pid]` — a full linear scan of df_history (~200K rows). With ~500 players × 21 targets = **10,500 O(N) filter operations per scan**. The `build_data_cache()` step already builds O(1) lookups for the latest row; hit rates should be pre-computed there.

### 4.2 Per-Player Rolling Recalculation in auto_refresh_data

`auto_refresh_data` loops over all updated players and recomputes rolling features:
```python
for pid in updated_pids:         # up to ~500 players
    for stat in base_stats:      # ~35 stats
        combined.loc[mask, f'{stat}_L5'] = vals.rolling(5, ...).mean().values
        ...  # 6 rolling variants per stat
```
This is ~500 × 35 × 6 = **105,000 individual rolling window computations** done in Python-level loops. Vectorizing with `groupby.transform` across all players at once would be 10–50x faster.

### 4.3 `add_schedule_density` Loop Over All Players

```python
for player_id, group in df.groupby('PLAYER_ID'):
    counts = get_rolling_count(group)
    games_7d_list.extend(counts)
df['GAMES_7D'] = games_7d_list
```
This is a Python-level for loop over all players building a list. This breaks if `df.groupby` iteration order doesn't match the original DataFrame order. The pandas rolling-on-DatetimeIndex approach can be vectorized with `groupby().transform()`.

### 4.4 `groupby().apply()` in Feature Engineering Functions

`add_rebound_specific_features`, `add_blocks_specific_features`, and similar functions use:
```python
df.groupby(['OPPONENT', 'SEASON_ID']).apply(
    lambda x: (x['FGA'] - x['FG3A']).shift(1).rolling(10, ...).mean()
).reset_index(level=[0, 1], drop=True)
```
`apply()` with a lambda processes each group sequentially in Python. `transform()` or pre-aggregating then merging back would be substantially faster, especially on full season datasets.

### 4.5 FanDuel API Sleep Between Games

`fanduel.py` `get_all_odds` sleeps `0.5s` between each game's props fetch:
```python
time.sleep(0.5)
```
With 15 games this adds 7.5 seconds of idle waiting. The Odds API allows slightly faster polling; `0.25s` or a concurrent fetch with `asyncio` would reduce wait time.

### 4.6 features.py Full Rebuild Every Run

The entire feature pipeline processes all ~200K rows on every run, even if only the last few days of games have changed. There is no incremental or partial-rebuild path. This is acceptable for scheduled overnight runs but makes iterative development slow (~90–180 seconds per feature change test).

---

## 5. Improvement Recommendations (Ranked by Impact)

### Priority 1 — Critical Bugs (fix immediately)

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | PrizePicks cross-league cache contamination | `prizepicks.py` | Use per-league cache filenames matching FanDuel's pattern (`prizepicks_cache_{league}.json`). Key by `league_filter` in `_load_cache`/`_save_cache`. |
| 2 | Grader can't grade AI Scanner output | `grader.py`, `scanner.py` | Either: (a) have scanner save grader-compatible CSV with `Player/Stat/Line/Side` columns to `output/nba/scans/`; or (b) add a column-remapping step in `grade_file`. Document which tool's output the grader expects. |
| 3 | Super Scanner uses stale data | `nba_cli.py:get_ai_predictions` | Call `df_history = auto_refresh_data(df_history)` after `load_data()` to match Tool 3 behavior. |
| 4 | DST bug in FanDuel date parsing | `fanduel.py` | Replace `timedelta(hours=5)` with `pytz.timezone('US/Eastern')` or `zoneinfo.ZoneInfo('America/New_York')` conversion. |

### Priority 2 — High-Impact Correctness

| # | Issue | File | Fix |
|---|-------|------|-----|
| 5 | `LOG_TRANSFORM_TARGETS` duplicated | `backtester.py` | Import from `train.py`: add to the existing `from src.sports.nba.train import ...` line. |
| 6 | `tune_train.py` stale feature list | `tune_train.py` | Replace the hardcoded `FEATURES` list with a call to `get_features_for_target(target)` from `train.py`. |
| 7 | `ensure_combo_stats` defined twice | `features.py` | Delete lines 668–675 (the second identical definition). |
| 8 | `valid_input` scope leak | `scanner.py` ~1530 | Extract `opp_win_pct` inside the model loop and carry it forward, or read it from `last_row.get('OPP_WIN_PCT')` directly. |
| 9 | Traded-player availability detection bug | `scanner.py:analyze_player_availability` | Look up team schedule using all teams the player has played for in the current season, not just their last TEAM_ID. |

### Priority 3 — Performance (run-time improvements)

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 10 | Hit-rate O(N) per player×stat | High (10K+ redundant filters per scan) | Pre-compute L5/L10/L20/H2H hit rates in `build_data_cache` using vectorized groupby. |
| 11 | `auto_refresh_data` per-player rolling loops | High (100K ops) | Replace Python for-loops with `combined.groupby('PLAYER_ID')[stat].transform(lambda x: x.rolling(...).mean())` for each stat. |
| 12 | `add_schedule_density` for-loop | Medium | Use `df.groupby('PLAYER_ID')['GAME_DATE'].transform(lambda x: x.expanding().apply(...))` or resample approach. |
| 13 | `groupby().apply()` in feature functions | Medium | Replace with `groupby().transform()` for all rolling-window calculations. |
| 14 | `auto_refresh_data` fetches full season | Low-Medium | Add `date_from_nullable=(latest_date + timedelta(days=1)).strftime('%m/%d/%Y')` to the nba_api call to fetch only new games. |

### Priority 4 — Maintainability

| # | Issue | File | Fix |
|---|-------|------|-----|
| 15 | scanner.py is 2,472 lines | `scanner.py` | Split into: `scanner_data.py` (load/refresh), `scanner_features.py` (hit rates, availability), `scanner_inference.py` (prediction loop), `scanner_display.py` (output formatting). |
| 16 | config.py side effects at import | `config.py` | Move the two `print()` statements into a dedicated `print_mode_info()` function called explicitly from CLI entry points. |
| 17 | Duplicate `drop_duplicates` call | `nba_cli.py` lines 351–352 | Delete one of the two identical calls. |
| 18 | Duplicate Stage 3 header | `features.py` ~796 | Delete one of the two `print("\n--- STAGE 3: HISTORICAL FEATURES ---")` lines. |
| 19 | Bare exceptions in builder.py | `builder.py` | Change `print(f"❌ Error fetching {season}...")` to `print(f"❌ Error fetching {season}: {e}")` so failures are debuggable. |
| 20 | Position file never refreshed mid-season | `builder.py:fetch_player_positions` | Remove the `if os.path.exists(POSITION_FILE): return` early exit, or add a `force_refresh` parameter for trades. |

---

## 6. Model Quality Notes (non-code)

- **BLK (35% directional accuracy)** is below the 53.5% break-even at any edge level. The config correctly classifies it as RISKY/SPECULATIVE. Adding more block-specific features has diminishing returns because block events are inherently high-variance Bernoulli trials with a low mean (0–2 per game).
- **The LINE_ADJUSTMENT_FACTORS in analyzer.py are hardcoded** (e.g. `'PTS': 0.035`). These are described as "empirically derived" but there is no derivation script. A calibration tool that fits these factors from the historical scan/grade CSV pairs would make the analyzer self-improving.
- **The 70/30 split in backtester.py is identical to train.py**. The test set used for backtesting is the same set used for early stopping in training. For unbiased backtest estimates, a held-out third partition or genuine out-of-sample period (e.g. the current season) should be used.
- **Correlation constraints** (averaging `PRA` model with `PTS+REB+AST`) artificially pull combo-stat predictions toward their components. If the PRA model is well-calibrated independently, this averaging introduces systematic bias. Consider removing it and relying on per-target calibration instead.

---

## 7. Summary Statistics

| File | Lines | Key Concern |
|------|-------|-------------|
| `scanner.py` | 2,472 | Monolith; needs split |
| `features.py` | 846 | Duplicate function; slow `.apply()` |
| `nba_cli.py` | ~500 | Stale data in Super Scanner |
| `analyzer.py` | 378 | Hardcoded calibration factors |
| `backtester.py` | 267 | Not truly independent of training split |
| `train.py` | 266 | Good overall |
| `grader.py` | 279 | Wrong path for AI Scanner output |
| `prizepicks.py` | 473 | Cross-league cache bug |
| `fanduel.py` | 271 | DST bug |
| `builder.py` | 208 | Silent errors; positions never refreshed |
