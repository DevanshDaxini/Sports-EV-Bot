# Sports EV Bot

A multi-sport prop betting analysis engine. Uses XGBoost + LightGBM ensemble models trained on historical data to project player stats, then compares those projections against PrizePicks lines and FanDuel odds to find positive expected value (+EV) opportunities.

Supports NBA (21 stat models) and Tennis (7 market models).

---

## How It Works

Three independent analysis layers, usable alone or combined:

### 1. AI Scanner

Trains XGBoost + LightGBM ensemble models on historical player data to predict stat lines. For NBA, uses nba_api game logs with 451 engineered features per player-game (rolling averages, streak/consistency signals, opponent-allowed stats by position, usage rate, expected possessions, rest days, home/away splits). For Tennis, uses Jeff Sackmann's open match data with 150+ features (surface, rank, H2H, recent form, fatigue).

Each model outputs a projected stat value. The scanner compares this projection to the PrizePicks line and shows the delta.

### 2. Odds Scanner

Fetches live odds from FanDuel via the-odds-api and compares them to PrizePicks lines. Removes the vig from FanDuel two-way markets to calculate the true implied probability, then adjusts for any line difference between platforms using logarithmic scaling. Dynamic per-stat thresholds control how large a line discrepancy is allowed before filtering it out.

Output: a ranked list of plays sorted by implied win percentage.

### 3. Super Scanner

Combines layers 1 and 2. Finds plays where both the AI projection and the FanDuel-implied probability agree on the same side (Over or Under). A combined confidence score weights the math edge and AI edge equally, adjusted by stat volatility. Only plays where both signals agree are surfaced.

---

## Project Structure

```
sports_ev_bot/
├── main.py                          # entry point, sport selection menu
├── src/
│   ├── cli/
│   │   ├── nba_cli.py               # NBA menu (super/odds/ai scanner)
│   │   └── tennis_cli.py            # Tennis menu (scanner + setup tools)
│   ├── core/
│   │   ├── analyzers/
│   │   │   └── analyzer.py          # PropsAnalyzer: vig removal, line diff adjustment
│   │   ├── odds_providers/
│   │   │   ├── fanduel.py           # FanDuel client (the-odds-api)
│   │   │   └── prizepicks.py        # PrizePicks client (partner API)
│   │   ├── config.py                # shared cross-sport config
│   │   ├── utils.py                 # shared utilities
│   │   └── visualizer.py            # accuracy plot generation
│   └── sports/
│       ├── nba/
│       │   ├── builder.py           # download game logs + 1H splits via nba_api
│       │   ├── features.py          # engineer 451 training features
│       │   ├── train.py             # train 21 XGBoost+LightGBM ensemble models
│       │   ├── scanner.py           # AI scanner, player scout, game scanning
│       │   ├── injuries.py          # live injury scraper (ESPN + CBS)
│       │   ├── grader.py            # backtest grading
│       │   ├── backtester.py        # historical backtesting
│       │   ├── config.py            # NBA-specific constants
│       │   └── mappings.py          # stat name normalization maps
│       └── tennis/
│           ├── builder.py           # download ATP/WTA data from Sackmann GitHub
│           ├── features.py          # engineer 150+ training features
│           ├── train.py             # train 7 XGBoost models
│           ├── scanner.py           # tennis scanner + scout
│           ├── rankings.py          # live ATP/WTA ranking lookups
│           ├── config.py            # tennis-specific constants
│           └── mappings.py          # stat name normalization maps
├── data/
│   ├── nba/
│   │   ├── raw/                     # raw game logs (gitignored)
│   │   └── processed/               # engineered features (gitignored)
│   └── tennis/
│       ├── raw/                     # ATP/WTA match CSVs
│       ├── processed/               # training dataset (gitignored, 1.2GB)
│       ├── rankings_cache/          # cached ATP/WTA rankings
│       └── projections/             # scan output
├── models/
│   ├── nba/                         # XGBoost .json models (gitignored)
│   └── tennis/                      # XGBoost .json models (gitignored)
└── output/                          # scan results, CSVs (gitignored)
```

---

## NBA Models

21 XGBoost + LightGBM ensemble models (60% XGBoost / 40% LightGBM). Trained on 4 seasons of game logs with exponential recency sample weights and Optuna Bayesian hyperparameter optimization.

| Target  | MAE   | R²    | Directional Accuracy |
|---------|-------|-------|----------------------|
| PTS     | 4.77  | 0.461 | 60.2%                |
| REB     | 1.94  | 0.411 | 57.9%                |
| AST     | 1.43  | 0.465 | 55.5%                |
| FG3M    | 0.94  | 0.240 | 54.6%                |
| FG3A    | 1.56  | 0.514 | 54.0%                |
| BLK     | 0.54  | 0.164 | 39.7%                |
| STL     | 0.77  | 0.021 | 61.1%                |
| TOV     | 0.96  | 0.203 | 62.0%                |
| PRA     | 6.09  | 0.516 | 61.1%                |
| PR      | 5.62  | 0.471 | 60.6%                |
| PA      | 5.21  | 0.523 | 60.5%                |
| RA      | 2.66  | 0.429 | 58.6%                |
| SB      | 0.98  | 0.068 | 62.4%                |
| FGM     | 1.84  | 0.413 | 58.3%                |
| FGA     | 2.79  | 0.570 | 60.5%                |
| FTM     | 1.48  | 0.348 | 50.5%                |
| FTA     | 1.77  | 0.360 | 53.1%                |
| FPTS    | 7.77  | 0.459 | 61.4%                |
| PTS_1H  | 3.12  | 0.352 | 58.8%                |
| PRA_1H  | 3.84  | 0.432 | 59.9%                |
| FPTS_1H | 4.93  | 0.378 | 61.0%                |

MAE = Mean Absolute Error. R² = coefficient of determination. Directional Accuracy = % of test games where model correctly predicted Over vs Under relative to the PrizePicks line. Metrics from held-out test set (most recent 30% of games, including 2025-26 playoffs).

### Key Features (451 total)

**Rolling averages**
- Season, L20 (EWM), L10, L5 — for all primary stats
- L5 and L10 medians for robustness to outliers

**Streak & consistency signals**
- `{stat}_STREAK` = L5 avg − L20 avg (hot/cold indicator)
- `{stat}_CONSISTENCY` = L5 median / season avg (reliability signal)

**Expected possessions**
- `EXP_POSS` = team pace / 100 × usage rate / 100 × L5 minutes
- `EXP_POSS_SEASON` = same but using season-average minutes

**Opponent defense**
- Opponent-allowed stats filtered by position (G / F / C)
- Season-level and L5-level opponent ratings

**Context**
- Usage rate with transfer from injured teammates
- Schedule density: back-to-back, 4-in-6, days rest
- Home / away splits
- 1st-half splits (for _1H targets)

**Leakage prevention**
- Training excludes the target game's own stats from all rolling windows

### Model Architecture

- **Ensemble**: 60% XGBoost / 40% LightGBM weighted average
- **Recency weights**: `exp(-0.001 × days_ago)` — exponential decay so recent games matter more
- **Hyperparameter tuning**: Optuna Bayesian optimization, 30 trials per target, 3-fold TimeSeriesSplit cross-validation
- **Log transform**: applied to low-count targets (FG3M, BLK, STL, TOV, SB) to stabilize variance

---

## Tennis Models

7 XGBoost regression models. Trained on ~1M ATP/WTA matches.

| Target          | MAE   | R²    | Directional Accuracy |
|-----------------|-------|-------|----------------------|
| Total Sets      | 0.30  | 0.696 | 85.0%                |
| Total Games     | 2.99  | 0.697 | 84.0%                |
| Games Won       | 2.03  | 0.658 | 80.7%                |
| Aces            | 2.45  | 0.476 | 76.4%                |
| Break Pts Won   | 1.11  | 0.463 | 76.1%                |
| Double Faults   | 1.61  | 0.282 | 71.0%                |
| Total Tiebreaks | 0.43  | 0.211 | 67.7%                |

### Key Features Used

- Surface type (hard, clay, grass, carpet)
- ATP/WTA ranking with fuzzy name matching
- Head-to-head record
- Recent form (last 5/10/20 matches)
- Slam vs non-slam (best-of-5 vs best-of-3)
- Fatigue: days since last match, matches in last 7/14/30 days

---

## Line Difference Handling

When PrizePicks and FanDuel have different lines for the same player/stat, the system adjusts the implied probability using logarithmic scaling:

```
adjustment = factor * log(1 + |line_diff|) / log(2)
```

Each stat has its own adjustment factor (e.g. 3.5% per point for PTS, 6.0% per steal for STL) and its own maximum allowed line difference (e.g. 4.0 for PTS, 2.0 for BLK). This prevents unrealistic win percentages on large discrepancies while still capturing real edges.

If PrizePicks has a lower line than FanDuel, only the Over side is shown. If higher, only the Under. If lines match, both sides are shown.

---

## Setup

### Requirements

- Python 3.10+
- `ODDS_API_KEY` from [the-odds-api.com](https://the-odds-api.com) (required for FanDuel odds)

### Installation

```bash
git clone https://github.com/DevanshDaxini/Sports-EV-Bot.git
cd Sports-EV-Bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
ODDS_API_KEY=your_key_here
```

### First-time Setup (NBA)

Run these in order from the main menu:

1. **Build Data** — downloads 4 seasons of game logs + 1st-half splits via nba_api
2. **Engineer Features** — computes 451 features per player-game
3. **Train Models** — trains 21 XGBoost + LightGBM ensemble models (~10 min)

### First-time Setup (Tennis)

1. **Build Data** — downloads ATP/WTA match history from Sackmann GitHub
2. **Engineer Features** — computes 150+ features per match (~5 min)
3. **Train Models** — trains 7 XGBoost models (~3 min)

---

## Usage

```bash
python main.py
```

Select a sport, then choose a tool:

- **Super Scanner** — finds plays where math odds and AI projection agree
- **Odds Scanner** — pure FanDuel vs PrizePicks line comparison
- **AI Scanner** — standalone AI predictions with player scouting

The player scout shows per-stat projections, PrizePicks lines (including goblin/demon alt lines marked with (G)/(D)), FanDuel-implied win percentages, and Over/Under recommendations.

---

## API Usage

- **PrizePicks**: partner API, free, no key needed. Cached for 10 minutes.
- **FanDuel (via the-odds-api)**: requires API key, costs credits per call. Cached for 10 minutes. The Odds Scanner auto-detects the active game slate date to minimize unnecessary calls.
- **nba_api**: free, no key needed. Used for game schedules and historical data. Features a custom resilience layer with 30-second timeouts and automatic 5-attempt retries to handle frequent `stats.nba.com` connection issues.
- **ESPN/CBS Sports**: scraped for live injury reports. No key needed.

---

## Disclaimer

This software is for educational and research purposes only. Sports betting involves significant financial risk. No guarantee of profit. Not responsible for financial losses.
