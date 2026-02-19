# Sports EV Bot

A multi-sport prop betting analysis engine. Uses XGBoost regression models trained on historical data to project player stats, then compares those projections against PrizePicks lines and FanDuel odds to find positive expected value (+EV) opportunities.

Supports NBA (17 stat models) and Tennis (7 market models).

---

## How It Works

The system runs three independent analysis layers that can be used alone or combined:

### 1. AI Scanner

Trains XGBoost models on historical player data to predict stat lines. For NBA, it uses nba_api game logs with 100+ engineered features (rolling averages, opponent-allowed stats by position, usage rate, rest days, home/away splits). For Tennis, it uses Jeff Sackmann's open match data with 150+ features (surface, rank, H2H, recent form, fatigue).

Each model outputs a projected stat value. The scanner compares this projection to the PrizePicks line and shows the delta.

### 2. Odds Scanner

Fetches live odds from FanDuel via the-odds-api and compares them to PrizePicks lines. The analyzer removes the vig from FanDuel two-way markets to calculate the true implied probability, then adjusts for any line difference between platforms using logarithmic scaling. Dynamic per-stat thresholds control how large a line discrepancy is allowed before filtering it out.

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
│       │   ├── builder.py           # download game logs via nba_api
│       │   ├── features.py          # engineer 100+ training features
│       │   ├── train.py             # train 17 XGBoost models
│       │   ├── scanner.py           # AI scanner, player scout, game scanning
│       │   ├── injuries.py          # live injury scraper (ESPN + CBS)
│       │   ├── grader.py            # backtest grading
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

17 XGBoost regression models. Trained on ~3 seasons of game logs.

| Target | MAE   | R2    | Directional Accuracy |
|--------|-------|-------|----------------------|
| PTS    | 2.03  | 0.882 | 89.4%                |
| FGM    | 0.79  | 0.873 | 87.8%                |
| PA     | 2.82  | 0.848 | 87.2%                |
| PR     | 3.31  | 0.805 | 85.5%                |
| PRA    | 3.95  | 0.788 | 84.8%                |
| FGA    | 2.30  | 0.698 | 83.8%                |
| FG3A   | 1.27  | 0.681 | 80.9%                |
| FTA    | 1.37  | 0.520 | 76.9%                |
| FG3M   | 0.71  | 0.606 | 76.5%                |
| RA     | 2.60  | 0.458 | 74.3%                |
| FTM    | 1.19  | 0.484 | 73.8%                |
| REB    | 1.91  | 0.434 | 72.1%                |
| AST    | 1.42  | 0.466 | 72.2%                |
| STL    | 0.76  | 0.086 | 71.0%                |
| TOV    | 0.95  | 0.250 | 61.4%                |
| SB     | 0.97  | 0.133 | 53.4%                |
| BLK    | 0.57  | 0.184 | 35.0%                |

MAE = Mean Absolute Error. R2 = coefficient of determination. Directional Accuracy = percentage of games where the model correctly predicted Over vs Under relative to the PrizePicks line.

### Key Features Used

- Rolling averages (season, last 10, last 5 games)
- Opponent-allowed stats filtered by position (G/F/C)
- Usage rate and missing usage from injured teammates
- Schedule density (back-to-back, 4-in-6, days rest)
- Home/away splits
- Feature leakage prevention: training excludes the target game's stats from all rolling windows

---

## Tennis Models

7 XGBoost regression models. Trained on ~1M ATP/WTA matches.

| Target          | MAE   | R2    | Directional Accuracy |
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
git clone https://github.com/DevanshDaxini/nba_ev_bot.git
cd nba_ev_bot
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

1. Build Data -- downloads ~3 seasons of game logs via nba_api
2. Engineer Features -- computes 100+ features per player-game
3. Train Models -- trains 17 XGBoost models (~5 min)

### First-time Setup (Tennis)

1. Build Data -- downloads ATP/WTA match history from Sackmann GitHub
2. Engineer Features -- computes 150+ features per match (~5 min)
3. Train Models -- trains 7 XGBoost models (~3 min)

---

## Usage

```bash
python main.py
```

Select a sport, then choose a tool:

- **Super Scanner** -- finds plays where math odds and AI projection agree
- **Odds Scanner** -- pure FanDuel vs PrizePicks line comparison
- **AI Scanner** -- standalone AI predictions with player scouting

The player scout shows per-stat projections, PrizePicks lines (including goblin/demon alt lines marked with (G)/(D)), FanDuel-implied win percentages, and Over/Under recommendations.

---

## API Usage

- **PrizePicks**: partner API, free, no key needed. Cached for 10 minutes.
- **FanDuel (via the-odds-api)**: requires API key, costs credits per call. Cached for 10 minutes. The Odds Scanner auto-detects the active game slate date to minimize unnecessary calls.
- **nba_api**: free, no key needed. Used for game schedules and historical data.
- **ESPN/CBS Sports**: scraped for live injury reports. No key needed.

---

## Disclaimer

This software is for educational and research purposes only. Sports betting involves significant financial risk. No guarantee of profit. Not responsible for financial losses.
