# Prop Correlation & Projection Engine

### Technical Audit & Predictive Performance Documentation

An advanced sports analytics suite developed in **Python** that identifies high-value player prop opportunities. The system utilizes a dual-layered approach: cross-referencing **XGBoost machine learning projections** against **FanDuel’s market-implied probabilities** to isolate mathematical edges.

---

## Evidence of Model Legitimacy

The model's credibility is built on verifiable regression metrics and directional accuracy. Below are the performance benchmarks from the most recent training cycle.

### 1. High-Conviction Performance

We track **Directional Accuracy** (the % of games where the AI correctly predicts the Over/Under side) and **** (the model's predictive power).

| Stat Category | Directional Accuracy |  Score | Mean Absolute Error (MAE) |
| --- | --- | --- | --- |
| **Points (PTS)** | **89.7%** | **0.747** | **2.93** |
| **Pts+Asts (PA)** | **86.5%** | **0.759** | **1.21** |
| **PRA** | **88.4%** | **0.729** | **3.15** |
| **Assists (AST)** | **78.5%** | **0.483** | **1.29** |

> **Note on Legitimacy:** An  of **~0.75** for Points indicates the model explains 75% of the scoring variance—a benchmark considered "Elite" in quantitative sports modeling.

### 2. Profitability Benchmark

The engine is graded against a **54.1% win rate threshold**, the mathematical breakeven point for PrizePicks 5-man flex plays. Currently, **Elite Tier** models operate with a **~30% margin** above this breakeven line.

---

## Core Architectural Features

The model's reliability is a result of several specialized sub-systems designed to handle the volatility of professional basketball.

### Teammate "Usage Vacuum" Logic

The engine monitors **Missing Usage**. When a high-volume player is ruled **OUT**, the system calculates the percentage of team possessions suddenly available.

* **Feature:** `MISSING_USAGE` / `USAGE_VACUUM`
* **Impact:** Identifies "blow-up" opportunities for secondary players before the betting market can adjust the lines.

### Feature Leakage Prevention

To ensure "honest" metrics, `train.py` strictly prevents the model from "peeking" at relevant stats from the game it is trying to predict.

* **Logic:** The model is blocked from seeing any rolling averages or opponent-allowed stats that contain the target variable for that specific game date.

### Fatigue & Schedule Density Mapping

Performance is adjusted based on the human element of the NBA schedule.

* **`IS_4_IN_6`**: Flags teams playing their 4th game in 6 nights.
* **`DAYS_REST`**: Adjusts for the performance "rust" of long layoffs versus the fatigue of back-to-back games.

### Position-Specific Defensive Analysis

Instead of general defensive rankings, the bot utilizes **OPP_ALLOWED** stats filtered by position.

* **Logic:** Calculates how many points or rebounds an opponent allows specifically to **Guards, Forwards, or Centers**.

---

## Project Structure

```text
sports_ev_bot/
├── main.py
├── .env
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── utils.py
│   │   ├── visualizer.py
│   │   ├── odds_providers/
│   │   │   ├── __init__.py
│   │   │   ├── fanduel.py
│   │   │   └── prizepicks.py
│   │   └── analyzers/
│   │       ├── __init__.py
│   │       └── analyzer.py
│   ├── sports/
│   │   ├── __init__.py
│   │   ├── nba/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── mappings.py
│   │       ├── builder.py
│   │       ├── features.py
│   │       ├── scanner.py
│   │       ├── train.py
│   │       ├── tune_train.py
│   │       ├── injuries.py
│   │       ├── grader.py
│   │       └── check_features.py
│   │      
│   └── cli/
│       ├── __init__.py
│       └── nba_cli.py
├── data/
│   └── nba/
│       ├── raw/
│       ├── processed/
│       └── projections/
├── models/
│   └── nba/
└── output/
    └── nba/
        ├── scans/
        └── analysis_plots/

```

---

## Usage & Validation

1. **Configure:** Add your `ODDS_API_KEY` to the `.env` file.
2. **Build:** Run `builder.py`, `features.py`, and `train.py` to sync latest data and train the AI.
3. **Scan:** Run `python3 main.py` to identify high-conviction edges.
4. **Audit:** Run `python3 -m src.visualizer` to generate live accuracy plots in the `analysis_plots/` folder.

---

## Disclaimer

This software is for **educational and research purposes only**. Sports betting involves significant financial risk. I do not guarantee profit and am not responsible for any financial losses.

---
