# NBA Prop Correlation & Projection Engine

An advanced, dual-layered sports analytics tool developed in **Python** that identifies high-value player prop opportunities by correlating market-implied probabilities with custom machine learning projections.

This system is designed to automate the detection of "sharp" plays by finding consensus between efficient market prices (FanDuel) and independent performance modeling (AI).

---

## 🛠 Features

* **Correlated Logic Engine:** Automatically identifies plays where both the betting market (Math) and the AI model (Data) agree on the outcome (Over/Under).
* **Dual-Source Integration:**
* **Market Scanner:** Fetches real-time odds via **The-Odds-API** for FanDuel and calculates implied win percentages.
* **AI Projection Scanner:** Integrates custom player projections to find discrepancies against **PrizePicks** lines.


* **Weighted Confidence Scoring:** Implements a normalized scoring algorithm (0-100) that balances:
* **Implied Win %:** Market-based probability derived from American odds.
* **AI Edge:** The percentage difference between the model's projection and the current line.
* **Volatility Weighting:** Adjusts scores based on the historical reliability of specific statistics (e.g., Rebounds are weighted higher than 3-Point Makes).


* **Efficient Data Management:** Features a multi-level caching system (In-memory and Disk) for FanDuel data to minimize API credit consumption.
* **Automated Logging:** Exports the Top 20 "High-Conviction" plays to `program_runs/correlated_plays.csv` for post-game performance grading.

---

## 📂 Project Architecture

Based on the current project structure:

```text
├── main.py                 # Primary entry point & CLI menu
├── visualizer.py           # Data visualization and charting
├── requirements.txt        # Project dependencies
├── .gitignore              # Git exclusion rules
├── src/                    # Source code directory
│   ├── analyzer.py         # Statistical analysis & edge calculation
│   ├── builder.py          # Dataset construction logic
│   ├── config.py           # Global settings, stat maps, and API keys
│   ├── fanduel.py          # Market data client with disk caching
│   ├── features.py         # Feature engineering for AI models
│   ├── grader.py           # Post-game performance tracking
│   ├── prizepicks.py       # Board scraper for pick'em platforms
│   ├── scanner.py          # Core scanning and correlation logic
│   ├── train.py            # AI model training scripts
│   ├── tune_train.py       # Hyperparameter tuning and optimization
│   └── utils.py            # Shared helper functions
├── csvFiles/               # Raw data storage
├── data/                   # Processed datasets
├── fanduel_cache/          # Local JSON storage for market odds
├── model_images/           # Visualized model performance metrics
├── models/                 # Saved machine learning model files (.pkl, .h5)
└── program_runs/           # CSV exports of historical scanner runs

```

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.8+
* Valid API Key from **The-Odds-API**

### 2. Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/NBA_EV_BOT.git
cd NBA_EV_BOT

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Configure Environment:**
Create a `.env` file in the `src/` directory:
```env
ODDS_API_KEY=your_api_key_here

```



### 3. Usage

Run the main script to launch the interactive scanner:

```bash
python main.py

```

---

## 📊 Scoring Methodology

The bot uses a **Balanced Ranking Formula** to prevent AI outliers from skewing results:

| Component | Logic |
| --- | --- |
| **Math Rank** | Normalizes Implied Win% (Scales 51%–56% to 0–10). |
| **AI Rank** | Normalizes AI Margin (Scales 0%–25% edge to 0–10). |
| **Volatility** | Multiplier based on stat predictability (e.g., REB = 1.15, FG3M = 0.90). |

---

## ⚖️ Disclaimer

This software is intended for **educational and research purposes only**. Sports betting involves significant risk. I do not guarantee profit and are not responsible for any financial losses incurred through the use of this tool.
