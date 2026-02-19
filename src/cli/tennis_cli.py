"""
Tennis CLI - Main Entry Point for the Tennis EV Bot

Provides the interactive menu system for tennis analysis:
    1. 🚀 Super Scanner       - Math + AI correlated plays
    2. 💰 Odds Scanner        - FanDuel vs PrizePicks arbitrage
    3. 🎾 AI Scanner          - Today/Tomorrow/Scout/Grade
    4. 🔨 Build Data          - Download Sackmann ATP/WTA match history
    5. ⚙️  Engineer Features   - Build training features
    6. 🤖 Train Models        - Train/retrain XGBoost models
    7. 📊 Model Metrics       - View current model accuracy
    8. 🏆 Rankings Debug      - Test player rank lookups

No external API key required for AI Scanner. Uses:
    - Sackmann GitHub (training data + rankings, free)
    - PrizePicks partner API (lines, free)
Odds/Super Scanners require ODDS_API_KEY in .env for FanDuel via the-odds-api.

Usage:
    Called from main.py → main_menu()
    Or directly: $ python3 -m src.cli.tennis_cli
"""

import os
import sys
import warnings
import requests
import time
import json
import unicodedata
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(_BASE, 'output', 'tennis', 'scans')


# ---------------------------------------------------------------------------
# MENU ACTIONS
# ---------------------------------------------------------------------------

def run_scanner():
    """Launch the full interactive tennis scanner (today/tomorrow/scout/grade)."""
    try:
        from src.sports.tennis.scanner import main as scanner_main
        scanner_main()
    except ImportError as e:
        print(f"Import error: {e}")
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"Scanner error: {e}")
        import traceback; traceback.print_exc()
        input("\nPress Enter to continue...")


# --- SETUP ---
def run_builder():
    """Download Sackmann ATP/WTA historical data."""
    print("\n" + "=" * 55)
    print("   BUILD TENNIS DATA")
    print("=" * 55)
    print("Downloads ATP + WTA match history from Sackmann GitHub")
    print("Source: github.com/JeffSackmann (free, no API key)")
    print("")

    confirm = input("This may take 1-2 minutes. Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    try:
        from src.sports.tennis.builder import (
            fetch_tour_data, ATP_BASE_URL, WTA_BASE_URL,
            ATP_FALLBACKS, WTA_FALLBACKS, ATP_OUTPUT, WTA_OUTPUT
        )
        fetch_tour_data('atp', ATP_BASE_URL, ATP_FALLBACKS, ATP_OUTPUT)
        fetch_tour_data('wta', WTA_BASE_URL, WTA_FALLBACKS, WTA_OUTPUT)
        print("\nData build complete!")
        print("   Next: Run 'Engineer Features' then 'Train Models'.")
    except Exception as e:
        print(f"\nBuilder error: {e}")

    input("\nPress Enter to continue...")


def run_feature_engineering():
    """Run feature engineering pipeline on raw data."""
    print("\n" + "=" * 55)
    print("   FEATURE ENGINEERING")
    print("=" * 55)
    print("Building 150+ features from raw match data (~5 min)...")
    print("")

    try:
        from src.sports.tennis.features import build_features
        build_features()
        print("\nFeatures built!")
        print("   Next: Run 'Train Models'.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()

    input("\nPress Enter to continue...")


def run_training():
    """Train/retrain all 7 XGBoost tennis models."""
    print("\n" + "=" * 55)
    print("   TRAIN TENNIS MODELS")
    print("=" * 55)
    print("Training 7 XGBoost models (~3 min)...")
    print("")

    try:
        from src.sports.tennis.train import train_and_evaluate
        train_and_evaluate()
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback; traceback.print_exc()

    input("\nPress Enter to continue...")


# --- REPORTING ---
def view_metrics():
    """Display saved model accuracy metrics."""
    metrics_path = os.path.join(_BASE, 'models', 'tennis', 'model_metrics.csv')

    print("\n" + "=" * 55)
    print("   TENNIS MODEL METRICS")
    print("=" * 55)

    if not os.path.exists(metrics_path):
        print("No metrics found. Run 'Train Models' first.")
        input("\nPress Enter to continue...")
        return

    df = pd.read_csv(metrics_path)

    print(f"\n{'TARGET':<22} {'MAE':>6} {'R²':>6} {'DIR%':>7} {'TRAIN':>8}")
    print("-" * 55)
    for _, row in df.iterrows():
        print(f"{row['target']:<22} {row['mae']:>6.3f} {row['r2']:>6.3f} "
              f"{row['dir_accuracy']:>6.1f}% {int(row['train_rows']):>7,}")

    print("\nMAE  = Mean Absolute Error (lower is better)")
    print("DIR% = Directional accuracy — did we predict Over/Under correctly")
    print(f"\nLast trained: {df['trained_at'].iloc[-1]}")

    input("\nPress Enter to continue...")


def run_rankings_debug():
    """Test player rank lookups and surface detection."""
    print("\n" + "=" * 55)
    print("   RANKINGS DEBUG")
    print("=" * 55)

    try:
        from src.sports.tennis.rankings import TennisRankings
        r = TennisRankings()
        r.load(force_refresh=True)

        print("\nTest player rank lookups:")
        test_players = [
            'Carlos Alcaraz', 'Jannik Sinner', 'Novak Djokovic', 'Taylor Fritz',
            'Iga Swiatek', 'Aryna Sabalenka', 'Coco Gauff', 'Madison Keys'
        ]
        for p in test_players:
            rank = r.get_rank(p)
            tour = r.get_tour(p).upper()
            rank_str = f"#{int(rank)}" if rank != 50.0 else "Unknown"
            print(f"  {p:<28} {rank_str:>6}  ({tour})")

        print("\nTest surface lookups:")
        tournaments = [
            'Australian Open', 'Roland Garros', 'Wimbledon', 'US Open',
            'Indian Wells', 'Miami Open', 'Madrid Open', 'Halle'
        ]
        for t in tournaments:
            print(f"  {t:<30} → {r.get_surface(t)}")

        print("\nTest slam detection:")
        for t in ['Australian Open', 'Indian Wells', 'Wimbledon', 'Miami Open']:
            print(f"  {t:<30} → best-of-5 = {r.is_slam(t)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()

    input("\nPress Enter to continue...")


# ---------------------------------------------------------------------------
# MAIN MENU
# ---------------------------------------------------------------------------

def main_menu():
    """Main tennis menu — called from main.py."""

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("\n" + "=" * 55)
        print("   TENNIS EV BOT")
        print("=" * 55)
        print(f"   {datetime.now().strftime('%A, %B %d, %Y')}")
        print("=" * 55)

        print("\nANALYSIS")
        print("1. AI Scanner            -- Scan / Scout / Grade")

        print("\nSETUP  (run once in order)")
        print("2. Build Data            -- Download ATP + WTA history")
        print("3. Engineer Features     -- Build training features")
        print("4. Train Models          -- Train all 7 XGBoost models")

        print("\nREPORTING")
        print("5. Model Metrics         -- Accuracy by market")
        print("6. Rankings Debug        -- Test rank/surface lookups")

        print("\n" + "=" * 55)
        print("0. Back")
        print("=" * 55)

        choice = input("\nSelect: ").strip()

        if   choice == '1': run_scanner()
        elif choice == '2': run_builder()
        elif choice == '3': run_feature_engineering()
        elif choice == '4': run_training()
        elif choice == '5': view_metrics()
        elif choice == '6': run_rankings_debug()
        elif choice == '0': break
        else:
            print("\nInvalid selection.")
            input("Press Enter to try again...")


if __name__ == "__main__":
    main_menu()
