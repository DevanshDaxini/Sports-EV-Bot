"""
sports_ev_bot - Entry Point

Run this file to launch the interactive menu.
    $ python main.py

To add a new sport (e.g., CBB) in the future:
    1. Populate src/sports/cbb/config.py
    2. Create src/sports/cbb/mappings.py, builder.py, features.py, train.py
    3. Add a CBB menu option in src/cli/nba_cli.py (or create src/cli/cbb_cli.py)
"""

import sys
import os

# Ensure the project root is on the path so all src.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli.nba_cli import main_menu

if __name__ == "__main__":
    main_menu()
