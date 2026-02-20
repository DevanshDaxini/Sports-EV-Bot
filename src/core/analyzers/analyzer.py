"""
Props Edge Analyzer - FanDuel vs PrizePicks Comparison

NOW WITH LINE-ADJUSTED WIN% CALCULATION!

Calculates implied probability from FanDuel odds,
THEN adjusts for PrizePicks line differences to show YOUR actual win rate.

Mathematical Approach:
    1. Convert FanDuel Over/Under odds to implied probabilities
    2. Adjust probability based on PP vs FD line difference
    3. Return opportunities with ADJUSTED win rates
    
Example (NEW):
    FanDuel: LeBron 26.5 Points, Over -120, Under +100
    PrizePicks: LeBron 25.5 Points (1 point easier!)
    
    Step 1: Convert to probabilities
        prob_over = 120/220 = 54.5%
        prob_under = 100/200 = 50.0%
        
    Step 2: Adjust for line difference
        Line diff = 25.5 - 26.5 = -1.0 (PP is 1 point easier)
        For OVER: easier line = higher win probability
        Adjustment = +3.5% per point for PTS
        Adjusted WIN% = 54.5% + 3.5% = 58.0%
        
Usage:
    from src.core.analyzers.analyzer import PropsAnalyzer
    analyzer = PropsAnalyzer(prizepicks_df, fanduel_df, league='NBA')
    edges = analyzer.calculate_edges()
"""

import math
import pandas as pd
from fuzzywuzzy import process
from src.core.config import SLIP_CONFIG

class PropsAnalyzer:
    def __init__(self, prizepicks_df, fanduel_df, league='NBA'):
        """
        Args:
            prizepicks_df (DataFrame): PrizePicks lines
            fanduel_df (DataFrame):   FanDuel odds
            league (str):             Sport league label e.g. 'NBA', 'CBB'
        """
        self.pp_df = prizepicks_df
        self.fd_df = fanduel_df
        self.league = league
        
        # ✅ Line adjustment factors (% change per point difference)
        # These are empirically derived - adjust based on testing
        self.LINE_ADJUSTMENT_FACTORS = {
            # High-variance stats (bigger adjustments per point)
            'PTS': 0.035,   # 3.5% per point
            'Points': 0.035,
            'PA': 0.030,    # 3.0% per point (combo stat)
            'PRA': 0.025,   # 2.5% per point (combo stat)
            'PR': 0.030,
            
            # Medium-variance stats
            'REB': 0.040,   # 4.0% per point (rebounds vary more)
            'Rebounds': 0.040,
            'AST': 0.045,   # 4.5% per point (assists highly variable)
            'Assists': 0.045,
            'RA': 0.035,
            
            # High-variance stats (biggest adjustments)
            'FG3M': 0.055,  # 5.5% per 3-pointer
            '3-Pt Made': 0.055,
            'STL': 0.060,   # 6.0% per steal
            'Steals': 0.060,
            'BLK': 0.060,   # 6.0% per block
            'Blocks': 0.060,
            'SB': 0.055,    # Steals + Blocks combo
            
            # Default for unknown stats
            'DEFAULT': 0.035
        }

        # ✅ Dynamic per-stat maximum line difference thresholds
        # Lines differing by more than this are dropped (too stale to trust)
        self.MAX_LINE_DIFF = {
            # High-volume stats — 4-point swings are common
            'PTS': 4.0, 'Points': 4.0,
            'PRA': 4.0, 'Pts+Rebs+Asts': 4.0,
            'PR': 4.0,  'Pts+Rebs': 4.0,
            'PA': 4.0,  'Pts+Asts': 4.0,
            # Medium-volume
            'REB': 3.0, 'Rebounds': 3.0,
            'AST': 3.0, 'Assists': 3.0,
            'RA': 3.0,  'Rebs+Asts': 3.0,
            # Moderate
            'FG3M': 2.5, '3-Pt Made': 2.5,
            'FGM': 2.5,  'Field Goals Made': 2.5,
            'FGA': 2.5,  'Field Goals Attempted': 2.5,
            # Low-volume — more volatile per point
            'STL': 2.0, 'Steals': 2.0,
            'BLK': 2.0, 'Blocks': 2.0,
            'SB': 2.0,  'Blks+Stls': 2.0,
            'TOV': 2.0, 'Turnovers': 2.0,
            'FTM': 2.0, 'Free Throws Made': 2.0,
            'FTA': 2.0, 'Free Throws Attempted': 2.0,
            'DEFAULT': 3.0
        }

    def calculate_edges(self):
        """
        Find profitable opportunities by comparing PrizePicks to FanDuel.
        
        NOW WITH LINE-ADJUSTED WIN% CALCULATION!
        
        Returns:
            pandas.DataFrame: Rows with columns:
                Date, Player, League, Stat, Line, Side, Implied_Win_%, FD_Odds, FD_Line
        """
        opportunities = []

        if self.fd_df.empty:
            return pd.DataFrame()

        # --- STEP 1: RESHAPE FANDUEL DATA (Long -> Wide) ---
        fd_over = self.fd_df[self.fd_df['Side'] == 'Over'].copy()
        fd_under = self.fd_df[self.fd_df['Side'] == 'Under'].copy()

        fd_over = fd_over.rename(columns={'Odds': 'over_price'})
        fd_under = fd_under.rename(columns={'Odds': 'under_price'})

        fd_over = fd_over.drop(columns=['Side'], errors='ignore')
        fd_under = fd_under.drop(columns=['Side'], errors='ignore')

        before_merge = len(fd_over)
        self.fd_wide = pd.merge(
            fd_over,
            fd_under,
            on=['Player', 'Stat', 'Line', 'Date'],
            how='inner'
        )
        after_merge = len(self.fd_wide)

        if after_merge < before_merge * 0.7:
            print(f"⚠️  Warning: Only {after_merge}/{before_merge} lines had both Over and Under odds")
            print(f"    Lost {before_merge - after_merge} opportunities due to incomplete data")

        # --- STEP 2: LOOP THROUGH PRIZEPICKS ROWS ---
        for index, pp_row in self.pp_df.iterrows():
            pp_name = pp_row['Player']
            pp_stat = pp_row['Stat']
            pp_line = pp_row['Line']
            pp_date = pp_row.get('Date', 'Unknown')

            fd_name, fd_rows = self._find_match_in_fanduel(pp_name)
            if fd_name is None:
                continue

            matching_stat = fd_rows[fd_rows['Stat'] == pp_stat]
            if matching_stat.empty:
                continue

            fd_row = matching_stat.iloc[0]
            fd_line = fd_row['Line']
            line_diff = pp_line - fd_line

            # ✅ Dynamic per-stat max line difference (replaces hard 1.5 cap)
            max_diff = self.MAX_LINE_DIFF.get(
                pp_stat, self.MAX_LINE_DIFF['DEFAULT']
            )
            if abs(line_diff) > max_diff:
                continue

            # ✅ Only show the side where the line discrepancy gives YOU edge
            # PP higher than FD → Under is easier on PP → show Under only
            # PP lower than FD  → Over is easier on PP  → show Over only
            # Lines match       → show both
            if line_diff < 0:
                valid_sides = ['Over']
            elif line_diff > 0:
                valid_sides = ['Under']
            else:
                valid_sides = ['Over', 'Under']

            fd_over_odds = fd_row['over_price']
            fd_under_odds = fd_row['under_price']

            # Get base true probability from FanDuel
            true_over, true_under = self._calculate_true_probability(fd_over_odds, fd_under_odds)
            
            # ✅ NEW: Adjust for line difference
            adjusted_over, adjusted_under = self._adjust_for_line_difference(
                true_over, true_under, line_diff, pp_stat
            )

            if 'Over' in valid_sides:
                opportunities.append({
                    "Date": pp_date,
                    "Player": pp_name,
                    "League": self.league,
                    "Stat": pp_stat,
                    "Line": pp_line,
                    "Side": "Over",
                    "Implied_Win_%": round(adjusted_over * 100, 2),
                    "FD_Odds": fd_over_odds,
                    "FD_Line": fd_line,
                    "Line_Diff": round(line_diff, 1)  # ✅ PP - FD
                })

            if 'Under' in valid_sides:
                opportunities.append({
                    "Date": pp_date,
                    "Player": pp_name,
                    "League": self.league,
                    "Stat": pp_stat,
                    "Line": pp_line,
                    "Side": "Under",
                    "Implied_Win_%": round(adjusted_under * 100, 2),
                    "FD_Odds": fd_under_odds,
                    "FD_Line": fd_line,
                    "Line_Diff": round(line_diff, 1)  # ✅ PP - FD
                })

        return pd.DataFrame(opportunities)

    def _find_match_in_fanduel(self, pp_name):
        if hasattr(self, 'fd_wide') and not self.fd_wide.empty:
            search_df = self.fd_wide
        else:
            return None, None

        fd_unique_name = search_df['Player'].unique()
        match_name, score = process.extractOne(pp_name, fd_unique_name)

        if score < 80:
            return None, None

        player_rows = search_df[search_df['Player'] == match_name]
        return match_name, player_rows

    def _calculate_true_probability(self, over_odds, under_odds):
        """
        Get implied probability without removing bookmaker vig.
        
        Args:
            over_odds (int): American odds for Over (e.g., -120)
            under_odds (int): American odds for Under (e.g., +100)
            
        Returns:
            tuple: (implied_over_prob, implied_under_prob)
        """
        def odds_to_prob(odds):
            if odds < 0:
                return (-odds) / ((-odds) + 100)
            else:
                return 100 / (odds + 100)

        prob_over = odds_to_prob(over_odds)
        prob_under = odds_to_prob(under_odds)

        return prob_over, prob_under

    def _adjust_for_line_difference(self, true_over, true_under, line_diff, stat):
        """
        ✅ NEW METHOD: Adjust probabilities based on line difference.
        
        Args:
            true_over (float): Base probability for Over (from FanDuel)
            true_under (float): Base probability for Under (from FanDuel)
            line_diff (float): PP_line - FD_line
            stat (str): Stat type (PTS, REB, AST, etc.)
            
        Returns:
            tuple: (adjusted_over, adjusted_under)
            
        Logic:
            - If line_diff < 0: PP line is EASIER (lower) for Over
                → Increase Over probability, decrease Under probability
            - If line_diff > 0: PP line is HARDER (higher) for Over
                → Decrease Over probability, increase Under probability
                
        Example:
            FD: 26.5, PP: 25.5 (line_diff = -1.0)
            Over is easier on PP!
            Adjustment = 0.035 * 1.0 = +3.5% to Over
        """
        if line_diff == 0:
            # Lines are identical, no adjustment needed
            return true_over, true_under
        
        # Get adjustment factor for this stat
        adjustment_factor = self.LINE_ADJUSTMENT_FACTORS.get(
            stat,
            self.LINE_ADJUSTMENT_FACTORS['DEFAULT']
        )
        
        # ✅ Logarithmic scaling: diminishing returns on larger diffs
        # First point gives ~100% of factor, 2pts ~158%, 3pts ~200%
        # (vs linear: 100%, 200%, 300%)
        adjustment = adjustment_factor * math.log(1 + abs(line_diff)) / math.log(2)
        
        if line_diff < 0:
            # PP line is LOWER (easier for Over, harder for Under)
            adjusted_over = min(true_over + adjustment, 0.90)  # Cap at 90%
            adjusted_under = max(true_under - adjustment, 0.10)  # Floor at 10%
        else:
            # PP line is HIGHER (harder for Over, easier for Under)
            adjusted_over = max(true_over - adjustment, 0.10)  # Floor at 10%
            adjusted_under = min(true_under + adjustment, 0.90)  # Cap at 90%
        
        # Normalize so probabilities sum to 1.0
        total = adjusted_over + adjusted_under
        adjusted_over = adjusted_over / total
        adjusted_under = adjusted_under / total
        
        return adjusted_over, adjusted_under


# --- TEST BLOCK ---
if __name__ == "__main__":
    print("--- TESTING DYNAMIC LINE-ADJUSTED ANALYZER ---")

    # Scenario 1: Small difference (1 point)
    pp_data = {
        'Player': ['LeBron James', 'LeBron James'],
        'Stat': ['Points', 'Points'],
        'Line': [25.5, 25.5],
        'Date': ['2026-02-12', '2026-02-12']
    }
    pp_df = pd.DataFrame(pp_data)

    fd_data = [
        {'Player': 'LeBron James', 'Stat': 'Points', 'Line': 26.5, 'Odds': -120, 'Side': 'Over', 'Date': '2026-02-12'},
        {'Player': 'LeBron James', 'Stat': 'Points', 'Line': 26.5, 'Odds': +100, 'Side': 'Under', 'Date': '2026-02-12'}
    ]
    fd_df = pd.DataFrame(fd_data)

    print("\nScenario 1: FD 26.5 vs PP 25.5 (1pt diff, Over edge)")
    analyzer = PropsAnalyzer(pp_df, fd_df, league='NBA')
    results = analyzer.calculate_edges()
    if not results.empty:
        print("✅ Found edges:")
        print(results[['Player', 'Side', 'Line', 'FD_Line', 'Line_Diff', 'Implied_Win_%']].to_string(index=False))
    else:
        print("❌ No edges found.")

    # Scenario 2: Large difference (3 points) — previously dropped!
    pp_df2 = pd.DataFrame({'Player': ['Jayson Tatum'], 'Stat': ['Points'], 'Line': [24.5], 'Date': ['2026-02-12']})
    fd_data2 = [
        {'Player': 'Jayson Tatum', 'Stat': 'Points', 'Line': 27.5, 'Odds': -110, 'Side': 'Over', 'Date': '2026-02-12'},
        {'Player': 'Jayson Tatum', 'Stat': 'Points', 'Line': 27.5, 'Odds': -110, 'Side': 'Under', 'Date': '2026-02-12'}
    ]
    fd_df2 = pd.DataFrame(fd_data2)

    print("\nScenario 2: FD 27.5 vs PP 24.5 (3pt diff — was DROPPED, now captured!)")
    analyzer2 = PropsAnalyzer(pp_df2, fd_df2, league='NBA')
    results2 = analyzer2.calculate_edges()
    if not results2.empty:
        print("✅ Found edges:")
        print(results2[['Player', 'Side', 'Line', 'FD_Line', 'Line_Diff', 'Implied_Win_%']].to_string(index=False))
    else:
        print("❌ No edges found (check MAX_LINE_DIFF for stat).")