"""
Props Edge Analyzer - FanDuel vs PrizePicks Comparison

Calculates "true probability" by removing bookmaker vig from FanDuel odds,
then identifies opportunities where true probability exceeds PrizePicks breakeven rate.

Mathematical Approach:
    1. Convert FanDuel Over/Under odds to implied probabilities
    2. Remove vig: true_prob = implied_prob / (sum of both sides)
    3. Compare to PrizePicks hurdle rate (e.g., 54.25% for 5-man flex)
    4. Return profitable opportunities
    
Example:
    FanDuel: LeBron 25.5 Points, Over -120, Under +100
    
    Step 1: Convert to probabilities
        prob_over = 120/220 = 54.5%
        prob_under = 100/200 = 50.0%
        
    Step 2: Remove vig
        market_total = 54.5% + 50.0% = 104.5% (vig = 4.5%)
        true_over = 54.5% / 104.5% = 52.2%
        true_under = 50.0% / 104.5% = 47.8%
        
    Step 3: Compare to PrizePicks
        If true_over (52.2%) < hurdle (54.25%) → Skip (no edge)
        
Usage:
    from src.core.analyzers.analyzer import PropsAnalyzer
    analyzer = PropsAnalyzer(prizepicks_df, fanduel_df, league='NBA')
    edges = analyzer.calculate_edges()
"""

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
                                      Written into the output rows.
        """
        self.pp_df = prizepicks_df
        self.fd_df = fanduel_df
        self.league = league

    def calculate_edges(self):
        """
        Find profitable opportunities by comparing PrizePicks to FanDuel.
        
        Returns:
            pandas.DataFrame: Rows with columns:
                Date, Player, League, Stat, Line, Side, Implied_Win_%, FD_Odds
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

            valid_sides = ['Over', 'Under']

            if line_diff != 0:
                if abs(line_diff) > 1.5:
                    continue
                if line_diff < 0:
                    valid_sides = ['Over']
                elif line_diff > 0:
                    valid_sides = ['Under']

            fd_over_odds = fd_row['over_price']
            fd_under_odds = fd_row['under_price']

            true_over, true_under = self._calculate_true_probability(fd_over_odds, fd_under_odds)

            if 'Over' in valid_sides:
                opportunities.append({
                    "Date": pp_date,
                    "Player": pp_name,
                    "League": self.league,   # ← no longer hardcoded 'NBA'
                    "Stat": pp_stat,
                    "Line": pp_line,
                    "Side": "Over",
                    "Implied_Win_%": round(true_over * 100, 2),
                    "FD_Odds": fd_over_odds
                })

            if 'Under' in valid_sides:
                opportunities.append({
                    "Date": pp_date,
                    "Player": pp_name,
                    "League": self.league,   # ← no longer hardcoded 'NBA'
                    "Stat": pp_stat,
                    "Line": pp_line,
                    "Side": "Under",
                    "Implied_Win_%": round(true_under * 100, 2),
                    "FD_Odds": fd_under_odds
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
        def odds_to_prob(odds):
            if odds < 0:
                return (-odds) / ((-odds) + 100)
            else:
                return 100 / (odds + 100)

        prob_over = odds_to_prob(over_odds)
        prob_under = odds_to_prob(under_odds)
        market_total = prob_over + prob_under
        true_over_prob = prob_over / market_total
        true_under_prob = prob_under / market_total

        return true_over_prob, true_under_prob


# --- TEST BLOCK ---
if __name__ == "__main__":
    print("--- TESTING ANALYZER LOGIC ---")

    pp_data = {
        'Player': ['LeBron James'],
        'Stat': ['Points'],
        'Line': [25.5],
        'Date': ['2026-02-12']
    }
    pp_df = pd.DataFrame(pp_data)

    fd_data = [
        {'Player': 'LeBron James', 'Stat': 'Points', 'Line': 25.5, 'Odds': -120, 'Side': 'Over', 'Date': '2026-02-12'},
        {'Player': 'LeBron James', 'Stat': 'Points', 'Line': 25.5, 'Odds': -110, 'Side': 'Under', 'Date': '2026-02-12'}
    ]
    fd_df = pd.DataFrame(fd_data)

    analyzer = PropsAnalyzer(pp_df, fd_df, league='NBA')
    results = analyzer.calculate_edges()

    if not results.empty:
        print("\n✅ Success! Found edges:")
        print(results[['Player', 'Side', 'Implied_Win_%', 'FD_Odds']])
    else:
        print("\n❌ No edges found.")
