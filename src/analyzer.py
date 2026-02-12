import pandas as pd
from fuzzywuzzy import process
from src.config import SLIP_CONFIG

class PropsAnalyzer:
    def __init__(self, prizepicks_df, fanduel_df):
        self.pp_df = prizepicks_df
        self.fd_df = fanduel_df

    def calculate_edges(self):
        """
        1. Reshapes FanDuel data to have Over/Under odds in one row.
        2. Iterates through PrizePicks lines to find matches.
        3. Calculates 'True Probability' and returns profitable plays.
        """
        opportunities = []
        
        # --- STEP 1: RESHAPE FANDUEL DATA (Long -> Wide) ---
        # The FD scanner gives us separate rows for Over and Under.
        # We need to merge them into one row per line to do the math.
        
        if self.fd_df.empty:
            return pd.DataFrame()

        # Separate the Overs and Unders
        fd_over = self.fd_df[self.fd_df['Side'] == 'Over'].copy()
        fd_under = self.fd_df[self.fd_df['Side'] == 'Under'].copy()
        
        # Rename 'Odds' column to specific price columns
        fd_over = fd_over.rename(columns={'Odds': 'over_price'})
        fd_under = fd_under.rename(columns={'Odds': 'under_price'})
        
        # Drop 'Side' column as it is no longer needed after split
        fd_over = fd_over.drop(columns=['Side'], errors='ignore')
        fd_under = fd_under.drop(columns=['Side'], errors='ignore')
        
        # Merge them back together on Player, Stat, Line, and Date
        # We use 'inner' merge because we need BOTH sides to calculate fair odds
        self.fd_wide = pd.merge(
            fd_over, 
            fd_under, 
            on=['Player', 'Stat', 'Line', 'Date'], 
            how='inner'
        )
        # ---------------------------------------------------

        # 2. Loop through every row in the PrizePicks DataFrame
        for index, pp_row in self.pp_df.iterrows():
            
            pp_name = pp_row['Player']
            pp_stat = pp_row['Stat']
            pp_line = pp_row['Line']
            pp_date = pp_row.get('Date', 'Unknown')

            # Find matching player in our new "Wide" FanDuel data
            fd_name, fd_rows = self._find_match_in_fanduel(pp_name)

            if fd_name is None:
                continue

            # Check if the stats match (e.g. Points vs Points)
            matching_stat = fd_rows[fd_rows['Stat'] == pp_stat]
            
            if matching_stat.empty:
                continue

            # Get the matching FanDuel line
            # (In rare cases duplicates might exist, we take the first one)
            fd_row = matching_stat.iloc[0]
            fd_line = fd_row['Line']
            
            line_diff = pp_line - fd_line

            # 1. Define Valid Sides
            valid_sides = ['Over', 'Under']

            # 2. Handle Discrepancies (Line differences)
            if line_diff != 0:
                # If the line differs by too much (e.g. 15.5 vs 22.5), it's probably a different market
                if abs(line_diff) > 1.5:
                    continue

                # If PP line is lower (20.5 vs 21.5), the OVER is easier to hit
                if line_diff < 0:
                    valid_sides = ['Over'] 
                # If PP line is higher (22.5 vs 21.5), the UNDER is easier to hit
                elif line_diff > 0:
                    valid_sides = ['Under']
            
            # 3. Calculate True Probability
            # Now we can safely access these columns because we created them in Step 1
            fd_over_odds = fd_row['over_price']
            fd_under_odds = fd_row['under_price']

            true_over, true_under = self._calculate_true_probability(
                                    fd_over_odds, 
                                    fd_under_odds)
            
            # 4. Store Opportunities
            if 'Over' in valid_sides:
                opportunities.append({
                    "Date": pp_date,
                    "Player": pp_name,
                    "League": "NBA",
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
                    "League": "NBA",
                    "Stat": pp_stat,
                    "Line": pp_line,
                    "Side": "Under",
                    "Implied_Win_%": round(true_under * 100, 2),
                    "FD_Odds": fd_under_odds
                })

        return pd.DataFrame(opportunities)

    def _find_match_in_fanduel(self, pp_name):
        """
        Helper: Uses fuzzy matching to find 
        the closest player name in the transformed Fanduel data.
        """
        # We search in self.fd_wide now, not self.fd_df
        if hasattr(self, 'fd_wide') and not self.fd_wide.empty:
            search_df = self.fd_wide
        else:
            return None, None

        fd_unique_name = search_df['Player'].unique()
        
        match_name, score = process.extractOne(pp_name, fd_unique_name)
        
        if score < 85:
            return None, None
        
        player_rows = search_df[search_df['Player'] == match_name]

        return match_name, player_rows

    def _calculate_true_probability(self, over_odds, under_odds):
        """
        Helper: Converts American Odds to True Probability (removing vig).
        """
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
    
    # Mock PrizePicks Data
    pp_data = {
        'Player': ['LeBron James'], 
        'Stat': ['Points'],
        'Line': [25.5],
        'Date': ['2026-02-12']
    }
    pp_df = pd.DataFrame(pp_data)

    # Mock FanDuel Data (Long Format - simulating your current issue)
    fd_data = [
        {'Player': 'LeBron James', 'Stat': 'Points', 'Line': 25.5, 'Odds': -120, 'Side': 'Over', 'Date': '2026-02-12'},
        {'Player': 'LeBron James', 'Stat': 'Points', 'Line': 25.5, 'Odds': -110, 'Side': 'Under', 'Date': '2026-02-12'}
    ]
    fd_df = pd.DataFrame(fd_data)

    print("Running analysis on mock data...")
    analyzer = PropsAnalyzer(pp_df, fd_df)
    results = analyzer.calculate_edges()

    if not results.empty:
        print("\n✅ Success! Found edges:")
        print(results[['Player', 'Side', 'Implied_Win_%', 'FD_Odds']])
    else:
        print("\n❌ No edges found.")