import pandas as pd
from fuzzywuzzy import process
from src.config import SLIP_CONFIG

class PropsAnalyzer:
    def __init__(self, prizepicks_df, fanduel_df):
        self.pp_df = prizepicks_df
        self.fd_df = fanduel_df

    def calculate_edges(self):
        """
        Iterates through PrizePicks lines, finds the matching FanDuel line,
        calculates the 'True Probability', and 
        checks if it beats the Hurdle Rate.
        """
        opportunities = []
        
        # 1. Loop through every row in the PrizePicks DataFrame
        for index, pp_row in self.pp_df.iterrows():
            
            # getting all the stats from prizepicks
            pp_name = pp_row['Player']
            pp_stat = pp_row['Stat']
            pp_line = pp_row['Line']

            fd_name, fd_rows = self._find_match_in_fanduel(pp_name)

            # This checks if the row is empty
            if fd_name is None:
                continue

            # Check if the lines match 
            matching_stat = fd_rows[fd_rows['Stat'] == pp_stat]
            
            if matching_stat.empty:
                continue

            fd_line = matching_stat.iloc[0]['Line']
            line_diff = pp_line - fd_line

            # 1. Define Valid Sides
            # If lines match, both sides are valid to check
            valid_sides = ['Over', 'Under']

            # 2. Handle Discrepancies
            if line_diff != 0:
                # If the difference is huge 
                # (e.g. 5 points), it's probably bad data. Skip.
                if abs(line_diff) > 1.5:
                    continue

                # PP (10.5) < FD (11.5) -> The OVER 
                # is easier on PP. The UNDER is harder.
                if line_diff < 0:
                    valid_sides = ['Over'] # We only care about the Over
                
                # PP (11.5) > FD (10.5) -> The UNDER 
                # is easier on PP. The OVER is harder.
                elif line_diff > 0:
                    valid_sides = ['Under']
            
            # getting the true probablity from fanduel.
            fd_over_odds = matching_stat.iloc[0]['over_price']
            fd_under_odds = matching_stat.iloc[0]['under_price']

            true_over, true_under = self._calculate_true_probability(
                                    fd_over_odds, 
                                    fd_under_odds)
            
            # looping through to see if that are any props that match the
            # params setup to consider it as a good option for a certain bet.
            for slip_type, data in SLIP_CONFIG.items():
                hurdle = data['hurdle']

                if 'Over' in valid_sides and true_over * 100 > hurdle:
                    opportunities.append({
                        "Player": pp_name,
                        "League": "NBA",
                        "Stat": pp_stat,
                        "Line": pp_line,
                        "Side": "Over",
                        "Implied_Win_%": round(true_over * 100, 2),
                        "Slip_Type": slip_type,
                        "Hurdle": hurdle,
                        "FD_Odds": fd_over_odds
                    })

                # Check the UNDER -> ADD "and 'Under' in valid_sides"
                if 'Under' in valid_sides and true_under * 100 > hurdle:
                    opportunities.append({
                        "Player": pp_name,
                        "League": "NBA",
                        "Stat": pp_stat,
                        "Line": pp_line,
                        "Side": "Under",
                        "Implied_Win_%": round(true_under * 100, 2),
                        "Slip_Type": slip_type,
                        "Hurdle": hurdle,
                        "FD_Odds": fd_under_odds
                    })

        return pd.DataFrame(opportunities)

    def _find_match_in_fanduel(self, pp_name):
        """
        Helper: Uses fuzzy matching to find 
        the closest player name in self.fd_df.
        Returns: (actual_name, row_data) or (None, None)
        """
        fd_unique_name = self.fd_df['Player'].unique()
        
        match_name, score = process.extractOne(pp_name, fd_unique_name)
        
        # checks if you are 85 percent sure that the name is the same
        if score < 85:
            return None, None
        
        player_rows = self.fd_df[self.fd_df['Player'] == match_name]

        return match_name, player_rows

    def _calculate_true_probability(self, over_odds, under_odds):
        """
        Helper: Converts American Odds 
        (e.g. -110, -110) to a Percentage (0-100).
        """
        
        prob_over = 0
        prob_under = 0
        
        if over_odds < 0:
            prob_over = (-over_odds) / ((-over_odds) + 100)
        elif over_odds > 0:
            prob_over = (100) / (over_odds + 100)

        if under_odds < 0:
            prob_under = (-under_odds) / ((-under_odds) + 100)
        elif under_odds > 0:
            prob_under = (100) / (under_odds + 100)

        Market_Total = prob_over + prob_under

        true_over_prob = prob_over / Market_Total
        true_under_prob = prob_under / Market_Total

        return true_over_prob, true_under_prob
    
# Testing Below

if __name__ == "__main__":
    print("--- TESTING ANALYZER LOGIC ---")
    
    # 1. Create Fake PrizePicks Data
    # We need columns: 'Player', 'Stat', 'Line'
    mock_pp_data = {
        'Player': ['LeBron James', 'Steph Curry'], 
        'Stat': ['Points', 'Points'],
        'Line': [25.5, 29.5] 
    }
    pp_df = pd.DataFrame(mock_pp_data)

    # 2. Create Fake FanDuel Data
    # We need columns: 'Player', 'Stat', 'Line', 'over_price', 'under_price'
    # Scenario A: LeBron is a HUGE favorite to go OVER (Should be a Win)
    # Scenario B: Steph is a normal bet (Should be ignored)
    mock_fd_data = {
        'Player': ['LeBron James', 'Stephen Curry'], 
        'Stat': ['Points', 'Points'],
        'Line': [25.5, 29.5],
        'over_price': [-200, -110],  # -200 is massive odds (66%)
        'under_price': [150, -110]
    }
    fd_df = pd.DataFrame(mock_fd_data)

    # 3. Run the Analyzer
    print("Running analysis on mock data...")
    analyzer = PropsAnalyzer(pp_df, fd_df)
    results = analyzer.calculate_edges()

    # 4. Check Results
    if not results.empty:
        print("\nFound these edges:")
        print(results)
    else:
        print("\nNo edges found. (Did something break?)")