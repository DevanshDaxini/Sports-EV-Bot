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
            
            # --- YOUR HOMEWORK STARTS HERE ---
            
            # TODO: Get the player name, stat type 
            # (Points, etc), and line from pp_row
            
            # TODO: Call self._find_match_in_fanduel() 
            # to see if this player exists in FanDuel data
            # match_name, match_row = ...
            
            # TODO: If no match is found, continue to the next iteration
            
            # TODO: Check if the lines match 
            # (e.g. is PrizePicks 24.5 == FanDuel 24.5?)
            # Advanced: Later you can add logic to 
            # handle 10.5 vs 11.5 differences.
            
            # TODO: Calculate the 'True Probability' 
            # (remove the vig) using the helper function
            # true_prob = self._calculate_true_probability(...)
            
            # TODO: Check against SLIP_CONFIG
            # Loop through '2_man_power', 
            # '5_man_flex', etc. in SLIP_CONFIG.
            # If true_prob > slip_config['hurdle'], 
            # append it to the 'opportunities' list.
            
            pass 

        return pd.DataFrame(opportunities)

    def _find_match_in_fanduel(self, pp_name):
        """
        Helper: Uses fuzzy matching to find 
        the closest player name in self.fd_df.
        Returns: (actual_name, row_data) or (None, None)
        """
        # TODO: Get all unique player names from self.fd_df
        
        # TODO: Use process.extractOne(pp_name, all_fd_names) 
        # to find the best match
        
        # TODO: Check the score. If < 85 (or your threshold), return None.
        
        # TODO: Return the row from self.fd_df that corresponds to that name
        return None, None

    def _calculate_true_probability(self, over_odds, under_odds):
        """
        Helper: Converts American Odds 
        (e.g. -110, -110) to a Percentage (0-100).
        """
        # TODO: Convert American Odds to Implied Probability (Decimal)
        # Formula: if odds < 0: prob = (-odds) / (-odds + 100)
        
        # TODO: Sum the implied probabilities of 
        # Over + Under (It will be > 100% due to juice)
        
        # TODO: Remove the juice (Devig)
        # Formula: True_Over = Implied_Over / (Implied_Over + Implied_Under)
        
        return 0.0