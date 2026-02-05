from src.prizepicks import PrizePicksClient
from src.fanduel import FanDuelClient
from src.analyzer import PropsAnalyzer

def main():
    print("--- 1. Fetching PrizePicks Lines ---")
    pp = PrizePicksClient()
    pp_df = pp.fetch_board()
    print(f"Got {len(pp_df)} PrizePicks props.")

    print("\n--- 2. Fetching FanDuel Odds ---")
    fd = FanDuelClient()
    # Limit to 5 games for testing to save API quota
    fd_df = fd.get_all_odds() 
    print(f"Got {len(fd_df)} FanDuel props.")

    print("\n--- 3. Analyzing for +EV Bets ---")
    analyzer = PropsAnalyzer(pp_df, fd_df)
    edges = analyzer.calculate_edges()

    if not edges.empty:
        # Sort by Win % so the best bets are at the top
        best_bets = edges.sort_values(by='Implied_Win_%', 
                                      ascending=False).head(20)
        
        print("\nTOP BETS FOUND:")
        # detailed view
        print(best_bets[['Player', 'Stat', 'Side', 'Line', 
                         'Implied_Win_%', 'Slip_Type', 'Hurdle']])
        
        # Save to CSV
        edges.to_csv("final_picks.csv", index=False)
        print("\nSaved full list to 'final_picks.csv'")
    else:
        print("No bets found! (This might mean no " \
            "games match, or no edges > 54% today)")

if __name__ == "__main__":
    main()