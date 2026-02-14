"""
sports_ev_bot - Multi-Sport Entry Point

Interactive menu system for NBA, CBB, and upcoming sports betting analysis.

Features:
    - Sport selection menu
    - Separate workflows for each sport
    - Shared core functionality (FanDuel, PrizePicks APIs)
    
Sports Supported:
    - NBA (Professional Basketball) - ACTIVE
    - CBB (College Basketball) - ACTIVE
    - WNBA (Women's Basketball) - COMING SOON
    - MLB (Major League Baseball) - COMING SOON
    - NFL (Football) - COMING SOON

Usage:
    $ python main.py
    
Then select your sport and follow the prompts.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main entry point - Sport selection menu.
    
    Workflow:
        1. Display sport menu
        2. User selects sport
        3. Launch sport-specific CLI
        4. Return to sport menu (or exit)
    """
    while True:
        # Clear screen for clean display
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*60)
        print("  " + "🏀"*8 + "  SPORTS ANALYTICS HUB  " + "🏀"*8)
        print("="*60)
        
        print("\n🎯 PROFESSIONAL BETTING ANALYSIS PLATFORM")
        print("   AI-Powered Predictions • Multi-Sport Support • Real-Time Odds\n")
        
        print("="*60)
        print("SELECT SPORT")
        print("="*60)
        
        # Active Sports
        print("\n🟢 ACTIVE")
        
        print("\n1. 🏀 NBA")
        print("   Professional Basketball")
        print("   ⭐ Elite Models: PTS (89%), FGM (88%), PA (87%)")
        print("   📊 Active Models: 13 stats")
        print("   💰 PrizePicks Breakeven: 54.1%")
        
        # Coming Soon Sports
        print("\n" + "-"*60)
        print("🔜 COMING SOON")
        
        print("\n2. 🏀 WNBA")
        print("   Women's Basketball")
        print("   📅 Target: May 2026")
        
        print("\n3. ⚾ MLB")
        print("   Major League Baseball")
        print("   📅 Target: Spring 2026")
        
        print("\n4. 🏈 NFL")
        print("   Football")
        print("   📅 Target: Summer 2026")
        
        print("\n" + "="*60)
        print("\n0. 🚪 Exit")
        
        print("\n" + "="*60)
        
        choice = input("\nSelect Sport (1-4, 0 to exit): ").strip()
        
        # ================================================================
        # ACTIVE SPORTS
        # ================================================================
        
        if choice == '1':
            # Launch NBA
            try:
                from src.cli.nba_cli import main_menu as nba_menu
                nba_menu()
            except ImportError as e:
                print(f"\n❌ Error loading NBA module: {e}")
                print("   Make sure src/cli/nba_cli.py exists")
                input("\nPress Enter to continue...")
            except Exception as e:
                print(f"\n❌ NBA module error: {e}")
                input("\nPress Enter to continue...")
        
        # ================================================================
        # COMING SOON SPORTS
        # ================================================================
        
        elif choice == '2':
            # WNBA placeholder
            show_coming_soon_wnba()
        
        elif choice == '3':
            # MLB placeholder
            show_coming_soon_mlb()
        
        elif choice == '4':
            # NFL placeholder
            show_coming_soon_nfl()
        
        # ================================================================
        # EXIT
        # ================================================================
        
        elif choice == '0':
            # Exit
            print("\n" + "="*60)
            print("  👋 GOODBYE!")
            print("="*60)
            print("\n📊 Session Summary:")
            print("   Thanks for using Sports Analytics Hub")
            print("   Good luck with your bets! 🎯")
            print("\n💡 Tips:")
            print("   - Stick to ELITE tier models (highest accuracy)")
            print("   - Check injury reports before betting")
            print("   - Manage bankroll wisely (never bet more than 3%)")
            print("\n🔮 Coming Soon: WNBA (May 2026), MLB (Spring 2026), NFL (Summer 2026)")
            print("\n" + "="*60 + "\n")
            break
        
        else:
            print("\n❌ Invalid selection. Please choose 1-5 or 0.")
            input("Press Enter to try again...")

# ============================================================================
# COMING SOON SCREENS
# ============================================================================

def show_coming_soon_wnba():
    """Display WNBA coming soon information."""
    print("\n" + "="*60)
    print("  🏀 WNBA MODULE - IN DEVELOPMENT")
    print("="*60)
    
    print("\n📊 Overview:")
    print("   - 40 games/season (manageable sample size)")
    print("   - Similar dynamics to NBA")
    print("   - Growing betting market")
    print("   - Season: May - October")
    
    print("\n🔧 Planned Features:")
    print("   ✅ Tempo-adjusted predictions")
    print("   ✅ Usage vacuum (injury impact)")
    print("   ✅ Efficiency metrics (TS%, Usage Rate)")
    print("   ✅ Defensive matchups by position")
    print("   ✅ Home/away splits")
    print("   ✅ Back-to-back game detection")
    
    print("\n📈 Expected Performance:")
    print("   - Target: 75-80% directional accuracy")
    print("   - Similar to CBB (smaller sample than NBA)")
    
    print("\n🛠️  Implementation:")
    print("   - Adapting NBA codebase")
    print("   - WNBA API for data collection")
    print("   - Separate models for 12 teams")
    
    print("\n📅 Timeline:")
    print("   - Development: April 2026")
    print("   - Testing: May 2026 (season start)")
    print("   - Production: June 2026")
    
    input("\n📅 Estimated Release: May 2026 | Press Enter to return...")

def show_coming_soon_mlb():
    """Display MLB coming soon information."""
    print("\n" + "="*60)
    print("  ⚾ MLB MODULE - PLANNING PHASE")
    print("="*60)
    
    print("\n📊 Overview:")
    print("   - 162 games/season (large sample)")
    print("   - Pitcher matchups are CRITICAL")
    print("   - Park factors matter significantly")
    print("   - Weather affects outcomes")
    print("   - Season: April - October")
    
    print("\n🔧 Planned Features:")
    print("   ✅ Pitcher vs Batter matchups")
    print("   ✅ Park factors (dimensions, altitude)")
    print("   ✅ Weather integration (wind, temp)")
    print("   ✅ Platoon splits (vs LHP/RHP)")
    print("   ✅ Bullpen strength")
    print("   ✅ Vegas total correlation")
    print("   ✅ Umpire tendencies")
    
    print("\n📈 Target Props:")
    print("   - Hits, Home Runs, RBIs, Stolen Bases")
    print("   - Strikeouts (pitcher & batter)")
    print("   - Total Bases")
    print("   - Pitcher Outs, Earned Runs")
    
    print("\n⚠️  Challenges:")
    print("   - Complex pitcher/batter history")
    print("   - Weather data integration")
    print("   - Park-specific adjustments")
    
    print("\n📚 Data Sources:")
    print("   - Baseball Savant (Statcast)")
    print("   - FanGraphs")
    print("   - MLB Stats API")
    
    print("\n📅 Timeline:")
    print("   - Research: Feb-Mar 2026")
    print("   - Development: Mar-Apr 2026")
    print("   - Testing: Apr 2026 (Opening Day)")
    print("   - Production: May 2026")
    
    input("\n📅 Estimated Release: Spring 2026 | Press Enter to return...")

def show_coming_soon_nfl():
    """Display NFL coming soon information."""
    print("\n" + "="*60)
    print("  🏈 NFL MODULE - PLANNING PHASE")
    print("="*60)
    
    print("\n📊 Overview:")
    print("   - Only 17 games/season (small sample!)")
    print("   - High variance sport")
    print("   - Vegas lines are very sharp")
    print("   - Weather critical for outdoor games")
    print("   - Season: September - February")
    
    print("\n🔧 Planned Features:")
    print("   ✅ Weather integration (wind, rain, snow)")
    print("   ✅ Vegas spread correlation")
    print("   ✅ Snap count percentage")
    print("   ✅ Target share (WR/TE)")
    print("   ✅ Game script modeling")
    print("   ✅ Defensive matchups")
    print("   ✅ Injury impact (especially O-line)")
    
    print("\n📈 Target Props:")
    print("   - Passing: Yards, TDs, Completions")
    print("   - Rushing: Yards, Attempts, TDs")
    print("   - Receiving: Receptions, Yards, TDs")
    print("   - Defense: Tackles, Sacks")
    
    print("\n⚠️  Critical Challenges:")
    print("   - Small sample size (17 games)")
    print("   - High variance")
    print("   - Sharp Vegas lines")
    print("   - Weather unpredictability")
    
    print("\n🎯 Approach:")
    print("   - Smaller rolling windows (L3, L6)")
    print("   - Heavy Vegas correlation")
    print("   - Mandatory weather API")
    print("   - Target: 65-70% accuracy")
    
    print("\n📅 Timeline:")
    print("   - Research: May-Jun 2026")
    print("   - Development: Jul-Aug 2026")
    print("   - Testing: Sep 2026 (Week 1)")
    print("   - Production: Oct 2026")
    
    input("\n📅 Estimated Release: Summer 2026 | Press Enter to return...")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!\n")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("\nPlease report this issue if it persists.")