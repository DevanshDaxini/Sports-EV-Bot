import pandas as pd
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import matplotlib.ticker as mtick

def plot_feature_importance():
    """Visualizes which stats are weighting the model most."""
    
    # CORRECTED: Point explicitly to the .json file
    model_path = 'models/PTS_model.json'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Could not find {model_path}")
        print("   -> Check if your folder is named 'models' and contains .json files")
        return

    # Load the model using XGBoost's native JSON loader
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # Get importance
    importance = model.get_booster().get_score(importance_type='weight')
    importance = pd.Series(importance).sort_values(ascending=True)

    # Plot
    plt.figure(figsize=(10, 8))
    importance.plot(kind='barh', color='skyblue')
    plt.title('Feature Importance: What is the Model Weighting?')
    plt.xlabel('Weight (F-Score)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("✅ Saved: feature_importance.png")

def plot_win_rate():
    """Visualizes accuracy with bulletproof data cleaning."""
    history_file = "program_runs/win_rate_history.csv"
    if not os.path.exists(history_file):
        print("⚠️ No history file found yet.")
        return

    df = pd.read_csv(history_file)
    
    # --- CRITICAL FIX: FORCE NUMERIC CONVERSION ---
    # 1. Ensure it's a string, strip '%', and coerce errors to NaN
    # This handles "50%", "0.5", and even garbage text like "Win_Rate" in the middle of the file
    df['Win_Rate'] = pd.to_numeric(
        df['Win_Rate'].astype(str).str.replace('%', ''), 
        errors='coerce'
    )
    
    # 2. Drop any rows that couldn't be converted (like repeated headers)
    df = df.dropna(subset=['Win_Rate'])
    
    # 3. Auto-Scale: If data is 0.51, convert to 51.0
    if df['Win_Rate'].mean() < 1.0:
        print("ℹ️ Detected decimal format (e.g., 0.51). Converting to percentage.")
        df['Win_Rate'] = df['Win_Rate'] * 100
        
    # ----------------------------------------------

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Plot Green Line
    plt.plot(df['Date'], df['Win_Rate'], marker='o', markersize=8, 
             linestyle='-', color='green', linewidth=3, label='Model Accuracy')
    
    # Plot Red Breakeven Line
    plt.axhline(y=54.1, color='red', linestyle='--', linewidth=2, label='PrizePicks Breakeven (54.1%)')
    
    # Y-Axis Settings (40% to 60%)
    plt.ylim(40, 60)
    plt.yticks(range(40, 61, 2)) # Ticks every 2%
    
    plt.title('NBA Bot Accuracy Tracker', fontsize=16, fontweight='bold')
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('win_rate_trend.png')
    print("✅ Saved: win_rate_trend.png (Cleaned & Scaled)")

if __name__ == "__main__":
    print("📊 Generating Visualizations...")
    plot_feature_importance()
    plot_win_rate()
    print("🚀 Done! Check your folder for .png files.")
    
#To run: python3 -m src.visualizer