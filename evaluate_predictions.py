import pandas as pd
import os
import glob
import re
from datetime import datetime

print("🧭 Current Working Directory:", os.getcwd())

PRED_PATH = "data/history"
DATA_PATH = "data/sp500_data.csv"

# Step 1: Gather prediction files with proper date format, skip placeholders
files = sorted([
    f for f in glob.glob(f"{PRED_PATH}/predictions_*.csv")
    if os.path.getsize(f) > 0 and re.search(r"predictions_\d{4}-\d{2}-\d{2}\.csv", f)
])

print("📁 Searching for:", f"{PRED_PATH}/predictions_*.csv")
print("🔍 Valid files found:", files)

if not completed_files:
    print("❌ No completed prediction files available for evaluation.")
    exit()

latest_file = completed_files[-1]

# Step 2: Get the most recent COMPLETED prediction file (excluding today)
today_str = datetime.now().strftime("%Y-%m-%d")
completed_files = [f for f in files if today_str not in f]

if not completed_files:
    print("❌ No completed prediction files found (today’s file excluded).")
    exit()

latest_file = completed_files[-1]
print(f"🗂️ Evaluating predictions in: {latest_file}")

# Step 3: Load predictions and market data
df_preds = pd.read_csv(latest_file)
df_data = pd.read_csv(DATA_PATH)

# Step 4: Evaluation logic
def get_next_price(ticker, timestamp):
    subset = df_data[(df_data["Ticker"] == ticker) & (df_data["Timestamp"] > timestamp)]
    if subset.empty:
        return None
    return subset.iloc[0]["Price"]

results = []
for _, row in df_preds.iterrows():
    next_price = get_next_price(row["Ticker"], row["Timestamp"])
    if next_price is None or pd.isna(next_price):
        outcome = "Unknown"
    else:
        delta = ((next_price - row["Price"]) / row["Price"]) * 100
        if row["Suggested Action"] == "Buy" and delta > 0.5:
            outcome = "Correct"
        elif row["Suggested Action"] == "Sell" and delta < -0.5:
            outcome = "Correct"
        elif row["Suggested Action"] == "Hold" and abs(delta) < 0.5:
            outcome = "Correct"
        else:
            outcome = "Incorrect"

    results.append({**row, "Comparison Price": next_price, "Actual Outcome": outcome})

# Step 5: Overwrite file with evaluated results
pd.DataFrame(results).to_csv(latest_file, index=False)
print("✅ Evaluation complete. Updated:", latest_file)