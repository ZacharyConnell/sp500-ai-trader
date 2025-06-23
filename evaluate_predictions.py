import pandas as pd
import os
import glob

PRED_PATH = "data/history"
DATA_PATH = "data/sp500_data.csv"

# Step 1: Locate the most recent predictions file
files = sorted(glob.glob(f"{PRED_PATH}/predictions_*.csv"))
if not files:
    print("❌ No prediction files found in history.")
    exit()

latest_file = files[-1]
print(f"🗂️ Evaluating predictions in: {latest_file}")

# Step 2: Load predictions and historical data
df_preds = pd.read_csv(latest_file)
df_data = pd.read_csv(DATA_PATH)

# Step 3: Evaluation logic
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

# Step 4: Save results back to the same file
pd.DataFrame(results).to_csv(latest_file, index=False)
print("✅ Evaluation complete. Updated:", latest_file)