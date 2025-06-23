import pandas as pd
from datetime import datetime, timedelta
import os

PRED_PATH = "data/history"
DATA_PATH = "data/sp500_data.csv"

today = datetime.now().date()
yesterday = today - timedelta(days=1)

yesterday_file = f"{PRED_PATH}/predictions_{yesterday}.csv"
if not os.path.exists(yesterday_file):
    print("❌ No predictions file for yesterday.")
    exit()

df_preds = pd.read_csv(yesterday_file)
df_data = pd.read_csv(DATA_PATH)

def get_next_price(ticker, timestamp):
    subset = df_data[(df_data["Ticker"] == ticker) & (df_data["Timestamp"] > timestamp)]
    if subset.empty:
        return None
    return subset.iloc[0]["Price"]

results = []
for _, row in df_preds.iterrows():
    next_price = get_next_price(row["Ticker"], row["Timestamp"])
    if next_price is None:
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

pd.DataFrame(results).to_csv(yesterday_file, index=False)
print("✅ Evaluation complete. Updated yesterday's prediction file.")