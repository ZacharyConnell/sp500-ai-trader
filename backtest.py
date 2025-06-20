import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
import os

DATA_FILE = "data/sp500_data.csv"
MODEL_FILE = "models/lstm_model.h5"
SUMMARY_FILE = "data/backtest_summary.csv"
TRADE_LOG_FILE = "data/backtest_trades.csv"
EQUITY_PLOT_DIR = "data/equity_curves"

# --- Backtest parameters ---
STEPS = 10
THRESHOLD = 0.8
MIN_TRADES = 5
START_DATE = None  # e.g., "2023-01-01"

def compute_indicators(df):
    df['MA'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).mean())
    df['STD'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).std())

    def rsi(x, period=14):
        delta = x.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))

    df['RSI'] = df.groupby("Ticker")["Price"].transform(rsi)
    return df.dropna()

def backtest_all():
    model = load_model(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = compute_indicators(df).sort_values(["Ticker", "Timestamp"])

    if START_DATE:
        df = df[df["Timestamp"] >= pd.to_datetime(START_DATE)]

    features = ["Price", "Sentiment", "MA", "STD", "RSI"]
    summary = []
    trades = []

    os.makedirs(EQUITY_PLOT_DIR, exist_ok=True)

    for ticker in df['Ticker'].unique():
        sub = df[df['Ticker'] == ticker].reset_index(drop=True)
        if len(sub) < STEPS + 2:
            continue

        # Fit a scaler for this ticker's data
        scaler = MinMaxScaler()
        scaler.fit(sub[features])
        equity = [1]
        returns = []
        signal_log = []

        try:
            for i in range(STEPS, len(sub) - 1):
                X_input = sub[features].iloc[i - STEPS:i]
                X_scaled = scaler.transform(X_input)
                X = np.expand_dims(X_scaled, axis=0)

                pred = model.predict(X, verbose=0)[0][0]
                today_price = sub['Price'].iloc[i]
                next_price = sub['Price'].iloc[i + 1]
                pct_change = (next_price - today_price) / today_price
                ts = sub['Timestamp'].iloc[i]

                if pred > THRESHOLD:
                    returns.append(pct_change)
                    signal_log.append((ts, "Buy", pct_change))
                    equity.append(equity[-1] * (1 + pct_change))
                elif pred < 1 - THRESHOLD:
                    returns.append(-pct_change)
                    signal_log.append((ts, "Sell", -pct_change))
                    equity.append(equity[-1] * (1 - pct_change))
                else:
                    signal_log.append((ts, "Hold", 0))
                    equity.append(equity[-1])
        except Exception as e:
            print(f"⚠️ {ticker} skipped: {e}")
            continue

        if len(returns) >= MIN_TRADES:
            total_return = equity[-1] - 1
            win_rate = np.mean([r > 0 for r in returns])
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) else 0

            summary.append({
                "Ticker": ticker,
                "Trades": len(returns),
                "Total Return %": round(total_return * 100, 2),
                "Win Rate %": round(win_rate * 100, 2),
                "Sharpe Ratio": round(sharpe, 2)
            })

            for t, signal, r in signal_log:
                if signal != "Hold":
                    trades.append({
                        "Ticker": ticker,
                        "Date": t,
                        "Signal": signal,
                        "Return": round(r * 100, 2)
                    })

            # Save the equity curve for the ticker as a PNG image.
            plt.figure()
            plt.plot(equity, color="green" if total_return > 0 else "red")
            plt.title(f"{ticker} Equity Curve")
            plt.xlabel("Signal Index")
            plt.ylabel("Cumulative Return")
            plt.tight_layout()
            plt.savefig(f"{EQUITY_PLOT_DIR}/{ticker}_equity.png")
            plt.close()

    summary_df = pd.DataFrame(summary).sort_values("Total Return %", ascending=False)
    trades_df = pd.DataFrame(trades)

    os.makedirs("data", exist_ok=True)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    trades_df.to_csv(TRADE_LOG_FILE, index=False)

    print(f"✅ Backtest complete. Saved to {SUMMARY_FILE}")
    print(summary_df.head(10))

if __name__ == "__main__":
    backtest_all()