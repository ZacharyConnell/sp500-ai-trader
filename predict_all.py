import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import shutil
import sys

print("🐍 Python interpreter:", sys.executable)

# === File Paths ===
DATA_FILE = "data/sp500_data.csv"
MODEL_FILE = "models/lstm_model.keras"
OUTPUT_FILE = "data/predictions_today.csv"
LOG_FILE = "data/predictions_log.csv"
HISTORY_DIR = "data/history"
BACKUP_DIR = "models/backups"

# === Configs ===
EXPORT_TIMESTAMPED_CSV = True
BACKUP_MODEL = True
SEQUENCE_LENGTH = 10  # Match upgraded model.py
FEATURES = ['Price', 'Sentiment', 'MA', 'STD', 'RSI', 'Momentum', 'MACD', 'ATR', 'EMA_diff']

# === Feature Engineering ===
def engineer_features(df):
    df = df.copy()
    df['MA'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).mean())
    df['STD'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).std())
    df['RSI'] = df.groupby('Ticker')['Price'].transform(lambda x: rsi(x))
    df['Momentum'] = df.groupby('Ticker')['Price'].transform(lambda x: x.diff(3))
    df['MACD'] = (
        df.groupby('Ticker')['Price'].transform(lambda x: x.ewm(span=12).mean()) -
        df.groupby('Ticker')['Price'].transform(lambda x: x.ewm(span=26).mean())
    )
    df['ATR'] = df.groupby('Ticker')['Price'].transform(lambda x: x.pct_change().abs().rolling(14).mean())
    df['EMA_diff'] = (
        df.groupby('Ticker')['Price'].transform(lambda x: x.ewm(span=5).mean()) -
        df.groupby('Ticker')['Price'].transform(lambda x: x.ewm(span=20).mean())
    )
    df['RSI'] = df['RSI'].fillna(0)
    df = df.dropna(subset=['Sentiment', 'Price'])
    return df

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

def get_sector_encoding_map(df):
    df["Sector"] = df["Sector"].astype("category")
    return {sector: idx for idx, sector in enumerate(df["Sector"].cat.categories)}

def calibrate_confidence(prob, temperature=1.5):
    logit = np.log(prob / (1 - prob + 1e-8))
    scaled_logit = logit / temperature
    return 1 / (1 + np.exp(-scaled_logit))

def interpret_vol_class(vol_probs):
    return ["Low", "Medium", "High"][np.argmax(vol_probs)]

def explain_action(action, confidence, est_return):
    if action == "Buy":
        return f"High confidence ({round(confidence, 1)}%) and expected return > 1%"
    elif action == "Sell":
        return f"Low confidence ({round(confidence, 1)}%) and expected return < -1%"
    else:
        return f"Neutral outlook: return not strong or confidence moderate"

# === Main Prediction Function ===
def predict_all():
    if not os.path.exists(MODEL_FILE):
        print("❌ Model not found.")
        return
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        print("❌ Missing or empty data file.")
        return

    model = load_model(MODEL_FILE, compile=False)
    df = pd.read_csv(DATA_FILE).sort_values(["Ticker", "Timestamp"])
    df = engineer_features(df)

    if "Sector" not in df.columns:
        print("❌ Missing 'Sector' column.")
        return

    sector_map = get_sector_encoding_map(df)
    n_sectors = len(sector_map)
    predictions = []
    total = 0

    for ticker in df["Ticker"].unique():
        sub = df[df["Ticker"] == ticker].tail(SEQUENCE_LENGTH)
        if len(sub) < SEQUENCE_LENGTH:
            continue
        try:
            if sub[FEATURES].isnull().any().any():
                continue
            sector = sub["Sector"].iloc[-1]
            if sector not in sector_map:
                continue

            X_main = MinMaxScaler().fit_transform(sub[FEATURES])
            sector_onehot = np.zeros(n_sectors)
            sector_onehot[sector_map[sector]] = 1
            sector_seq = np.repeat(sector_onehot.reshape(1, -1), SEQUENCE_LENGTH, axis=0)
            X = np.concatenate([X_main, sector_seq], axis=1).reshape(1, SEQUENCE_LENGTH, -1)

            pred_class, pred_return, pred_vol = model.predict(X, verbose=0)
            raw_prob = pred_class[0][0]
            confidence = calibrate_confidence(raw_prob) * 100
            est_return = pred_return[0][0] * 100
            vol_class = interpret_vol_class(pred_vol[0])

            if confidence > 60 and est_return > 1:
                action = "Buy"
            elif confidence < 40 and est_return < -1:
                action = "Sell"
            else:
                action = "Hold"

            reason = explain_action(action, confidence, est_return)
            latest = sub.iloc[-1]

            predictions.append({
                "Ticker": ticker,
                "Sector": sector,
                "Price": round(latest["Price"], 2),
                "Sentiment": round(latest["Sentiment"], 3),
                "Prediction": "↑" if confidence > 50 else "↓",
                "Confidence": round(confidence, 2),
                "Expected Return %": round(est_return, 2),
                "Volatility Class": vol_class,
                "Suggested Action": action,
                "Reason": reason,
                "Timestamp": latest["Timestamp"]
            })
            total += 1
        except Exception as e:
            print(f"⚠️ {ticker}: {e}")
            continue

    if not predictions:
        print("⚠️ No predictions generated.")
        return

    df_preds = pd.DataFrame(predictions).sort_values("Confidence", ascending=False)

    if not df_preds[df_preds["Suggested Action"] == "Buy"].empty:
        print("\n⭐ Top Picks for Today:")
        print(df_preds[df_preds["Suggested Action"] == "Buy"][["Ticker", "Expected Return %", "Confidence", "Reason"]].head().to_string(index=False))
    else:
        print("\n📉 No high-confidence Buy signals today.")

    os.makedirs("data", exist_ok=True)
    df_preds.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Predictions saved to {OUTPUT_FILE}")

    if EXPORT_TIMESTAMPED_CSV:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d")
        archive = f"{HISTORY_DIR}/predictions_{ts}.csv"
        df_preds.to_csv(archive, index=False)
        print(f"🗂️ Archived to {archive}")

    try:
        existing = pd.read_csv(LOG_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing = pd.DataFrame()

    combined = pd.concat([existing, df_preds]).drop_duplicates(subset=["Ticker", "Timestamp"])
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"])
    combined = combined[combined["Timestamp"] >= datetime.now() - pd.Timedelta(days=30)]
    combined.to_csv(LOG_FILE, index=False)
    print(f"🕒 Log updated at {LOG_FILE}")

    if BACKUP_MODEL:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_name = f"lstm_model_{datetime.now().strftime('%Y-%m-%d')}.keras"
        shutil.copyfile(MODEL_FILE, f"{BACKUP_DIR}/{backup_name}")
        print(f"📦 Model backed up as {backup_name}")

    print(f"\n🔍 {total} ticker(s) processed.")

if __name__ == "__main__":
    predict_all()