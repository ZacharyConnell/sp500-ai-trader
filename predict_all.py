import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import shutil

# --- File Paths ---
DATA_FILE = "data/sp500_data.csv"
MODEL_FILE = "models/lstm_model.h5"
OUTPUT_FILE = "data/predictions_today.csv"
LOG_FILE = "data/predictions_log.csv"
HISTORY_DIR = "data/history"
BACKUP_DIR = "models/backups"

# --- Configs ---
EXPORT_TIMESTAMPED_CSV = True
BACKUP_MODEL = True

# --- Feature Engineering ---
def engineer_features(df):
    df['MA'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).mean())
    df['STD'] = df.groupby('Ticker')['Price'].transform(lambda x: x.rolling(5).std())

    def rsi(series, period=14):
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))

    df['RSI'] = df.groupby('Ticker')['Price'].transform(lambda x: rsi(x))
    return df.dropna()

def get_sector_mapping():
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return dict(zip(table["Symbol"], table["GICS Sector"]))
    except Exception as e:
        print(f"⚠️ Failed to fetch sector mapping: {e}")
        return {}

def calibrate_confidence(prob, temperature=1.5):
    logit = np.log(prob / (1 - prob + 1e-8))
    scaled_logit = logit / temperature
    return 1 / (1 + np.exp(-scaled_logit))

def interpret_vol_class(vol_probs):
    return ["Low", "Medium", "High"][np.argmax(vol_probs)]

# --- Main Prediction Function ---
def predict_all():
    if not os.path.exists(MODEL_FILE):
        print("❌ Model not found. Run model.py first.")
        return

    model = load_model(MODEL_FILE)
    df = pd.read_csv(DATA_FILE).sort_values(by=["Ticker", "Timestamp"])
    df = engineer_features(df)
    sector_map = get_sector_mapping()

    features = ['Price', 'Sentiment', 'MA', 'STD', 'RSI']
    predictions = []

    for ticker in df['Ticker'].unique():
        sub = df[df['Ticker'] == ticker].tail(10)
        if len(sub) < 10:
            continue
        try:
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(sub[features])
            X = np.expand_dims(X_scaled, axis=0)

            pred_class, pred_return, pred_vol = model.predict(X, verbose=0)
            raw_prob = pred_class[0][0]
            confidence = calibrate_confidence(raw_prob)
            est_return = pred_return[0][0] * 100
            vol_class = interpret_vol_class(pred_vol[0])
            prediction = "↑" if confidence > 0.5 else "↓"

            action = (
                "Buy" if confidence > 0.6 and est_return > 1 else
                "Sell" if confidence < 0.4 and est_return < -1 else
                "Hold"
            )

            latest = sub.iloc[-1]
            predictions.append({
                "Ticker": ticker,
                "Sector": sector_map.get(ticker, "Unknown"),
                "Price": round(latest["Price"], 2),
                "Sentiment": round(latest["Sentiment"], 3),
                "Prediction": prediction,
                "Confidence": round(confidence * 100, 2),
                "Expected Return %": round(est_return, 2),
                "Volatility Class": vol_class,
                "Suggested Action": action,
                "Timestamp": latest["Timestamp"]
            })
        except Exception as e:
            print(f"⚠️ {ticker} skipped: {e}")

    df_preds = pd.DataFrame(predictions).sort_values("Confidence", ascending=False)
    os.makedirs("data", exist_ok=True)
    df_preds.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Predictions saved to {OUTPUT_FILE}")

    if EXPORT_TIMESTAMPED_CSV:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d")
        archive_file = f"{HISTORY_DIR}/predictions_{timestamp_str}.csv"
        df_preds.to_csv(archive_file, index=False)
        print(f"🗂️ Archived to {archive_file}")

    if os.path.exists(LOG_FILE):
        existing = pd.read_csv(LOG_FILE)
        combined = pd.concat([existing, df_preds]).drop_duplicates(
            subset=["Ticker", "Timestamp"], keep="last"
        )
        combined["Timestamp"] = pd.to_datetime(combined["Timestamp"])
        combined = combined[combined["Timestamp"] >= datetime.now() - pd.Timedelta(days=30)]
    else:
        combined = df_preds

    combined.to_csv(LOG_FILE, index=False)
    print(f"🕒 Historical log saved to {LOG_FILE}")

    if BACKUP_MODEL:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_name = f"lstm_model_{datetime.now().strftime('%Y-%m-%d')}.h5"
        shutil.copyfile(MODEL_FILE, f"{BACKUP_DIR}/{backup_name}")
        print(f"📦 Model backed up as {backup_name}")

if __name__ == "__main__":
    predict_all()