import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import shutil

# --- File Paths ---
DATA_FILE = "data/sp500_data.csv"
MODEL_FILE = "models/lstm_model.keras"
OUTPUT_FILE = "data/predictions_today.csv"
LOG_FILE = "data/predictions_log.csv"
HISTORY_DIR = "data/history"
BACKUP_DIR = "models/backups"

# --- Configs ---
EXPORT_TIMESTAMPED_CSV = True
BACKUP_MODEL = True
SEQUENCE_LENGTH = 6  # match model.py's `steps=6`

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
    df['RSI'] = df['RSI'].fillna(0)
    df = df.dropna(subset=['Sentiment', 'MA', 'STD', 'Price'])
    return df

def get_sector_encoding_map(df):
    df["Sector"] = df["Sector"].astype("category")
    sector_categories = df["Sector"].cat.categories
    return {sector: idx for idx, sector in enumerate(sector_categories)}

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

# --- Main Prediction Function ---
def predict_all():
    if not os.path.exists(MODEL_FILE):
        print("❌ Model not found. Run model.py first.")
        return
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        print("❌ Data file is missing or empty. Run collector.py first.")
        return

    model = load_model(MODEL_FILE, compile=False)
    df = pd.read_csv(DATA_FILE).sort_values(by=["Ticker", "Timestamp"])
    df = engineer_features(df)

    if "Sector" not in df.columns:
        print("❌ Missing 'Sector' column in the data.")
        return

    sector_map = get_sector_encoding_map(df)
    n_sectors = len(sector_map)
    features = ['Price', 'Sentiment', 'MA', 'STD', 'RSI']
    predictions = []

    for ticker in df['Ticker'].unique():
        sub = df[df['Ticker'] == ticker].tail(SEQUENCE_LENGTH)
        if len(sub) < SEQUENCE_LENGTH:
            continue
        try:
            if sub[features].isnull().any().any():
                continue
            sector = sub["Sector"].iloc[-1]
            if sector not in sector_map:
                continue

            X_scaled = MinMaxScaler().fit_transform(sub[features])
            sector_onehot = np.zeros(n_sectors)
            sector_onehot[sector_map[sector]] = 1
            sector_3d = np.repeat(sector_onehot.reshape(1, -1), SEQUENCE_LENGTH, axis=0)
            X_full = np.concatenate([X_scaled, sector_3d], axis=1).reshape(1, SEQUENCE_LENGTH, -1)

            pred_class, pred_return, pred_vol = model.predict(X_full, verbose=0)
            raw_prob = pred_class[0][0]
            confidence = calibrate_confidence(raw_prob) * 100
            est_return = pred_return[0][0] * 100
            vol_class = interpret_vol_class(pred_vol[0])
            prediction = "↑" if confidence > 50 else "↓"

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
                "Prediction": prediction,
                "Confidence": round(confidence, 2),
                "Expected Return %": round(est_return, 2),
                "Volatility Class": vol_class,
                "Suggested Action": action,
                "Reason": reason,
                "Timestamp": latest["Timestamp"]
            })
        except Exception:
            continue

    if not predictions:
        print("⚠️ No predictions generated.")
        return

    df_preds = pd.DataFrame(predictions).sort_values("Confidence", ascending=False)

    # Top picks
    top_picks = df_preds[df_preds["Suggested Action"] == "Buy"]
    if not top_picks.empty:
        print("\n⭐ Top Picks for Today:")
        print(top_picks[["Ticker", "Expected Return %", "Confidence", "Reason"]].head(5).to_string(index=False))
    else:
        print("\n📉 No high-confidence Buy signals today.")

    # Export
    os.makedirs("data", exist_ok=True)
    df_preds.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Predictions saved to {OUTPUT_FILE}")

    if EXPORT_TIMESTAMPED_CSV:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d")
        archive_file = f"{HISTORY_DIR}/predictions_{timestamp_str}.csv"
        df_preds.to_csv(archive_file, index=False)
        print(f"🗂️ Archived to {archive_file}")

    try:
        existing = pd.read_csv(LOG_FILE)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        existing = pd.DataFrame()

    combined = pd.concat([existing, df_preds]).drop_duplicates(
        subset=["Ticker", "Timestamp"], keep="last"
    )
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"])
    combined = combined[combined["Timestamp"] >= datetime.now() - pd.Timedelta(days=30)]
    combined.to_csv(LOG_FILE, index=False)
    print(f"🕒 Historical log saved to {LOG_FILE}")

    if BACKUP_MODEL:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_name = f"lstm_model_{datetime.now().strftime('%Y-%m-%d')}.keras"
        shutil.copyfile(MODEL_FILE, f"{BACKUP_DIR}/{backup_name}")
        print(f"📦 Model backed up as {backup_name}")

if __name__ == "__main__":
    predict_all()