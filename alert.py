import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os
import csv
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATA_FILE = "data/sp500_data.csv"
MODEL_FILE = "models/lstm_model.h5"
ALERT_LOG = "data/alerts_log.csv"

# Email config (loaded from environment)
SENDER = os.environ.get("ALERT_EMAIL")
RECEIVER = os.environ.get("ALERT_RECEIVER") or SENDER  # Optional separate receiver
PASSWORD = os.environ.get("ALERT_PASSWORD")

if not SENDER or not PASSWORD:
    raise ValueError("Missing ALERT_EMAIL or ALERT_PASSWORD in environment.")

def send_alert(ticker, confidence, direction):
    msg = MIMEText(f"{ticker} is predicted to go {direction.upper()} with {confidence:.2f}% confidence.")
    msg["Subject"] = f"📢 Stock Alert: {ticker} {direction.upper()}"
    msg["From"] = SENDER
    msg["To"] = RECEIVER
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER, PASSWORD)
            server.sendmail(SENDER, RECEIVER, msg.as_string())
        print(f"✅ Alert sent for {ticker}")
        # Log the alert only if the email was sent successfully.
        log_alert(ticker, confidence, direction)
    except Exception as e:
        print(f"❌ Failed to send alert for {ticker}: {e}")

def log_alert(ticker, confidence, direction):
    os.makedirs("data", exist_ok=True)
    log_exists = os.path.exists(ALERT_LOG)
    with open(ALERT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["Timestamp", "Ticker", "Confidence", "Direction"])
        writer.writerow([datetime.now().isoformat(), ticker, round(confidence, 2), direction])

def predict_tickers():
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Run model.py first.")
        return
    model = load_model(MODEL_FILE)
    df = pd.read_csv(DATA_FILE).sort_values("Timestamp")

    # Fit a scaler on all the data once
    scaler = MinMaxScaler()
    scaler.fit(df[['Price', 'Sentiment']])

    for ticker in df['Ticker'].unique():
        sub = df[df['Ticker'] == ticker].sort_values("Timestamp").tail(10)
        if len(sub) < 10:
            continue
        try:
            X_input = sub[['Price', 'Sentiment']].values
            X_scaled = scaler.transform(X_input)
            X = np.expand_dims(X_scaled, axis=0)
            pred = model.predict(X, verbose=0)[0][0]
            confidence = pred * 100
            # Alert if the prediction is very high or very low
            if pred > 0.9 or pred < 0.1:
                direction = "up" if pred > 0.5 else "down"
                send_alert(ticker, confidence, direction)
        except Exception as e:
            print(f"{ticker} failed: {e}")

if __name__ == "__main__":
    print(f"🚨 Running stock alert check at {datetime.now()}")
    predict_tickers()