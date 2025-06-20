import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download sentiment model (runs once)
nltk.download("vader_lexicon", quiet=True)
analyzer = SentimentIntensityAnalyzer()

HEADERS = {"User-Agent": "Mozilla/5.0"}
DATA_FILE = "data/sp500_data.csv"
BATCH_SIZE = 20
SLEEP_SECONDS = 2

def get_sp500_tickers():
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]['Symbol'].tolist()
        # Fix problematic tickers like BRK.B → BRK-B
        return [t.replace(".", "-") for t in tickers]
    except Exception as e:
        print(f"❌ Error fetching tickers: {e}")
        return []

def get_sentiment(ticker):
    try:
        url = f"https://news.google.com/search?q={ticker}%20stock"
        res = requests.get(url, headers=HEADERS, timeout=8)
        soup = BeautifulSoup(res.text, "html.parser")
        headlines = soup.find_all("a", class_="DY5T1d", limit=3)
        scores = [analyzer.polarity_scores(h.text)['compound'] for h in headlines if h.text]
        return ticker, sum(scores) / len(scores) if scores else 0
    except Exception:
        return ticker, 0

def collect_data():
    tickers = get_sp500_tickers()
    if not tickers:
        print("❌ No tickers found.")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sector mapping
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        sector_map = dict(zip(table["Symbol"].str.replace(".", "-"), table["GICS Sector"]))
    except Exception:
        sector_map = {}

    # Sentiment scores in parallel
    sentiments = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        future_map = {executor.submit(get_sentiment, t): t for t in tickers}
        for future in as_completed(future_map):
            t, score = future.result()
            sentiments[t] = score

    rows = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        print(f"📦 Fetching price data for batch {i//BATCH_SIZE + 1}")
        try:
            df_batch = yf.download(
                batch, period="1d", interval="1m",
                progress=False, group_by="ticker", threads=True
            )
            if df_batch.empty:
                print(f"⚠️ Batch {i//BATCH_SIZE + 1} returned no data.")
                continue
        except Exception as e:
            print(f"⚠️ Error fetching batch {i//BATCH_SIZE + 1}: {e}")
            time.sleep(5)
            continue

        for t in batch:
            try:
                # Check if multi-indexed (multi-ticker) or flat (single-ticker)
                if isinstance(df_batch.columns, pd.MultiIndex):
                    if t in df_batch.columns.get_level_values(0):
                        series = df_batch[t]["Close"].dropna()
                    else:
                        continue
                else:
                    series = df_batch["Close"].dropna()

                if series.empty:
                    continue

                latest_price = series.iloc[-1]
                sentiment = sentiments.get(t, 0)
                sector = sector_map.get(t, "Unknown")
                rows.append({
                    "Ticker": t,
                    "Price": round(latest_price, 2),
                    "Sentiment": round(sentiment, 3),
                    "Sector": sector,
                    "Timestamp": now
                })
            except Exception as e:
                print(f"⚠️ Skipping {t}: {e}")
                continue

        time.sleep(SLEEP_SECONDS)

    if not rows:
        print("⚠️ No data collected.")
        return

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    write_header = not os.path.exists(DATA_FILE)
    df.to_csv(DATA_FILE, mode="a", header=write_header, index=False)
    print(f"✅ Collected {len(df)} records at {now}")

if __name__ == "__main__":
    collect_data()