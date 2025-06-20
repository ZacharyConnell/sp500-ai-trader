import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

HEADERS = {"User-Agent": "Mozilla/5.0"}
DATA_FILE = "data/sp500_data.csv"

def get_sp500_tickers():
    """
    Scrapes the S&P 500 ticker symbols from Wikipedia.
    """
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"❌ Error fetching S&P 500 tickers: {e}")
        return []

def get_sentiment(ticker):
    """
    Scrapes news headlines related to a ticker and calculates the average sentiment.
    """
    try:
        url = f"https://news.google.com/search?q={ticker}%20stock"
        res = requests.get(url, headers=HEADERS, timeout=8)
        soup = BeautifulSoup(res.text, "html.parser")
        headlines = soup.find_all("a", class_="DY5T1d", limit=3)
        scores = [analyzer.polarity_scores(headline.text)['compound'] for headline in headlines if headline.text]
        avg_score = sum(scores) / len(scores) if scores else 0
        return ticker, avg_score
    except Exception as e:
        print(f"⚠️ Error getting sentiment for {ticker}: {e}")
        return ticker, 0

def collect_data():
    tickers = get_sp500_tickers()
    if not tickers:
        print("❌ No tickers found.")
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get sector mapping from Wikipedia's S&P 500 companies table
    try:
        sector_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        sector_map = dict(zip(sector_table["Symbol"], sector_table["GICS Sector"]))
    except Exception as e:
        print(f"⚠️ Could not fetch sector mapping: {e}")
        sector_map = {}

    # Parallel sentiment scraping using ThreadPoolExecutor
    sentiments = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_ticker = {executor.submit(get_sentiment, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                ticker, score = future.result()
                sentiments[ticker] = score
            except Exception as e:
                print(f"⚠️ Error processing sentiment for {t}: {e}")
                sentiments[t] = 0

    # Collect stock prices, sentiments, and sector data for each ticker
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            price = info.get("regularMarketPrice")
            if price is None:
                continue
            sentiment = sentiments.get(ticker, 0)
            sector = sector_map.get(ticker, "Unknown")
            rows.append({
                "Ticker": ticker,
                "Price": price,
                "Sentiment": sentiment,
                "Sector": sector,
                "Timestamp": now
            })
        except Exception as e:
            print(f"⚠️ Error collecting data for {ticker}: {e}")
            continue

    if not rows:
        print("⚠️ No data collected.")
        return

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    # Append to file if it exists; otherwise, write header.
    write_header = not os.path.exists(DATA_FILE)
    df.to_csv(DATA_FILE, mode='a', header=write_header, index=False)
    print(f"✅ Collected data for {len(df)} S&P 500 stocks at {now}")

if __name__ == "__main__":
    collect_data()