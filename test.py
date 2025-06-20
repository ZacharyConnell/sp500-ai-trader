import pandas as pd

df = pd.read_csv("data/sp500_data.csv")
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
print(df.shape)
print(df[['Ticker', 'Price', 'Sector']].head())
