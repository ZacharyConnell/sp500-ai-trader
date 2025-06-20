import yfinance as yf

class Portfolio:
    def __init__(self, cash=100000):
        self.cash = cash
        self.holdings = {}  # {ticker: {'shares': int, 'avg_price': float}}

    def buy(self, ticker, price, shares):
        cost = price * shares
        if self.cash >= cost:
            self.cash -= cost
            if ticker in self.holdings:
                current = self.holdings[ticker]
                total_shares = current['shares'] + shares
                new_avg = (
                    (current['avg_price'] * current['shares']) + (price * shares)
                ) / total_shares
                self.holdings[ticker] = {'shares': total_shares, 'avg_price': new_avg}
            else:
                self.holdings[ticker] = {'shares': shares, 'avg_price': price}
            print(f"✅ Bought {shares} shares of {ticker} at ${price:.2f}")
        else:
            print("❌ Not enough cash to complete purchase.")

    def sell(self, ticker, price, shares):
        if ticker in self.holdings and self.holdings[ticker]['shares'] >= shares:
            revenue = price * shares
            self.cash += revenue
            self.holdings[ticker]['shares'] -= shares
            if self.holdings[ticker]['shares'] == 0:
                del self.holdings[ticker]
            print(f"✅ Sold {shares} shares of {ticker} at ${price:.2f}")
        else:
            print("❌ Not enough shares to sell.")

    def value(self):
        total = self.cash
        for ticker, position in self.holdings.items():
            try:
                price = yf.Ticker(ticker).info.get("regularMarketPrice")
                if price:
                    total += price * position['shares']
            except Exception:
                continue
        return round(total, 2)

    def summary(self):
        print(f"💰 Cash: ${round(self.cash, 2)}")
        for ticker, position in self.holdings.items():
            print(f"{ticker}: {position['shares']} shares @ ${round(position['avg_price'], 2)}")