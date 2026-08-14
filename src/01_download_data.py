import yfinance as yf
import os

# Stocks
stocks = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "V",
    "WMT"
]

# Market indicators
market_symbols = [
    "SPY",
    "QQQ",
    "DIA",
    "^VIX"
]

os.makedirs("data/raw", exist_ok=True)

for ticker in stocks + market_symbols:

    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print(f"WARNING: No data found for {ticker}")
        continue

    # Flatten columns if necessary
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    file_path = f"data/raw/{ticker.replace('^', '')}.csv"

    data.to_csv(file_path, index=False)

    print(f"Saved: {file_path}")

print("\nDownload completed.")