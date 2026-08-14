import yfinance as yf
import pandas as pd


def get_realtime_data(ticker):

    ticker = ticker.upper()

    print(f"Getting latest market data for {ticker}...")

    data = yf.download(
        ticker,
        period="1d",
        interval="1m",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No current market data available for {ticker}"
        )

    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    return data


def get_latest_price(ticker):

    data = get_realtime_data(ticker)

    latest = data.iloc[-1]

    return {
        "Ticker": ticker.upper(),
        "Price": float(latest["Close"]),
        "Open": float(latest["Open"]),
        "High": float(latest["High"]),
        "Low": float(latest["Low"]),
        "Volume": float(latest["Volume"])
    }


if __name__ == "__main__":

    ticker = input(
        "Enter stock ticker: "
    ).strip().upper()

    try:

        result = get_latest_price(
            ticker
        )

        print()
        print("=" * 50)
        print("REAL-TIME MARKET DATA")
        print("=" * 50)

        for key, value in result.items():

            if key == "Price":
                print(
                    f"{key:10}: ${value:.2f}"
                )

            elif key == "Ticker":
                print(
                    f"{key:10}: {value}"
                )

            else:
                print(
                    f"{key:10}: {value}"
                )

        print("=" * 50)

    except Exception as e:

        print(
            f"ERROR: {e}"
        )