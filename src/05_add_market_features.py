import pandas as pd
import os

feature_folder = "data/processed/features"

market_files = {
    "SPY": "SPY.csv",
    "QQQ": "QQQ.csv",
    "DIA": "DIA.csv",
    "VIX": "VIX.csv"
}


def load_market_data(file, prefix):

    path = os.path.join(
        feature_folder,
        file
    )

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    result = pd.DataFrame()

    result["Date"] = df["Date"]

    result[f"{prefix}_Return"] = (
        df["Close"].pct_change()
    )

    result[f"{prefix}_SMA20"] = (
        df["Close"].rolling(20).mean()
    )

    result[f"{prefix}_Volatility"] = (
        df["Close"]
        .pct_change()
        .rolling(20)
        .std()
    )

    return result


spy = load_market_data(
    "SPY.csv",
    "SPY"
)

qqq = load_market_data(
    "QQQ.csv",
    "QQQ"
)

dia = load_market_data(
    "DIA.csv",
    "DIA"
)

vix = load_market_data(
    "VIX.csv",
    "VIX"
)

market = spy.merge(
    qqq,
    on="Date",
    how="outer"
)

market = market.merge(
    dia,
    on="Date",
    how="outer"
)

market = market.merge(
    vix,
    on="Date",
    how="outer"
)

market = market.sort_values("Date")

market = market.ffill()

market.to_csv(
    "data/processed/market_features.csv",
    index=False
)

print("Market features saved.")