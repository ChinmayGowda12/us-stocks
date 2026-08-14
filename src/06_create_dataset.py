import pandas as pd
import os

feature_folder = "data/processed/features"
market_path = "data/processed/market_features.csv"

output_folder = "data/processed/final"

os.makedirs(output_folder, exist_ok=True)

market = pd.read_csv(market_path)

market["Date"] = pd.to_datetime(
    market["Date"]
)

files = os.listdir(feature_folder)

for file in files:

    if file in [
        "SPY.csv",
        "QQQ.csv",
        "DIA.csv",
        "VIX.csv"
    ]:
        continue

    if not file.endswith(".csv"):
        continue

    ticker = file.replace(".csv", "")

    print(f"Creating dataset for {ticker}")

    path = os.path.join(
        feature_folder,
        file
    )

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values("Date")

    # Merge market information
    df = df.merge(
        market,
        on="Date",
        how="left"
    )

    # Future close
    df["Future_Close_1D"] = (
        df["Close"].shift(-1)
    )

    # Future return
    df["Future_Return_1D"] = (
        df["Future_Close_1D"] /
        df["Close"] - 1
    )

    # Target
    df["Target"] = (
        df["Future_Return_1D"] > 0
    ).astype(int)

    # Remove rows containing NaN
    df = df.dropna()

    output_path = os.path.join(
        output_folder,
        f"{ticker}.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"{ticker}: {len(df)} rows saved"
    )

print("\nDataset creation completed.")