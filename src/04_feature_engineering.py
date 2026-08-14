import pandas as pd
import numpy as np
import os

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volatility import BollingerBands

input_folder = "data/processed"
output_folder = "data/processed/features"

os.makedirs(output_folder, exist_ok=True)


def create_features(df):

    df = df.copy()

    # -----------------------------
    # Returns
    # -----------------------------

    df["Return_1D"] = df["Close"].pct_change()

    df["Return_5D"] = df["Close"].pct_change(5)

    df["Return_10D"] = df["Close"].pct_change(10)

    df["Return_20D"] = df["Close"].pct_change(20)

    # -----------------------------
    # Moving averages
    # -----------------------------

    df["SMA_5"] = df["Close"].rolling(5).mean()

    df["SMA_20"] = df["Close"].rolling(20).mean()

    df["SMA_50"] = df["Close"].rolling(50).mean()

    df["SMA_200"] = df["Close"].rolling(200).mean()

    df["EMA_12"] = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["EMA_26"] = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    # -----------------------------
    # RSI
    # -----------------------------

    rsi = RSIIndicator(
        close=df["Close"],
        window=14
    )

    df["RSI"] = rsi.rsi()

    # -----------------------------
    # MACD
    # -----------------------------

    macd = MACD(
        close=df["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = macd.macd_signal()

    df["MACD_Difference"] = macd.macd_diff()

    # -----------------------------
    # ATR
    # -----------------------------

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["ATR"] = atr.average_true_range()

    # -----------------------------
    # Bollinger Bands
    # -----------------------------

    bb = BollingerBands(
        close=df["Close"],
        window=20,
        window_dev=2
    )

    df["BB_High"] = bb.bollinger_hband()

    df["BB_Low"] = bb.bollinger_lband()

    df["BB_Middle"] = bb.bollinger_mavg()

    # -----------------------------
    # Volatility
    # -----------------------------

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
    )

    # -----------------------------
    # Volume
    # -----------------------------

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    df["Volume_SMA_20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["Volume_Ratio"] = (
        df["Volume"] /
        df["Volume_SMA_20"]
    )

    # -----------------------------
    # Price position
    # -----------------------------

    df["Distance_SMA20"] = (
        df["Close"] / df["SMA_20"] - 1
    )

    df["Distance_SMA50"] = (
        df["Close"] / df["SMA_50"] - 1
    )

    df["Distance_SMA200"] = (
        df["Close"] / df["SMA_200"] - 1
    )

    return df


files = os.listdir(input_folder)

for file in files:

    if not file.endswith(".csv"):
        continue

    ticker = file.replace(".csv", "")

    print(f"Creating features for {ticker}")

    path = os.path.join(input_folder, file)

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = create_features(df)

    output_path = os.path.join(
        output_folder,
        file
    )

    df.to_csv(output_path, index=False)

    print(f"Saved {output_path}")

print("\nFeature engineering completed.")