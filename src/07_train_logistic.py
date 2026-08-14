import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

TICKER = "AAPL"

path = (
    f"data/processed/final/{TICKER}.csv"
)

df = pd.read_csv(path)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values("Date")

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "SMA_200",
    "EMA_12",
    "EMA_26",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Difference",
    "ATR",
    "BB_High",
    "BB_Low",
    "BB_Middle",
    "Volatility_20D",
    "Volume_Change",
    "Volume_Ratio",
    "Distance_SMA20",
    "Distance_SMA50",
    "Distance_SMA200",
    "SPY_Return",
    "SPY_Volatility",
    "QQQ_Return",
    "QQQ_Volatility",
    "DIA_Return",
    "DIA_Volatility",
    "VIX_Return",
    "VIX_Volatility"
]

X = df[FEATURES]

y = df["Target"]

# 80/20 chronological split
split = int(len(df) * 0.8)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]

# Scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# Model
model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_scaled,
    y_train
)

predictions = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    classification_report(
        y_test,
        predictions
    )
)

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    f"models/{TICKER}_logistic.pkl"
)

joblib.dump(
    scaler,
    f"models/{TICKER}_scaler.pkl"
)

print("Model saved.")