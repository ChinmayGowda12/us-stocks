import pandas as pd
import os
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

TICKER = "AAPL"

df = pd.read_csv(
    f"data/processed/final/{TICKER}.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values("Date")

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return_1D", "Return_5D",
    "Return_10D", "Return_20D",
    "SMA_5", "SMA_20",
    "SMA_50", "SMA_200",
    "EMA_12", "EMA_26",
    "RSI",
    "MACD", "MACD_Signal",
    "MACD_Difference",
    "ATR",
    "BB_High", "BB_Low",
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

split = int(
    len(df) * 0.8
)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"XGBoost Accuracy: {accuracy:.4f}"
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
    f"models/{TICKER}_xgboost.pkl"
)

print("XGBoost saved.")