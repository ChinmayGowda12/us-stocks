import pandas as pd
import joblib
import matplotlib.pyplot as plt

TICKER = "AAPL"

model = joblib.load(
    f"models/{TICKER}_xgboost.pkl"
)

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

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance)

importance.head(15).plot(
    x="Feature",
    y="Importance",
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Top XGBoost Features"
)

plt.tight_layout()

plt.show()