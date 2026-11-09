import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

STOCKS = [
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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_FOLDER = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# K-MEANS FEATURES
# ============================================================

KMEANS_FEATURES = [
    "Return_20D",
    "Volatility_20D",
    "RSI",
    "MACD",
    "Volume_Ratio",
    "Sharpe_Ratio",
    "Maximum_Drawdown",
    "ATR"
]


def perform_kmeans(stock_data):

    df = stock_data.copy()

    missing = [
        column
        for column in KMEANS_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing K-Means features: "
            + ", ".join(missing)
        )

    df = df.dropna(
        subset=KMEANS_FEATURES
    ).copy()

    if len(df) < 3:
        raise ValueError(
            "At least 3 valid stocks are required for K-Means."
        )

    X = df[
        KMEANS_FEATURES
    ].astype(float).values

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    n_clusters = min(
        3,
        len(df)
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df["KMeans_Cluster"] = (
        kmeans.fit_predict(
            X_scaled
        )
    )

    joblib.dump(
        kmeans,
        os.path.join(
            MODEL_FOLDER,
            "kmeans_model.pkl"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            MODEL_FOLDER,
            "kmeans_scaler.pkl"
        )
    )

    return df
