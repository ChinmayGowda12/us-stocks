import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"

DEFAULT_TICKER = "AAPL"

# Number of days needed to calculate long indicators such as SMA200
HISTORY_PERIOD = "2y"

# Benchmark/index tickers used by the feature set
BENCHMARKS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "DIA": "DIA",
    "VIX": "^VIX"
}


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data(ticker, period=HISTORY_PERIOD):

    print(f"Downloading market data for {ticker}...")

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No market data found for {ticker}"
        )

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.copy()

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{ticker}: Missing columns: {missing}"
        )

    data = data[required_columns].copy()

    data = data.dropna()

    return data


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(data, period=14):

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - previous_close)
    tr3 = abs(low - previous_close)

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(period).mean()

    return atr


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data, ticker):

    df = data.copy()

    close = df["Close"]

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    df["Return_1D"] = close.pct_change(1)
    df["Return_5D"] = close.pct_change(5)
    df["Return_10D"] = close.pct_change(10)
    df["Return_20D"] = close.pct_change(20)

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA_12"] = close.ewm(
        span=12,
        adjust=False
    ).mean()

    df["EMA_26"] = close.ewm(
        span=26,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = calculate_rsi(close)

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    df["MACD"] = (
        df["EMA_12"] -
        df["EMA_26"]
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["MACD_Difference"] = (
        df["MACD"] -
        df["MACD_Signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["ATR"] = calculate_atr(df)

    # --------------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------------

    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["BB_Middle"] = bb_middle

    df["BB_High"] = (
        bb_middle +
        2 * bb_std
    )

    df["BB_Low"] = (
        bb_middle -
        2 * bb_std
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # VOLUME FEATURES
    # --------------------------------------------------------

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    average_volume = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["Volume_Ratio"] = (
        df["Volume"] /
        average_volume
    )

    # --------------------------------------------------------
    # DISTANCE FROM MOVING AVERAGES
    # --------------------------------------------------------

    df["Distance_SMA20"] = (
        close / df["SMA_20"] - 1
    )

    df["Distance_SMA50"] = (
        close / df["SMA_50"] - 1
    )

    df["Distance_SMA200"] = (
        close / df["SMA_200"] - 1
    )

    # --------------------------------------------------------
    # BENCHMARK FEATURES
    # --------------------------------------------------------

    benchmark_data = {}

    for name, symbol in BENCHMARKS.items():

        try:

            benchmark = yf.download(
                symbol,
                period=HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            if isinstance(
                benchmark.columns,
                pd.MultiIndex
            ):
                benchmark.columns = (
                    benchmark.columns
                    .get_level_values(0)
                )

            benchmark = benchmark["Close"]

            benchmark = benchmark.reindex(
                df.index
            ).ffill()

            benchmark_return = (
                benchmark.pct_change()
            )

            if name == "VIX":

                df["VIX_Return"] = (
                    benchmark_return
                )

                df["VIX_Volatility"] = (
                    benchmark_return
                    .rolling(20)
                    .std()
                )

            else:

                df[f"{name}_Return"] = (
                    benchmark_return
                )

                df[f"{name}_Volatility"] = (
                    benchmark_return
                    .rolling(20)
                    .std()
                )

        except Exception as e:

            print(
                f"Warning: Could not download "
                f"{name}: {e}"
            )

            if name == "VIX":

                df["VIX_Return"] = 0.0
                df["VIX_Volatility"] = 0.0

            else:

                df[f"{name}_Return"] = 0.0
                df[f"{name}_Volatility"] = 0.0

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def find_model(ticker):

    possible_files = [
        f"{ticker}_xgboost.pkl",
        f"{ticker}_model.pkl",
        f"{ticker}.pkl",
        f"{ticker}_xgb.pkl"
    ]

    for filename in possible_files:

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No trained model found for {ticker} "
        f"inside {MODEL_DIR}"
    )


# ============================================================
# MODEL FEATURE HANDLING
# ============================================================

def get_model_features(model):

    if hasattr(model, "feature_names_in_"):

        return list(
            model.feature_names_in_
        )

    if hasattr(model, "get_booster"):

        try:

            return list(
                model.get_booster()
                .feature_names
            )

        except Exception:
            pass

    return None


# ============================================================
# PREDICTION
# ============================================================

def predict_stock(ticker):

    ticker = ticker.upper().strip()

    print()
    print("=" * 60)
    print(f"REAL-TIME PREDICTION: {ticker}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model_path = find_model(ticker)

    print(
        f"Model: {model_path}"
    )

    model = joblib.load(
        model_path
    )

    # --------------------------------------------------------
    # Download recent data
    # --------------------------------------------------------

    data = download_data(
        ticker
    )

    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    df = create_features(
        data,
        ticker
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    if df.empty:
        raise ValueError(
            "Not enough data after "
            "feature calculation."
        )

    latest = df.iloc[-1]

    # --------------------------------------------------------
    # Determine model features
    # --------------------------------------------------------

    model_features = get_model_features(
        model
    )

    if model_features is None:

        # Fallback to the feature set
        model_features = [
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

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "The following model features "
            "are missing:\n"
            + "\n".join(missing_features)
        )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    X_latest = pd.DataFrame(
        [latest[model_features].values],
        columns=model_features
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X_latest
    )[0]

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = (
            model.predict_proba(
                X_latest
            )[0]
        )

        classes = list(
            model.classes_
        )

        probability_map = dict(
            zip(
                classes,
                probabilities
            )
        )

        # Assumption:
        # class 1 = UP
        # class 0 = DOWN

        up_probability = (
            probability_map.get(1, 0)
        )

        down_probability = (
            probability_map.get(0, 0)
        )

    else:

        if prediction == 1:

            up_probability = 1.0
            down_probability = 0.0

        else:

            up_probability = 0.0
            down_probability = 1.0

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    if up_probability >= 0.60:

        signal = "UP"

    elif down_probability >= 0.60:

        signal = "DOWN"

    else:

        signal = "UNCERTAIN"

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(
        f"Stock           : {ticker}"
    )

    print(
        f"Date            : "
        f"{df.index[-1].date()}"
    )

    print(
        f"Latest Price    : "
        f"${float(latest['Close']):.2f}"
    )

    print(
        f"RSI             : "
        f"{float(latest['RSI']):.2f}"
    )

    print(
        f"MACD            : "
        f"{float(latest['MACD']):.4f}"
    )

    print(
        f"Probability UP  : "
        f"{up_probability * 100:.2f}%"
    )

    print(
        f"Probability DOWN: "
        f"{down_probability * 100:.2f}%"
    )

    print(
        f"Prediction      : "
        f"{signal}"
    )

    print("=" * 60)

    return {
        "Ticker": ticker,
        "Date": df.index[-1].date(),
        "Price": float(latest["Close"]),
        "RSI": float(latest["RSI"]),
        "MACD": float(latest["MACD"]),
        "UP_%": up_probability * 100,
        "DOWN_%": down_probability * 100,
        "Signal": signal
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1:

        ticker = sys.argv[1].upper()

    else:

        ticker = input(
            "Enter stock ticker: "
        ).strip().upper()

    try:

        predict_stock(
            ticker
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)
        print(e)
        print("=" * 60)