import os
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from risk_metrics import (
    calculate_risk_metrics,
    determine_risk_level
)

from kmeans_clustering import perform_kmeans

# HMM: Hidden Markov Model for market-regime detection
# Install once with:
# pip install hmmlearn
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

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

MODEL_FOLDER = "models"
OUTPUT_FOLDER = "models"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "all_stock_predictions.csv"
)

HISTORY_PERIOD = "5y"
REALTIME_INTERVAL = "1m"

# HMM settings
HMM_COMPONENTS = 3
HMM_RANDOM_STATE = 42
HMM_N_ITER = 200

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# XGBOOST FEATURES
#
# THESE ARE KEPT UNCHANGED SO THE EXISTING XGBOOST PIPELINE
# CONTINUES TO USE THE SAME FEATURE SET.
# ============================================================

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

# ============================================================
# DOWNLOAD DAILY DATA
# ============================================================

def download_daily_data(ticker):
    print(f"Downloading historical data for {ticker}...")

    data = yf.download(
        ticker,
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No historical data available for {ticker}"
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    if "Date" not in data.columns:
        data = data.rename(
            columns={data.columns[0]: "Date"}
        )

    data["Date"] = pd.to_datetime(data["Date"])

    try:
        if data["Date"].dt.tz is not None:
            data["Date"] = data["Date"].dt.tz_localize(None)
    except Exception:
        pass

    return data


# ============================================================
# GET REAL-TIME DATA
# ============================================================

def get_realtime_data(ticker):
    print(f"Getting latest market data for {ticker}...")

    data = yf.download(
        ticker,
        period="1d",
        interval=REALTIME_INTERVAL,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No real-time data available for {ticker}"
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna(subset=["Close"])

    if data.empty:
        raise ValueError(
            f"Real-time data contains no valid prices for {ticker}"
        )

    latest = data.iloc[-1]

    return {
        "Price": float(latest["Close"]),
        "Open": float(latest["Open"]),
        "High": float(latest["High"]),
        "Low": float(latest["Low"]),
        "Volume": float(latest["Volume"])
    }


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(period).mean()
    average_loss = loss.rolling(period).mean()

    rs = (
        average_gain /
        average_loss.replace(0, np.nan)
    )

    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# ATR
# ============================================================

def calculate_atr_feature(df, period=14):
    previous_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]

    tr2 = (
        df["High"] -
        previous_close
    ).abs()

    tr3 = (
        df["Low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(period).mean()

    return atr


# ============================================================
# TECHNICAL FEATURES
# ============================================================

def create_stock_features(
    stock_df,
    spy_df,
    qqq_df,
    dia_df,
    vix_df
):
    df = stock_df.copy()

    df = df.sort_values("Date")
    df = df.reset_index(drop=True)

    # Price returns
    df["Return_1D"] = df["Close"].pct_change(1)
    df["Return_5D"] = df["Close"].pct_change(5)
    df["Return_10D"] = df["Close"].pct_change(10)
    df["Return_20D"] = df["Close"].pct_change(20)

    # SMA
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # EMA
    df["EMA_12"] = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    df["EMA_26"] = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    # RSI
    df["RSI"] = calculate_rsi(
        df["Close"],
        14
    )

    # MACD
    df["MACD"] = (
        df["EMA_12"] -
        df["EMA_26"]
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_Difference"] = (
        df["MACD"] -
        df["MACD_Signal"]
    )

    # ATR
    df["ATR"] = calculate_atr_feature(
        df,
        14
    )

    # Bollinger Bands
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()

    df["BB_Middle"] = rolling_mean
    df["BB_High"] = (
        rolling_mean +
        2 * rolling_std
    )
    df["BB_Low"] = (
        rolling_mean -
        2 * rolling_std
    )

    # Volatility
    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
    )

    # Volume
    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    volume_average = (
        df["Volume"].rolling(20).mean()
    )

    df["Volume_Ratio"] = (
        df["Volume"] /
        volume_average
    )

    # Distance from moving averages
    df["Distance_SMA20"] = (
        df["Close"] /
        df["SMA_20"] -
        1
    )

    df["Distance_SMA50"] = (
        df["Close"] /
        df["SMA_50"] -
        1
    )

    df["Distance_SMA200"] = (
        df["Close"] /
        df["SMA_200"] -
        1
    )

    # Market features
    market_data = {
        "SPY": spy_df,
        "QQQ": qqq_df,
        "DIA": dia_df,
        "VIX": vix_df
    }

    for name, market_df in market_data.items():

        market = market_df.copy()

        market = market[
            ["Date", "Close"]
        ].copy()

        market = market.sort_values("Date")

        market["Return"] = (
            market["Close"].pct_change()
        )

        market["Volatility"] = (
            market["Return"]
            .rolling(20)
            .std()
        )

        rename_map = {
            "Return": f"{name}_Return",
            "Volatility": f"{name}_Volatility"
        }

        df = df.merge(
            market[
                [
                    "Date",
                    "Return",
                    "Volatility"
                ]
            ].rename(columns=rename_map),
            on="Date",
            how="left"
        )

    return df


# ============================================================
# HMM FEATURE / REGIME DETECTION
#
# IMPORTANT:
# HMM IS KEPT SEPARATE FROM THE EXISTING XGBOOST FEATURES.
# Therefore the existing XGBoost model does NOT need to be
# retrained just to run this integrated version.
#
# The HMM learns hidden market regimes from:
#   1. daily return
#   2. rolling volatility
#   3. normalized volume ratio
#
# The three hidden states are sorted by their average return:
#   Bearish  -> lowest mean return
#   Neutral  -> middle mean return
#   Bullish  -> highest mean return
#
# State numbers themselves are arbitrary in HMM, so we map them
# to meaningful labels after fitting.
# ============================================================

def prepare_hmm_data(df):
    hmm_df = pd.DataFrame(index=df.index)

    hmm_df["Return"] = df["Close"].pct_change()

    hmm_df["Volatility"] = (
        hmm_df["Return"]
        .rolling(20)
        .std()
    )

    volume_mean = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    hmm_df["Volume_Ratio"] = (
        df["Volume"] /
        volume_mean
    )

    hmm_df = hmm_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = hmm_df.notna().all(axis=1)

    return hmm_df, valid_mask


def fit_hmm_regime(df):
    """
    Fit a 3-state Gaussian HMM on historical daily data.

    Returns:
        model
        hmm_features
        valid_mask
        state_to_regime
    """

    hmm_features, valid_mask = prepare_hmm_data(df)

    X = hmm_features.loc[
        valid_mask,
        [
            "Return",
            "Volatility",
            "Volume_Ratio"
        ]
    ].values

    if len(X) < 100:
        raise ValueError(
            "Not enough valid historical observations "
            "to fit the HMM."
        )

    # Standardize features because return, volatility and
    # volume ratio have different numerical scales.
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std == 0] = 1.0

    X_scaled = (
        (X - mean) /
        std
    )

    model = GaussianHMM(
        n_components=HMM_COMPONENTS,
        covariance_type="diag",
        n_iter=HMM_N_ITER,
        random_state=HMM_RANDOM_STATE
    )

    model.fit(X_scaled)

    hidden_states = model.predict(X_scaled)

    state_statistics = []

    for state in range(HMM_COMPONENTS):

        state_mask = (
            hidden_states == state
        )

        if state_mask.sum() == 0:
            mean_return = 0.0
            mean_volatility = 0.0
        else:
            mean_return = float(
                X[state_mask, 0].mean()
            )

            mean_volatility = float(
                X[state_mask, 1].mean()
            )

        state_statistics.append(
            {
                "state": state,
                "mean_return": mean_return,
                "mean_volatility": mean_volatility
            }
        )

    # Sort states by mean historical return.
    # This makes the labels interpretable.
    sorted_states = sorted(
        state_statistics,
        key=lambda x: x["mean_return"]
    )

    state_to_regime = {}

    regime_names = [
        "BEARISH",
        "NEUTRAL",
        "BULLISH"
    ]

    for regime_name, state_info in zip(
        regime_names,
        sorted_states
    ):
        state_to_regime[
            state_info["state"]
        ] = regime_name

    # Save preprocessing parameters inside the returned object.
    hmm_info = {
        "model": model,
        "mean": mean,
        "std": std,
        "state_to_regime": state_to_regime,
        "state_statistics": state_statistics
    }

    return hmm_info


def predict_hmm_regime(
    df,
    hmm_info
):
    """
    Predict the current hidden market regime.

    The HMM is fitted using the complete 5-year historical
    dataset and the current row is classified using the same
    preprocessing parameters.
    """

    hmm_features, valid_mask = prepare_hmm_data(df)

    model = hmm_info["model"]
    mean = hmm_info["mean"]
    std = hmm_info["std"]
    state_to_regime = hmm_info["state_to_regime"]

    # We need the latest valid observation.
    valid_rows = hmm_features.loc[
        valid_mask,
        [
            "Return",
            "Volatility",
            "Volume_Ratio"
        ]
    ]

    if valid_rows.empty:
        raise ValueError(
            "No valid observation available for HMM prediction."
        )

    latest = valid_rows.iloc[-1].values.astype(float)

    latest_scaled = (
        (latest - mean) /
        std
    ).reshape(1, -1)

    # Hidden state probabilities.
    state_probabilities = model.predict_proba(
        latest_scaled
    )[0]

    current_state = int(
        np.argmax(state_probabilities)
    )

    regime = state_to_regime.get(
        current_state,
        "UNKNOWN"
    )

    confidence = float(
        state_probabilities[current_state]
    )

    # State characteristics for the current state.
    state_stats = {
        item["state"]: item
        for item in hmm_info["state_statistics"]
    }

    current_stats = state_stats.get(
        current_state,
        {}
    )

    return {
        "HMM_State": current_state,
        "HMM_Regime": regime,
        "HMM_Confidence_%": confidence * 100,
        "HMM_State_Mean_Return": float(
            current_stats.get(
                "mean_return",
                np.nan
            )
        ),
        "HMM_State_Mean_Volatility": float(
            current_stats.get(
                "mean_volatility",
                np.nan
            )
        )
    }


def get_hmm_adjusted_context(
    xgb_signal,
    hmm_regime
):
    """
    HMM is an unsupervised regime detector, not a directional
    classifier. Therefore this function DOES NOT change the
    XGBoost probability.

    It only gives a transparent context label showing whether
    the XGBoost directional signal agrees with the detected
    market regime.
    """

    if (
        xgb_signal == "UP" and
        hmm_regime == "BULLISH"
    ):
        return "ALIGNED_BULLISH"

    if (
        xgb_signal == "DOWN" and
        hmm_regime == "BEARISH"
    ):
        return "ALIGNED_BEARISH"

    if (
        xgb_signal == "UP" and
        hmm_regime == "BEARISH"
    ):
        return "CONFLICT"

    if (
        xgb_signal == "DOWN" and
        hmm_regime == "BULLISH"
    ):
        return "CONFLICT"

    return "NEUTRAL_CONTEXT"


# ============================================================
# UPDATE LATEST ROW WITH REAL-TIME PRICE
# ============================================================

def update_with_realtime(
    features_df,
    realtime
):
    df = features_df.copy()

    latest_index = df.index[-1]

    df.loc[
        latest_index,
        "Open"
    ] = realtime["Open"]

    df.loc[
        latest_index,
        "High"
    ] = realtime["High"]

    df.loc[
        latest_index,
        "Low"
    ] = realtime["Low"]

    df.loc[
        latest_index,
        "Close"
    ] = realtime["Price"]

    df.loc[
        latest_index,
        "Volume"
    ] = realtime["Volume"]

    close = df["Close"]

    df["Return_1D"] = close.pct_change()
    df["Return_5D"] = close.pct_change(5)
    df["Return_10D"] = close.pct_change(10)
    df["Return_20D"] = close.pct_change(20)

    df["SMA_5"] = close.rolling(5).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    df["EMA_12"] = close.ewm(
        span=12,
        adjust=False
    ).mean()

    df["EMA_26"] = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["RSI"] = calculate_rsi(
        close,
        14
    )

    df["MACD"] = (
        df["EMA_12"] -
        df["EMA_26"]
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["MACD_Difference"] = (
        df["MACD"] -
        df["MACD_Signal"]
    )

    df["ATR"] = calculate_atr_feature(
        df,
        14
    )

    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()

    df["BB_Middle"] = rolling_mean

    df["BB_High"] = (
        rolling_mean +
        2 * rolling_std
    )

    df["BB_Low"] = (
        rolling_mean -
        2 * rolling_std
    )

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
    )

    df["Volume_Change"] = (
        df["Volume"].pct_change()
    )

    df["Volume_Ratio"] = (
        df["Volume"] /
        df["Volume"].rolling(20).mean()
    )

    df["Distance_SMA20"] = (
        close /
        df["SMA_20"] -
        1
    )

    df["Distance_SMA50"] = (
        close /
        df["SMA_50"] -
        1
    )

    df["Distance_SMA200"] = (
        close /
        df["SMA_200"] -
        1
    )

    return df


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    print()
    print("Downloading market indicators...")

    spy = download_daily_data("SPY")
    qqq = download_daily_data("QQQ")
    dia = download_daily_data("DIA")
    vix = download_daily_data("^VIX")

    return (
        spy,
        qqq,
        dia,
        vix
    )


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

def load_model(ticker):

    model_path = os.path.join(
        MODEL_FOLDER,
        f"{ticker}_xgboost.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(model_path)

    return model


# ============================================================
# PREDICT ONE STOCK
# ============================================================

def predict_stock(
    ticker,
    spy,
    qqq,
    dia,
    vix
):

    print()
    print("=" * 70)
    print(f"PROCESSING {ticker}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load existing XGBoost model
    # --------------------------------------------------------

    model = load_model(ticker)

    # --------------------------------------------------------
    # Historical stock data
    # --------------------------------------------------------

    stock_data = download_daily_data(ticker)

    # --------------------------------------------------------
    # Real-time data
    # --------------------------------------------------------

    realtime = get_realtime_data(ticker)

    # --------------------------------------------------------
    # Create existing XGBoost features
    # --------------------------------------------------------

    features_df = create_stock_features(
        stock_data,
        spy,
        qqq,
        dia,
        vix
    )

    # --------------------------------------------------------
    # Update latest row with real-time data
    # --------------------------------------------------------

    features_df = update_with_realtime(
        features_df,
        realtime
    )

    # --------------------------------------------------------
    # Make sure all ORIGINAL XGBoost features exist
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing features: " +
            ", ".join(missing_features)
        )

    # --------------------------------------------------------
    # Get latest row
    # --------------------------------------------------------

    latest = features_df.iloc[-1]

    # --------------------------------------------------------
    # XGBoost model input
    #
    # IMPORTANT:
    # HMM features are NOT inserted into this dataframe.
    # This preserves compatibility with the existing XGBoost
    # models trained on FEATURES.
    # --------------------------------------------------------

    model_input = (
        features_df[
            FEATURES
        ]
        .iloc[-1:]
        .copy()
    )

    if model_input.isna().any().any():

        nan_columns = (
            model_input.columns[
                model_input.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Latest model row contains missing values: " +
            ", ".join(nan_columns)
        )

    # --------------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------------

    prediction = model.predict(
        model_input
    )[0]

    probabilities = (
        model.predict_proba(
            model_input
        )[0]
    )

    classes = list(model.classes_)

    if 1 in classes:
        up_probability = probabilities[
            classes.index(1)
        ]
    else:
        up_probability = float(
            probabilities[-1]
        )

    down_probability = 1 - up_probability

    if up_probability >= 0.60:
        signal = "UP"
    elif up_probability <= 0.40:
        signal = "DOWN"
    else:
        signal = "UNCERTAIN"

    # ========================================================
    # HMM MARKET REGIME
    # ========================================================

    print()
    print(
        f"Fitting {HMM_COMPONENTS}-state HMM "
        f"for {ticker}..."
    )

    hmm_info = fit_hmm_regime(
        features_df
    )

    hmm_result = predict_hmm_regime(
        features_df,
        hmm_info
    )

    hmm_context = get_hmm_adjusted_context(
        signal,
        hmm_result["HMM_Regime"]
    )

    # ========================================================
    # RISK METRICS
    # ========================================================

    risk_df = stock_data.copy()

    risk_df["Date"] = pd.to_datetime(
        risk_df["Date"]
    )

    market_df = spy[
        [
            "Date",
            "Close"
        ]
    ].copy()

    market_df["Date"] = pd.to_datetime(
        market_df["Date"]
    )

    risk = calculate_risk_metrics(
        risk_df,
        market_df
    )

    risk_level = determine_risk_level(
        risk["Volatility"],
        risk["Maximum_Drawdown"],
        risk["VaR_95"]
    )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "Ticker": ticker,

        "Date": str(
            pd.Timestamp.now().date()
        ),

        "Price": float(
            realtime["Price"]
        ),

        # Existing XGBoost outputs
        "UP_%": float(
            up_probability * 100
        ),

        "DOWN_%": float(
            down_probability * 100
        ),

        "Signal": signal,

        # HMM outputs
        "HMM_State": int(
            hmm_result["HMM_State"]
        ),

        "HMM_Regime": (
            hmm_result["HMM_Regime"]
        ),

        "HMM_Confidence_%": float(
            hmm_result["HMM_Confidence_%"]
        ),

        "HMM_State_Mean_Return": float(
            hmm_result[
                "HMM_State_Mean_Return"
            ]
        ),

        "HMM_State_Mean_Volatility": float(
            hmm_result[
                "HMM_State_Mean_Volatility"
            ]
        ),

        "HMM_Context": hmm_context,

        # Existing indicators
        "RSI": float(
            latest["RSI"]
        ),

        "MACD": float(
            latest["MACD"]
        ),

        # K-Means helper columns
        "KMeans_Return_20D": float(
            latest["Return_20D"]
        ),

        "KMeans_Volatility_20D": float(
            latest["Volatility_20D"]
        ),

        "KMeans_RSI": float(
            latest["RSI"]
        ),

        "KMeans_MACD": float(
            latest["MACD"]
        ),

        "KMeans_Volume_Ratio": float(
            latest["Volume_Ratio"]
        ),

        # Risk metrics
        "Volatility": float(
            risk["Volatility"]
        ),

        "Maximum_Drawdown": float(
            risk["Maximum_Drawdown"]
        ),

        "Sharpe_Ratio": float(
            risk["Sharpe_Ratio"]
        ),

        "VaR_95": float(
            risk["VaR_95"]
        ),

        "Beta": float(
            risk["Beta"]
        ),

        "ATR": float(
            risk["ATR"]
        ),

        "Risk_Level": risk_level
    }


# ============================================================
# FORMAT DISPLAY
# ============================================================

def format_display(df):

    display_df = df.copy()

    display_df["Price"] = display_df[
        "Price"
    ].apply(
        lambda x:
        f"${x:.2f}"
        if pd.notna(x)
        else "N/A"
    )

    display_df["UP_%"] = display_df[
        "UP_%"
    ].apply(
        lambda x:
        f"{x:.2f}%"
        if pd.notna(x)
        else "N/A"
    )

    display_df["DOWN_%"] = display_df[
        "DOWN_%"
    ].apply(
        lambda x:
        f"{x:.2f}%"
        if pd.notna(x)
        else "N/A"
    )

    display_df["HMM_Confidence_%"] = (
        display_df[
            "HMM_Confidence_%"
        ].apply(
            lambda x:
            f"{x:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    display_df["HMM_State_Mean_Return"] = (
        display_df[
            "HMM_State_Mean_Return"
        ].apply(
            lambda x:
            f"{x * 100:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    display_df["HMM_State_Mean_Volatility"] = (
        display_df[
            "HMM_State_Mean_Volatility"
        ].apply(
            lambda x:
            f"{x * 100:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    display_df["RSI"] = display_df[
        "RSI"
    ].apply(
        lambda x:
        f"{x:.2f}"
        if pd.notna(x)
        else "N/A"
    )

    display_df["MACD"] = display_df[
        "MACD"
    ].apply(
        lambda x:
        f"{x:.4f}"
        if pd.notna(x)
        else "N/A"
    )

    display_df["Volatility"] = display_df[
        "Volatility"
    ].apply(
        lambda x:
        f"{x * 100:.2f}%"
        if pd.notna(x)
        else "N/A"
    )

    display_df["Maximum_Drawdown"] = (
        display_df[
            "Maximum_Drawdown"
        ].apply(
            lambda x:
            f"{x * 100:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    display_df["Sharpe_Ratio"] = display_df[
        "Sharpe_Ratio"
    ].apply(
        lambda x:
        f"{x:.2f}"
        if pd.notna(x)
        else "N/A"
    )

    display_df["VaR_95"] = display_df[
        "VaR_95"
    ].apply(
        lambda x:
        f"{x * 100:.2f}%"
        if pd.notna(x)
        else "N/A"
    )

    display_df["Beta"] = display_df[
        "Beta"
    ].apply(
        lambda x:
        f"{x:.2f}"
        if pd.notna(x)
        else "N/A"
    )

    display_df["ATR"] = display_df[
        "ATR"
    ].apply(
        lambda x:
        f"{x:.2f}"
        if pd.notna(x)
        else "N/A"
    )

    return display_df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)
    print(
        "                 ALL STOCK PREDICTIONS + HMM REGIME DETECTION"
    )
    print("=" * 120)

    print()
    print(
        "Stocks:",
        ", ".join(STOCKS)
    )

    print()
    print(
        "Existing XGBoost features: PRESERVED"
    )

    print(
        "HMM: 3-state hidden market-regime detection"
    )

    # ========================================================
    # DOWNLOAD MARKET DATA
    # ========================================================

    try:

        (
            spy,
            qqq,
            dia,
            vix
        ) = load_market_data()

    except Exception as e:

        print()
        print(
            "ERROR downloading market data:"
        )

        print(e)

        return

    # ========================================================
    # PROCESS ALL STOCKS
    # ========================================================

    results = []

    for ticker in STOCKS:

        try:

            result = predict_stock(
                ticker,
                spy,
                qqq,
                dia,
                vix
            )

            results.append(result)

            print()
            print(
                f"{ticker} completed successfully."
            )

        except Exception as e:

            print()
            print(
                f"ERROR processing {ticker}:"
            )

            print(e)

            results.append({

                "Ticker": ticker,

                "Date": str(
                    pd.Timestamp.now().date()
                ),

                "Price": np.nan,

                "UP_%": np.nan,

                "DOWN_%": np.nan,

                "Signal": "ERROR",

                "HMM_State": np.nan,

                "HMM_Regime": "ERROR",

                "HMM_Confidence_%": np.nan,

                "HMM_State_Mean_Return": np.nan,

                "HMM_State_Mean_Volatility": np.nan,

                "HMM_Context": "ERROR",

                "RSI": np.nan,

                "MACD": np.nan,

                "KMeans_Return_20D": np.nan,

                "KMeans_Volatility_20D": np.nan,

                "KMeans_RSI": np.nan,

                "KMeans_MACD": np.nan,

                "KMeans_Volume_Ratio": np.nan,

                "Volatility": np.nan,

                "Maximum_Drawdown": np.nan,

                "Sharpe_Ratio": np.nan,

                "VaR_95": np.nan,

                "Beta": np.nan,

                "ATR": np.nan,

                "Risk_Level": "ERROR"
            })

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    if len(results) == 0:

        print()
        print(
            "No predictions were generated."
        )

        return

    results_df = pd.DataFrame(results)

    # ========================================================
    # K-MEANS CLUSTERING
    # ========================================================

    kmeans_columns = [
        "Ticker",
        "KMeans_Return_20D",
        "KMeans_Volatility_20D",
        "KMeans_RSI",
        "KMeans_MACD",
        "KMeans_Volume_Ratio",
        "Sharpe_Ratio",
        "Maximum_Drawdown",
        "ATR"
    ]

    kmeans_input = results_df[
        kmeans_columns
    ].copy()

    kmeans_input = kmeans_input.rename(
        columns={
            "KMeans_Return_20D":
                "Return_20D",

            "KMeans_Volatility_20D":
                "Volatility_20D",

            "KMeans_RSI":
                "RSI",

            "KMeans_MACD":
                "MACD",

            "KMeans_Volume_Ratio":
                "Volume_Ratio"
        }
    )

    try:

        kmeans_result = perform_kmeans(
            kmeans_input
        )

        results_df = results_df.merge(
            kmeans_result[
                [
                    "Ticker",
                    "KMeans_Cluster"
                ]
            ],
            on="Ticker",
            how="left"
        )

        print()
        print(
            "K-Means clustering completed successfully."
        )

    except Exception as e:

        print()
        print(
            "WARNING: K-Means clustering could not be completed:"
        )

        print(e)

        results_df[
            "KMeans_Cluster"
        ] = np.nan

    # Remove internal K-Means helper columns.
    helper_columns = [
        "KMeans_Return_20D",
        "KMeans_Volatility_20D",
        "KMeans_RSI",
        "KMeans_MACD",
        "KMeans_Volume_Ratio"
    ]

    results_df = results_df.drop(
        columns=helper_columns,
        errors="ignore"
    )

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    display_df = format_display(
        results_df
    )

    print()
    print("=" * 220)

    print(
        "                    FINAL PREDICTIONS + HMM + RISK"
    )

    print("=" * 220)

    print()

    display_columns = [
        "Ticker",
        "Date",
        "Price",
        "UP_%",
        "DOWN_%",
        "Signal",
        "HMM_State",
        "HMM_Regime",
        "HMM_Confidence_%",
        "HMM_Context",
        "KMeans_Cluster",
        "RSI",
        "MACD",
        "Volatility",
        "Maximum_Drawdown",
        "Sharpe_Ratio",
        "VaR_95",
        "Beta",
        "ATR",
        "Risk_Level"
    ]

    print(
        display_df[
            display_columns
        ].to_string(index=False)
    )

    # ========================================================
    # SAVE LOCATION
    # ========================================================

    print()
    print("=" * 80)

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 80)

    # ========================================================
    # STRONGEST UP SIGNALS
    # ========================================================

    valid = results_df[
        results_df["Signal"] != "ERROR"
    ].copy()

    if not valid.empty:

        up_stocks = (
            valid
            .sort_values(
                "UP_%",
                ascending=False
            )
        )

        print()
        print("=" * 80)
        print("STRONGEST UP SIGNALS")
        print("=" * 80)

        strong_up = up_stocks[
            up_stocks["UP_%"] >= 60
        ]

        if strong_up.empty:

            print(
                "No stocks have UP probability >= 60%."
            )

        else:

            for _, row in strong_up.iterrows():

                print(
                    f"{row['Ticker']:<8}"
                    f"{row['UP_%']:.2f}% UP"
                    f" | HMM: "
                    f"{row['HMM_Regime']:<8}"
                    f" | Context: "
                    f"{row['HMM_Context']:<17}"
                    f" | Risk: "
                    f"{row['Risk_Level']}"
                )

        # ====================================================
        # DOWN SIGNALS
        # ====================================================

        print()
        print("=" * 80)
        print("STRONG DOWN SIGNALS")
        print("=" * 80)

        down_stocks = (
            valid
            .sort_values(
                "DOWN_%",
                ascending=False
            )
        )

        strong_down = down_stocks[
            down_stocks["DOWN_%"] >= 60
        ]

        if strong_down.empty:

            print(
                "No stocks have DOWN probability >= 60%."
            )

        else:

            for _, row in strong_down.iterrows():

                print(
                    f"{row['Ticker']:<8}"
                    f"{row['DOWN_%']:.2f}% DOWN"
                    f" | HMM: "
                    f"{row['HMM_Regime']:<8}"
                    f" | Context: "
                    f"{row['HMM_Context']:<17}"
                    f" | Risk: "
                    f"{row['Risk_Level']}"
                )

    # ========================================================
    # HMM REGIME SUMMARY
    # ========================================================

    if not valid.empty:

        print()
        print("=" * 80)
        print("HMM MARKET REGIME SUMMARY")
        print("=" * 80)

        for _, row in valid.iterrows():

            confidence = row[
                "HMM_Confidence_%"
            ]

            print(
                f"{row['Ticker']:<8}"
                f"Regime: "
                f"{row['HMM_Regime']:<8}"
                f" | Confidence: "
                f"{confidence:.2f}%"
                f" | XGBoost Signal: "
                f"{row['Signal']:<9}"
                f" | Context: "
                f"{row['HMM_Context']}"
            )

    # ========================================================
    # K-MEANS CLUSTER SUMMARY
    # ========================================================

    if "KMeans_Cluster" in valid.columns:

        clustered = valid.dropna(
            subset=["KMeans_Cluster"]
        ).copy()

        if not clustered.empty:

            print()
            print("=" * 80)
            print("K-MEANS STOCK CLUSTERS")
            print("=" * 80)

            for (
                cluster_id,
                cluster_data
            ) in clustered.groupby(
                "KMeans_Cluster"
            ):

                tickers = ", ".join(
                    cluster_data[
                        "Ticker"
                    ].tolist()
                )

                print(
                    f"Cluster "
                    f"{int(cluster_id)}: "
                    f"{tickers}"
                )

    # ========================================================
    # RISK SUMMARY
    # ========================================================

    if not valid.empty:

        print()
        print("=" * 80)
        print("RISK SUMMARY")
        print("=" * 80)

        for _, row in valid.iterrows():

            print(
                f"{row['Ticker']:<8}"
                f"Risk: "
                f"{row['Risk_Level']:<8}"
                f"Volatility: "
                f"{row['Volatility'] * 100:.2f}%"
                f" | Drawdown: "
                f"{row['Maximum_Drawdown'] * 100:.2f}%"
                f" | Beta: "
                f"{row['Beta']:.2f}"
            )

    print()
    print("=" * 80)

    print(
        "ALL STOCK PREDICTIONS COMPLETED"
    )

    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
