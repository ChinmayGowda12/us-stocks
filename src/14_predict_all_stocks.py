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

# Historical period used for feature calculation
HISTORY_PERIOD = "5y"

# Real-time interval
REALTIME_INTERVAL = "1m"


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# FEATURES
#
# THESE MUST MATCH THE FEATURES USED DURING TRAINING
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

    print(
        f"Downloading historical data for {ticker}..."
    )

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

    # --------------------------------------------------------
    # Handle MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    data = data.reset_index()

    # --------------------------------------------------------
    # Make sure Date exists
    # --------------------------------------------------------

    if "Date" not in data.columns:

        data = data.rename(
            columns={
                data.columns[0]: "Date"
            }
        )

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    # --------------------------------------------------------
    # Remove timezone
    # --------------------------------------------------------

    try:

        if data["Date"].dt.tz is not None:

            data["Date"] = (
                data["Date"]
                .dt.tz_localize(None)
            )

    except Exception:

        pass

    return data


# ============================================================
# GET REAL-TIME DATA
# ============================================================

def get_realtime_data(ticker):

    print(
        f"Getting latest market data for {ticker}..."
    )

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

    # --------------------------------------------------------
    # Handle MultiIndex
    # --------------------------------------------------------

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    data = data.dropna(
        subset=["Close"]
    )

    if data.empty:

        raise ValueError(
            f"Real-time data contains no valid prices for {ticker}"
        )

    latest = data.iloc[-1]

    return {

        "Price": float(
            latest["Close"]
        ),

        "Open": float(
            latest["Open"]
        ),

        "High": float(
            latest["High"]
        ),

        "Low": float(
            latest["Low"]
        ),

        "Volume": float(
            latest["Volume"]
        )
    }


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = (
        gain
        .rolling(period)
        .mean()
    )

    average_loss = (
        loss
        .rolling(period)
        .mean()
    )

    rs = (
        average_gain
        /
        average_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    return rsi


# ============================================================
# ATR
# ============================================================

def calculate_atr_feature(
    df,
    period=14
):

    previous_close = (
        df["Close"]
        .shift(1)
    )

    tr1 = (
        df["High"]
        -
        df["Low"]
    )

    tr2 = (
        df["High"]
        -
        previous_close
    ).abs()

    tr3 = (
        df["Low"]
        -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = (
        true_range
        .rolling(period)
        .mean()
    )

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

    # --------------------------------------------------------
    # Basic cleanup
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    )

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Price returns
    # --------------------------------------------------------

    df["Return_1D"] = (
        df["Close"]
        .pct_change(1)
    )

    df["Return_5D"] = (
        df["Close"]
        .pct_change(5)
    )

    df["Return_10D"] = (
        df["Close"]
        .pct_change(10)
    )

    df["Return_20D"] = (
        df["Close"]
        .pct_change(20)
    )

    # --------------------------------------------------------
    # SMA
    # --------------------------------------------------------

    df["SMA_5"] = (
        df["Close"]
        .rolling(5)
        .mean()
    )

    df["SMA_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["SMA_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["SMA_200"] = (
        df["Close"]
        .rolling(200)
        .mean()
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA_12"] = (
        df["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    df["EMA_26"] = (
        df["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = calculate_rsi(
        df["Close"],
        14
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    df["MACD"] = (
        df["EMA_12"]
        -
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
        df["MACD"]
        -
        df["MACD_Signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["ATR"] = calculate_atr_feature(
        df,
        14
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    rolling_mean = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    rolling_std = (
        df["Close"]
        .rolling(20)
        .std()
    )

    df["BB_Middle"] = (
        rolling_mean
    )

    df["BB_High"] = (
        rolling_mean
        +
        2 * rolling_std
    )

    df["BB_Low"] = (
        rolling_mean
        -
        2 * rolling_std
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
    )

    volume_average = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        /
        volume_average
    )

    # --------------------------------------------------------
    # Distance from moving averages
    # --------------------------------------------------------

    df["Distance_SMA20"] = (
        df["Close"]
        /
        df["SMA_20"]
        - 1
    )

    df["Distance_SMA50"] = (
        df["Close"]
        /
        df["SMA_50"]
        - 1
    )

    df["Distance_SMA200"] = (
        df["Close"]
        /
        df["SMA_200"]
        - 1
    )

    # --------------------------------------------------------
    # MARKET FEATURES
    # --------------------------------------------------------

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

        market = market.sort_values(
            "Date"
        )

        market["Return"] = (
            market["Close"]
            .pct_change()
        )

        market["Volatility"] = (
            market["Return"]
            .rolling(20)
            .std()
        )

        if name == "SPY":

            df = df.merge(
                market[
                    [
                        "Date",
                        "Return",
                        "Volatility"
                    ]
                ].rename(
                    columns={
                        "Return":
                            "SPY_Return",

                        "Volatility":
                            "SPY_Volatility"
                    }
                ),
                on="Date",
                how="left"
            )

        elif name == "QQQ":

            df = df.merge(
                market[
                    [
                        "Date",
                        "Return",
                        "Volatility"
                    ]
                ].rename(
                    columns={
                        "Return":
                            "QQQ_Return",

                        "Volatility":
                            "QQQ_Volatility"
                    }
                ),
                on="Date",
                how="left"
            )

        elif name == "DIA":

            df = df.merge(
                market[
                    [
                        "Date",
                        "Return",
                        "Volatility"
                    ]
                ].rename(
                    columns={
                        "Return":
                            "DIA_Return",

                        "Volatility":
                            "DIA_Volatility"
                    }
                ),
                on="Date",
                how="left"
            )

        elif name == "VIX":

            df = df.merge(
                market[
                    [
                        "Date",
                        "Return",
                        "Volatility"
                    ]
                ].rename(
                    columns={
                        "Return":
                            "VIX_Return",

                        "Volatility":
                            "VIX_Volatility"
                    }
                ),
                on="Date",
                how="left"
            )

    return df


# ============================================================
# UPDATE LATEST ROW WITH REAL-TIME PRICE
# ============================================================

def update_with_realtime(
    features_df,
    realtime
):

    df = features_df.copy()

    latest_index = df.index[-1]

    # --------------------------------------------------------
    # Replace latest market values
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Recalculate price-dependent latest features
    # --------------------------------------------------------

    close = df["Close"]

    df["Return_1D"] = (
        close.pct_change()
    )

    df["Return_5D"] = (
        close.pct_change(5)
    )

    df["Return_10D"] = (
        close.pct_change(10)
    )

    df["Return_20D"] = (
        close.pct_change(20)
    )

    df["SMA_5"] = (
        close.rolling(5).mean()
    )

    df["SMA_20"] = (
        close.rolling(20).mean()
    )

    df["SMA_50"] = (
        close.rolling(50).mean()
    )

    df["SMA_200"] = (
        close.rolling(200).mean()
    )

    df["EMA_12"] = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    df["EMA_26"] = (
        close
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["RSI"] = calculate_rsi(
        close,
        14
    )

    df["MACD"] = (
        df["EMA_12"]
        -
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
        df["MACD"]
        -
        df["MACD_Signal"]
    )

    df["ATR"] = calculate_atr_feature(
        df,
        14
    )

    rolling_mean = (
        close
        .rolling(20)
        .mean()
    )

    rolling_std = (
        close
        .rolling(20)
        .std()
    )

    df["BB_Middle"] = (
        rolling_mean
    )

    df["BB_High"] = (
        rolling_mean
        +
        2 * rolling_std
    )

    df["BB_Low"] = (
        rolling_mean
        -
        2 * rolling_std
    )

    df["Volatility_20D"] = (
        df["Return_1D"]
        .rolling(20)
        .std()
    )

    df["Volume_Change"] = (
        df["Volume"]
        .pct_change()
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        /
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["Distance_SMA20"] = (
        close
        /
        df["SMA_20"]
        - 1
    )

    df["Distance_SMA50"] = (
        close
        /
        df["SMA_50"]
        - 1
    )

    df["Distance_SMA200"] = (
        close
        /
        df["SMA_200"]
        - 1
    )

    return df


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    print()
    print(
        "Downloading market indicators..."
    )

    spy = download_daily_data(
        "SPY"
    )

    qqq = download_daily_data(
        "QQQ"
    )

    dia = download_daily_data(
        "DIA"
    )

    vix = download_daily_data(
        "^VIX"
    )

    return (
        spy,
        qqq,
        dia,
        vix
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(ticker):

    model_path = os.path.join(
        MODEL_FOLDER,
        f"{ticker}_xgboost.pkl"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(
        model_path
    )

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

    print(
        f"PROCESSING {ticker}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        ticker
    )

    # --------------------------------------------------------
    # Historical stock data
    # --------------------------------------------------------

    stock_data = download_daily_data(
        ticker
    )

    # --------------------------------------------------------
    # Real-time data
    # --------------------------------------------------------

    realtime = get_realtime_data(
        ticker
    )

    # --------------------------------------------------------
    # Create features
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
    # Make sure all features exist
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in features_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features: "
            +
            ", ".join(
                missing_features
            )
        )

    # --------------------------------------------------------
    # Get latest row
    # --------------------------------------------------------

    latest = (
        features_df
        .iloc[-1]
    )

    # --------------------------------------------------------
    # Check NaN
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
                model_input.isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Latest model row contains "
            "missing values: "
            +
            ", ".join(
                nan_columns
            )
        )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = model.predict(
        model_input
    )[0]

    probabilities = (
        model.predict_proba(
            model_input
        )[0]
    )

    # --------------------------------------------------------
    # Determine probability UP
    #
    # XGBoost binary classifier:
    # probability for class 1 = UP
    # --------------------------------------------------------

    classes = list(
        model.classes_
    )

    if 1 in classes:

        up_probability = (
            probabilities[
                classes.index(1)
            ]
        )

    else:

        up_probability = float(
            probabilities[-1]
        )

    down_probability = (
        1
        -
        up_probability
    )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    if up_probability >= 0.60:

        signal = "UP"

    elif up_probability <= 0.40:

        signal = "DOWN"

    else:

        signal = "UNCERTAIN"

    # ========================================================
    # RISK METRICS
    # ========================================================

    # Use historical daily data for risk calculations.
    # This avoids treating 1-minute data as daily returns.

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

    # --------------------------------------------------------
    # Calculate risk metrics
    # --------------------------------------------------------

    risk = calculate_risk_metrics(
        risk_df,
        market_df
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

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
            pd.Timestamp.now()
            .date()
        ),

        "Price": float(
            realtime["Price"]
        ),

        "UP_%": float(
            up_probability * 100
        ),

        "DOWN_%": float(
            down_probability * 100
        ),

        "Signal": signal,

        "RSI": float(
            latest["RSI"]
        ),

        "MACD": float(
            latest["MACD"]
        ),

        # ----------------------------------------------------
        # RISK METRICS
        # ----------------------------------------------------

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

def format_display(
    df
):

    display_df = df.copy()

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    display_df["Price"] = (
        display_df["Price"]
        .apply(
            lambda x:
            f"${x:.2f}"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    display_df["UP_%"] = (
        display_df["UP_%"]
        .apply(
            lambda x:
            f"{x:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    display_df["DOWN_%"] = (
        display_df["DOWN_%"]
        .apply(
            lambda x:
            f"{x:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    display_df["RSI"] = (
        display_df["RSI"]
        .apply(
            lambda x:
            f"{x:.2f}"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    display_df["MACD"] = (
        display_df["MACD"]
        .apply(
            lambda x:
            f"{x:.4f}"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    display_df["Volatility"] = (
        display_df["Volatility"]
        .apply(
            lambda x:
            f"{x * 100:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # Maximum Drawdown
    # --------------------------------------------------------

    display_df["Maximum_Drawdown"] = (
        display_df[
            "Maximum_Drawdown"
        ]
        .apply(
            lambda x:
            f"{x * 100:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # Sharpe
    # --------------------------------------------------------

    display_df["Sharpe_Ratio"] = (
        display_df["Sharpe_Ratio"]
        .apply(
            lambda x:
            f"{x:.2f}"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # VaR
    # --------------------------------------------------------

    display_df["VaR_95"] = (
        display_df["VaR_95"]
        .apply(
            lambda x:
            f"{x * 100:.2f}%"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # Beta
    # --------------------------------------------------------

    display_df["Beta"] = (
        display_df["Beta"]
        .apply(
            lambda x:
            f"{x:.2f}"
            if pd.notna(x)
            else "N/A"
        )
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    display_df["ATR"] = (
        display_df["ATR"]
        .apply(
            lambda x:
            f"{x:.2f}"
            if pd.notna(x)
            else "N/A"
        )
    )

    return display_df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)

    print(
        "                         ALL STOCK PREDICTIONS"
    )

    print("=" * 100)

    print()
    print(
        "Stocks:",
        ", ".join(STOCKS)
    )

    print()

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

            results.append(
                result
            )

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

            # ------------------------------------------------
            # Add error row
            # ------------------------------------------------

            results.append({

                "Ticker": ticker,

                "Date": str(
                    pd.Timestamp.now()
                    .date()
                ),

                "Price": np.nan,

                "UP_%": np.nan,

                "DOWN_%": np.nan,

                "Signal": "ERROR",

                "RSI": np.nan,

                "MACD": np.nan,

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

    results_df = pd.DataFrame(
        results
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
    print("=" * 180)

    print(
        "                         FINAL PREDICTIONS + RISK"
    )

    print("=" * 180)

    print()

    print(
        display_df[
            [
                "Ticker",
                "Date",
                "Price",
                "UP_%",
                "DOWN_%",
                "Signal",
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
        ].to_string(
            index=False
        )
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

        print(
            "STRONGEST UP SIGNALS"
        )

        print("=" * 80)

        strong_up = up_stocks[
            up_stocks["UP_%"] >= 60
        ]

        if strong_up.empty:

            print(
                "No stocks have UP probability >= 60%."
            )

        else:

            for _, row in (
                strong_up.iterrows()
            ):

                print(
                    f"{row['Ticker']:<8}"
                    f"{row['UP_%']:.2f}% UP"
                    f" | Risk: "
                    f"{row['Risk_Level']}"
                )

        # ====================================================
        # DOWN SIGNALS
        # ====================================================

        print()
        print("=" * 80)

        print(
            "STRONG DOWN SIGNALS"
        )

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

            for _, row in (
                strong_down.iterrows()
            ):

                print(
                    f"{row['Ticker']:<8}"
                    f"{row['DOWN_%']:.2f}% DOWN"
                    f" | Risk: "
                    f"{row['Risk_Level']}"
                )

    # ========================================================
    # RISK SUMMARY
    # ========================================================

    if not valid.empty:

        print()
        print("=" * 80)

        print(
            "RISK SUMMARY"
        )

        print("=" * 80)

        for _, row in (
            valid.iterrows()
        ):

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