import os
import subprocess
import sys

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="US Stock ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_FOLDER = os.path.join(
    BASE_DIR,
    "models"
)

DATA_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final"
)

PREDICTION_FILE = os.path.join(
    MODELS_FOLDER,
    "all_stock_predictions.csv"
)

RANKING_FILE = os.path.join(
    MODELS_FOLDER,
    "stock_ranking.csv"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    'US Stock Market ML Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'XGBoost predictions, technical indicators, '
    'risk metrics and stock ranking'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def find_column(df, possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def format_percentage(value):

    try:

        if pd.isna(value):
            return "N/A"

        value = float(value)

        # Handle values already stored as percentages
        if abs(value) > 1:
            return f"{value:.2f}%"

        return f"{value:.2%}"

    except Exception:

        return "N/A"


def format_number(value, decimals=2):

    try:

        if pd.isna(value):
            return "N/A"

        return f"{float(value):.{decimals}f}"

    except Exception:

        return "N/A"


def get_prediction_file():

    if not os.path.exists(
        PREDICTION_FILE
    ):

        return None

    try:

        df = pd.read_csv(
            PREDICTION_FILE
        )

        df = clean_column_names(df)

        return df

    except Exception as error:

        st.error(
            "Unable to read prediction file:"
        )

        st.code(
            str(error)
        )

        return None


def get_historical_data(ticker):

    file_path = os.path.join(
        DATA_FOLDER,
        f"{ticker}.csv"
    )

    if not os.path.exists(
        file_path
    ):

        return None

    try:

        df = pd.read_csv(
            file_path
        )

        df = clean_column_names(df)

        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df = df.sort_values(
                "Date"
            )

        return df

    except Exception:

        return None


# ============================================================
# LOAD PREDICTIONS
# ============================================================

predictions = get_prediction_file()


if predictions is None:

    st.error(
        "Prediction file was not found."
    )

    st.info(
        "Run all_stock_predictions.py first."
    )

    st.write(
        "Expected file:"
    )

    st.code(
        "models/all_stock_predictions.csv"
    )

    st.stop()


# ============================================================
# CHECK TICKER COLUMN
# ============================================================

ticker_column = find_column(
    predictions,
    [
        "Ticker",
        "ticker",
        "Symbol",
        "symbol"
    ]
)


if ticker_column is None:

    st.error(
        "Ticker column was not found "
        "in all_stock_predictions.csv."
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(predictions.columns)
    )

    st.stop()


# ============================================================
# NORMALIZE TICKERS
# ============================================================

predictions[ticker_column] = (
    predictions[ticker_column]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# IDENTIFY IMPORTANT COLUMNS
# ============================================================

signal_column = find_column(
    predictions,
    [
        "Signal",
        "signal",
        "Prediction",
        "prediction"
    ]
)

up_column = find_column(
    predictions,
    [
        "UP_%",
        "UP%",
        "Probability_UP",
        "Probability UP",
        "UP_Probability"
    ]
)

down_column = find_column(
    predictions,
    [
        "DOWN_%",
        "DOWN%",
        "Probability_DOWN",
        "Probability DOWN",
        "DOWN_Probability"
    ]
)

price_column = find_column(
    predictions,
    [
        "Price",
        "Close",
        "Latest Price"
    ]
)

rsi_column = find_column(
    predictions,
    [
        "RSI",
        "rsi"
    ]
)

macd_column = find_column(
    predictions,
    [
        "MACD",
        "macd"
    ]
)

risk_column = find_column(
    predictions,
    [
        "Risk_Level",
        "Risk Level",
        "Risk"
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Dashboard Controls"
)


# ------------------------------------------------------------
# Refresh
# ------------------------------------------------------------

if st.sidebar.button(
    "🔄 Refresh Dashboard",
    use_container_width=True
):

    st.cache_data.clear()

    st.rerun()


# ------------------------------------------------------------
# Stock selection
# ------------------------------------------------------------

tickers = sorted(
    predictions[ticker_column]
    .dropna()
    .unique()
    .tolist()
)


if len(tickers) == 0:

    st.error(
        "No stocks were found."
    )

    st.stop()


selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    tickers
)


# ============================================================
# SUMMARY CALCULATIONS
# ============================================================

total_stocks = len(tickers)


if signal_column is not None:

    signal_values = (
        predictions[signal_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    up_count = (
        signal_values
        .eq("UP")
        .sum()
    )

    down_count = (
        signal_values
        .eq("DOWN")
        .sum()
    )

    uncertain_count = (
        signal_values
        .eq("UNCERTAIN")
        .sum()
    )

else:

    up_count = 0
    down_count = 0
    uncertain_count = 0


# ============================================================
# MARKET SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Market Summary'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Stocks Analyzed",
        total_stocks
    )


with col2:

    st.metric(
        "UP Signals",
        up_count
    )


with col3:

    st.metric(
        "DOWN Signals",
        down_count
    )


with col4:

    st.metric(
        "UNCERTAIN",
        uncertain_count
    )


# ============================================================
# ALL STOCK PREDICTIONS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'All Stock Predictions'
    '</div>',
    unsafe_allow_html=True
)


display_columns = []

preferred_columns = [

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

    "Risk_Level",

    "Rank"
]


for column in preferred_columns:

    if column in predictions.columns:

        display_columns.append(
            column
        )


# Add any columns not explicitly listed
for column in predictions.columns:

    if column not in display_columns:

        display_columns.append(
            column
        )


display_df = predictions[
    display_columns
].copy()


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# STOCK RANKING
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Stock Ranking'
    '</div>',
    unsafe_allow_html=True
)


if os.path.exists(
    RANKING_FILE
):

    try:

        ranking = pd.read_csv(
            RANKING_FILE
        )

        ranking = clean_column_names(
            ranking
        )

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:

        st.warning(
            "Could not read stock ranking file."
        )

        st.code(
            str(error)
        )

else:

    # If ranking file doesn't exist,
    # create a basic ranking from UP probability.

    if up_column is not None:

        ranking = predictions.copy()

        try:

            ranking["_UP_NUMERIC"] = pd.to_numeric(
                ranking[up_column],
                errors="coerce"
            )

            ranking = ranking.sort_values(
                "_UP_NUMERIC",
                ascending=False
            )

            ranking.insert(
                0,
                "Rank",
                range(
                    1,
                    len(ranking) + 1
                )
            )

            ranking_columns = []

            for column in [
                "Rank",
                ticker_column,
                price_column,
                up_column,
                down_column,
                signal_column,
                risk_column
            ]:

                if (
                    column is not None
                    and column in ranking.columns
                    and column not in ranking_columns
                ):

                    ranking_columns.append(
                        column
                    )

            ranking = ranking[
                ranking_columns
            ]

            st.dataframe(
                ranking,
                use_container_width=True,
                hide_index=True
            )

        except Exception:

            st.info(
                "Ranking file not available."
            )

    else:

        st.info(
            "Ranking file not available."
        )


# ============================================================
# SELECTED STOCK
# ============================================================

st.markdown(
    '<div class="section-title">'
    f'{selected_ticker} Analysis'
    '</div>',
    unsafe_allow_html=True
)


selected_rows = predictions[
    predictions[ticker_column]
    == selected_ticker
]


if selected_rows.empty:

    st.warning(
        "No prediction data found for "
        f"{selected_ticker}."
    )

    st.stop()


selected = selected_rows.iloc[-1]


# ============================================================
# SELECTED STOCK MAIN METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


# ------------------------------------------------------------
# Price
# ------------------------------------------------------------

with col1:

    if price_column is not None:

        price = selected[
            price_column
        ]

        try:

            st.metric(
                "Current Price",
                f"${float(price):,.2f}"
            )

        except Exception:

            st.metric(
                "Current Price",
                str(price)
            )

    else:

        st.metric(
            "Current Price",
            "N/A"
        )


# ------------------------------------------------------------
# UP probability
# ------------------------------------------------------------

with col2:

    if up_column is not None:

        st.metric(
            "UP Probability",
            format_percentage(
                selected[
                    up_column
                ]
            )
        )

    else:

        st.metric(
            "UP Probability",
            "N/A"
        )


# ------------------------------------------------------------
# DOWN probability
# ------------------------------------------------------------

with col3:

    if down_column is not None:

        st.metric(
            "DOWN Probability",
            format_percentage(
                selected[
                    down_column
                ]
            )
        )

    else:

        st.metric(
            "DOWN Probability",
            "N/A"
        )


# ------------------------------------------------------------
# Signal
# ------------------------------------------------------------

with col4:

    if signal_column is not None:

        st.metric(
            "Prediction",
            str(
                selected[
                    signal_column
                ]
            )
        )

    else:

        st.metric(
            "Prediction",
            "N/A"
        )


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Technical Indicators'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    if rsi_column is not None:

        st.metric(
            "RSI",
            format_number(
                selected[
                    rsi_column
                ],
                2
            )
        )

    else:

        st.metric(
            "RSI",
            "N/A"
        )


with col2:

    if macd_column is not None:

        st.metric(
            "MACD",
            format_number(
                selected[
                    macd_column
                ],
                4
            )
        )

    else:

        st.metric(
            "MACD",
            "N/A"
        )


with col3:

    if risk_column is not None:

        st.metric(
            "Risk Level",
            str(
                selected[
                    risk_column
                ]
            )
        )

    else:

        st.metric(
            "Risk Level",
            "N/A"
        )


# ============================================================
# RISK METRICS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Risk Metrics'
    '</div>',
    unsafe_allow_html=True
)


risk_columns = [

    "Volatility",
    "Maximum_Drawdown",
    "Sharpe_Ratio",
    "VaR_95",
    "Beta",
    "ATR"
]


risk_available = [

    column

    for column in risk_columns

    if column in predictions.columns

]


if len(risk_available) == 0:

    st.info(
        "Risk metrics are not present "
        "in the prediction CSV yet."
    )

else:

    risk_col_count = min(
        len(risk_available),
        3
    )

    for start in range(
        0,
        len(risk_available),
        3
    ):

        row_columns = st.columns(
            min(
                3,
                len(
                    risk_available[
                        start:start + 3
                    ]
                )
            )
        )

        for index, column in enumerate(
            risk_available[
                start:start + 3
            ]
        ):

            with row_columns[index]:

                value = selected[
                    column
                ]

                if column in [
                    "Volatility",
                    "Maximum_Drawdown",
                    "VaR_95"
                ]:

                    display_value = (
                        format_percentage(
                            value
                        )
                    )

                elif column == "Sharpe_Ratio":

                    display_value = (
                        format_number(
                            value,
                            2
                        )
                    )

                elif column == "Beta":

                    display_value = (
                        format_number(
                            value,
                            2
                        )
                    )

                elif column == "ATR":

                    display_value = (
                        format_number(
                            value,
                            2
                        )
                    )

                else:

                    display_value = str(
                        value
                    )

                st.metric(
                    column,
                    display_value
                )


# ============================================================
# HISTORICAL DATA
# ============================================================

historical = get_historical_data(
    selected_ticker
)


if historical is None:

    st.warning(
        f"Historical data for "
        f"{selected_ticker} was not found."
    )

else:

    # --------------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------------

    if "Close" in historical.columns:

        st.markdown(
            '<div class="section-title">'
            f'{selected_ticker} Price History'
            '</div>',
            unsafe_allow_html=True
        )

        price_chart = historical[
            [
                "Date",
                "Close"
            ]
        ].dropna()

        if not price_chart.empty:

            price_chart = (
                price_chart
                .set_index("Date")
            )

            st.line_chart(
                price_chart,
                use_container_width=True
            )


    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    moving_average_columns = [

        column

        for column in [
            "Close",
            "SMA_20",
            "SMA_50",
            "SMA_200"
        ]

        if column in historical.columns
    ]


    if len(
        moving_average_columns
    ) > 1:

        st.markdown(
            '<div class="section-title">'
            'Price and Moving Averages'
            '</div>',
            unsafe_allow_html=True
        )

        ma_chart = historical[
            [
                "Date"
            ]
            +
            moving_average_columns
        ].dropna(
            subset=["Date"]
        )

        ma_chart = (
            ma_chart
            .set_index("Date")
        )

        st.line_chart(
            ma_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # RSI CHART
    # --------------------------------------------------------

    if "RSI" in historical.columns:

        st.markdown(
            '<div class="section-title">'
            'RSI'
            '</div>',
            unsafe_allow_html=True
        )

        rsi_chart = historical[
            [
                "Date",
                "RSI"
            ]
        ].dropna()

        if not rsi_chart.empty:

            rsi_chart = (
                rsi_chart
                .set_index("Date")
            )

            st.line_chart(
                rsi_chart,
                use_container_width=True
            )


    # --------------------------------------------------------
    # MACD CHART
    # --------------------------------------------------------

    macd_columns = [

        column

        for column in [
            "MACD",
            "MACD_Signal"
        ]

        if column in historical.columns
    ]


    if len(macd_columns) > 0:

        st.markdown(
            '<div class="section-title">'
            'MACD'
            '</div>',
            unsafe_allow_html=True
        )

        macd_chart = historical[
            [
                "Date"
            ]
            +
            macd_columns
        ].dropna(
            subset=["Date"]
        )

        macd_chart = (
            macd_chart
            .set_index("Date")
        )

        st.line_chart(
            macd_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # VOLUME CHART
    # --------------------------------------------------------

    if "Volume" in historical.columns:

        st.markdown(
            '<div class="section-title">'
            'Trading Volume'
            '</div>',
            unsafe_allow_html=True
        )

        volume_chart = historical[
            [
                "Date",
                "Volume"
            ]
        ].dropna()

        if not volume_chart.empty:

            volume_chart = (
                volume_chart
                .set_index("Date")
            )

            st.bar_chart(
                volume_chart,
                use_container_width=True
            )


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "View Prediction Data"
):

    st.dataframe(
        selected_rows,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "US Stock Market ML Dashboard | "
    "XGBoost + Technical Indicators + "
    "Risk Metrics + Stock Ranking"
)