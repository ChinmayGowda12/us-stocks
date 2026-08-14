import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

TICKER = "AAPL"

DATA_PATH = f"data/processed/final/{TICKER}.csv"
MODEL_PATH = f"models/{TICKER}_xgboost.pkl"

# Probability threshold
# If model probability of UP >= 60%, take a long position.
PROBABILITY_THRESHOLD = 0.60

# Transaction cost per position change.
# 0.0005 = 0.05%
TRANSACTION_COST = 0.0005

# Starting portfolio value
INITIAL_CAPITAL = 10000.0

# Create output directory
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. CHECK FILES
# ============================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n\n"
        "Make sure you have completed the data preparation steps."
    )


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n\n"
        "Make sure you have trained the XGBoost model first."
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

df = df.reset_index(drop=True)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Start date: {df['Date'].iloc[0].date()}"
)

print(
    f"End date: {df['Date'].iloc[-1].date()}"
)


# ============================================================
# 4. FEATURES
# ============================================================

FEATURES = [

    # Price
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    # Returns
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",

    # Moving averages
    "SMA_5",
    "SMA_20",
    "SMA_50",
    "SMA_200",

    # Exponential moving averages
    "EMA_12",
    "EMA_26",

    # Momentum
    "RSI",

    # MACD
    "MACD",
    "MACD_Signal",
    "MACD_Difference",

    # Volatility
    "ATR",

    # Bollinger Bands
    "BB_High",
    "BB_Low",
    "BB_Middle",

    # Volatility
    "Volatility_20D",

    # Volume
    "Volume_Change",
    "Volume_Ratio",

    # Distance from moving averages
    "Distance_SMA20",
    "Distance_SMA50",
    "Distance_SMA200",

    # Market features
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
# 5. CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "\nThe following features are missing:\n"
        + "\n".join(missing_features)
        + "\n\n"
        "Check that your feature-engineering step was completed."
    )


# ============================================================
# 6. REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=FEATURES + [
        "Future_Return_1D"
    ]
)

df = df.reset_index(drop=True)

print(
    f"Rows after cleaning: {len(df)}"
)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
#
# We keep the data chronological.
#
# First 80% = training period
# Last 20%  = test period
#
# The test period is never used to train the model.

split_index = int(
    len(df) * 0.80
)

train = df.iloc[
    :split_index
].copy()

test = df.iloc[
    split_index:
].copy()

print("\n======================================")
print("TRAIN / TEST SPLIT")
print("======================================")

print(
    f"Training rows: {len(train)}"
)

print(
    f"Testing rows:  {len(test)}"
)

print(
    f"\nTraining period:"
)

print(
    train["Date"].iloc[0].date(),
    "to",
    train["Date"].iloc[-1].date()
)

print(
    f"\nTesting period:"
)

print(
    test["Date"].iloc[0].date(),
    "to",
    test["Date"].iloc[-1].date()
)


# ============================================================
# 8. LOAD TRAINED XGBOOST MODEL
# ============================================================

print("\nLoading XGBoost model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 9. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

test["Probability_UP"] = (
    model.predict_proba(
        test[FEATURES]
    )[:, 1]
)


# ============================================================
# 10. CREATE TRADING SIGNAL
# ============================================================

# Example:
#
# Probability >= 0.60
#       ↓
#       BUY / LONG
#
# Probability < 0.60
#       ↓
#       NO POSITION
#
# We are NOT shorting in this first version.

test["Position"] = (
    test["Probability_UP"]
    >= PROBABILITY_THRESHOLD
).astype(int)


# ============================================================
# 11. DETERMINE WHEN A TRADE OCCURS
# ============================================================

# Position:
#
# 0 = no position
# 1 = long
#
# A change from:
#
# 0 → 1 = BUY
# 1 → 0 = SELL
#
# produces a transaction cost.

test["Position_Change"] = (
    test["Position"]
    .diff()
    .abs()
)

# First row:
# There is no previous position.
#
# Treat the first position as a trade if
# the model enters a position.

if len(test) > 0:

    test.loc[
        test.index[0],
        "Position_Change"
    ] = abs(
        test.loc[
            test.index[0],
            "Position"
        ]
    )


# ============================================================
# 12. TRANSACTION COST
# ============================================================

test["Transaction_Cost"] = (
    test["Position_Change"]
    * TRANSACTION_COST
)


# ============================================================
# 13. STRATEGY RETURN
# ============================================================

# Future_Return_1D:
#
# Tomorrow's close / today's close - 1
#
# If Position = 1:
#     We participate in the return.
#
# If Position = 0:
#     We earn 0%.
#
# Then subtract transaction costs.

test["Gross_Strategy_Return"] = (
    test["Position"]
    * test["Future_Return_1D"]
)

test["Strategy_Return"] = (
    test["Gross_Strategy_Return"]
    - test["Transaction_Cost"]
)


# ============================================================
# 14. BUY AND HOLD RETURN
# ============================================================

test["BuyHold_Return"] = (
    test["Future_Return_1D"]
)


# ============================================================
# 15. EQUITY CURVE
# ============================================================

test["Strategy_Growth"] = (
    1 + test["Strategy_Return"]
).cumprod()

test["BuyHold_Growth"] = (
    1 + test["BuyHold_Return"]
).cumprod()


# Convert growth into actual portfolio value.

test["Strategy_Equity"] = (
    INITIAL_CAPITAL
    * test["Strategy_Growth"]
)

test["BuyHold_Equity"] = (
    INITIAL_CAPITAL
    * test["BuyHold_Growth"]
)


# ============================================================
# 16. TOTAL RETURNS
# ============================================================

strategy_final_value = (
    test["Strategy_Equity"].iloc[-1]
)

buyhold_final_value = (
    test["BuyHold_Equity"].iloc[-1]
)

strategy_return = (
    strategy_final_value
    / INITIAL_CAPITAL
    - 1
)

buyhold_return = (
    buyhold_final_value
    / INITIAL_CAPITAL
    - 1
)


# ============================================================
# 17. MAXIMUM DRAWDOWN
# ============================================================

running_peak = (
    test["Strategy_Equity"]
    .cummax()
)

drawdown = (
    test["Strategy_Equity"]
    / running_peak
    - 1
)

maximum_drawdown = (
    drawdown.min()
)


# ============================================================
# 18. SHARPE RATIO
# ============================================================

daily_returns = (
    test["Strategy_Return"]
)

return_std = (
    daily_returns.std()
)

if return_std != 0:

    sharpe_ratio = (
        daily_returns.mean()
        / return_std
    ) * np.sqrt(252)

else:

    sharpe_ratio = 0.0


# ============================================================
# 19. WIN RATE
# ============================================================

# Only look at days when the strategy
# actually had a position.

trading_days = test[
    test["Position"] == 1
]

if len(trading_days) > 0:

    winning_days = (
        trading_days[
            trading_days["Strategy_Return"] > 0
        ]
    )

    win_rate = (
        len(winning_days)
        / len(trading_days)
    )

else:

    win_rate = 0.0


# ============================================================
# 20. NUMBER OF TRADES
# ============================================================

number_of_entries = (
    (
        test["Position"].diff()
        == 1
    )
    .sum()
)

number_of_exits = (
    (
        test["Position"].diff()
        == -1
    )
    .sum()
)


# ============================================================
# 21. AVERAGE TRADE RETURN
# ============================================================

if len(trading_days) > 0:

    average_trade_return = (
        trading_days["Strategy_Return"]
        .mean()
    )

else:

    average_trade_return = 0.0


# ============================================================
# 22. PROFIT FACTOR
# ============================================================

profits = (
    trading_days[
        trading_days["Strategy_Return"] > 0
    ]["Strategy_Return"]
    .sum()
)

losses = (
    trading_days[
        trading_days["Strategy_Return"] < 0
    ]["Strategy_Return"]
    .sum()
)

if losses != 0:

    profit_factor = (
        profits / abs(losses)
    )

else:

    profit_factor = np.inf


# ============================================================
# 23. MODEL DIRECTIONAL ACCURACY
# ============================================================

test["Prediction"] = (
    test["Probability_UP"] >= 0.50
).astype(int)

test["Correct"] = (
    test["Prediction"]
    == test["Target"]
)

directional_accuracy = (
    test["Correct"].mean()
)


# ============================================================
# 24. PRINT RESULTS
# ============================================================

print("\n")
print("======================================")
print("       BACKTEST RESULTS")
print("======================================")

print(
    f"\nTicker: {TICKER}"
)

print(
    f"Initial Capital: "
    f"${INITIAL_CAPITAL:,.2f}"
)

print(
    f"Probability Threshold: "
    f"{PROBABILITY_THRESHOLD:.2f}"
)

print(
    f"Transaction Cost: "
    f"{TRANSACTION_COST:.4%}"
)

print("\n--------------------------------------")

print(
    f"Strategy Final Value: "
    f"${strategy_final_value:,.2f}"
)

print(
    f"Buy & Hold Final Value: "
    f"${buyhold_final_value:,.2f}"
)

print("\n--------------------------------------")

print(
    f"Strategy Return: "
    f"{strategy_return:.2%}"
)

print(
    f"Buy & Hold Return: "
    f"{buyhold_return:.2%}"
)

print(
    f"Strategy vs Buy & Hold: "
    f"{strategy_return - buyhold_return:.2%}"
)

print("\n--------------------------------------")

print(
    f"Maximum Drawdown: "
    f"{maximum_drawdown:.2%}"
)

print(
    f"Sharpe Ratio: "
    f"{sharpe_ratio:.2f}"
)

print(
    f"Win Rate: "
    f"{win_rate:.2%}"
)

print(
    f"Average Trade Return: "
    f"{average_trade_return:.2%}"
)

print(
    f"Profit Factor: "
    f"{profit_factor:.2f}"
)

print("\n--------------------------------------")

print(
    f"Directional Accuracy: "
    f"{directional_accuracy:.2%}"
)

print(
    f"Number of Entries: "
    f"{number_of_entries}"
)

print(
    f"Number of Exits: "
    f"{number_of_exits}"
)

print(
    f"Total Position Days: "
    f"{len(trading_days)}"
)

print("\n======================================")


# ============================================================
# 25. SAVE BACKTEST DATA
# ============================================================

output_csv = (
    f"outputs/{TICKER}_backtest.csv"
)

test.to_csv(
    output_csv,
    index=False
)

print(
    f"\nBacktest data saved to:"
)

print(
    output_csv
)


# ============================================================
# 26. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Ticker",
        "Initial Capital",
        "Final Strategy Value",
        "Final Buy & Hold Value",
        "Strategy Return",
        "Buy & Hold Return",
        "Maximum Drawdown",
        "Sharpe Ratio",
        "Win Rate",
        "Average Trade Return",
        "Profit Factor",
        "Directional Accuracy",
        "Number of Entries",
        "Number of Exits"
    ],

    "Value": [

        TICKER,
        INITIAL_CAPITAL,
        strategy_final_value,
        buyhold_final_value,
        strategy_return,
        buyhold_return,
        maximum_drawdown,
        sharpe_ratio,
        win_rate,
        average_trade_return,
        profit_factor,
        directional_accuracy,
        number_of_entries,
        number_of_exits
    ]
})

summary_path = (
    f"outputs/{TICKER}_backtest_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"Summary saved to:"
)

print(
    summary_path
)


# ============================================================
# 27. CREATE EQUITY CURVE CHART
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    test["Date"],
    test["Strategy_Equity"],
    label="ML Strategy"
)

plt.plot(
    test["Date"],
    test["BuyHold_Equity"],
    label="Buy & Hold"
)

plt.title(
    f"{TICKER} - ML Strategy vs Buy & Hold"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Portfolio Value ($)"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

chart_path = (
    f"outputs/{TICKER}_backtest.png"
)

plt.savefig(
    chart_path,
    dpi=150
)

plt.show()

print(
    f"\nChart saved to:"
)

print(
    chart_path
)


# ============================================================
# 28. DONE
# ============================================================

print("\n======================================")
print("BACKTEST COMPLETED SUCCESSFULLY")
print("======================================")