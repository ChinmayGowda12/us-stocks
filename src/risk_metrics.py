import numpy as np
import pandas as pd


# ============================================================
# RISK METRICS MODULE
# ============================================================
#
# This file calculates:
#
# 1. Annualized Volatility
# 2. Maximum Drawdown
# 3. Sharpe Ratio
# 4. Historical VaR (95%)
# 5. Beta vs SPY
# 6. ATR
# 7. Overall Risk Level
#
# ============================================================


# ============================================================
# CALCULATE ATR
# ============================================================

def calculate_atr(df, window=14):

    data = df.copy()

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = close.shift(1)

    true_range_1 = high - low

    true_range_2 = (
        high - previous_close
    ).abs()

    true_range_3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            true_range_1,
            true_range_2,
            true_range_3
        ],
        axis=1
    ).max(axis=1)

    atr = (
        true_range
        .rolling(window)
        .mean()
    )

    return atr.iloc[-1]


# ============================================================
# CALCULATE BETA
# ============================================================

def calculate_beta(
    stock_df,
    market_df
):

    stock = stock_df.copy()

    market = market_df.copy()

    # --------------------------------------------------------
    # Sort by date
    # --------------------------------------------------------

    stock = stock.sort_values(
        "Date"
    )

    market = market.sort_values(
        "Date"
    )

    # --------------------------------------------------------
    # Calculate returns
    # --------------------------------------------------------

    stock_returns = (
        stock
        .set_index("Date")["Close"]
        .pct_change()
        .rename("Stock_Return")
    )

    market_returns = (
        market
        .set_index("Date")["Close"]
        .pct_change()
        .rename("Market_Return")
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [
            stock_returns,
            market_returns
        ],
        axis=1
    ).dropna()

    # --------------------------------------------------------
    # Check sufficient data
    # --------------------------------------------------------

    if len(combined) < 30:

        return np.nan

    # --------------------------------------------------------
    # Market variance
    # --------------------------------------------------------

    market_variance = (
        combined["Market_Return"]
        .var()
    )

    if market_variance == 0:

        return np.nan

    # --------------------------------------------------------
    # Beta
    # --------------------------------------------------------

    beta = (

        combined["Stock_Return"]
        .cov(
            combined["Market_Return"]
        )

        /

        market_variance

    )

    return beta


# ============================================================
# CALCULATE ALL RISK METRICS
# ============================================================

def calculate_risk_metrics(
    stock_df,
    market_df=None
):

    df = stock_df.copy()

    # --------------------------------------------------------
    # Sort data
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    )

    # --------------------------------------------------------
    # Calculate daily returns
    # --------------------------------------------------------

    returns = (
        df["Close"]
        .pct_change()
        .dropna()
    )

    # --------------------------------------------------------
    # Check sufficient data
    # --------------------------------------------------------

    if len(returns) < 30:

        raise ValueError(
            "Not enough historical data "
            "to calculate risk metrics."
        )

    # ========================================================
    # 1. ANNUALIZED VOLATILITY
    # ========================================================

    daily_volatility = (
        returns.std()
    )

    annualized_volatility = (

        daily_volatility

        *

        np.sqrt(252)

    )

    # ========================================================
    # 2. MAXIMUM DRAWDOWN
    # ========================================================

    cumulative_max = (
        df["Close"]
        .cummax()
    )

    drawdown = (

        df["Close"]

        /

        cumulative_max

        - 1

    )

    maximum_drawdown = (
        drawdown.min()
    )

    # ========================================================
    # 3. SHARPE RATIO
    # ========================================================
    #
    # Assumption:
    # risk-free rate = 0 for this simple model.
    #
    # Sharpe = annualized return / annualized volatility
    #
    # ========================================================

    annualized_return = (
        returns.mean() * 252
    )

    if annualized_volatility != 0:

        sharpe_ratio = (

            annualized_return

            /

            annualized_volatility

        )

    else:

        sharpe_ratio = np.nan

    # ========================================================
    # 4. VALUE AT RISK - 95%
    # ========================================================
    #
    # Historical 5th percentile of daily returns.
    #
    # Example:
    # VaR = -0.025
    #
    # approximately means the 5th percentile daily return
    # is -2.5%.
    #
    # ========================================================

    var_95 = (
        returns.quantile(0.05)
    )

    # ========================================================
    # 5. BETA
    # ========================================================

    if market_df is not None:

        beta = calculate_beta(

            stock_df,

            market_df

        )

    else:

        beta = np.nan

    # ========================================================
    # 6. ATR
    # ========================================================

    atr = calculate_atr(
        df,
        window=14
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "Volatility": (
            annualized_volatility
        ),

        "Maximum_Drawdown": (
            maximum_drawdown
        ),

        "Sharpe_Ratio": (
            sharpe_ratio
        ),

        "VaR_95": (
            var_95
        ),

        "Beta": (
            beta
        ),

        "ATR": (
            atr
        )

    }


# ============================================================
# DETERMINE OVERALL RISK LEVEL
# ============================================================

def determine_risk_level(
    volatility,
    maximum_drawdown,
    var_95
):

    # --------------------------------------------------------
    # Convert to absolute values because risk magnitude matters
    # --------------------------------------------------------

    volatility = abs(
        volatility
    )

    maximum_drawdown = abs(
        maximum_drawdown
    )

    var_95 = abs(
        var_95
    )

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    if (

        volatility >= 0.40

        or

        maximum_drawdown >= 0.30

        or

        var_95 >= 0.04

    ):

        return "HIGH"

    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    elif (

        volatility >= 0.20

        or

        maximum_drawdown >= 0.15

        or

        var_95 >= 0.02

    ):

        return "MEDIUM"

    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    else:

        return "LOW"