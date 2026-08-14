import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "models/all_stock_predictions.csv"

OUTPUT_FILE = "models/stock_rankings.csv"


# ============================================================
# NORMALIZE VALUE
# ============================================================

def normalize_series(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = series.min()
    maximum = series.max()

    if pd.isna(minimum) or pd.isna(maximum):

        return pd.Series(
            0.5,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            0.5,
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# CREATE RANKING SCORE
# ============================================================

def calculate_ranking_score(df):

    data = df.copy()

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "UP_%",
        "DOWN_%",
        "RSI",
        "MACD",
        "Volatility",
        "Maximum_Drawdown",
        "Sharpe_Ratio",
        "VaR_95",
        "Beta",
        "ATR"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Prediction strength
    # --------------------------------------------------------

    data["Prediction_Strength"] = (
        data["UP_%"] / 100
    )

    # --------------------------------------------------------
    # Confidence
    #
    # 50% = 0
    # 100% = 1
    # --------------------------------------------------------

    data["Confidence"] = (
        (
            data["UP_%"] - 50
        ).abs()
        / 50
    )

    # --------------------------------------------------------
    # Risk components
    # --------------------------------------------------------

    if "Volatility" in data.columns:

        volatility_risk = (
            normalize_series(
                data["Volatility"]
            )
        )

    else:

        volatility_risk = pd.Series(
            0.5,
            index=data.index
        )

    if "Maximum_Drawdown" in data.columns:

        drawdown_risk = (
            normalize_series(
                data["Maximum_Drawdown"]
                .abs()
            )
        )

    else:

        drawdown_risk = pd.Series(
            0.5,
            index=data.index
        )

    if "VaR_95" in data.columns:

        var_risk = (
            normalize_series(
                data["VaR_95"].abs()
            )
        )

    else:

        var_risk = pd.Series(
            0.5,
            index=data.index
        )

    if "Beta" in data.columns:

        beta_risk = (
            normalize_series(
                data["Beta"].abs()
            )
        )

    else:

        beta_risk = pd.Series(
            0.5,
            index=data.index
        )

    # --------------------------------------------------------
    # Combined risk score
    # --------------------------------------------------------

    data["Risk_Score"] = (

        volatility_risk * 0.35

        +

        drawdown_risk * 0.30

        +

        var_risk * 0.20

        +

        beta_risk * 0.15

    )

    # --------------------------------------------------------
    # Technical score
    # --------------------------------------------------------

    technical_score = pd.Series(
        0.5,
        index=data.index
    )

    # RSI
    if "RSI" in data.columns:

        rsi_score = (
            1 -
            (
                (
                    data["RSI"] - 50
                ).abs()
                / 50
            )
        )

        rsi_score = rsi_score.clip(
            0,
            1
        )

        technical_score = (
            technical_score * 0.5
            +
            rsi_score * 0.5
        )

    # MACD
    if "MACD" in data.columns:

        macd_positive = (
            data["MACD"] > 0
        ).astype(float)

        technical_score = (
            technical_score * 0.7
            +
            macd_positive * 0.3
        )

    data["Technical_Score"] = (
        technical_score
    )

    # --------------------------------------------------------
    # Risk-adjusted score
    #
    # Higher prediction strength
    # Higher confidence
    # Better technicals
    # Lower risk
    # --------------------------------------------------------

    data["Ranking_Score"] = (

        data["Prediction_Strength"] * 0.40

        +

        data["Confidence"] * 0.20

        +

        data["Technical_Score"] * 0.15

        +

        (
            1 -
            data["Risk_Score"]
        ) * 0.25

    )

    # --------------------------------------------------------
    # Convert to percentage
    # --------------------------------------------------------

    data["Ranking_Score"] = (
        data["Ranking_Score"] * 100
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    data = data.sort_values(
        "Ranking_Score",
        ascending=False
    )

    data["Rank"] = range(
        1,
        len(data) + 1
    )

    # --------------------------------------------------------
    # Ranking category
    # --------------------------------------------------------

    def ranking_category(score):

        if score >= 75:

            return "EXCELLENT"

        elif score >= 60:

            return "STRONG"

        elif score >= 45:

            return "MODERATE"

        elif score >= 30:

            return "WEAK"

        else:

            return "VERY WEAK"

    data["Ranking"] = (
        data["Ranking_Score"]
        .apply(
            ranking_category
        )
    )

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)

    print(
        "                 STOCK RANKING ENGINE"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_FILE
    ):

        print(
            f"ERROR: Input file not found:"
        )

        print(
            INPUT_FILE
        )

        print()
        print(
            "Run 14_predict_all_stocks.py first."
        )

        return

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Remove errors
    # --------------------------------------------------------

    df = df[
        df["Signal"] != "ERROR"
    ].copy()

    if df.empty:

        print(
            "No valid stock predictions found."
        )

        return

    # --------------------------------------------------------
    # Calculate ranking
    # --------------------------------------------------------

    ranked = calculate_ranking_score(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    ranked.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()

    print(
        f"{'Rank':<6}"
        f"{'Ticker':<10}"
        f"{'UP_%':<10}"
        f"{'Signal':<12}"
        f"{'Risk':<10}"
        f"{'Score':<10}"
        f"{'Rating':<12}"
    )

    print("-" * 80)

    for _, row in ranked.iterrows():

        print(
            f"{int(row['Rank']):<6}"
            f"{row['Ticker']:<10}"
            f"{row['UP_%']:.2f}%"
            f"{'':<4}"
            f"{row['Signal']:<12}"
            f"{row.get('Risk_Level', 'N/A'):<10}"
            f"{row['Ranking_Score']:.2f}"
            f"{'':<3}"
            f"{row['Ranking']:<12}"
        )

    # --------------------------------------------------------
    # Best stocks
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "TOP RANKED STOCKS"
    )

    print("=" * 80)

    top = ranked.head(5)

    for _, row in top.iterrows():

        print(
            f"{int(row['Rank'])}. "
            f"{row['Ticker']} - "
            f"{row['Ranking_Score']:.2f} "
            f"({row['Ranking']})"
        )

    # --------------------------------------------------------
    # Strong UP candidates
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "STRONG UP CANDIDATES"
    )

    print("=" * 80)

    up = ranked[
        ranked["UP_%"] >= 60
    ].copy()

    if up.empty:

        print(
            "No stocks above 60% UP probability."
        )

    else:

        for _, row in up.iterrows():

            print(
                f"{row['Ticker']:<8}"
                f"{row['UP_%']:.2f}% UP"
            )

    # --------------------------------------------------------
    # Strong DOWN candidates
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "STRONG DOWN CANDIDATES"
    )

    print("=" * 80)

    down = ranked[
        ranked["DOWN_%"] >= 60
    ].copy()

    if down.empty:

        print(
            "No stocks above 60% DOWN probability."
        )

    else:

        for _, row in down.iterrows():

            print(
                f"{row['Ticker']:<8}"
                f"{row['DOWN_%']:.2f}% DOWN"
            )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "Ranking saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()