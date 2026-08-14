import pandas as pd
import os
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


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

DATA_FOLDER = "data/processed/final"

MODEL_FOLDER = "models"

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# FEATURES
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
# TRAIN ONE STOCK
# ============================================================

def train_stock(ticker):

    print("\n")
    print("=" * 60)
    print(f"TRAINING MODEL FOR {ticker}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    file_path = os.path.join(
        DATA_FOLDER,
        f"{ticker}.csv"
    )

    if not os.path.exists(file_path):

        print(
            f"ERROR: {file_path} not found."
        )

        return None

    df = pd.read_csv(
        file_path
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    )

    # --------------------------------------------------------
    # Remove missing data
    # --------------------------------------------------------

    df = df.dropna(
        subset=FEATURES + ["Target"]
    )

    print(
        f"Total rows: {len(df)}"
    )

    # --------------------------------------------------------
    # X and Y
    # --------------------------------------------------------

    X = df[FEATURES]

    y = df["Target"]

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split = int(
        len(df) * 0.80
    )

    X_train = X.iloc[:split]

    X_test = X.iloc[split:]

    y_train = y.iloc[:split]

    y_test = y.iloc[split:]

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=500,

        max_depth=5,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=5,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42

    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        f"\nTraining {ticker}..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            y_test,
            probabilities
        )

    except ValueError:

        auc = float("nan")

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n")
    print("-" * 60)
    print(f"RESULTS FOR {ticker}")
    print("-" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_FOLDER,
        f"{ticker}_xgboost.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved to: {model_path}"
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    results = pd.DataFrame({

        "Date": df["Date"].iloc[split:].values,

        "Actual": y_test.values,

        "Prediction": predictions,

        "Probability_UP": probabilities

    })

    prediction_path = os.path.join(
        MODEL_FOLDER,
        f"{ticker}_predictions.csv"
    )

    results.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"Predictions saved to: "
        f"{prediction_path}"
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "Ticker": ticker,

        "Rows": len(df),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": auc

    }


# ============================================================
# TRAIN ALL STOCKS
# ============================================================

all_results = []


for ticker in STOCKS:

    result = train_stock(
        ticker
    )

    if result is not None:

        all_results.append(
            result
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

if len(all_results) > 0:

    summary = pd.DataFrame(
        all_results
    )

    summary = summary.sort_values(
        "Accuracy",
        ascending=False
    )

    summary_path = (
        "models/all_stocks_results.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\n")
    print("=" * 60)
    print("ALL STOCKS TRAINING COMPLETED")
    print("=" * 60)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nSummary saved to: "
        f"{summary_path}"
    )