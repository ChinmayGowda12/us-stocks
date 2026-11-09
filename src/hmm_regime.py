import os
import warnings
import joblib
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM


warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

N_STATES = 3

MODEL_FOLDER = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "models"
)

HMM_MODEL_FILE = os.path.join(
    MODEL_FOLDER,
    "hmm_market_regime.pkl"
)

HMM_MAPPING_FILE = os.path.join(
    MODEL_FOLDER,
    "hmm_state_mapping.pkl"
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


# ============================================================
# PREPARE HMM DATA
# ============================================================

def prepare_hmm_data(spy_data):

    df = spy_data.copy()

    # --------------------------------------------------------
    # Make sure Date is datetime
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Calculate daily return
    # --------------------------------------------------------

    df["HMM_Return"] = (
        df["Close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna(
        subset=["HMM_Return"]
    ).copy()

    if len(df) < 100:

        raise ValueError(
            "Not enough SPY data for HMM training. "
            "At least 100 observations are required."
        )

    return df


# ============================================================
# TRAIN HMM
# ============================================================

def train_hmm(spy_data):

    print()
    print("=" * 70)
    print("TRAINING HMM MARKET REGIME MODEL")
    print("=" * 70)

    df = prepare_hmm_data(
        spy_data
    )

    # --------------------------------------------------------
    # HMM input
    #
    # Multiply returns by 100 so that the numerical scale is
    # easier for the Gaussian HMM to work with.
    # --------------------------------------------------------

    X = (
        df["HMM_Return"]
        .values
        * 100
    ).reshape(
        -1,
        1
    )

    # --------------------------------------------------------
    # Create HMM
    # --------------------------------------------------------

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=200,
        random_state=42
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(
        X
    )

    # --------------------------------------------------------
    # Predict hidden states
    # --------------------------------------------------------

    states = model.predict(
        X
    )

    df["HMM_State"] = states

    # ========================================================
    # DETERMINE STATE CHARACTERISTICS
    # ========================================================

    state_returns = {}

    for state in range(
        N_STATES
    ):

        state_data = df[
            df["HMM_State"] == state
        ]

        if len(state_data) == 0:

            state_returns[state] = 0.0

        else:

            state_returns[state] = float(
                state_data["HMM_Return"]
                .mean()
            )

    # --------------------------------------------------------
    # Sort states by average return
    # --------------------------------------------------------

    sorted_states = sorted(
        state_returns,
        key=state_returns.get
    )

    # --------------------------------------------------------
    # Assign human-readable regimes
    # --------------------------------------------------------

    regime_mapping = {
        sorted_states[0]: "Bearish",
        sorted_states[1]: "Sideways",
        sorted_states[2]: "Bullish"
    }

    df["Market_Regime"] = (
        df["HMM_State"]
        .map(regime_mapping)
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        HMM_MODEL_FILE
    )

    joblib.dump(
        regime_mapping,
        HMM_MAPPING_FILE
    )

    print()
    print(
        "HMM model saved to:"
    )

    print(
        HMM_MODEL_FILE
    )

    print()
    print(
        "HMM state mapping:"
    )

    for state, regime in regime_mapping.items():

        print(
            f"State {state} -> {regime}"
        )

    print()
    print(
        "Average returns by state:"
    )

    for state in sorted(
        state_returns
    ):

        print(
            f"State {state}: "
            f"{state_returns[state] * 100:.4f}%"
        )

    return (
        df,
        model,
        regime_mapping
    )


# ============================================================
# GET CURRENT MARKET REGIME
# ============================================================

def get_current_regime(
    spy_data
):

    # --------------------------------------------------------
    # Check whether trained model exists
    # --------------------------------------------------------

    if not os.path.exists(
        HMM_MODEL_FILE
    ):

        raise FileNotFoundError(
            "HMM model not found. "
            "Train the HMM model first."
        )

    if not os.path.exists(
        HMM_MAPPING_FILE
    ):

        raise FileNotFoundError(
            "HMM state mapping not found."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(
        HMM_MODEL_FILE
    )

    regime_mapping = joblib.load(
        HMM_MAPPING_FILE
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    df = prepare_hmm_data(
        spy_data
    )

    X = (
        df["HMM_Return"]
        .values
        * 100
    ).reshape(
        -1,
        1
    )

    # --------------------------------------------------------
    # Predict states
    # --------------------------------------------------------

    states = model.predict(
        X
    )

    current_state = int(
        states[-1]
    )

    current_regime = (
        regime_mapping
        .get(
            current_state,
            "Unknown"
        )
    )

    return {
        "HMM_State": current_state,
        "Market_Regime": current_regime
    }


# ============================================================
# TRAIN AND GET CURRENT REGIME
# ============================================================

def train_and_get_current_regime(
    spy_data
):

    df, model, regime_mapping = train_hmm(
        spy_data
    )

    current_state = int(
        df["HMM_State"].iloc[-1]
    )

    current_regime = (
        regime_mapping
        .get(
            current_state,
            "Unknown"
        )
    )

    return {
        "HMM_State": current_state,
        "Market_Regime": current_regime
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import yfinance as yf

    print()
    print(
        "Downloading SPY data..."
    )

    spy = yf.download(
        "SPY",
        period="5y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if isinstance(
        spy.columns,
        pd.MultiIndex
    ):

        spy.columns = (
            spy.columns
            .get_level_values(0)
        )

    spy = spy.reset_index()

    if "Date" not in spy.columns:

        spy = spy.rename(
            columns={
                spy.columns[0]: "Date"
            }
        )

    result = train_and_get_current_regime(
        spy
    )

    print()
    print("=" * 70)
    print("CURRENT MARKET REGIME")
    print("=" * 70)

    print(
        f"HMM State: "
        f"{result['HMM_State']}"
    )

    print(
        f"Market Regime: "
        f"{result['Market_Regime']}"
    )

    print("=" * 70)