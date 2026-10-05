# preprocessing.py - Clean data and create features for the ML model

import numpy as np
import pandas as pd

# Features used when predicting raw price (Nominal mode)
NOMINAL_FEATURE_COLUMNS = [
    "Previous_Close",   # yesterday's closing price
    "Daily_Return",     # % change from yesterday to today
    "MA_5",             # average close over last 5 days
    "MA_20",            # average close over last 20 days
    "High_Low_Range",   # today's high minus today's low
    "Volume_Change",    # % change in trading volume
    "Volatility",       # how much returns varied over last 10 days
]

# Features used when predicting % return (Stationary mode — more robust)
STATIONARY_FEATURE_COLUMNS = [
    "Daily_Return",     # % change today
    "Ratio_MA5",        # MA_5 divided by Close (scale-free)
    "Ratio_MA20",       # MA_20 divided by Close (scale-free)
    "Rel_High_Low",     # High-Low range divided by Close (scale-free)
    "Volume_Change",    # % change in volume
    "Volatility",       # 10-day rolling std of returns
]

# Alias so app.py can import FEATURE_COLUMNS without breaking
FEATURE_COLUMNS = NOMINAL_FEATURE_COLUMNS

TARGET_PRICE_COLUMN  = "Target"         # next day's closing price
TARGET_RETURN_COLUMN = "Target_Return"  # next day's % return


def preprocess_and_engineer_features(raw_df):
    """
    Step 1 — Clean the raw data.
    Step 2 — Build 7 technical features from OHLCV columns.
    Step 3 — Create the target (what we want to predict).
    Step 4 — Split into training data and the latest row for prediction.

    Returns:
        full_df        — all rows with features (used for charts)
        model_df       — rows where both features AND target exist (used for training)
        latest_row     — the most recent row (used to predict tomorrow)
        audit_stats    — a small summary of what was cleaned
    """
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)

    initial_rows = len(df)

    # --- Step 1: Remove bad rows ---
    # Drop rows where Close is zero/negative or High is less than Low (impossible)
    bad_rows = (df["Close"] <= 0) | (df["High"] < df["Low"])
    df = df[~bad_rows].reset_index(drop=True)
    invalid_removed = int(bad_rows.sum())

    # Replace zero volume with NaN then forward-fill
    # (zero volume causes division-by-zero when computing Volume_Change)
    df["Volume"] = df["Volume"].replace(0, np.nan).ffill().bfill()

    # --- Step 2: Build features ---
    df["Previous_Close"]  = df["Close"].shift(1)
    df["Daily_Return"]    = df["Close"].pct_change()                        # (Close_t - Close_{t-1}) / Close_{t-1}
    df["MA_5"]            = df["Close"].rolling(5).mean()
    df["MA_20"]           = df["Close"].rolling(20).mean()
    df["High_Low_Range"]  = df["High"] - df["Low"]
    df["Volume_Change"]   = df["Volume"].pct_change()
    df["Volatility"]      = df["Daily_Return"].rolling(10).std()

    # Scale-free versions (divide by Close so price level doesn't matter)
    df["Ratio_MA5"]       = df["MA_5"]  / df["Close"]
    df["Ratio_MA20"]      = df["MA_20"] / df["Close"]
    df["Rel_High_Low"]    = df["High_Low_Range"] / df["Close"]

    # --- Step 3: Create targets ---
    df["Target"]          = df["Close"].shift(-1)                           # next day's price
    df["Target_Return"]   = df["Target"] / df["Close"] - 1                 # next day's % return

    # Replace any infinity values (e.g. from dividing by zero) with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # --- Step 4: Split rows ---
    all_features = list(set(NOMINAL_FEATURE_COLUMNS + STATIONARY_FEATURE_COLUMNS))

    # latest_row: the last row that has all features computed (target will be NaN — that's fine)
    has_features = df[all_features].notna().all(axis=1)
    latest_row = df[has_features].iloc[[-1]].copy()

    # model_df: rows where features AND both targets are all present (used for training)
    model_df = df.dropna(
        subset=all_features + [TARGET_PRICE_COLUMN, TARGET_RETURN_COLUMN]
    ).reset_index(drop=True)

    audit_stats = {
        "initial_rows":        initial_rows,
        "duplicates_removed":  initial_rows - len(df) - invalid_removed,
        "invalid_rows_removed": invalid_removed,
        "usable_training_rows": len(model_df),
        "warmup_rows_dropped": initial_rows - len(model_df) - 1,
        "nominal_features":    len(NOMINAL_FEATURE_COLUMNS),
        "stationary_features": len(STATIONARY_FEATURE_COLUMNS),
    }

    return df, model_df, latest_row, audit_stats
