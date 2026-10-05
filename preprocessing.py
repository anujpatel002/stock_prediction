"""
preprocessing.py - Data Cleaning and Feature Engineering Module
Performs data sanitation, robust verification, rolling-window calculations,
and both nominal and stationary feature engineering for stock price prediction.
"""

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

# Nominal features (Baseline mode)
NOMINAL_FEATURE_COLUMNS: List[str] = [
    "Previous_Close",
    "Daily_Return",
    "MA_5",
    "MA_20",
    "High_Low_Range",
    "Volume_Change",
    "Volatility",
]

# Stationary scale-invariant features (Advanced mode)
STATIONARY_FEATURE_COLUMNS: List[str] = [
    "Daily_Return",
    "Ratio_MA5",
    "Ratio_MA20",
    "Rel_High_Low",
    "Volume_Change",
    "Volatility",
]

# Alias for backwards compatibility
FEATURE_COLUMNS: List[str] = NOMINAL_FEATURE_COLUMNS

TARGET_PRICE_COLUMN: str = "Target"
TARGET_RETURN_COLUMN: str = "Target_Return"


def preprocess_and_engineer_features(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Cleans raw OHLCV stock data, computes financial technical features
    (both raw nominal indicators and stationary scale-invariant ratios),
    and structures training/unseen prediction splits.

    Parameters:
        raw_df (pd.DataFrame): Raw historical dataframe with OHLCV columns.

    Returns:
        Tuple:
            - full_df (pd.DataFrame): Entire dataframe with calculated features.
            - model_df (pd.DataFrame): Clean historical dataset with target for ML.
            - latest_features (pd.DataFrame): Latest single row for next trading day prediction.
            - audit_stats (dict): Dictionary of preprocessing audit metrics.
    """
    df = raw_df.copy()

    # 1. Verification and Sorting
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)

    initial_rows = len(df)

    # 2. Duplicate Detection and Removal
    duplicates_count = int(df.duplicated(subset=["Date"]).sum())
    if duplicates_count > 0:
        df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    # 3. Invalid-value checking
    invalid_mask = (df["Close"] <= 0) | (df["High"] < df["Low"])
    invalid_rows_count = int(invalid_mask.sum())
    if invalid_rows_count > 0:
        df = df[~invalid_mask].reset_index(drop=True)

    # Handle zero or missing volume to prevent division by zero in pct_change
    df["Volume"] = df["Volume"].replace(0, np.nan).ffill().bfill()

    # 4. Feature Engineering
    # (a) Previous Close & Daily Return
    df["Previous_Close"] = df["Close"].shift(1)
    df["Daily_Return"] = (df["Close"] / df["Previous_Close"]) - 1

    # (b) Moving Averages
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()

    # (c) High-Low Range
    df["High_Low_Range"] = df["High"] - df["Low"]

    # (d) Volume Change
    df["Volume_Change"] = df["Volume"].pct_change()

    # (e) Historical Volatility (10-day rolling std of returns)
    df["Volatility"] = df["Daily_Return"].rolling(window=10).std()

    # (f) Stationary Scale-Invariant Ratios (Relative to current Close)
    # Eliminates non-stationarity and extrapolation barrier for tree models
    df["Ratio_MA5"] = df["MA_5"] / df["Close"]
    df["Ratio_MA20"] = df["MA_20"] / df["Close"]
    df["Rel_High_Low"] = df["High_Low_Range"] / df["Close"]

    # (g) Targets
    # Nominal price target
    df["Target"] = df["Close"].shift(-1)
    # Stationary percentage return target
    df["Target_Return"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]

    # 5. Clean up infinities and rolling window warm-up NaNs
    df = df.replace([np.inf, -np.inf], np.nan)

    all_features = list(set(NOMINAL_FEATURE_COLUMNS + STATIONARY_FEATURE_COLUMNS))

    # Extract latest row (has all features computed, but tomorrow target is NaN)
    valid_features_mask = df[all_features].notna().all(axis=1)
    latest_features = df[valid_features_mask].iloc[[-1]].copy()

    # Model training dataset: where both features AND targets are present
    model_df = df.dropna(subset=all_features + [TARGET_PRICE_COLUMN, TARGET_RETURN_COLUMN]).copy().reset_index(drop=True)

    audit_stats = {
        "initial_rows": initial_rows,
        "duplicates_removed": duplicates_count,
        "invalid_rows_removed": invalid_rows_count,
        "nominal_features": len(NOMINAL_FEATURE_COLUMNS),
        "stationary_features": len(STATIONARY_FEATURE_COLUMNS),
        "usable_training_rows": len(model_df),
        "warmup_rows_dropped": initial_rows - len(model_df) - 1,
    }

    return df, model_df, latest_features, audit_stats
