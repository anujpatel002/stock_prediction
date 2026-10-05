# model.py - Train a machine learning model and predict tomorrow's stock price

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import (
    NOMINAL_FEATURE_COLUMNS,
    STATIONARY_FEATURE_COLUMNS,
    TARGET_PRICE_COLUMN,
    TARGET_RETURN_COLUMN,
)

# Human-readable names shown in the UI dropdown
SUPPORTED_ALGORITHMS = {
    "random_forest":    "Random Forest Regressor",
    "linear_regression":"Linear Regression (OLS)",
    "knn":              "K-Nearest Neighbors (KNN)",
    "decision_tree":    "Decision Tree Regressor",
}

# Readable labels for the feature importance chart
FEATURE_LABELS = {
    "Previous_Close":  "Previous Close",
    "Daily_Return":    "Daily Return",
    "MA_5":            "5-Day Moving Avg",
    "MA_20":           "20-Day Moving Avg",
    "High_Low_Range":  "High-Low Range",
    "Volume_Change":   "Volume Change",
    "Volatility":      "10-Day Volatility",
    "Ratio_MA5":       "5-Day MA Ratio",
    "Ratio_MA20":      "20-Day MA Ratio",
    "Rel_High_Low":    "Intraday Spread Ratio",
}


def _build_model(algorithm: str, seed: int = 42):
    """
    Return the right scikit-learn model object for the chosen algorithm.
    Linear models and KNN need StandardScaler, so we wrap them in a Pipeline.
    """
    # Models that need feature scaling
    scaled = {
        "linear_regression": LinearRegression(),
        "knn":               KNeighborsRegressor(n_neighbors=5),
    }
    # Tree-based models — no scaling needed
    trees = {
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "decision_tree": DecisionTreeRegressor(max_depth=6, random_state=seed),
    }

    key = algorithm.lower().strip()

    if key in scaled:
        return Pipeline([("scaler", StandardScaler()), ("reg", scaled[key])])
    if key in trees:
        return trees[key]

    # Default fallback
    return trees["random_forest"]


def _get_importances(model, feature_cols: list) -> np.ndarray:
    """
    Get how important each feature was for the model's predictions.
    - Tree models have a built-in .feature_importances_ attribute.
    - Linear models use the size of their coefficients.
    - Returns a normalized array (values add up to 1.0).
    """
    # Tree models (Random Forest, Decision Tree)
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_

    # Linear models inside a Pipeline
    elif hasattr(model, "named_steps") and hasattr(model.named_steps["reg"], "coef_"):
        raw = np.abs(model.named_steps["reg"].coef_)

    # KNN has no built-in importance — give every feature equal weight
    else:
        return np.ones(len(feature_cols)) / len(feature_cols)

    total = raw.sum()
    return raw / total if total > 0 else np.ones(len(raw)) / len(raw)


def train_and_evaluate(
    model_df,
    latest_features,
    mode: str = "stationary",
    algorithm: str = "random_forest",
    train_ratio: float = 0.80,
    random_state: int = 42,
) -> dict:
    """
    Train the model on historical data and predict tomorrow's closing price.

    model_df        — cleaned DataFrame with features + target columns
    latest_features — the most recent row (used to predict tomorrow)
    mode            — 'stationary' (predict % return) or 'nominal' (predict raw price)
    algorithm       — which ML algorithm to use
    train_ratio     — fraction of data used for training (0.80 = 80%)

    Returns a dict with metrics, test results, feature importances, and the prediction.
    """
    if len(model_df) < 20:
        raise ValueError(
            f"Only {len(model_df)} rows available. Need at least 20 trading days of data."
        )

    # Pick features and target column based on mode
    if mode == "stationary":
        features = STATIONARY_FEATURE_COLUMNS
        target   = TARGET_RETURN_COLUMN
    else:
        features = NOMINAL_FEATURE_COLUMNS
        target   = TARGET_PRICE_COLUMN

    # --- Chronological 80/20 split (NO shuffling — future data must not leak into training) ---
    split = int(len(model_df) * train_ratio)
    train = model_df.iloc[:split]
    test  = model_df.iloc[split:]

    X_train, y_train = train[features], train[target]
    X_test,  y_test  = test[features],  test[target]
    actual_prices    = test[TARGET_PRICE_COLUMN]   # always compare in ₹

    # --- Train ---
    model = _build_model(algorithm, seed=random_state)
    model.fit(X_train, y_train)

    # --- Predict on test set ---
    if mode == "stationary":
        # Model predicts % return → convert back to price
        pred_returns   = model.predict(X_test)
        test_preds     = test["Close"].values * (1.0 + pred_returns)
    else:
        test_preds = model.predict(X_test)

    # --- Evaluation metrics ---
    mae  = float(mean_absolute_error(actual_prices, test_preds))
    rmse = float(np.sqrt(mean_squared_error(actual_prices, test_preds)))
    r2   = float(r2_score(actual_prices, test_preds))

    # --- Test results table (for the Actual vs Predicted chart) ---
    test_results_df = pd.DataFrame({
        "Date":            test["Date"].values,
        "Actual_Price":    actual_prices.values,
        "Predicted_Price": test_preds,
        "Residual":        actual_prices.values - test_preds,
    })

    # --- Feature importance ---
    importances = _get_importances(model, features)
    feature_importance_df = pd.DataFrame({
        "Feature":    [FEATURE_LABELS.get(f, f) for f in features],
        "Raw_Feature": features,
        "Importance":  importances,
    }).sort_values("Importance", ascending=True)

    # --- Predict tomorrow ---
    latest_X      = latest_features[features]
    current_close = float(latest_features["Close"].values[0])
    last_date     = pd.to_datetime(latest_features["Date"].values[0])

    if mode == "stationary":
        pred_return        = float(model.predict(latest_X)[0])
        predicted_price    = current_close * (1.0 + pred_return)
        change_pct         = pred_return * 100.0
    else:
        predicted_price    = float(model.predict(latest_X)[0])
        change_pct         = (predicted_price - current_close) / current_close * 100.0

    change_abs = predicted_price - current_close

    return {
        "model":                model,
        "algorithm_key":        algorithm,
        "algorithm_name":       SUPPORTED_ALGORITHMS.get(algorithm, "ML Regressor"),
        "mode":                 mode,
        "train_size":           len(train),
        "test_size":            len(test),
        "metrics":              {"MAE": mae, "RMSE": rmse, "R2": r2},
        "test_results_df":      test_results_df,
        "feature_importance_df": feature_importance_df,
        "prediction": {
            "last_date":           last_date,
            "current_close":       current_close,
            "predicted_next_close": predicted_price,
            "expected_change_abs": change_abs,
            "expected_change_pct": change_pct,
        },
    }
