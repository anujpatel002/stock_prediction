"""
model.py - Machine Learning Model Training, Prediction, and Evaluation
Supports all applicable Scikit-Learn regression algorithms:
  - Random Forest Regressor (Ensemble Bagging)
  - Gradient Boosting Regressor (Ensemble Boosting)
  - Decision Tree Regressor (Single Tree)
  - Linear Regression (Ordinary Least Squares Baseline)
  - Ridge Regression (L2 Regularized)
  - Support Vector Regressor (SVR - RBF Kernel)
  - K-Nearest Neighbors (KNN Regressor)
Also supports both Stationary Returns and Nominal Price modeling strategies.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import (
    NOMINAL_FEATURE_COLUMNS,
    STATIONARY_FEATURE_COLUMNS,
    TARGET_PRICE_COLUMN,
    TARGET_RETURN_COLUMN,
)

SUPPORTED_ALGORITHMS: Dict[str, str] = {
    "random_forest": "Random Forest Regressor",
    "linear_regression": "Linear Regression (OLS)",
    "knn": "K-Nearest Neighbors (KNN)",
    "decision_tree": "Decision Tree Regressor",
}


def _instantiate_model(algorithm_key: str, random_state: int = 42):
    """Factory function to build the requested algorithm pipeline."""
    key = algorithm_key.lower().strip()

    if key == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
    elif key == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            random_state=random_state,
        )
    elif key == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=6,
            random_state=random_state,
        )
    elif key == "linear_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        )
    elif key == "ridge":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0)),
            ]
        )
    elif key == "svr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", SVR(C=1.0, epsilon=0.01)),
            ]
        )
    elif key == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", KNeighborsRegressor(n_neighbors=5)),
            ]
        )
    else:
        # Default fallback to Random Forest
        return RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )


def _extract_feature_importances(model, X_test, y_test, feature_cols: List[str]) -> np.ndarray:
    """Extracts or estimates normalized feature importance for any regressor."""
    try:
        # 1. Direct Gini importance (Random Forest, Gradient Boosting, Decision Tree)
        if hasattr(model, "feature_importances_"):
            raw = np.array(model.feature_importances_)
            total = raw.sum()
            return raw / total if total > 0 else np.ones(len(raw)) / len(raw)

        # 2. Linear / Ridge coefficients from pipeline
        if hasattr(model, "named_steps") and hasattr(model.named_steps["reg"], "coef_"):
            raw = np.abs(model.named_steps["reg"].coef_)
            total = raw.sum()
            return raw / total if total > 0 else np.ones(len(raw)) / len(raw)

        # 3. Model-agnostic Permutation Importance (SVR, KNN)
        perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
        raw = np.maximum(0, perm.importances_mean)
        total = raw.sum()
        if total > 0:
            return raw / total
        return np.ones(len(feature_cols)) / len(feature_cols)

    except Exception:
        return np.ones(len(feature_cols)) / len(feature_cols)


def train_and_evaluate(
    model_df: pd.DataFrame,
    latest_features: pd.DataFrame,
    mode: str = "stationary",
    algorithm: str = "random_forest",
    train_ratio: float = 0.80,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains the selected ML regression algorithm on chronological historical stock data
    and generates evaluation metrics along with next-trading-day predictions.

    Parameters:
        model_df (pd.DataFrame): Preprocessed dataframe with features and targets.
        latest_features (pd.DataFrame): Most recent single row of features.
        mode (str): 'stationary' (relative percentage returns) or 'nominal' (absolute price level).
        algorithm (str): Key identifying the ML algorithm from SUPPORTED_ALGORITHMS.
        train_ratio (float): Chronological train split ratio (default 0.80).
        random_state (int): Reproducibility seed.

    Returns:
        dict: Containing model, metrics, test evaluation dataframe,
              feature importances, and next-day forecast.
    """
    total_samples = len(model_df)
    if total_samples < 20:
        raise ValueError(
            f"Insufficient samples ({total_samples}) for reliable time-series modeling. "
            "The stock may be very newly listed with fewer than 20 trading sessions. "
            "Please select an equity with at least 1 month of trading history."
        )

    # 1. Select Features and Target based on Modeling Strategy
    if mode == "stationary":
        feature_cols = STATIONARY_FEATURE_COLUMNS
        target_col = TARGET_RETURN_COLUMN
    else:
        feature_cols = NOMINAL_FEATURE_COLUMNS
        target_col = TARGET_PRICE_COLUMN

    # 2. Chronological Train/Test Split (Strictly NO random shuffling)
    split_index = int(total_samples * train_ratio)
    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    y_test_price = test_df[TARGET_PRICE_COLUMN]

    # 3. Model Initialization and Training
    model = _instantiate_model(algorithm, random_state=random_state)
    model.fit(X_train, y_train)

    # 4. Model Inference & Price Reconstruction
    if mode == "stationary":
        test_pred_returns = model.predict(X_test)
        test_predictions = test_df["Close"].values * (1.0 + test_pred_returns)
    else:
        test_predictions = model.predict(X_test)

    # 5. Evaluation Metrics Calculation
    mae = float(mean_absolute_error(y_test_price, test_predictions))
    mse = float(mean_squared_error(y_test_price, test_predictions))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test_price, test_predictions))

    # 6. Build Test Results DataFrame
    test_results_df = pd.DataFrame(
        {
            "Date": test_df["Date"].values,
            "Actual_Price": y_test_price.values,
            "Predicted_Price": test_predictions,
            "Residual": y_test_price.values - test_predictions,
        }
    )

    # 7. Feature Importance Extraction
    readable_names = {
        "Daily_Return": "Daily Return",
        "Ratio_MA5": "5-Day MA Ratio",
        "Ratio_MA20": "20-Day MA Ratio",
        "Rel_High_Low": "Intraday Spread Ratio",
        "Volume_Change": "Volume Change",
        "Volatility": "10-Day Volatility",
        "Previous_Close": "Previous Close",
        "MA_5": "5-Day Moving Avg",
        "MA_20": "20-Day Moving Avg",
        "High_Low_Range": "High-Low Range",
    }
    feature_labels = [readable_names.get(col, col) for col in feature_cols]
    importances = _extract_feature_importances(model, X_test, y_test, feature_cols)

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_labels,
            "Raw_Feature": feature_cols,
            "Importance": importances,
        }
    ).sort_values("Importance", ascending=True)

    # 8. Next Trading Day Prediction
    latest_X = latest_features[feature_cols]
    current_close = float(latest_features["Close"].values[0])
    last_trading_date = pd.to_datetime(latest_features["Date"].values[0])

    if mode == "stationary":
        next_pred_return = float(model.predict(latest_X)[0])
        predicted_next_close = float(current_close * (1.0 + next_pred_return))
        expected_change_pct = next_pred_return * 100.0
        expected_change_abs = predicted_next_close - current_close
    else:
        predicted_next_close = float(model.predict(latest_X)[0])
        expected_change_abs = predicted_next_close - current_close
        expected_change_pct = (expected_change_abs / current_close) * 100.0

    algo_display_name = SUPPORTED_ALGORITHMS.get(algorithm.lower().strip(), "Machine Learning Regressor")

    return {
        "model": model,
        "algorithm_key": algorithm,
        "algorithm_name": algo_display_name,
        "mode": mode,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        },
        "test_results_df": test_results_df,
        "feature_importance_df": feature_importance_df,
        "prediction": {
            "last_date": last_trading_date,
            "current_close": current_close,
            "predicted_next_close": predicted_next_close,
            "expected_change_abs": expected_change_abs,
            "expected_change_pct": expected_change_pct,
        },
    }
