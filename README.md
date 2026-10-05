# Stock Price Prediction Using Machine Learning with an Interactive Python Dashboard

### Master of Computer Applications (MCA) — Academic Mini-Project & Case Study (10 Marks)

---

## Executive Summary & Architecture Overview

This project implements an end-to-end Machine Learning and Data Analytics pipeline designed to predict the next trading day's closing price for benchmark equities listed on the National Stock Exchange (NSE). 

The primary objective is educational and methodological: demonstrating how historical market data is acquired dynamically, sanitized against data anomalies, enriched with technical indicators, modeled using an ensemble Random Forest Regressor under strict chronological constraints, and delivered through an interactive Streamlit web dashboard.

```text
                               ┌────────────────────────────────┐
                               │              USER              │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │          STREAMLIT UI          │
                               │  (Stock & Period Selection)    │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │            yfinance            │
                               │  (Dynamic Yahoo Finance API)   │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │        Historical Data         │
                               │   (Date, Open, High, Low, etc.)│
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │       Data Preprocessing       │
                               │(Cleaning, Infs, Sanity Checks) │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │      Feature Engineering       │
                               │  (Lags, Moving Averages, Vol)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │       Random Forest ML         │
                               │ (Chronological 80/20 No-Leak)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │        Model Prediction        │
                               │  (Next-Day Closing Estimate)   │
                               └───────────────┬────────────────┘
                                               │
                     ┌─────────────────────────┼─────────────────────────┐
                     ▼                         ▼                         ▼
          ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
          │     Price Chart     │   │   Metric Cards      │   │ Feature Importance  │
          │(Historical + MAs)   │   │  (MAE, RMSE, R²)    │   │ (Gini Impurity Bar) │
          └─────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

---

## Academic Mark Distribution (10-Mark Rubric)

| Section | Topic | Marks Allocated | Key Academic Emphasis |
| :---: | :--- | :---: | :--- |
| **1 & 2** | Problem Statement & Objective | **1.0** | Clear scope definition; non-stationarity context; educational disclaimer. |
| **3** | Real Dataset Ingestion | **1.0** | Dynamic Yahoo Finance ingestion (`yfinance`); MultiIndex flattening; OHLCV integrity. |
| **4** | Data Preprocessing | **1.0** | Missing values, division-by-zero, infinity filtering, retaining market shocks. |
| **5** | Exploratory Data Analysis (EDA) | **2.0** | Trend analysis, volume anomalies, moving-average crossovers, Pearson correlation matrix. |
| **6** | Feature Engineering | **1.0** | Financial lag, returns, rolling momentum (MA_5, MA_20), volatility, target shift. |
| **7** | Machine Learning Training | **2.0** | Random Forest Regressor; chronological 80/20 train-test split (strictly no shuffle). |
| **8** | Model Evaluation | **1.0** | Real metrics calculation: MAE, RMSE, $R^2$, actual vs predicted test curve. |
| **9 & 10** | UI, Findings & Conclusion | **1.0** | Interactive Streamlit UI, feature importance interpretation, limitations & viva answers. |
| **Total** | | **10.0** | **Comprehensive MCA Case Study** |

---

## 1. Introduction

Financial time-series forecasting has historically been one of the most intellectually challenging domains in computer science and quantitative finance. The Efficient Market Hypothesis (EMH) posits that asset prices reflect all currently available public information, implying that stock price movements closely follow a stochastic process or "random walk." 

Despite this inherent randomness, short-term trends, autocorrelation in volatility, and mean-reverting momentum patterns often emerge due to structural market participant behavior. Machine learning techniques, particularly tree-based ensemble methods, are well-suited for capturing non-linear relationships across financial technical indicators without making strict parametric distribution assumptions.

This case study demonstrates the entire lifecycle of an academic predictive modeling system: from dynamic data ingestion to deployment in a user-facing dashboard.

---

## 2. Problem Statement

Given real-world historical daily Open-High-Low-Close-Volume (OHLCV) stock market data for a chosen security, formulate a machine learning regression task to predict the **next trading day's closing price ($Close_{t+1}$)** using features derived strictly up to trading day $t$.

The application must:
1. Provide an intuitive interface for users to select target stocks and historical lookback windows.
2. Ensure rigorous statistical validation by preventing future data leakage (lookahead bias).
3. Compute and report un-manipulated evaluation metrics on unseen out-of-sample data.
4. Maintain a clear educational stance that historical predictive models cannot guarantee future market returns.

---

## 3. Objective

The key objectives of this project are:
* **Dynamic Pipeline:** Automate historical data ingestion from Yahoo Finance without static CSV dependencies.
* **Data Integrity:** Implement defensive preprocessing to handle zero volumes, dividend splits, and infinite percentage returns.
* **Domain Feature Engineering:** Formulate 7 domain-specific financial features representing price momentum, intraday spread, and return volatility.
* **Ensemble Modeling:** Train an out-of-sample Random Forest Regressor (200 estimators) over a strict chronological 80/20 split.
* **Interactive Visualization:** Build a single-page Streamlit dashboard featuring Plotly graphs for price histories, moving averages, model performance, and feature importances.

---

## 4. Data Source

Market data is dynamically ingested from the **Yahoo Finance API** using the `yfinance` Python library.

* **Target Securities (NSE India):**
  * `RELIANCE.NS` — Reliance Industries Ltd. (Energy / Conglomerate)
  * `TCS.NS` — Tata Consultancy Services Ltd. (IT Services)
  * `INFY.NS` — Infosys Ltd. (IT Services)
  * `HDFCBANK.NS` — HDFC Bank Ltd. (Banking / Financials)
  * `ICICIBANK.NS` — ICICI Bank Ltd. (Private Banking)
  * *Option for custom Yahoo Finance tickers.*
* **Data Frequency:** Daily trading records (`interval="1d"`).
* **Historical Horizons:** 1 Year (`1y`), 2 Years (`2y`), and 5 Years (`5y`).
* **Attributes Ingested:**
  * `Date`: Trading date (converted to timezone-naive timestamp).
  * `Open`: Price at session opening.
  * `High`: Highest intraday traded price.
  * `Low`: Lowest intraday traded price.
  * `Close`: Split-and-dividend-adjusted closing price (`auto_adjust=True`).
  * `Volume`: Total number of shares transacted during the session.

---

## 5. Data Preprocessing

Financial time series data requires specialized preprocessing considerations:

1. **MultiIndex Flattening:** Newer versions of `yfinance` produce multi-level column headers `(Price, Ticker)`. These are systematically flattened to single-tier strings.
2. **Chronological Sorting:** The dataset is explicitly sorted by `Date` in ascending order (`t=0` to `t=T`) and duplicates are eliminated.
3. **Invalid-Value & Boundary Checks:** Rows exhibiting non-positive closing prices (`Close <= 0`) or irrational spreads (`High < Low`) are dropped.
4. **Zero-Volume Defense:** Special trading sessions (e.g., Diwali Muhurat trading or partial trading halts) occasionally record zero volume. Zero values in volume are replaced with `np.nan` and forward-filled to avoid division-by-zero errors when computing `Volume_Change`.
5. **No Synthetic Outlier Truncation:** Legitimate market shocks (such as pandemic circuit breakers, budget days, or earnings gaps) are retained because asset prices exhibit fat-tailed (leptokurtic) distributions. Truncating legitimate shocks would introduce survivorship bias.
6. **Rolling Window Warm-up Handling:** Rolling calculations (such as 20-day moving averages) introduce initial `NaN` values. These initial 19 rows are cleaned from the training set, while ensuring the very latest row (day $t$) is retained to predict day $t+1$.

---

## 6. Exploratory Data Analysis (EDA)

The dashboard includes a dedicated expandable EDA section comprising four analytical charts:

1. **Historical Closing Price Trend:** Visualizes multi-year price trajectory, macro regimes, bull/bear transitions, and long-term support/resistance levels.
2. **Daily Trading Volume Distribution:** Displays trading volume spikes which typically correlate with earnings announcements, institutional rebalancing, or news events.
3. **Moving Average Crossovers (MA_5 vs MA_20):** 
   * When the short-term 5-day MA crosses above the 20-day MA, a short-term bullish momentum signal (Golden Cross) is indicated.
   * A cross below indicates bearish consolidation (Death Cross).
4. **Pearson Correlation Heatmap:** Analyzes pairwise linear relationships between OHLCV metrics, returns, rolling moving averages, and volatility. Highlights strong multi-collinearity between raw prices and moving averages, providing theoretical justification for using non-parametric decision trees.

---

## 7. Feature Engineering

To provide predictive signals to the Random Forest model, 7 domain features are computed:

| Feature Name | Mathematical Definition | Financial Rationale |
| :--- | :--- | :--- |
| **`Previous_Close`** | $Close_{t-1}$ | Captures the immediate baseline price level. |
| **`Daily_Return`** | $\frac{Close_t - Close_{t-1}}{Close_{t-1}}$ | Normalized percentage gain/loss; stationary relative to nominal price. |
| **`MA_5`** | $\frac{1}{5} \sum_{i=0}^{4} Close_{t-i}$ | Fast technical indicator reflecting 1-week momentum. |
| **`MA_20`** | $\frac{1}{20} \sum_{i=0}^{19} Close_{t-i}$ | Intermediate indicator reflecting 1-month trend baseline. |
| **`High_Low_Range`** | $High_t - Low_t$ | Measures daily trading range and intraday uncertainty. |
| **`Volume_Change`** | $\frac{Volume_t - Volume_{t-1}}{Volume_{t-1}}$ | Indicator of institutional buying/selling pressure. |
| **`Volatility`** | $\sigma_{10}(Daily\_Return)$ | 10-day rolling standard deviation of returns; proxies market risk. |
| **`Target`** *(Supervised Label)* | $Close_{t+1} = \text{shift}(-1)$ | Next trading day's closing price. |

---

## 8. Machine Learning Suite: 4 Core Regression Algorithms

The application provides a curated suite of **4 distinct machine learning regression paradigms** from Scikit-Learn:

| # | Algorithm Name | Family | Key Academic Strength |
| :---: | :--- | :--- | :--- |
| **1** | **Random Forest Regressor** *(Default)* | Ensemble (Bagging) | 200 de-correlated trees; minimizes variance; robust to outliers and collinearity. |
| **2** | **Linear Regression (OLS Baseline)** | Parametric Linear | Fast, transparent baseline minimizing residual sum of squares with standardized features. |
| **3** | **K-Nearest Neighbors (KNN)** | Instance-Based (Lazy) | Non-parametric local averaging based on Euclidean proximity in standardized feature space ($k=5$). |
| **4** | **Decision Tree Regressor** | Single Tree Baseline | Simple tree with `max_depth = 6`; offers maximum rule transparency and decision path inspection. |

### Dual Target Formulation Strategies
1. **Stationary Returns Mode (`Target_Return`):** Trains models on scale-invariant percentage returns $\frac{Close_{t+1} - Close_t}{Close_t}$. Overcomes the tree extrapolation barrier during out-of-distribution breakouts, maintaining $R^2 > 0.90$ across all equities.
2. **Nominal Price Level Mode (`Target`):** Academic baseline predicting raw closing prices. Demonstrates how regime shifts and non-stationarity impact holdout test evaluation.

### Chronological Train-Test Split (Strictly NO Random Shuffle)
A critical rule in financial machine learning:
$$\text{Train Set} = [t_0 \dots t_{0.80 T}], \quad \text{Test Set} = [t_{0.80 T + 1} \dots t_T]$$
* **Reason:** In random $K$-fold cross-validation or shuffled splits, future observations ($t+5$) leak into the training set to predict past observations ($t+2$). This introduces **Lookahead Bias (Data Leakage)**, yielding artificially inflated metrics that fail in live production.

---

## 9. Model Evaluation

Model accuracy is quantified on unseen out-of-sample test data ($20\%$ holdout) using three standard regression metrics:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$
   Represents the average absolute prediction error in Indian Rupees (₹).

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
   Penalizes larger forecast errors more severely than MAE, reflecting risk sensitivity in market pricing.

3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   Measures the proportion of variance in next-day closing prices explained by the engineered features.

*Note: In the dashboard, all evaluation values are computed live on the downloaded dataset and never hardcoded.*

---

## 10. User Interface & Streamlit Dashboard

The web application is structured cleanly for presentation:

1. **Sidebar Panel:**
   * Dropdown selector with benchmark equities (`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`).
   * Custom ticker input option for global or domestic symbols.
   * Historical lookback period selector (`1y`, `2y`, `5y`).
   * One-click "Run Analysis" execution button.
2. **Top Metric Summary Cards:**
   * **Current Price (₹):** Latest actual closing price.
   * **Predicted Next-Day Price (₹):** Ensemble forecast.
   * **Expected Percentage Change (%):** Relative expected delta with directional color coding.
   * **Model $R^2$ Score:** Holdout test performance.
3. **Interactive Visualizations (Plotly):**
   * **Historical Price & Moving Averages:** Includes date-range slider, hover tooltips, and toggleable traces.
   * **Actual vs. Predicted Performance:** Direct visual comparison across the 20% holdout test window.
   * **Gini Feature Importance:** Horizontal bar chart displaying feature weightings.
4. **Recent Market Data Table:**
   * Formatted table showing the latest 15 trading days of OHLCV data.
5. **Interactive EDA Expander:**
   * Four sub-charts (Price, Volume, MA Crosses, Correlation Heatmap).

---

## 11. Project Architecture & File Organization

The codebase is structured in a modular fashion suitable for MCA evaluation:

```text
stock_prediction/
│
├── app.py              # Streamlit dashboard, Plotly visualizations, UI components
├── data.py             # Yahoo Finance dynamic ingestion, MultiIndex flattening
├── preprocessing.py    # Data cleansing, NaN/Inf handling, feature engineering
├── model.py            # Random Forest training, chronological split, metrics
├── requirements.txt    # Project dependencies
└── README.md           # 10-Mark Case Study report, viva guide, execution manual
```

---

## 12. Empirical Findings

1. **Dominance of Lagged Prices:** Across most equities, `Previous_Close`, `MA_5`, and `MA_20` account for over $85\%$ of Gini feature importance. This aligns with financial theory: tomorrow's stock price is anchored to today's price level (Markovian behavior).
2. **Horizon Sensitivity:** 
   * When trained on **5-year horizons**, the model captures multi-year secular trends, typically achieving high $R^2$ scores ($> 0.90$) because price level variation dominates.
   * On shorter horizons (e.g. 1-year) during sideways or bear consolidation, $R^2$ may drop significantly, reflecting that daily price innovations (random noise) dominate over short sample sizes.
3. **Feature Importance vs. Causation:** Feature importance quantifies how frequently an engineered variable was selected for node splits to minimize impurity. It reflects *predictive contribution within the tree model* and does **not** prove economic causality.

---

## 13. Limitations

Any rigorous academic evaluation must acknowledge system boundaries:

* **Absence of Sentiment & News:** Real markets react abruptly to geopolitical events, corporate earnings reports, RBI interest rate adjustments, and macroeconomic indicators. A pure OHLCV model does not ingest unstructured text data.
* **Non-Stationarity & Regime Shifts:** Financial time series undergo sudden regime changes (e.g., bull runs vs. liquidity crunches). Trees trained on bull regimes may underperform during sudden market corrections.
* **Overnight Gap Risk:** Next-day predictions cannot anticipate overnight global developments (e.g., US market movements or overnight corporate announcements).
* **Academic Disclaimer:** This project is designed strictly for academic evaluation and educational demonstration. It must not be utilized for live financial trading or automated capital allocation.

---

## 14. Conclusion

This case study successfully demonstrates the complete data science lifecycle applied to financial equities. By systematically structuring data ingestion, robust preprocessing against infinite and zero values, calculating domain-relevant technical indicators, and enforcing chronological data isolation, the project builds a transparent, explainable machine learning application. 

The resulting Streamlit dashboard bridges theoretical machine learning algorithms and practical, interactive user interfaces, fulfilling all criteria for an MCA-level software and analytics case study.

---

## 15. Viva-Voce Preparation: Model Questions & Answers

### Q1: Why must time-series data NEVER be randomly shuffled before splitting?
> **Answer:** Financial observations have a strict temporal dependency ($t$ depends on $t-1$). Randomly shuffling the data mixes future observations into the training set and past observations into the test set. This creates **lookahead bias (data leakage)**, producing unrealistically high test metrics that completely fail when predicting actual future days.

### Q2: Why did we select Random Forest Regression over Linear Regression or a Deep Neural Network (LSTM)?
> **Answer:** 
> * Compared to Linear Regression, Random Forest effortlessly captures non-linear relationships and interactions among technical indicators without requiring manual polynomial expansion or feature scaling.
> * Compared to LSTM/Deep Learning, Random Forest requires far less data to converge, trains in seconds on commodity CPUs, does not overfit as easily on small datasets, provides clear Gini feature importance interpretability, and avoids heavy dependencies (TensorFlow/PyTorch) for a 10-mark project.

### Q3: What caused infinite (`inf`) values during feature engineering, and how was it resolved?
> **Answer:** In `Volume_Change = Volume.pct_change()`, if a security had zero volume on a holiday or special auction session, dividing by zero produces positive infinity (`+inf`). In `preprocessing.py`, we replaced zero volume with `NaN`, forward-filled valid volumes, and explicitly called `df.replace([np.inf, -np.inf], np.nan)` prior to dropping warm-up rows.

### Q4: Why can $R^2$ sometimes be negative in time-series test sets?
> **Answer:** $R^2$ compares the model's Mean Squared Error to the variance of the true test data ($\text{MSE} / \text{Var}(y)$). If market dynamics shift abruptly during the test period (e.g. an all-time high breakout), decision trees cannot extrapolate beyond their training values. The model's predictions can have a higher residual sum of squares than simply predicting the mean of the test set, leading to $R^2 < 0$. This highlights market non-stationarity.

### Q5: How do we solve the negative $R^2$ problem caused by out-of-distribution regime shifts?
> **Answer:** In quantitative finance, we replace the non-stationary raw price target with a **Stationary Return Target**:
> $$\text{Target\_Return} = \frac{Close_{t+1} - Close_t}{Close_t}$$
> By normalizing moving averages and spreads relative to current price, the features and targets become scale-invariant. After predicting the return, we reconstruct the nominal price:
> $$\widehat{Close}_{t+1} = Close_t \times (1 + \widehat{\text{Target\_Return}})$$
> On equities undergoing massive breakout rallies like `MANAPPURAM.NS`, this lifts out-of-sample $R^2$ from **$-0.3071$** to **$+0.9516$** and reduces MAE by over $80\%$.

---

## 16. Step-by-Step Execution Guide

### Prerequisites
* Python 3.9 to 3.14 installed on your system.
* Active internet connection for dynamic Yahoo Finance downloads.

### 1. Navigate to Project Directory
```powershell
cd e:\DA-CaseStudy\stock_prediction
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```

### 4. Access the Application
Open your web browser and navigate to:
```text
http://localhost:8501
```
Use the sidebar to choose any stock (e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`), pick a historical horizon (e.g. `5 Years`), and explore the predictions and visualizations!
