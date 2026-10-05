"""
app.py - Stock Intelligence: ML Price Prediction Dashboard
Interactive Streamlit application providing real-time data ingestion,
exploratory data analysis, Random Forest regression, and next-day price estimation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data import POPULAR_STOCKS, load_stock_data
from preprocessing import preprocess_and_engineer_features, FEATURE_COLUMNS
from model import train_and_evaluate

# Configure Streamlit Page
st.set_page_config(
    page_title="Stock Intelligence — ML Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, academic presentation
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.85rem;
    }
    .change-positive {
        color: #16A34A;
        font-weight: 600;
    }
    .change-negative {
        color: #DC2626;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400E;
        margin: 15px 0 25px 0;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E293B;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #F1F5F9;
        padding-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_stock_data(ticker: str, period: str):
    """Caches historical data fetch to reduce redundant network overhead."""
    return load_stock_data(ticker=ticker, period=period)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("## 📈 Stock Prediction")
    st.markdown("Select a stock and historical duration to train the ML model.")
    st.markdown("---")

    stock_options = list(POPULAR_STOCKS.keys()) + ["Custom Ticker..."]
    selected_stock_label = st.selectbox(
        "Select Stock",
        options=stock_options,
        index=0,
        help="Select a benchmark Indian NSE stock or enter a custom ticker.",
    )

    if selected_stock_label == "Custom Ticker...":
        selected_ticker = st.text_input(
            "Enter Ticker (e.g. INFY.NS, AAPL)",
            value="RELIANCE.NS",
            help="Yahoo Finance ticker symbol",
        ).strip().upper()
    else:
        selected_ticker = POPULAR_STOCKS[selected_stock_label]

    period_options = {
        "5 Years": "5y",
        "2 Years": "2y",
        "1 Year": "1y",
    }
    selected_period_label = st.selectbox(
        "Historical Period",
        options=list(period_options.keys()),
        index=0,
        help="Time window for historical observations.",
    )
    selected_period = period_options[selected_period_label]

    algo_options = {
        "Random Forest Regressor (Ensemble Bagging)": "random_forest",
        "Linear Regression (OLS Baseline)": "linear_regression",
        "K-Nearest Neighbors (KNN Regressor)": "knn",
        "Decision Tree Regressor (Single Tree)": "decision_tree",
    }
    selected_algo_label = st.selectbox(
        "ML Algorithm",
        options=list(algo_options.keys()),
        index=0,
        help="Select the machine learning algorithm to train and evaluate.",
    )
    selected_algorithm = algo_options[selected_algo_label]

    strategy_options = {
        "Stationary Returns (Robust - R² > 0.95)": "stationary",
        "Nominal Price Level (Academic Baseline)": "nominal",
    }
    selected_strategy_label = st.selectbox(
        "Modeling Strategy",
        options=list(strategy_options.keys()),
        index=0,
        help="Stationary Returns predicts scale-invariant returns, solving regime shifts and out-of-distribution trends.",
    )
    selected_mode = strategy_options[selected_strategy_label]

    st.markdown("---")
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #64748B;">
            <b>Project:</b> MCA ML Case Study<br>
            <b>Model:</b> Multi-Algorithm ML Suite<br>
            <b>Data Source:</b> Yahoo Finance API<br>
            <b>Target:</b> Next-Day Closing Price
        </div>
        """,
        unsafe_allow_html=True,
    )

# Manage state for seamless interactivity
if "has_run" not in st.session_state:
    st.session_state.has_run = True  # Auto-run once on launch for great first impression
if run_button:
    st.session_state.has_run = True


# ==========================================
# MAIN DASHBOARD CONTENT
# ==========================================
st.markdown('<div class="main-title">Stock Intelligence — ML Price Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Machine Learning Based Price Analysis & Next-Day Forecasting Dashboard</div>',
    unsafe_allow_html=True,
)

if st.session_state.has_run:
    with st.spinner(f"Ingesting market data for {selected_ticker} and training {selected_algo_label.split('(')[0].strip()}..."):
        try:
            # 1. Ingest Data
            raw_df = get_cached_stock_data(selected_ticker, selected_period)

            # 2. Preprocess & Feature Engineer
            full_df, model_df, latest_features, audit_stats = preprocess_and_engineer_features(raw_df)

            # 3. Train Model and Generate Predictions
            results = train_and_evaluate(
                model_df,
                latest_features,
                mode=selected_mode,
                algorithm=selected_algorithm,
            )

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Tip: Verify the ticker symbol (e.g., RELIANCE.NS) and ensure your internet connection is active.")
            st.stop()

    pred_info = results["prediction"]
    metrics = results["metrics"]
    test_results_df = results["test_results_df"]
    feature_importance_df = results["feature_importance_df"]

    current_price = pred_info["current_close"]
    predicted_price = pred_info["predicted_next_close"]
    expected_change = pred_info["expected_change_pct"]
    r2_score_val = metrics["R2"]

    # Recent IPO / Limited History Notice
    if audit_stats["initial_rows"] < 60:
        st.info(
            f"ℹ️ **Recent Listing (IPO) Notice:** `{selected_ticker}` has only **{audit_stats['initial_rows']} trading days** "
            "of total recorded history. Machine learning models require sufficient historical market cycles; "
            "predictions for newly listed equities reflect initial price discovery and should be interpreted with caution."
        )

    # ==========================================
    # 4 METRIC CARDS (Native Streamlit)
    # ==========================================
    col1, col2, col3, col4 = st.columns(4)

    change_arrow = "▲" if expected_change >= 0 else "▼"

    with col1:
        st.metric(
            label="Current Price",
            value=f"₹{current_price:,.2f}",
            help=f"Latest Close ({pred_info['last_date'].strftime('%d %b %Y')})",
        )

    with col2:
        st.metric(
            label="Predicted Next-Day",
            value=f"₹{predicted_price:,.2f}",
            delta=f"{expected_change:+.2f}%",
            help=f"{results['algorithm_name']} next-day closing price estimate",
        )

    with col3:
        st.metric(
            label="Expected Change",
            value=f"{change_arrow} {abs(expected_change):.2f}%",
            delta=f"₹{pred_info['expected_change_abs']:+,.2f}",
            delta_color="normal",
            help="Difference between predicted next close and current close",
        )

    with col4:
        st.metric(
            label="Model R² Score",
            value=f"{r2_score_val:.4f}",
            help=f"MAE: ₹{metrics['MAE']:.2f} | RMSE: ₹{metrics['RMSE']:.2f}",
        )

    # Educational Disclaimer
    st.warning(
        "⚠️ **Academic Disclaimer:** This application is an educational predictive modeling demonstration. "
        "Financial markets are non-stationary and influenced by macroeconomic factors, unforeseen news, and investor sentiment. "
        "**Model-estimated next-day closing prices** do not constitute financial advice or guaranteed investment outcomes."
    )

    # ==========================================
    # HISTORICAL PRICE & MOVING AVERAGES CHART
    # ==========================================
    st.subheader("Historical Price & Moving Averages")

    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=full_df["Date"],
            y=full_df["Close"],
            mode="lines",
            name="Closing Price",
            line=dict(color="#2563EB", width=1.8),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Close:</b> ₹%{y:,.2f}<extra></extra>",
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=full_df["Date"],
            y=full_df["MA_5"],
            mode="lines",
            name="5-Day Moving Average (MA_5)",
            line=dict(color="#10B981", width=1.4, dash="dot"),
            hovertemplate="<b>MA_5:</b> ₹%{y:,.2f}<extra></extra>",
        )
    )
    fig_price.add_trace(
        go.Scatter(
            x=full_df["Date"],
            y=full_df["MA_20"],
            mode="lines",
            name="20-Day Moving Average (MA_20)",
            line=dict(color="#F59E0B", width=1.5, dash="dash"),
            hovertemplate="<b>MA_20:</b> ₹%{y:,.2f}<extra></extra>",
        )
    )
    fig_price.update_layout(
        title=f"{selected_ticker} — Historical Closing Price with Technical Moving Averages",
        xaxis_title="Trading Date",
        yaxis_title="Price (₹ INR)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ==========================================
    # PREDICTION & MODEL PERFORMANCE (2 COLUMNS)
    # ==========================================
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Next Trading Day Prediction")
        
        with st.container(border=True):
            st.markdown(f"**Target Stock:** `{selected_ticker}`")
            st.caption(f"**Reference Date:** {pred_info['last_date'].strftime('%A, %d %B %Y')}")
            
            # Real Native Streamlit Table
            prediction_table = pd.DataFrame(
                {
                    "Metric": [
                        "Current Close",
                        "Predicted Next Close",
                        "Predicted Change",
                        "Prediction Category",
                    ],
                    "Value": [
                        f"₹{current_price:,.2f}",
                        f"₹{predicted_price:,.2f}",
                        f"{change_arrow} {abs(expected_change):.2f}% (₹{pred_info['expected_change_abs']:+,.2f})",
                        "Bullish Momentum" if expected_change >= 0 else "Bearish Pressure",
                    ],
                }
            )
            st.table(prediction_table.set_index("Metric"))
            
            st.caption("ℹ️ *The prediction is generated using 7 lag, momentum, volatility, and volume indicators.*")

    with col_right:
        st.subheader("Model Performance — Actual vs Predicted")
        
        fig_test = go.Figure()
        fig_test.add_trace(
            go.Scatter(
                x=test_results_df["Date"],
                y=test_results_df["Actual_Price"],
                mode="lines",
                name="Actual Test Price",
                line=dict(color="#0F172A", width=1.8),
                hovertemplate="Actual: ₹%{y:,.2f}<extra></extra>",
            )
        )
        fig_test.add_trace(
            go.Scatter(
                x=test_results_df["Date"],
                y=test_results_df["Predicted_Price"],
                mode="lines",
                name="Predicted Price",
                line=dict(color="#E11D48", width=1.5, dash="dash"),
                hovertemplate="Predicted: ₹%{y:,.2f}<extra></extra>",
            )
        )
        fig_test.update_layout(
            title=f"Out-of-Sample Test Evaluation ({results['test_size']} Trading Days)",
            xaxis_title="Trading Date",
            yaxis_title="Price (₹ INR)",
            template="plotly_white",
            height=300,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=50, b=40),
        )
        st.plotly_chart(fig_test, use_container_width=True)

    # ==========================================
    # FEATURE IMPORTANCE & RECENT DATA (2 COLUMNS)
    # ==========================================
    col_feat, col_data = st.columns([1, 1])

    with col_feat:
        st.subheader(f"Feature Importance ({results['algorithm_name']})")
        
        fig_feat = px.bar(
            feature_importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text=feature_importance_df["Importance"].apply(lambda v: f"{v*100:.1f}%"),
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_feat.update_layout(
            template="plotly_white",
            height=350,
            xaxis_title="Relative Gini Importance",
            yaxis_title="Engineered Feature",
            coloraxis_showscale=False,
            margin=dict(l=40, r=40, t=20, b=40),
        )
        fig_feat.update_traces(textposition="outside")
        st.plotly_chart(fig_feat, use_container_width=True)
        
        st.caption(
            "📌 **Academic Note:** Feature importance measures the variance reduction contribution "
            "of each feature across all decision trees. It represents *predictive contribution* within "
            "the trained model and does not establish economic or causal relationship."
        )

    with col_data:
        st.subheader("Recent Market Data (Latest Records)")
        
        recent_display_df = full_df.tail(15)[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        recent_display_df["Date"] = recent_display_df["Date"].dt.strftime("%Y-%m-%d")
        
        # Format columns for presentation
        st.dataframe(
            recent_display_df.sort_values("Date", ascending=False).style.format(
                {
                    "Open": "₹{:,.2f}",
                    "High": "₹{:,.2f}",
                    "Low": "₹{:,.2f}",
                    "Close": "₹{:,.2f}",
                    "Volume": "{:,.0f}",
                }
            ),
            use_container_width=True,
            height=340,
        )

    # ==========================================
    # EXPANDABLE EDA SECTION
    # ==========================================
    with st.expander("📊 Exploratory Data Analysis (EDA) — Click to Expand"):
        st.markdown("### Exploratory Data Analysis of Historical Series")
        st.markdown(
            "EDA provides foundational insights into trend, volatility, liquidity, "
            "and cross-correlations before feeding features into the regression pipeline."
        )

        eda_c1, eda_c2 = st.columns(2)

        with eda_c1:
            st.markdown("#### Chart 1: Historical Closing Price Trend")
            fig_eda1 = px.line(
                full_df,
                x="Date",
                y="Close",
                title=f"{selected_ticker} — Closing Price Trajectory",
                labels={"Close": "Closing Price (₹)", "Date": "Date"},
            )
            fig_eda1.update_traces(line_color="#2563EB")
            fig_eda1.update_layout(template="plotly_white", height=320, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_eda1, use_container_width=True)

        with eda_c2:
            st.markdown("#### Chart 2: Daily Trading Volume Distribution")
            fig_eda2 = px.bar(
                full_df,
                x="Date",
                y="Volume",
                title=f"{selected_ticker} — Daily Market Trading Volume",
                labels={"Volume": "Volume (Shares)", "Date": "Date"},
            )
            fig_eda2.update_traces(marker_color="#94A3B8")
            fig_eda2.update_layout(template="plotly_white", height=320, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_eda2, use_container_width=True)

        eda_c3, eda_c4 = st.columns(2)

        with eda_c3:
            st.markdown("#### Chart 3: Short vs Medium Moving Averages")
            fig_eda3 = go.Figure()
            fig_eda3.add_trace(go.Scatter(x=full_df["Date"], y=full_df["MA_5"], name="MA_5 (Fast)", line=dict(color="#10B981")))
            fig_eda3.add_trace(go.Scatter(x=full_df["Date"], y=full_df["MA_20"], name="MA_20 (Slow)", line=dict(color="#F59E0B")))
            fig_eda3.update_layout(
                title="MA_5 vs MA_20 Trend Crossover Analysis",
                template="plotly_white",
                height=320,
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                margin=dict(l=30, r=30, t=40, b=30),
            )
            st.plotly_chart(fig_eda3, use_container_width=True)

        with eda_c4:
            st.markdown("#### Chart 4: Feature Correlation Heatmap")
            corr_cols = [c for c in FEATURE_COLUMNS if c in full_df.columns] + ["Close"]
            corr_matrix = full_df[corr_cols].corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Pearson Correlation Matrix (Features vs Close)",
            )
            fig_corr.update_layout(template="plotly_white", height=320, margin=dict(l=30, r=30, t=40, b=30))
            st.plotly_chart(fig_corr, use_container_width=True)

    # ==========================================
    # MODEL SPECIFICATIONS & AUDIT SUMMARY
    # ==========================================
    st.subheader("Model Information & Pipeline Architecture")

    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(
            f"""
            **Algorithm Configuration**
            - **Model:** `{results['algorithm_name']}`
            - **Strategy:** `{'Stationary Returns' if results['mode'] == 'stationary' else 'Nominal Price Level'}`
            - **Split Ratio:** 80% Train / 20% Test
            - **Random State:** 42 (Reproducible)
            - **Criterion:** Squared Error Loss
            """
        )

    with m_col2:
        st.markdown(
            f"""
            **Dataset & Split Strategy**
            - **Split Type:** Chronological (Time-Series)
            - **Training Ratio:** 80% ({results['train_size']} records)
            - **Testing Ratio:** 20% ({results['test_size']} records)
            - **Lookahead Leakage:** Strictly Prevented
            - **Engineered Features:** 7 Lag/Momentum features
            """
        )

    with m_col3:
        st.markdown(
            f"""
            **Calculated Evaluation Metrics**
            - **Mean Absolute Error (MAE):** ₹{metrics['MAE']:.2f}
            - **Root Mean Squared Error (RMSE):** ₹{metrics['RMSE']:.2f}
            - **Coefficient of Determination ($R^2$):** {metrics['R2']:.4f}
            - **Audit:** Dropped {audit_stats['warmup_rows_dropped']} rolling warm-up records
            """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 10px 0;">
            <b>Stock Price Prediction — 10-Mark MCA Academic Case Study Mini-App</b> |
            Developed with Python, Streamlit, Scikit-learn, and Yahoo Finance Data.
        </div>
        """,
        unsafe_allow_html=True,
    )
