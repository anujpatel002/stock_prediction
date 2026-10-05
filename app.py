# app.py - Stock Price Prediction Dashboard (Streamlit)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data import POPULAR_STOCKS, load_stock_data
from preprocessing import preprocess_and_engineer_features, FEATURE_COLUMNS
from model import train_and_evaluate, SUPPORTED_ALGORITHMS

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cache data so we don't re-download on every button click ──────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(ticker, period):
    return load_stock_data(ticker, period)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Prediction")
    st.caption("Select a stock and click Run Analysis.")
    st.divider()

    stock_choices = list(POPULAR_STOCKS.keys()) + ["Custom Ticker..."]
    stock_label = st.selectbox("Stock", stock_choices)

    if stock_label == "Custom Ticker...":
        ticker = st.text_input("Enter ticker (e.g. INFY.NS)", "RELIANCE.NS").strip().upper()
    else:
        ticker = POPULAR_STOCKS[stock_label]

    period = st.selectbox("History", {"5 Years": "5y", "2 Years": "2y", "1 Year": "1y"}.keys())
    period_code = {"5 Years": "5y", "2 Years": "2y", "1 Year": "1y"}[period]

    algo_label = st.selectbox("Algorithm", list(SUPPORTED_ALGORITHMS.values()))
    algo_key   = {v: k for k, v in SUPPORTED_ALGORITHMS.items()}[algo_label]

    st.divider()
    run = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Auto-run once on first load
if "ready" not in st.session_state:
    st.session_state.ready = True
if run:
    st.session_state.ready = True


# ── Main Page ─────────────────────────────────────────────────────────────────
st.title("Stock Price Prediction — ML Dashboard")
st.caption("MCA / BCA Academic Case Study | Data: Yahoo Finance | Model: Scikit-Learn")

if not st.session_state.ready:
    st.info("Select a stock in the sidebar and click **Run Analysis**.")
    st.stop()

with st.spinner(f"Downloading data and training {algo_label}..."):
    try:
        raw_df = get_data(ticker, period_code)
        full_df, model_df, latest_row, audit = preprocess_and_engineer_features(raw_df)
        results = train_and_evaluate(model_df, latest_row, mode="stationary", algorithm=algo_key)
    except Exception as err:
        st.error(f"Error: {err}")
        st.info("Check the ticker symbol and your internet connection.")
        st.stop()

pred    = results["prediction"]
metrics = results["metrics"]

# ── 4 Metric Cards ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
arrow = "▲" if pred["expected_change_pct"] >= 0 else "▼"

c1.metric("Current Price",       f"₹{pred['current_close']:,.2f}")
c2.metric("Predicted Next-Day",  f"₹{pred['predicted_next_close']:,.2f}",
          delta=f"{pred['expected_change_pct']:+.2f}%")
c3.metric("Expected Change",
          f"{arrow} {abs(pred['expected_change_pct']):.2f}%",
          delta=f"₹{pred['expected_change_abs']:+,.2f}")
c4.metric("Model R²",            f"{metrics['R2']:.4f}",
          help=f"MAE ₹{metrics['MAE']:.2f} | RMSE ₹{metrics['RMSE']:.2f}")

st.warning(
    "⚠️ Academic project only. Predictions are NOT financial advice. "
    "Stock markets are influenced by news, sentiment, and events no model can foresee."
)

# ── Historical Price Chart ────────────────────────────────────────────────────
st.subheader("Historical Price & Moving Averages")

fig = go.Figure()
fig.add_trace(go.Scatter(x=full_df["Date"], y=full_df["Close"],
                         name="Close",  line=dict(color="#2563EB", width=1.8)))
fig.add_trace(go.Scatter(x=full_df["Date"], y=full_df["MA_5"],
                         name="MA 5",   line=dict(color="#10B981", width=1.4, dash="dot")))
fig.add_trace(go.Scatter(x=full_df["Date"], y=full_df["MA_20"],
                         name="MA 20",  line=dict(color="#F59E0B", width=1.5, dash="dash")))
fig.update_layout(
    title=f"{ticker} — Closing Price with Moving Averages",
    xaxis_title="Date", yaxis_title="Price (₹)",
    template="plotly_white", height=420, hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True)),
    legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
    margin=dict(l=40, r=40, t=55, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ── Actual vs Predicted  +  Feature Importance ───────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Actual vs Predicted (Test Set)")
    tdf = results["test_results_df"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=tdf["Date"], y=tdf["Actual_Price"],
                              name="Actual",    line=dict(color="#0F172A", width=1.8)))
    fig2.add_trace(go.Scatter(x=tdf["Date"], y=tdf["Predicted_Price"],
                              name="Predicted", line=dict(color="#E11D48", width=1.5, dash="dash")))
    fig2.update_layout(
        title=f"Out-of-Sample Test ({results['test_size']} days)",
        xaxis_title="Date", yaxis_title="Price (₹)",
        template="plotly_white", height=340, hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader(f"Feature Importance — {results['algorithm_name']}")
    fidf = results["feature_importance_df"]
    fig3 = px.bar(fidf, x="Importance", y="Feature", orientation="h",
                  text=fidf["Importance"].map(lambda v: f"{v*100:.1f}%"),
                  color="Importance", color_continuous_scale="Blues")
    fig3.update_layout(
        template="plotly_white", height=340,
        xaxis_title="Relative Importance", yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=20, b=40),
    )
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Higher bar = feature contributed more to the model's decisions.")

# ── Prediction Summary  +  Recent Data ───────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Next-Day Prediction Summary")
    with st.container(border=True):
        st.markdown(f"**Stock:** `{ticker}`  |  **Reference date:** {pred['last_date'].strftime('%d %b %Y')}")
        st.table(pd.DataFrame({
            "Metric": ["Current Close", "Predicted Next Close", "Change", "Signal"],
            "Value":  [
                f"₹{pred['current_close']:,.2f}",
                f"₹{pred['predicted_next_close']:,.2f}",
                f"{arrow} {abs(pred['expected_change_pct']):.2f}%  (₹{pred['expected_change_abs']:+,.2f})",
                "Bullish 📈" if pred["expected_change_pct"] >= 0 else "Bearish 📉",
            ],
        }).set_index("Metric"))

with col_b:
    st.subheader("Recent Market Data (Last 15 Days)")
    recent = full_df.tail(15)[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        recent.sort_values("Date", ascending=False).style.format({
            "Open": "₹{:,.2f}", "High": "₹{:,.2f}",
            "Low":  "₹{:,.2f}", "Close": "₹{:,.2f}",
            "Volume": "{:,.0f}",
        }),
        use_container_width=True, height=340,
    )

# ── EDA Section ───────────────────────────────────────────────────────────────
with st.expander("📊 Exploratory Data Analysis (EDA) — Click to expand"):
    e1, e2 = st.columns(2)

    with e1:
        st.markdown("**Chart 1: Closing Price Trend**")
        st.plotly_chart(
            px.line(full_df, x="Date", y="Close",
                    title=f"{ticker} — Price History",
                    labels={"Close": "Price (₹)"},
                    template="plotly_white", height=300),
            use_container_width=True,
        )

    with e2:
        st.markdown("**Chart 2: Daily Trading Volume**")
        st.plotly_chart(
            px.bar(full_df, x="Date", y="Volume",
                   title=f"{ticker} — Volume",
                   labels={"Volume": "Shares Traded"},
                   template="plotly_white", height=300,
                   color_discrete_sequence=["#94A3B8"]),
            use_container_width=True,
        )

    e3, e4 = st.columns(2)

    with e3:
        st.markdown("**Chart 3: MA_5 vs MA_20 Crossover**")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=full_df["Date"], y=full_df["MA_5"],
                                    name="MA 5 (fast)",  line=dict(color="#10B981")))
        fig_ma.add_trace(go.Scatter(x=full_df["Date"], y=full_df["MA_20"],
                                    name="MA 20 (slow)", line=dict(color="#F59E0B")))
        fig_ma.update_layout(title="Moving Average Crossover",
                             template="plotly_white", height=300,
                             xaxis_title="Date", yaxis_title="Price (₹)",
                             margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_ma, use_container_width=True)

    with e4:
        st.markdown("**Chart 4: Feature Correlation Heatmap**")
        corr_cols = [c for c in FEATURE_COLUMNS if c in full_df.columns] + ["Close"]
        fig_corr = px.imshow(
            full_df[corr_cols].corr(),
            text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Matrix",
            height=300,
        )
        fig_corr.update_layout(template="plotly_white", margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_corr, use_container_width=True)

# ── Model Info ────────────────────────────────────────────────────────────────
st.subheader("Model & Pipeline Summary")
i1, i2, i3 = st.columns(3)

i1.markdown(f"""
**Algorithm**
- Model: `{results['algorithm_name']}`
- Strategy: `Stationary Returns`
- Split: 80% train / 20% test
- Random seed: 42
""")

i2.markdown(f"""
**Dataset**
- Total usable rows: {results['train_size'] + results['test_size']}
- Training rows: {results['train_size']}
- Test rows: {results['test_size']}
- Split type: Chronological (no shuffle)
""")

i3.markdown(f"""
**Evaluation (on test set)**
- MAE:  ₹{metrics['MAE']:.2f}
- RMSE: ₹{metrics['RMSE']:.2f}
- R²:   {metrics['R2']:.4f}
- Warm-up rows dropped: {audit['warmup_rows_dropped']}
""")

st.divider()
st.caption("Stock Price Prediction — MCA Academic Case Study | Python · Streamlit · Scikit-Learn · Yahoo Finance")
