"""
data.py - Real Historical Stock Market Data Ingestion
Module for fetching dynamic daily OHLCV data from Yahoo Finance via yfinance.
"""

from typing import Dict, Optional
import pandas as pd
import yfinance as yf

# Predefined list of benchmark Indian NSE stocks for the case study
POPULAR_STOCKS: Dict[str, str] = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Infosys Limited (INFY)": "INFY.NS",
    "HDFC Bank Limited": "HDFCBANK.NS",
    "ICICI Bank Limited": "ICICIBANK.NS",
}

AVAILABLE_PERIODS = ["1y", "2y", "5y"]


def load_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Dynamically downloads daily historical OHLCV data for a given ticker from Yahoo Finance.

    Parameters:
        ticker (str): Ticker symbol (e.g., 'RELIANCE.NS', 'TCS.NS')
        period (str): Historical data horizon ('1y', '2y', '5y')

    Returns:
        pd.DataFrame: Cleaned dataframe containing ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    ticker_clean = ticker.strip().upper()
    
    # Try fetching with user-provided ticker
    df = yf.download(
        tickers=ticker_clean,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # If empty and ticker lacks exchange suffix (e.g. INDOMIM), automatically try .NS or .BO
    if (df is None or df.empty) and ("." not in ticker_clean):
        for fallback_suffix in [".NS", ".BO"]:
            fallback_ticker = f"{ticker_clean}{fallback_suffix}"
            df_fallback = yf.download(
                tickers=fallback_ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if df_fallback is not None and not df_fallback.empty:
                df = df_fallback
                ticker_clean = fallback_ticker
                break

    if df is None or df.empty:
        raise ValueError(
            f"No historical data could be retrieved for ticker '{ticker_clean}'. "
            "For Indian stocks, ensure the exchange suffix is included (e.g., 'INDOMIM.NS' or 'RELIANCE.NS')."
        )

    # Flatten MultiIndex columns created by recent yfinance releases
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to turn Date into an explicit column
    df = df.reset_index()

    # Ensure Date column exists and is timezone-naive datetime
    date_col = next((c for c in df.columns if str(c).lower() == "date"), None)
    if date_col is None:
        raise ValueError("Downloaded dataset does not contain a 'Date' index or column.")

    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # Required OHLCV columns
    expected_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Required OHLCV column '{col}' missing from data feed.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep and arrange standard columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # Sort strictly by Date ascending
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)

    return df
