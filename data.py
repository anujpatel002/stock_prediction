# data.py - Download stock price data from Yahoo Finance

import pandas as pd
import yfinance as yf

# Popular Indian NSE stocks shown in the dropdown
POPULAR_STOCKS = {
    "Reliance Industries":           "RELIANCE.NS",
    "Tata Consultancy Services":     "TCS.NS",
    "Infosys":                       "INFY.NS",
    "HDFC Bank":                     "HDFCBANK.NS",
    "ICICI Bank":                    "ICICIBANK.NS",
}


def load_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download daily OHLCV data for a stock from Yahoo Finance.

    ticker : stock symbol, e.g. 'RELIANCE.NS'
    period : how far back to go — '1y', '2y', or '5y'
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    ticker = ticker.strip().upper()

    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)

    # If nothing came back, try adding .NS (Indian exchange) automatically
    if df.empty and "." not in ticker:
        df = yf.download(ticker + ".NS", period=period, interval="1d",
                         auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(
            f"No data found for '{ticker}'. "
            "Check the ticker symbol (e.g. RELIANCE.NS) and your internet connection."
        )

    # Newer yfinance returns multi-level columns like (Close, RELIANCE.NS) — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Make sure the date column is named 'Date' and has no timezone info
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # Keep only the 6 standard columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # Convert all price/volume columns to numbers (in case any are strings)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("Date").reset_index(drop=True)
