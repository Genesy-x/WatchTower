import ccxt
import pandas as pd
from datetime import datetime

binance = ccxt.binance()

def fetch_ohlcv(symbol: str, timeframe: str = "1d", limit: int = 500):
    """
    Fetch OHLCV (price candles) for a given symbol from Binance.
    """
    try:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"[ERROR] Fetching OHLCV for {symbol}: {e}")
        return pd.DataFrame()  # Empty on error

def fetch_market_data(binance_symbol: str, coingecko_id: str, timeframe: str = "1d", limit: int = 500):
    """
    Return OHLCV only (CoinGecko removed to avoid rate limits, as not used).
    """
    ohlcv_df = fetch_ohlcv(binance_symbol, timeframe, limit)
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": {}  # Empty dict, since not used in logic
    }