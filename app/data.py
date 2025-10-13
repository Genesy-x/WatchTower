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

import requests
import pandas as pd

COINDESK_API_KEY = "74e1e197fe44b98d6c1cfd466095fa9fa4c2a57edea008936b2ac1d5ad5167d1"  
COINDESK_BASE = "https://data-api.coindesk.com/spot/v1/historical/hours"

def fetch_historical_ohlc(market: str = "binance", instrument: str = "BTC-USDT", start: str = None, end: str = None, limit: int = 500, aggregate: int = 1):
    params = {
        "market": market,
        "instrument": instrument,
        "limit": limit,
        "aggregate": aggregate,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "api_key": COINDESK_API_KEY
    }
    if start:
        params["start"] = start  # YYYY-MM-DD
    if end:
        params["end"] = end
    try:
        response = requests.get(COINDESK_BASE, params=params)
        response.raise_for_status()
        data = response.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume_base", "volume_quote"]]  # Adjust columns
        return df
    except Exception as e:
        print(f"[ERROR] Fetching CoinDesk OHLC for {instrument}: {e}")
        return pd.DataFrame()

# Use in fetch_market_data or directly in main.py for backtests