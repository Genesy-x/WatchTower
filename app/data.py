import ccxt
import requests
import pandas as pd
from datetime import datetime
import time  # For delays

binance = ccxt.binance()

COINGECKO_API = "https://api.coingecko.com/api/v3"

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

def fetch_market_caps(coingecko_ids: list):
    """
    Batch fetch market cap for multiple IDs in one call.
    """
    try:
        url = f"{COINGECKO_API}/coins/markets"
        params = {"vs_currency": "usd", "ids": ','.join(coingecko_ids)}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        caps = {item["id"]: {
            "id": item.get("id", ""),
            "symbol": item.get("symbol", ""),
            "name": item.get("name", ""),
            "market_cap": item.get("market_cap", 0),
            "volume_24h": item.get("total_volume", 0),
            "price": item.get("current_price", 0),
            "timestamp": datetime.utcnow().isoformat()
        } for item in data}
        return caps
    except Exception as e:
        print(f"[ERROR] Batch fetching market caps: {e}")
        return {id: {"market_cap": 0} for id in coingecko_ids}  # Defaults

def fetch_market_data(binance_symbol: str, coingecko_id: str, timeframe: str = "1d", limit: int = 500):
    """
    Combine OHLCV + market cap/volume in one structure.
    """
    ohlcv_df = fetch_ohlcv(binance_symbol, timeframe, limit)
    time.sleep(1)  # Delay to avoid rate limits
    cap_data = fetch_market_caps([coingecko_id]).get(coingecko_id, {"market_cap": 0})
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": cap_data
    }