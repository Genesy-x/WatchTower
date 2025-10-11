import ccxt
import requests
import pandas as pd
from datetime import datetime

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
        return pd.DataFrame()  # Return empty DF on error

def fetch_market_cap(symbol_id: str = "bitcoin"):
    """
    Fetch market cap and volume data from CoinGecko using the coin ID.
    """
    try:
        url = f"{COINGECKO_API}/coins/markets"
        params = {"vs_currency": "usd", "ids": symbol_id}
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise on bad status
        data = response.json()
        if not data:
            raise ValueError("Empty response")
        item = data[0]
        return {
            "id": item.get("id", symbol_id),
            "symbol": item.get("symbol", ""),
            "name": item.get("name", ""),
            "market_cap": item.get("market_cap", 0),
            "volume_24h": item.get("total_volume", 0),
            "price": item.get("current_price", 0),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] Fetching market cap for {symbol_id}: {e}")
        return {
            "id": symbol_id,
            "symbol": "",
            "name": "",
            "market_cap": 0,
            "volume_24h": 0,
            "price": 0,
            "timestamp": datetime.utcnow().isoformat()
        }  # Return defaults on error

def fetch_market_data(binance_symbol: str, coingecko_id: str, timeframe: str = "1d", limit: int = 500):
    """
    Combine OHLCV + market cap/volume in one structure.
    """
    ohlcv_df = fetch_ohlcv(binance_symbol, timeframe, limit)
    cap_data = fetch_market_cap(coingecko_id)
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": cap_data
    }