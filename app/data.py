import ccxt
import requests
import pandas as pd
from datetime import datetime

binance = ccxt.binance()

COINGECKO_API = "https://api.coingecko.com/api/v3"

def fetch_ohlcv(symbol: str, timeframe: str = "1d", limit: int = 500):
    """
    Fetch OHLCV (price candles) for a given symbol from Binance.
    Default timeframe is 1d, limit = 500.
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
        return None

def fetch_market_cap(symbol_id: str = "bitcoin"):
    """
    Fetch market cap and volume data from CoinGecko using the coin ID.
    """
    try:
        url = f"{COINGECKO_API}/coins/markets"
        params = {"vs_currency": "usd", "ids": symbol_id}
        response = requests.get(url, params=params)
        data = response.json()[0]

        return {
            "id": data["id"],
            "symbol": data["symbol"],
            "name": data["name"],
            "market_cap": data["market_cap"],
            "volume_24h": data["total_volume"],
            "price": data["current_price"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"[ERROR] Fetching market cap for {symbol_id}: {e}")
        return None

def fetch_market_data(binance_symbol: str, coingecko_id: str, timeframe: str = "1d", limit: int = 500):
    """
    Combine OHLCV + market cap/volume in one structure.
    """
    ohlcv_df = fetch_ohlcv(binance_symbol, timeframe=timeframe, limit=limit)
    cap_data = fetch_market_cap(coingecko_id)

    return {
        "ohlcv": ohlcv_df,
        "fundamentals": cap_data
    }