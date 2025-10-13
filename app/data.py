import pandas as pd
from datetime import datetime
import requests

COINDESK_API_KEY = "74e1e197fe44b98d6c1cfd466095fa9fa4c2a57edea008936b2ac1d5ad5167d1"  # Replace with your own API key if needed
COINDESK_BASE = "https://data-api.coindesk.com/spot/v1/historical/hours"

def fetch_ohlcv(market: str = "binance", instrument: str = "BTC-USDT", start: str = None, end: str = None, timeframe: str = "1d", limit: int = 500, aggregate: int = 1):
    """
    Fetch historical OHLCV data from CoinDesk API.
    - market: Exchange like "binance"
    - instrument: Pair like "BTC-USDT"
    - timeframe: Not directly supported; aggregate param simulates (e.g., aggregate=24 for 1d from hours)
    - limit: Number of candles (adjust based on aggregate)
    """
    try:
        params = {
            "market": market,
            "instrument": instrument,
            "limit": limit,
            "aggregate": aggregate,  # E.g., 24 for daily from hourly data
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON",
            "api_key": COINDESK_API_KEY
        }
        if start:
            params["start"] = start  # YYYY-MM-DD
        if end:
            params["end"] = end

        response = requests.get(COINDESK_BASE, params=params)
        response.raise_for_status()
        data = response.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume_base", "volume_quote"]]  # Adjust to match original OHLCV
        df.rename(columns={"volume_base": "volume"}, inplace=True)  # Simplify to match original
        return df
    except Exception as e:
        print(f"[ERROR] Fetching CoinDesk OHLC for {instrument}: {e}")
        return pd.DataFrame()  # Empty on error

def fetch_market_data(binance_symbol: str, coingecko_id: str, timeframe: str = "1d", limit: int = 500):
    """
    Return OHLCV from CoinDesk (adapted for historical use; fundamentals empty as not used).
    Note: binance_symbol used for instrument (e.g., "BTCUSDT" -> "BTC-USDT")
    """
    instrument = binance_symbol.replace("USDT", "-USDT")  # Adapt to CoinDesk format
    ohlcv_df = fetch_ohlcv(market="binance", instrument=instrument, timeframe=timeframe, limit=limit)
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": {}  # Empty dict, since not used in logic
    }