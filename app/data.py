import requests
import pandas as pd

COINDESK_API_KEY = "74e1e197fe44b98d6c1cfd466095fa9fa4c2a57edea008936b2ac1d5ad5167d1"  # Replace if needed
COINDESK_BASE = "https://data-api.coindesk.com/spot/v1/historical/hours"

def fetch_historical_ohlc_btc(start: str = "2024-01-01", end: str = None, limit: int = 500, aggregate: int = 24):
    """Fetch 1d historical OHLCV data for BTCUSDT on Binance."""
    params = {
        "market": "binance",
        "instrument": "BTCUSDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": aggregate,  # 24 for 1d from hourly
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }

    try:
        response = requests.get(COINDESK_BASE, params=params)
        response.raise_for_status()
        json_data = response.json()
        data_list = json_data.get("Data", [])
        df = pd.DataFrame(data_list)
        df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
        df.set_index("timestamp", inplace=True)
        df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        return df
    except Exception as e:
        print(f"[ERROR] Fetching CoinDesk OHLC for BTCUSDT: {e}")
        return pd.DataFrame()

def fetch_historical_ohlc_eth(start: str = "2024-01-01", end: str = None, limit: int = 500, aggregate: int = 24):
    """Fetch 1d historical OHLCV data for ETHUSDT on Binance."""
    params = {
        "market": "binance",
        "instrument": "ETHUSDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": aggregate,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }

    try:
        response = requests.get(COINDESK_BASE, params=params)
        response.raise_for_status()
        json_data = response.json()
        data_list = json_data.get("Data", [])
        df = pd.DataFrame(data_list)
        df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
        df.set_index("timestamp", inplace=True)
        df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        return df
    except Exception as e:
        print(f"[ERROR] Fetching CoinDesk OHLC for ETHUSDT: {e}")
        return pd.DataFrame()

def fetch_historical_ohlc_sol(start: str = "2024-01-01", end: str = None, limit: int = 500, aggregate: int = 24):
    """Fetch 1d historical OHLCV data for SOLUSDT on Binance."""
    params = {
        "market": "binance",
        "instrument": "SOLUSDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": aggregate,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }

    try:
        response = requests.get(COINDESK_BASE, params=params)
        response.raise_for_status()
        json_data = response.json()
        data_list = json_data.get("Data", [])
        df = pd.DataFrame(data_list)
        df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
        df.set_index("timestamp", inplace=True)
        df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        return df
    except Exception as e:
        print(f"[ERROR] Fetching CoinDesk OHLC for SOLUSDT: {e}")
        return pd.DataFrame()

def fetch_historical_ohlc_xaut(start: str = "2024-01-01", end: str = None, limit: int = 500, aggregate: int = 24):
    """Fetch 1d historical OHLCV data for XAUTUSDT on Kraken."""
    params = {
        "market": "kraken",
        "instrument": "XAUTUSDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": aggregate,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }

    try:
        response = requests.get(COINDESK_BASE, params=params)
        response.raise_for_status()
        json_data = response.json()
        data_list = json_data.get("Data", [])
        df = pd.DataFrame(data_list)
        df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
        df.set_index("timestamp", inplace=True)
        df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        df.columns = ["open", "high", "low", "close", "volume"]
        return df
    except Exception as e:
        print(f"[ERROR] Fetching CoinDesk OHLC for XAUTUSDT: {e}")
        return pd.DataFrame()

def fetch_market_data(binance_symbol: str, timeframe: str = "1d", limit: int = 500):
    """
    Fetch OHLCV for a single asset based on symbol, using specific fetch function.
    """
    instrument_map = {
        "BTCUSDT": fetch_historical_ohlc_btc,
        "ETHUSDT": fetch_historical_ohlc_eth,
        "SOLUSDT": fetch_historical_ohlc_sol,
        "PAXGUSDT": fetch_historical_ohlc_xaut  # GOLD mapping
    }
    fetch_func = instrument_map.get(binance_symbol)
    if not fetch_func:
        print(f"[ERROR] No fetch function for {binance_symbol}")
        return {"ohlcv": pd.DataFrame(), "fundamentals": {}}
    
    ohlcv_df = fetch_func(limit=limit)
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": {}  # Empty
    }