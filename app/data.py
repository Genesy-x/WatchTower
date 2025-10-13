import requests
import pandas as pd

COINDESK_API_KEY = "74e1e197fe44b98d6c1cfd466095fa9fa4c2a57edea008936b2ac1d5ad5167d1"  # Replace if needed
COINDESK_BASE = "https://data-api.coindesk.com/spot/v1/historical/hours"

def fetch_historical_ohlc_btc(start: str = None, end: str = None, limit: int = 10, aggregate: int = 1):
    """Fetch historical OHLCV data for BTC-USDT."""
    params = {
        "market": "binance",
        "instrument": "BTC-USDT",
        "limit": limit,
        "aggregate": aggregate,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end

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
        print(f"[ERROR] Fetching CoinDesk OHLC for BTC-USDT: {e}")
        return pd.DataFrame()

def fetch_historical_ohlc_eth(start: str = None, end: str = None, limit: int = 10, aggregate: int = 1):
    """Fetch historical OHLCV data for ETH-USDT."""
    params = {
        "market": "binance",
        "instrument": "ETH-USDT",
        "limit": limit,
        "aggregate": aggregate,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end

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
        print(f"[ERROR] Fetching CoinDesk OHLC for ETH-USDT: {e}")
        return pd.DataFrame()

def fetch_historical_ohlc_sol(start: str = None, end: str = None, limit: int = 10, aggregate: int = 1):
    """Fetch historical OHLCV data for SOL-USDT."""
    params = {
        "market": "binance",
        "instrument": "SOL-USDT",
        "limit": limit,
        "aggregate": aggregate,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end

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
        print(f"[ERROR] Fetching CoinDesk OHLC for SOL-USDT: {e}")
        return pd.DataFrame()

def fetch_market_data(binance_symbol: str, timeframe: str = "1d", limit: int = 10):
    """
    Fetch OHLCV for a single asset based on symbol, using specific fetch function.
    No coingecko_id needed since we use CoinDesk only.
    """
    instrument_map = {
        "BTCUSDT": fetch_historical_ohlc_btc,
        "ETHUSDT": fetch_historical_ohlc_eth,
        "SOLUSDT": fetch_historical_ohlc_sol
    }
    fetch_func = instrument_map.get(binance_symbol)
    if not fetch_func:
        print(f"[ERROR] No fetch function for {binance_symbol}")
        return {"ohlcv": pd.DataFrame(), "fundamentals": {}}
    
    ohlcv_df = fetch_func(limit=limit, aggregate=24 if timeframe == "1d" else 1)
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": {}  # Empty
    }