import requests
import pandas as pd
import time
from datetime import datetime
from app.db.database import SessionLocal, OHLCVData

COINDESK_API_KEY = "74e1e197fe44b98d6c1cfd466095fa9fa4c2a57edea008936b2ac1d5ad5167d1"
COINDESK_BASE = "https://data-api.coindesk.com/spot/v1/historical/days"

def get_latest_timestamp(instrument: str):
    """Get the latest timestamp for an instrument from Neon."""
    db = SessionLocal()
    try:
        latest = db.query(OHLCVData.timestamp).filter(OHLCVData.instrument == instrument).order_by(OHLCVData.timestamp.desc()).first()
        return latest[0] if latest else pd.Timestamp("2024-01-01")
    finally:
        db.close()

def fetch_historical_ohlc_btc(start: str = None, end: str = None, limit: int = 1):
    """Fetch daily historical OHLCV data for BTC-USDT on Binance."""
    if start is None:
        start = "2024-01-01"  # Default start
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    params = {
        "market": "binance",
        "instrument": "BTC-USDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": 1,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    headers = {"Content-type": "application/json; charset=UTF-8"}

    for attempt in range(3):
        try:
            response = requests.get(COINDESK_BASE, params=params, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            data_list = json_data.get("Data", [])
            if not data_list:
                print(f"[WARNING] Empty data for BTC-USDT, attempt {attempt+1}")
                time.sleep(2)
                continue
            df = pd.DataFrame(data_list)
            df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df.set_index("timestamp", inplace=True)
            # Filter by start date
            df = df[df.index >= pd.Timestamp(start)]
            df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            print(f"[SUCCESS] Fetched {len(df)} rows for BTC-USDT from {df.index.min().date()} to {df.index.max().date()}")
            return df
        except Exception as e:
            print(f"[ERROR] Fetching CoinDesk OHLC for BTC-USDT (attempt {attempt+1}): {e}")
            time.sleep(2)
    print("[ERROR] Failed to fetch BTC-USDT after retries")
    return pd.DataFrame()

def fetch_historical_ohlc_eth(start: str = None, end: str = None, limit: int = 1):
    """Fetch daily historical OHLCV data for ETH-USDT on Binance."""
    if start is None:
        start = "2024-01-01"
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    params = {
        "market": "binance",
        "instrument": "ETH-USDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": 1,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    headers = {"Content-type": "application/json; charset=UTF-8"}

    for attempt in range(3):
        try:
            response = requests.get(COINDESK_BASE, params=params, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            data_list = json_data.get("Data", [])
            if not data_list:
                print(f"[WARNING] Empty data for ETH-USDT, attempt {attempt+1}")
                time.sleep(2)
                continue
            df = pd.DataFrame(data_list)
            df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df.set_index("timestamp", inplace=True)
            df = df[df.index >= pd.Timestamp(start)]
            df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            print(f"[SUCCESS] Fetched {len(df)} rows for ETH-USDT from {df.index.min().date()} to {df.index.max().date()}")
            return df
        except Exception as e:
            print(f"[ERROR] Fetching CoinDesk OHLC for ETH-USDT (attempt {attempt+1}): {e}")
            time.sleep(2)
    print("[ERROR] Failed to fetch ETH-USDT after retries")
    return pd.DataFrame()

def fetch_historical_ohlc_sol(start: str = None, end: str = None, limit: int = 1):
    """Fetch daily historical OHLCV data for SOL-USDT on Binance."""
    if start is None:
        start = "2024-01-01"
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    params = {
        "market": "binance",
        "instrument": "SOL-USDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": 1,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    headers = {"Content-type": "application/json; charset=UTF-8"}

    for attempt in range(3):
        try:
            response = requests.get(COINDESK_BASE, params=params, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            data_list = json_data.get("Data", [])
            if not data_list:
                print(f"[WARNING] Empty data for SOL-USDT, attempt {attempt+1}")
                time.sleep(2)
                continue
            df = pd.DataFrame(data_list)
            df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df.set_index("timestamp", inplace=True)
            df = df[df.index >= pd.Timestamp(start)]
            df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            print(f"[SUCCESS] Fetched {len(df)} rows for SOL-USDT from {df.index.min().date()} to {df.index.max().date()}")
            return df
        except Exception as e:
            print(f"[ERROR] Fetching CoinDesk OHLC for SOL-USDT (attempt {attempt+1}): {e}")
            time.sleep(2)
    print("[ERROR] Failed to fetch SOL-USDT after retries")
    return pd.DataFrame()

def fetch_historical_ohlc_paxg(start: str = None, end: str = None, limit: int = 1):
    """Fetch daily historical OHLCV data for PAXG-USDT on Binance."""
    if start is None:
        start = "2024-01-01"
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    params = {
        "market": "binance",
        "instrument": "PAXG-USDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": 1,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    headers = {"Content-type": "application/json; charset=UTF-8"}

    for attempt in range(3):
        try:
            response = requests.get(COINDESK_BASE, params=params, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            data_list = json_data.get("Data", [])
            if not data_list:
                print(f"[WARNING] Empty data for PAXG-USDT, attempt {attempt+1}")
                time.sleep(2)
                continue
            df = pd.DataFrame(data_list)
            df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df.set_index("timestamp", inplace=True)
            df = df[df.index >= pd.Timestamp(start)]
            df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            print(f"[SUCCESS] Fetched {len(df)} rows for PAXG-USDT from {df.index.min().date()} to {df.index.max().date()}")
            return df
        except Exception as e:
            print(f"[ERROR] Fetching CoinDesk OHLC for PAXG-USDT (attempt {attempt+1}): {e}")
            time.sleep(2)
    print("[ERROR] Failed to fetch PAXG-USDT after retries")
    return pd.DataFrame()

def fetch_historical_ohlc_sui(start: str = None, end: str = None, limit: int = 1):
    """Fetch daily historical OHLCV data for SUI-USDT on Binance.
    
    Note: SUI data is only available from May 2023 onwards.
    """
    if start is None:
        start = "2023-05-01"  # SUI launched May 2023
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    params = {
        "market": "binance",
        "instrument": "SUI-USDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": 1,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    headers = {"Content-type": "application/json; charset=UTF-8"}

    for attempt in range(3):
        try:
            response = requests.get(COINDESK_BASE, params=params, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            data_list = json_data.get("Data", [])
            if not data_list:
                print(f"[WARNING] Empty data for SUI-USDT, attempt {attempt+1}")
                time.sleep(2)
                continue
            df = pd.DataFrame(data_list)
            df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df.set_index("timestamp", inplace=True)
            df = df[df.index >= pd.Timestamp(start)]
            df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            print(f"[SUCCESS] Fetched {len(df)} rows for SUI-USDT from {df.index.min().date()} to {df.index.max().date()}")
            return df
        except Exception as e:
            print(f"[ERROR] Fetching CoinDesk OHLC for SUI-USDT (attempt {attempt+1}): {e}")
            time.sleep(2)
    print("[ERROR] Failed to fetch SUI-USDT after retries")
    return pd.DataFrame()

def fetch_historical_ohlc_bnb(start: str = None, end: str = None, limit: int = 1):
    """Fetch daily historical OHLCV data for BNB-USDT on Binance."""
    if start is None:
        start = "2024-01-01"
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    params = {
        "market": "binance",
        "instrument": "BNB-USDT",
        "start": start,
        "end": end,
        "limit": limit,
        "aggregate": 1,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    headers = {"Content-type": "application/json; charset=UTF-8"}

    for attempt in range(3):
        try:
            response = requests.get(COINDESK_BASE, params=params, headers=headers)
            response.raise_for_status()
            json_data = response.json()
            data_list = json_data.get("Data", [])
            if not data_list:
                print(f"[WARNING] Empty data for BNB-USDT, attempt {attempt+1}")
                time.sleep(2)
                continue
            df = pd.DataFrame(data_list)
            df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df.set_index("timestamp", inplace=True)
            df = df[df.index >= pd.Timestamp(start)]
            df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            print(f"[SUCCESS] Fetched {len(df)} rows for BNB-USDT from {df.index.min().date()} to {df.index.max().date()}")
            return df
        except Exception as e:
            print(f"[ERROR] Fetching CoinDesk OHLC for BNB-USDT (attempt {attempt+1}): {e}")
            time.sleep(2)
    print("[ERROR] Failed to fetch BNB-USDT after retries")
    return pd.DataFrame()

def fetch_market_data(binance_symbol: str, timeframe: str = "1d", limit: int = 700, start: str = None, end: str = None):
    """
    Fetch OHLCV for a single asset based on symbol, using specific fetch function.
    """
    instrument_map = {
        "BTCUSDT": fetch_historical_ohlc_btc,
        "ETHUSDT": fetch_historical_ohlc_eth,
        "SOLUSDT": fetch_historical_ohlc_sol,
        "SUIUSDT": fetch_historical_ohlc_sui,
        "BNBUSDT": fetch_historical_ohlc_bnb,
        "PAXGUSDT": fetch_historical_ohlc_paxg
    }
    fetch_func = instrument_map.get(binance_symbol)
    if not fetch_func:
        print(f"[ERROR] No fetch function for {binance_symbol}")
        return {"ohlcv": pd.DataFrame(), "fundamentals": {}}
   
    ohlcv_df = fetch_func(start=start, end=end, limit=limit)
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": {}
    }
