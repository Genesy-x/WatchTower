import requests
import pandas as pd
import time
from datetime import datetime


COINDESK_API_KEY = "74e1e197fe44b98d6c1cfd466095fa9fa4c2a57edea008936b2ac1d5ad5167d1"  # Replace if needed
COINDESK_BASE = "https://data-api.coindesk.com/spot/v1/historical/days"


def fetch_historical_ohlc_btc(start: str = "2023-01-01", end: str = datetime.now().strftime("%Y-%m-%d"), limit: int = 700, aggregate: int = 1):
   """Fetch daily historical OHLCV data for BTC-USDT on Binance."""
   params = {
       "market": "binance",
       "instrument": "BTC-USDT",
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
           df = df[df.index >= pd.Timestamp("2023-01-01")]
           df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
           df.columns = ["open", "high", "low", "close", "volume"]
           print(f"[SUCCESS] Fetched {len(df)} rows for BTC-USDT from {df.index.min().date()} to {df.index.max().date()}")
           return df
       except Exception as e:
           print(f"[ERROR] Fetching CoinDesk OHLC for BTC-USDT (attempt {attempt+1}): {e}")
           time.sleep(2)
   print("[ERROR] Failed to fetch BTC-USDT after retries")
   return pd.DataFrame()


def fetch_historical_ohlc_eth(start: str = "2023-01-01", end: str = datetime.now().strftime("%Y-%m-%d"), limit: int = 700, aggregate: int = 1):
   """Fetch daily historical OHLCV data for ETH-USDT on Binance."""
   params = {
       "market": "binance",
       "instrument": "ETH-USDT",
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
           df = df[df.index >= pd.Timestamp("2023-01-01")]
           df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
           df.columns = ["open", "high", "low", "close", "volume"]
           print(f"[SUCCESS] Fetched {len(df)} rows for ETH-USDT from {df.index.min().date()} to {df.index.max().date()}")
           return df
       except Exception as e:
           print(f"[ERROR] Fetching CoinDesk OHLC for ETH-USDT (attempt {attempt+1}): {e}")
           time.sleep(2)
   print("[ERROR] Failed to fetch ETH-USDT after retries")
   return pd.DataFrame()


def fetch_historical_ohlc_sol(start: str = "2023-01-01", end: str = datetime.now().strftime("%Y-%m-%d"), limit: int = 700, aggregate: int = 1):
   """Fetch daily historical OHLCV data for SOL-USDT on Binance."""
   params = {
       "market": "binance",
       "instrument": "SOL-USDT",
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
           df = df[df.index >= pd.Timestamp("2023-01-01")]
           df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
           df.columns = ["open", "high", "low", "close", "volume"]
           print(f"[SUCCESS] Fetched {len(df)} rows for SOL-USDT from {df.index.min().date()} to {df.index.max().date()}")
           return df
       except Exception as e:
           print(f"[ERROR] Fetching CoinDesk OHLC for SOL-USDT (attempt {attempt+1}): {e}")
           time.sleep(2)
   print("[ERROR] Failed to fetch SOL-USDT after retries")
   return pd.DataFrame()


def fetch_historical_ohlc_paxg(start: str = "2023-01-01", end: str = datetime.now().strftime("%Y-%m-%d"), limit: int = 700, aggregate: int = 1):
   """Fetch daily historical OHLCV data for PAXG-USDT on Binance."""
   params = {
       "market": "binance",
       "instrument": "PAXG-USDT",
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
           df = df[df.index >= pd.Timestamp("2023-01-01")]
           df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
           df.columns = ["open", "high", "low", "close", "volume"]
           print(f"[SUCCESS] Fetched {len(df)} rows for PAXG-USDT from {df.index.min().date()} to {df.index.max().date()}")
           return df
       except Exception as e:
           print(f"[ERROR] Fetching CoinDesk OHLC for PAXG-USDT (attempt {attempt+1}): {e}")
           time.sleep(2)
   print("[ERROR] Failed to fetch PAXG-USDT after retries")
   return pd.DataFrame()


def fetch_market_data(binance_symbol: str, timeframe: str = "1d", limit: int = 700):
   """
   Fetch OHLCV for a single asset based on symbol, using specific fetch function.
   """
   instrument_map = {
       "BTCUSDT": fetch_historical_ohlc_btc,
       "ETHUSDT": fetch_historical_ohlc_eth,
       "SOLUSDT": fetch_historical_ohlc_sol,
       "PAXGUSDT": fetch_historical_ohlc_paxg
   }
   fetch_func = instrument_map.get(binance_symbol)
   if not fetch_func:
       print(f"[ERROR] No fetch function for {binance_symbol}")
       return {"ohlcv": pd.DataFrame(), "fundamentals": {}}
  
   ohlcv_df = fetch_func(limit=limit)
   return {
       "ohlcv": ohlcv_df,
       "fundamentals": {}  # Empty, but needed for tournament
   }
