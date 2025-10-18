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
        return latest[0] if latest else pd.Timestamp("2023-01-01")  # Changed to 2023 for more history
    finally:
        db.close()

def fetch_ohlc_generic(symbol: str, market_pair: str, start: str = None, end: str = None, limit: int = 1):
    """
    Generic fetch function for any symbol
    
    Args:
        symbol: Internal symbol (BTC, ETH, etc.)
        market_pair: CoinDesk pair (BTC-USDT, ETH-USDT, etc.)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        limit: Max rows to fetch (use 1 for daily update, 1000 for backfill)
    """
    if start is None:
        latest = get_latest_timestamp(symbol)
        start = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    params = {
        "market": "binance",
        "instrument": market_pair,
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
            response = requests.get(COINDESK_BASE, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            data_list = json_data.get("Data", [])
            
            if not data_list:
                print(f"[WARNING] No new data for {market_pair}, attempt {attempt+1}")
                time.sleep(2)
                continue
            
            df = pd.DataFrame(data_list)
            df["timestamp"] = pd.to_datetime(df["TIMESTAMP"], unit="s")
            df.set_index("timestamp", inplace=True)
            df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
            df.columns = ["open", "high", "low", "close", "volume"]
            
            print(f"[SUCCESS] Fetched {len(df)} rows for {market_pair} from {df.index.min().date()} to {df.index.max().date()}")
            return df
            
        except Exception as e:
            print(f"[ERROR] Fetching {market_pair} (attempt {attempt+1}): {e}")
            time.sleep(2)
    
    print(f"[ERROR] Failed to fetch {market_pair} after retries")
    return pd.DataFrame()

# Wrapper functions for each asset
def fetch_historical_ohlc_btc(start: str = None, end: str = None, limit: int = 1):
    return fetch_ohlc_generic("BTC", "BTC-USDT", start, end, limit)

def fetch_historical_ohlc_eth(start: str = None, end: str = None, limit: int = 1):
    return fetch_ohlc_generic("ETH", "ETH-USDT", start, end, limit)

def fetch_historical_ohlc_sol(start: str = None, end: str = None, limit: int = 1):
    return fetch_ohlc_generic("SOL", "SOL-USDT", start, end, limit)

def fetch_historical_ohlc_xaut(start: str = None, end: str = None, limit: int = 1):
    return fetch_ohlc_generic("PAXG", "PAXG-USDT", start, end, limit)

def fetch_market_data(binance_symbol: str, timeframe: str = "1d", limit: int = 1):
    """
    Fetch OHLCV for a single asset
    """
    instrument_map = {
        "BTCUSDT": fetch_historical_ohlc_btc,
        "ETHUSDT": fetch_historical_ohlc_eth,
        "SOLUSDT": fetch_historical_ohlc_sol,
        "PAXGUSDT": fetch_historical_ohlc_xaut
    }
    fetch_func = instrument_map.get(binance_symbol)
    if not fetch_func:
        print(f"[ERROR] No fetch function for {binance_symbol}")
        return {"ohlcv": pd.DataFrame(), "fundamentals": {}}
    
    ohlcv_df = fetch_func(limit=limit)
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": {}
    }

def backfill_historical_data(symbol: str, start_date: str = "2023-01-01"):
    """
    Backfill historical data for an asset from start_date to latest
    
    Usage:
        from app.data import backfill_historical_data
        backfill_historical_data("BTCUSDT", "2023-01-01")
    """
    print(f"[BACKFILL] Starting backfill for {symbol} from {start_date}")
    
    instrument_map = {
        "BTCUSDT": ("BTC", "BTC-USDT"),
        "ETHUSDT": ("ETH", "ETH-USDT"),
        "SOLUSDT": ("SOL", "SOL-USDT"),
        "PAXGUSDT": ("PAXG", "PAXG-USDT")
    }
    
    if symbol not in instrument_map:
        print(f"[ERROR] Unknown symbol {symbol}")
        return
    
    instrument, pair = instrument_map[symbol]
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch all historical data (CoinDesk limit is ~2000 rows per request)
    df = fetch_ohlc_generic(instrument, pair, start=start_date, end=end_date, limit=2000)
    
    if df.empty:
        print(f"[BACKFILL] No data fetched for {symbol}")
        return
    
    # Store in database
    db = SessionLocal()
    stored_count = 0
    
    try:
        for index, row in df.iterrows():
            existing = db.query(OHLCVData).filter(
                OHLCVData.instrument == instrument,
                OHLCVData.timestamp == index
            ).first()
            
            if not existing:
                record = OHLCVData(
                    instrument=instrument,
                    timestamp=index.to_pydatetime(),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                )
                db.add(record)
                stored_count += 1
        
        db.commit()
        print(f"[BACKFILL] ✓ Stored {stored_count} new rows for {symbol}")
        
    except Exception as e:
        print(f"[BACKFILL] ✗ Failed: {e}")
        db.rollback()
    finally:
        db.close()

# CLI helper for backfilling
if __name__ == "__main__":
    print("Backfilling historical data for all assets...")
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PAXGUSDT"]:
        backfill_historical_data(symbol, "2023-01-01")
        time.sleep(1)  # Rate limiting
    print("Backfill complete!")