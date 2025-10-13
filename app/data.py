import requests
import pandas as pd

COINDESK_API_KEY = "74e1e197fe44b98d6c1cfd466095fa9fa4c2a57edea008936b2ac1d5ad5167d1"  # Replace with your own if needed
COINDESK_BASE = "https://data-api.coindesk.com/spot/v1/historical/hours"

def fetch_historical_ohlc_batch(market: str = "binance", instruments: list = ["BTC-USDT"], start: str = None, end: str = None, limit: int = 20, aggregate: int = 1):
    """
    Batch fetch historical OHLCV from CoinDesk for multiple instruments in one call.
    - instruments: List of pairs like ["BTC-USDT", "ETH-USDT"]
    """
    instrument_str = ','.join(instruments)
    params = {
        "market": market,
        "instrument": instrument_str,
        "limit": limit,
        "aggregate": aggregate,
        "fill": "true",
        "apply_mapping": "true",
        "response_format": "JSON",
        "groups": "ID,VOLUME,OHLC",
        "api_key": COINDESK_API_KEY
    }
    if start:
        params["start"] = start  # YYYY-MM-DD
    if end:
        params["end"] = end

    try:
        response = requests.get(COINDESK_BASE, params=params, headers={"Content-type": "application/json; charset=UTF-8"})
        response.raise_for_status()
        json_data = response.json()
        data = json_data["data"]
        dfs = {}
        for item in data:
            instrument = item["instrument"]
            df = pd.DataFrame(item["ohlcv"])  # Adjust based on actual response structure
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            dfs[instrument] = df[["open", "high", "low", "close", "volume_base"]]  # Adapt columns
        return dfs  # Dict of {instrument: DF}
    except Exception as e:
        print(f"[ERROR] Batch fetching CoinDesk OHLC: {e}")
        return {}  # Empty dict on error

def fetch_market_data(binance_symbol: str, coingecko_id: str, timeframe: str = "1d", limit: int = 20):
    """
    Fetch for single asset (wrapper for batch with one instrument).
    """
    instrument = binance_symbol.replace("USDT", "-USDT")  # Adapt format
    dfs = fetch_historical_ohlc_batch(market="binance", instruments=[instrument], limit=limit, aggregate=24 if timeframe == "1d" else 1)
    ohlcv_df = dfs.get(instrument, pd.DataFrame())
    return {
        "ohlcv": ohlcv_df,
        "fundamentals": {}  # Empty, as not used
    }