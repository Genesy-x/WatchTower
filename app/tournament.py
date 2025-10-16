import time
from app.data import fetch_market_data
from app.indicators import compute_indicators
from app.scoring import score_coin

def run_tournament(assets, assets_data=None):
    """
    Run the WatchTower tournament for a list of assets.
    
    Args:
        assets: list of (symbol, cg_id) tuples
        assets_data: dict of pre-computed indicator data {instrument: df}
                    If provided, use this instead of fetching fresh data
    
    Returns: list of dicts sorted by score
    """
    results = []

    for symbol, cg_id in assets:
        print(f"⚙️  Processing {symbol}...")
        try:
            instrument = symbol.replace("USDT", "")
            
            # Use pre-computed data if available
            if assets_data and instrument in assets_data:
                df = assets_data[instrument]
                print(f"[DEBUG] Using pre-loaded data for {symbol}: {len(df)} rows")
            else:
                # Fall back to fetching fresh data
                data = fetch_market_data(symbol, "1d", 700)
                df = compute_indicators(data["ohlcv"])
                print(f"[DEBUG] Fetched fresh data for {symbol}: {len(df)} rows")
            
            if df.empty or len(df) < 100:  # Minimum 100 days for validity
                print(f"[WARNING] Insufficient data for {symbol}: {len(df)} rows")
                results.append({
                    "symbol": symbol,
                    "name": symbol.replace("USDT", ""),
                    "score": 0.0,
                    "price": 0,
                    "market_cap": 0
                })
                continue
            
            # Mock fundamentals
            fundamentals = {
                "name": symbol.replace("USDT", ""),
                "price": df["close"].iloc[-1] if not df.empty else 0,
                "market_cap": 0,
                "volume_24h": df["volume"].iloc[-1] if not df.empty else 0
            }
            score = score_coin(df, fundamentals)
            results.append({
                "symbol": symbol,
                "name": fundamentals["name"],
                "score": score,
                "price": fundamentals["price"],
                "market_cap": fundamentals["market_cap"],
            })
            time.sleep(0.1)  # Small delay to avoid rate limits
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")
            results.append({
                "symbol": symbol,
                "name": symbol.replace("USDT", ""),
                "score": 0.0,
                "price": 0,
                "market_cap": 0
            })

    # Sort by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results