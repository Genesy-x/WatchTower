import time
from app.data import fetch_market_data
from app.indicators import compute_indicators
from app.scoring import score_coin

def run_tournament(assets):
    """
    Run the WatchTower tournament for a list of assets.
    Returns: list of dicts sorted by score
    """
    results = []

    for symbol, cg_id in assets:
        print(f"⚙️  Processing {symbol}...")
        try:
            data = fetch_market_data(symbol, "1d", 700)
            df = compute_indicators(data["ohlcv"])
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
            time.sleep(1)  # Avoid rate limits
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
