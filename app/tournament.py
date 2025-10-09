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
            data = fetch_market_data(symbol, cg_id)
            df = compute_indicators(data["ohlcv"])
            score = score_coin(df, data["fundamentals"])
            results.append({
                "symbol": symbol,
                "name": data["fundamentals"]["name"],
                "score": score,
                "price": data["fundamentals"]["price"],
                "market_cap": data["fundamentals"]["market_cap"],
            })
            time.sleep(1)  # avoid rate limits
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

    # Sort by score descending
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results
