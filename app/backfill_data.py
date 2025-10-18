"""
Utility script to backfill historical data

Usage:
    python backfill_data.py                    # Backfill all assets from 2023-01-01
    python backfill_data.py --start 2022-01-01  # Custom start date
    python backfill_data.py --asset BTCUSDT    # Single asset only
"""

import argparse
import time
from app.data import backfill_historical_data

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PAXGUSDT"]

def main():
    parser = argparse.ArgumentParser(description="Backfill historical OHLCV data")
    parser.add_argument('--start', type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument('--asset', type=str, help="Single asset to backfill (e.g., BTCUSDT)")
    args = parser.parse_args()
    
    assets_to_process = [args.asset] if args.asset else ASSETS
    
    print(f"{'='*60}")
    print(f"BACKFILLING HISTORICAL DATA")
    print(f"{'='*60}")
    print(f"Start date: {args.start}")
    print(f"Assets: {', '.join(assets_to_process)}")
    print(f"{'='*60}\n")
    
    for asset in assets_to_process:
        if asset not in ASSETS:
            print(f"[WARNING] Unknown asset {asset}, skipping...")
            continue
        
        print(f"\n📊 Processing {asset}...")
        backfill_historical_data(asset, args.start)
        time.sleep(2)  # Rate limiting between requests
    
    print(f"\n{'='*60}")
    print(f"✅ BACKFILL COMPLETE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()