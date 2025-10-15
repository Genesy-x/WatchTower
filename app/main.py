from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data import fetch_market_data
from app.indicators import compute_indicators
from app.strategies.universal_rs import compute_relative_strength, rotate_equity, compute_metrics
from app.tournament import run_tournament
from app.db.database import SessionLocal, BacktestRun, OHLCVData
from app.equity import compute_equity
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
import os
import uvicorn

app = FastAPI(title="WatchTower Backend", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://super-salamander-d9eb9b.netlify.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALL_ASSETS = [
    ("BTCUSDT", "bitcoin"),
    ("ETHUSDT", "ethereum"),
    ("SOLUSDT", "solana"),
    ("PAXGUSDT", "pax-gold")
]

def store_single_asset(db, name, timeframe: str = "1d", limit: int = 1):
    try:
        market_data = fetch_market_data(name, timeframe, limit)
        ohlcv = market_data["ohlcv"]
        instrument = name.replace("USDT", "")
        print(f"Storing OHLCV for {instrument}: {ohlcv.head()}")
        if not ohlcv.empty:
            for index, row in ohlcv.iterrows():
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
            db.commit()
            print(f"Successfully committed {len(ohlcv)} new rows for {instrument} to Neon")
        else:
            print(f"[WARNING] No new data for {instrument}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to store {instrument}: {e}")
        db.rollback()
        return False

@app.get("/")
async def root():
    """Health check endpoint to trigger logs."""
    print("Server is running!")
    return {"status": "healthy"}

@app.get("/store-btc")
async def store_btc():
    db = SessionLocal()
    success = store_single_asset(db, "BTCUSDT")
    db.close()
    return {"status": "BTC data stored" if success else "Failed to store BTC data"}

@app.get("/store-eth")
async def store_eth():
    db = SessionLocal()
    success = store_single_asset(db, "ETHUSDT")
    db.close()
    return {"status": "ETH data stored" if success else "Failed to store ETH data"}

@app.get("/store-sol")
async def store_sol():
    db = SessionLocal()
    success = store_single_asset(db, "SOLUSDT")
    db.close()
    return {"status": "SOL data stored" if success else "Failed to store SOL data"}

@app.get("/store-xaut")
async def store_xaut():
    db = SessionLocal()
    success = store_single_asset(db, "PAXGUSDT")
    db.close()
    return {"status": "XAUT data stored" if success else "Failed to store XAUT data"}

@app.get("/backtest")
async def backtest(start_date: str = "2024-01-01", limit: int = 700, used_assets: int = 3,
                   use_gold: bool = True, benchmark: str = "BTC", timeframe: str = "1d"):
    try:
        # Fetch data from Neon (implement a query function if needed)
        db = SessionLocal()
        assets_data = {}
        for symbol, _ in ALL_ASSETS[:used_assets + 1]:
            instrument = symbol.replace("USDT", "")
            query = db.query(OHLCVData).filter(OHLCVData.instrument == instrument).order_by(OHLCVData.timestamp).all()
            if query:
                df = pd.DataFrame([(q.timestamp, q.open, q.high, q.low, q.close, q.volume) for q in query],
                                columns=["timestamp", "open", "high", "low", "close", "volume"])
                df.set_index("timestamp", inplace=True)
                assets_data[instrument] = compute_indicators(df)
        db.close()

        if not assets_data:
            return {"error": "No data available in Neon"}

        gold_data = assets_data.get("XAUT", pd.DataFrame())
        if gold_data.empty and use_gold:
            market_data = fetch_market_data("PAXGUSDT", timeframe, limit)
            gold_data = compute_indicators(market_data["ohlcv"])

        top_assets = list(assets_data.keys())[:used_assets]
        rs_data = compute_relative_strength({k: v for k, v in assets_data.items() if k in top_assets}, filtered=True)
        equity_filtered, alloc_hist_filtered, switches_filtered = rotate_equity(
            rs_data, {k: v for k, v in assets_data.items() if k in top_assets}, gold_data, start_date=start_date, use_gold=use_gold
        )
        metrics_filtered = compute_metrics(equity_filtered)

        # Generate signal from allocation history for strategy equity
        strategy_df = assets_data[top_assets[0]]  # Use top asset as base
        strategy_df["signal"] = 0
        for date, alloc in alloc_hist_filtered.items():
            if date in strategy_df.index:
                strategy_df.loc[date, "signal"] = 1 if alloc != "CASH" else 0
        strategy_df, strategy_metrics = compute_equity(strategy_df)

        # Benchmark equity
        if benchmark in assets_data:
            benchmark_df = assets_data[benchmark]
            benchmark_df["signal"] = 1  # Always invested for buy-and-hold
            benchmark_df, benchmark_metrics = compute_equity(benchmark_df)
        else:
            benchmark_df = pd.DataFrame(index=strategy_df.index, data={"close": 1.0})
            benchmark_df["signal"] = 1
            benchmark_df, benchmark_metrics = compute_equity(benchmark_df)

        # Asset table with equity from compute_equity
        asset_table = [{"symbol": f"{k}USDT", "score": 0} for k in top_assets[:used_assets] + (["XAUT"] if use_gold else [])]
        for asset in asset_table:
            symbol = asset["symbol"].replace("USDT", "")
            if symbol in assets_data:
                asset_df = assets_data[symbol]
                asset_df["signal"] = 1  # Buy-and-hold for each asset
                _, asset_metrics = compute_equity(asset_df)
                asset["equity"] = asset_metrics["final_equity"]

        top3 = [f"{k}USDT" for k in list(assets_data.keys())[:3]]
        current_alloc = alloc_hist_filtered[-1] if alloc_hist_filtered else "CASH"

        metrics_table = {
            **metrics_filtered,
            "PositionChanges": switches_filtered,
            "EquityMaxDD": strategy_metrics["max_drawdown_%"],
            "NetProfit": strategy_metrics["total_return_%"]
        }

        response = {
            "metrics": metrics_table,
            "final_equity_filtered": strategy_df["equity"].iloc[-1],
            "switches": switches_filtered,
            "current_allocation": current_alloc,
            "top3": top3,
            "asset_table": asset_table,
            "equity_curve_filtered": strategy_df["equity"].to_dict(),
            "buy_hold_equity": benchmark_df["bh_equity"].to_dict()
        }

        strategy_df.index = strategy_df.index.astype(str)

        db = SessionLocal()
        run = BacktestRun(
            start_date=pd.to_datetime(start_date).to_pydatetime(),
            end_date=strategy_df.index[-1],
            metrics=metrics_table,
            equity_curve=strategy_df["equity"].to_dict(),
            alloc_hist=alloc_hist_filtered,
            switches=switches_filtered
        )
        db.add(run)
        db.commit()
        db.close()

        return response
    except Exception as e:
        return {"error": str(e)}

@app.get("/rebalance")
async def rebalance(used_assets: int = 3, use_gold: bool = True, timeframe: str = "1d", limit: int = 1):
    try:
        # Fetch data from Neon
        db = SessionLocal()
        assets_data = {}
        for symbol, _ in ALL_ASSETS[:used_assets + 1]:
            instrument = symbol.replace("USDT", "")
            query = db.query(OHLCVData).filter(OHLCVData.instrument == instrument).order_by(OHLCVData.timestamp).all()
            if query:
                df = pd.DataFrame([(q.timestamp, q.open, q.high, q.low, q.close, q.volume) for q in query],
                                columns=["timestamp", "open", "high", "low", "close", "volume"])
                df.set_index("timestamp", inplace=True)
                assets_data[instrument] = compute_indicators(df)
        db.close()

        if not assets_data:
            return {"error": "No data available in Neon"}

        gold_data = assets_data.get("XAUT", pd.DataFrame())
        if gold_data.empty and use_gold:
            market_data = fetch_market_data("PAXGUSDT", timeframe, limit)
            gold_data = compute_indicators(market_data["ohlcv"])

        top_assets = list(assets_data.keys())[:used_assets]
        rs_data = compute_relative_strength({k: v for k, v in assets_data.items() if k in top_assets}, filtered=True)
        equity_filtered, alloc_hist, switches = rotate_equity(
            rs_data, {k: v for k, v in assets_data.items() if k in top_assets}, gold_data, use_gold=use_gold
        )

        # Generate signal for rebalance equity
        rebalance_df = assets_data[top_assets[0]]  # Use top asset as base
        rebalance_df["signal"] = 0
        for date, alloc in alloc_hist.items():
            if date in rebalance_df.index:
                rebalance_df.loc[date, "signal"] = 1 if alloc != "CASH" else 0
        rebalance_df, rebalance_metrics = compute_equity(rebalance_df)

        asset_table = [{"symbol": f"{k}USDT", "score": 0} for k in top_assets + (["XAUT"] if use_gold else [])]
        for asset in asset_table:
            symbol = asset["symbol"].replace("USDT", "")
            if symbol in assets_data:
                asset_df = assets_data[symbol]
                asset_df["signal"] = 1  # Buy-and-hold for each asset
                _, asset_metrics = compute_equity(asset_df)
                asset["equity"] = asset_metrics["final_equity"]

        top3 = [f"{k}USDT" for k in list(assets_data.keys())[:3]]
        current_alloc = alloc_hist[-1]

        return {
            "current_allocation": current_alloc,
            "top3": top3,
            "asset_table": asset_table,
            "latest_equity": rebalance_df["equity"].iloc[-1],
            "switches": switches
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/store-btc")
async def store_btc():
    db = SessionLocal()
    success = store_single_asset(db, "BTCUSDT")
    db.close()
    return {"status": "BTC data stored" if success else "Failed to store BTC data"}

@app.get("/store-eth")
async def store_eth():
    db = SessionLocal()
    success = store_single_asset(db, "ETHUSDT")
    db.close()
    return {"status": "ETH data stored" if success else "Failed to store ETH data"}

@app.get("/store-sol")
async def store_sol():
    db = SessionLocal()
    success = store_single_asset(db, "SOLUSDT")
    db.close()
    return {"status": "SOL data stored" if success else "Failed to store SOL data"}

@app.get("/store-paxg")
async def store_paxg():
    db = SessionLocal()
    success = store_single_asset(db, "PAXGUSDT")
    db.close()
    return {"status": "PAXG data stored" if success else "Failed to store PAXG data"}

# Scheduler for daily updates post-UTC close
def daily_update():
    db = SessionLocal()
    for name, _ in ALL_ASSETS:
        store_single_asset(db, name)
    db.close()
    print(f"Daily update completed at {datetime.utcnow()} UTC")

scheduler = BackgroundScheduler()
scheduler.add_job(daily_update, 'cron', hour=0, minute=5, timezone='UTC')  # 5 mins after UTC 00:00
scheduler.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)