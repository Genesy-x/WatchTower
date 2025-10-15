from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data import fetch_market_data
from app.indicators import compute_indicators
from app.strategies.universal_rs import compute_relative_strength, rotate_equity, compute_metrics
from app.tournament import run_tournament
from app.db.database import SessionLocal, BacktestRun, OHLCVData
from datetime import datetime
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

def fetch_and_store_raw_data(assets=ALL_ASSETS, timeframe: str = "1d", limit: int = 700):
    """
    Fetch and store raw OHLCV data for assets in the database.
    """
    db = SessionLocal()
    try:
        for name, _ in assets:
            market_data = fetch_market_data(name, timeframe, limit)
            ohlcv = market_data["ohlcv"]
            symbol = name.replace("USDT", "")
            print(f"Storing OHLCV for {symbol}: {ohlcv.head()}")
            for index, row in ohlcv.iterrows():
                existing = db.query(OHLCVData).filter(
                    OHLCVData.symbol == symbol,
                    OHLCVData.timestamp == index
                ).first()
                if not existing:
                    record = OHLCVData(
                        symbol=symbol,
                        timestamp=index,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    db.add(record)
        db.commit()
        print("Data committed to Neon database")
    except Exception as e:
        print(f"[ERROR] Failed to store data in Neon: {e}")
        db.rollback()
    finally:
        db.close()

@app.get("/")
async def root():
    """Health check endpoint to trigger logs."""
    print("Server is running!")
    return {"status": "healthy"}

@app.get("/backtest")
async def backtest(start_date: str = "2024-01-01", limit: int = 700, used_assets: int = 3,
                   use_gold: bool = True, benchmark: str = "BTC", timeframe: str = "1d"):
    try:
        tournament_results = run_tournament(ALL_ASSETS[:used_assets + 1])
        if not tournament_results:
            return {"error": "No tournament results available"}
        assets_data = {}
        top_assets = [result["symbol"] for result in tournament_results[:used_assets]]
        for symbol, _ in ALL_ASSETS[:used_assets + 1]:
            market_data = fetch_market_data(symbol, timeframe, limit)
            ohlcv = market_data["ohlcv"]
            print(f"Raw OHLCV for {symbol}: {ohlcv.head()}")
            if ohlcv.empty:
                print(f"Empty OHLCV for {symbol}")
                continue
            assets_data[symbol.replace("USDT", "")] = compute_indicators(ohlcv)

        gold_market = fetch_market_data("PAXGUSDT", timeframe, limit)
        gold_data = compute_indicators(gold_market["ohlcv"])
        print(f"Raw OHLCV for GOLD: {gold_market['ohlcv'].head()}")

        rs_data = compute_relative_strength({k: v for k, v in assets_data.items() if k in top_assets}, filtered=True)
        equity_filtered, alloc_hist_filtered, switches_filtered = rotate_equity(
            rs_data, {k: v for k, v in assets_data.items() if k in top_assets}, gold_data, start_date=start_date, use_gold=use_gold
        )
        metrics_filtered = compute_metrics(equity_filtered)

        if benchmark in assets_data:
            bh_returns = assets_data[benchmark]["close"].pct_change().fillna(0)
            bh_equity = (1 + bh_returns).cumprod().reindex(equity_filtered.index).fillna(1)
        else:
            bh_equity = pd.Series(1.0, index=equity_filtered.index)

        asset_table = tournament_results[:used_assets + 1]
        for asset in asset_table:
            symbol = asset["symbol"].replace("USDT", "")
            if symbol in assets_data:
                equity = (1 + assets_data[symbol]["close"].pct_change()).cumprod().fillna(1)
                asset["equity"] = round(equity.iloc[-1], 2)

        top3 = [result["symbol"].replace("USDT", "") for result in tournament_results[:3]]
        current_alloc = alloc_hist_filtered[-1] if alloc_hist_filtered else "CASH"

        metrics_table = {
            **metrics_filtered,
            "PositionChanges": switches_filtered,
            "EquityMaxDD": metrics_filtered["MaxDD"],
            "NetProfit": metrics_filtered["NetProfit"]
        }

        response = {
            "metrics": metrics_table,
            "final_equity_filtered": equity_filtered.iloc[-1],
            "switches": switches_filtered,
            "current_allocation": current_alloc,
            "top3": top3,
            "asset_table": asset_table,
            "equity_curve_filtered": equity_filtered.to_dict(),
            "buy_hold_equity": bh_equity.to_dict()
        }

        equity_filtered.index = equity_filtered.index.astype(str)

        fetch_and_store_raw_data()
        db = SessionLocal()
        run = BacktestRun(
            start_date=pd.to_datetime(start_date).to_pydatetime(),
            end_date=equity_filtered.index[-1],
            metrics=metrics_table,
            equity_curve=equity_filtered.to_dict(),
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
async def rebalance(used_assets: int = 3, use_gold: bool = True, timeframe: str = "1d", limit: int = 700):
    try:
        tournament_results = run_tournament(ALL_ASSETS[:used_assets + 1])
        if not tournament_results:
            return {"error": "No tournament results available"}
        assets_data = {}
        top_assets = [result["symbol"] for result in tournament_results[:used_assets]]
        for symbol, _ in ALL_ASSETS[:used_assets + 1]:
            market_data = fetch_market_data(symbol, timeframe, limit)
            ohlcv = market_data["ohlcv"]
            assets_data[symbol.replace("USDT", "")] = compute_indicators(ohlcv)

        gold_market = fetch_market_data("PAXGUSDT", timeframe, limit)
        gold_data = compute_indicators(gold_market["ohlcv"])

        rs_data = compute_relative_strength({k: v for k, v in assets_data.items() if k in top_assets}, filtered=True)
        equity_filtered, alloc_hist, switches = rotate_equity(
            rs_data, {k: v for k, v in assets_data.items() if k in top_assets}, gold_data, use_gold=use_gold
        )

        asset_table = tournament_results[:used_assets + 1]
        for asset in asset_table:
            symbol = asset["symbol"].replace("USDT", "")
            if symbol in assets_data:
                equity = (1 + assets_data[symbol]["close"].pct_change()).cumprod().fillna(1)
                asset["equity"] = round(equity.iloc[-1], 2)

        top3 = [result["symbol"].replace("USDT", "") for result in tournament_results[:3]]
        current_alloc = alloc_hist[-1]

        return {
            "current_allocation": current_alloc,
            "top3": top3,
            "asset_table": asset_table,
            "latest_equity": equity_filtered.iloc[-1],
            "switches": switches
        }
    except Exception as e:
        return {"error": str(e)}

# Scheduler for daily rebalancing
def scheduled_rebalance():
    fetch_and_store_raw_data()
    response = backtest()
    print("Scheduled rebalance complete:", response["metrics"])

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_rebalance, 'interval', days=1)
scheduler.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)