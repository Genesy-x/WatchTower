from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data import fetch_market_data
from app.indicators import compute_indicators
from app.strategies.universal_rs import compute_relative_strength, rotate_equity, compute_metrics
from app.tournament import run_tournament
from app.db.database import SessionLocal, BacktestRun, OHLCVData
from app.equity import compute_equity
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
    allow_methods=["*"],  # Corrected from [""] to ["*"]
    allow_headers=["*"],  # Corrected from [""] to ["*"]
)

ALL_ASSETS = [
    ("BTCUSDT", "bitcoin"),
    ("ETHUSDT", "ethereum"),
    ("SOLUSDT", "solana"),
    ("PAXGUSDT", "pax-gold")
]

def store_single_asset(db, name, timeframe: str = "1d", limit: int = 700):
    try:
        market_data = fetch_market_data(name, timeframe, limit)
        ohlcv = market_data["ohlcv"]
        instrument = name.replace("USDT", "")
        print(f"Storing OHLCV for {instrument}: {ohlcv.head()}")
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
        print(f"Successfully committed rows for {instrument} to Neon")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to store {instrument}: {e}")
        db.rollback()
        return False

def fetch_and_store_raw_data(assets=ALL_ASSETS, timeframe: str = "1d", limit: int = 700):
    db = SessionLocal()
    for name, _ in assets:
        store_single_asset(db, name, timeframe, limit)
    db.close()

@app.get("/")
async def root():
    """Health check endpoint to trigger logs."""
    print("Server is running!")
    return {"status": "healthy"}

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