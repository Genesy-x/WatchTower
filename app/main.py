from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data import fetch_market_data
from app.indicators import compute_indicators
from app.strategies.universal_rs import compute_relative_strength, rotate_equity, compute_metrics
from app.db.database import SessionLocal, BacktestRun
from datetime import datetime
from app.db.database import OHLCVData
import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
# import redis
import os

app = FastAPI(title="WatchTower Backend", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://super-salamander-d9eb9b.netlify.app", "http://localhost:3000"],  # Allow Netlify and local frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for caching (set REDIS_URL env var, e.g., redis://localhost:6379/0)
# redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

ALL_ASSETS = {
    "BTC": {"symbol": "BTCUSDT", "cg_id": "bitcoin"},
    "ETH": {"symbol": "ETHUSDT", "cg_id": "ethereum"},
    "SOL": {"symbol": "SOLUSDT", "cg_id": "solana"},
    "XRP": {"symbol": "XRPUSDT", "cg_id": "ripple"},
    "DOGE": {"symbol": "DOGEUSDT", "cg_id": "dogecoin"},
    "SUI": {"symbol": "SUIUSDT", "cg_id": "sui"},
    "BNB": {"symbol": "BNBUSDT", "cg_id": "binancecoin"},
    "TRX": {"symbol": "TRXUSDT", "cg_id": "tron"},
    "ADA": {"symbol": "ADAUSDT", "cg_id": "cardano"},
    "LINK": {"symbol": "LINKUSDT", "cg_id": "chainlink"},
}

GOLD = {"symbol": "PAXGUSDT", "cg_id": "pax-gold"}

def cached_fetch_market_data(symbol: str, cg_id: str, timeframe: str = "1d", limit: int = 500):
    # Disabled caching; direct fetch
    return fetch_market_data(symbol, cg_id, timeframe, limit)

@app.get("/backtest")
async def backtest(start_date: str = "2023-01-01", limit: int = 500, used_assets: int = 6,
                   use_gold: bool = True, benchmark: str = "BTC", timeframe: str = "1d"):
    try:
        assets = dict(list(ALL_ASSETS.items())[:used_assets])
        assets_data = {}
        for name, info in assets.items():
            market_data = cached_fetch_market_data(info["symbol"], info["cg_id"], timeframe, limit)
            ohlcv = market_data["ohlcv"]
            assets_data[name] = compute_indicators(ohlcv)

        gold_market = cached_fetch_market_data(GOLD["symbol"], GOLD["cg_id"], timeframe, limit)
        gold_data = compute_indicators(gold_market["ohlcv"])

        # Unfiltered RS
        rs_data_unfiltered = compute_relative_strength(assets_data, filtered=False)
        equity_unfiltered, alloc_hist_unfiltered, switches_unfiltered = rotate_equity(
            rs_data_unfiltered, assets_data, gold_data, start_date=start_date, use_gold=use_gold
        )
        metrics_unfiltered = compute_metrics(equity_unfiltered)

        # Filtered RS
        rs_data_filtered = compute_relative_strength(assets_data, filtered=True)
        equity_filtered, alloc_hist_filtered, switches_filtered = rotate_equity(
            rs_data_filtered, assets_data, gold_data, start_date=start_date, use_gold=use_gold
        )
        metrics_filtered = compute_metrics(equity_filtered)

        # Buy & Hold for benchmark
        if benchmark in assets_data:
            bh_returns = assets_data[benchmark]["close"].pct_change().fillna(0)
            bh_equity = (1 + bh_returns).cumprod().reindex(equity_filtered.index).fillna(1)
        else:
            bh_equity = pd.Series(1.0, index=equity_filtered.index)

        # Asset table data (for mainT)
        asset_table = []
        rs_last = rs_data_filtered.iloc[-1]
        ranks = rs_last.rank(ascending=False, pct=False, method='min')  # 1 for top
        for name, df in assets_data.items():
            tpi_last = df["TPI"].iloc[-1]
            signal = "Long" if tpi_last > 0 else "Short" if tpi_last < 0 else "Neutral"
            asset_equity = (1 + df["close"].pct_change()).cumprod().fillna(1)
            asset_returns = (asset_equity.iloc[-1] - 1) * 100
            asset_maxdd = ((asset_equity / asset_equity.cummax()) - 1).min() * 100
            asset_table.append({
                "name": name,
                "rank": ranks[name] if not pd.isna(ranks[name]) else "N/A",
                "signal": signal,
                "returns": round(asset_returns, 2),
                "max_dd": round(asset_maxdd, 2)
            })

        # Top 3 (for miniT)
        top_assets = rs_last[~np.isneginf(rs_last)].sort_values(ascending=False).index[:3].tolist()

        # Current allocation
        current_alloc = alloc_hist_filtered[-1] if alloc_hist_filtered else "CASH"

        # Metrics table (combined for filtered)
        metrics_table = {
            **metrics_filtered,
            "PositionChanges": switches_filtered,
            "EquityMaxDD": metrics_filtered["MaxDD"],  # Already %
            "NetProfit": metrics_filtered["NetProfit"]
        }

        response = {
            "metrics": metrics_table,
            "final_equity_filtered": equity_filtered.iloc[-1],
            "final_equity_unfiltered": equity_unfiltered.iloc[-1],
            "switches": switches_filtered,
            "current_allocation": current_alloc,
            "top3": top_assets,
            "asset_table": asset_table,
            "equity_curve_filtered": equity_filtered.to_dict(),
            "equity_curve_unfiltered": equity_unfiltered.to_dict(),
            "buy_hold_equity": bh_equity.to_dict()
        }

        # Convert index to str for JSON serialization
        equity_filtered.index = equity_filtered.index.astype(str)

        # Store in DB
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
async def rebalance(used_assets: int = 6, use_gold: bool = True, timeframe: str = "12h", limit: int = 168):  # 1 week for recent data
    try:
        assets = dict(list(ALL_ASSETS.items())[:used_assets])
        assets_data = {}
        for name, info in assets.items():
            market_data = cached_fetch_market_data(info["symbol"], info["cg_id"], timeframe, limit)
            ohlcv = market_data["ohlcv"]
            assets_data[name] = compute_indicators(ohlcv)

        gold_market = cached_fetch_market_data(GOLD["symbol"], GOLD["cg_id"], timeframe, limit)
        gold_data = compute_indicators(gold_market["ohlcv"])

        rs_data = compute_relative_strength(assets_data, filtered=True)
        # Use recent data, no start_date for live
        equity_filtered, alloc_hist, switches = rotate_equity(rs_data, assets_data, gold_data, use_gold=use_gold)

        rs_last = rs_data.iloc[-1]
        top_assets = rs_last[~np.isneginf(rs_last)].sort_values(ascending=False).index[:3].tolist()
        current_alloc = alloc_hist[-1]

        asset_table = []
        ranks = rs_last.rank(ascending=False, pct=False, method='min')
        for name, df in assets_data.items():
            tpi_last = df["TPI"].iloc[-1]
            signal = "Long" if tpi_last > 0 else "Short" if tpi_last < 0 else "Neutral"
            asset_returns = df["close"].pct_change().sum() * 100  # Recent returns sum for live
            asset_maxdd = 0  # Simplified for live; full in backtest
            asset_table.append({
                "name": name,
                "rank": ranks[name] if not pd.isna(ranks[name]) else "N/A",
                "signal": signal,
                "returns": round(asset_returns, 2),
                "max_dd": round(asset_maxdd, 2)
            })

        return {
            "current_allocation": current_alloc,
            "top3": top_assets,
            "asset_table": asset_table,
            "latest_equity": equity_filtered.iloc[-1],
            "switches": switches
        }
    except Exception as e:
        return {"error": str(e)}

# Scheduler for 12h rebalancing (runs backtest and stores)
def scheduled_rebalance():
    # Call backtest logic with defaults
    response = backtest()
    print("Scheduled rebalance complete:", response["metrics"])

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_rebalance, 'interval', hours=12)
scheduler.start()

def fetch_and_store_raw_data(used_assets: int = 6, timeframe: str = "1d", limit: int = 500):
    db = SessionLocal()
    assets = dict(list(ALL_ASSETS.items())[:used_assets])
    for name, info in assets.items():
        market_data = fetch_market_data(info["symbol"], info["cg_id"], timeframe, limit)
        ohlcv = market_data["ohlcv"]
        for index, row in ohlcv.iterrows():
            existing = db.query(OHLCVData).filter(
                OHLCVData.symbol == name,
                OHLCVData.timestamp == index
            ).first()
            if not existing:
                record = OHLCVData(
                    symbol=name,
                    timestamp=index,
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                db.add(record)
    db.commit()
    db.close()

# Call in scheduler or backtest
def scheduled_rebalance():
    fetch_and_store_raw_data()  # Store/update raw data
    response = backtest()
    print("Scheduled rebalance complete:", response["metrics"])