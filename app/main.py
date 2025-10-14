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
import os

app = FastAPI(title="WatchTower Backend", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://super-salamander-d9eb9b.netlify.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALL_ASSETS = {
    "BTC": {"symbol": "BTCUSDT"},
    "ETH": {"symbol": "ETHUSDT"},
    "SOL": {"symbol": "SOLUSDT"},
    "GOLD": {"symbol": "PAXGUSDT"}  # Added GOLD as an asset
}

def fetch_and_store_raw_data(used_assets: int = 4, timeframe: str = "1d", limit: int = 500):
    """
    Fetch and store raw OHLCV data for assets in the database.
    """
    db = SessionLocal()
    assets = dict(list(ALL_ASSETS.items())[:used_assets])
    for name, info in assets.items():
        market_data = fetch_market_data(info["symbol"], timeframe, limit)
        ohlcv = market_data["ohlcv"]
        print(f"Storing OHLCV for {name}: {ohlcv.head()}")
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
    print("Data committed to Neon database")
    db.close()

@app.get("/backtest")
async def backtest(start_date: str = "2024-01-01", limit: int = 500, used_assets: int = 4,
                   use_gold: bool = True, benchmark: str = "BTC", timeframe: str = "1d"):
    try:
        assets = dict(list(ALL_ASSETS.items())[:used_assets])
        assets_data = {}
        for name, info in assets.items():
            market_data = fetch_market_data(info["symbol"], timeframe, limit)
            ohlcv = market_data["ohlcv"]
            print(f"Raw OHLCV for {name}: {ohlcv.head()}")
            if ohlcv.empty:
                print(f"Empty OHLCV for {name}")
                continue
            assets_data[name] = compute_indicators(ohlcv)

        # Ensure gold_data is always fetched, even if not used in rotation
        gold_market = fetch_market_data(ALL_ASSETS["GOLD"]["symbol"], timeframe, limit)
        gold_data = compute_indicators(gold_market["ohlcv"])
        print(f"Raw OHLCV for GOLD: {gold_market['ohlcv'].head()}")

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

        # Asset table data (tournament-style)
        asset_table = []
        rs_last = rs_data_filtered.iloc[-1]
        ranks = rs_last.rank(ascending=False, pct=False, method='min')
        for name, df in assets_data.items():
            tpi_last = df["TPI"].iloc[-1]
            signal = "Long" if tpi_last > 0 else "Short" if tpi_last < 0 else "Neutral"
            asset_equity = (1 + df["close"].pct_change()).cumprod().fillna(1)
            asset_returns = (asset_equity.iloc[-1] - 1) * 100
            asset_maxdd = ((asset_equity / asset_equity.cummax()) - 1).min() * 100
            asset_table.append({
                "name": name,
                "rank": int(ranks[name]) if not pd.isna(ranks[name]) else "N/A",
                "signal": signal,
                "returns": round(asset_returns, 2),
                "max_dd": round(asset_maxdd, 2)
            })

        # Top 3 (including GOLD if in top)
        top_assets = rs_last[~np.isneginf(rs_last)].sort_values(ascending=False).index[:3].tolist()

        # Current allocation
        current_alloc = alloc_hist_filtered[-1] if alloc_hist_filtered else "CASH"

        # Metrics table
        metrics_table = {
            **metrics_filtered,
            "PositionChanges": switches_filtered,
            "EquityMaxDD": metrics_filtered["MaxDD"],
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
async def rebalance(used_assets: int = 4, use_gold: bool = True, timeframe: str = "1d", limit: int = 500):
    try:
        assets = dict(list(ALL_ASSETS.items())[:used_assets])
        assets_data = {}
        for name, info in assets.items():
            market_data = fetch_market_data(info["symbol"], timeframe, limit)
            ohlcv = market_data["ohlcv"]
            assets_data[name] = compute_indicators(ohlcv)

        gold_market = fetch_market_data(ALL_ASSETS["GOLD"]["symbol"], timeframe, limit)
        gold_data = compute_indicators(gold_market["ohlcv"])

        rs_data = compute_relative_strength(assets_data, filtered=True)
        equity_filtered, alloc_hist, switches = rotate_equity(rs_data, assets_data, gold_data, use_gold=use_gold)

        rs_last = rs_data.iloc[-1]
        top_assets = rs_last[~np.isneginf(rs_last)].sort_values(ascending=False).index[:3].tolist()
        current_alloc = alloc_hist[-1]

        asset_table = []
        ranks = rs_last.rank(ascending=False, pct=False, method='min')
        for name, df in assets_data.items():
            tpi_last = df["TPI"].iloc[-1]
            signal = "Long" if tpi_last > 0 else "Short" if tpi_last < 0 else "Neutral"
            asset_returns = df["close"].pct_change().sum() * 100
            asset_maxdd = 0
            asset_table.append({
                "name": name,
                "rank": int(ranks[name]) if not pd.isna(ranks[name]) else "N/A",
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

# Scheduler for 1d rebalancing
def scheduled_rebalance():
    fetch_and_store_raw_data()
    response = backtest()
    print("Scheduled rebalance complete:", response["metrics"])

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_rebalance, 'interval', days=1)  # Changed to 1 day for 1d timeframe
scheduler.start()