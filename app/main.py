from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data import fetch_market_data
from app.strategy_manager import compute_indicators, uses_relative_strength
from app.strategies.universal_rs import compute_relative_strength, compute_metrics
from app.tournament import run_tournament
from app.db.database import SessionLocal, BacktestRun, OHLCVData
from app.equity import compute_equity
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
import os
import uvicorn
from sqlalchemy.exc import OperationalError
import time

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
    ("SUIUSDT", "sui"),
    ("BNBUSDT", "bnb"),
    ("PAXGUSDT", "pax-gold")
]

def store_single_asset(db, name, timeframe: str = "1d", limit: int = 700, start_date: str = None, end_date: str = None):
    """Store OHLCV data - SIMPLIFIED"""
    try:
        instrument = name.replace("USDT", "")
        
        print(f"[STORE] {instrument}: start={start_date}, end={end_date}, limit={limit}")
        
        # Use simple fetch_market_data with direct parameters
        market_data = fetch_market_data(name, timeframe, limit=limit, start=start_date, end=end_date)
        ohlcv = market_data["ohlcv"]
        
        print(f"Received {len(ohlcv)} rows for {instrument}")
        
        if not ohlcv.empty:
            stored_count = 0
            skipped_count = 0
            
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
                    stored_count += 1
                else:
                    skipped_count += 1
            
            db.commit()
            print(f"Committed {stored_count} new, skipped {skipped_count} duplicates")
            
            return {
                "success": True, 
                "stored": stored_count, 
                "skipped": skipped_count,
                "total": len(ohlcv),
                "date_range": f"{ohlcv.index.min().date()} to {ohlcv.index.max().date()}"
            }
        else:
            return {"success": False, "error": "No data returned"}
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return {"success": False, "error": str(e)}

def query_neon_with_retry(instrument, max_retries=3, delay=5):
    """Query Neon with retries for SSL issues."""
    db = SessionLocal()
    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] Querying Neon for {instrument}, attempt {attempt + 1}")
            query = db.query(OHLCVData).filter(OHLCVData.instrument == instrument).order_by(OHLCVData.timestamp).all()
            if query:
                df = pd.DataFrame([(q.timestamp, q.open, q.high, q.low, q.close, q.volume) for q in query],
                                columns=["timestamp", "open", "high", "low", "close", "volume"])
                df.set_index("timestamp", inplace=True)
                print(f"[DEBUG] Successfully queried {len(df)} rows for {instrument}")
                return df
            print(f"[WARNING] No data found for {instrument}")
            return pd.DataFrame()
        except OperationalError as e:
            print(f"[ERROR] Neon query failed for {instrument} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            return pd.DataFrame()
        finally:
            db.close()

@app.get("/")
async def root():
    """Health check endpoint to trigger logs."""
    print("Server is running!")
    return {"status": "healthy"}

@app.get("/store-btc")
async def store_btc(limit: int = 365, start_date: str = None, end_date: str = None):
    """
    Store BTC data. 
    Examples:
      /store-btc                                           # Store latest missing data
      /store-btc?start_date=2023-01-01&end_date=2023-12-31 # Backfill 2023 (auto limit=365)
      /store-btc?start_date=2023-01-01&limit=500           # Backfill with custom limit
    
    NOTE: When using start_date, limit defaults to 365. Override if needed.
    """
    db = SessionLocal()
    
    # If start_date is provided but limit is default, increase limit for backfill
    if start_date and limit == 365:
        # Calculate days between dates if end_date provided
        if end_date:
            from datetime import datetime
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            limit = min((end - start).days + 1, 500)  # Cap at 500
            print(f"[AUTO-LIMIT] Calculated limit={limit} for date range")
    
    result = store_single_asset(db, "BTCUSDT", limit=limit, start_date=start_date, end_date=end_date)
    db.close()
    return result

@app.get("/store-eth")
async def store_eth(limit: int = 365, start_date: str = None, end_date: str = None):
    """Store ETH data"""
    db = SessionLocal()
    if start_date and end_date and limit == 365:
        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        limit = min((end - start).days + 1, 500)
    result = store_single_asset(db, "ETHUSDT", limit=limit, start_date=start_date, end_date=end_date)
    db.close()
    return result

@app.get("/store-sol")
async def store_sol(limit: int = 365, start_date: str = None, end_date: str = None):
    """Store SOL data"""
    db = SessionLocal()
    if start_date and end_date and limit == 365:
        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        limit = min((end - start).days + 1, 500)
    result = store_single_asset(db, "SOLUSDT", limit=limit, start_date=start_date, end_date=end_date)
    db.close()
    return result

@app.get("/store-bnb")
async def store_bnb(limit: int = 365, start_date: str = None, end_date: str = None):
    """Store BNB data"""
    db = SessionLocal()
    if start_date and end_date and limit == 365:
        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        limit = min((end - start).days + 1, 500)
    result = store_single_asset(db, "BNBUSDT", limit=limit, start_date=start_date, end_date=end_date)
    db.close()
    return result

@app.get("/store-paxg")
async def store_paxg(limit: int = 365, start_date: str = None, end_date: str = None):
    """Store PAXG data"""
    db = SessionLocal()
    if start_date and end_date and limit == 365:
        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        limit = min((end - start).days + 1, 500)
    result = store_single_asset(db, "PAXGUSDT", limit=limit, start_date=start_date, end_date=end_date)
    db.close()
    return result

@app.get("/store-all")
async def store_all(limit: int = 1, start_date: str = None):
    """
    Store data for all assets at once
    WARNING: May timeout on large limits. Use individual endpoints for backfilling.
    
    Examples:
      /store-all                           # Store today for all
      /store-all?limit=30                  # Store last 30 days for all
    """
    db = SessionLocal()
    results = {}
    
    for symbol, _ in ALL_ASSETS:
        print(f"[STORE-ALL] Processing {symbol}...")
        result = store_single_asset(db, symbol, limit=limit, start_date=start_date)
        results[symbol] = result
    
    db.close()
    return {"status": "completed", "results": results}

@app.get("/backtest")
async def backtest(start_date: str = "2023-01-01", limit: int = 700, used_assets: int = 3,
                   use_gold: bool = True, benchmark: str = "BTC", timeframe: str = "1d"):
    try:
        print("[DEBUG] Starting backtest")
        
        # CRITICAL FIX: Always load BNB and GOLD, regardless of used_assets
        # First load the rotation candidates
        assets_data = {}
        rotation_assets = []
        
        for symbol, _ in ALL_ASSETS[:used_assets + 1]:  # Load used_assets + 1 for tournament
            instrument = symbol.replace("USDT", "")
            df = query_neon_with_retry(instrument)
            if not df.empty:
                assets_data[instrument] = compute_indicators(df)
                rotation_assets.append((symbol, _))
                print(f"[DEBUG] Loaded rotation asset {instrument}: {len(df)} rows")
        
        # CRITICAL: Now load BNB if not already loaded
        if "BNB" not in assets_data:
            print("[DEBUG] Loading BNB from Neon...")
            bnb_df = query_neon_with_retry("BNB")
            if not bnb_df.empty:
                assets_data["BNB"] = compute_indicators(bnb_df)
                print(f"[DEBUG] Loaded BNB: {len(bnb_df)} rows")
            else:
                print("[WARNING] BNB data not found in Neon!")
        
        # CRITICAL: Always load GOLD (PAXG) from Neon
        print("[DEBUG] Loading GOLD (PAXG) from Neon...")
        gold_df = query_neon_with_retry("PAXG")
        if not gold_df.empty:
            gold_data = compute_indicators(gold_df)
            print(f"[DEBUG] Loaded GOLD from Neon: {len(gold_df)} rows")
        else:
            print("[WARNING] GOLD data not found in Neon! Fetching from API...")
            market_data = fetch_market_data("PAXGUSDT", timeframe, limit)
            gold_data = compute_indicators(market_data["ohlcv"])
            print(f"[DEBUG] Fetched GOLD from API: {len(market_data['ohlcv'])} rows")

        if not assets_data:
            return {"error": "No data available in Neon"}

        # Find common start date (earliest date where we have data for major assets)
        common_start = pd.to_datetime(start_date)
        
        # Only consider BTC/ETH/SOL/BNB for common start (not SUI which starts later in May 2023)
        major_assets = ["BTC", "ETH", "SOL", "BNB"]
        asset_start_dates = {}
        
        for asset_name in major_assets:
            if asset_name in assets_data and not assets_data[asset_name].empty:
                asset_start = assets_data[asset_name].index.min()
                asset_start_dates[asset_name] = asset_start
                print(f"[DEBUG] {asset_name} data starts: {asset_start.date()}")
        
        # Use the requested start_date or earliest available
        if asset_start_dates:
            earliest_available = max(asset_start_dates.values())
            if earliest_available > common_start:
                print(f"[WARNING] Requested {common_start.date()} but earliest data is {earliest_available.date()}")
                common_start = earliest_available
        
        print(f"[DEBUG] Using common start date: {common_start.date()}")
        print(f"[DEBUG] This ensures all major assets (BTC/ETH/SOL/BNB) have data from this point")

        # Run tournament to get top assets
        print("[DEBUG] Running tournament...")
        print(f"[DEBUG] Processing {len(rotation_assets)} assets: {[s for s, _ in rotation_assets]}")
        tournament_results = run_tournament(rotation_assets, assets_data=assets_data)
        print(f"[DEBUG] Tournament results ({len(tournament_results)} assets): {[r['symbol'] for r in tournament_results]}")

        top_assets = [result["symbol"].replace("USDT", "") for result in tournament_results[:used_assets]]
        print(f"[DEBUG] Top {used_assets} assets for rotation: {top_assets}")
        
        # Align all assets to common start date
        aligned_assets = {}
        for asset in top_assets:
            if asset in assets_data:
                aligned_assets[asset] = assets_data[asset][assets_data[asset].index >= common_start]
                print(f"[DEBUG] Aligned {asset}: {len(aligned_assets[asset])} rows from {aligned_assets[asset].index.min().date()} to {aligned_assets[asset].index.max().date()}")
        
        # Show if BNB or GOLD could be added to rotation pool
        if "BNB" in assets_data and "BNB" not in aligned_assets:
            print(f"[INFO] BNB available but not in top {used_assets} assets")
        
        # Align gold data too
        if not gold_data.empty:
            gold_data_aligned = gold_data[gold_data.index >= common_start]
            print(f"[DEBUG] Aligned GOLD: {len(gold_data_aligned)} rows from {gold_data_aligned.index.min().date()} to {gold_data_aligned.index.max().date()}")
            print(f"[DEBUG] GOLD will be used as defensive position when use_gold={use_gold}")
        else:
            gold_data_aligned = pd.DataFrame()
            print("[WARNING] No GOLD data available! Strategy will use CASH instead.")
        
        # Get rotation function based on active strategy
        use_rs = uses_relative_strength()
        
        if use_rs:
            # Momentum-based rotation (simple strategy)
            from app.strategies.universal_rs import rotate_equity
            rs_data = compute_relative_strength(aligned_assets, filtered=True)
            print(f"[DEBUG] RS data shape: {rs_data.shape if not rs_data.empty else 'Empty'}")
            
            equity_filtered, alloc_hist_filtered, switches_filtered = rotate_equity(
                rs_data, aligned_assets, gold_data_aligned, start_date=str(common_start.date()), use_gold=use_gold
            )
        else:
            # Signal-based rotation (QB strategy)
            from app.strategies.qb_rotation import rotate_equity_qb
            print(f"[DEBUG] Using QB signal-based rotation")
            equity_filtered, alloc_hist_filtered, switches_filtered = rotate_equity_qb(
                aligned_assets, gold_data_aligned, start_date=str(common_start.date()), use_gold=use_gold
            )
        
        print(f"[DEBUG] Equity filtered length: {len(equity_filtered)}")

        metrics_filtered = compute_metrics(equity_filtered)
        print(f"[DEBUG] Metrics computed: {metrics_filtered}")

        # The equity_filtered IS the strategy equity - don't recalculate it!
        strategy_equity = equity_filtered.copy()
        
        # Calculate strategy metrics from the rotation equity
        strategy_returns = strategy_equity.pct_change().dropna()
        if not strategy_returns.empty and strategy_returns.std() != 0:
            strategy_sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365)
        else:
            strategy_sharpe = 0
        
        strategy_drawdowns = (strategy_equity / strategy_equity.cummax()) - 1
        strategy_max_dd = strategy_drawdowns.min() * 100
        strategy_total_return = (strategy_equity.iloc[-1] - 1) * 100
        
        strategy_metrics = {
            'total_return_%': strategy_total_return,
            'buy_and_hold_%': 0,
            'max_drawdown_%': strategy_max_dd,
            'final_equity': strategy_equity.iloc[-1],
            'sharpe': strategy_sharpe
        }
        
        print(f"[DEBUG] Strategy metrics: {strategy_metrics}")

        # Benchmark equity - simple buy and hold FROM COMMON START
        if benchmark in assets_data:
            benchmark_df = assets_data[benchmark][assets_data[benchmark].index >= common_start]
            
            if not benchmark_df.empty:
                # Calculate buy & hold properly: normalize to start at 1.0
                first_close = benchmark_df['close'].iloc[0]
                benchmark_equity = benchmark_df['close'] / first_close
                
                # Align to strategy equity index
                benchmark_equity = benchmark_equity.reindex(equity_filtered.index, method='ffill')
                
                print(f"[DEBUG] Benchmark ({benchmark}): {first_close:.2f} -> {benchmark_df['close'].iloc[-1]:.2f}")
            else:
                benchmark_equity = pd.Series(1.0, index=equity_filtered.index)
        else:
            benchmark_equity = pd.Series(1.0, index=equity_filtered.index)
        
        benchmark_total_return = (benchmark_equity.iloc[-1] - 1) * 100
        strategy_metrics['buy_and_hold_%'] = benchmark_total_return
        
        print(f"[DEBUG] Benchmark return: {benchmark_total_return:.2f}%")

        # Asset table equity calculation - use common start date
        asset_table = tournament_results[:used_assets + 1] if tournament_results else []
        for asset in asset_table:
            symbol = asset["symbol"].replace("USDT", "")
            if symbol in assets_data:
                asset_df = assets_data[symbol].copy()
                asset_df = asset_df[asset_df.index >= common_start]
                
                if len(asset_df) > 0:
                    first_close = asset_df["close"].iloc[0]
                    last_close = asset_df["close"].iloc[-1]
                    
                    # Return as percentage gain
                    return_pct = ((last_close - first_close) / first_close) * 100
                    asset["equity"] = return_pct
                    
                    print(f"[DEBUG] {symbol}: {first_close:.2f} -> {last_close:.2f} = {return_pct:.2f}%")
                else:
                    asset["equity"] = 0.0
            else:
                asset["equity"] = 0.0

        top3 = [result["symbol"] for result in tournament_results[:3]] if tournament_results else []
        
        # Extract current allocation
        current_alloc = str(alloc_hist_filtered[-1]) if len(alloc_hist_filtered) > 0 else "CASH"
        print(f"[DEBUG] Current allocation: {current_alloc}")

        metrics_table = {
            **metrics_filtered,
            "PositionChanges": switches_filtered,
            "EquityMaxDD": strategy_metrics["max_drawdown_%"],
            "NetProfit": strategy_metrics["total_return_%"]
        }

        # Convert all numpy types to native Python types
        metrics_table_clean = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
            for k, v in metrics_table.items()
        }
        
        response = {
            "metrics": metrics_table_clean,
            "final_equity_filtered": float(strategy_equity.iloc[-1]) if not strategy_equity.empty else 0,
            "switches": int(switches_filtered),
            "current_allocation": current_alloc,
            "top3": top3,
            "asset_table": asset_table,
            "equity_curve_filtered": {str(k): float(v) for k, v in strategy_equity.items()} if not strategy_equity.empty else {},
            "buy_hold_equity": {str(k): float(v) for k, v in benchmark_equity.items()} if not benchmark_equity.empty else {}
        }

        # Store in database
        end_date = equity_filtered.index[-1].to_pydatetime() if not equity_filtered.empty else datetime.now()

        db = SessionLocal()
        
        equity_dict = {str(k): float(v) for k, v in strategy_equity.items()}
        alloc_dict = {str(k): str(v) for k, v in zip(equity_filtered.index, alloc_hist_filtered)}
        
        metrics_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
            for k, v in metrics_table.items()
        }
        
        run = BacktestRun(
            start_date=pd.to_datetime(start_date).to_pydatetime(),
            end_date=end_date,
            metrics=metrics_serializable,
            equity_curve=equity_dict,
            alloc_hist=alloc_dict,
            switches=int(switches_filtered)
        )
        db.add(run)
        db.commit()
        db.close()

        return response
    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/rebalance")
async def rebalance(used_assets: int = 3, use_gold: bool = True, timeframe: str = "1d", limit: int = 1):
    try:
        print("[DEBUG] Starting rebalance")
        
        # Load rotation candidates
        assets_data = {}
        rotation_assets = []
        
        for symbol, _ in ALL_ASSETS[:used_assets + 1]:
            instrument = symbol.replace("USDT", "")
            df = query_neon_with_retry(instrument)
            if not df.empty:
                assets_data[instrument] = compute_indicators(df)
                rotation_assets.append((symbol, _))
                print(f"[DEBUG] Loaded {instrument}: {len(df)} rows")
        
        # Always load GOLD
        print("[DEBUG] Loading GOLD (PAXG) from Neon...")
        gold_df = query_neon_with_retry("PAXG")
        if not gold_df.empty:
            gold_data = compute_indicators(gold_df)
            print(f"[DEBUG] Loaded GOLD: {len(gold_df)} rows")
        else:
            print("[WARNING] Fetching GOLD from API...")
            market_data = fetch_market_data("PAXGUSDT", timeframe, limit=700)
            gold_data = compute_indicators(market_data["ohlcv"])

        if not assets_data:
            return {"error": "No data available in Neon"}

        # Run tournament
        print("[DEBUG] Running tournament...")
        tournament_results = run_tournament(rotation_assets, assets_data=assets_data)
        print(f"[DEBUG] Tournament results: {[r['symbol'] for r in tournament_results]}")

        top_assets = [result["symbol"].replace("USDT", "") for result in tournament_results[:used_assets]]
        print(f"[DEBUG] Top assets: {top_assets}")
        
        # Get rotation function
        use_rs = uses_relative_strength()
        
        if use_rs:
            from app.strategies.universal_rs import rotate_equity
            rs_data = compute_relative_strength({k: assets_data[k] for k in top_assets if k in assets_data}, filtered=True)
            
            equity_filtered, alloc_hist, switches = rotate_equity(
                rs_data, {k: assets_data[k] for k in top_assets if k in assets_data}, gold_data, use_gold=use_gold
            )
        else:
            from app.strategies.qb_rotation import rotate_equity_qb
            print(f"[DEBUG] Using QB rotation")
            equity_filtered, alloc_hist, switches = rotate_equity_qb(
                {k: assets_data[k] for k in top_assets if k in assets_data}, gold_data, use_gold=use_gold
            )
        
        rebalance_equity = equity_filtered.copy()
        
        # Calculate metrics
        rebalance_returns = rebalance_equity.pct_change().dropna()
        if not rebalance_returns.empty and rebalance_returns.std() != 0:
            rebalance_sharpe = (rebalance_returns.mean() / rebalance_returns.std()) * np.sqrt(365)
        else:
            rebalance_sharpe = 0
        
        rebalance_drawdowns = (rebalance_equity / rebalance_equity.cummax()) - 1
        rebalance_max_dd = rebalance_drawdowns.min() * 100
        rebalance_total_return = (rebalance_equity.iloc[-1] - 1) * 100

        # Asset table
        common_start = None
        for asset_name, asset_df in assets_data.items():
            if not asset_df.empty:
                asset_start = asset_df.index.min()
                if common_start is None or asset_start > common_start:
                    common_start = asset_start

        asset_table = tournament_results[:used_assets + 1] if tournament_results else []
        for asset in asset_table:
            symbol = asset["symbol"].replace("USDT", "")
            if symbol in assets_data:
                asset_df = assets_data[symbol].copy()
                
                if common_start is not None:
                    asset_df = asset_df[asset_df.index >= common_start]
                
                if len(asset_df) > 0:
                    first_close = asset_df["close"].iloc[0]
                    last_close = asset_df["close"].iloc[-1]
                    return_pct = ((last_close - first_close) / first_close) * 100
                    asset["equity"] = return_pct
                else:
                    asset["equity"] = 0.0
            else:
                asset["equity"] = 0.0

        top3 = [result["symbol"] for result in tournament_results[:3]] if tournament_results else []
        current_alloc = str(alloc_hist[-1]) if len(alloc_hist) > 0 else "CASH"
        print(f"[DEBUG REBALANCE] Current allocation: {current_alloc}")

        return {
            "current_allocation": current_alloc,
            "top3": top3,
            "asset_table": asset_table,
            "latest_equity": float(rebalance_equity.iloc[-1]) if not rebalance_equity.empty else 0,
            "switches": int(switches)
        }
    except Exception as e:
        print(f"[ERROR] Rebalance failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Scheduler for daily updates
def daily_update():
    """Runs daily at 00:05 UTC"""
    print(f"[SCHEDULER] Daily update started at {datetime.utcnow()} UTC")
    db = SessionLocal()
    
    try:
        for symbol, _ in ALL_ASSETS:
            print(f"[SCHEDULER] Fetching new data for {symbol}...")
            result = store_single_asset(db, symbol, timeframe="1d", limit=1)
            
            if result.get("success"):
                stored = result.get("stored", 0)
                if stored > 0:
                    print(f"[SCHEDULER] ✓ {symbol} updated ({stored} rows)")
                else:
                    print(f"[SCHEDULER] ○ {symbol} already up to date")
            else:
                print(f"[SCHEDULER] ✗ {symbol} failed: {result.get('error')}")
        
        print(f"[SCHEDULER] Completed at {datetime.utcnow()} UTC")
        
    except Exception as e:
        print(f"[SCHEDULER ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

scheduler = BackgroundScheduler(timezone='UTC')
scheduler.add_job(
    daily_update, 
    'cron', 
    hour=0, 
    minute=5,
    timezone='UTC'
)
scheduler.start()

print("[STARTUP] Scheduler initialized. Daily updates at 00:05 UTC")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)