"""
QB Strategy Rotation Logic - CORRECTED TO MATCH PINESCRIPT
The QB signal determines if we HOLD an asset, not just rank it.
Only switch when current holding's QB turns bearish OR another asset becomes significantly stronger.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def rotate_equity_qb(assets: dict, gold_df: pd.DataFrame, start_date: str = None, use_gold: bool = True, 
                     momentum_threshold: float = 0.10) -> tuple:
    """
    QB-specific rotation - CORRECTED LOGIC
    
    Key principles:
    1. QB=1 means "this asset is in a bullish state" - we can hold it
    2. QB=-1 means "this asset is bearish" - we should NOT hold it
    3. We only switch if:
       a) Current holding becomes bearish (QB=-1), OR
       b) Another bullish asset has significantly better momentum (>10% edge by default)
    
    This reduces switching while still rotating to stronger assets when needed.
    
    Args:
        momentum_threshold: How much better momentum (as decimal) needed to trigger a switch
                          e.g., 0.10 = 10% better momentum required
    """
    # Get common index across all assets
    all_indices = [df.index for df in assets.values()]
    if gold_df is not None and not gold_df.empty:
        all_indices.append(gold_df.index)
    
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.intersection(idx)
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        common_index = common_index[common_index >= start_dt]
        print(f"[DEBUG QB] Filtering from start_date: {start_dt} -> {len(common_index)} days")
    
    if common_index.empty:
        raise ValueError("No overlapping dates across assets")
    
    # Log the actual date range being used
    print(f"[DEBUG QB] Backtest period: {common_index[0].date()} to {common_index[-1].date()} ({len(common_index)} days)")
    print(f"[DEBUG QB] Assets in rotation: {list(assets.keys())}")
    print(f"[DEBUG QB] Momentum threshold for switching: {momentum_threshold*100:.1f}%")
    
    # Prepare data
    qb_signals = {}
    momentum_data = {}
    close_prices = {}
    
    for name, df in assets.items():
        df_reindexed = df.reindex(common_index)
        qb_signals[name] = df_reindexed['QB'].fillna(0)
        momentum_data[name] = df_reindexed['Momentum'].fillna(0)
        close_prices[name] = df_reindexed['close'].ffill()
    
    # Add GOLD
    if gold_df is not None and not gold_df.empty:
        gold_reindexed = gold_df.reindex(common_index)
        gold_qb = gold_reindexed['QB'].fillna(0)
        gold_close = gold_reindexed['close'].ffill()
    else:
        gold_qb = pd.Series(0, index=common_index)
        gold_close = pd.Series(1, index=common_index)
    
    # Convert to DataFrames
    qb_df = pd.DataFrame(qb_signals)
    momentum_df = pd.DataFrame(momentum_data)
    close_df = pd.DataFrame(close_prices)
    
    # Initialize tracking
    equity = pd.Series(1.0, index=common_index, name='Equity')
    holding = 'CASH'  # Start with CASH
    alloc_hist = []
    switches = 0
    
    for i in range(len(common_index)):
        date = common_index[i]
        
        # STEP 1: Record what we're HOLDING at start of this day
        alloc_hist.append(holding)
        
        # STEP 2: Calculate return from what we HELD during this period
        if i == 0:
            period_return = 0
        elif holding == 'CASH':
            period_return = 0
        elif holding == 'GOLD':
            period_return = (gold_close.iloc[i] - gold_close.iloc[i-1]) / gold_close.iloc[i-1]
        elif holding in close_df.columns:
            period_return = (close_df.iloc[i][holding] - close_df.iloc[i-1][holding]) / close_df.iloc[i-1][holding]
        else:
            period_return = 0
        
        # Update equity
        if i > 0:
            equity.iloc[i] = equity.iloc[i-1] * (1 + period_return)
        
        # Debug first few days
        if i < 5:
            print(f"[DEBUG QB] Day {i} ({date.date()}): HOLDING={holding}, Return={period_return*100:.2f}%, Equity={equity.iloc[i]:.4f}")
        
        # STEP 3: Decide what to hold TOMORROW based on TODAY'S signals
        current_qb = qb_df.iloc[i]
        current_momentum = momentum_df.iloc[i]
        
        # Find all assets with bullish QB signal
        bullish_assets = current_qb[current_qb == 1]
        
        # DECISION LOGIC (CORRECTED):
        if bullish_assets.empty:
            # No bullish crypto assets -> check GOLD or go CASH
            if use_gold and gold_qb.iloc[i] == 1:
                next_holding = 'GOLD'
            else:
                next_holding = 'CASH'
        else:
            # We have bullish assets. Now decide if we should switch.
            
            # Case A: Current holding is bearish -> MUST switch
            if holding in current_qb.index and current_qb[holding] != 1:
                # Current asset turned bearish, pick best bullish asset
                bullish_momentum = current_momentum[bullish_assets.index]
                next_holding = bullish_momentum.idxmax()
                if i < 10:
                    print(f"[DEBUG QB]   FORCED SWITCH: {holding} turned bearish (QB={current_qb[holding]})")
            
            # Case B: Current holding is still bullish -> only switch if much better option exists
            elif holding in bullish_assets.index:
                current_momentum_val = current_momentum[holding]
                bullish_momentum = current_momentum[bullish_assets.index]
                best_asset = bullish_momentum.idxmax()
                best_momentum_val = bullish_momentum[best_asset]
                
                # Only switch if the best asset has significantly better momentum
                momentum_edge = best_momentum_val - current_momentum_val
                
                if momentum_edge > momentum_threshold:
                    next_holding = best_asset
                    if i < 10:
                        print(f"[DEBUG QB]   MOMENTUM SWITCH: {best_asset} has {momentum_edge*100:.1f}% edge over {holding}")
                else:
                    next_holding = holding  # Stay put
                    if i < 5:
                        print(f"[DEBUG QB]   HOLDING: {holding} still bullish, edge={momentum_edge*100:.1f}% < threshold")
            
            # Case C: We're in CASH/GOLD but now have bullish assets -> enter best one
            else:
                bullish_momentum = current_momentum[bullish_assets.index]
                next_holding = bullish_momentum.idxmax()
                if i < 10:
                    print(f"[DEBUG QB]   ENTERING: {next_holding} (from {holding})")
        
        # Track switches
        if next_holding != holding:
            switches += 1
            if i < 10 or i > len(common_index) - 5:
                print(f"[DEBUG QB] Switch #{switches} at {date.date()}: {holding} -> {next_holding}")
        
        if i < 5:
            qb_dict = {k: v for k, v in current_qb.items()}
            print(f"[DEBUG QB]   Signals: QB={qb_dict}")
        
        # Update holding for tomorrow
        holding = next_holding
    
    print(f"\n[DEBUG QB] ===== FINAL SUMMARY =====")
    print(f"[DEBUG QB] Total switches: {switches}")
    print(f"[DEBUG QB] Switch rate: {switches/len(common_index)*100:.1f}% of days")
    print(f"[DEBUG QB] Final holding (what we're holding NOW): {holding}")
    print(f"[DEBUG QB] alloc_hist[-1] (last recorded): {alloc_hist[-1] if alloc_hist else 'None'}")
    print(f"[DEBUG QB] Last 10 holdings: {alloc_hist[-10:]}")
    print(f"[DEBUG QB] Final equity: {equity.iloc[-1]:.4f}")
    print(f"[DEBUG QB] Total return: {(equity.iloc[-1] - 1) * 100:.2f}%")
    
    # Analyze allocation distribution
    from collections import Counter
    alloc_counts = Counter(alloc_hist)
    print(f"\n[DEBUG QB] Allocation distribution:")
    for asset, count in alloc_counts.most_common():
        pct = count / len(alloc_hist) * 100
        print(f"  {asset}: {count} days ({pct:.1f}%)")
    
    return equity.ffill(), alloc_hist, switches

def compute_metrics(equity: pd.Series) -> dict:
    """Calculate performance metrics"""
    days = len(equity)
    if days < 2:
        return {"CAGR": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "Omega": 0.0, "MaxDD": 0.0, "NetProfit": 0.0, "Days": days}

    final_equity = equity.iloc[-1]
    cagr = (final_equity ** (365 / days)) - 1 if final_equity > 0 else -1
    net_profit = (final_equity - 1) * 100

    returns = equity.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return {"CAGR": cagr, "Sharpe": 0.0, "Sortino": 0.0, "Omega": 0.0, "MaxDD": 0.0, "NetProfit": net_profit, "Days": days}

    sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
    downside_std = returns[returns < 0].std() if not returns[returns < 0].empty else 1e-6
    sortino = (returns.mean() / downside_std) * np.sqrt(365)
    drawdowns = (equity / equity.cummax()) - 1
    maxdd = drawdowns.min() * 100

    above = returns[returns > 0].sum()
    below = abs(returns[returns <= 0].sum())
    omega = above / below if below > 0 else float('inf')

    return {"CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "Omega": omega, "MaxDD": maxdd, "NetProfit": net_profit, "Days": days}