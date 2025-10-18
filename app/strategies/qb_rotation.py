"""
QB Strategy Rotation Logic - FIXED FOR REPAINTING
Uses signal-based allocation with proper lagging
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def rotate_equity_qb(assets: dict, gold_df: pd.DataFrame, start_date: str = None, use_gold: bool = True) -> tuple:
    """
    QB-specific rotation with NO REPAINTING
    
    Key fix: We hold an asset and earn its return, THEN decide what to hold next day
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
    
    if common_index.empty:
        raise ValueError("No overlapping dates across assets")
    
    # Prepare data
    qb_signals = {}
    momentum_data = {}
    close_prices = {}
    
    for name, df in assets.items():
        df_reindexed = df.reindex(common_index)
        qb_signals[name] = df_reindexed['QB'].fillna(0)
        momentum_data[name] = df_reindexed['Momentum'].fillna(0)
        close_prices[name] = df_reindexed['close'].fillna(method='ffill')
    
    # Add GOLD
    if gold_df is not None and not gold_df.empty:
        gold_reindexed = gold_df.reindex(common_index)
        gold_qb = gold_reindexed['QB'].fillna(0)
        gold_close = gold_reindexed['close'].fillna(method='ffill')
    else:
        gold_qb = pd.Series(0, index=common_index)
        gold_close = pd.Series(1, index=common_index)
    
    # Convert to DataFrames
    qb_df = pd.DataFrame(qb_signals)
    momentum_df = pd.DataFrame(momentum_data)
    close_df = pd.DataFrame(close_prices)
    
    # Initialize tracking
    equity = pd.Series(1.0, index=common_index, name='Equity')
    holding = None  # What we're CURRENTLY holding
    alloc_hist = []
    switches = 0
    
    for i in range(len(common_index)):
        date = common_index[i]
        
        # STEP 1: Calculate return from what we HELD during this period
        if i == 0:
            period_return = 0  # No return on first day
        elif holding == 'CASH':
            period_return = 0
        elif holding == 'GOLD':
            period_return = (gold_close.iloc[i] - gold_close.iloc[i-1]) / gold_close.iloc[i-1]
        elif holding in close_df.columns:
            period_return = (close_df.iloc[i][holding] - close_df.iloc[i-1][holding]) / close_df.iloc[i-1][holding]
        else:
            period_return = 0
        
        # Update equity based on what we HELD
        if i > 0:
            equity.iloc[i] = equity.iloc[i-1] * (1 + period_return)
        
        # Debug first few days
        if i < 5:
            print(f"[DEBUG QB] Day {i} ({date}): HELD {holding}, Return={period_return*100:.2f}%, Equity={equity.iloc[i]:.4f}")
        
        # STEP 2: Decide what to hold TOMORROW based on TODAY'S signal
        current_qb = qb_df.iloc[i]
        current_momentum = momentum_df.iloc[i]
        bullish_assets = current_qb[current_qb == 1]
        
        if not bullish_assets.empty:
            # Among bullish assets, pick strongest by momentum
            bullish_momentum = current_momentum[bullish_assets.index]
            next_holding = bullish_momentum.idxmax()
        elif use_gold and gold_qb.iloc[i] == 1:
            next_holding = 'GOLD'
        else:
            next_holding = 'CASH'
        
        # Track switches
        if next_holding != holding:
            switches += 1
            if i < 10:
                logger.info(f"Switch at {date}: {holding} -> {next_holding}")
        
        if i < 5:
            print(f"[DEBUG QB] Signal says hold {next_holding} tomorrow (QB={current_qb.to_dict()})")
        
        # Update holding for next period
        holding = next_holding
        alloc_hist.append(holding)
    
    print(f"[DEBUG QB] Total switches: {switches}")
    print(f"[DEBUG QB] Final allocation: {alloc_hist[-1] if alloc_hist else 'None'}")
    print(f"[DEBUG QB] Final equity: {equity.iloc[-1]:.4f}")
    
    return equity.ffill(), alloc_hist, switches

def compute_metrics(equity: pd.Series) -> dict:
    """Same metrics calculation"""
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