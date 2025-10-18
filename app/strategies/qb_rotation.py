"""
QB Strategy Rotation Logic
Different from momentum rotation - uses signal-based allocation
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def rotate_equity_qb(assets: dict, gold_df: pd.DataFrame, start_date: str = None, use_gold: bool = True) -> tuple:
    """
    QB-specific rotation: Uses QB signal instead of momentum ranking
    
    Logic:
    - Find assets with QB = 1 (bullish signal)
    - Rank those by score/momentum
    - Allocate to strongest bullish asset
    - If no bullish assets, go to GOLD if its QB = 1, else CASH
    
    Args:
        assets: dict of {name: df} with QB column
        gold_df: DataFrame with gold data and QB signal
        start_date: backtest start date
        use_gold: whether to use gold as safe haven
    
    Returns:
        equity, alloc_hist, switches
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
        qb_signals[name] = df_reindexed['QB'].fillna(0)  # Already shifted in indicator calculation
        momentum_data[name] = df_reindexed['Momentum'].fillna(0)
        close_prices[name] = df_reindexed['close'].fillna(method='ffill')  # Store actual close prices
    
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
    current_alloc = None
    alloc_hist = []
    switches = 0
    previous_close = {}  # Track previous day's close for each asset
    
    for i in range(len(common_index)):
        date = common_index[i]
        
        # QB signals are already shifted in the indicator calculation
        current_qb = qb_df.iloc[i]
        current_momentum = momentum_df.iloc[i]
        
        # Find assets with bullish signal (QB = 1)
        bullish_assets = current_qb[current_qb == 1]
        
        if not bullish_assets.empty:
            # Among bullish assets, pick the one with highest momentum
            bullish_momentum = current_momentum[bullish_assets.index]
            top_asset = bullish_momentum.idxmax()
            
            if top_asset != current_alloc:
                switches += 1
                current_alloc = top_asset
                logger.info(f"Switch at {date}: {top_asset} (QB=1, Momentum={bullish_momentum[top_asset]:.4f})")
            
            # Calculate return properly: (today_close - yesterday_close) / yesterday_close
            if i > 0 and top_asset in previous_close:
                today_close = close_df.iloc[i][top_asset]
                yesterday_close = previous_close[top_asset]
                period_return = (today_close - yesterday_close) / yesterday_close if yesterday_close != 0 else 0
            else:
                period_return = 0  # First day, no return
            
            alloc = top_asset
            
        else:
            # No bullish crypto assets - check GOLD
            if use_gold and gold_qb.iloc[i] == 1:
                if current_alloc != 'GOLD':
                    switches += 1
                    current_alloc = 'GOLD'
                    logger.info(f"Switch at {date}: GOLD (No bullish crypto, GOLD QB=1)")
                
                # Calculate GOLD return
                if i > 0 and 'GOLD' in previous_close:
                    today_close = gold_close.iloc[i]
                    yesterday_close = previous_close['GOLD']
                    period_return = (today_close - yesterday_close) / yesterday_close if yesterday_close != 0 else 0
                else:
                    period_return = 0
                
                alloc = 'GOLD'
            else:
                # Go to CASH
                if current_alloc != 'CASH':
                    switches += 1
                    current_alloc = 'CASH'
                    logger.info(f"Switch at {date}: CASH (No bullish signals)")
                
                period_return = 0
                alloc = 'CASH'
        
        # Update equity
        if i > 0:
            equity.iloc[i] = equity.iloc[i-1] * (1 + period_return)
        
        alloc_hist.append(alloc)
        
        # Store today's close prices for next iteration
        previous_close = {asset: close_df.iloc[i][asset] for asset in close_df.columns}
        previous_close['GOLD'] = gold_close.iloc[i]
    
    print(f"[DEBUG QB] Total switches: {switches}")
    print(f"[DEBUG QB] Final allocation: {alloc_hist[-1] if alloc_hist else 'None'}")
    print(f"[DEBUG QB] Final equity: {equity.iloc[-1]:.4f}")
    
    return equity.ffill(), alloc_hist, switches

def compute_metrics(equity: pd.Series) -> dict:
    """Same metrics calculation as original"""
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