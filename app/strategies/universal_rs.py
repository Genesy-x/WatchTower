import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_relative_strength(assets: dict, filtered: bool = True) -> pd.DataFrame:
    """
    Compute RS ranks, optionally filtering by TPI >0.
    """
    momentum_dict = {name: df["Momentum"] for name, df in assets.items()}
    momentum_df = pd.DataFrame(momentum_dict).dropna(how='all').ffill()  # Changed to 'all' and ffill for partial overlaps

    if momentum_df.empty:
        raise ValueError("No overlapping data.")

    if filtered:
        tpi_dict = {name: df["TPI"] for name, df in assets.items()}
        tpi_df = pd.DataFrame(tpi_dict).reindex(momentum_df.index).ffill()
        momentum_df = momentum_df.where(tpi_df > 0, -np.inf)

    rs_data = momentum_df.rank(axis=1, pct=True, method='min')
    return rs_data

def rotate_equity(rs_data: pd.DataFrame, assets: dict, gold_df: pd.DataFrame, start_date: str = None, use_gold: bool = True) -> tuple:
    """
    Simulate rotation. Handles cash/gold fallback if no qualifying asset.
    Returns equity, alloc_hist, switches.
    """
    if start_date:
        start_dt = pd.to_datetime(start_date)
        rs_data = rs_data[rs_data.index >= start_dt]

    if rs_data.empty:
        raise ValueError("No data.")

    returns_dict = {name: df["close"].reindex(rs_data.index).pct_change().fillna(0) for name, df in assets.items()}
    returns_df = pd.DataFrame(returns_dict)

    gold_returns = gold_df["close"].reindex(rs_data.index).pct_change().fillna(0)
    gold_tpi = gold_df["TPI"].reindex(rs_data.index).fillna(0)

    equity = pd.Series(1.0, index=rs_data.index, name='Equity')
    current_alloc = None
    current_use_gold2 = False
    alloc_hist = []
    switches = 0

    for i in range(len(rs_data)):
        # Update use_gold2 based on previous GOLD TPI
        if i > 0:
            gold_tpi_prev = gold_tpi.iloc[i-1]
            if gold_tpi_prev > 0 and use_gold:
                current_use_gold2 = True
            if gold_tpi_prev < 0:
                current_use_gold2 = False

        row = rs_data.iloc[i]
        if np.all(np.isneginf(row)) or row.max() == -np.inf:
            top = 'cash'
        else:
            top = row.idxmax()

        if top != current_alloc:
            switches += 1
            current_alloc = top
            logger.info(f"Switch at {rs_data.index[i]}: {top}")

        if top != 'cash':
            period_return = returns_df.iloc[i][top]
        else:
            period_return = gold_returns.iloc[i] if current_use_gold2 else 0

        if i > 0:
            equity.iloc[i] = equity.iloc[i-1] * (1 + period_return)
        alloc_hist.append(top if top != 'cash' else 'GOLD' if current_use_gold2 else 'CASH')

    return equity.ffill(), alloc_hist, switches

def compute_metrics(equity: pd.Series) -> dict:
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
    maxdd = drawdowns.min() * 100  # As %

    # Omega (simple gain/loss ratio)
    above = returns[returns > 0].sum()
    below = abs(returns[returns <= 0].sum())
    omega = above / below if below > 0 else float('inf')

    return {"CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "Omega": omega, "MaxDD": maxdd, "NetProfit": net_profit, "Days": days}