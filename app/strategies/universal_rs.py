import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def rotate_equity(rs_data: pd.DataFrame, assets: dict, gold_df: pd.DataFrame, start_date: str = None, use_gold: bool = True) -> tuple:
    """
    Simulate rotation. Handles cash/gold fallback if no qualifying asset.
    Returns equity, alloc_hist, switches.
    """
    if start_date:
        start_dt = pd.to_datetime(start_date)
        rs_data = rs_data[rs_data.index >= start_dt]

    if rs_data.empty:
        raise ValueError("No RS data available.")

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
        row = rs_data.iloc[i]
        # Check for invalid data (NaN or inf)
        if row.isna().any() or np.isinf(row).any():
            logger.warning(f"Invalid data at index {rs_data.index[i]}: {row}")
            top = 'cash'
        elif row.empty or row.dropna().empty:
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
            # Use gold if TPI positive, otherwise cash
            gold_tpi_prev = gold_tpi.iloc[i] if i > 0 else gold_tpi.iloc[0]
            current_use_gold2 = gold_tpi_prev > 0 and use_gold
            period_return = gold_returns.iloc[i] if current_use_gold2 else 0

        if i > 0:
            equity.iloc[i] = equity.iloc[i-1] * (1 + period_return)
        alloc_hist.append(top if top != 'cash' else 'GOLD' if current_use_gold2 else 'CASH')

    return equity.ffill(), alloc_hist, switches