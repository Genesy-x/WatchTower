import pandas as pd
import numpy as np

def compute_equity(df: pd.DataFrame, fee: float = 0.0005, rebalance_interval: int = 12):
    """
    Simulates an equity curve similar to the PineScript f_equity logic.
    It assumes we're switching allocations based on 'signal' column (1 = long, 0 = cash).
    Args:
        df: DataFrame with 'close' column and optional 'signal' column.
        fee: Transaction fee per trade (default 0.0005 or 0.05%).
        rebalance_interval: Number of periods between rebalances (default 12, e.g., 12 hours or days).
    Returns:
        tuple: (DataFrame with equity metrics, dict of performance metrics)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    df = df.copy().reset_index(drop=True)

    # Generate signal if not present (assume 1 if no signal, adjust based on strategy)
    if "signal" not in df.columns:
        print("[WARNING] No 'signal' column found, assuming always in position")
        df["signal"] = 1

    # Compute log returns
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["ret"].fillna(0, inplace=True)

    # Apply trading signal (1 = invested, 0 = cash)
    df["strategy_ret"] = df["ret"] * df["signal"]

    # Apply trading fees (only when signal changes)
    df["signal_change"] = df["signal"].diff().fillna(0).abs()
    df["fee_cost"] = df["signal_change"] * fee
    df["strategy_ret_net"] = df["strategy_ret"] - df["fee_cost"]

    # Compute cumulative equity
    df["equity"] = (1 + df["strategy_ret_net"]).cumprod()

    # Compute Buy & Hold baseline
    df["bh_equity"] = (1 + df["ret"]).cumprod()

    # Rolling metrics (annualized based on rebalance_interval)
    df["rolling_vol"] = df["strategy_ret"].rolling(30).std() * np.sqrt(365 * 24 / rebalance_interval)
    df["rolling_sharpe"] = (
        df["strategy_ret"].rolling(30).mean() / df["strategy_ret"].rolling(30).std()
    ) * np.sqrt(365 * 24 / rebalance_interval)

    # Basic performance summary
    final_equity = df["equity"].iloc[-1]
    bh_final = df["bh_equity"].iloc[-1]
    total_ret = (final_equity - 1) * 100
    bh_ret = (bh_final - 1) * 100
    max_dd = (df["equity"] / df["equity"].cummax() - 1).min() * 100

    metrics = {
        "total_return_%": round(total_ret, 2),
        "buy_and_hold_%": round(bh_ret, 2),
        "max_drawdown_%": round(max_dd, 2),
        "final_equity": round(final_equity, 3),
        "sharpe": round(df["rolling_sharpe"].iloc[-1], 2) if not np.isnan(df["rolling_sharpe"].iloc[-1]) else None,
    }

    return df, metrics