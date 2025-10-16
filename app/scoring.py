import numpy as np
import pandas as pd

def score_coin(df: pd.DataFrame, fundamentals: dict):
    """
    Compute a weighted performance score for a given coin.
    Inputs:
        df: DataFrame with OHLCV + indicators (RSI, momentum, etc.)
        fundamentals: dict with 'market_cap', 'volume_24h', etc.
    Returns:
        score (float)
    """
    if df.empty or len(df) < 2:
        return 0.0

    latest = df.iloc[-1].dropna()
    if latest.empty:
        return 0.0

    # Extract scalar values with explicit checks
    momentum = latest.get("Momentum")
    momentum_score = np.tanh(momentum / 100) if pd.notna(momentum) and not isinstance(momentum, pd.Series) else 0
    if isinstance(momentum, pd.Series) and not momentum.empty:
        momentum_score = np.tanh(momentum.iloc[-1] / 100)  # Use iloc[-1] for last value

    rsi = latest.get("RSI")
    rsi_score = 1 - abs(50 - rsi) / 50 if pd.notna(rsi) and not isinstance(rsi, pd.Series) else 0.5
    if isinstance(rsi, pd.Series) and not rsi.empty:
        rsi_score = 1 - abs(50 - rsi.iloc[-1]) / 50  # Use iloc[-1] for last value

    sma50 = latest.get("SMA50", 0)
    sma200 = latest.get("SMA200", 0)
    sma_score = (sma50 - sma200) / (sma200 or 1) if pd.notna(sma50) and pd.notna(sma200) and not isinstance(sma50, pd.Series) and not isinstance(sma200, pd.Series) else 0
    if (isinstance(sma50, pd.Series) or isinstance(sma200, pd.Series)) and not sma50.empty and not sma200.empty:
        sma_score = (sma50.iloc[-1] - sma200.iloc[-1]) / (sma200.iloc[-1] or 1)  # Use iloc[-1] for last values

    vol_ratio = (fundamentals.get("volume_24h", 0) / (fundamentals.get("market_cap", 1) or 1))
    vol_score = np.tanh(vol_ratio * 1e3)

    score = (
        0.35 * momentum_score +
        0.25 * rsi_score +
        0.25 * sma_score +
        0.15 * vol_score
    )

    return round(max(0, min(score * 100, 100)), 2)