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
        return 0.0  # Default score for empty data

    latest = df.iloc[-1].dropna()  # Drop NaN values from the latest row
    if latest.empty:
        return 0.0  # Handle case where all values are NaN

    # Normalize indicators to 0–1 scale, handle missing values explicitly
    momentum = latest.get("Momentum")
    momentum_score = np.tanh(momentum / 100) if pd.notna(momentum) and isinstance(momentum, (int, float)) else 0

    rsi = latest.get("RSI")
    rsi_score = 1 - abs(50 - rsi) / 50 if pd.notna(rsi) and isinstance(rsi, (int, float)) else 0.5  # Neutral if NA

    sma50 = latest.get("SMA50", 0)
    sma200 = latest.get("SMA200", 0)
    sma_trend = (sma50 - sma200) / (sma200 or 1) if pd.notna(sma50) and pd.notna(sma200) else 0
    sma_score = np.tanh(sma_trend * 10)

    # Volume-to-market-cap ratio
    vol_ratio = (fundamentals.get("volume_24h", 0) / (fundamentals.get("market_cap", 1) or 1))
    vol_score = np.tanh(vol_ratio * 1e3)

    # Weighted combination
    score = (
        0.35 * momentum_score +
        0.25 * rsi_score +
        0.25 * sma_score +
        0.15 * vol_score
    )

    return round(max(0, min(score * 100, 100)), 2)
