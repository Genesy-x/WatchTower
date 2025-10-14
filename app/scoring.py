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

    latest = df.iloc[-1]

    # Normalize indicators to 0–1 scale
    momentum_score = np.tanh(latest["Momentum"] / 100) if pd.notna(latest["Momentum"]) else 0
    rsi_score = 1 - abs(50 - latest["RSI"]) / 50 if pd.notna(latest["RSI"]) else 0.5  # Neutral if NA
    sma_trend = (latest["SMA50"] - latest["SMA200"]) / latest["SMA200"] if pd.notna(latest["SMA50"]) and pd.notna(latest["SMA200"]) else 0
    sma_score = np.tanh(sma_trend * 10)

    # Volume-to-market-cap ratio (handle zero division)
    vol_ratio = fundamentals["volume_24h"] / (fundamentals["market_cap"] or 1)  # Avoid division by zero
    vol_score = np.tanh(vol_ratio * 1e3)

    # Weighted combination
    score = (
        0.35 * momentum_score +
        0.25 * rsi_score +
        0.25 * sma_score +
        0.15 * vol_score
    )

    return round(max(0, min(score * 100, 100)), 2)
