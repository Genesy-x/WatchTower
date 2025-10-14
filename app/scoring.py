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

    # Normalize indicators to 0–1 scale, handle missing values
    momentum_score = np.tanh(latest.get("Momentum", 0) / 100) if "Momentum" in latest else 0
    rsi_score = 1 - abs(50 - latest.get("RSI", 50)) / 50 if "RSI" in latest else 0.5  # Neutral if NA
    sma_trend = (latest.get("SMA50", 0) - latest.get("SMA200", 0)) / (latest.get("SMA200", 1) or 1) if "SMA50" in latest and "SMA200" in latest else 0
    sma_score = np.tanh(sma_trend * 10)

    # Volume-to-market-cap ratio (handle zero division)
    vol_ratio = (fundamentals.get("volume_24h", 0) / (fundamentals.get("market_cap", 1) or 1))  # Avoid division by zero
    vol_score = np.tanh(vol_ratio * 1e3)

    # Weighted combination
    score = (
        0.35 * momentum_score +
        0.25 * rsi_score +
        0.25 * sma_score +
        0.15 * vol_score
    )

    return round(max(0, min(score * 100, 100)), 2)
