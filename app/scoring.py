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

    latest = df.iloc[-1]  # latest candle

    # Normalize indicators to 0–1 scale
    momentum_score = np.tanh(latest["momentum"] / 100)
    rsi_score = 1 - abs(50 - latest["rsi"]) / 50  # closer to 50 = more stable
    sma_trend = (latest["sma_20"] - latest["sma_50"]) / latest["sma_50"]
    sma_score = np.tanh(sma_trend * 10)

    # Volume-to-market-cap ratio
    vol_ratio = fundamentals["volume_24h"] / fundamentals["market_cap"]
    vol_score = np.tanh(vol_ratio * 1e3)

    # Weighted combination (tune weights later)
    score = (
        0.35 * momentum_score +
        0.25 * rsi_score +
        0.25 * sma_score +
        0.15 * vol_score
    )

    # Scale 0–100
    return round(max(0, min(score * 100, 100)), 2)
