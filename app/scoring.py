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

    latest = df.iloc[-1]
    
    # Helper function to safely extract scalar values
    def get_scalar(value):
        """Extract scalar from Series or return value as-is"""
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            return value.iloc[-1] if len(value) > 0 else None
        return value
    
    # Extract and validate momentum
    momentum = get_scalar(latest.get("Momentum"))
    momentum_score = np.tanh(momentum / 100) if pd.notna(momentum) else 0

    # Extract and validate RSI
    rsi = get_scalar(latest.get("RSI"))
    rsi_score = 1 - abs(50 - rsi) / 50 if pd.notna(rsi) else 0.5

    # Extract and validate SMAs
    sma50 = get_scalar(latest.get("SMA50"))
    sma200 = get_scalar(latest.get("SMA200"))
    
    if pd.notna(sma50) and pd.notna(sma200) and sma200 != 0:
        sma_score = (sma50 - sma200) / sma200
    else:
        sma_score = 0

    # Volume score
    vol_ratio = (fundamentals.get("volume_24h", 0) / max(fundamentals.get("market_cap", 1), 1))
    vol_score = np.tanh(vol_ratio * 1e3)

    # Weighted score
    score = (
        0.35 * momentum_score +
        0.25 * rsi_score +
        0.25 * sma_score +
        0.15 * vol_score
    )

    return round(max(0, min(score * 100, 100)), 2)