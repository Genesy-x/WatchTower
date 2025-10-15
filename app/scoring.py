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
        print(f"[DEBUG] Empty or insufficient data in score_coin: {df.head()}")
        return 0.0

    latest = df.iloc[-1].dropna()
    if latest.empty:
        print(f"[DEBUG] All NaN in latest row: {df.iloc[-1]}")
        return 0.0

    print(f"[DEBUG] Latest row for scoring: {latest.to_dict()}")  # Debug as dict

    # Extract scalar values explicitly
    momentum = latest.get("Momentum")
    momentum_score = np.tanh(momentum / 100) if pd.notna(momentum) and not isinstance(momentum, pd.Series) else 0
    if isinstance(momentum, pd.Series):
        print(f"[DEBUG] Momentum is Series: {momentum}")
        momentum_score = np.tanh(momentum.iloc[0] / 100) if not momentum.empty else 0

    rsi = latest.get("RSI")
    rsi_score = 1 - abs(50 - rsi) / 50 if pd.notna(rsi) and not isinstance(rsi, pd.Series) else 0.5
    if isinstance(rsi, pd.Series):
        print(f"[DEBUG] RSI is Series: {rsi}")
        rsi_score = 1 - abs(50 - rsi.iloc[0]) / 50 if not rsi.empty else 0.5

    sma50 = latest.get("SMA50", 0)
    sma200 = latest.get("SMA200", 0)
    sma_trend = (sma50 - sma200) / (sma200 or 1) if pd.notna(sma50) and pd.notna(sma200) and not isinstance(sma50, pd.Series) and not isinstance(sma200, pd.Series) else 0
    if isinstance(sma50, pd.Series) or isinstance(sma200, pd.Series):
        print(f"[DEBUG] SMA50 or SMA200 is Series: {sma50}, {sma200}")
        sma_trend = (sma50.iloc[0] - sma200.iloc[0]) / (sma200.iloc[0] or 1) if not sma50.empty and not sma200.empty else 0
    sma_score = np.tanh(sma_trend * 10)

    vol_ratio = (fundamentals.get("volume_24h", 0) / (fundamentals.get("market_cap", 1) or 1))
    vol_score = np.tanh(vol_ratio * 1e3)

    score = (
        0.35 * momentum_score +
        0.25 * rsi_score +
        0.25 * sma_score +
        0.15 * vol_score
    )

    return round(max(0, min(score * 100, 100)), 2)
