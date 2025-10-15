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
    if df.empty | (len(df) < 2):  # Bitwise OR
        print(f"[DEBUG] Empty or insufficient data in score_coin: {df.head()}")
        return 0.0

    latest = df.iloc[-1].dropna()
    if latest.empty:
        print(f"[DEBUG] All NaN in latest row: {df.iloc[-1]}")
        return 0.0

    print(f"[DEBUG] Latest row for scoring: {latest.to_dict()}")

    # Extract scalar values with bitwise checks
    momentum = latest.get("Momentum")
    print(f"[DEBUG] Momentum value: {momentum}, type: {type(momentum)}")
    momentum_score = np.tanh(momentum / 100) if (pd.notna(momentum)) & (not isinstance(momentum, pd.Series)) else 0
    if isinstance(momentum, pd.Series) & (not momentum.empty):
        print(f"[DEBUG] Momentum is Series: {momentum}")
        momentum_score = np.tanh(momentum.item() / 100)

    rsi = latest.get("RSI")
    print(f"[DEBUG] RSI value: {rsi}, type: {type(rsi)}")
    rsi_score = 1 - abs(50 - rsi) / 50 if (pd.notna(rsi)) & (not isinstance(rsi, pd.Series)) else 0.5
    if isinstance(rsi, pd.Series) & (not rsi.empty):
        print(f"[DEBUG] RSI is Series: {rsi}")
        rsi_score = 1 - abs(50 - rsi.item()) / 50

    sma50 = latest.get("SMA50", 0)
    sma200 = latest.get("SMA200", 0)
    print(f"[DEBUG] SMA50 value: {sma50}, type: {type(sma50)}, SMA200 value: {sma200}, type: {type(sma200)}")
    sma_trend = (sma50 - sma200) / (sma200 or 1) if (pd.notna(sma50)) & (pd.notna(sma200)) & (not isinstance(sma50, pd.Series)) & (not isinstance(sma200, pd.Series)) else 0
    if (isinstance(sma50, pd.Series) | isinstance(sma200, pd.Series)) & (not sma50.empty) & (not sma200.empty):
        print(f"[DEBUG] SMA50 or SMA200 is Series: {sma50}, {sma200}")
        sma_trend = (sma50.item() - sma200.item()) / (sma200.item() or 1)
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