import pandas as pd
import numpy as np

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[DEBUG] Input to compute_indicators: {df.head()}")  # Debug input
    # RSI manual
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Momentum"] = df["close"].pct_change(1)
    df["SMA50"] = df["close"].rolling(window=50).mean()
    df["SMA200"] = df["close"].rolling(window=200).mean()
    df["VolumeChange"] = df["volume"].pct_change(5)

    # DEMA manual
    ema1 = df["close"].ewm(span=14, adjust=False).mean()
    df["DEMA"] = 2 * ema1 - ema1.ewm(span=14, adjust=False).mean()

    # VIDYA
    std = df["close"].rolling(window=14).std()
    df["VIDYA"] = df["close"].ewm(alpha=2 / (14 + 1) * (1 - std / df["close"].replace(0, 1)), adjust=False).mean()

    # ALMA
    df["ALMA"] = df["close"].rolling(window=9).apply(lambda x: np.average(x, weights=np.arange(1, 10)), raw=True)

    # BBands
    sma = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["BB_upper"] = sma + 2 * std
    df["BB_lower"] = sma - 2 * std
    df["BB_PCT"] = (df["close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # Signals
    df["SIG_DEMA"] = np.where(df["close"] > df["DEMA"], 1, -1)
    df["SIG_VIDYA"] = np.where(df["close"] > df["VIDYA"], 1, -1)
    df["SIG_ALMA"] = np.where(df["close"] > df["ALMA"], 1, -1)
    df["SIG_BB"] = np.where(df["BB_PCT"] > 0.5, 1, -1)

    # TPI
    df["TPI"] = df[["SIG_DEMA", "SIG_VIDYA", "SIG_ALMA", "SIG_BB"]].mean(axis=1)

    result = df.dropna()
    print(f"[DEBUG] Output from compute_indicators: {result.head()}")  # Debug output
    return result