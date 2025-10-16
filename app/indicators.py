import pandas as pd
import numpy as np

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[DEBUG] Input to compute_indicators: {df.head()}")  # Debug input
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
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

    # VIDYA - Fix: Calculate alpha as scalar or apply row by row
    std = df["close"].rolling(window=14).std()
    close_safe = df["close"].replace(0, 1)  # Avoid division by zero
    
    # Calculate VIDYA properly - apply element-wise with proper alpha handling
    vidya = pd.Series(index=df.index, dtype=float)
    vidya.iloc[0] = df["close"].iloc[0]
    
    for i in range(1, len(df)):
        # Calculate alpha for this specific row
        std_val = std.iloc[i]
        close_val = close_safe.iloc[i]
        
        # Ensure we have valid values
        if pd.isna(std_val) or pd.isna(close_val) or close_val == 0:
            alpha = 2 / (14 + 1)  # Default alpha
        else:
            alpha = (2 / (14 + 1)) * (1 - std_val / close_val)
            # Clamp alpha between 0 and 1
            alpha = max(0.0001, min(alpha, 1.0))
        
        # Apply EMA formula: VIDYA[i] = alpha * close[i] + (1 - alpha) * VIDYA[i-1]
        vidya.iloc[i] = alpha * df["close"].iloc[i] + (1 - alpha) * vidya.iloc[i-1]
    
    df["VIDYA"] = vidya

    # ALMA
    df["ALMA"] = df["close"].rolling(window=9).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)) if len(x) == 9 else np.nan, 
        raw=True
    )

    # BBands
    sma = df["close"].rolling(window=20).mean()
    std_bb = df["close"].rolling(window=20).std()
    df["BB_upper"] = sma + 2 * std_bb
    df["BB_lower"] = sma - 2 * std_bb
    
    # Avoid division by zero in BB_PCT
    bb_range = df["BB_upper"] - df["BB_lower"]
    df["BB_PCT"] = np.where(
        bb_range != 0,
        (df["close"] - df["BB_lower"]) / bb_range,
        0.5  # Default to middle if no range
    )

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