import pandas as pd
import pandas_ta as ta
import numpy as np

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["Momentum"] = df["close"].pct_change(1)  # ROC(1) equivalent
    df["SMA50"] = ta.sma(df["close"], length=50)
    df["SMA200"] = ta.sma(df["close"], length=200)
    df["VolumeChange"] = df["volume"].pct_change(5)

    # For UNI/TPI approximation
    df["DEMA"] = ta.dema(df["close"], length=14)
    df["VIDYA"] = ta.vidya(df["close"], length=14)
    df["ALMA"] = ta.alma(df["close"], length=9)
    bb = ta.bbands(df["close"], length=20)
    
    # Use index-based access to avoid KeyError on column names
    if bb is not None and len(bb.columns) >= 3:
        lower = bb.iloc[:, 0]  # Lower band
        middle = bb.iloc[:, 1]  # Middle band (not used)
        upper = bb.iloc[:, 2]  # Upper band
        df["BB_PCT"] = (df["close"] - lower) / (upper - lower)
    else:
        df["BB_PCT"] = np.nan  # Fallback if bbands fails

    # Signals (+1 bull, -1 bear)
    df["SIG_DEMA"] = np.where(df["close"] > df["DEMA"], 1, -1)
    df["SIG_VIDYA"] = np.where(df["close"] > df["VIDYA"], 1, -1)
    df["SIG_ALMA"] = np.where(df["close"] > df["ALMA"], 1, -1)
    df["SIG_BB"] = np.where(df["BB_PCT"] > 0.5, 1, -1)

    # TPI: Average of signals (Trend Probability Index)
    df["TPI"] = df[["SIG_DEMA", "SIG_VIDYA", "SIG_ALMA", "SIG_BB"]].mean(axis=1)

    return df.dropna()