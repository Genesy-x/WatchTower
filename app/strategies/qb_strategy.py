"""
QB Trading Strategy
Converted from Pine Script to Python
"""
import pandas as pd
import numpy as np

def calculate_rsi(src, period=14):
    """Calculate RSI indicator"""
    delta = src.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate SMMA (RMA in Pine Script)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_zscore(src, period=40):
    """Calculate Z-Score"""
    mean = src.rolling(window=period).mean()
    std = src.rolling(window=period).std()
    zscore = (src - mean) / std
    return zscore

def calculate_roc(src, period=30):
    """Calculate Rate of Change"""
    roc = ((src - src.shift(period)) / src.shift(period)) * 100
    return roc

def smma(series, period):
    """Calculate Smoothed Moving Average (SMMA/RMA)"""
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_stdev(series, period=30):
    """Calculate Standard Deviation"""
    return series.rolling(window=period).std()

def alma(src, window=9, sigma=6, offset=0.85):
    """Calculate Arnaud Legoux Moving Average (ALMA)"""
    m = offset * (window - 1)
    s = window / sigma
    
    alma_values = []
    src_array = src.values
    
    for i in range(len(src)):
        if i < window - 1:
            alma_values.append(np.nan)
        else:
            wtd_sum = 0
            cum_wt = 0
            for j in range(window):
                w = np.exp(-((j - m) ** 2) / (2 * s * s))
                wtd_sum += src_array[i - window + 1 + j] * w
                cum_wt += w
            alma_values.append(wtd_sum / cum_wt)
    
    return pd.Series(alma_values, index=src.index)

def normalize(src, length):
    """Normalize values between -0.5 and 0.5"""
    lowest = src.rolling(window=length).min()
    highest = src.rolling(window=length).max()
    smoothed_value = (src - lowest) / (highest - lowest) - 0.5
    return smoothed_value

def ma_stdev(ma, ma2, src):
    """Calculate MA standard deviation signal"""
    # Base MA Standard Deviation
    base_ma_sd = calculate_stdev(ma, 30)
    upper_base_ma = (ma + base_ma_sd) * 1
    lower_base_ma = (ma - base_ma_sd) * 1
    
    # Base signals
    base_ma_long_signal = src > upper_base_ma
    base_ma_short_signal = src < lower_base_ma
    
    # Normalized MA Calculations
    normalized_base_ma = ma2
    normalized_ma = -1 * normalized_base_ma / src
    
    # Normalized Standard Deviation
    normalized_sd = calculate_stdev(normalized_ma, 30)
    normalized_lower_bound = normalized_ma - normalized_sd
    
    # Normalized signals
    normalized_long_signal = normalized_lower_bound > -1
    normalized_short_signal = normalized_ma < -1
    
    # Final signal with state preservation
    final_sig = pd.Series(0, index=src.index)
    
    for i in range(len(src)):
        if i > 0:
            final_sig.iloc[i] = final_sig.iloc[i-1]  # Preserve previous state
        
        if base_ma_long_signal.iloc[i] and normalized_long_signal.iloc[i]:
            final_sig.iloc[i] = 1
        elif base_ma_short_signal.iloc[i] and normalized_short_signal.iloc[i]:
            final_sig.iloc[i] = -1
    
    return final_sig

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for QB strategy indicators
    This function signature matches the expected interface
    """
    df = df.copy()
    src = df['close']
    
    # OSCILLATORS - Midline Cross
    rsi_ = calculate_rsi(src, 14)
    zscore_ = calculate_zscore(src, 40)
    roc_ = calculate_roc(src, 30)
    
    rsi = smma(rsi_, 2)
    zscore = smma(zscore_, 2)
    roc = smma(roc_, 2)
    
    rsis = pd.Series(np.where(rsi > 50, 1, np.where(rsi < 50, -1, 0)), index=src.index)
    zscores = pd.Series(np.where(zscore > 0, 1, np.where(zscore < 0, -1, 0)), index=src.index)
    rocs = pd.Series(np.where(roc > 0, 1, np.where(roc < 0, -1, 0)), index=src.index)
    
    avg = pd.Series(np.where((rsis + zscores + rocs) / 3 > 0, 1, -1), index=src.index)
    
    # OSCILLATORS - Stdev
    sd_filt1 = calculate_stdev(rsi_, 30)
    norm_dn1 = rsi_ + sd_filt1 * 1
    norm_up1 = rsi_ - sd_filt1 * 1
    
    sd_filt2 = calculate_stdev(zscore_, 30)
    norm_dn2 = zscore_ + sd_filt2 * 1
    norm_up2 = zscore_ - sd_filt2 * 1
    
    sd_filt4 = calculate_stdev(roc_, 30)
    norm_dn4 = roc_ + sd_filt4 * 1
    norm_up4 = roc_ - sd_filt4 * 1
    
    # State preservation for stdev signals
    rsi_stdevs = pd.Series(0, index=src.index)
    zscore_stdevs = pd.Series(0, index=src.index)
    roc_stdevs = pd.Series(0, index=src.index)
    
    for i in range(len(src)):
        if i > 0:
            rsi_stdevs.iloc[i] = rsi_stdevs.iloc[i-1]
            zscore_stdevs.iloc[i] = zscore_stdevs.iloc[i-1]
            roc_stdevs.iloc[i] = roc_stdevs.iloc[i-1]
        
        # RSI stdev
        if not pd.isna(norm_up1.iloc[i]) and norm_up1.iloc[i] > 50:
            rsi_stdevs.iloc[i] = 1
        elif not pd.isna(norm_dn1.iloc[i]) and norm_dn1.iloc[i] < 50:
            rsi_stdevs.iloc[i] = -1
        
        # Zscore stdev
        if not pd.isna(norm_up2.iloc[i]) and norm_up2.iloc[i] > 0:
            zscore_stdevs.iloc[i] = 1
        elif not pd.isna(norm_dn2.iloc[i]) and norm_dn2.iloc[i] < 0:
            zscore_stdevs.iloc[i] = -1
        
        # ROC stdev
        if not pd.isna(norm_up4.iloc[i]) and norm_up4.iloc[i] > 0:
            roc_stdevs.iloc[i] = 1
        elif not pd.isna(norm_dn4.iloc[i]) and norm_dn4.iloc[i] < 0:
            roc_stdevs.iloc[i] = -1
    
    avg2 = pd.Series(np.where((rsi_stdevs + zscore_stdevs + roc_stdevs) / 3 > 0, 1, -1), index=src.index)
    
    # MA'S - Normalized
    ma1 = src.rolling(window=7).mean()
    ma2 = src.ewm(span=7, adjust=False).mean()
    ma3 = alma(src, window=7, sigma=6, offset=0.85)
    
    zscore1 = normalize(ma1, 30)
    zscore2 = normalize(ma2, 30)
    zscore3 = normalize(ma3, 30)
    
    ma1s = pd.Series(np.where(zscore1 > 0, 1, np.where(zscore1 < 0, -1, 0)), index=src.index)
    ma2s = pd.Series(np.where(zscore2 > 0, 1, np.where(zscore2 < 0, -1, 0)), index=src.index)
    ma3s = pd.Series(np.where(zscore3 > 0, 1, np.where(zscore3 < 0, -1, 0)), index=src.index)
    
    avg3 = pd.Series(np.where((ma1s + ma2s + ma3s) / 3 > 0, 1, -1), index=src.index)
    
    # MA'S - Slope
    MA1 = src.rolling(window=30).mean()
    MA2 = src.ewm(span=30, adjust=False).mean()
    MA3 = alma(src, window=30, sigma=6, offset=0.85)
    
    MA1_2 = src.rolling(window=14).mean()
    MA2_2 = src.ewm(span=14, adjust=False).mean()
    MA3_2 = alma(src, window=14, sigma=6, offset=0.85)
    
    mastdev1 = ma_stdev(MA1_2, MA1, src)
    mastdev2 = ma_stdev(MA2_2, MA2, src)
    mastdev3 = ma_stdev(MA3_2, MA3, src)
    
    avg5 = pd.Series(np.where((mastdev1 + mastdev2 + mastdev3) / 3 > 0, 1, -1), index=src.index)
    
    # Final average
    avg_ = (avg + avg2 + avg3 + avg5) / 4
    
    # Generate QB signal with state preservation
    QB = pd.Series(np.nan, index=src.index)
    
    for i in range(len(src)):
        if i > 0 and not pd.isna(QB.iloc[i-1]):
            QB.iloc[i] = QB.iloc[i-1]  # Preserve previous state
        
        if not pd.isna(avg_.iloc[i]):
            if avg_.iloc[i] > 0.34:
                QB.iloc[i] = 1
            elif avg_.iloc[i] < -0.34:
                QB.iloc[i] = -1
    
    # Add required columns for compatibility with rotation strategy
    df['QB'] = QB
    df['Signal'] = avg_
    df['TPI'] = QB  # TPI used for filtering in rotation
    df['Momentum'] = df['close'].pct_change(1)  # Used for ranking
    
    # Store intermediate values for potential analysis
    df['RSI'] = rsi
    df['ZScore'] = zscore
    df['ROC'] = roc
    
    result = df.dropna()
    print(f"[DEBUG QB] Computed QB indicators, {len(result)} valid rows")
    return result