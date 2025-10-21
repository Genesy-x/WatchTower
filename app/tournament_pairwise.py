"""
Pairwise Tournament System - Exact PineScript "Majors Sync" Logic

This implements the tournament where each asset competes against every other asset
using relative strength signals on their price ratios.

Example for 4 assets (BTC, ETH, SOL, SUI):
- BTC gets +1 for each: ETHBTC<0, SOLBTC<0, SUIBTC<0 (max 3 points)
- ETH gets +1 for each: ETHBTC>0, SOLETH<0, SUIETH<0 (max 3 points)
- SOL gets +1 for each: SOLBTC>0, SOLETH>0, SUISOL<0 (max 3 points)
- SUI gets +1 for each: SUIBTC>0, SUIETH>0, SUISOL>0 (max 3 points)

The asset with the highest score is the winner!
"""
import pandas as pd
import numpy as np
from itertools import combinations

def compute_pair_tpi(asset1_df: pd.DataFrame, asset2_df: pd.DataFrame, 
                     asset1_name: str, asset2_name: str) -> pd.Series:
    """
    Compute TPI (Trend Power Index) for asset1/asset2 ratio
    
    Returns: 
        Series with values:
        +1 = asset1 outperforming asset2
        -1 = asset2 outperforming asset1
         0 = neutral
    
    This uses QB indicator logic on the price ratio
    """
    # Align indices
    common_index = asset1_df.index.intersection(asset2_df.index)
    
    if common_index.empty:
        print(f"[WARNING] No common dates between {asset1_name} and {asset2_name}")
        return pd.Series(0, index=asset1_df.index)
    
    # Calculate ratio (asset1 / asset2)
    asset1_close = asset1_df['close'].reindex(common_index).ffill()
    asset2_close = asset2_df['close'].reindex(common_index).ffill()
    
    ratio = asset1_close / asset2_close
    
    # Create OHLCV dataframe for the ratio
    # We use the ratio as all price fields (simplified)
    ratio_df = pd.DataFrame({
        'close': ratio,
        'open': ratio,
        'high': ratio,
        'low': ratio,
        'volume': asset1_df['volume'].reindex(common_index).fillna(0)
    })
    
    # Import and compute QB indicators on the ratio
    from app.strategies.qb_strategy import compute_indicators
    
    try:
        ratio_indicators = compute_indicators(ratio_df)
        
        # Get the QB signal (already shifted to avoid lookahead)
        pair_tpi = ratio_indicators['QB'].fillna(0)
        
        # Convert to +1/-1 format with state preservation
        tpi_signal = pd.Series(0, index=pair_tpi.index)
        for i in range(len(pair_tpi)):
            if i > 0:
                tpi_signal.iloc[i] = tpi_signal.iloc[i-1]  # Preserve state
            
            if pair_tpi.iloc[i] == 1:
                tpi_signal.iloc[i] = 1
            elif pair_tpi.iloc[i] == -1:
                tpi_signal.iloc[i] = -1
        
        # Reindex to full asset1 index
        result = tpi_signal.reindex(asset1_df.index, fill_value=0)
        
        print(f"[PAIR] {asset1_name}/{asset2_name}: {len(result)} rows, "
              f"Bullish={sum(result > 0)}, Bearish={sum(result < 0)}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to compute pair TPI for {asset1_name}/{asset2_name}: {e}")
        return pd.Series(0, index=asset1_df.index)


def run_pairwise_tournament(assets_data: dict, start_date: str = None) -> tuple:
    """
    Run complete pairwise tournament
    
    For N assets, computes N*(N-1)/2 pair signals
    Then scores each asset based on how many pairs it dominates
    
    Returns:
        scores_df: DataFrame with daily scores for each asset
        pair_signals: Dict of pair signals for debugging
    """
    asset_names = sorted(assets_data.keys())  # Sort for consistent ordering
    n_assets = len(asset_names)
    
    print(f"\n{'='*60}")
    print(f"PAIRWISE TOURNAMENT: {n_assets} assets")
    print(f"Assets: {asset_names}")
    print(f"Pairs to compute: {n_assets * (n_assets - 1) // 2}")
    print(f"{'='*60}\n")
    
    # Get common index
    all_indices = [df.index for df in assets_data.values()]
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.intersection(idx)
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        common_index = common_index[common_index >= start_dt]
    
    print(f"[TOURNAMENT] Date range: {common_index[0].date()} to {common_index[-1].date()}")
    print(f"[TOURNAMENT] Total days: {len(common_index)}\n")
    
    # Compute all pairwise TPIs
    pair_signals = {}
    pair_count = 0
    total_pairs = n_assets * (n_assets - 1) // 2
    
    for asset1, asset2 in combinations(asset_names, 2):
        pair_count += 1
        pair_name = f"{asset1}/{asset2}"
        
        print(f"[{pair_count}/{total_pairs}] Computing {pair_name}...")
        
        tpi = compute_pair_tpi(
            assets_data[asset1], 
            assets_data[asset2],
            asset1,
            asset2
        )
        
        pair_signals[pair_name] = tpi.reindex(common_index, fill_value=0)
    
    print(f"\n[TOURNAMENT] All pair signals computed!\n")
    
    # Initialize score dataframe
    scores_df = pd.DataFrame(0, index=common_index, columns=asset_names)
    
    # Score each asset on each day
    print("[TOURNAMENT] Computing daily scores...\n")
    
    for date_idx in range(len(common_index)):
        date = common_index[date_idx]
        
        for asset_name in asset_names:
            score = 0
            
            # Check against all other assets
            for other_asset in asset_names:
                if asset_name == other_asset:
                    continue
                
                # Get the pair signal
                if asset_name < other_asset:  # Lexicographic order
                    pair_name = f"{asset_name}/{other_asset}"
                    signal = pair_signals[pair_name].iloc[date_idx]
                    # Positive signal means asset_name > other_asset
                    if signal > 0:
                        score += 1
                else:
                    pair_name = f"{other_asset}/{asset_name}"
                    signal = pair_signals[pair_name].iloc[date_idx]
                    # Negative signal means asset_name > other_asset
                    if signal < 0:
                        score += 1
            
            scores_df.loc[date, asset_name] = score
        
        # Debug first few days
        if date_idx < 3:
            scores_row = scores_df.loc[date]
            print(f"[TOURNAMENT] Day {date_idx} ({date.date()}): {scores_row.to_dict()}")
    
    print(f"\n[TOURNAMENT] Tournament scoring complete!")
    print(f"[TOURNAMENT] Score range: 0 (weakest) to {n_assets-1} (strongest)\n")
    
    return scores_df, pair_signals


def get_ranked_assets_with_signals(scores_row: pd.Series, assets_data: dict, 
                                   date_idx: int) -> list:
    """
    Get assets ranked by tournament score, with their individual bullish/bearish status
    
    Returns: List of tuples (asset_name, score, is_bullish, qb_value)
    Sorted by score (descending)
    """
    rankings = []
    
    for asset_name in scores_row.index:
        score = scores_row[asset_name]
        
        # Check individual asset signal (QB)
        if asset_name in assets_data:
            asset_df = assets_data[asset_name]
            
            if date_idx < len(asset_df):
                qb_value = asset_df.iloc[date_idx]['QB'] if 'QB' in asset_df.columns else 0
                is_bullish = (qb_value == 1)
            else:
                qb_value = 0
                is_bullish = False
        else:
            qb_value = 0
            is_bullish = False
        
        rankings.append((asset_name, int(score), is_bullish, qb_value))
    
    # Sort by score (descending), then by name for tie-breaking
    rankings.sort(key=lambda x: (-x[1], x[0]))
    
    return rankings


def analyze_tournament_results(scores_df: pd.DataFrame, assets_data: dict):
    """
    Print detailed tournament analysis
    """
    print(f"\n{'='*60}")
    print("TOURNAMENT ANALYSIS")
    print(f"{'='*60}\n")
    
    asset_names = scores_df.columns.tolist()
    
    # Overall statistics
    print("Overall Score Statistics:")
    print("-" * 40)
    for asset in asset_names:
        scores = scores_df[asset]
        print(f"{asset:6s}: Mean={scores.mean():.2f}, "
              f"Min={scores.min():.0f}, Max={scores.max():.0f}, "
              f"Std={scores.std():.2f}")
    
    print(f"\n{'='*60}\n")
    
    # Leadership analysis
    print("Leadership Days (days as #1 ranked):")
    print("-" * 40)
    
    for asset in asset_names:
        days_as_leader = (scores_df.idxmax(axis=1) == asset).sum()
        pct = days_as_leader / len(scores_df) * 100
        print(f"{asset:6s}: {days_as_leader:4d} days ({pct:5.1f}%)")
    
    print(f"\n{'='*60}\n")