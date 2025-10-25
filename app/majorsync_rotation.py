"""
MajorSync Rotation Strategy - Exact PineScript Implementation

Allocation Modes:
- Aggressive:      100% winner
- Semi-Aggressive: 80% winner, 20% second place
- Conservative:    60% winner, 30% second, 10% third

Key Rules (EXACT PINESCRIPT LOGIC):
1. Tournament ranks assets using pairwise comparisons (see tournament_pairwise.py)
2. Check if top N assets from tournament are INDIVIDUALLY bullish (QB=1 on PREVIOUS bar)
3. Only allocate to assets that are BOTH:
   - Ranked in top N by tournament
   - Individually bullish (QB[1]=1)
4. Any unallocated portion goes to GOLD (if GOLD.QB[1]=1) or CASH
"""
import pandas as pd
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def rotate_equity_majorsync(
    assets_data: dict,
    tournament_scores: pd.DataFrame,
    gold_df: pd.DataFrame,
    start_date: str = None,
    use_gold: bool = True,
    allocation_mode: str = "Aggressive"
) -> tuple:
    """
    MajorSync rotation with pairwise tournament rankings + individual trend filter
    
    Args:
        assets_data: Dict of asset DataFrames with OHLCV + indicators (including QB)
        tournament_scores: DataFrame from run_pairwise_tournament()
        gold_df: GOLD/PAXG DataFrame with QB indicator
        start_date: Start date for backtest
        use_gold: Use GOLD as defensive allocation vs CASH
        allocation_mode: "Aggressive", "Semi-Aggressive", or "Conservative"
    
    Returns:
        (equity_series, allocation_history, switches)
    """
    
    # Get common index
    common_index = tournament_scores.index
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        common_index = common_index[common_index >= start_dt]
        tournament_scores = tournament_scores.loc[common_index]
    
    print(f"\n{'='*60}")
    print(f"MAJORSYNC ROTATION (EXACT PINESCRIPT LOGIC)")
    print(f"{'='*60}")
    print(f"Period: {common_index[0].date()} to {common_index[-1].date()}")
    print(f"Days: {len(common_index)}")
    print(f"Assets: {list(assets_data.keys())}")
    print(f"Allocation Mode: {allocation_mode}")
    print(f"Use GOLD: {use_gold}")
    print(f"{'='*60}\n")
    
    # Set allocation percentages
    ALLOCATION_MODES = {
        "Aggressive": [1.0, 0.0, 0.0],       # 100% winner
        "Semi-Aggressive": [0.8, 0.2, 0.0],  # 80/20
        "Conservative": [0.6, 0.3, 0.1]      # 60/30/10
    }
    
    if allocation_mode not in ALLOCATION_MODES:
        print(f"[WARNING] Unknown mode '{allocation_mode}', using Aggressive")
        allocation_mode = "Aggressive"
    
    alloc_weights = ALLOCATION_MODES[allocation_mode]
    print(f"[ALLOCATION] Weights: 1st={alloc_weights[0]*100:.0f}%, "
          f"2nd={alloc_weights[1]*100:.0f}%, 3rd={alloc_weights[2]*100:.0f}%\n")
    
    # Prepare price data aligned to common_index
    close_prices = {}
    qb_signals = {}
    
    for name, df in assets_data.items():
        aligned_df = df.reindex(common_index)
        close_prices[name] = aligned_df['close'].ffill()
        qb_signals[name] = aligned_df['QB'].fillna(0)
    
    # GOLD data
    if gold_df is not None and not gold_df.empty:
        gold_reindexed = gold_df.reindex(common_index)
        gold_qb = gold_reindexed['QB'].fillna(0)
        gold_close = gold_reindexed['close'].ffill()
    else:
        gold_qb = pd.Series(0, index=common_index)
        gold_close = pd.Series(1, index=common_index)
    
    # Initialize tracking
    equity = pd.Series(1.0, index=common_index, name='Equity')
    alloc_hist = []
    switches = 0
    prev_primary = None
    
    # Process each day
    for i in range(len(common_index)):
        date = common_index[i]
        
        # Get tournament rankings for this day
        scores_today = tournament_scores.loc[date]
        ranked_assets = scores_today.sort_values(ascending=False)
        
        # ============================================================
        # CRITICAL FIX: Use PREVIOUS bar's QB signal (avoid lookahead)
        # ============================================================
        
        # Get top 3 from tournament
        top3_assets = list(ranked_assets.index[:3])
        
        # Check which top3 are individually bullish on PREVIOUS bar
        bullish_top3 = []
        
        if i == 0:
            # First bar: no previous bar, assume neutral
            pass
        else:
            prev_date = common_index[i-1]
            
            for asset_name in top3_assets:
                if asset_name in qb_signals:
                    qb_prev = qb_signals[asset_name].loc[prev_date]
                    
                    if qb_prev == 1:
                        score = ranked_assets[asset_name]
                        bullish_top3.append((asset_name, score))
        
        # ============================================================
        # ALLOCATION LOGIC (matching PineScript lines 332-357)
        # ============================================================
        
        allocation = {}
        primary_holding = None
        
        # Determine weights for bullish top3
        final_alloc_weights = {}
        
        for rank_idx, (asset_name, score) in enumerate(bullish_top3[:3]):
            if rank_idx < len(alloc_weights) and alloc_weights[rank_idx] > 0:
                final_alloc_weights[asset_name] = alloc_weights[rank_idx]
        
        # Calculate total allocated
        total_allocated = sum(final_alloc_weights.values())
        
        if total_allocated == 0:
            # No bullish crypto assets -> GOLD or CASH
            if i > 0:
                prev_date = common_index[i-1]
                gold_qb_prev = gold_qb.loc[prev_date]
                
                if use_gold and gold_qb_prev == 1:
                    allocation['GOLD'] = 1.0
                    primary_holding = 'GOLD'
                else:
                    allocation['CASH'] = 1.0
                    primary_holding = 'CASH'
            else:
                allocation['CASH'] = 1.0
                primary_holding = 'CASH'
        else:
            # Allocate to bullish assets
            for asset_name, weight in final_alloc_weights.items():
                allocation[asset_name] = weight
            
            # Remainder to CASH
            if total_allocated < 1.0:
                allocation['CASH'] = round(1.0 - total_allocated, 10)
            
            # Primary = highest weighted asset
            primary_holding = max(final_alloc_weights, key=final_alloc_weights.get)
        
        # Record allocation
        alloc_hist.append(primary_holding if primary_holding else 'CASH')
        
        # ============================================================
        # CALCULATE WEIGHTED RETURN
        # ============================================================
        
        if i == 0:
            weighted_return = 0
        else:
            weighted_return = 0
            
            for asset_name, weight in allocation.items():
                if weight == 0:
                    continue
                
                if asset_name == 'CASH':
                    asset_return = 0
                elif asset_name == 'GOLD':
                    asset_return = (gold_close.iloc[i] - gold_close.iloc[i-1]) / gold_close.iloc[i-1]
                elif asset_name in close_prices:
                    asset_return = (close_prices[asset_name].iloc[i] - close_prices[asset_name].iloc[i-1]) / close_prices[asset_name].iloc[i-1]
                else:
                    asset_return = 0
                
                weighted_return += asset_return * weight
        
        # Update equity
        if i > 0:
            equity.iloc[i] = equity.iloc[i-1] * (1 + weighted_return)
        
        # Track switches
        if primary_holding != prev_primary and prev_primary is not None:
            switches += 1
            if switches <= 10 or i > len(common_index) - 5:
                print(f"[SWITCH #{switches}] {date.date()}: {prev_primary} -> {primary_holding}")
        
        prev_primary = primary_holding
        
        # Debug first few days
        if i < 5:
            print(f"\n[DAY {i}] {date.date()}")
            print(f"  Tournament top3: {top3_assets}")
            if i > 0:
                print(f"  QB signals (prev bar):")
                for asset in top3_assets:
                    if asset in qb_signals:
                        qb_val = qb_signals[asset].iloc[i-1]
                        print(f"    {asset}: {qb_val}")
            print(f"  Bullish top3: {[name for name, _ in bullish_top3]}")
            print(f"  Allocation: {allocation}")
            print(f"  Primary: {primary_holding}")
            print(f"  Weighted return: {weighted_return*100:.2f}%")
            print(f"  Equity: {equity.iloc[i]:.4f}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("MAJORSYNC SUMMARY")
    print(f"{'='*60}")
    print(f"Total switches: {switches}")
    print(f"Switch rate: {switches/len(common_index)*100:.1f}% of days")
    print(f"Final holding: {alloc_hist[-1]}")
    print(f"Final equity: {equity.iloc[-1]:.4f}")
    print(f"Total return: {(equity.iloc[-1] - 1) * 100:.2f}%")
    
    # Allocation distribution
    alloc_counts = Counter(alloc_hist)
    print(f"\nAllocation Distribution:")
    print("-" * 40)
    for asset, count in alloc_counts.most_common():
        pct = count / len(alloc_hist) * 100
        print(f"  {asset:6s}: {count:4d} days ({pct:5.1f}%)")
    
    print(f"{'='*60}\n")
    
    return equity.ffill(), alloc_hist, switches


def compute_metrics(equity: pd.Series) -> dict:
    """Calculate comprehensive performance metrics"""
    days = len(equity)
    
    if days < 2:
        return {
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Omega": 0.0,
            "MaxDD": 0.0,
            "NetProfit": 0.0,
            "Days": days
        }
    
    final_equity = equity.iloc[-1]
    
    # CAGR
    years = days / 365.0
    cagr = (final_equity ** (1 / years)) - 1 if final_equity > 0 and years > 0 else -1
    
    # Net profit
    net_profit = (final_equity - 1) * 100
    
    # Returns
    returns = equity.pct_change().dropna()
    
    if returns.empty or returns.std() == 0:
        return {
            "CAGR": cagr,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Omega": 0.0,
            "MaxDD": 0.0,
            "NetProfit": net_profit,
            "Days": days
        }
    
    # Sharpe ratio (annualized)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0
    
    # Sortino ratio (annualized)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if not downside_returns.empty else 1e-6
    sortino = (returns.mean() / downside_std) * np.sqrt(365)
    
    # Maximum drawdown
    cummax = equity.cummax()
    drawdowns = (equity / cummax) - 1
    max_dd = drawdowns.min() * 100
    
    # Omega ratio
    positive_returns = returns[returns > 0].sum()
    negative_returns = abs(returns[returns <= 0].sum())
    omega = positive_returns / negative_returns if negative_returns > 0 else float('inf')
    
    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Omega": float(omega),
        "MaxDD": float(max_dd),
        "NetProfit": float(net_profit),
        "Days": int(days)
    }