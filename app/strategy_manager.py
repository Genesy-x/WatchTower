"""
Strategy Manager - Allows easy swapping between different trading strategies

To add a new strategy:
1. Create a new file in app/strategies/ (e.g., my_strategy.py)
2. Implement a compute_indicators(df) function that returns a DataFrame with:
   - 'TPI': Trend Power Index or equivalent filter signal
   - 'Momentum': Used for ranking assets
   - Other columns as needed
3. Add your strategy to AVAILABLE_STRATEGIES dict below
4. Set ACTIVE_STRATEGY to your strategy name

Required columns for any strategy:
- 'TPI': Filtering signal (positive = bullish, negative = bearish)
- 'Momentum': Used for relative strength ranking
- 'close': Price data (usually already present)
"""

# Import strategy modules here
from app.indicators import compute_indicators as simple_strategy
from app.strategies.qb_strategy import compute_indicators as qb_strategy

# ============================================
# STRATEGY CONFIGURATION
# ============================================

# Available strategies
AVAILABLE_STRATEGIES = {
    'simple': {
        'name': 'Simple Momentum Strategy',
        'function': simple_strategy,
        'description': 'Uses DEMA, VIDYA, ALMA, BBands for signals'
    },
    'qb': {
        'name': 'QB Trading System',
        'function': qb_strategy,
        'description': 'Advanced oscillator + MA system with state preservation'
    }
}

# ============================================
# SET YOUR ACTIVE STRATEGY HERE
# ============================================
ACTIVE_STRATEGY = 'qb'  # Change this to switch strategies: 'simple' or 'qb'
# ============================================

def get_active_strategy():
    """Get the currently active strategy"""
    if ACTIVE_STRATEGY not in AVAILABLE_STRATEGIES:
        raise ValueError(f"Strategy '{ACTIVE_STRATEGY}' not found. Available: {list(AVAILABLE_STRATEGIES.keys())}")
    
    strategy = AVAILABLE_STRATEGIES[ACTIVE_STRATEGY]
    print(f"[STRATEGY] Using: {strategy['name']}")
    print(f"[STRATEGY] Description: {strategy['description']}")
    return strategy['function']

def compute_indicators(df):
    """
    Wrapper function that calls the active strategy's compute_indicators
    This is the main entry point used throughout the application
    """
    strategy_func = get_active_strategy()
    return strategy_func(df)

def list_strategies():
    """List all available strategies"""
    print("\n" + "="*60)
    print("AVAILABLE TRADING STRATEGIES")
    print("="*60)
    for key, strategy in AVAILABLE_STRATEGIES.items():
        active = " [ACTIVE]" if key == ACTIVE_STRATEGY else ""
        print(f"\n{key}{active}:")
        print(f"  Name: {strategy['name']}")
        print(f"  Description: {strategy['description']}")
    print("\n" + "="*60)
    print(f"\nCurrent active strategy: {ACTIVE_STRATEGY}")
    print("="*60 + "\n")

# Print active strategy on import
print(f"\n{'='*60}")
print(f"ACTIVE STRATEGY: {AVAILABLE_STRATEGIES[ACTIVE_STRATEGY]['name']}")
print(f"{'='*60}\n")