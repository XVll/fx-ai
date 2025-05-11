# data/utils/indicators.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Union

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        series: Price series
        period: EMA period

    Returns:
        Series with EMA values
    """
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(prices: pd.Series,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fast_ema = calculate_ema(prices, fast_period)
    slow_ema = calculate_ema(prices, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def calculate_vwap(bars_df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume-Weighted Average Price.

    Args:
        bars_df: DataFrame with OHLCV bars

    Returns:
        Series with VWAP values
    """
    if bars_df.empty:
        return pd.Series(dtype=float)

    # Calculate typical price
    if all(col in bars_df.columns for col in ['high', 'low', 'close']):
        typical_price = (bars_df['high'] + bars_df['low'] + bars_df['close']) / 3
    else:
        # Fallback to close if H/L not available
        typical_price = bars_df['close']

    # Calculate VWAP
    if 'volume' in bars_df.columns:
        cumulative_tp_vol = (typical_price * bars_df['volume']).cumsum()
        cumulative_vol = bars_df['volume'].cumsum()

        # Avoid division by zero
        vwap = np.where(cumulative_vol > 0, cumulative_tp_vol / cumulative_vol, typical_price)
        return pd.Series(vwap, index=bars_df.index)
    else:
        # Fallback if volume not available
        return typical_price

def calculate_support_resistance(bars_df: pd.DataFrame,
                                 window: int = 20,
                                 threshold: float = 0.005) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate dynamic support and resistance levels.

    Args:
        bars_df: DataFrame with OHLCV bars
        window: Lookback window for pivots
        threshold: Minimum price change threshold (as fraction)

    Returns:
        Tuple of (Support levels, Resistance levels)
    """
    if bars_df.empty or 'high' not in bars_df.columns or 'low' not in bars_df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Initialize output series
    support = pd.Series(index=bars_df.index)
    resistance = pd.Series(index=bars_df.index)

    # Find pivot lows (support)
    for i in range(window, len(bars_df) - window):
        # Check if this is a local low
        if all(bars_df.iloc[i-window:i]['low'] >= bars_df.iloc[i]['low']) and \
                all(bars_df.iloc[i+1:i+window+1]['low'] > bars_df.iloc[i]['low']):
            support.iloc[i] = bars_df.iloc[i]['low']

    # Find pivot highs (resistance)
    for i in range(window, len(bars_df) - window):
        # Check if this is a local high
        if all(bars_df.iloc[i-window:i]['high'] <= bars_df.iloc[i]['high']) and \
                all(bars_df.iloc[i+1:i+window+1]['high'] < bars_df.iloc[i]['high']):
            resistance.iloc[i] = bars_df.iloc[i]['high']

    # Forward fill support and resistance levels
    support = support.ffill()
    resistance = resistance.ffill()

    return support, resistance

def calculate_half_dollar_levels(price_range: Tuple[float, float],
                                 include_quarters: bool = True) -> List[float]:
    """
    Calculate whole and half dollar price levels within a range.

    Args:
        price_range: Tuple of (min_price, max_price)
        include_quarters: Whether to include quarter dollar levels

    Returns:
        List of price levels
    """
    min_price, max_price = price_range

    # Find the nearest whole dollar below min_price
    start_dollar = np.floor(min_price)
    # Find the nearest whole dollar above max_price
    end_dollar = np.ceil(max_price)

    levels = []

    # Generate whole dollar levels
    dollar_levels = np.arange(start_dollar, end_dollar + 1, 1.0)
    levels.extend(dollar_levels)

    # Generate half dollar levels
    half_dollar_levels = np.arange(start_dollar + 0.5, end_dollar, 1.0)
    levels.extend(half_dollar_levels)

    # Generate quarter dollar levels if requested
    if include_quarters:
        quarter_levels = np.arange(start_dollar + 0.25, end_dollar, 0.5)
        three_quarter_levels = np.arange(start_dollar + 0.75, end_dollar, 0.5)
        levels.extend(quarter_levels)
        levels.extend(three_quarter_levels)

    # Sort and filter levels within range
    levels = sorted([level for level in levels if min_price <= level <= max_price])

    return levels