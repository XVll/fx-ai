# data/utils/cleaning.py
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional

def clean_ohlc_data(df: pd.DataFrame, price_cols: List[str] = None) -> pd.DataFrame:
    """
    Clean OHLCV data by removing or fixing common issues.
    
    Args:
        df: DataFrame with OHLCV data
        price_cols: List of price columns to clean (defaults to ['open', 'high', 'low', 'close'])
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df

    # Default price columns if not specified
    if price_cols is None:
        price_cols = ['open', 'high', 'low', 'close']

    # Make a copy to avoid modifying the original
    df_clean = df.copy()

    # Remove rows with NaN values in critical columns
    for col in price_cols:
        if col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[col])

    # Remove rows with zero or negative prices
    for col in price_cols:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] > 0]

    # Ensure high >= low
    if 'high' in df_clean.columns and 'low' in df_clean.columns:
        # Find inconsistent rows
        invalid_mask = df_clean['high'] < df_clean['low']
        if invalid_mask.any():
            # For invalid rows, swap high and low
            temp = df_clean.loc[invalid_mask, 'high'].copy()
            df_clean.loc[invalid_mask, 'high'] = df_clean.loc[invalid_mask, 'low']
            df_clean.loc[invalid_mask, 'low'] = temp

    # Ensure high >= open/close >= low
    for col in ['open', 'close']:
        if col in df_clean.columns:
            if 'high' in df_clean.columns:
                df_clean['high'] = np.maximum(df_clean['high'], df_clean[col])
            if 'low' in df_clean.columns:
                df_clean['low'] = np.minimum(df_clean['low'], df_clean[col])

    # Handle volume-related issues
    if 'volume' in df_clean.columns:
        # Remove rows with negative volume
        df_clean = df_clean[df_clean['volume'] >= 0]

        # Replace NaN volume with 0
        df_clean['volume'] = df_clean['volume'].fillna(0)

    # Detect and remove outliers in price (e.g., sudden jumps/drops)
    for col in price_cols:
        if col in df_clean.columns:
            # Calculate price changes
            pct_change = df_clean[col].pct_change().abs()

            # Find jumps more than 50% (extreme outliers)
            outliers = pct_change > 0.5

            # Remove outliers
            df_clean = df_clean[~outliers]

    return df_clean

def clean_trades_data(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean trades data by removing or fixing common issues.
    
    Args:
        trades_df: DataFrame with trades data
        
    Returns:
        Cleaned DataFrame
    """
    if trades_df.empty:
        return trades_df

    # Make a copy to avoid modifying the original
    df_clean = trades_df.copy()

    # Remove rows with NaN or zero/negative prices
    if 'price' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['price'])
        df_clean = df_clean[df_clean['price'] > 0]

    # Remove rows with NaN or negative sizes
    if 'size' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['size'])
        df_clean = df_clean[df_clean['size'] > 0]

    # Remove likely erroneous trades (extremely large jumps)
    if 'price' in df_clean.columns:
        # Calculate price changes
        pct_change = df_clean['price'].pct_change().abs()

        # Find jumps more than 50% (extreme outliers)
        outliers = pct_change > 0.5

        # Remove outliers
        df_clean = df_clean[~outliers]

    return df_clean

def clean_quotes_data(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean quotes data by removing or fixing common issues.
    
    Args:
        quotes_df: DataFrame with quotes data
        
    Returns:
        Cleaned DataFrame
    """
    if quotes_df.empty:
        return quotes_df

    # Make a copy to avoid modifying the original
    df_clean = quotes_df.copy()

    # Remove rows with NaN prices
    price_cols = [col for col in df_clean.columns if '_px_' in col]
    size_cols = [col for col in df_clean.columns if '_sz_' in col]

    # Drop rows with NaN price values
    for col in price_cols:
        df_clean = df_clean.dropna(subset=[col])

    # Ensure prices are positive
    for col in price_cols:
        df_clean = df_clean[df_clean[col] > 0]

    # Ensure sizes are non-negative
    for col in size_cols:
        df_clean = df_clean[df_clean[col] >= 0]

    # Ensure bid < ask for L1 quotes
    if 'bid_px_00' in df_clean.columns and 'ask_px_00' in df_clean.columns:
        # Find quotes where bid >= ask (crossed market)
        crossed = df_clean['bid_px_00'] >= df_clean['ask_px_00']

        # Remove crossed quotes
        df_clean = df_clean[~crossed]

    return df_clean