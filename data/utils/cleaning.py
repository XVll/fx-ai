# data/utils/cleaning.py
import pandas as pd
import numpy as np
from typing import List


def clean_ohlc_data(df: pd.DataFrame, price_cols: List[str] = None) -> pd.DataFrame:
    """
    Clean OHLCV data by removing or fixing common issues.
    Specifically adapted for low float, high volatility momentum stocks.

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
        price_cols = ["open", "high", "low", "close"]

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
    if "high" in df_clean.columns and "low" in df_clean.columns:
        # Find inconsistent rows
        invalid_mask = df_clean["high"] < df_clean["low"]
        if invalid_mask.any():
            # For invalid rows, swap high and low
            temp = df_clean.loc[invalid_mask, "high"].copy()
            df_clean.loc[invalid_mask, "high"] = df_clean.loc[invalid_mask, "low"]
            df_clean.loc[invalid_mask, "low"] = temp

    # Ensure high >= open/close >= low
    for col in ["open", "close"]:
        if col in df_clean.columns:
            if "high" in df_clean.columns:
                df_clean["high"] = np.maximum(df_clean["high"], df_clean[col])
            if "low" in df_clean.columns:
                df_clean["low"] = np.minimum(df_clean["low"], df_clean[col])

    # Handle volume-related issues
    if "volume" in df_clean.columns:
        # Remove rows with negative volume
        df_clean = df_clean[df_clean["volume"] >= 0]
        # Replace NaN volume with 0
        df_clean["volume"] = df_clean["volume"].fillna(0)

    # For low float momentum stocks, we need to be careful with outlier detection
    # since these stocks can have legitimate huge price moves
    # Use a more permissive approach but still catch clear errors
    for col in price_cols:
        if col in df_clean.columns:
            # Calculate price changes
            pct_change = df_clean[col].pct_change().abs()

            # For low float stocks, even 100%+ moves can be legitimate in seconds
            # Only filter out extreme outliers that are likely data errors
            extreme_outliers = pct_change > 5.0  # 500% change as threshold
            if extreme_outliers.any():
                df_clean = df_clean[~extreme_outliers]

    return df_clean


def clean_trades_data(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean trades data by removing or fixing common issues.
    Specifically adapted for low float stocks with high volatility.

    Args:
        trades_df: DataFrame with trades data

    Returns:
        Cleaned DataFrame
    """
    if trades_df.empty:
        return trades_df

    # Make a copy to avoid modifying the original
    df_clean = trades_df.copy()

    # Remove rows with NaN prices
    if "price" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["price"])
        df_clean = df_clean[df_clean["price"] > 0]

    # Remove rows with NaN or negative sizes
    if "size" in df_clean.columns:
        df_clean = df_clean.dropna(subset=["size"])
        df_clean = df_clean[df_clean["size"] > 0]

    # Special logic for momentum stocks with legitimate large price jumps
    # Set a very high threshold for removing price outliers
    if "price" in df_clean.columns:
        # Calculate price jumps
        price_jumps = df_clean["price"].pct_change().abs()

        # Only filter out extreme jumps that are likely data errors
        # For momentum stocks, even 100% jumps can happen in a split second
        extreme_jumps = price_jumps > 5.0  # 500% jump as threshold for removal
        if extreme_jumps.any():
            df_clean = df_clean[~extreme_jumps]

    return df_clean


def clean_quotes_data(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean quotes data by removing or fixing common issues.
    Specifically adapted for low float, high volatility momentum stocks.

    Args:
        quotes_df: DataFrame with quotes data

    Returns:
        Cleaned DataFrame
    """
    if quotes_df.empty:
        return quotes_df

    # Make a copy to avoid modifying the original
    df_clean = quotes_df.copy()

    # Find price and size columns based on standard naming
    price_cols = [
        col for col in df_clean.columns if "price" in col.lower() or "px" in col.lower()
    ]
    size_cols = [
        col for col in df_clean.columns if "size" in col.lower() or "sz" in col.lower()
    ]

    # Drop rows with NaN price values in essential columns
    essential_price_cols = ["bid_price", "ask_price", "bid_px_00", "ask_px_00"]
    essential_cols = [col for col in essential_price_cols if col in df_clean.columns]
    if essential_cols:
        df_clean = df_clean.dropna(subset=essential_cols)

    # Ensure prices are positive
    for col in price_cols:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] > 0]

    # Ensure sizes are non-negative
    for col in size_cols:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] >= 0]

    # Ensure bid < ask for all levels (allowing for zero bid/ask)
    bid_cols = [col for col in price_cols if "bid" in col.lower()]
    ask_cols = [col for col in price_cols if "ask" in col.lower()]

    for bid_col in bid_cols:
        for ask_col in ask_cols:
            # Estimate if they're at the same level by looking at column names
            if (
                bid_col[-2:] == ask_col[-2:]
            ):  # Same level, e.g., bid_px_00 and ask_px_00
                # Find crossed quotes (bid >= ask) where both values are > 0
                crossed = (
                    (df_clean[bid_col] >= df_clean[ask_col])
                    & (df_clean[bid_col] > 0)
                    & (df_clean[ask_col] > 0)
                )
                if crossed.any():
                    df_clean = df_clean[~crossed]

    return df_clean
