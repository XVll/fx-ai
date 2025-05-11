#!/usr/bin/env python
"""
Process 2025 Data

This script processes data from the correct 2025 dates based on the files you have
"""
import os
import sys
import argparse
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

# Import Databento
try:
    import databento as db

    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    print("ERROR: Databento package is required! Install with: pip install databento")
    sys.exit(1)


class DBNReader:
    """Reader for DBN files"""

    def __init__(self, data_dir):
        """Initialize with data directory"""
        self.data_dir = data_dir
        self.file_paths = []

        # Scan for files
        self._scan_files()

    def _scan_files(self):
        """Scan for all .dbn.zst files in directory and subdirectories"""
        print(f"Scanning for compressed DBN files in {self.data_dir}")

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.dbn.zst'):
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)

        print(f"\nFound {len(self.file_paths)} compressed DBN files")

        # Print the first few files
        if self.file_paths:
            print("Sample files:")
            for file in self.file_paths[:5]:
                print(f"  {os.path.basename(file)}")
            if len(self.file_paths) > 5:
                print(f"  ... and {len(self.file_paths) - 5} more")

    def inspect_file(self, file_path):
        """Inspect a specific file's content"""
        try:
            print(f"\nInspecting: {os.path.basename(file_path)}")

            # Try to open the file
            store = db.DBNStore.from_file(file_path)

            # Print basic info
            print(f"  Scheme type: {store.schema}")
            print(f"  Symbols: {store.symbols}")

            # Try to get data
            df = store.to_df()

            if df.empty:
                print("  DataFrame is empty")
            else:
                print(f"  DataFrame shape: {df.shape}")
                print(f"  Index type: {type(df.index).__name__}")
                print(f"  Time range: {df.index.min()} to {df.index.max()}")
                print(f"  Columns: {', '.join(df.columns[:9])}" +
                      (f", ..." if len(df.columns) > 9 else ""))

                # Print a few rows
                print("\n  Sample data:")
                print(df.head(3).to_string(index=False))

            return df

        except Exception as e:
            print(f"  Error inspecting file: {e}")
            traceback.print_exc()
            return None

    def extract_trade_data(self, date_to_extract, symbol=None):
        """Extract trade data for a specific date"""
        print(f"\nExtracting trade data for {date_to_extract}...")

        # Make sure date_to_extract is a date object
        if isinstance(date_to_extract, str):
            date_to_extract = datetime.strptime(date_to_extract, '%Y-%m-%d').date()

        # Trading day start and end times (US Eastern time)
        start_time = datetime.combine(date_to_extract, datetime.strptime("09:30:00", "%H:%M:%S").time())
        end_time = datetime.combine(date_to_extract, datetime.strptime("16:00:00", "%H:%M:%S").time())

        # Try to make these timezone-aware
        try:
            start_time_utc = pd.Timestamp(start_time).tz_localize('America/New_York').tz_convert('UTC')
            end_time_utc = pd.Timestamp(end_time).tz_localize('America/New_York').tz_convert('UTC')
            use_tz = True
        except:
            start_time_utc = start_time
            end_time_utc = end_time
            use_tz = False

        # Format date for filename matching
        date_str = date_to_extract.strftime('%Y%m%d')

        # Find trade files for this date
        trade_files = [f for f in self.file_paths if
                       'trades' in os.path.basename(f) and date_str in os.path.basename(f)]
        print(f"  Found {len(trade_files)} trade files for {date_str}")

        # If no exact date match, try looking for files with date ranges
        if not trade_files:
            range_files = [f for f in self.file_paths if 'trades' in os.path.basename(f)]
            for file in range_files:
                # Check if file might contain our date
                try:
                    # Try to extract date range from filename
                    basename = os.path.basename(file)
                    parts = basename.split('.')[0].split('-')
                    dates = [p for p in parts if len(p) == 8 and p.isdigit()]

                    if len(dates) >= 2:
                        start_date = datetime.strptime(dates[0], '%Y%m%d').date()
                        end_date = datetime.strptime(dates[1], '%Y%m%d').date()
                        if start_date <= date_to_extract <= end_date:
                            trade_files.append(file)
                except:
                    # If can't extract from filename, we'll check file contents
                    pass

        if not trade_files:
            print("  No trade files found for this date. Will check all files.")
            trade_files = [f for f in self.file_paths if 'trades' in os.path.basename(f)]

        # Extract trades from each file
        all_trades = []

        for file_path in trade_files:
            basename = os.path.basename(file_path)
            print(f"  Checking {basename}...")

            try:
                # Read the file
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                if df.empty:
                    print(f"    No data in {basename}")
                    continue

                # Filter for the symbol if provided
                if symbol and 'symbol' in df.columns:
                    df = df[df['symbol'] == symbol]

                # Make sure index is timezone-aware if needed
                if use_tz and df.index.tz is None:
                    print("    Converting index to UTC timezone")
                    df.index = pd.to_datetime(df.index).tz_localize('UTC')

                # Filter for the date range
                mask = (df.index >= start_time_utc) & (df.index <= end_time_utc)
                filtered_df = df[mask]

                if filtered_df.empty:
                    print(f"    No trades in {basename} for the specified time range")
                else:
                    print(f"    Found {len(filtered_df)} trades in {basename}")
                    filtered_df = filtered_df.copy()  # Make a copy to avoid view vs copy warning
                    all_trades.append(filtered_df)

            except Exception as e:
                print(f"    Error processing {basename}: {e}")

        # Combine all trades
        if not all_trades:
            print("  No trades found for the specified date")
            return None

        trades_df = pd.concat(all_trades)
        trades_df = trades_df.sort_index()

        print(f"  Total trades: {len(trades_df)}")

        # Show some statistics
        if not trades_df.empty:
            print("\nTrade statistics:")

            if 'price' in trades_df.columns:
                print(f"  Min price: {trades_df['price'].min()}")
                print(f"  Max price: {trades_df['price'].max()}")
                print(f"  Avg price: {trades_df['price'].mean():.4f}")

            if 'size' in trades_df.columns:
                print(f"  Min size: {trades_df['size'].min()}")
                print(f"  Max size: {trades_df['size'].max()}")
                print(f"  Avg size: {trades_df['size'].mean():.1f}")
                print(f"  Total volume: {trades_df['size'].sum()}")

        return trades_df

    def extract_ohlc_data(self, date_to_extract, timeframe='1m', symbol=None):
        print(f"\nExtracting {timeframe} OHLC data for {date_to_extract}...")

        # Make sure date_to_extract is a date object
        if isinstance(date_to_extract, str):
            date_to_extract = datetime.strptime(date_to_extract, '%Y-%m-%d').date()

        # Trading day start and end times (US Eastern time)
        start_time = datetime.combine(date_to_extract, datetime.strptime("09:30:00", "%H:%M:%S").time())
        end_time = datetime.combine(date_to_extract, datetime.strptime("16:00:00", "%H:%M:%S").time())

        # Try to make these timezone-aware
        try:
            start_time_utc = pd.Timestamp(start_time).tz_localize('America/New_York').tz_convert('UTC')
            end_time_utc = pd.Timestamp(end_time).tz_localize('America/New_York').tz_convert('UTC')
            use_tz = True
        except:
            start_time_utc = start_time
            end_time_utc = end_time
            use_tz = False

        # Format date for filename matching
        date_str = date_to_extract.strftime('%Y%m%d')

        # Find OHLC files for this date and timeframe
        ohlc_files = [f for f in self.file_paths if
                      f'ohlcv-{timeframe}' in os.path.basename(f) and date_str in os.path.basename(f)]

        # If no exact date match, look for files with date ranges
        if not ohlc_files:
            range_files = [f for f in self.file_paths if f'ohlcv-{timeframe}' in os.path.basename(f)]
            for file in range_files:
                # Check if file might contain our date
                try:
                    # Try to extract date range from filename
                    basename = os.path.basename(file)
                    parts = basename.split('.')[0].split('-')
                    dates = [p for p in parts if len(p) == 8 and p.isdigit()]

                    if len(dates) >= 2:
                        start_date = datetime.strptime(dates[0], '%Y%m%d').date()
                        end_date = datetime.strptime(dates[1], '%Y%m%d').date()
                        if start_date <= date_to_extract <= end_date:
                            ohlc_files.append(file)
                except:
                    # If can't extract from filename, we'll check file contents
                    pass

        print(f"  Found {len(ohlc_files)} {timeframe} OHLC files")

        # If no files found and timeframe is 5m, we'll try to create from 1m later
        if not ohlc_files and timeframe != '5m':
            print(f"  No {timeframe} OHLC files found that match the date")
            return None

        # Extract OHLC data from each file
        all_bars = []

        for file_path in ohlc_files:
            basename = os.path.basename(file_path)
            print(f"  Checking {basename}...")

            try:
                # Read the file
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                if df.empty:
                    print(f"    No data in {basename}")
                    continue

                # Filter for the symbol if provided
                if symbol and 'symbol' in df.columns:
                    df = df[df['symbol'] == symbol]

                # Make sure index is timezone-aware if needed
                if use_tz and df.index.tz is None:
                    print("    Converting index to UTC timezone")
                    df.index = pd.to_datetime(df.index).tz_localize('UTC')

                # Filter for the date range
                mask = (df.index >= start_time_utc) & (df.index <= end_time_utc)
                filtered_df = df[mask]

                if filtered_df.empty:
                    print(f"    No bars in {basename} for the specified time range")
                else:
                    print(f"    Found {len(filtered_df)} bars in {basename}")
                    filtered_df = filtered_df.copy()  # Make a copy to avoid view vs copy warning
                    all_bars.append(filtered_df)

            except Exception as e:
                print(f"    Error processing {basename}: {e}")

        # Combine all bars
        if not all_bars:
            # If timeframe is 5m and we couldn't find direct data, try to create from 1m
            if timeframe == '5m':
                print("  Trying to create 5m bars from 1m data...")
                bars_1m = self.extract_ohlc_data(date_to_extract, '1m', symbol)

                if bars_1m is not None and not bars_1m.empty:
                    resampled = bars_1m.resample('5min').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })

                    print(f"  Created {len(resampled)} 5m bars from 1m data")
                    return resampled

            print(f"  No {timeframe} bars found for the specified date")
            return None

        bars_df = pd.concat(all_bars)
        bars_df = bars_df.sort_index()

        print(f"  Total {timeframe} bars: {len(bars_df)}")

        # Show some statistics
        if not bars_df.empty:
            print(f"\n{timeframe} OHLC statistics:")

            if all(col in bars_df.columns for col in ['open', 'high', 'low', 'close']):
                print(f"  Open range: {bars_df['open'].min()} to {bars_df['open'].max()}")
                print(f"  High range: {bars_df['high'].min()} to {bars_df['high'].max()}")
                print(f"  Low range: {bars_df['low'].min()} to {bars_df['low'].max()}")
                print(f"  Close range: {bars_df['close'].min()} to {bars_df['close'].max()}")

            if 'volume' in bars_df.columns:
                print(f"  Min volume: {bars_df['volume'].min()}")
                print(f"  Max volume: {bars_df['volume'].max()}")
                print(f"  Avg volume: {bars_df['volume'].mean():.1f}")
                print(f"  Total volume: {bars_df['volume'].sum()}")

        return bars_df

    def process_trading_day(self, date_to_process, output_dir=None):
        """Process a complete trading day"""
        # Make sure date_to_process is a date object
        if isinstance(date_to_process, str):
            date_to_process = datetime.strptime(date_to_process, '%Y-%m-%d').date()

        date_str = date_to_process.strftime('%Y-%m-%d')
        print(f"\n{'-' * 60}")
        print(f"Processing data for {date_str}")
        print(f"{'-' * 60}")

        # Extract all data types
        bars_1m = self.extract_ohlc_data(date_to_process, '1m')
        bars_5m = self.extract_ohlc_data(date_to_process, '5m')
        trades = self.extract_trade_data(date_to_process)

        return {
            '1m': bars_1m,
            '5m': bars_5m,
            'trades': trades
        }


def main():
    parser = argparse.ArgumentParser(description='Process 2025 Data')
    parser.add_argument('--data-dir', type=str, default='../dnb/Mlgo', help='Directory containing DBN files')
    parser.add_argument('--date', type=str, default='2025-02-03', help='Process a specific date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save outputs')

    args = parser.parse_args()

    # Create the reader
    reader = DBNReader(args.data_dir)

    # Inspect a specific file if requested
    for file in reader.file_paths:
        reader.inspect_file(file)

    # Process the date
    try:
        date = datetime.strptime(args.date, '%Y-%m-%d').date()
        reader.process_trading_day(date, args.output_dir)
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.")


if __name__ == "__main__":
    main()
