"""Momentum Scanner for identifying high-value training days and reset points."""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import databento as db
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data.utils.helpers import ensure_timezone_aware


class MomentumScanner:
    """Scans historical data to identify momentum days and quality reset points using activity-based scoring."""
    
    # Momentum day thresholds
    MIN_DAILY_MOVE = 0.10  # 10% minimum intraday movement
    MAX_DAILY_MOVE = 0.30  # 30% maximum (beyond this might be too volatile)
    MIN_VOLUME_MULTIPLIER = 2.0  # Minimum 2x average volume
    
    # Reset point parameters
    MIN_RESET_POINTS_PER_DAY = 20  # Minimum reset points for training coverage
    MAX_RESET_POINTS_PER_DAY = 50  # Maximum reset points per day
    TARGET_RESET_POINTS = 30  # Target number of reset points
    
    # Activity scoring parameters
    VOLUME_RATIO_WEIGHT = 0.5  # Weight for volume ratio in activity score
    PRICE_CHANGE_WEIGHT = 0.5  # Weight for price change in activity score
    
    # Directional filtering
    DIRECTION_FILTER = 'both'  # Options: 'front_side', 'back_side', 'both'
    
    def __init__(self, data_dir: str, output_dir: str, 
                 direction_filter: str = 'both',
                 logger: Optional[logging.Logger] = None):
        """Initialize the momentum scanner.
        
        Args:
            data_dir: Directory containing Databento files
            output_dir: Directory to save index files
            direction_filter: Filter for momentum direction ('front_side', 'back_side', 'both')
            logger: Optional logger instance
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        self.direction_filter = direction_filter
        
        # Index paths
        self.day_index_path = self.output_dir / "momentum_days.parquet"
        self.reset_index_path = self.output_dir / "reset_points.parquet"
        
        # Cache for average volumes
        self._avg_volume_cache: Dict[str, float] = {}
        
    def scan_all_symbols(self, symbols: Optional[List[str]] = None, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        max_workers: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scan all symbols for momentum days and reset points.
        
        Args:
            symbols: List of symbols to scan (None for all)
            start_date: Start date for scanning
            end_date: End date for scanning
            max_workers: Number of parallel workers
            
        Returns:
            Tuple of (day_index_df, reset_points_df)
        """
        # Discover available symbols if not provided
        if symbols is None:
            symbols = self._discover_symbols()
            
        self.logger.info(f"Scanning {len(symbols)} symbols for momentum days...")
        
        # Process symbols in parallel
        all_day_records = []
        all_reset_records = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit scanning tasks
            future_to_symbol = {
                executor.submit(self._scan_symbol, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Process results as they complete
            with tqdm(total=len(symbols), desc="Scanning symbols") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        day_records, reset_records = future.result()
                        all_day_records.extend(day_records)
                        all_reset_records.extend(reset_records)
                    except Exception as e:
                        self.logger.error(f"Error scanning {symbol}: {e}")
                    finally:
                        pbar.update(1)
        
        # Create DataFrames
        day_df = pd.DataFrame(all_day_records)
        reset_df = pd.DataFrame(all_reset_records)
        
        # Save indices
        if not day_df.empty:
            day_df.to_parquet(self.day_index_path)
            self.logger.info(f"Saved {len(day_df)} momentum days to {self.day_index_path}")
            
        if not reset_df.empty:
            reset_df.to_parquet(self.reset_index_path)
            self.logger.info(f"Saved {len(reset_df)} reset points to {self.reset_index_path}")
            
        return day_df, reset_df
    
    def _discover_symbols(self) -> List[str]:
        """Discover all available symbols from data directory."""
        symbols = set()
        
        # Walk through data directory looking for metadata files
        for metadata_path in self.data_dir.rglob("metadata.json"):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    query_symbols = metadata.get('query', {}).get('symbols', [])
                    symbols.update(str(s).upper() for s in query_symbols)
            except Exception as e:
                self.logger.warning(f"Error reading {metadata_path}: {e}")
                
        return sorted(list(symbols))
    
    def _scan_symbol(self, symbol: str, start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> Tuple[List[Dict], List[Dict]]:
        """Scan a single symbol for momentum days and reset points.
        
        Returns:
            Tuple of (day_records, reset_records)
        """
        day_records = []
        reset_records = []
        
        # Get 1-second OHLCV data for the symbol
        ohlcv_files = self._find_ohlcv_files(symbol, '1s')
        if not ohlcv_files:
            self.logger.warning(f"No 1s OHLCV data found for {symbol}")
            return day_records, reset_records
            
        # Get average daily volume for comparison
        avg_volume = self._get_average_volume(symbol)
        
        # Process each file (typically contains multiple days)
        for file_path in ohlcv_files:
            try:
                # Load data
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()
                
                if df.empty:
                    continue
                    
                # Ensure proper datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                    
                # Filter by date range if provided
                if start_date:
                    df = df[df.index >= ensure_timezone_aware(start_date)]
                if end_date:
                    df = df[df.index <= ensure_timezone_aware(end_date)]
                    
                if df.empty:
                    continue
                    
                # Group by date and analyze each day
                for date, day_data in df.groupby(df.index.date):
                    day_record, day_reset_points = self._analyze_day(
                        symbol, date, day_data, avg_volume
                    )
                    
                    if day_record:
                        day_records.append(day_record)
                        reset_records.extend(day_reset_points)
                        
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                
        return day_records, reset_records
    
    def _analyze_day(self, symbol: str, date: pd.Timestamp, 
                    day_data: pd.DataFrame, avg_volume: float) -> Tuple[Optional[Dict], List[Dict]]:
        """Analyze a single day for momentum characteristics using activity-based scoring.
        
        Returns:
            Tuple of (day_record, reset_points) or (None, [])
        """
        # Filter to regular trading hours (4 AM - 8 PM ET)
        day_data = self._filter_trading_hours(day_data)
        if len(day_data) < 100:  # Need sufficient data points
            return None, []
            
        # Calculate daily metrics
        open_price = day_data['open'].iloc[0]
        close_price = day_data['close'].iloc[-1]
        high_price = day_data['high'].max()
        low_price = day_data['low'].min()
        total_volume = day_data['volume'].sum()
        
        # Calculate price movement
        daily_return = (close_price - open_price) / open_price
        max_move_up = (high_price - open_price) / open_price
        max_move_down = (open_price - low_price) / open_price
        max_intraday_move = max(max_move_up, max_move_down)
        
        # Determine momentum direction
        is_front_side = daily_return > 0.05  # 5% positive move
        is_back_side = daily_return < -0.05  # 5% negative move
        
        # Apply direction filter
        if self.direction_filter == 'front_side' and not is_front_side:
            return None, []
        elif self.direction_filter == 'back_side' and not is_back_side:
            return None, []
        
        # Check momentum criteria
        is_momentum_day = (
            self.MIN_DAILY_MOVE <= max_intraday_move <= self.MAX_DAILY_MOVE and
            total_volume >= avg_volume * self.MIN_VOLUME_MULTIPLIER
        )
        
        if not is_momentum_day:
            return None, []
            
        # Find halts
        halt_count = self._count_halts(symbol, date)
        
        # Calculate activity score (simple and effective)
        volume_ratio = min(total_volume / avg_volume, 10.0) / 10.0  # Normalize to 0-1
        price_change = min(max_intraday_move, 0.30) / 0.30  # Normalize to 0-1
        activity_score = (volume_ratio * self.VOLUME_RATIO_WEIGHT + 
                         price_change * self.PRICE_CHANGE_WEIGHT)
        
        # Create day record
        day_record = {
            'symbol': symbol,
            'date': pd.Timestamp(date),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': total_volume,
            'daily_return': daily_return,
            'max_intraday_move': max_intraday_move,
            'volume_multiplier': total_volume / avg_volume,
            'halt_count': halt_count,
            'activity_score': activity_score,
            'is_front_side': is_front_side,
            'is_back_side': is_back_side,
            'file_paths': [str(f) for f in self._find_day_files(symbol, date)]
        }
        
        # Find reset points within the day
        reset_points = self._find_reset_points(symbol, date, day_data, activity_score)
        
        return day_record, reset_points
    
    def _find_reset_points(self, symbol: str, date: pd.Timestamp,
                          day_data: pd.DataFrame, day_activity_score: float) -> List[Dict]:
        """Find reset points within a momentum day using activity-based scoring."""
        reset_points = []
        
        # Resample to 1-minute bars for activity calculation
        minute_bars = day_data.resample('1min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(minute_bars) < 60:  # Need at least 1 hour of data
            return reset_points
            
        # Calculate rolling metrics for activity scoring
        minute_bars['volume_sma'] = minute_bars['volume'].rolling(10).mean()
        minute_bars['volume_ratio'] = minute_bars['volume'] / minute_bars['volume_sma']
        minute_bars['price_change'] = minute_bars['close'].pct_change(5).abs()  # 5-minute price change
        
        # Calculate activity score for each minute
        minute_bars['activity_score'] = (
            minute_bars['volume_ratio'].clip(0, 5) / 5 * self.VOLUME_RATIO_WEIGHT +
            minute_bars['price_change'].clip(0, 0.05) / 0.05 * self.PRICE_CHANGE_WEIGHT
        )
        
        # Add direction information
        minute_bars['price_direction'] = minute_bars['close'].pct_change(5)
        minute_bars['is_positive_move'] = minute_bars['price_direction'] > 0.001
        minute_bars['is_negative_move'] = minute_bars['price_direction'] < -0.001
        
        # Filter out low activity periods
        minute_bars = minute_bars[minute_bars['activity_score'] > 0.1]
        
        if len(minute_bars) < self.MIN_RESET_POINTS_PER_DAY:
            # If not enough high-activity points, use all available
            minute_bars = day_data.resample('1min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            minute_bars['activity_score'] = 0.5  # Default score
            minute_bars['is_positive_move'] = True
            minute_bars['is_negative_move'] = False
        
        # Generate reset points using cumulative distribution weighted by activity
        reset_points = self._generate_activity_weighted_reset_points(
            minute_bars, symbol, date, day_activity_score
        )
        
        return reset_points
    
    def _generate_activity_weighted_reset_points(self, minute_bars: pd.DataFrame,
                                               symbol: str, date: pd.Timestamp,
                                               day_activity_score: float) -> List[Dict]:
        """Generate reset points using activity-weighted distribution."""
        reset_points = []
        
        # Clean data
        minute_bars = minute_bars.dropna()
        if len(minute_bars) == 0:
            return reset_points
            
        # Calculate cumulative activity scores
        activity_scores = minute_bars['activity_score'].fillna(0.1).values
        cumulative_activity = np.cumsum(activity_scores)
        total_activity = cumulative_activity[-1]
        
        if total_activity == 0:
            # Fallback to uniform distribution
            indices = np.linspace(0, len(minute_bars) - 1, self.TARGET_RESET_POINTS, dtype=int)
        else:
            # Generate reset points weighted by activity
            target_points = min(self.TARGET_RESET_POINTS, len(minute_bars))
            reset_thresholds = np.linspace(0, total_activity, target_points + 1)[1:]
            
            indices = []
            for threshold in reset_thresholds:
                idx = np.searchsorted(cumulative_activity, threshold)
                if idx < len(minute_bars):
                    indices.append(idx)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            indices = unique_indices
        
        # Create reset points from selected indices
        for idx in indices:
            if idx >= len(minute_bars):
                continue
                
            row = minute_bars.iloc[idx]
            
            reset_points.append({
                'symbol': symbol,
                'date': date,
                'timestamp': row.name,  # Index is timestamp
                'activity_score': row.get('activity_score', 0.5),
                'day_activity_score': day_activity_score,
                'combined_score': row.get('activity_score', 0.5) * day_activity_score,
                'price': row['close'],
                'volume': row['volume'],
                'is_positive_move': row.get('is_positive_move', True),
                'is_negative_move': row.get('is_negative_move', False),
                'volume_ratio': row.get('volume_ratio', 1.0),
                'price_change': row.get('price_change', 0.0)
            })
        
        # Ensure minimum reset points
        if len(reset_points) < self.MIN_RESET_POINTS_PER_DAY and len(minute_bars) > self.MIN_RESET_POINTS_PER_DAY:
            # Add uniformly distributed points to reach minimum
            additional_needed = self.MIN_RESET_POINTS_PER_DAY - len(reset_points)
            used_timestamps = {p['timestamp'] for p in reset_points}
            
            # Find unused timestamps
            all_timestamps = set(minute_bars.index)
            unused_timestamps = sorted(all_timestamps - used_timestamps)
            
            if unused_timestamps:
                # Select evenly spaced additional points
                step = max(1, len(unused_timestamps) // additional_needed)
                for i in range(0, len(unused_timestamps), step):
                    if len(reset_points) >= self.MIN_RESET_POINTS_PER_DAY:
                        break
                        
                    timestamp = unused_timestamps[i]
                    row = minute_bars.loc[timestamp]
                    
                    reset_points.append({
                        'symbol': symbol,
                        'date': date,
                        'timestamp': timestamp,
                        'activity_score': 0.3,  # Lower score for filler points
                        'day_activity_score': day_activity_score,
                        'combined_score': 0.3 * day_activity_score,
                        'price': row['close'],
                        'volume': row['volume'],
                        'is_positive_move': row.get('is_positive_move', True),
                        'is_negative_move': row.get('is_negative_move', False),
                        'volume_ratio': row.get('volume_ratio', 1.0),
                        'price_change': row.get('price_change', 0.0)
                    })
        
        # Sort by timestamp
        reset_points.sort(key=lambda x: x['timestamp'])
        
        # Limit to maximum
        return reset_points[:self.MAX_RESET_POINTS_PER_DAY]
    
    def _is_within_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within trading hours (4 AM - 8 PM ET)."""
        # Convert to ET for checking  
        try:
            et_time = timestamp.tz_convert('US/Eastern').time()
            return pd.Timestamp('04:00').time() <= et_time <= pd.Timestamp('20:00').time()
        except:
            # Fallback for timezone issues
            return True
    
    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to regular trading hours (4 AM - 8 PM ET)."""
        # Convert to ET timezone for filtering
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert('America/New_York')
        
        # Filter to 4 AM - 8 PM ET
        mask = (df_et.index.hour >= 4) & (df_et.index.hour < 20)
        return df[mask]
    
    def _get_average_volume(self, symbol: str) -> float:
        """Get average daily volume for a symbol."""
        if symbol in self._avg_volume_cache:
            return self._avg_volume_cache[symbol]
            
        # Calculate from daily bars if available
        daily_files = self._find_ohlcv_files(symbol, '1d')
        if daily_files:
            try:
                volumes = []
                for file_path in daily_files[-5:]:  # Last 5 files
                    store = db.DBNStore.from_file(file_path)
                    df = store.to_df()
                    if not df.empty and 'volume' in df.columns:
                        volumes.extend(df['volume'].values)
                        
                if volumes:
                    avg_vol = np.median(volumes)  # Use median to handle outliers
                    self._avg_volume_cache[symbol] = avg_vol
                    return avg_vol
            except Exception as e:
                self.logger.warning(f"Error calculating average volume for {symbol}: {e}")
                
        # Default fallback
        return 1_000_000
    
    def _count_halts(self, symbol: str, date: pd.Timestamp) -> int:
        """Count trading halts for a symbol on a given date."""
        halt_count = 0
        
        # Look for status files
        status_files = self._find_status_files(symbol, date)
        for file_path in status_files:
            try:
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()
                
                # Filter to the specific date
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                    
                df = df[df.index.date == date]
                
                # Count halt events
                if 'action' in df.columns:
                    # Databento status codes: 8=HALTED, 9=PAUSED, 10=SUSPENDED
                    halt_count += df['action'].isin([8, 9, 10]).sum()
                    
            except Exception as e:
                self.logger.warning(f"Error counting halts from {file_path}: {e}")
                
        return halt_count
    
    
    def _find_ohlcv_files(self, symbol: str, timeframe: str) -> List[Path]:
        """Find OHLCV files for a symbol and timeframe."""
        files = []
        pattern = f"*{symbol.lower()}*.ohlcv-{timeframe}.dbn*"
        
        for file_path in self.data_dir.rglob(pattern):
            files.append(file_path)
            
        return sorted(files)
    
    def _find_status_files(self, symbol: str, date: pd.Timestamp) -> List[Path]:
        """Find status files for a symbol on a date."""
        files = []
        date_str = date.strftime('%Y%m%d')
        pattern = f"*{date_str}*.status.dbn*"
        
        for file_path in self.data_dir.rglob(pattern):
            # Verify it's for the right symbol
            if self._file_contains_symbol(file_path, symbol):
                files.append(file_path)
                
        return files
    
    def _find_day_files(self, symbol: str, date: pd.Timestamp) -> List[Path]:
        """Find all data files for a symbol on a specific date."""
        files = []
        date_str = date.strftime('%Y%m%d')
        
        # Look for files with the date in the name
        for pattern in [f"*{date_str}*.dbn*", f"*{symbol.lower()}*.dbn*"]:
            for file_path in self.data_dir.rglob(pattern):
                if self._file_contains_symbol(file_path, symbol):
                    files.append(file_path)
                    
        return list(set(files))  # Remove duplicates
    
    def _file_contains_symbol(self, file_path: Path, symbol: str) -> bool:
        """Check if a file contains data for a specific symbol."""
        # Check parent directory for metadata
        metadata_path = file_path.parent / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    symbols = metadata.get('query', {}).get('symbols', [])
                    return symbol.upper() in [str(s).upper() for s in symbols]
            except:
                pass
                
        # Fallback to filename check
        return symbol.lower() in str(file_path).lower()
    
    def load_index(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load existing momentum indices from disk.
        
        Returns:
            Tuple of (day_index_df, reset_points_df)
        """
        day_df = pd.DataFrame()
        reset_df = pd.DataFrame()
        
        if self.day_index_path.exists():
            day_df = pd.read_parquet(self.day_index_path)
            self.logger.info(f"Loaded {len(day_df)} momentum days from index")
            
        if self.reset_index_path.exists():
            reset_df = pd.read_parquet(self.reset_index_path)
            self.logger.info(f"Loaded {len(reset_df)} reset points from index")
            
        return day_df, reset_df
    
    def query_momentum_days(self, symbol: str, min_activity: float = 0.5,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           direction: Optional[str] = None) -> pd.DataFrame:
        """Query momentum days for a symbol.
        
        Args:
            symbol: Symbol to query
            min_activity: Minimum activity score
            start_date: Start date filter
            end_date: End date filter
            direction: Filter by direction ('front_side', 'back_side', None for both)
            
        Returns:
            DataFrame of matching momentum days
        """
        day_df, _ = self.load_index()
        
        if day_df.empty:
            return pd.DataFrame()
            
        # Filter by symbol
        mask = day_df['symbol'] == symbol.upper()
        
        # Filter by activity score
        mask &= day_df['activity_score'] >= min_activity
        
        # Filter by direction
        if direction == 'front_side':
            mask &= day_df['is_front_side'] == True
        elif direction == 'back_side':
            mask &= day_df['is_back_side'] == True
        
        # Filter by date range
        if start_date:
            mask &= day_df['date'] >= pd.Timestamp(start_date)
        if end_date:
            mask &= day_df['date'] <= pd.Timestamp(end_date)
            
        return day_df[mask].sort_values('activity_score', ascending=False)
    
    def query_reset_points(self, symbol: str, date: Optional[pd.Timestamp] = None,
                          min_activity: float = 0.5,
                          direction: Optional[str] = None) -> pd.DataFrame:
        """Query reset points for a symbol.
        
        Args:
            symbol: Symbol to query
            date: Specific date (None for all)
            min_activity: Minimum combined activity score
            direction: Filter by direction ('positive', 'negative', None for both)
            
        Returns:
            DataFrame of matching reset points
        """
        _, reset_df = self.load_index()
        
        if reset_df.empty:
            return pd.DataFrame()
            
        # Filter by symbol
        mask = reset_df['symbol'] == symbol.upper()
        
        # Filter by date
        if date:
            mask &= reset_df['date'] == date
            
        # Filter by activity
        mask &= reset_df['combined_score'] >= min_activity
        
        # Filter by direction
        if direction == 'positive':
            mask &= reset_df['is_positive_move'] == True
        elif direction == 'negative':
            mask &= reset_df['is_negative_move'] == True
        
        return reset_df[mask].sort_values('combined_score', ascending=False)
    
    def get_top_momentum_days(self, symbol: str, top_n: int = 10,
                             direction: Optional[str] = None) -> pd.DataFrame:
        """Get top N momentum days by activity score.
        
        Args:
            symbol: Symbol to query
            top_n: Number of top days to return
            direction: Filter by direction ('front_side', 'back_side', None for both)
            
        Returns:
            DataFrame of top momentum days
        """
        days = self.query_momentum_days(symbol, min_activity=0.0, direction=direction)
        return days.head(top_n)
    
    def get_momentum_day_summary(self) -> pd.DataFrame:
        """Get summary statistics for all momentum days.
        
        Returns:
            DataFrame with summary by symbol
        """
        day_df, _ = self.load_index()
        
        if day_df.empty:
            return pd.DataFrame()
            
        summary = day_df.groupby('symbol').agg({
            'date': 'count',
            'activity_score': ['mean', 'max'],
            'max_intraday_move': ['mean', 'max'],
            'volume_multiplier': ['mean', 'max'],
            'is_front_side': 'sum',
            'is_back_side': 'sum'
        })
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={
            'date_count': 'total_days',
            'activity_score_mean': 'avg_activity',
            'activity_score_max': 'max_activity',
            'max_intraday_move_mean': 'avg_move',
            'max_intraday_move_max': 'max_move',
            'volume_multiplier_mean': 'avg_volume_mult',
            'volume_multiplier_max': 'max_volume_mult',
            'is_front_side_sum': 'front_side_days',
            'is_back_side_sum': 'back_side_days'
        })
        
        return summary.sort_values('total_days', ascending=False)