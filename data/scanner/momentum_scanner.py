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
    """Scans historical data to identify momentum days and quality reset points."""
    
    # Momentum day thresholds
    MIN_DAILY_MOVE = 0.10  # 10% minimum intraday movement
    MAX_DAILY_MOVE = 0.30  # 30% maximum (beyond this might be too volatile)
    MIN_VOLUME_MULTIPLIER = 2.0  # Minimum 2x average volume
    
    # Reset point parameters
    MIN_RESET_POINTS_PER_DAY = 10
    MAX_RESET_POINTS_PER_DAY = 20
    RESET_LOOKBACK_MINUTES = 30  # Look back 30 minutes for pattern context
    
    # Pattern types
    PATTERN_TYPES = [
        'breakout',      # Price breaking through resistance
        'flush',         # Sharp downward move
        'bounce',        # Recovery from low
        'consolidation', # Sideways movement after big move
        'gap_fill',      # Filling morning gap
        'momentum_cont', # Continuation of existing trend
    ]
    
    def __init__(self, data_dir: str, output_dir: str, logger: Optional[logging.Logger] = None):
        """Initialize the momentum scanner.
        
        Args:
            data_dir: Directory containing Databento files
            output_dir: Directory to save index files
            logger: Optional logger instance
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or logging.getLogger(__name__)
        
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
        """Analyze a single day for momentum characteristics.
        
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
        daily_range = (high_price - low_price) / open_price
        max_move_up = (high_price - open_price) / open_price
        max_move_down = (open_price - low_price) / open_price
        max_intraday_move = max(max_move_up, max_move_down)
        
        # Check momentum criteria
        is_momentum_day = (
            self.MIN_DAILY_MOVE <= max_intraday_move <= self.MAX_DAILY_MOVE and
            total_volume >= avg_volume * self.MIN_VOLUME_MULTIPLIER
        )
        
        if not is_momentum_day:
            return None, []
            
        # Find halts
        halt_count = self._count_halts(symbol, date)
        
        # Calculate quality score
        quality_score = self._calculate_day_quality(
            max_intraday_move, total_volume / avg_volume, halt_count
        )
        
        # Create day record
        day_record = {
            'symbol': symbol,
            'date': pd.Timestamp(date),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': total_volume,
            'max_intraday_move': max_intraday_move,
            'volume_multiplier': total_volume / avg_volume,
            'halt_count': halt_count,
            'quality_score': quality_score,
            'file_paths': [str(f) for f in self._find_day_files(symbol, date)]
        }
        
        # Find reset points within the day
        reset_points = self._find_reset_points(symbol, date, day_data, quality_score)
        
        return day_record, reset_points
    
    def _find_reset_points(self, symbol: str, date: pd.Timestamp,
                          day_data: pd.DataFrame, day_quality: float) -> List[Dict]:
        """Find quality reset points within a momentum day."""
        reset_points = []
        
        # Resample to 1-minute bars for pattern detection
        minute_bars = day_data.resample('1min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        if len(minute_bars) < 60:  # Need at least 1 hour of data
            return reset_points
            
        # Calculate technical indicators
        minute_bars['sma_10'] = minute_bars['close'].rolling(10).mean()
        minute_bars['sma_30'] = minute_bars['close'].rolling(30).mean()
        minute_bars['volume_sma'] = minute_bars['volume'].rolling(10).mean()
        minute_bars['price_velocity'] = minute_bars['close'].pct_change(5)
        minute_bars['volume_ratio'] = minute_bars['volume'] / minute_bars['volume_sma']
        
        # Identify potential reset points
        potential_points = []
        
        for i in range(self.RESET_LOOKBACK_MINUTES, len(minute_bars)):
            timestamp = minute_bars.index[i]
            lookback = minute_bars.iloc[i-self.RESET_LOOKBACK_MINUTES:i+1]
            
            # Detect patterns
            pattern = self._detect_pattern(lookback)
            if pattern:
                quality = self._calculate_reset_quality(
                    lookback, pattern, timestamp.hour
                )
                
                potential_points.append({
                    'timestamp': timestamp,
                    'pattern': pattern,
                    'quality': quality,
                    'price': minute_bars['close'].iloc[i],
                    'volume': minute_bars['volume'].iloc[i],
                    'velocity': minute_bars['price_velocity'].iloc[i]
                })
        
        # Select best reset points (distributed throughout the day)
        selected_points = self._select_best_reset_points(
            potential_points, 
            min_points=self.MIN_RESET_POINTS_PER_DAY,
            max_points=self.MAX_RESET_POINTS_PER_DAY
        )
        
        # Create reset point records
        for point in selected_points:
            reset_points.append({
                'symbol': symbol,
                'date': date,
                'timestamp': point['timestamp'],
                'pattern_type': point['pattern'],
                'local_quality_score': point['quality'],
                'day_quality_score': day_quality,
                'combined_score': point['quality'] * day_quality,
                'price': point['price'],
                'volume': point['volume'],
                'price_velocity': point['velocity']
            })
            
        return reset_points
    
    def _detect_pattern(self, lookback: pd.DataFrame) -> Optional[str]:
        """Detect momentum patterns in lookback window."""
        if len(lookback) < 10:
            return None
            
        # Calculate metrics
        price_change = (lookback['close'].iloc[-1] - lookback['close'].iloc[0]) / lookback['close'].iloc[0]
        high_to_low = (lookback['high'].max() - lookback['low'].min()) / lookback['close'].iloc[0]
        recent_velocity = lookback['close'].pct_change().iloc[-5:].mean()
        volume_surge = lookback['volume'].iloc[-5:].mean() / lookback['volume'].iloc[:-5].mean()
        
        # Pattern detection logic
        if price_change > 0.02 and volume_surge > 1.5 and recent_velocity > 0:
            if lookback['close'].iloc[-1] > lookback['high'].iloc[:-5].max():
                return 'breakout'
            else:
                return 'momentum_cont'
                
        elif price_change < -0.02 and volume_surge > 1.5:
            return 'flush'
            
        elif abs(price_change) < 0.01 and high_to_low > 0.02:
            if lookback['low'].iloc[-5:].min() > lookback['low'].iloc[:-5].min():
                return 'bounce'
            else:
                return 'consolidation'
                
        elif self._is_gap_fill(lookback):
            return 'gap_fill'
            
        return None
    
    def _is_gap_fill(self, lookback: pd.DataFrame) -> bool:
        """Check if current action represents gap fill pattern."""
        # Simple gap fill detection - would need access to previous day's close
        # For now, return False
        return False
    
    def _calculate_reset_quality(self, lookback: pd.DataFrame, 
                                pattern: str, hour: int) -> float:
        """Calculate quality score for a reset point."""
        score = 0.5  # Base score
        
        # Pattern quality multipliers
        pattern_scores = {
            'breakout': 1.2,
            'flush': 1.1,
            'bounce': 1.0,
            'momentum_cont': 0.9,
            'consolidation': 0.8,
            'gap_fill': 1.0
        }
        score *= pattern_scores.get(pattern, 1.0)
        
        # Time of day multiplier
        if 9 <= hour <= 10:  # Market open
            score *= 1.2
        elif 15 <= hour <= 16:  # Power hour
            score *= 1.1
        elif 11 <= hour <= 14:  # Midday
            score *= 0.9
            
        # Volume quality
        volume_ratio = lookback['volume'].iloc[-5:].mean() / lookback['volume'].mean()
        score *= min(1.5, max(0.5, volume_ratio))
        
        # Price action clarity (low noise)
        returns = lookback['close'].pct_change().dropna()
        if len(returns) > 0:
            noise_ratio = returns.std() / abs(returns.mean()) if returns.mean() != 0 else 10
            clarity_multiplier = 1.0 / (1.0 + noise_ratio)
            score *= clarity_multiplier
            
        return min(1.0, max(0.0, score))
    
    def _select_best_reset_points(self, points: List[Dict], 
                                 min_points: int, max_points: int) -> List[Dict]:
        """Select best reset points with good distribution throughout the day."""
        if len(points) <= min_points:
            return points
            
        # Sort by quality
        points.sort(key=lambda x: x['quality'], reverse=True)
        
        # Take top quality points
        selected = points[:max_points]
        
        # Ensure time distribution
        if len(selected) > min_points:
            # Group by hour and ensure coverage
            hourly_groups = {}
            for point in selected:
                hour = point['timestamp'].hour
                if hour not in hourly_groups:
                    hourly_groups[hour] = []
                hourly_groups[hour].append(point)
                
            # Rebalance if too concentrated
            final_selected = []
            points_per_hour = max_points // len(hourly_groups) if hourly_groups else 1
            
            for hour, hour_points in sorted(hourly_groups.items()):
                # Take best from each hour
                hour_points.sort(key=lambda x: x['quality'], reverse=True)
                final_selected.extend(hour_points[:points_per_hour])
                
            # Fill remaining slots with best overall
            remaining = max_points - len(final_selected)
            if remaining > 0:
                used_timestamps = {p['timestamp'] for p in final_selected}
                unused = [p for p in selected if p['timestamp'] not in used_timestamps]
                final_selected.extend(unused[:remaining])
                
            return final_selected[:max_points]
        
        return selected
    
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
    
    def _calculate_day_quality(self, max_move: float, volume_mult: float, 
                              halt_count: int) -> float:
        """Calculate overall quality score for a momentum day."""
        # Base score from price movement
        if max_move >= 0.20:  # 20%+ move
            base_score = 0.9
        elif max_move >= 0.15:  # 15-20% move
            base_score = 0.7
        else:  # 10-15% move
            base_score = 0.5
            
        # Volume multiplier
        if volume_mult >= 5.0:
            volume_score = 1.2
        elif volume_mult >= 3.0:
            volume_score = 1.1
        else:
            volume_score = 1.0
            
        # Halt penalty (some halts are good for volatility training)
        if halt_count == 0:
            halt_score = 1.0
        elif halt_count <= 2:
            halt_score = 0.9
        else:
            halt_score = 0.7
            
        # Combine scores
        final_score = base_score * volume_score * halt_score
        return min(1.0, max(0.0, final_score))
    
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
    
    def query_momentum_days(self, symbol: str, min_quality: float = 0.5,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Query momentum days for a symbol.
        
        Args:
            symbol: Symbol to query
            min_quality: Minimum quality score
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame of matching momentum days
        """
        day_df, _ = self.load_index()
        
        if day_df.empty:
            return pd.DataFrame()
            
        # Filter by symbol
        mask = day_df['symbol'] == symbol.upper()
        
        # Filter by quality
        mask &= day_df['quality_score'] >= min_quality
        
        # Filter by date range
        if start_date:
            mask &= day_df['date'] >= pd.Timestamp(start_date)
        if end_date:
            mask &= day_df['date'] <= pd.Timestamp(end_date)
            
        return day_df[mask].sort_values('quality_score', ascending=False)
    
    def query_reset_points(self, symbol: str, date: Optional[pd.Timestamp] = None,
                          pattern_type: Optional[str] = None,
                          min_quality: float = 0.5) -> pd.DataFrame:
        """Query reset points for a symbol.
        
        Args:
            symbol: Symbol to query
            date: Specific date (None for all)
            pattern_type: Filter by pattern type
            min_quality: Minimum combined quality score
            
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
            
        # Filter by pattern
        if pattern_type:
            mask &= reset_df['pattern_type'] == pattern_type
            
        # Filter by quality
        mask &= reset_df['combined_score'] >= min_quality
        
        return reset_df[mask].sort_values('combined_score', ascending=False)