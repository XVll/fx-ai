"""Market Simulator - Pre-calculated features for efficient training

This simulator pre-computes all market state and features for each second of the trading day,
allowing for O(1) lookups during training. Features are calculated once and stored in a 
DataFrame for efficient access.

IMPORTANT: Handles warmup data by loading previous day's data for lookback windows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

from data.data_manager import DataManager
from data.feature_cache_manager import FeatureCacheManager
from feature.simple_feature_manager import SimpleFeatureManager
from feature.contexts import MarketContext
from config.schemas import ModelConfig, SimulationConfig
from dashboard.event_stream import event_stream, EventType

# Market hours configuration
MARKET_HOURS = {
    "PREMARKET_START": pd.Timestamp("04:00:00").time(),
    "REGULAR_START": pd.Timestamp("09:30:00").time(),
    "REGULAR_END": pd.Timestamp("16:00:00").time(),
    "POSTMARKET_END": pd.Timestamp("20:00:00").time(),
    "TIMEZONE": "America/New_York"
}


@dataclass
class MarketState:
    """Complete market state at a point in time"""
    timestamp: pd.Timestamp
    current_price: float
    best_bid: float
    best_ask: float
    bid_size: int
    ask_size: int
    mid_price: float
    spread: float
    market_session: str
    is_halted: bool
    intraday_high: float
    intraday_low: float
    session_volume: float
    session_trades: int
    session_vwap: float
    
    # Features - stored as numpy arrays
    hf_features: np.ndarray  # (seq_len, feat_dim)
    mf_features: np.ndarray  # (seq_len, feat_dim)
    lf_features: np.ndarray  # (seq_len, feat_dim)
    # static_features removed - moved to lf branch
    

class MarketSimulator:
    """Market simulator with pre-calculated features for entire trading day
    
    This version properly handles warmup data by loading previous day's data
    to ensure we have enough lookback for all feature windows.
    """
    
    def __init__(self, 
                 symbol: str,
                 data_manager: DataManager,
                 model_config: ModelConfig,
                 simulation_config: SimulationConfig,
                 feature_manager: Optional[SimpleFeatureManager] = None,
                 feature_cache_manager: Optional[FeatureCacheManager] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the market simulator
        
        Args:
            symbol: Trading symbol
            data_manager: Data manager for loading market data
            model_config: Model configuration with feature dimensions
            simulation_config: Simulation configuration
            feature_manager: Optional feature manager (will create if not provided)
            feature_cache_manager: Optional feature cache manager (will create if not provided)
            logger: Optional logger
        """
        self.symbol = symbol
        self.data_manager = data_manager
        self.model_config = model_config
        self.simulation_config = simulation_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize feature manager if not provided
        if feature_manager is None:
            self.feature_manager = SimpleFeatureManager(
                symbol=symbol,
                config=model_config,
                logger=self.logger
            )
        else:
            self.feature_manager = feature_manager
        
        # Initialize feature cache manager if not provided
        if feature_cache_manager is None:
            self.feature_cache_manager = FeatureCacheManager(logger=self.logger)
        else:
            self.feature_cache_manager = feature_cache_manager
            
        # Market timezone
        self.market_tz = ZoneInfo(MARKET_HOURS["TIMEZONE"])
        self.utc_tz = ZoneInfo("UTC")
        
        # Pre-computed market states DataFrame
        self.df_market_state = None
        
        # Current state tracking
        self.current_index = 0
        self.current_date = None

        
        # Cache for precomputed states
        self._precomputed_cache = {}  # {(symbol, date): df_market_state}
        
        # Previous day data cache
        self.prev_day_data = {}
        
        # Combined data with warmup
        self.combined_bars_1s = None
        self.combined_bars_1m = None
        self.combined_bars_5m = None
        self.combined_trades = None
        self.combined_quotes = None
    
    def __del__(self):
        """Cleanup method to ensure feature cache is saved"""
        try:
            if hasattr(self, 'feature_cache_manager'):
                self.feature_cache_manager.unload_session()
        except Exception:
            pass  # Ignore errors during cleanup
        
    def initialize_day(self, date: datetime) -> bool:
        """Initialize simulator for a specific trading day
        
        This pre-computes all market states and features for the entire day.
        Loads previous day data for warmup to handle early morning feature extraction.
        
        Args:
            date: Trading date to initialize
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing market simulator for {self.symbol} on {date}")
            
            # If we're switching to a different date, unload the previous cache session first
            new_cache_date = pd.Timestamp(date).date()
            if (self.current_date is not None and self.current_date != new_cache_date):
                self.logger.info(f"Switching from {self.current_date} to {new_cache_date}, unloading cache session")
                self.feature_cache_manager.unload_session()
            
            # Load feature cache session for this day
            if not self.feature_cache_manager.load_session(self.symbol, new_cache_date, self.feature_manager):
                self.logger.warning(f"Failed to load feature cache session for {self.symbol} on {new_cache_date}")
            
            # Check cache first
            cache_key = (self.symbol, pd.Timestamp(date).date())
            if cache_key in self._precomputed_cache:
                self.logger.info(f"Using cached precomputed states for {self.symbol} on {date}")
                self.df_market_state = self._precomputed_cache[cache_key]
                self.current_index = 0
                self.current_date = pd.Timestamp(date).date()
                return True
            
            # Load current day data
            day_data = self.data_manager.load_day(self.symbol, date)
            if not day_data:
                self.logger.error(f"No data available for {self.symbol} on {date}")
                return False
                
            # Get previous day summary data
            self.prev_day_data = self._get_previous_day_data(date)
            
            # Load previous day's full data for warmup
            prev_day_full_data = self._load_previous_day_full_data(date)
            
            # Combine current and previous day data
            self._combine_data_with_warmup(day_data, prev_day_full_data)
            
            # Build uniform timeline for current day only (4 AM - 8 PM ET)
            timeline = self._build_timeline(date)
            
            # Pre-compute all states and features
            self.df_market_state = self._precompute_states(timeline, day_data)
            
            # Cache the precomputed states
            if self.df_market_state is not None and not self.df_market_state.empty:
                self._precomputed_cache[cache_key] = self.df_market_state
                # Limit cache size to prevent memory issues
                if len(self._precomputed_cache) > 5:
                    # Remove oldest entry
                    oldest_key = next(iter(self._precomputed_cache))
                    del self._precomputed_cache[oldest_key]
            
            if self.df_market_state is None or self.df_market_state.empty:
                self.logger.error("Failed to pre-compute market states")
                return False
                
            # Reset to beginning
            self.current_index = 0
            self.current_date = pd.Timestamp(date).date()
            
            self.logger.info(f"Successfully initialized {len(self.df_market_state)} market states with warmup data")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize day: {e}")
            return False
            
    def _load_previous_day_full_data(self, date: datetime) -> Dict[str, pd.DataFrame]:
        """Load previous trading day's full data for warmup"""
        # First check if data_manager already has previous day data cached
        # This would have been loaded when we called load_day for the current day
        prev_data_from_cache = {
            'trades': self.data_manager.get_previous_day_data('trades'),
            'quotes': self.data_manager.get_previous_day_data('quotes'),
            'status': self.data_manager.get_previous_day_data('status'),
            'bars_1s': self.data_manager.get_previous_day_data('bars_1s'),
            'bars_1m': self.data_manager.get_previous_day_data('bars_1m'),
            'bars_5m': self.data_manager.get_previous_day_data('bars_5m'),
        }
        
        # Check if we actually have previous day data
        has_prev_data = any(df is not None and not df.empty for df in prev_data_from_cache.values())
        
        if has_prev_data:
            self.logger.info("Using cached previous day data for warmup")
            return prev_data_from_cache
            
        # If no cached data, try to find and load previous trading day
        prev_date = self._find_previous_trading_day(date)
        
        if prev_date is None:
            self.logger.warning("No previous trading day found - starting without warmup data")
            return {}
            
        self.logger.info(f"Loading previous day data from {prev_date} for warmup")
        
        # Try to load the previous day
        try:
            prev_day_data = self.data_manager.load_day(self.symbol, prev_date)
            return prev_day_data or {}
        except Exception as e:
            self.logger.warning(f"Failed to load previous day data: {e}")
            return {}
            
    def _combine_data_with_warmup(self, current_data: Dict[str, pd.DataFrame], 
                                  prev_data: Dict[str, pd.DataFrame]):
        """Combine current day data with previous day for warmup"""
        
        # Extract and combine trades
        current_trades = current_data.get('trades', pd.DataFrame())
        prev_trades = prev_data.get('trades', pd.DataFrame())
        
        if not prev_trades.empty and not current_trades.empty:
            self.combined_trades = pd.concat([prev_trades, current_trades]).sort_index()
        else:
            self.combined_trades = current_trades
            
        # Extract and combine quotes
        current_quotes = current_data.get('quotes', pd.DataFrame())
        prev_quotes = prev_data.get('quotes', pd.DataFrame())
        
        if not prev_quotes.empty and not current_quotes.empty:
            self.combined_quotes = pd.concat([prev_quotes, current_quotes]).sort_index()
        else:
            self.combined_quotes = current_quotes
            
        # Build 1s bars from combined trades
        if not self.combined_trades.empty:
            # Get time range for both days
            start_time = self.combined_trades.index[0].floor('1s')
            end_time = self.combined_trades.index[-1].ceil('1s')
            full_timeline = pd.date_range(start=start_time, end=end_time, freq='1s')
            
            self.combined_bars_1s = self._build_1s_bars_from_trades(self.combined_trades, full_timeline)
            
            # Aggregate to higher timeframes
            self.combined_bars_1m = self._aggregate_bars(self.combined_bars_1s, '1min')
            self.combined_bars_5m = self._aggregate_bars(self.combined_bars_1s, '5min')
        else:
            # No combined data - create empty DataFrames but with proper structure
            self.combined_bars_1s = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'trade_count'])
            self.combined_bars_1m = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'trade_count'])
            self.combined_bars_5m = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'trade_count'])
            
        if not self.combined_bars_1s.empty:
            self.logger.info(f"Combined data spans {len(self.combined_bars_1s)} seconds with warmup")
        else:
            self.logger.warning("No warmup data available - will use synthetic data for early hours")
        
    def _find_previous_trading_day(self, date: datetime) -> Optional[datetime]:
        """Find the previous trading day, checking against actual data availability"""
        current_date = pd.Timestamp(date).date()
        
        # Check up to 10 days back (to handle long weekends/holidays)
        for days_back in range(1, 11):
            check_date = current_date - timedelta(days=days_back)
            
            # Skip weekends first
            if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                continue
                
            # Try to check if data exists for this date
            # We can use momentum index if available
            if self.data_manager.momentum_scanner:
                momentum_days = self.data_manager.get_momentum_days(self.symbol)
                if not momentum_days.empty:
                    # Check if this date exists in momentum days
                    if any(momentum_days['date'].dt.date == check_date):
                        return check_date
                    continue
            
            # Otherwise, just return the first non-weekend day
            # The actual data loading will handle if it doesn't exist
            return check_date
            
        # If we couldn't find anything in 10 days, return None
        return None
        
    def _build_timeline(self, date: datetime) -> pd.DatetimeIndex:
        """Build uniform 1-second timeline for full trading day"""
        date_obj = pd.Timestamp(date).date()
        
        # Create ET timezone timestamps for 4 AM - 8 PM
        start_et = pd.Timestamp(date_obj).tz_localize(self.market_tz).replace(
            hour=MARKET_HOURS["PREMARKET_START"].hour,
            minute=MARKET_HOURS["PREMARKET_START"].minute,
            second=0
        )
        end_et = pd.Timestamp(date_obj).tz_localize(self.market_tz).replace(
            hour=MARKET_HOURS["POSTMARKET_END"].hour,
            minute=MARKET_HOURS["POSTMARKET_END"].minute,
            second=0
        )
        
        # Convert to UTC
        start_utc = start_et.tz_convert(self.utc_tz)
        end_utc = end_et.tz_convert(self.utc_tz)
        
        # Create 1-second timeline
        return pd.date_range(start=start_utc, end=end_utc, freq='1s')
        
    def _get_previous_day_data(self, date: datetime) -> Dict[str, float]:
        """Get previous trading day summary data"""
        prev_day_data = self.data_manager.get_previous_day_data('bars_1d')
        
        if prev_day_data is not None and not prev_day_data.empty:
            last_row = prev_day_data.iloc[-1]
            return {
                'open': float(last_row.get('open', 0)),
                'high': float(last_row.get('high', 0)),
                'low': float(last_row.get('low', 0)),
                'close': float(last_row.get('close', 0)),
                'volume': float(last_row.get('volume', 0)),
                'vwap': float(last_row.get('vwap', last_row.get('close', 0)))
            }
        
        # Return reasonable defaults when no previous day data exists
        return {'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 0, 'vwap': 100.0}
        
    @staticmethod
    def _extract_features_batch(batch_data: List[Tuple[int, pd.Timestamp, pd.Series]], 
                               shared_data: Dict) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Extract features for a batch of timestamps in parallel
        
        This is the worker function for parallel processing.
        """
        # Recreate feature manager in each process
        feature_manager = SimpleFeatureManager(
            symbol=shared_data['symbol'],
            config=shared_data['model_config'],
            logger=logging.getLogger(__name__)
        )
        
        results = []
        for idx, timestamp, row in batch_data:
            # Build windows for this timestamp
            hf_window = MarketSimulator._build_hf_window_for_batch(timestamp, shared_data)
            mf_window = MarketSimulator._build_mf_window_for_batch(timestamp, shared_data)  
            lf_window = MarketSimulator._build_lf_window_for_batch(timestamp, shared_data)
            
            # Create context
            context = MarketContext(
                timestamp=timestamp,
                current_price=row['current_price'],
                hf_window=hf_window,
                mf_1m_window=mf_window,
                lf_5m_window=lf_window,
                prev_day_close=shared_data['prev_day_data'].get('close', 0),
                prev_day_high=shared_data['prev_day_data'].get('high', 0),
                prev_day_low=shared_data['prev_day_data'].get('low', 0),
                session_high=row['intraday_high'],
                session_low=row['intraday_low'],
                session=row['market_session'],
                market_cap=1e9,
                session_volume=row['session_volume'],
                session_trades=row['session_trades'],
                session_vwap=row['session_vwap']
            )
            
            # Extract features
            features = feature_manager.extract_features(context)
            
            hf_feat = features.get('hf', np.zeros((shared_data['hf_seq_len'], shared_data['hf_feat_dim'])))
            mf_feat = features.get('mf', np.zeros((shared_data['mf_seq_len'], shared_data['mf_feat_dim'])))
            lf_feat = features.get('lf', np.zeros((shared_data['lf_seq_len'], shared_data['lf_feat_dim'])))
            # Static features have been moved to lf branch, no longer separate
            
            results.append((idx, hf_feat, mf_feat, lf_feat))
            
        return results
        
    @staticmethod
    def _build_hf_window_for_batch(current_ts: pd.Timestamp, shared_data: Dict) -> List[Dict[str, Any]]:
        """Build HF window using shared data for parallel processing"""
        window_size = shared_data['hf_seq_len']
        start_ts = current_ts - pd.Timedelta(seconds=window_size - 1)
        
        window = []
        combined_trades = shared_data.get('combined_trades', pd.DataFrame())
        combined_quotes = shared_data.get('combined_quotes', pd.DataFrame())
        combined_bars_1s = shared_data.get('combined_bars_1s', pd.DataFrame())
        
        for i in range(window_size):
            ts = start_ts + pd.Timedelta(seconds=i)
            
            # Get trades in this second
            second_trades = []
            if not combined_trades.empty:
                mask = (combined_trades.index >= ts) & (combined_trades.index < ts + pd.Timedelta(seconds=1))
                for _, trade in combined_trades[mask].iterrows():
                    second_trades.append({
                        'price': float(trade['price']),
                        'size': int(trade.get('size', 100)),
                        'conditions': []
                    })
                    
            # Get quotes in this second
            second_quotes = []
            if not combined_quotes.empty:
                mask = (combined_quotes.index >= ts) & (combined_quotes.index < ts + pd.Timedelta(seconds=1))
                for _, quote in combined_quotes[mask].iterrows():
                    second_quotes.append({
                        'bid_price': float(quote.get('bid_price', 0)),
                        'ask_price': float(quote.get('ask_price', 0)),
                        'bid_size': int(quote.get('bid_size', 0)),
                        'ask_size': int(quote.get('ask_size', 0))
                    })
                    
            # Get 1s bar
            bar_1s = None
            if not combined_bars_1s.empty and ts in combined_bars_1s.index:
                bar = combined_bars_1s.loc[ts]
                bar_1s = {
                    'timestamp': ts,
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': float(bar['volume']),
                    'is_synthetic': False
                }
                
            window.append({
                'timestamp': ts,
                'trades': second_trades,
                'quotes': second_quotes,
                '1s_bar': bar_1s
            })
            
        return window
        
    @staticmethod
    def _build_mf_window_for_batch(current_ts: pd.Timestamp, shared_data: Dict) -> List[Dict[str, Any]]:
        """Build MF window using shared data for parallel processing"""
        window_size = shared_data['mf_seq_len']
        current_minute = current_ts.floor('1min')
        combined_bars_1m = shared_data.get('combined_bars_1m', pd.DataFrame())
        prev_day_close = shared_data['prev_day_data'].get('close', 0)
        
        window = []
        
        for i in range(window_size):
            bar_ts = current_minute - pd.Timedelta(minutes=window_size - 1 - i)
            
            if not combined_bars_1m.empty and bar_ts in combined_bars_1m.index:
                bar = combined_bars_1m.loc[bar_ts]
                window.append({
                    'timestamp': bar_ts,
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': float(bar['volume']),
                    'is_synthetic': False
                })
            else:
                # Synthetic bar
                last_price = prev_day_close
                if window:
                    last_price = window[-1]['close']
                elif not combined_bars_1m.empty:
                    prev_bars = combined_bars_1m[combined_bars_1m.index < bar_ts]
                    if not prev_bars.empty:
                        last_price = prev_bars.iloc[-1]['close']
                        
                if last_price == 0:
                    last_price = 10.0
                        
                window.append({
                    'timestamp': bar_ts,
                    'open': last_price,
                    'high': last_price,
                    'low': last_price,
                    'close': last_price,
                    'volume': 0,
                    'is_synthetic': True
                })
                
        return window
        
    @staticmethod
    def _build_lf_window_for_batch(current_ts: pd.Timestamp, shared_data: Dict) -> List[Dict[str, Any]]:
        """Build LF window using shared data for parallel processing"""
        window_size = shared_data['lf_seq_len']
        current_5m = current_ts.floor('5min')
        combined_bars_5m = shared_data.get('combined_bars_5m', pd.DataFrame())
        prev_day_close = shared_data['prev_day_data'].get('close', 0)
        
        window = []
        
        for i in range(window_size):
            bar_ts = current_5m - pd.Timedelta(minutes=5 * (window_size - 1 - i))
            
            if not combined_bars_5m.empty and bar_ts in combined_bars_5m.index:
                bar = combined_bars_5m.loc[bar_ts]
                window.append({
                    'timestamp': bar_ts,
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': float(bar['volume']),
                    'is_synthetic': False
                })
            else:
                # Synthetic bar
                last_price = prev_day_close
                if window:
                    last_price = window[-1]['close']
                elif not combined_bars_5m.empty:
                    prev_bars = combined_bars_5m[combined_bars_5m.index < bar_ts]
                    if not prev_bars.empty:
                        last_price = prev_bars.iloc[-1]['close']
                        
                if last_price == 0:
                    last_price = 10.0
                        
                window.append({
                    'timestamp': bar_ts,
                    'open': last_price,
                    'high': last_price,
                    'low': last_price,
                    'close': last_price,
                    'volume': 0,
                    'is_synthetic': True
                })
                
        return window
        
    def _precompute_states(self, timeline: pd.DatetimeIndex, 
                          day_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Pre-compute all market states and features for the timeline
        
        Uses vectorized operations for much faster processing.
        """
        self.logger.info(f"Starting vectorized pre-computation of {len(timeline)} market states...")
        
        # Extract current day data
        trades_df = day_data.get('trades', pd.DataFrame())
        quotes_df = day_data.get('quotes', pd.DataFrame())
        status_df = day_data.get('status', pd.DataFrame())
        
        # Create base DataFrame with timeline
        df_states = pd.DataFrame(index=timeline)
        
        # Initialize default values with proper float types
        last_price = float(self.prev_day_data.get('close', 100.0))  # Default fallback
        df_states['current_price'] = float(last_price)
        df_states['best_bid'] = float(last_price * 0.999)
        df_states['best_ask'] = float(last_price * 1.001)
        df_states['bid_size'] = 100
        df_states['ask_size'] = 100
        df_states['is_halted'] = False
        
        # Vectorized quote processing
        if not quotes_df.empty:
            self.logger.info("Processing quotes vectorized...")
            # Forward fill quotes to timeline
            quotes_reindexed = quotes_df.reindex(timeline, method='ffill')
            mask = quotes_reindexed.notna().any(axis=1)
            df_states.loc[mask, 'best_bid'] = quotes_reindexed.loc[mask, 'bid_price'].fillna(df_states['best_bid'])
            df_states.loc[mask, 'best_ask'] = quotes_reindexed.loc[mask, 'ask_price'].fillna(df_states['best_ask'])
            df_states.loc[mask, 'bid_size'] = quotes_reindexed.loc[mask, 'bid_size'].fillna(df_states['bid_size'])
            df_states.loc[mask, 'ask_size'] = quotes_reindexed.loc[mask, 'ask_size'].fillna(df_states['ask_size'])
        
        # Vectorized 1s bar processing  
        if not self.combined_bars_1s.empty:
            self.logger.info("Processing 1s bars vectorized...")
            bars_in_timeline = self.combined_bars_1s.reindex(timeline, method='ffill')
            valid_bars = bars_in_timeline.dropna(subset=['close'])
            if not valid_bars.empty:
                df_states.loc[valid_bars.index, 'current_price'] = valid_bars['close']
                
        # Calculate derived fields vectorized
        df_states['mid_price'] = (df_states['best_bid'] + df_states['best_ask']) / 2
        df_states['spread'] = df_states['best_ask'] - df_states['best_bid']
        
        # Fix invalid spreads
        invalid_spread = (df_states['best_bid'] >= df_states['best_ask']) | (df_states['best_bid'] <= 0)
        df_states.loc[invalid_spread, 'spread'] = df_states.loc[invalid_spread, 'current_price'] * 0.001
        df_states.loc[invalid_spread, 'best_bid'] = df_states.loc[invalid_spread, 'current_price'] - df_states.loc[invalid_spread, 'spread'] / 2
        df_states.loc[invalid_spread, 'best_ask'] = df_states.loc[invalid_spread, 'current_price'] + df_states.loc[invalid_spread, 'spread'] / 2
        
        # Recalculate mid_price after spread corrections
        df_states['mid_price'] = (df_states['best_bid'] + df_states['best_ask']) / 2
        
        # Market session vectorized
        df_states['market_session'] = df_states.index.to_series().apply(self._get_market_session)
        
        # Session statistics (cumulative)
        if not self.combined_bars_1s.empty:
            bars_in_day = self.combined_bars_1s.reindex(timeline).fillna(0)
            df_states['session_volume'] = bars_in_day['volume'].cumsum()
            df_states['session_trades'] = bars_in_day.get('trade_count', 0).cumsum()
            df_states['session_value'] = (bars_in_day['close'] * bars_in_day['volume']).cumsum()
            df_states['session_vwap'] = df_states['session_value'] / df_states['session_volume'].replace(0, 1)
            df_states['intraday_high'] = bars_in_day['high'].expanding().max().fillna(last_price)
            df_states['intraday_low'] = bars_in_day['low'].expanding().min().fillna(last_price)
        else:
            df_states['session_volume'] = 0.0
            df_states['session_trades'] = 0  
            df_states['session_vwap'] = df_states['current_price'].astype(float)
            df_states['intraday_high'] = df_states['current_price'].astype(float)
            df_states['intraday_low'] = df_states['current_price'].astype(float)
        
        # FAST APPROACH: Skip feature extraction during initialization
        # Features will be calculated on-demand during training for better performance
        self.logger.info("Skipping feature pre-computation for faster initialization...")
        self.logger.info("Features will be calculated on-demand during training")
        
        # Pre-allocate empty feature arrays (will be populated on-demand)
        num_states = len(df_states)
        empty_hf = np.zeros((self.model_config.hf_seq_len, self.model_config.hf_feat_dim), dtype=np.float32)
        empty_mf = np.zeros((self.model_config.mf_seq_len, self.model_config.mf_feat_dim), dtype=np.float32)
        empty_lf = np.zeros((self.model_config.lf_seq_len, self.model_config.lf_feat_dim), dtype=np.float32)
        # Static features have been moved to lf branch
        
        # Fill with placeholder arrays - features computed on-demand
        df_states['hf_features'] = [empty_hf.copy() for _ in range(num_states)]
        df_states['mf_features'] = [empty_mf.copy() for _ in range(num_states)]
        df_states['lf_features'] = [empty_lf.copy() for _ in range(num_states)]
        # Static features have been moved to lf branch
        
        # Mark that features need computation
        df_states['features_computed'] = False
        
        # Calculate memory usage (for placeholder arrays)
        placeholder_memory_mb = (num_states * 100 * 4) / (1024 * 1024)  # Small placeholder overhead
        
        self.logger.info(f"Feature dimensions per timestamp (computed on-demand):")
        self.logger.info(f"  - HF: {self.model_config.hf_seq_len}x{self.model_config.hf_feat_dim} = {self.model_config.hf_seq_len * self.model_config.hf_feat_dim} values")
        self.logger.info(f"  - MF: {self.model_config.mf_seq_len}x{self.model_config.mf_feat_dim} = {self.model_config.mf_seq_len * self.model_config.mf_feat_dim} values")
        self.logger.info(f"  - LF: {self.model_config.lf_seq_len}x{self.model_config.lf_feat_dim} = {self.model_config.lf_seq_len * self.model_config.lf_feat_dim} values")
        # Static features have been moved to lf branch
        self.logger.info(f"Placeholder memory usage: {placeholder_memory_mb:.1f} MB for {num_states} states")
        self.logger.info(f"Completed FAST initialization of {len(df_states)} market states (features computed on-demand)")
        return df_states
        
    def _build_1s_bars_from_trades(self, trades_df: pd.DataFrame, 
                                   timeline: pd.DatetimeIndex) -> pd.DataFrame:
        """Build 1-second OHLCV bars from trades"""
        if trades_df.empty:
            # Return empty bars with timeline index
            return pd.DataFrame(index=timeline, columns=['open', 'high', 'low', 'close', 'volume', 'trade_count'])
            
        # Group trades by second
        trades_df['timestamp_1s'] = trades_df.index.floor('1s')
        
        # Aggregate to 1s bars
        bars = trades_df.groupby('timestamp_1s').agg({
            'price': ['first', 'max', 'min', 'last'],
            'size': 'sum'
        })
        
        # Flatten column names
        bars.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Add trade count
        trade_counts = trades_df.groupby('timestamp_1s').size()
        bars['trade_count'] = trade_counts
        
        # Reindex to full timeline with forward fill
        bars = bars.reindex(timeline)
        bars['volume'] = bars['volume'].fillna(0)
        bars['trade_count'] = bars['trade_count'].fillna(0)
        
        # Forward fill prices
        for col in ['open', 'high', 'low', 'close']:
            bars[col] = bars[col].ffill()
            
        return bars
        
    def _aggregate_bars(self, bars_1s: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Aggregate 1s bars to higher timeframe"""
        if bars_1s.empty:
            return pd.DataFrame()
            
        # Resample to target frequency
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'trade_count': 'sum'
        }
        
        bars = bars_1s.resample(freq).agg(agg_dict)
        
        # Forward fill prices
        for col in ['open', 'high', 'low', 'close']:
            bars[col] = bars[col].ffill()
            
        return bars
        
    def _get_market_session(self, timestamp: pd.Timestamp) -> str:
        """Determine market session for timestamp"""
        local_time = timestamp.tz_convert(self.market_tz).time()
        
        if local_time < MARKET_HOURS["PREMARKET_START"]:
            return "CLOSED"
        elif local_time < MARKET_HOURS["REGULAR_START"]:
            return "PREMARKET"
        elif local_time < MARKET_HOURS["REGULAR_END"]:
            return "REGULAR"
        elif local_time <= MARKET_HOURS["POSTMARKET_END"]:
            return "POSTMARKET"
        else:
            return "CLOSED"
            
    def _get_quote_at_time(self, quotes_df: pd.DataFrame, 
                          timestamp: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """Get quote state at or before timestamp"""
        if quotes_df.empty:
            return None
            
        # Get quotes up to timestamp
        valid_quotes = quotes_df[quotes_df.index <= timestamp]
        if valid_quotes.empty:
            return None
            
        # Get last quote
        last_quote = valid_quotes.iloc[-1]
        return {
            'bid_price': float(last_quote.get('bid_price', 0)),
            'ask_price': float(last_quote.get('ask_price', 0)),
            'bid_size': int(last_quote.get('bid_size', 0)),
            'ask_size': int(last_quote.get('ask_size', 0))
        }
        
    def _is_halted_at_time(self, status_df: pd.DataFrame, 
                          timestamp: pd.Timestamp) -> bool:
        """Check if trading is halted at timestamp"""
        if status_df.empty:
            return False
            
        # Get status up to timestamp
        valid_status = status_df[status_df.index <= timestamp]
        if valid_status.empty:
            return False
            
        # Check last status
        last_status = valid_status.iloc[-1]
        return bool(last_status.get('is_halted', False))
        
    def _build_hf_window_with_warmup(self, current_ts: pd.Timestamp) -> List[Dict[str, Any]]:
        """Build high-frequency data window using combined data"""
        window_size = self.model_config.hf_seq_len
        
        # Get window timestamps
        end_ts = current_ts
        start_ts = current_ts - pd.Timedelta(seconds=window_size - 1)
        
        window = []
        
        # Build window for each second
        for i in range(window_size):
            ts = start_ts + pd.Timedelta(seconds=i)
            
            # Get trades in this second from combined data
            second_trades = []
            if not self.combined_trades.empty:
                mask = (self.combined_trades.index >= ts) & (self.combined_trades.index < ts + pd.Timedelta(seconds=1))
                for _, trade in self.combined_trades[mask].iterrows():
                    second_trades.append({
                        'price': float(trade['price']),
                        'size': int(trade.get('size', 100)),
                        'conditions': []
                    })
                    
            # Get quotes in this second from combined data
            second_quotes = []
            if not self.combined_quotes.empty:
                mask = (self.combined_quotes.index >= ts) & (self.combined_quotes.index < ts + pd.Timedelta(seconds=1))
                for _, quote in self.combined_quotes[mask].iterrows():
                    second_quotes.append({
                        'bid_price': float(quote.get('bid_price', 0)),
                        'ask_price': float(quote.get('ask_price', 0)),
                        'bid_size': int(quote.get('bid_size', 0)),
                        'ask_size': int(quote.get('ask_size', 0))
                    })
                    
            # Get 1s bar from combined data
            bar_1s = None
            if not self.combined_bars_1s.empty and ts in self.combined_bars_1s.index:
                bar = self.combined_bars_1s.loc[ts]
                bar_1s = {
                    'timestamp': ts,
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': float(bar['volume']),
                    'is_synthetic': False
                }
                
            window.append({
                'timestamp': ts,
                'trades': second_trades,
                'quotes': second_quotes,
                '1s_bar': bar_1s
            })
            
        return window
        
    def _build_mf_window_with_warmup(self, current_ts: pd.Timestamp) -> List[Dict[str, Any]]:
        """Build medium-frequency (1m) data window using combined data"""
        window_size = self.model_config.mf_seq_len
        
        # Align to minute boundary
        current_minute = current_ts.floor('1min')
        
        window = []
        
        for i in range(window_size):
            bar_ts = current_minute - pd.Timedelta(minutes=window_size - 1 - i)
            
            if not self.combined_bars_1m.empty and bar_ts in self.combined_bars_1m.index:
                bar = self.combined_bars_1m.loc[bar_ts]
                window.append({
                    'timestamp': bar_ts,
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': float(bar['volume']),
                    'is_synthetic': False
                })
            else:
                # Synthetic bar - use last known price from combined data
                last_price = self.prev_day_data.get('close', 0)
                if window:
                    last_price = window[-1]['close']
                elif not self.combined_bars_1m.empty:
                    # Find last known price before this timestamp
                    prev_bars = self.combined_bars_1m[self.combined_bars_1m.index < bar_ts]
                    if not prev_bars.empty:
                        last_price = prev_bars.iloc[-1]['close']
                
                # If still no price (no previous day data), use a default
                if last_price == 0:
                    # Try to get from any available current day data
                    if not self.combined_bars_1s.empty:
                        # Get first available price from current day
                        first_prices = self.combined_bars_1s[self.combined_bars_1s['close'] > 0]
                        if not first_prices.empty:
                            last_price = first_prices.iloc[0]['close']
                    
                    # Final fallback - use a reasonable default
                    if last_price == 0:
                        last_price = 10.0  # Default price if absolutely no data
                        
                window.append({
                    'timestamp': bar_ts,
                    'open': last_price,
                    'high': last_price,
                    'low': last_price,
                    'close': last_price,
                    'volume': 0,
                    'is_synthetic': True
                })
                
        return window
        
    def _build_lf_window_with_warmup(self, current_ts: pd.Timestamp) -> List[Dict[str, Any]]:
        """Build low-frequency (5m) data window using combined data"""
        window_size = self.model_config.lf_seq_len
        
        # Align to 5-minute boundary
        current_5m = current_ts.floor('5min')
        
        window = []
        
        for i in range(window_size):
            bar_ts = current_5m - pd.Timedelta(minutes=5 * (window_size - 1 - i))
            
            if not self.combined_bars_5m.empty and bar_ts in self.combined_bars_5m.index:
                bar = self.combined_bars_5m.loc[bar_ts]
                window.append({
                    'timestamp': bar_ts,
                    'open': float(bar['open']),
                    'high': float(bar['high']),
                    'low': float(bar['low']),
                    'close': float(bar['close']),
                    'volume': float(bar['volume']),
                    'is_synthetic': False
                })
            else:
                # Synthetic bar - use last known price from combined data
                last_price = self.prev_day_data.get('close', 0)
                if window:
                    last_price = window[-1]['close']
                elif not self.combined_bars_5m.empty:
                    # Find last known price before this timestamp
                    prev_bars = self.combined_bars_5m[self.combined_bars_5m.index < bar_ts]
                    if not prev_bars.empty:
                        last_price = prev_bars.iloc[-1]['close']
                
                # If still no price (no previous day data), use a default
                if last_price == 0:
                    # Try to get from any available current day data
                    if not self.combined_bars_1s.empty:
                        # Get first available price from current day
                        first_prices = self.combined_bars_1s[self.combined_bars_1s['close'] > 0]
                        if not first_prices.empty:
                            last_price = first_prices.iloc[0]['close']
                    
                    # Final fallback - use a reasonable default
                    if last_price == 0:
                        last_price = 10.0  # Default price if absolutely no data
                        
                window.append({
                    'timestamp': bar_ts,
                    'open': last_price,
                    'high': last_price,
                    'low': last_price,
                    'close': last_price,
                    'volume': 0,
                    'is_synthetic': True
                })
                
        return window
        
    def get_market_state(self, timestamp: Optional[pd.Timestamp] = None) -> Optional[MarketState]:
        """Get market state at specific timestamp or current index
        
        Args:
            timestamp: Optional timestamp to query. If None, uses current index
            
        Returns:
            MarketState object or None if not found
        """
        if self.df_market_state is None or self.df_market_state.empty:
            return None
            
        if timestamp is not None:
            # Find exact or closest timestamp
            if timestamp in self.df_market_state.index:
                row = self.df_market_state.loc[timestamp]
            else:
                # Find closest previous timestamp
                valid_times = self.df_market_state.index[self.df_market_state.index <= timestamp]
                if valid_times.empty:
                    return None
                row = self.df_market_state.loc[valid_times[-1]]
        else:
            # Use current index
            if self.current_index >= len(self.df_market_state):
                return None
            row = self.df_market_state.iloc[self.current_index]
            
        # Convert row to MarketState
        return MarketState(
            timestamp=row.name,
            current_price=row['current_price'],
            best_bid=row['best_bid'],
            best_ask=row['best_ask'],
            bid_size=row['bid_size'],
            ask_size=row['ask_size'],
            mid_price=row['mid_price'],
            spread=row['spread'],
            market_session=row['market_session'],
            is_halted=row['is_halted'],
            intraday_high=row['intraday_high'],
            intraday_low=row['intraday_low'],
            session_volume=row['session_volume'],
            session_trades=row['session_trades'],
            session_vwap=row['session_vwap'],
            hf_features=row['hf_features'],
            mf_features=row['mf_features'],
            lf_features=row['lf_features'],
            # static_features removed - moved to lf branch
        )
        
    def get_current_features(self) -> Optional[Dict[str, np.ndarray]]:
        """Get features for current timestamp (computed on-demand with persistent caching)"""
        if self.df_market_state is None or self.current_index >= len(self.df_market_state):
            return None
            
        # Get current row
        row = self.df_market_state.iloc[self.current_index]
        timestamp = row.name
        
        # Check if features already computed in memory for this timestamp
        if not row.get('features_computed', False):
            # Check persistent cache first
            cached_features = self.feature_cache_manager.get_cached_features(timestamp)
            
            if cached_features is not None:
                # Load from persistent cache

                # Update the DataFrame with cached features
                self.df_market_state.at[timestamp, 'hf_features'] = cached_features.get('hf', np.zeros((self.model_config.hf_seq_len, self.model_config.hf_feat_dim)))
                self.df_market_state.at[timestamp, 'mf_features'] = cached_features.get('mf', np.zeros((self.model_config.mf_seq_len, self.model_config.mf_feat_dim)))
                self.df_market_state.at[timestamp, 'lf_features'] = cached_features.get('lf', np.zeros((self.model_config.lf_seq_len, self.model_config.lf_feat_dim)))
                self.df_market_state.at[timestamp, 'features_computed'] = True
                
                return cached_features
            
            # Compute features on-demand (cache miss)
            try:
                self.logger.debug(f"DEBUG: Computing features on-demand for {timestamp}")
                
                # Build windows for this timestamp
                self.logger.debug(f"DEBUG: Building HF window for {timestamp}")
                hf_window = self._build_hf_window_with_warmup(timestamp)
                self.logger.debug(f"DEBUG: HF window built, size: {len(hf_window)}")
                
                self.logger.debug(f"DEBUG: Building MF window for {timestamp}")
                mf_window = self._build_mf_window_with_warmup(timestamp)
                self.logger.debug(f"DEBUG: MF window built, size: {len(mf_window)}")
                
                self.logger.debug(f"DEBUG: Building LF window for {timestamp}")
                lf_window = self._build_lf_window_with_warmup(timestamp)
                self.logger.debug(f"DEBUG: LF window built, size: {len(lf_window)}")
                
                # Create context
                context = MarketContext(
                    timestamp=timestamp,
                    current_price=row['current_price'],
                    hf_window=hf_window,
                    mf_1m_window=mf_window,
                    lf_5m_window=lf_window,
                    prev_day_close=self.prev_day_data.get('close', 0),
                    prev_day_high=self.prev_day_data.get('high', 0),
                    prev_day_low=self.prev_day_data.get('low', 0),
                    session_high=row['intraday_high'],
                    session_low=row['intraday_low'],
                    session=row['market_session'],
                    market_cap=1e9,
                    session_volume=row['session_volume'],
                    session_trades=row['session_trades'],
                    session_vwap=row['session_vwap']
                )
                
                # Extract features
                self.logger.debug(f"DEBUG: About to extract features for timestamp {timestamp}")
                features = self.feature_manager.extract_features(context)
                self.logger.debug(f"DEBUG: Features extracted successfully")
                
                # Cache the newly computed features
                if features is not None:
                    self.feature_cache_manager.cache_features(timestamp, features)
                    self.logger.debug(f"DEBUG: Features cached for {timestamp}")
                
                # Update the DataFrame with computed features
                self.df_market_state.at[timestamp, 'hf_features'] = features.get('hf', np.zeros((self.model_config.hf_seq_len, self.model_config.hf_feat_dim)))
                self.df_market_state.at[timestamp, 'mf_features'] = features.get('mf', np.zeros((self.model_config.mf_seq_len, self.model_config.mf_feat_dim)))
                self.df_market_state.at[timestamp, 'lf_features'] = features.get('lf', np.zeros((self.model_config.lf_seq_len, self.model_config.lf_feat_dim)))
                # Static features have been moved to lf branch
                self.df_market_state.at[timestamp, 'features_computed'] = True
                
                # Return computed features
                return features
                
            except Exception as e:
                self.logger.error(f"On-demand feature computation failed for {timestamp}: {e}")
                # Return zeros on failure
                return {
                    'hf': np.zeros((self.model_config.hf_seq_len, self.model_config.hf_feat_dim)),
                    'mf': np.zeros((self.model_config.mf_seq_len, self.model_config.mf_feat_dim)),
                    'lf': np.zeros((self.model_config.lf_seq_len, self.model_config.lf_feat_dim)),
                    # Static features have been moved to lf branch
                }
        
        # Features already computed in memory - return cached values
        return {
            'hf': row['hf_features'],
            'mf': row['mf_features'],
            'lf': row['lf_features']
        }
        
    def get_current_market_data(self) -> Optional[Dict[str, Any]]:
        """Get market data (without features) for current timestamp"""
        state = self.get_market_state()
        if state is None:
            return None
        
        # Emit market update event for dashboard
        event_stream.emit_market_update(
            symbol=self.symbol,
            price=float(state.current_price),
            bid=float(state.best_bid),
            ask=float(state.best_ask),
            volume=int(state.session_volume),
            bid_size=int(state.bid_size),
            ask_size=int(state.ask_size),
            high=float(state.intraday_high),
            low=float(state.intraday_low),
            vwap=float(state.session_vwap),
            is_halted=state.is_halted,
            market_session=state.market_session,
            timestamp=state.timestamp  # Add timestamp for proper time display
        )
            
        return {
            'timestamp': state.timestamp,
            'current_price': state.current_price,
            'best_bid': state.best_bid,
            'best_ask': state.best_ask,
            'bid_size': state.bid_size,
            'ask_size': state.ask_size,
            'mid_price': state.mid_price,
            'spread': state.spread,
            'market_session': state.market_session,
            'is_halted': state.is_halted,
            'intraday_high': state.intraday_high,
            'intraday_low': state.intraday_low,
            'session_volume': state.session_volume,
            'session_trades': state.session_trades,
            'session_vwap': state.session_vwap
        }
        
    def step(self) -> bool:
        """Advance to next timestamp
        
        Returns:
            True if successful, False if at end of data
        """
        if self.df_market_state is None:
            return False
            
        self.current_index += 1
        return self.current_index < len(self.df_market_state)
        
    def reset(self, start_index: Optional[int] = None) -> bool:
        """Reset to beginning or specific index
        
        Args:
            start_index: Optional starting index. If None, resets to beginning
            
        Returns:
            True if successful
        """
        if self.df_market_state is None:
            return False
            
        if start_index is None:
            self.current_index = 0
        else:
            self.current_index = max(0, min(start_index, len(self.df_market_state) - 1))
            
        return True
        
    def set_time(self, timestamp: pd.Timestamp) -> bool:
        """Jump to specific timestamp
        
        Args:
            timestamp: Timestamp to jump to
            
        Returns:
            True if successful, False if timestamp not found
        """
        if self.df_market_state is None:
            return False
            
        # Find timestamp in index
        if timestamp in self.df_market_state.index:
            self.current_index = self.df_market_state.index.get_loc(timestamp)
            return True
        else:
            # Find closest previous timestamp
            valid_times = self.df_market_state.index[self.df_market_state.index <= timestamp]
            if valid_times.empty:
                return False
            self.current_index = self.df_market_state.index.get_loc(valid_times[-1])
            return True
            
    def get_time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get available time range
        
        Returns:
            Tuple of (start_time, end_time)
        """
        if self.df_market_state is None or self.df_market_state.empty:
            return (None, None)
            
        return (self.df_market_state.index[0], self.df_market_state.index[-1])
        
    def is_done(self) -> bool:
        """Check if at end of data"""
        if self.df_market_state is None:
            return True
            
        return self.current_index >= len(self.df_market_state) - 1
        
    def get_progress(self) -> float:
        """Get progress through the day as percentage"""
        if self.df_market_state is None or self.df_market_state.empty:
            return 0.0
            
        return self.current_index / max(1, len(self.df_market_state) - 1) * 100
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current day"""
        if self.df_market_state is None:
            return {}
            
        return {
            'date': self.current_date,
            'symbol': self.symbol,
            'total_seconds': len(self.df_market_state),
            'current_index': self.current_index,
            'progress_pct': self.get_progress(),
            'price_range': {
                'high': self.df_market_state['intraday_high'].max(),
                'low': self.df_market_state['intraday_low'].min()
            },
            'total_volume': self.df_market_state['session_volume'].iloc[-1] if not self.df_market_state.empty else 0,
            'total_trades': self.df_market_state['session_trades'].iloc[-1] if not self.df_market_state.empty else 0,
            'warmup_info': {
                'has_warmup': self.combined_bars_1s is not None and not self.combined_bars_1s.empty,
                'warmup_start': self.combined_bars_1s.index[0] if self.combined_bars_1s is not None and not self.combined_bars_1s.empty else None,
                'warmup_seconds': len(self.combined_bars_1s) - len(self.df_market_state) if self.combined_bars_1s is not None else 0
            }
        }
        
        
    def get_1m_candle_data(self, lookback_minutes: int = 390) -> List[Dict[str, Any]]:
        """Get 1-minute candle data for dashboard display
        
        Args:
            lookback_minutes: Number of minutes to look back (default 390 = full trading day)
            
        Returns:
            List of candle dictionaries suitable for dashboard
        """
        if self.combined_bars_1m is None or self.combined_bars_1m.empty:
            return []
            
        # Get current timestamp
        if self.df_market_state is None or self.current_index >= len(self.df_market_state):
            return []
            
        current_time = self.df_market_state.index[self.current_index]
        
        # Get candles up to current time
        candles = self.combined_bars_1m[self.combined_bars_1m.index <= current_time].tail(lookback_minutes)
        
        # Convert to dashboard format
        candle_list = []
        for timestamp, row in candles.iterrows():
            candle_list.append({
                'timestamp': timestamp.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
            
        return candle_list
    
    def close(self):
        """Clean up resources"""
        if self.df_market_state is not None:
            del self.df_market_state
            self.df_market_state = None
            
        # Clear cache
        self._precomputed_cache.clear()
            
        # Clean up combined data
        if self.combined_bars_1s is not None:
            del self.combined_bars_1s
            self.combined_bars_1s = None
        if self.combined_bars_1m is not None:
            del self.combined_bars_1m
            self.combined_bars_1m = None
        if self.combined_bars_5m is not None:
            del self.combined_bars_5m
            self.combined_bars_5m = None
        if self.combined_trades is not None:
            del self.combined_trades
            self.combined_trades = None
        if self.combined_quotes is not None:
            del self.combined_quotes
            self.combined_quotes = None
            
        self.current_index = 0
        self.current_date = None
        self.prev_day_data = {}