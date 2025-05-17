# Extremely lightweight dummy data provider for ultra-fast debugging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

from data.provider.data_provider import HistoricalDataProvider
from data.utils.helpers import ensure_timezone_aware


class DummyDataProvider(HistoricalDataProvider):
    """
    Ultra-lightweight dummy data provider that generates minimal data
    for the fastest possible debugging experience.

    This provider sacrifices data completeness and realism for speed,
    generating only the essential data needed to test the trading system.
    """

    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """Initialize the minimal dummy provider."""
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # Configure data generation (minimal defaults)
        self.debug_window_mins = self.config.get('debug_window_mins', 30)  # Just 30 mins of data by default.yaml
        self.price_range = self.config.get('price_range', (5.0, 10.0))  # Narrower range
        self.volatility = self.config.get('volatility', 0.02)
        self.random_seed = self.config.get('random_seed', 42)

        # Use a very small number of events for speed
        self.num_squeezes = self.config.get('num_squeezes', 2)  # Just 2 squeezes in the window
        self.squeeze_magnitude = self.config.get('squeeze_magnitude', 0.2)  # 20% price jump

        # Extreme data reduction settings
        self.data_sparsity = self.config.get('data_sparsity', 5)  # Only generate every Nth second
        self.quotes_per_bar = self.config.get('quotes_per_bar', 3)  # Minimal quotes
        self.trades_per_bar = self.config.get('trades_per_bar', 2)  # Minimal trades

        # Random generator
        self.rng = np.random.RandomState(self.random_seed)

        # Available symbols (just a few for testing)
        self.available_symbols = self.config.get('symbols', ['MLGO', 'AAPL'])

        # Cached data - everything in memory
        self._cached_data = {}

        # Pre-generate all needed data immediately
        for symbol in self.available_symbols:
            self._generate_minimal_dataset(symbol)

        self.logger.info(f"MinimalDummyProvider initialized with {len(self.available_symbols)} symbols, "
                         f"{self.debug_window_mins} minutes of sparse data")

    def _generate_minimal_dataset(self, symbol: str):
        """Generate a complete minimal dataset for a symbol."""
        # Choose a fixed reference date (doesn't matter what it is)
        base_date = datetime(2025, 1, 1, 9, 30, 0, tzinfo=datetime.now().astimezone().tzinfo)
        end_date = base_date + timedelta(minutes=self.debug_window_mins)

        # Generate 1s bars first - they're the foundation
        self._generate_1s_bars(symbol, base_date, end_date)

        # Generate derived data
        self._generate_1m_bars(symbol, base_date, end_date)
        self._generate_5m_bars(symbol, base_date, end_date)
        self._generate_trades(symbol, base_date, end_date)
        self._generate_quotes(symbol, base_date, end_date)
        self._generate_status(symbol, base_date, end_date)

        # Also generate daily data
        self._generate_daily_bars(symbol, base_date - timedelta(days=30), end_date)

    def _generate_1s_bars(self, symbol: str, start_time: datetime, end_time: datetime):
        """Generate minimal 1s bars."""
        # Only generate every Nth second for sparsity
        timestamps = []
        current = start_time
        while current <= end_time:
            if (current.second % self.data_sparsity) == 0:  # Only keep every Nth second
                timestamps.append(current)
            current += timedelta(seconds=1)

        if not timestamps:
            return

        # Set base price from symbol
        symbol_seed = sum(ord(c) for c in symbol) % 100
        price_min, price_max = self.price_range
        base_price = price_min + (symbol_seed / 100) * (price_max - price_min)

        # Super-simple price generation
        n_bars = len(timestamps)
        prices = np.zeros(n_bars)
        price = base_price

        # Plan squeeze events - randomly place them
        squeeze_indices = self.rng.choice(range(n_bars), size=self.num_squeezes, replace=False)

        # Generate prices
        for i in range(n_bars):
            # Normal small price movement
            price_change = self.rng.normal(0, self.volatility)

            # Add squeeze if scheduled
            if i in squeeze_indices:
                price_change += self.squeeze_magnitude

            # Update price
            price *= (1 + price_change)
            prices[i] = price

        # Generate minimal OHLC data
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,  # Simple high
            'low': prices * 0.995,  # Simple low
            'close': prices,
            'volume': self.rng.randint(100, 1000, size=n_bars),
            'timeframe': '1s'
        }, index=pd.DatetimeIndex(timestamps))

        # Cache
        cache_key = f"{symbol}_bars_1s_{start_time.isoformat()}_{end_time.isoformat()}"
        self._cached_data[cache_key] = df

    def _generate_1m_bars(self, symbol: str, start_time: datetime, end_time: datetime):
        """Generate 1m bars by aggregating/subsampling 1s bars."""
        # Get the 1s bars
        one_s_key = f"{symbol}_bars_1s_{start_time.isoformat()}_{end_time.isoformat()}"
        if one_s_key not in self._cached_data:
            return

        df_1s = self._cached_data[one_s_key]

        # Resample to 1-minute, taking only a few points
        minutes = []
        current = start_time.replace(second=0, microsecond=0)
        while current <= end_time:
            minutes.append(current)
            current += timedelta(minutes=1)

        # Create a minimal dataset with just enough data
        df_1m = pd.DataFrame({
            'open': self.rng.uniform(0.99, 1.01, size=len(minutes)) * 10,
            'high': self.rng.uniform(1.01, 1.03, size=len(minutes)) * 10,
            'low': self.rng.uniform(0.97, 0.99, size=len(minutes)) * 10,
            'close': self.rng.uniform(0.98, 1.02, size=len(minutes)) * 10,
            'volume': self.rng.randint(500, 5000, size=len(minutes)),
            'timeframe': '1m'
        }, index=pd.DatetimeIndex(minutes))

        # Cache
        cache_key = f"{symbol}_bars_1m_{start_time.isoformat()}_{end_time.isoformat()}"
        self._cached_data[cache_key] = df_1m

    def _generate_5m_bars(self, symbol: str, start_time: datetime, end_time: datetime):
        """Generate 5m bars by subsampling 1m bars."""
        # Create at 5-minute intervals
        five_mins = []
        current = start_time.replace(minute=start_time.minute - start_time.minute % 5,
                                     second=0, microsecond=0)
        while current <= end_time:
            five_mins.append(current)
            current += timedelta(minutes=5)

        # Create minimal 5m bars
        if not five_mins:
            return

        df_5m = pd.DataFrame({
            'open': self.rng.uniform(0.98, 1.02, size=len(five_mins)) * 10,
            'high': self.rng.uniform(1.01, 1.05, size=len(five_mins)) * 10,
            'low': self.rng.uniform(0.95, 0.99, size=len(five_mins)) * 10,
            'close': self.rng.uniform(0.97, 1.03, size=len(five_mins)) * 10,
            'volume': self.rng.randint(2000, 20000, size=len(five_mins)),
            'timeframe': '5m'
        }, index=pd.DatetimeIndex(five_mins))

        # Cache
        cache_key = f"{symbol}_bars_5m_{start_time.isoformat()}_{end_time.isoformat()}"
        self._cached_data[cache_key] = df_5m

    def _generate_daily_bars(self, symbol: str, start_time: datetime, end_time: datetime):
        """Generate minimal daily bars."""
        # Create daily intervals, skipping weekends
        days = []
        current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        while current <= end_time:
            if current.weekday() < 5:  # Mon-Fri
                days.append(current)
            current += timedelta(days=1)

        if not days:
            return

        # Create minimal daily data
        df_daily = pd.DataFrame({
            'open': self.rng.uniform(0.97, 1.03, size=len(days)) * 10,
            'high': self.rng.uniform(1.02, 1.08, size=len(days)) * 10,
            'low': self.rng.uniform(0.92, 0.98, size=len(days)) * 10,
            'close': self.rng.uniform(0.95, 1.05, size=len(days)) * 10,
            'volume': self.rng.randint(50000, 500000, size=len(days)),
            'timeframe': '1d'
        }, index=pd.DatetimeIndex(days))

        # Cache
        cache_key = f"{symbol}_bars_1d_{start_time.isoformat()}_{end_time.isoformat()}"
        self._cached_data[cache_key] = df_daily

    def _generate_trades(self, symbol: str, start_time: datetime, end_time: datetime):
        """Generate minimal trade data."""
        # Get 1s bars
        one_s_key = f"{symbol}_bars_1s_{start_time.isoformat()}_{end_time.isoformat()}"
        if one_s_key not in self._cached_data:
            return

        df_1s = self._cached_data[one_s_key]

        # Generate minimal trades
        all_trades = []

        for idx, bar in df_1s.iterrows():
            # Only create a few trades per bar
            for i in range(self.trades_per_bar):
                trade_time = idx + timedelta(milliseconds=self.rng.randint(0, 999))

                trade = {
                    'price': bar['close'] * (1 + self.rng.normal(0, 0.0005)),
                    'size': self.rng.randint(50, 200),
                    'side': 'B' if i % 2 == 0 else 'A',  # Alternate sides
                    'exchange': 'NASDAQ',
                    'conditions': [],
                    'trade_id': f"{i}"
                }

                all_trades.append((trade_time, trade))

        # Sort and create dataframe
        if not all_trades:
            return

        all_trades.sort(key=lambda x: x[0])
        times, trades = zip(*all_trades)

        trades_df = pd.DataFrame(trades, index=pd.DatetimeIndex(times))

        # Cache
        cache_key = f"{symbol}_trades_{start_time.isoformat()}_{end_time.isoformat()}"
        self._cached_data[cache_key] = trades_df

    def _generate_quotes(self, symbol: str, start_time: datetime, end_time: datetime):
        """Generate minimal quote data."""
        # Get 1s bars
        one_s_key = f"{symbol}_bars_1s_{start_time.isoformat()}_{end_time.isoformat()}"
        if one_s_key not in self._cached_data:
            return

        df_1s = self._cached_data[one_s_key]

        # Generate minimal quotes
        all_quotes = []

        for idx, bar in df_1s.iterrows():
            mid = bar['close']

            # Only create a few quotes per bar
            for i in range(self.quotes_per_bar):
                quote_time = idx + timedelta(milliseconds=self.rng.randint(0, 999))

                spread = mid * 0.001  # 0.1% spread

                quote = {
                    'bid_price': mid - spread / 2,
                    'ask_price': mid + spread / 2,
                    'bid_size': self.rng.randint(50, 500),
                    'ask_size': self.rng.randint(50, 500),
                    'bid_count': 5,
                    'ask_count': 5,
                    'exchange': 'NASDAQ'
                }

                all_quotes.append((quote_time, quote))

        # Sort and create dataframe
        if not all_quotes:
            return

        all_quotes.sort(key=lambda x: x[0])
        times, quotes = zip(*all_quotes)

        quotes_df = pd.DataFrame(quotes, index=pd.DatetimeIndex(times))

        # Cache
        cache_key = f"{symbol}_quotes_{start_time.isoformat()}_{end_time.isoformat()}"
        self._cached_data[cache_key] = quotes_df

    def _generate_status(self, symbol: str, start_time: datetime, end_time: datetime):
        """Generate minimal status data."""
        # Just make one record for the whole window
        status_records = [{
            'timestamp': start_time,
            'status': 'TRADING',
            'reason': 'SCHEDULED',
            'is_trading': True,
            'is_halted': False,
            'is_short_sell_restricted': False
        }]

        # Create dataframe
        status_df = pd.DataFrame(status_records)
        status_df.set_index('timestamp', inplace=True)

        # Cache
        cache_key = f"{symbol}_status_{start_time.isoformat()}_{end_time.isoformat()}"
        self._cached_data[cache_key] = status_df

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get simple metadata for a symbol."""
        if symbol not in self.available_symbols:
            raise ValueError(f"Symbol {symbol} not available in dummy provider")

        return {
            'symbol': symbol,
            'description': f"{symbol} Inc. - Minimal Test Data",
            'exchange': 'NASDAQ',
            'asset_type': 'STOCK',
            'sector': 'TECHNOLOGY',
            'float': 1000000,
            'avg_volume': 500000
        }

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        return self.available_symbols.copy()

    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get OHLCV bars - returns pre-generated minimal data."""
        # Validate
        if symbol not in self.available_symbols:
            raise ValueError(f"Symbol {symbol} not available in dummy provider")

        if timeframe not in ["1s", "1m", "5m", "1d"]:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        # Convert times to normalized form
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Create a key for the cache
        cache_key = f"{symbol}_bars_{timeframe}_{start_utc.isoformat()}_{end_utc.isoformat()}"

        # Try to get from cache using exact key
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        # Try to find a key that contains a superset of the data
        for k, df in self._cached_data.items():
            if f"{symbol}_bars_{timeframe}_" in k:
                # Found data for this symbol and timeframe
                # No need to filter - just return everything
                return df

        # No data found - return empty
        return pd.DataFrame()

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get trades - returns pre-generated minimal data."""
        # Convert times
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Cache key
        cache_key = f"{symbol}_trades_{start_utc.isoformat()}_{end_utc.isoformat()}"

        # Try to get from cache using exact key
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        # Try to find a key that contains a superset of the data
        for k, df in self._cached_data.items():
            if f"{symbol}_trades_" in k:
                # Found data for this symbol and type
                return df

        # No data found - return empty
        return pd.DataFrame()

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get quotes - returns pre-generated minimal data."""
        # Convert times
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Cache key
        cache_key = f"{symbol}_quotes_{start_utc.isoformat()}_{end_utc.isoformat()}"

        # Try to get from cache using exact key
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        # Try to find a key that contains a superset of the data
        for k, df in self._cached_data.items():
            if f"{symbol}_quotes_" in k:
                # Found data for this symbol and type
                return df

        # No data found - return empty
        return pd.DataFrame()

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get status - returns pre-generated minimal data."""
        # Convert times
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Cache key
        cache_key = f"{symbol}_status_{start_utc.isoformat()}_{end_utc.isoformat()}"

        # Try to get from cache using exact key
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        # Try to find a key that contains a superset of the data
        for k, df in self._cached_data.items():
            if f"{symbol}_status_" in k:
                # Found data for this symbol and type
                return df

        # Create a minimal status dataframe
        status_df = pd.DataFrame([{
            'status': 'TRADING',
            'reason': 'SCHEDULED',
            'is_trading': True,
            'is_halted': False,
            'is_short_sell_restricted': False
        }], index=[start_utc])

        return status_df