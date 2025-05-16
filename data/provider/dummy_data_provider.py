# data/provider/dummy/dummy_provider.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import random

from data.provider.data_provider import HistoricalDataProvider
from data.utils.helpers import ensure_timezone_aware


class DummyDataProvider(HistoricalDataProvider):
    """
    A dummy data provider that generates synthetic market data for testing.
    Implements the HistoricalDataProvider interface.
    """

    def __init__(self,
                 config: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the dummy data provider.

        Args:
            config: Configuration dictionary with optional parameters
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # Configure data generation parameters
        self.price_range = self.config.get('price_range', (2.0, 15.0))
        self.volatility = self.config.get('volatility', 0.02)
        self.trend_strength = self.config.get('trend_strength', 0.6)
        self.vol_baseline = self.config.get('vol_baseline', 5000)
        self.vol_variation = self.config.get('vol_variation', 0.5)
        self.random_seed = self.config.get('random_seed', 42)

        # Random generator for reproducibility
        self.rng = np.random.RandomState(self.random_seed)

        # Store available symbols
        self.available_symbols = self.config.get('symbols',
                                                 ['MLGO', 'AAPL', 'MSFT', 'NVDA', 'TSLA'])

        # Configuration for different trading session hours
        self.market_hours = {
            'pre_market_start': datetime.strptime('04:00:00', '%H:%M:%S').time(),
            'market_open': datetime.strptime('09:30:00', '%H:%M:%S').time(),
            'market_close': datetime.strptime('16:00:00', '%H:%M:%S').time(),
            'post_market_end': datetime.strptime('20:00:00', '%H:%M:%S').time(),
        }

        # Cached data for faster retrieval
        self._cached_data = {}

        self.logger.info(f"DummyDataProvider initialized with {len(self.available_symbols)} symbols")

    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if datetime is within market hours."""
        t = dt.time()
        return self.market_hours['market_open'] <= t <= self.market_hours['market_close']

    def _is_extended_hours(self, dt: datetime) -> bool:
        """Check if datetime is within extended trading hours."""
        t = dt.time()
        return ((self.market_hours['pre_market_start'] <= t < self.market_hours['market_open']) or
                (self.market_hours['market_close'] < t <= self.market_hours['post_market_end']))

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        if symbol not in self.available_symbols:
            raise ValueError(f"Symbol {symbol} not available in dummy provider")

        # Generate some basic symbol info
        return {
            'symbol': symbol,
            'description': f"{symbol} Inc. - Dummy Data",
            'exchange': 'NASDAQ',
            'asset_type': 'STOCK',
            'sector': 'TECHNOLOGY',
            'float': self.rng.randint(500000, 20000000),
            'avg_volume': self.vol_baseline
        }

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        return self.available_symbols.copy()

    def _generate_price_series(self, symbol: str, start_time: datetime,
                               end_time: datetime, interval_seconds: int = 1) -> pd.DataFrame:
        """
        Generate a synthetic price series with realistic properties.

        Args:
            symbol: Symbol to generate data for
            start_time: Start time
            end_time: End time
            interval_seconds: Interval in seconds between data points

        Returns:
            DataFrame with OHLCV price bars
        """
        # Generate timestamps within trading hours
        timestamps = []
        current = start_time

        while current <= end_time:
            # Skip non-trading hours
            if self._is_market_hours(current) or self._is_extended_hours(current):
                timestamps.append(current)

            current += timedelta(seconds=interval_seconds)

        if not timestamps:
            return pd.DataFrame()

        # Set initial price based on symbol
        symbol_seed = sum(ord(c) for c in symbol) % 100
        price_min, price_max = self.price_range
        initial_price = price_min + (symbol_seed / 100) * (price_max - price_min)

        # Generate price movements
        n = len(timestamps)

        # Create base random component
        daily_volatility = self.volatility * np.sqrt(252)
        noise = self.rng.normal(0, daily_volatility / np.sqrt(252 * 6.5 * 60 * 60 / interval_seconds), n)

        # Add some trend and mean reversion components
        t = np.linspace(0, 1, n)

        # Create a few trend changes
        n_trends = max(2, n // 2000)
        trend_points = self.rng.choice(range(1, n - 1), n_trends, replace=False)
        trend_points.sort()
        trend_points = np.concatenate(([0], trend_points, [n - 1]))

        # Generate trend directions
        trend_dirs = self.rng.choice([-1, 1], n_trends + 1)

        # Apply trends
        trends = np.zeros(n)
        for i in range(len(trend_points) - 1):
            start, end = trend_points[i], trend_points[i + 1]
            trends[start:end + 1] = np.linspace(0, trend_dirs[i] * self.trend_strength, end - start + 1)

        # Combine components
        changes = noise + trends * daily_volatility

        # Calculate log prices
        log_prices = np.cumsum(changes) + np.log(initial_price)
        prices = np.exp(log_prices)

        # Generate OHLCV data
        df = pd.DataFrame(index=timestamps)

        # For each bar, calculate open, high, low, close
        df['open'] = prices

        # Add random high/low within each bar
        high_multipliers = 1 + np.abs(self.rng.normal(0, self.volatility / 3, n))
        low_multipliers = 1 - np.abs(self.rng.normal(0, self.volatility / 3, n))

        df['high'] = prices * high_multipliers
        df['low'] = prices * low_multipliers

        # Ensure high >= open/close >= low
        df['high'] = np.maximum(df['high'], df['open'])
        df['low'] = np.minimum(df['low'], df['open'])

        # Shift to get next bar's open as current close
        df['close'] = df['open'].shift(-1).fillna(df['open'] * (1 + self.rng.normal(0, self.volatility / 5)))

        # Ensure high >= close and low <= close
        df['high'] = np.maximum(df['high'], df['close'])
        df['low'] = np.minimum(df['low'], df['close'])

        # Generate volume
        base_volume = self.vol_baseline

        # Volume tends to be higher at open and close
        time_factor = np.ones(n)
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60
            # Higher volume at open and close
            if 9.5 <= hour < 10.5 or 15 <= hour < 16:
                time_factor[i] = 2.0
            # Lower volume in middle of day
            elif 12 <= hour < 14:
                time_factor[i] = 0.7
            # Lower volume in extended hours
            elif hour < 9.5 or hour >= 16:
                time_factor[i] = 0.3

        # Random volume variations with some clustering
        vol_noise = np.abs(self.rng.normal(0, self.vol_variation, n))

        # Add occasional volume spikes
        spike_mask = self.rng.random(n) < 0.02  # 2% chance of volume spike
        vol_noise[spike_mask] *= self.rng.uniform(3, 10, np.sum(spike_mask))

        # Combine volume factors
        df['volume'] = base_volume * time_factor * (1 + vol_noise)
        df['volume'] = df['volume'].astype(int)

        # Add VWAP
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

        return df

    def _generate_trades_from_bars(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic trades from OHLCV bars."""
        if bars_df.empty:
            return pd.DataFrame()

        all_trades = []

        for idx, bar in bars_df.iterrows():
            # Generate random number of trades for this bar
            n_trades = max(1, int(np.round(0.1 * np.sqrt(bar['volume']))))

            # Limit to reasonable number
            n_trades = min(n_trades, 100)

            # Generate trades within this bar's time
            for _ in range(n_trades):
                # Get random price within bar range
                price = self.rng.uniform(bar['low'], bar['high'])

                # Calculate trade size
                size = max(1, int(bar['volume'] / n_trades * self.rng.uniform(0.5, 1.5)))

                # Assign random side
                if price > (bar['high'] + bar['low']) / 2:
                    side = 'A'  # Ask/Sell
                else:
                    side = 'B'  # Bid/Buy

                trade = {
                    'price': price,
                    'size': size,
                    'side': side,
                    'exchange': 'NASDAQ',
                    'conditions': ['@', 'T'],
                    'trade_id': f"T{idx.strftime('%H%M%S')}{self.rng.randint(10000, 99999)}",
                    'timestamp': idx + timedelta(milliseconds=self.rng.randint(0, 999))
                }

                all_trades.append(trade)

        if not all_trades:
            return pd.DataFrame()

        # Convert to DataFrame
        trades_df = pd.DataFrame(all_trades)

        # Use timestamp as index
        trades_df.set_index('timestamp', inplace=True)
        trades_df.sort_index(inplace=True)

        return trades_df

    def _generate_quotes_from_bars(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic quotes from OHLCV bars."""
        if bars_df.empty:
            return pd.DataFrame()

        all_quotes = []

        for idx, bar in bars_df.iterrows():
            # Generate multiple quotes per bar
            n_quotes = max(1, int(np.round(0.5 * np.sqrt(bar['volume']))))

            # Limit to reasonable number
            n_quotes = min(n_quotes, 200)

            # Generate timestamp offsets
            offsets = np.sort(self.rng.randint(0, 1000, n_quotes))

            # Previous values for continuity
            prev_bid = bar['low'] * 0.999
            prev_ask = bar['high'] * 1.001

            # Generate quotes within this bar's time
            for i, offset_ms in enumerate(offsets):
                # Calculate progression within bar (0 to 1)
                progress = i / max(1, n_quotes - 1)

                # Calculate bid and ask with some mean reversion
                target_mid = bar['low'] + progress * (bar['high'] - bar['low'])

                # Add some randomness to spread
                spread_bps = 5 + 15 * self.rng.random()  # 5-20 bps spread
                half_spread = target_mid * (spread_bps / 20000)  # half spread in price

                # Calculate bid and ask
                bid_price = target_mid - half_spread
                ask_price = target_mid + half_spread

                # Mean reversion
                bid_price = 0.7 * bid_price + 0.3 * prev_bid
                ask_price = 0.7 * ask_price + 0.3 * prev_ask

                # Update previous values
                prev_bid, prev_ask = bid_price, ask_price

                # Calculate sizes
                vol_factor = (bar['volume'] / self.vol_baseline) ** 0.5
                base_size = max(100, int(vol_factor * self.rng.uniform(100, 1000)))

                bid_size = int(base_size * self.rng.uniform(0.8, 1.2))
                ask_size = int(base_size * self.rng.uniform(0.8, 1.2))

                # Generate order counts
                bid_count = max(1, int(bid_size / (50 * self.rng.uniform(0.8, 1.2))))
                ask_count = max(1, int(ask_size / (50 * self.rng.uniform(0.8, 1.2))))

                quote = {
                    'bid_price': bid_price,
                    'ask_price': ask_price,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'bid_count': bid_count,
                    'ask_count': ask_count,
                    'exchange': 'NASDAQ',
                    'timestamp': idx + timedelta(milliseconds=offset_ms)
                }

                all_quotes.append(quote)

        if not all_quotes:
            return pd.DataFrame()

        # Convert to DataFrame
        quotes_df = pd.DataFrame(all_quotes)

        # Use timestamp as index
        quotes_df.set_index('timestamp', inplace=True)
        quotes_df.sort_index(inplace=True)

        return quotes_df

    def _get_status_data(self, symbol: str, start_time: datetime,
                         end_time: datetime) -> pd.DataFrame:
        """Generate synthetic status data."""
        # Create basic status history with market open/close events
        status_records = []

        current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)

        while current <= end_day:
            # Pre-market start
            pre_start = current.replace(
                hour=self.market_hours['pre_market_start'].hour,
                minute=self.market_hours['pre_market_start'].minute,
                second=0
            )
            if start_time <= pre_start <= end_time:
                status_records.append({
                    'timestamp': pre_start,
                    'status': 'PRE_OPEN',
                    'reason': 'SCHEDULED',
                    'is_trading': True,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

            # Market open
            market_open = current.replace(
                hour=self.market_hours['market_open'].hour,
                minute=self.market_hours['market_open'].minute,
                second=0
            )
            if start_time <= market_open <= end_time:
                status_records.append({
                    'timestamp': market_open,
                    'status': 'TRADING',
                    'reason': 'SCHEDULED',
                    'is_trading': True,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

            # Market close
            market_close = current.replace(
                hour=self.market_hours['market_close'].hour,
                minute=self.market_hours['market_close'].minute,
                second=0
            )
            if start_time <= market_close <= end_time:
                status_records.append({
                    'timestamp': market_close,
                    'status': 'POST_CLOSE',
                    'reason': 'SCHEDULED',
                    'is_trading': True,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

            # Post market end
            post_end = current.replace(
                hour=self.market_hours['post_market_end'].hour,
                minute=self.market_hours['post_market_end'].minute,
                second=0
            )
            if start_time <= post_end <= end_time:
                status_records.append({
                    'timestamp': post_end,
                    'status': 'CLOSED',
                    'reason': 'SCHEDULED',
                    'is_trading': False,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

            # Add occasional halt
            if self.rng.random() < 0.02:  # 2% chance of a halt per day
                halt_time = current + timedelta(hours=self.rng.randint(9, 16))
                if self._is_market_hours(halt_time) and start_time <= halt_time <= end_time:
                    halt_duration = timedelta(minutes=self.rng.randint(5, 30))

                    # Halt start
                    status_records.append({
                        'timestamp': halt_time,
                        'status': 'HALTED',
                        'reason': 'SURVEILLANCE_INTERVENTION',
                        'is_trading': False,
                        'is_halted': True,
                        'is_short_sell_restricted': False
                    })

                    # Halt end
                    status_records.append({
                        'timestamp': halt_time + halt_duration,
                        'status': 'TRADING',
                        'reason': 'SCHEDULED',
                        'is_trading': True,
                        'is_halted': False,
                        'is_short_sell_restricted': False
                    })

            # Move to next day
            current += timedelta(days=1)

        if not status_records:
            return pd.DataFrame()

        # Convert to DataFrame
        status_df = pd.DataFrame(status_records)

        # Use timestamp as index
        status_df.set_index('timestamp', inplace=True)
        status_df.sort_index(inplace=True)

        return status_df

    def _get_daily_bars(self, symbol: str, start_time: datetime,
                        end_time: datetime) -> pd.DataFrame:
        """Generate daily bars for a longer historical period."""
        # Create a unique key for caching
        cache_key = f"{symbol}_daily_{start_time.isoformat()}_{end_time.isoformat()}"

        # Check if already cached
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        # Generate dates (business days only)
        dates = []
        current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_day = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)

        while current <= end_day:
            # Skip weekends (0 = Monday, 6 = Sunday)
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)

        if not dates:
            return pd.DataFrame()

        # Set initial price based on symbol
        symbol_seed = sum(ord(c) for c in symbol) % 100
        price_min, price_max = self.price_range
        initial_price = price_min + (symbol_seed / 100) * (price_max - price_min)

        # Generate daily price changes
        n = len(dates)
        daily_volatility = self.volatility

        # Create random component
        noise = self.rng.normal(0, daily_volatility, n)

        # Create trend component
        trend = np.zeros(n)

        # Add a few trend changes
        n_trends = max(2, n // 30)  # Trend changes every 30 days on average
        trend_points = self.rng.choice(range(1, n - 1), n_trends, replace=False)
        trend_points.sort()
        trend_points = np.concatenate(([0], trend_points, [n - 1]))

        # Generate trend directions
        trend_dirs = self.rng.choice([-1, 1], n_trends + 1)

        # Apply trends
        for i in range(len(trend_points) - 1):
            start, end = trend_points[i], trend_points[i + 1]
            trend[start:end + 1] = np.linspace(0, trend_dirs[i] * self.trend_strength, end - start + 1)

        # Combine components
        changes = noise + trend * daily_volatility

        # Calculate log prices
        log_prices = np.cumsum(changes) + np.log(initial_price)
        prices = np.exp(log_prices)

        # Generate OHLCV data
        daily_data = []

        for i, date in enumerate(dates):
            # Generate open, high, low, close within daily range
            close = prices[i]

            # Previous close
            prev_close = prices[i - 1] if i > 0 else close * (1 - self.rng.normal(0, daily_volatility))

            # Generate open with gap
            gap = self.rng.normal(0, daily_volatility / 2)
            open_price = prev_close * (1 + gap)

            # Generate high and low
            daily_range = self.rng.uniform(1.5, 2.5) * daily_volatility
            high = max(open_price, close) * (1 + self.rng.uniform(0, daily_range))
            low = min(open_price, close) * (1 - self.rng.uniform(0, daily_range))

            # Generate volume
            vol_factor = self.rng.uniform(0.5, 1.5)
            if abs(close / prev_close - 1) > daily_volatility:
                vol_factor *= 2  # Higher volume on big moves

            volume = int(self.vol_baseline * 6.5 * 60 * 60 * vol_factor)  # Daily volume

            daily_data.append({
                'timestamp': date.replace(hour=16, minute=0, second=0),  # End of day timestamp
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'vwap': (high + low + close) / 3
            })

        # Convert to DataFrame
        daily_df = pd.DataFrame(daily_data)

        # Use timestamp as index
        daily_df.set_index('timestamp', inplace=True)

        # Cache the result
        self._cached_data[cache_key] = daily_df

        return daily_df

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get historical trades for a symbol in a time range.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized columns
        """
        # Validate symbol
        if symbol not in self.available_symbols:
            raise ValueError(f"Symbol {symbol} not available in dummy provider")

        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Create a unique key for caching
        cache_key = f"{symbol}_trades_{start_utc.isoformat()}_{end_utc.isoformat()}"

        # Check if already cached
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        # First get OHLCV bars - trades will be derived from these
        bars_df = self._generate_price_series(symbol, start_utc, end_utc)

        # Generate trades from bars
        trades_df = self._generate_trades_from_bars(bars_df)

        # Cache the result
        self._cached_data[cache_key] = trades_df

        return trades_df

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get historical quotes for a symbol in a time range.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized columns
        """
        # Validate symbol
        if symbol not in self.available_symbols:
            raise ValueError(f"Symbol {symbol} not available in dummy provider")

        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Create a unique key for caching
        cache_key = f"{symbol}_quotes_{start_utc.isoformat()}_{end_utc.isoformat()}"

        # Check if already cached
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        # First get OHLCV bars - quotes will be derived from these
        bars_df = self._generate_price_series(symbol, start_utc, end_utc)

        # Generate quotes from bars
        quotes_df = self._generate_quotes_from_bars(bars_df)

        # Cache the result
        self._cached_data[cache_key] = quotes_df

        return quotes_df

    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get OHLCV bars for a symbol, timeframe in a time range.

        Args:
            symbol: Symbol to get data for
            timeframe: Bar timeframe ("1s", "1m", "5m", "1d")
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized columns
        """
        # Validate symbol
        if symbol not in self.available_symbols:
            raise ValueError(f"Symbol {symbol} not available in dummy provider")

        # Validate timeframe
        if timeframe not in ["1s", "1m", "5m", "1d"]:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: 1s, 1m, 5m, 1d")

        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Create a unique key for caching
        cache_key = f"{symbol}_bars_{timeframe}_{start_utc.isoformat()}_{end_utc.isoformat()}"

        # Check if already cached
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        if timeframe == "1d":
            # Daily bars are generated differently
            bars_df = self._get_daily_bars(symbol, start_utc, end_utc)
        else:
            # Generate base 1-second bars
            interval_map = {
                "1s": 1,
                "1m": 60,
                "5m": 300
            }
            interval_seconds = interval_map[timeframe]

            if interval_seconds == 1:
                # For 1-second bars, generate directly
                bars_df = self._generate_price_series(symbol, start_utc, end_utc, interval_seconds)
            else:
                # For other timeframes, generate 1-second bars and resample
                base_df = self._generate_price_series(symbol, start_utc, end_utc, 1)

                if base_df.empty:
                    return pd.DataFrame()

                # Resample to the desired timeframe
                bars_df = base_df.resample(f'{interval_seconds}s').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'vwap': 'last'  # Just take the last VWAP for simplicity
                })

                # Filter out rows with no data
                bars_df = bars_df.dropna(subset=['open'])

        # Add timeframe column
        bars_df['timeframe'] = timeframe

        # Cache the result
        self._cached_data[cache_key] = bars_df

        return bars_df

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get status updates (halts, etc.) for a symbol in a time range.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized columns
        """
        # Validate symbol
        if symbol not in self.available_symbols:
            raise ValueError(f"Symbol {symbol} not available in dummy provider")

        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Create a unique key for caching
        cache_key = f"{symbol}_status_{start_utc.isoformat()}_{end_utc.isoformat()}"

        # Check if already cached
        if cache_key in self._cached_data:
            return self._cached_data[cache_key]

        # Generate status data
        status_df = self._get_status_data(symbol, start_utc, end_utc)

        # Cache the result
        self._cached_data[cache_key] = status_df

        return status_df