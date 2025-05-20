import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Union, Any
from zoneinfo import ZoneInfo

from data.provider.data_provider import HistoricalDataProvider
from data.utils.helpers import ensure_timezone_aware


class DummyDataProvider(HistoricalDataProvider):
    """
    Generates synthetic market data for testing purposes,
    following the same interface as Databento providers.

    This provider can generate realistic price action including:
    - Momentum squeezes
    - Pullbacks
    - Support/resistance tests at half/whole dollars
    - Realistic volume profiles
    - BBO data with spreads that tighten before momentum moves
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the dummy data provider with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Default settings if not provided in config
        self.debug_window_mins = self.config.get('debug_window_mins', 300)  # 5 hours by default
        self.data_sparsity = self.config.get('data_sparsity', 1)  # Every nth second to generate data for
        self.symbols = self.config.get('symbols', ['DUMMY'])
        self.num_squeezes = self.config.get('num_squeezes', 3)
        self.base_price = self.config.get('base_price', 5.00)  # Starting price around $5
        self.volatility = self.config.get('volatility', 0.05)  # 5% base volatility
        self.squeeze_magnitude = self.config.get('squeeze_magnitude', 0.30)  # 30% avg squeeze

        # Use fixed seed for reproducibility if provided
        self.seed = self.config.get('seed', None)
        self.np_random = np.random.RandomState(self.seed) if self.seed else np.random.RandomState()

        # Pre-generate data when initialized
        self._generated_data_cache = {}
        self._symbol_info_map = {}

        # Initialize symbol info
        for symbol in self.symbols:
            self._symbol_info_map[symbol] = {
                'symbol': symbol,
                'description': f'Synthetic data for {symbol}',
                'float_shares': self.np_random.randint(100000, 10000000),
                'sector': 'Technology',
                'avg_volume': self.np_random.randint(100000, 1000000),
            }

        self.logger.info(f"DummyDataProvider initialized with seed {self.seed} for symbols {self.symbols}")

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        if symbol in self._symbol_info_map:
            return self._symbol_info_map[symbol]

        return {
            'symbol': symbol,
            'description': f'Unknown symbol {symbol}',
            'is_synthetic': True
        }

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        return self.symbols

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get standardized trade data for a symbol.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized trade columns
        """
        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Generate trades data
        trades_df = self._generate_trades_data(symbol, start_utc, end_utc)

        # Return empty DataFrame if generation failed
        if trades_df is None or trades_df.empty:
            self.logger.warning(f"No trades data generated for {symbol} from {start_utc} to {end_utc}")
            return pd.DataFrame(columns=['price', 'size', 'side', 'exchange', 'conditions', 'trade_id'])

        return trades_df

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get standardized quote data for a symbol.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized quote columns
        """
        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Generate quotes data
        quotes_df = self._generate_quotes_data(symbol, start_utc, end_utc)

        # Return empty DataFrame if generation failed
        if quotes_df is None or quotes_df.empty:
            self.logger.warning(f"No quotes data generated for {symbol} from {start_utc} to {end_utc}")
            return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size',
                                         'bid_count', 'ask_count', 'side', 'exchange'])

        return quotes_df

    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get standardized OHLCV bar data for a symbol.

        Args:
            symbol: Symbol to get data for
            timeframe: Bar timeframe ("1s", "1m", "5m", "1d")
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized bar columns
        """
        # Validate timeframe
        if timeframe not in ["1s", "1m", "5m", "1d"]:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: 1s, 1m, 5m, 1d")

        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Generate bars data based on timeframe
        bars_df = self._generate_bars_data(symbol, timeframe, start_utc, end_utc)

        # Return empty DataFrame if generation failed
        if bars_df is None or bars_df.empty:
            self.logger.warning(f"No {timeframe} bars generated for {symbol} from {start_utc} to {end_utc}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'timeframe'])

        # Add timeframe column
        bars_df['timeframe'] = timeframe

        return bars_df

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """
        Get standardized status data for a symbol.

        Args:
            symbol: Symbol to get data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with standardized status columns
        """
        # Convert times to UTC
        start_utc = ensure_timezone_aware(start_time, is_end_time=False)
        end_utc = ensure_timezone_aware(end_time, is_end_time=True)

        # Generate status data
        status_df = self._generate_status_data(symbol, start_utc, end_utc)

        # Return empty DataFrame if generation failed
        if status_df is None or status_df.empty:
            self.logger.warning(f"No status data generated for {symbol} from {start_utc} to {end_utc}")
            return pd.DataFrame(columns=['status', 'reason', 'is_trading', 'is_halted',
                                         'is_short_sell_restricted'])

        return status_df

    def _generate_price_path(self, start_time: datetime, end_time: datetime) -> pd.Series:
        """
        Generate a synthetic price path with realistic momentum moves and consolidations.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Series with timestamps as index and prices as values
        """
        # Calculate how many seconds to generate
        time_range = (end_time - start_time).total_seconds()
        num_seconds = int(time_range // self.data_sparsity) + 1

        # Create timestamp index
        timestamps = [start_time + timedelta(seconds=i * self.data_sparsity) for i in range(num_seconds)]

        # Generate base random walk
        base_returns = self.np_random.normal(0.0, self.volatility / np.sqrt(24 * 60 * 60), num_seconds)

        # Add mean reversion to keep price within reasonable bounds
        mean_reversion = 0.02  # Strength of mean reversion
        base_returns = base_returns - mean_reversion * (np.log(self.base_price) - np.log(self.base_price))

        # Add squeezes at random intervals
        squeeze_points = self._generate_squeeze_points(num_seconds)

        # Apply squeezes
        for squeeze_idx, squeeze_magnitude, squeeze_duration in squeeze_points:
            # Half of squeeze_duration points before the peak have increasing returns
            for i in range(max(0, squeeze_idx - squeeze_duration // 2), squeeze_idx + 1):
                if 0 <= i < num_seconds:
                    # Gradually increase returns as we approach the peak
                    distance_to_peak = squeeze_idx - i
                    if distance_to_peak == 0:
                        # Max return at peak
                        base_returns[i] += squeeze_magnitude / 2
                    else:
                        # Increasing returns leading up to peak
                        position_factor = 1 - abs(distance_to_peak) / (squeeze_duration // 2)
                        base_returns[i] += squeeze_magnitude / 2 * position_factor

            # Half of squeeze_duration points after the peak have decreasing returns
            for i in range(squeeze_idx + 1, min(num_seconds, squeeze_idx + squeeze_duration // 2 + 1)):
                if 0 <= i < num_seconds:
                    # Gradually decrease returns as we move away from peak
                    distance_from_peak = i - squeeze_idx
                    position_factor = 1 - distance_from_peak / (squeeze_duration // 2)
                    base_returns[i] += -squeeze_magnitude / 4 * position_factor  # Smaller drop than rise

        # Ensure whole and half dollar resistance/support
        for i in range(1, num_seconds):
            # Get the previous log price
            prev_log_price = np.log(self.base_price) + np.sum(base_returns[:i])
            prev_price = np.exp(prev_log_price)

            # Check if we're approaching a whole or half dollar
            whole_dollar = int(np.round(prev_price))
            half_dollar = whole_dollar + 0.5

            # Determine if we're approaching from below or above
            next_log_price = prev_log_price + base_returns[i]
            next_price = np.exp(next_log_price)

            resistance_level = None

            # Check if we'd cross a whole dollar resistance
            if prev_price < whole_dollar < next_price:
                resistance_level = whole_dollar
            # Check if we'd cross a half dollar resistance
            elif prev_price < half_dollar < next_price:
                resistance_level = half_dollar

            # Apply resistance effect if needed
            if resistance_level is not None:
                # Probability of breaking through resistance
                break_prob = 0.4  # 40% chance of breaking through

                if self.np_random.random() > break_prob:
                    # Resistance holds, reduce the return to keep price below resistance
                    target_price = resistance_level - 0.01  # Just below resistance
                    target_log_price = np.log(target_price)
                    base_returns[i] = target_log_price - prev_log_price

        # Convert returns to prices
        log_prices = np.log(self.base_price) + np.cumsum(base_returns)
        prices = np.exp(log_prices)

        # Create Series
        price_series = pd.Series(prices, index=timestamps)

        return price_series

    def _generate_squeeze_points(self, num_seconds: int) -> List[tuple]:
        """
        Generate points where price squeezes (momentum spikes) occur.

        Args:
            num_seconds: Number of seconds in the price path

        Returns:
            List of tuples (index, magnitude, duration) for each squeeze
        """
        squeeze_points = []

        # Determine number of squeezes based on config
        n_squeezes = self.num_squeezes

        # Ensure at least one squeeze
        if n_squeezes < 1:
            n_squeezes = 1

        # Place squeezes throughout the time range, keeping them at least 10% of total time apart
        min_distance = max(1, int(num_seconds * 0.1))

        # First squeeze after the first 5-15% of the range
        first_squeeze_idx = self.np_random.randint(
            int(num_seconds * 0.05),
            int(num_seconds * 0.15)
        )

        # Randomize magnitudes and durations
        magnitudes = self.np_random.uniform(
            self.squeeze_magnitude * 0.5,
            self.squeeze_magnitude * 1.5,
            n_squeezes
        )

        # Durations in seconds (30s to 3min)
        durations = self.np_random.randint(30, 180, n_squeezes)

        # Add first squeeze
        squeeze_points.append((first_squeeze_idx, magnitudes[0], durations[0]))

        # Place remaining squeezes
        last_idx = first_squeeze_idx
        available_range = num_seconds - last_idx - min_distance

        for i in range(1, n_squeezes):
            # If we're out of range, stop adding squeezes
            if available_range <= 0:
                break

            # Random position for next squeeze, ensuring minimum distance
            next_distance = self.np_random.randint(min_distance, min_distance + available_range)
            next_idx = last_idx + next_distance

            # Add squeeze
            squeeze_points.append((next_idx, magnitudes[i], durations[i]))

            # Update for next iteration
            last_idx = next_idx
            available_range = num_seconds - last_idx - min_distance

        return squeeze_points

    def _generate_volume_profile(self, price_series: pd.Series, squeeze_points: List[tuple]) -> pd.Series:
        """
        Generate realistic volume profile based on price action.

        Args:
            price_series: Price path
            squeeze_points: List of squeeze points

        Returns:
            Series with timestamps as index and volume as values
        """
        # Base volume - lognormal distribution
        base_volume = self.np_random.lognormal(8, 1, len(price_series))

        # Add volume spikes at squeeze points
        for squeeze_idx, magnitude, duration in squeeze_points:
            # Volume builds up before squeeze and peaks during it
            start_idx = max(0, squeeze_idx - duration // 2)
            end_idx = min(len(base_volume), squeeze_idx + duration // 2)

            for i in range(start_idx, end_idx):
                # Volume is highest at squeeze_idx and tapers off
                distance = abs(i - squeeze_idx)
                volume_multiplier = 5 * (1 - distance / (duration // 2))

                if volume_multiplier > 0:
                    base_volume[i] *= (1 + volume_multiplier)

        # Add random volume spikes
        num_vol_spikes = max(3, len(price_series) // 300)  # 1 spike per ~5 minutes
        spike_indices = self.np_random.choice(range(len(base_volume)), num_vol_spikes, replace=False)

        for idx in spike_indices:
            base_volume[idx] *= self.np_random.uniform(2, 4)

        # Round to integers
        volume = np.round(base_volume).astype(int)

        # Create Series with same index as prices
        volume_series = pd.Series(volume, index=price_series.index)

        return volume_series

    def _generate_trades_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Generate trades data with timestamps, prices, sizes, etc.

        Args:
            symbol: Symbol to generate data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with trade data
        """
        # Use cache if available
        cache_key = f"{symbol}_trades_{start_time.isoformat()}_{end_time.isoformat()}"
        if cache_key in self._generated_data_cache:
            return self._generated_data_cache[cache_key]

        # Generate price and volume paths
        price_series = self._generate_price_path(start_time, end_time)

        # Get squeeze points for volume generation
        time_range = (end_time - start_time).total_seconds()
        num_seconds = int(time_range // self.data_sparsity) + 1
        squeeze_points = self._generate_squeeze_points(num_seconds)

        volume_series = self._generate_volume_profile(price_series, squeeze_points)

        # Create trades DataFrame
        trades_data = []
        trade_id = 100000

        # For each second, generate 0-5 trades depending on volume
        for ts, base_price in price_series.items():
            second_volume = volume_series.get(ts, 0)

            # Skip very low volume seconds
            if second_volume < 10:
                continue

            # Number of trades in this second based on volume
            num_trades = min(5, max(1, int(np.log10(second_volume))))

            # Divide volume among trades
            vol_per_trade = second_volume / num_trades

            # Generate each trade
            for i in range(num_trades):
                # Slightly adjust price and volume
                price_adjust = self.np_random.uniform(-0.02, 0.02)
                price = max(0.01, base_price * (1 + price_adjust))

                volume_adjust = self.np_random.uniform(0.5, 1.5)
                size = max(1, int(vol_per_trade * volume_adjust))

                # Determine trade side based on price movement
                if i > 0 and price > trades_data[-1]['price']:
                    side = 'B'  # Buy aggressor if price went up
                elif i > 0 and price < trades_data[-1]['price']:
                    side = 'A'  # Sell aggressor if price went down
                else:
                    side = self.np_random.choice(['B', 'A'], p=[0.5, 0.5])

                # Add trade data
                trade_data = {
                    'timestamp': ts + timedelta(milliseconds=self.np_random.randint(0, 1000)),
                    'price': price,
                    'size': size,
                    'side': side,
                    'exchange': 'NSDQ',
                    'conditions': [],
                    'trade_id': f"T{trade_id}"
                }
                trades_data.append(trade_data)
                trade_id += 1

        # Create DataFrame
        if not trades_data:
            return pd.DataFrame(columns=['price', 'size', 'side', 'exchange', 'conditions', 'trade_id'])

        trades_df = pd.DataFrame(trades_data)
        trades_df.set_index('timestamp', inplace=True)
        trades_df.sort_index(inplace=True)

        # Cache and return
        self._generated_data_cache[cache_key] = trades_df
        return trades_df

    def _generate_quotes_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Generate quotes data with bid/ask prices and sizes.

        Args:
            symbol: Symbol to generate data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with quote data
        """
        # Use cache if available
        cache_key = f"{symbol}_quotes_{start_time.isoformat()}_{end_time.isoformat()}"
        if cache_key in self._generated_data_cache:
            return self._generated_data_cache[cache_key]

        # Get trades data to use as a basis for quotes
        trades_df = self._generate_trades_data(symbol, start_time, end_time)

        if trades_df.empty:
            return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size',
                                         'bid_count', 'ask_count', 'side', 'exchange'])

        # Create time index for quotes (more frequent than trades)
        time_range = (end_time - start_time).total_seconds()
        num_seconds = int(time_range // self.data_sparsity) + 1
        timestamps = [start_time + timedelta(seconds=i * self.data_sparsity) for i in range(num_seconds)]

        # Generate squeeze points for spread analysis
        squeeze_points = self._generate_squeeze_points(num_seconds)
        squeeze_indices = [sp[0] for sp in squeeze_points]

        # Create quotes data
        quotes_data = []

        # Get the squeeze timestamps
        squeeze_timestamps = set()
        for idx in squeeze_indices:
            if 0 <= idx < len(timestamps):
                squeeze_range_start = max(0, idx - 10)
                squeeze_range_end = min(len(timestamps), idx + 10)
                for i in range(squeeze_range_start, squeeze_range_end):
                    squeeze_timestamps.add(timestamps[i])

        for i, ts in enumerate(timestamps):
            # Find closest trade price
            closest_trades = trades_df[trades_df.index <= ts].tail(5)

            if closest_trades.empty:
                closest_trades = trades_df.head(5)

            if closest_trades.empty:
                continue

            # Use average of recent trades for midpoint
            midpoint = closest_trades['price'].mean()

            # Determine spread based on proximity to squeeze points
            near_squeeze = ts in squeeze_timestamps

            if near_squeeze:
                # Tighter spreads before and during momentum moves
                spread_pct = self.np_random.uniform(0.001, 0.005)  # 0.1% to 0.5%
            else:
                # Wider spreads in normal conditions
                spread_pct = self.np_random.uniform(0.005, 0.02)  # 0.5% to 2%

            # Calculate bid and ask
            half_spread = midpoint * spread_pct / 2
            bid_price = midpoint - half_spread
            ask_price = midpoint + half_spread

            # Ensure minimum price of $0.01
            bid_price = max(0.01, bid_price)
            ask_price = max(bid_price + 0.01, ask_price)

            # Generate sizes
            base_size = self.np_random.lognormal(7, 1)

            if near_squeeze:
                # Larger sizes and more orders during squeeze periods
                size_multiplier = self.np_random.uniform(1.5, 3.0)
                count_multiplier = self.np_random.uniform(1.5, 2.0)
            else:
                size_multiplier = 1.0
                count_multiplier = 1.0

            bid_size = int(base_size * size_multiplier * self.np_random.uniform(0.8, 1.2))
            ask_size = int(base_size * size_multiplier * self.np_random.uniform(0.8, 1.2))

            bid_count = max(1, int(self.np_random.poisson(5 * count_multiplier)))
            ask_count = max(1, int(self.np_random.poisson(5 * count_multiplier)))

            # Determine side based on trade activity
            if not closest_trades.empty:
                recent_sides = closest_trades['side'].value_counts()
                if 'B' in recent_sides and 'A' in recent_sides:
                    if recent_sides['B'] > recent_sides['A']:
                        side = 'B'  # More buy trades recently
                    else:
                        side = 'A'  # More sell trades recently
                elif 'B' in recent_sides:
                    side = 'B'
                elif 'A' in recent_sides:
                    side = 'A'
                else:
                    side = 'N'  # No side information
            else:
                side = 'N'

            # Add quote data
            quote_data = {
                'timestamp': ts,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'bid_count': bid_count,
                'ask_count': ask_count,
                'side': side,
                'exchange': 'NSDQ'
            }
            quotes_data.append(quote_data)

        # Create DataFrame
        if not quotes_data:
            return pd.DataFrame(columns=['bid_price', 'ask_price', 'bid_size', 'ask_size',
                                         'bid_count', 'ask_count', 'side', 'exchange'])

        quotes_df = pd.DataFrame(quotes_data)
        quotes_df.set_index('timestamp', inplace=True)
        quotes_df.sort_index(inplace=True)

        # Cache and return
        self._generated_data_cache[cache_key] = quotes_df
        return quotes_df

    def _generate_bars_data(self, symbol: str, timeframe: str, start_time: datetime,
                            end_time: datetime) -> pd.DataFrame:
        """
        Generate OHLCV bar data at specified timeframe.

        Args:
            symbol: Symbol to generate data for
            timeframe: Bar timeframe ("1s", "1m", "5m", "1d")
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with bar data
        """
        # Use cache if available
        cache_key = f"{symbol}_bars_{timeframe}_{start_time.isoformat()}_{end_time.isoformat()}"
        if cache_key in self._generated_data_cache:
            return self._generated_data_cache[cache_key]

        # For 1s bars, use trades directly
        if timeframe == "1s":
            trades_df = self._generate_trades_data(symbol, start_time, end_time)

            if trades_df.empty:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

            # Resample to 1s bars
            bars_df = trades_df.resample('1S').agg({
                'price': ['first', 'max', 'min', 'last', 'count'],
                'size': 'sum'
            })

            # Drop rows with no data
            bars_df = bars_df.dropna()

            if bars_df.empty:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

            # Flatten MultiIndex columns
            bars_df.columns = ['open', 'high', 'low', 'close', 'count', 'volume']
            bars_df = bars_df.drop(columns=['count'])

        # For other timeframes, resample from 1s bars
        else:
            # Get 1s bars first
            bars_1s = self._generate_bars_data(symbol, "1s", start_time, end_time)

            if bars_1s.empty:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

            # Resample to target timeframe
            resample_rule = {
                "1m": "1min",
                "5m": "5min",
                "1d": "D"
            }.get(timeframe)

            if not resample_rule:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            bars_df = bars_1s.resample(resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            # Drop rows with no data
            bars_df = bars_df.dropna()

        # Cache and return
        self._generated_data_cache[cache_key] = bars_df
        return bars_df

    def _generate_status_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Generate status data for the symbol.

        Args:
            symbol: Symbol to generate data for
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with status data
        """
        # Use cache if available
        cache_key = f"{symbol}_status_{start_time.isoformat()}_{end_time.isoformat()}"
        if cache_key in self._generated_data_cache:
            return self._generated_data_cache[cache_key]

        # Create regular trading hours status
        # NYSE/NASDAQ market hours: 9:30 AM - 4:00 PM ET
        market_open_et = time(9, 30, 0)
        market_close_et = time(16, 0, 0)
        pre_market_open_et = time(4, 0, 0)
        post_market_close_et = time(20, 0, 0)

        # Create a status record for market open, close, and any transitions
        status_data = []

        # Convert to Eastern timezone for market hours
        start_time_et = start_time.astimezone(ZoneInfo("America/New_York"))
        end_time_et = end_time.astimezone(ZoneInfo("America/New_York"))

        current_date = start_time_et.date()
        end_date = end_time_et.date()

        while current_date <= end_date:
            # Pre-market open
            pre_open_time = datetime.combine(current_date, pre_market_open_et, ZoneInfo("America/New_York"))
            if start_time_et <= pre_open_time <= end_time_et:
                status_data.append({
                    'timestamp': pre_open_time.astimezone(ZoneInfo("UTC")),
                    'status': 'PRE_OPEN',
                    'reason': 'SCHEDULED',
                    'is_trading': True,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

            # Regular market open
            market_open_time = datetime.combine(current_date, market_open_et, ZoneInfo("America/New_York"))
            if start_time_et <= market_open_time <= end_time_et:
                status_data.append({
                    'timestamp': market_open_time.astimezone(ZoneInfo("UTC")),
                    'status': 'TRADING',
                    'reason': 'SCHEDULED',
                    'is_trading': True,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

            # Regular market close
            market_close_time = datetime.combine(current_date, market_close_et, ZoneInfo("America/New_York"))
            if start_time_et <= market_close_time <= end_time_et:
                status_data.append({
                    'timestamp': market_close_time.astimezone(ZoneInfo("UTC")),
                    'status': 'POST_CLOSE',
                    'reason': 'SCHEDULED',
                    'is_trading': True,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

            # Post-market close
            post_close_time = datetime.combine(current_date, post_market_close_et, ZoneInfo("America/New_York"))
            if start_time_et <= post_close_time <= end_time_et:
                status_data.append({
                    'timestamp': post_close_time.astimezone(ZoneInfo("UTC")),
                    'status': 'CLOSED',
                    'reason': 'SCHEDULED',
                    'is_trading': False,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

            # Move to next day
            current_date += timedelta(days=1)

        # Generate random halts if configuration says to
        if self.config.get('random_halts', False):
            # Get trading days in range
            trading_days = []
            current_date = start_time_et.date()
            while current_date <= end_time_et.date():
                # Skip weekends
                if current_date.weekday() < 5:  # Monday-Friday
                    trading_days.append(current_date)
                current_date += timedelta(days=1)

            # Add 0-2 random halts
            num_halts = self.np_random.randint(0, 3)
            for _ in range(num_halts):
                if not trading_days:
                    break

                # Pick a random trading day
                halt_date = self.np_random.choice(trading_days)

                # Random time during regular hours (9:30 AM - 4:00 PM ET)
                halt_hour = self.np_random.randint(9, 16)
                if halt_hour == 9:
                    halt_minute = self.np_random.randint(30, 60)
                elif halt_hour == 15:
                    halt_minute = self.np_random.randint(0, 59)
                else:
                    halt_minute = self.np_random.randint(0, 60)

                halt_time = datetime.combine(
                    halt_date,
                    time(halt_hour, halt_minute, 0),
                    ZoneInfo("America/New_York")
                )

                # Skip if outside our range
                if halt_time < start_time_et or halt_time > end_time_et:
                    continue

                # Add halt status
                halt_reasons = ['SURVEILLANCE', 'NEWS_PENDING', 'ORDER_IMBALANCE']
                status_data.append({
                    'timestamp': halt_time.astimezone(ZoneInfo("UTC")),
                    'status': 'HALTED',
                    'reason': self.np_random.choice(halt_reasons),
                    'is_trading': False,
                    'is_halted': True,
                    'is_short_sell_restricted': False
                })

                # Add resumption 5-30 minutes later
                resume_time = halt_time + timedelta(minutes=self.np_random.randint(5, 30))

                # Skip if outside our range
                if resume_time < start_time_et or resume_time > end_time_et:
                    continue

                status_data.append({
                    'timestamp': resume_time.astimezone(ZoneInfo("UTC")),
                    'status': 'TRADING',
                    'reason': 'SCHEDULED',
                    'is_trading': True,
                    'is_halted': False,
                    'is_short_sell_restricted': False
                })

        # Create DataFrame
        if not status_data:
            # Return at least the current status
            mid_time = start_time + (end_time - start_time) / 2
            mid_time_et = mid_time.astimezone(ZoneInfo("America/New_York"))
            current_time_et = mid_time_et.time()

            if market_open_et <= current_time_et <= market_close_et:
                status = 'TRADING'
                is_trading = True
            elif pre_market_open_et <= current_time_et < market_open_et:
                status = 'PRE_OPEN'
                is_trading = True
            elif market_close_et < current_time_et <= post_market_close_et:
                status = 'POST_CLOSE'
                is_trading = True
            else:
                status = 'CLOSED'
                is_trading = False

            status_data.append({
                'timestamp': mid_time,
                'status': status,
                'reason': 'SCHEDULED',
                'is_trading': is_trading,
                'is_halted': False,
                'is_short_sell_restricted': False
            })

        status_df = pd.DataFrame(status_data)
        status_df.set_index('timestamp', inplace=True)
        status_df.sort_index(inplace=True)

        # Cache and return
        self._generated_data_cache[cache_key] = status_df
        return status_df