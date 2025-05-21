# data/provider/dummy_data_provider.py
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union, Tuple, Any

from data.provider.data_provider import HistoricalDataProvider


class DummyDataProvider(HistoricalDataProvider):
    """
    Synthetic data provider that generates dummy OHLCV, trade, and quote data
    with momentum squeezes, pullbacks, and realistic volume patterns.

    Perfect for fast development and testing without the need for real data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dummy data provider.

        Args:
            config: Configuration dictionary with the following keys:
                - debug_window_mins: Minutes of data to generate
                - data_sparsity: Generate data every N seconds
                - num_squeezes: Number of momentum squeezes to simulate
                - base_price: Starting price for simulation
                - volatility: Base market volatility
                - squeeze_magnitude: Average magnitude of squeezes (as percent)
                - symbols: List of symbols to simulate (usually one)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Set defaults if not provided
        self.debug_window_mins = config.get('debug_window_mins', 120)  # 2 hours by default
        self.data_sparsity = config.get('data_sparsity', 1)  # Default to 1s data
        self.num_squeezes = config.get('num_squeezes', 2)
        self.base_price = config.get('base_price', 5.00)
        self.volatility = config.get('volatility', 0.03)
        self.squeeze_magnitude = config.get('squeeze_magnitude', 0.30)
        self.symbols = config.get('symbols', ['DUMMY'])

        # Set up random generator for reproducibility
        self.np_random = np.random.RandomState(42)

        # Cache generated data
        self.data_cache = {}

        # Generate synthetic data for all symbols up front
        for symbol in self.symbols:
            if symbol not in self.data_cache:
                self.data_cache[symbol] = self._generate_data_for_symbol(symbol)
                self.logger.info(
                    f"Generated dummy data for {symbol}: {self.debug_window_mins} mins at {self.data_sparsity}s intervals")

    def _generate_price_path(self, num_points: int, squeeze_points: List[Tuple[int, float]]) -> np.ndarray:
        """Generate a price path with specified squeezes at certain points."""
        # Generate base price path with random walk
        base_prices = np.zeros(num_points)
        base_prices[0] = self.base_price

        # Generate daily trend direction (-1 to +1)
        trend = self.np_random.uniform(-0.5, 0.5)

        # Add random walk with trend
        for i in range(1, num_points):
            random_change = self.np_random.normal(0, self.volatility)
            trend_change = trend * self.volatility * 0.5
            base_prices[i] = base_prices[i - 1] * (1 + random_change + trend_change)

        # Add squeezes
        for squeeze_idx, squeeze_strength in squeeze_points:
            if squeeze_idx < num_points - 30:  # Ensure enough room for the squeeze
                # Create the squeeze pattern (sharp rise followed by partial retracement)
                squeeze_length = int(20 + self.np_random.randint(10, 30))  # 30-50 seconds

                # Generate the squeeze points                
                squeeze_start_price = base_prices[squeeze_idx]
                squeeze_max = squeeze_start_price * (1 + squeeze_strength)

                # Calculate squeeze and retracement phases
                ramp_up_length = int(squeeze_length * 0.3)  # First 30% is ramp up
                hold_length = int(squeeze_length * 0.2)  # Next 20% is holding near high
                retracement_length = squeeze_length - ramp_up_length - hold_length  # Rest is retracement

                # Ramp up phase
                for j in range(ramp_up_length):
                    if squeeze_idx + j < num_points:
                        progress = j / ramp_up_length
                        # Accelerating rise with some noise
                        base_prices[squeeze_idx + j] = squeeze_start_price + progress ** 2 * (
                                    squeeze_max - squeeze_start_price) * \
                                                       (1 + self.np_random.normal(0, 0.01))

                # Hold phase with oscillation
                for j in range(hold_length):
                    if squeeze_idx + ramp_up_length + j < num_points:
                        oscillation = self.np_random.normal(0, 0.01) * squeeze_max
                        base_prices[squeeze_idx + ramp_up_length + j] = squeeze_max + oscillation

                # Retracement phase
                retracement_pct = self.np_random.uniform(0.3, 0.7)  # Retrace 30-70%
                retracement_target = squeeze_max - (squeeze_max - squeeze_start_price) * retracement_pct

                for j in range(retracement_length):
                    if squeeze_idx + ramp_up_length + hold_length + j < num_points:
                        progress = j / retracement_length
                        # Faster initial drop, then slower
                        drop_factor = 1 - (1 - progress) ** 2
                        base_prices[squeeze_idx + ramp_up_length + hold_length + j] = \
                            squeeze_max - drop_factor * (squeeze_max - retracement_target) * \
                            (1 + self.np_random.normal(0, 0.01))

        return base_prices

    def _generate_volume_profile(self, num_points: int, price_path: np.ndarray,
                                 squeeze_points: List[Tuple[int, float]]) -> np.ndarray:
        """Generate matching volume profile with elevated volumes during squeezes."""
        base_volumes = self.np_random.exponential(scale=1000, size=num_points)

        # Normalize volumes
        base_volumes = base_volumes / base_volumes.mean() * 5000  # Average 5000 shares

        # Add volume surges around squeezes
        for squeeze_idx, squeeze_strength in squeeze_points:
            if squeeze_idx < num_points:
                # Volume leads price - increase volume slightly before the price moves
                pre_squeeze_start = max(0, squeeze_idx - 5)
                for j in range(pre_squeeze_start, squeeze_idx):
                    if j < num_points:
                        # Gradual volume increase leading to the squeeze
                        ramp_factor = (j - pre_squeeze_start + 1) / (squeeze_idx - pre_squeeze_start + 1)
                        base_volumes[j] *= (1 + 5 * ramp_factor)  # Up to 6x normal volume

                # Surge during the squeeze
                squeeze_length = 30
                squeeze_peak = min(squeeze_idx + int(squeeze_length * 0.3), num_points - 1)

                # Volume climax at price breakout point
                for j in range(squeeze_idx, squeeze_peak):
                    if j < num_points:
                        # Volume surges with price velocity
                        price_velocity = (price_path[j] - price_path[j - 1]) / price_path[j - 1] if j > 0 else 0
                        base_volumes[j] *= (3 + 20 * abs(price_velocity))  # Volume spike based on price velocity

                # Elevated but decreasing volume after peak
                post_peak_length = min(squeeze_length - int(squeeze_length * 0.3), num_points - squeeze_peak)
                for j in range(post_peak_length):
                    if squeeze_peak + j < num_points:
                        # Gradual volume decrease after peak
                        decay_factor = 1 - (j / post_peak_length) ** 2
                        base_volumes[squeeze_peak + j] *= (1 + 5 * decay_factor)

        # Add random spikes throughout
        spike_indices = self.np_random.choice(num_points, size=int(num_points * 0.05), replace=False)
        for idx in spike_indices:
            base_volumes[idx] *= self.np_random.uniform(2, 5)

        return base_volumes.astype(int)

    def _generate_data_for_symbol(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Generate all necessary data for a symbol.

        Returns:
            Dictionary with dataframes for each data type
        """
        # Calculate time points
        total_seconds = self.debug_window_mins * 60
        num_points = total_seconds // self.data_sparsity + 1

        # Generate reference datetime range
        end_time = datetime.now(timezone.utc).replace(microsecond=0, second=0)
        start_time = end_time - timedelta(minutes=self.debug_window_mins)
        time_index = pd.date_range(start=start_time, end=end_time, periods=num_points)

        # Generate squeeze points for this symbol
        squeeze_points = []
        for i in range(self.num_squeezes):
            # Random squeeze timing and magnitude
            squeeze_idx = self.np_random.randint(30, num_points - 100)
            squeeze_strength = self.np_random.uniform(0.2, self.squeeze_magnitude * 1.5)
            squeeze_points.append((squeeze_idx, squeeze_strength))

        # Sort squeeze points by index
        squeeze_points.sort(key=lambda x: x[0])

        # Generate synthetic price path
        prices = self._generate_price_path(num_points, squeeze_points)
        volumes = self._generate_volume_profile(num_points, prices, squeeze_points)

        # 1-second bars (OHLCV)
        df_1s = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + self.np_random.uniform(0, 0.002, num_points)),
            'low': prices * (1 - self.np_random.uniform(0, 0.002, num_points)),
            'close': prices,
            'volume': volumes
        }, index=time_index)

        # Make OHLC consistent
        for i in range(len(df_1s)):
            df_1s.loc[df_1s.index[i], 'high'] = max(
                df_1s.loc[df_1s.index[i], 'open'],
                df_1s.loc[df_1s.index[i], 'close'],
                df_1s.loc[df_1s.index[i], 'high']
            )
            df_1s.loc[df_1s.index[i], 'low'] = min(
                df_1s.loc[df_1s.index[i], 'open'],
                df_1s.loc[df_1s.index[i], 'close'],
                df_1s.loc[df_1s.index[i], 'low']
            )

        # Generate 1-minute and 5-minute bars by resampling
        df_1m = df_1s.resample('1T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df_5m = df_1s.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df_1d = df_1s.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Generate trades data
        trades_per_bar = 5  # Average number of trades per second
        trade_rows = []

        for i in range(len(df_1s)):
            bar = df_1s.iloc[i]
            timestamp = df_1s.index[i]

            # Number of trades for this second varies with volume
            num_trades = max(1, int(bar['volume'] / 1000 * trades_per_bar))

            # Generate trades within this second
            for j in range(num_trades):
                trade_time = timestamp + timedelta(microseconds=self.np_random.randint(0, 1000000))

                # Calculate price with jitter around the OHLC
                price_base = self.np_random.uniform(bar['low'], bar['high'])

                # Determine if this is a buy or sell
                if price_base > bar['open']:
                    side = 'B'  # Buy aggressor - price going up
                elif price_base < bar['open']:
                    side = 'A'  # Sell aggressor - price going down
                else:
                    side = 'N'  # No side indicated

                # Trade size varies with volume
                size = max(100, min(10000, int(self.np_random.exponential(bar['volume'] / num_trades / 2))))

                trade_rows.append({
                    'timestamp': trade_time,
                    'price': price_base,
                    'size': size,
                    'side': side,
                    'exchange': 'XNAS',
                    'conditions': [],
                    'trade_id': f"{symbol}_{i}_{j}"
                })

        # Convert trades to dataframe
        if trade_rows:
            df_trades = pd.DataFrame(trade_rows)
            df_trades.set_index('timestamp', inplace=True)
            df_trades.sort_index(inplace=True)
        else:
            df_trades = pd.DataFrame(columns=['price', 'size', 'side', 'exchange', 'conditions', 'trade_id'])
            df_trades.index.name = 'timestamp'

        # Generate quotes data
        quotes_per_bar = 10  # More quote updates than trades
        quote_rows = []

        for i in range(len(df_1s)):
            bar = df_1s.iloc[i]
            timestamp = df_1s.index[i]

            # Number of quote updates for this second
            num_quotes = max(1, int(quotes_per_bar))

            # Generate quotes within this second
            for j in range(num_quotes):
                quote_time = timestamp + timedelta(microseconds=self.np_random.randint(0, 1000000))

                # Calculate bid and ask with realistic spread
                mid_price = bar['close']
                spread_bps = min(20, max(1, int(100 / mid_price)))  # Higher spread for lower prices

                # Tighter spreads during squeezes
                for sq_idx, sq_str in squeeze_points:
                    if abs(i - sq_idx) < 15:  # Near a squeeze point
                        spread_bps = max(1, spread_bps // 2)  # Tighter spread during momentum

                spread_amt = mid_price * (spread_bps / 10000)  # Convert bps to decimal

                # Sometimes flip bid/ask aggressively around NBBO to simulate momentum
                bid_price = mid_price - spread_amt / 2
                ask_price = mid_price + spread_amt / 2

                # During squeezes, raise bid aggressively 
                for sq_idx, sq_str in squeeze_points:
                    if i > sq_idx and i < sq_idx + 20:  # During a squeeze
                        if self.np_random.random() < 0.7:  # 70% chance
                            bid_price = mid_price - spread_amt / 4  # Tighter to mid

                # Randomize sizes with larger sizes for buyers during squeezes
                bid_size = max(100, int(self.np_random.exponential(2000)))
                ask_size = max(100, int(self.np_random.exponential(1500)))

                # Larger bids during squeezes
                for sq_idx, sq_str in squeeze_points:
                    if i > sq_idx - 5 and i < sq_idx + 10:  # Right before and during squeeze
                        bid_size = max(bid_size, int(bid_size * 2.5))  # Larger bids

                bid_count = max(1, int(bid_size / 500))
                ask_count = max(1, int(ask_size / 500))

                quote_rows.append({
                    'timestamp': quote_time,
                    'bid_price': bid_price,
                    'ask_price': ask_price,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'bid_count': bid_count,
                    'ask_count': ask_count,
                    'side': 'N',  # No side for quotes
                    'exchange': 'XNAS'
                })

        # Convert quotes to dataframe
        if quote_rows:
            df_quotes = pd.DataFrame(quote_rows)
            df_quotes.set_index('timestamp', inplace=True)
            df_quotes.sort_index(inplace=True)
        else:
            df_quotes = pd.DataFrame(columns=[
                'bid_price', 'ask_price', 'bid_size', 'ask_size',
                'bid_count', 'ask_count', 'side', 'exchange'
            ])
            df_quotes.index.name = 'timestamp'

        # Generate status data - trading status updates
        status_rows = []

        # Normal trading status
        status_rows.append({
            'timestamp': start_time,
            'status': 'TRADING',
            'reason': 'SCHEDULED',
            'is_trading': True,
            'is_halted': False,
            'is_short_sell_restricted': False
        })

        # Add a trading halt near a strong squeeze if there's one
        strong_squeezes = [(idx, strength) for idx, strength in squeeze_points if strength > 0.25]
        if strong_squeezes:
            sq_idx, sq_str = strong_squeezes[0]
            if sq_idx < num_points - 100:  # Ensure enough time after for resumption
                halt_time = time_index[sq_idx + 30]  # Halt shortly after the squeeze peak
                resume_time = halt_time + timedelta(minutes=5)  # 5 minute halt

                # Add halt
                status_rows.append({
                    'timestamp': halt_time,
                    'status': 'HALTED',
                    'reason': 'LULD_PAUSE',
                    'is_trading': False,
                    'is_halted': True,
                    'is_short_sell_restricted': False
                })

                # Add resumption 
                status_rows.append({
                    'timestamp': resume_time,
                    'status': 'TRADING',
                    'reason': 'SCHEDULED',
                    'is_trading': True,
                    'is_halted': False,
                    'is_short_sell_restricted': True  # Often SSR after halt
                })

        # Convert status to dataframe
        df_status = pd.DataFrame(status_rows)
        if not df_status.empty:
            df_status.set_index('timestamp', inplace=True)
            df_status.sort_index(inplace=True)

        # Return all dataframes in a dictionary
        return {
            'bars_1s': df_1s,
            'bars_1m': df_1m,
            'bars_5m': df_5m,
            'bars_1d': df_1d,
            'trades': df_trades,
            'quotes': df_quotes,
            'status': df_status
        }

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get metadata for a symbol."""
        if symbol not in self.symbols:
            self.logger.warning(f"Requested information for unknown symbol: {symbol}")
            return {"symbol": symbol, "description": "Unknown dummy symbol"}

        return {
            "symbol": symbol,
            "description": f"Dummy data for {symbol}",
            "is_synthetic": True,
            "exchange": "XNAS",
            "industry": "Technology"
        }

    def get_available_symbols(self) -> List[str]:
        """Get all available symbols."""
        return self.symbols.copy()

    def get_trades(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get trade data for a symbol within a time range."""
        if symbol not in self.data_cache:
            self.logger.warning(f"No data for symbol: {symbol}")
            return pd.DataFrame()

        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        trades_df = self.data_cache[symbol].get('trades', pd.DataFrame())
        if trades_df.empty:
            return trades_df

        # Slice by time range
        return trades_df[(trades_df.index >= start_time) & (trades_df.index <= end_time)]

    def get_quotes(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get quote data for a symbol within a time range."""
        if symbol not in self.data_cache:
            self.logger.warning(f"No data for symbol: {symbol}")
            return pd.DataFrame()

        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        quotes_df = self.data_cache[symbol].get('quotes', pd.DataFrame())
        if quotes_df.empty:
            return quotes_df

        # Slice by time range
        return quotes_df[(quotes_df.index >= start_time) & (quotes_df.index <= end_time)]

    def get_bars(self, symbol: str, timeframe: str, start_time: Union[datetime, str],
                 end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get bar data for a symbol within a time range."""
        if symbol not in self.data_cache:
            self.logger.warning(f"No data for symbol: {symbol}")
            return pd.DataFrame()

        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        # Map timeframe to data key
        data_key = f'bars_{timeframe}'
        if data_key not in self.data_cache[symbol]:
            self.logger.warning(f"Unsupported timeframe: {timeframe}")
            return pd.DataFrame()

        bars_df = self.data_cache[symbol][data_key]
        if bars_df.empty:
            return bars_df

        # Slice by time range
        return bars_df[(bars_df.index >= start_time) & (bars_df.index <= end_time)]

    def get_status(self, symbol: str, start_time: Union[datetime, str],
                   end_time: Union[datetime, str]) -> pd.DataFrame:
        """Get status data for a symbol within a time range."""
        if symbol not in self.data_cache:
            self.logger.warning(f"No data for symbol: {symbol}")
            return pd.DataFrame()

        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.Timestamp(start_time)
        if isinstance(end_time, str):
            end_time = pd.Timestamp(end_time)

        status_df = self.data_cache[symbol].get('status', pd.DataFrame())
        if status_df.empty:
            return status_df

        # Slice by time range
        return status_df[(status_df.index >= start_time) & (status_df.index <= end_time)]