import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from collections import deque

# Assuming the following files are in the PYTHONPATH or same directory
from config.config import MarketConfig, ModelConfig, EnvConfig, DataConfig #
from data.data_manager import DataManager #
from simulators.market_simulator import MARKET_HOURS, MarketSimulator

# --- Test Configuration ---
TEST_SYMBOL = "TESTSMBL"
SESSION_DATE_STR = "2025-03-27"
PREV_TRADING_DATE_STR = "2025-03-26"
DAY_BEFORE_PREV_TRADING_DATE_STR = "2025-03-25"

MARKET_TZ = ZoneInfo(MARKET_HOURS["TIMEZONE"]) #
UTC_TZ = ZoneInfo("UTC")

# Default start and end times for a typical simulation session
SESSION_START_ET_STR = f"{SESSION_DATE_STR} 09:30:00"
SESSION_END_ET_STR = f"{SESSION_DATE_STR} 16:00:00"

class TestMarketSimulator(unittest.TestCase):

    def _create_sample_datetime(self, date_str, time_obj, tzinfo=MARKET_TZ):
        return datetime.combine(date.fromisoformat(date_str), time_obj, tzinfo=tzinfo)

    def _generate_timestamps(self, start_dt_utc, end_dt_utc, delta):
        timestamps = []
        current_dt = start_dt_utc
        while current_dt <= end_dt_utc:
            timestamps.append(current_dt)
            current_dt += delta
        return timestamps

    def setUp(self):
        """Set up test fixtures, including mock objects and sample data."""
        self.market_config = MarketConfig()
        self.model_config = ModelConfig(hf_seq_len=60, mf_seq_len=30, lf_seq_len=15) # Modified for easier window testing
        self.env_config = EnvConfig()
        self.data_config = DataConfig(symbol=TEST_SYMBOL, start_date=SESSION_DATE_STR, end_date=SESSION_DATE_STR)

        self.mock_data_manager = MagicMock(spec=DataManager)

        # --- Sample Data Generation ---
        # Previous Trading Day (Day 1: 2025-03-26)
        self.prev_day_date = date.fromisoformat(PREV_TRADING_DATE_STR)
        self.prev_day_premarket_start_utc = self._create_sample_datetime(PREV_TRADING_DATE_STR, MARKET_HOURS["PREMARKET_START"]).astimezone(UTC_TZ)
        self.prev_day_regular_start_utc = self._create_sample_datetime(PREV_TRADING_DATE_STR, MARKET_HOURS["REGULAR_START"]).astimezone(UTC_TZ)
        self.prev_day_regular_end_utc = self._create_sample_datetime(PREV_TRADING_DATE_STR, MARKET_HOURS["REGULAR_END"]).astimezone(UTC_TZ)
        self.prev_day_postmarket_end_utc = self._create_sample_datetime(PREV_TRADING_DATE_STR, MARKET_HOURS["POSTMARKET_END"]).astimezone(UTC_TZ)

        # Current Session Day (Day 2: 2025-03-27)
        self.session_date = date.fromisoformat(SESSION_DATE_STR)
        self.session_day_premarket_start_utc = self._create_sample_datetime(SESSION_DATE_STR, MARKET_HOURS["PREMARKET_START"]).astimezone(UTC_TZ)
        self.session_day_regular_start_utc = self._create_sample_datetime(SESSION_DATE_STR, MARKET_HOURS["REGULAR_START"]).astimezone(UTC_TZ)
        self.session_day_regular_end_utc = self._create_sample_datetime(SESSION_DATE_STR, MARKET_HOURS["REGULAR_END"]).astimezone(UTC_TZ)
        self.session_day_postmarket_end_utc = self._create_sample_datetime(SESSION_DATE_STR, MARKET_HOURS["POSTMARKET_END"]).astimezone(UTC_TZ)

        # --- Mock DataFrames ---
        # Daily Bars for finding previous trading day
        daily_indices = [
            pd.Timestamp(f"{DAY_BEFORE_PREV_TRADING_DATE_STR} 00:00:00", tz=MARKET_TZ).tz_convert(UTC_TZ),
            pd.Timestamp(f"{PREV_TRADING_DATE_STR} 00:00:00", tz=MARKET_TZ).tz_convert(UTC_TZ),
            pd.Timestamp(f"{SESSION_DATE_STR} 00:00:00", tz=MARKET_TZ).tz_convert(UTC_TZ)
        ]
        self.sample_daily_bars_df = pd.DataFrame({
            'open': [100, 110, 120], 'high': [105, 115, 125],
            'low': [95, 105, 115], 'close': [102, 112, 122], # prev_day_close = 112
            'volume': [1000, 1100, 1200]
        }, index=pd.DatetimeIndex(daily_indices, name='timestamp'))

        # Previous Day Minute Bars (for VWAP and priming)
        prev_day_1m_bar_timestamps = self._generate_timestamps(
            self.prev_day_premarket_start_utc, self.prev_day_postmarket_end_utc, timedelta(minutes=1)
        )
        self.prev_day_1m_bars_df = pd.DataFrame({
            'open': np.linspace(110, 115, len(prev_day_1m_bar_timestamps)),
            'high': np.linspace(111, 116, len(prev_day_1m_bar_timestamps)),
            'low': np.linspace(109, 114, len(prev_day_1m_bar_timestamps)),
            'close': np.linspace(110.5, 115.5, len(prev_day_1m_bar_timestamps)), # Post market close around 115.5
            'volume': np.random.randint(10, 100, len(prev_day_1m_bar_timestamps))
        }, index=pd.DatetimeIndex(prev_day_1m_bar_timestamps, name='timestamp'))
        # Ensure a specific close for post-market
        self.prev_day_post_market_close_price = 115.88
        last_idx_prev_day = self.prev_day_1m_bars_df.index.get_loc(self.prev_day_postmarket_end_utc, method='nearest')
        self.prev_day_1m_bars_df.loc[self.prev_day_1m_bars_df.index[last_idx_prev_day], 'close'] = self.prev_day_post_market_close_price


        # Session Day Minute Bars
        session_day_1m_bar_timestamps = self._generate_timestamps(
            self.session_day_premarket_start_utc, self.session_day_postmarket_end_utc, timedelta(minutes=1)
        )
        self.session_day_1m_bars_df = pd.DataFrame({
            'open': np.linspace(120, 125, len(session_day_1m_bar_timestamps)),
            'high': np.linspace(121, 126, len(session_day_1m_bar_timestamps)),
            'low': np.linspace(119, 124, len(session_day_1m_bar_timestamps)),
            'close': np.linspace(120.5, 125.5, len(session_day_1m_bar_timestamps)),
            'volume': np.random.randint(10, 100, len(session_day_1m_bar_timestamps))
        }, index=pd.DatetimeIndex(session_day_1m_bar_timestamps, name='timestamp'))
        self.first_session_day_premarket_open = 120.0
        self.session_day_1m_bars_df.loc[self.session_day_premarket_start_utc, 'open'] = self.first_session_day_premarket_open
        self.session_day_1m_bars_df.loc[self.session_day_premarket_start_utc, 'high'] = self.first_session_day_premarket_open + 0.5
        self.session_day_1m_bars_df.loc[self.session_day_premarket_start_utc, 'low'] = self.first_session_day_premarket_open -0.5
        self.session_day_1m_bars_df.loc[self.session_day_premarket_start_utc, 'close'] = self.first_session_day_premarket_open + 0.1


        # Combine 1m bars for DataManager mock
        self.all_1m_bars_df = pd.concat([self.prev_day_1m_bars_df, self.session_day_1m_bars_df])

        # Session Day Trades and Quotes (sparse, for 1s bar aggregation and price updates)
        trade_times_utc = [
            self.session_day_premarket_start_utc + timedelta(seconds=10, milliseconds=100), # 120.1
            self.session_day_premarket_start_utc + timedelta(seconds=10, milliseconds=200), # 120.2
            self.session_day_premarket_start_utc + timedelta(minutes=1, seconds=5), # 120.3
            self.session_day_regular_start_utc + timedelta(seconds=30), # 122.0
        ]
        self.sample_trades_df = pd.DataFrame({
            'price': [120.1, 120.2, 120.3, 122.0],
            'size': [10, 5, 12, 100]
        }, index=pd.DatetimeIndex(trade_times_utc, name='timestamp'))

        quote_times_utc = [
            self.session_day_premarket_start_utc + timedelta(seconds=5), # Bid: 119.9, Ask: 120.1
            self.session_day_premarket_start_utc + timedelta(minutes=1), # Bid: 120.15, Ask: 120.25
            self.session_day_regular_start_utc + timedelta(seconds=15),# Bid: 121.9, Ask: 122.1
        ]
        self.sample_quotes_df = pd.DataFrame({
            'bid_price': [119.9, 120.15, 121.9],
            'ask_price': [120.1, 120.25, 122.1],
            'bid_size': [100, 200, 150],
            'ask_size': [120, 180, 160]
        }, index=pd.DatetimeIndex(quote_times_utc, name='timestamp'))

        # Mock DataManager methods
        def mock_get_bars(symbol, timeframe, start_time, end_time):
            if timeframe == "1d":
                return self.sample_daily_bars_df[(self.sample_daily_bars_df.index >= start_time) & (self.sample_daily_bars_df.index <= end_time)].copy()
            elif timeframe == "1m":
                return self.all_1m_bars_df[(self.all_1m_bars_df.index >= start_time) & (self.all_1m_bars_df.index <= end_time)].copy()
            elif timeframe == "5m": # For simplicity, allow direct 5m load for now, or could build from 1m
                df_5m = self.all_1m_bars_df.resample('5min', label='left', closed='left').agg(
                    {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                ).dropna()
                return df_5m[(df_5m.index >= start_time) & (df_5m.index <= end_time)].copy()
            return pd.DataFrame()

        def mock_load_data(symbols, start_time, end_time, data_types):
            # This mock should reflect the ranges MarketSimulator calculates
            # For the purpose of this test, we make sure prev_day and session_day data is "loaded"
            # if the time ranges overlap.
            loaded_data = {TEST_SYMBOL: {}}
            if "trades" in data_types:
                loaded_data[TEST_SYMBOL]["trades"] = self.sample_trades_df[
                    (self.sample_trades_df.index >= start_time) & (self.sample_trades_df.index <= end_time)
                ].copy()
            if "quotes" in data_types:
                loaded_data[TEST_SYMBOL]["quotes"] = self.sample_quotes_df[
                    (self.sample_quotes_df.index >= start_time) & (self.sample_quotes_df.index <= end_time)
                ].copy()
            return loaded_data

        self.mock_data_manager.get_bars.side_effect = mock_get_bars
        self.mock_data_manager.load_data.side_effect = mock_load_data

        # Initialize MarketSimulator instance for testing
        self.simulator = MarketSimulator(
            symbol=TEST_SYMBOL,
            data_manager=self.mock_data_manager,
            market_config=self.market_config,
            model_config=self.model_config,
            start_time=SESSION_START_ET_STR, # Sim will load data around this
            end_time=SESSION_END_ET_STR,
            logger=MagicMock() # Suppress logging during tests
        )

    ## Test Cases ##
    # --------------------
    def test_01_initialization_and_previous_day_data_loading(self):
        """
        Test correct initialization, identification of the previous trading day,
        and loading of its summary data.
        """
        self.assertIsNotNone(self.simulator)
        self.assertEqual(self.simulator.prev_day_date, self.prev_day_date) #

        prev_day_data = self.simulator.get_previous_day_data() #
        self.assertEqual(prev_day_data['date'], self.prev_day_date)
        self.assertEqual(prev_day_data['open'], 110)  # From sample_daily_bars_df
        self.assertEqual(prev_day_data['high'], 115)
        self.assertEqual(prev_day_data['low'], 105)
        self.assertEqual(prev_day_data['close'], 112) # From sample_daily_bars_df
        self.assertIsNotNone(prev_day_data['vwap']) # VWAP is calculated from 1m bars

        # Check if data loading ranges were calculated to include previous day
        # The actual call to load_data is mocked, but we check if MarketSimulator
        # *would* have asked for previous day's data by inspecting its raw DFs.
        # It's implicitly tested by the presence of prev_day_date and prev_day_data.
        # A more direct test would be to spy on _calculate_data_loading_ranges if needed.
        self.assertFalse(self.simulator.raw_1m_bars_df.empty)
        # Check if prev day's data is in raw_1m_bars_df (due to loading ranges)
        self.assertTrue(self.prev_day_1m_bars_df.index.min() in self.simulator.raw_1m_bars_df.index)
        self.assertTrue(self.prev_day_1m_bars_df.index.max() in self.simulator.raw_1m_bars_df.index)

    def test_02_timeline_generation(self):
        """
        Test the generation of agent timeline (1-second), 1-minute bar timeline,
        and 5-minute bar timeline for the session day (4 AM - 8 PM ET).
        """
        sim_start_utc = self._create_sample_datetime(SESSION_DATE_STR, MARKET_HOURS["PREMARKET_START"]).astimezone(UTC_TZ) #
        sim_end_utc = self._create_sample_datetime(SESSION_DATE_STR, MARKET_HOURS["POSTMARKET_END"]).astimezone(UTC_TZ) #

        # Agent Timeline (1-second)
        self.assertNotEmpty(self.simulator._agent_timeline_utc)
        self.assertEqual(self.simulator._agent_timeline_utc[0], sim_start_utc)
        self.assertEqual(self.simulator._agent_timeline_utc[-1], sim_end_utc)
        self.assertEqual(self.simulator._agent_timeline_utc[1] - self.simulator._agent_timeline_utc[0], timedelta(seconds=1))
        expected_agent_len = (sim_end_utc - sim_start_utc).total_seconds() + 1
        self.assertEqual(len(self.simulator._agent_timeline_utc), expected_agent_len)

        # 1-Minute Bar Timeline
        self.assertNotEmpty(self.simulator._one_min_bar_timeline)
        self.assertEqual(self.simulator._one_min_bar_timeline[0], sim_start_utc.replace(second=0, microsecond=0))
        self.assertEqual(self.simulator._one_min_bar_timeline[-1], sim_end_utc.replace(second=0, microsecond=0))
        self.assertEqual(self.simulator._one_min_bar_timeline[1] - self.simulator._one_min_bar_timeline[0], timedelta(minutes=1))

        # 5-Minute Bar Timeline
        self.assertNotEmpty(self.simulator._five_min_bar_timeline)
        start_5m = sim_start_utc.replace(minute=(sim_start_utc.minute // 5) * 5, second=0, microsecond=0)
        self.assertEqual(self.simulator._five_min_bar_timeline[0], start_5m)
        # End time for 5m can be tricky if POSTMARKET_END is not a multiple of 5min.
        # The loop goes `while current_time <= timeline_end`.
        self.assertTrue(self.simulator._five_min_bar_timeline[-1] <= sim_end_utc.replace(second=0, microsecond=0))
        self.assertEqual(self.simulator._five_min_bar_timeline[1] - self.simulator._five_min_bar_timeline[0], timedelta(minutes=5))

    def test_03_state_precomputation_at_4am_prime_with_prev_day_post_market(self):
        """
        Test that the state at 4:00 AM on session day is correctly primed
        using previous day's post-market data.
        """
        first_tick_utc = self.session_day_premarket_start_utc # 4:00:00 AM ET on session day
        state_at_4am = self.simulator.get_state_at_time(first_tick_utc) #

        self.assertIsNotNone(state_at_4am)
        self.assertEqual(state_at_4am['timestamp_utc'], first_tick_utc)

        # Price Priming: Should use previous day's actual close if no trades/quotes exactly at 4 AM
        # or the last known price from prev day's post-market from raw_trades/quotes if available
        # In our setup, _initialize_locf_values might pick up prev_day_close first.
        # Then _precompute_timeline_states iterates. If no trades/quotes for current_ts=4AM,
        # last_price from prev_day_close (112) or actual prev_day_post_market_close (115.88 if trades went up to there)
        # The test data for trades/quotes starts on session_day_premarket_start_utc + 5 seconds.
        # So at exactly 4:00:00, it should be based on prev_day_close or the last tick of prev day data used in LOCF init.
        # self.simulator.prev_day_close is 112 (from daily).
        # However, _initialize_locf_values can use head of raw_trades/quotes.
        # The loading range includes prev_day_postmarket_end_utc.
        # If raw_trades_df or raw_quotes_df has data near prev_day_postmarket_end_utc, it might be used.
        # Let's assume it uses the prev_day_close from daily bars if no closer trade/quote.
        # From _precompute_timeline_states:
        # last_price = prev_day_data.get('close') which is 112.0
        # then it's updated by _initialize_locf_values, which would look at self.raw_trades_df.
        # If raw_trades_df includes previous day's post market trades, it can update it.
        # Our self.sample_trades_df only has session day data.
        # self.prev_day_1m_bars_df has close of self.prev_day_post_market_close_price (115.88) at end of prev day.
        # The state precomputation's LOCF will carry this forward if there are no new trades/quotes at 4 AM.
        # The `current_1s_bar` will be synthetic using this `last_price`.

        # Check if `_initialize_locf_values` uses the latest from `raw_trades_df` or `raw_quotes_df`.
        # If these are empty for the previous day's tail, it falls back to `prev_day_close`.
        # The `_precompute_timeline_states` loop starts with `last_price = prev_day_data.get('close')`.
        # In our test setup, `sample_trades_df` and `sample_quotes_df` only start at `session_day_premarket_start_utc + 5s`.
        # So at exactly 4:00:00 AM, `last_price` should be `self.simulator.prev_day_close` (112.0).
        self.assertEqual(state_at_4am['current_price'], 112.0, "Price at 4AM should be prev day's daily close if no immediate data.")
        self.assertIsNone(state_at_4am['mid_price']) # No quotes exactly at 4:00:00

        # Test hf_data_window priming
        hf_window = state_at_4am['hf_data_window'] #
        self.assertEqual(len(hf_window), self.model_config.hf_seq_len * 2) #
        # This window looks back `hf_seq_len * 2` seconds. Since it's 4 AM,
        # it should be filled with `empty_hf_entry` or actual data if previous day's data was loaded
        # into raw_trades/quotes and falls into this lookback.
        # Given our `sample_trades_df` starts later, these should be mostly empty or synthetic.
        # The first entry might reflect the LOCF price.
        first_hf_entry_bar = hf_window[-1]['1s_bar'] # Most recent entry in the window
        self.assertIsNotNone(first_hf_entry_bar)
        self.assertTrue(first_hf_entry_bar['is_synthetic']) #
        self.assertEqual(first_hf_entry_bar['close'], 112.0)


        # Test 1m_bars_window priming
        mf_window = state_at_4am['1m_bars_window'] #
        self.assertEqual(len(mf_window), self.model_config.mf_seq_len * 2) #
        # This window looks back `mf_seq_len * 2` minutes.
        # It should contain actual bars from `self.prev_day_1m_bars_df`'s tail.
        last_bar_in_mf_window = mf_window[-1] # Most recent bar in window (should be for 3:59 AM prev day ET)
        expected_last_bar_ts = self.prev_day_postmarket_end_utc # This is 8PM ET prev day.
                                                              # The window ends AT or BEFORE current_ts (4AM).
                                                              # So the last bar is for the minute ending at 4AM,
                                                              # which is the 3:59 AM bar.
        # The bar for 4:00 AM on session_day_premarket_start_utc is part of the timeline.
        # _get_bars_window for current_ts=4:00:00 will fetch bars up to and including the bar whose timestamp is 4:00:00.
        # The mf_window_size is 60. So it should fetch 60 bars ending at 4:00:00 UTC of session day.
        # This means it will pull from self.prev_day_1m_bars_df and self.session_day_1m_bars_df.
        # The last bar in mf_window should be the one for self.session_day_premarket_start_utc
        self.assertEqual(last_bar_in_mf_window['timestamp'], self.session_day_premarket_start_utc)
        self.assertEqual(last_bar_in_mf_window['open'], self.first_session_day_premarket_open) # from session_day_1m_bars_df
        self.assertFalse(last_bar_in_mf_window.get('is_synthetic', False))

        # Check a bar from previous day is in the window
        a_prev_day_bar_ts = self.prev_day_postmarket_end_utc # 8PM ET prev day
        self.assertTrue(any(b['timestamp'] == a_prev_day_bar_ts for b in mf_window))
        prev_day_bar_in_window = next(b for b in mf_window if b['timestamp'] == a_prev_day_bar_ts)
        self.assertEqual(prev_day_bar_in_window['close'], self.prev_day_post_market_close_price)


    def test_04a_bar_integrity_1s_from_trades(self):
        """Test 1-second bar aggregation from trades."""
        # Time: 2025-03-27 04:00:10 ET (session_day_premarket_start_utc + 10s)
        # Trades at :10.100 (P:120.1, S:10) and :10.200 (P:120.2, S:5)
        target_ts_utc = self.session_day_premarket_start_utc + timedelta(seconds=10)
        state = self.simulator.get_state_at_time(target_ts_utc)
        bar_1s = state['current_1s_bar'] #

        self.assertIsNotNone(bar_1s)
        self.assertFalse(bar_1s['is_synthetic'])
        self.assertEqual(bar_1s['timestamp'], target_ts_utc)
        self.assertEqual(bar_1s['open'], 120.1)
        self.assertEqual(bar_1s['high'], 120.2)
        self.assertEqual(bar_1s['low'], 120.1)
        self.assertEqual(bar_1s['close'], 120.2)
        self.assertEqual(bar_1s['volume'], 15) # 10 + 5

    def test_04b_bar_integrity_1s_synthetic(self):
        """Test synthetic 1-second bar creation when no trades."""
        # Time: 2025-03-27 04:00:05 ET (session_day_premarket_start_utc + 5s)
        # Quote at :05, but no trade. Prev trade was before 4AM. LOCF from prev day close (112.0)
        # until first quote/trade of the day updates LOCF.
        # At 4:00:05, a quote B:119.9, A:120.1 arrives. This updates last_bid, last_ask.
        # last_price is still 112.0 as no trade yet for the session.
        target_ts_utc = self.session_day_premarket_start_utc + timedelta(seconds=5)
        state = self.simulator.get_state_at_time(target_ts_utc)
        bar_1s = state['current_1s_bar']

        self.assertIsNotNone(bar_1s)
        self.assertTrue(bar_1s['is_synthetic']) #
        self.assertEqual(bar_1s['timestamp'], target_ts_utc)
        # Price should be from LOCF.
        # Initial last_price is prev_day_close = 112.0. No trades in sample data before 4:00:10.
        self.assertEqual(bar_1s['open'], 112.0)
        self.assertEqual(bar_1s['high'], 112.0)
        self.assertEqual(bar_1s['low'], 112.0)
        self.assertEqual(bar_1s['close'], 112.0)
        self.assertEqual(bar_1s['volume'], 0.0)

    def test_04c_bar_integrity_1m_actual(self):
        """Test retrieval of actual 1-minute bars."""
        # Time: 2025-03-27 04:01:00 ET (session_day_premarket_start_utc + 1min)
        target_ts_utc = self.session_day_premarket_start_utc + timedelta(minutes=1)
        state = self.simulator.get_state_at_time(target_ts_utc)
        mf_window = state['1m_bars_window']
        bar_1m = mf_window[-1] # The bar for 04:01:00 is the one whose data is from 04:01:00 to 04:01:59

        self.assertIsNotNone(bar_1m)
        # The bar with timestamp 04:01:00 ET is the one we want
        self.assertEqual(bar_1m['timestamp'], target_ts_utc)
        self.assertFalse(bar_1m.get('is_synthetic', False))
        expected_bar_data = self.session_day_1m_bars_df.loc[target_ts_utc]
        self.assertEqual(bar_1m['open'], expected_bar_data['open'])
        self.assertEqual(bar_1m['high'], expected_bar_data['high'])
        self.assertEqual(bar_1m['low'], expected_bar_data['low'])
        self.assertEqual(bar_1m['close'], expected_bar_data['close'])
        self.assertEqual(bar_1m['volume'], expected_bar_data['volume'])

    def test_04d_bar_integrity_5m_from_1m_aggregation(self):
        """Test 5-minute bars created from 1-minute aggregation if direct 5m files were missing."""
        # This relies on the mock_get_bars for 5m to perform resampling if direct files not found.
        # For MarketSimulator, it gets data from _create_complete_bar_dictionaries.
        # Let's pick a time that ensures we have a full 5-min bar.
        # Example: bar for 4:05:00 ET (covers 4:05:00 to 4:09:59)
        target_bar_start_utc = self.session_day_premarket_start_utc + timedelta(minutes=5) # Bar for 4:05 AM
        state_ts_utc = target_bar_start_utc + timedelta(seconds=30) # Get state within that bar's formation
        state = self.simulator.get_state_at_time(state_ts_utc)
        lf_window = state['5m_bars_window'] #

        # Find the 4:05:00 bar in the window. The window ends at or before state_ts_utc.
        # The bar corresponding to state_ts_utc (4:05:30) is the 4:05:00 bar.
        bar_5m = lf_window[-1]
        self.assertEqual(bar_5m['timestamp'], target_bar_start_utc)

        # Manually aggregate the expected 5m bar from self.session_day_1m_bars_df
        bars_for_5m_agg = self.session_day_1m_bars_df[
            (self.session_day_1m_bars_df.index >= target_bar_start_utc) &
            (self.session_day_1m_bars_df.index < target_bar_start_utc + timedelta(minutes=5))
        ]
        self.assertFalse(bars_for_5m_agg.empty, "Should have 1m bars for aggregation")

        expected_open = bars_for_5m_agg['open'].iloc[0]
        expected_high = bars_for_5m_agg['high'].max()
        expected_low = bars_for_5m_agg['low'].min()
        expected_close = bars_for_5m_agg['close'].iloc[-1]
        expected_volume = bars_for_5m_agg['volume'].sum()

        self.assertFalse(bar_5m.get('is_synthetic', False))
        self.assertAlmostEqual(bar_5m['open'], expected_open)
        self.assertAlmostEqual(bar_5m['high'], expected_high)
        self.assertAlmostEqual(bar_5m['low'], expected_low)
        self.assertAlmostEqual(bar_5m['close'], expected_close)
        self.assertAlmostEqual(bar_5m['volume'], expected_volume)


    def test_05_price_propagation_and_locf(self):
        """Test price updates (current, bid, ask, mid) and LOCF."""
        # Time 1: 4:00:05 ET - Quote B:119.9, A:120.1. No trade yet. Prev day close 112.
        ts1 = self.session_day_premarket_start_utc + timedelta(seconds=5)
        state1 = self.simulator.get_state_at_time(ts1)
        self.assertEqual(state1['current_price'], 112.0) # LOCF from prev day daily close
        self.assertEqual(state1['best_bid_price'], 119.9)
        self.assertEqual(state1['best_ask_price'], 120.1)
        self.assertAlmostEqual(state1['mid_price'], 120.0)

        # Time 2: 4:00:09 ET - No new data after ts1. LOCF from ts1.
        ts2 = self.session_day_premarket_start_utc + timedelta(seconds=9)
        state2 = self.simulator.get_state_at_time(ts2)
        self.assertEqual(state2['current_price'], 112.0) # Still LOCF from prev day close
        self.assertEqual(state2['best_bid_price'], 119.9) # LOCF from ts1 quote
        self.assertEqual(state2['best_ask_price'], 120.1) # LOCF from ts1 quote
        self.assertAlmostEqual(state2['mid_price'], 120.0) # LOCF from ts1 quote

        # Time 3: 4:00:10 ET - Trades P:120.1, P:120.2. No new quote.
        ts3 = self.session_day_premarket_start_utc + timedelta(seconds=10)
        state3 = self.simulator.get_state_at_time(ts3)
        self.assertEqual(state3['current_price'], 120.2) # Updated by trade
        self.assertEqual(state3['best_bid_price'], 119.9) # LOCF from ts1 quote
        self.assertEqual(state3['best_ask_price'], 120.1) # LOCF from ts1 quote
        self.assertAlmostEqual(state3['mid_price'], 120.0)

        # Time 4: 4:01:00 ET - New Quote B:120.15, A:120.25. Trade at 4:01:05 (P:120.3)
        ts4_quote = self.session_day_premarket_start_utc + timedelta(minutes=1)
        state4_quote = self.simulator.get_state_at_time(ts4_quote)
        self.assertEqual(state4_quote['current_price'], 120.2) # LOCF from ts3 trade
        self.assertEqual(state4_quote['best_bid_price'], 120.15) # New quote
        self.assertEqual(state4_quote['best_ask_price'], 120.25) # New quote
        self.assertAlmostEqual(state4_quote['mid_price'], 120.20)

        ts4_trade = self.session_day_premarket_start_utc + timedelta(minutes=1, seconds=5)
        state4_trade = self.simulator.get_state_at_time(ts4_trade)
        self.assertEqual(state4_trade['current_price'], 120.3) # New trade
        self.assertEqual(state4_trade['best_bid_price'], 120.15) # LOCF from ts4_quote
        self.assertEqual(state4_trade['best_ask_price'], 120.25) # LOCF from ts4_quote

    def test_06_market_session_determination(self):
        """Test correct market session identification."""
        # Premarket
        ts_pre = self._create_sample_datetime(SESSION_DATE_STR, time(4, 30, 0)).astimezone(UTC_TZ)
        self.assertEqual(self.simulator._determine_market_session(ts_pre), "PREMARKET") #

        # Regular
        ts_reg = self._create_sample_datetime(SESSION_DATE_STR, time(10, 0, 0)).astimezone(UTC_TZ)
        self.assertEqual(self.simulator._determine_market_session(ts_reg), "REGULAR") #

        # Postmarket
        ts_post = self._create_sample_datetime(SESSION_DATE_STR, time(16, 30, 0)).astimezone(UTC_TZ)
        self.assertEqual(self.simulator._determine_market_session(ts_post), "POSTMARKET") #

        # Closed (before premarket)
        ts_closed_early = self._create_sample_datetime(SESSION_DATE_STR, time(3, 0, 0)).astimezone(UTC_TZ)
        self.assertEqual(self.simulator._determine_market_session(ts_closed_early), "CLOSED") #

        # Closed (after postmarket)
        ts_closed_late = self._create_sample_datetime(SESSION_DATE_STR, time(20, 30, 0)).astimezone(UTC_TZ)
        self.assertEqual(self.simulator._determine_market_session(ts_closed_late), "CLOSED") #

    def test_07_reset_and_step(self):
        """Test reset and step functionalities."""
        initial_state = self.simulator.reset() #
        self.assertIsNotNone(initial_state)
        initial_idx = self.simulator._current_agent_time_idx
        initial_ts = self.simulator.current_timestamp_utc

        # Default reset goes to an offset (min_offset)
        min_offset = max(self.model_config.hf_seq_len * 2, self.model_config.mf_seq_len*2, self.model_config.lf_seq_len*2)
        self.assertEqual(initial_idx, min_offset) # Default reset behavior might change based on window sizes
        self.assertEqual(initial_ts, self.simulator._agent_timeline_utc[min_offset])

        can_step = self.simulator.step() #
        self.assertTrue(can_step)
        self.assertEqual(self.simulator._current_agent_time_idx, initial_idx + 1)
        self.assertEqual(self.simulator.current_timestamp_utc, self.simulator._agent_timeline_utc[initial_idx + 1])
        self.assertNotEqual(self.simulator.get_current_market_state()['timestamp_utc'], initial_ts)

        # Step to the end
        while not self.simulator.is_done(): #
            self.assertTrue(self.simulator.step())

        self.assertTrue(self.simulator.is_done())
        self.assertFalse(self.simulator.step()) # Cannot step further
        self.assertEqual(self.simulator._current_agent_time_idx, len(self.simulator._agent_timeline_utc) - 1)

        # Test reset with random start
        self.simulator.reset(options={'random_start': True}) #
        random_idx = self.simulator._current_agent_time_idx
        self.assertTrue(random_idx >= min_offset)
        self.assertTrue(random_idx < len(self.simulator._agent_timeline_utc))


    def test_08_edge_case_no_previous_day_in_daily_data(self):
        """Test behavior when no previous trading day can be found in daily bars."""
        # Mock get_bars to return empty daily data for prev day lookup
        original_get_bars = self.mock_data_manager.get_bars
        def no_daily_bars_for_prev_lookup(symbol, timeframe, start_time, end_time):
            if timeframe == "1d" and end_time.date() == self.session_date : # The call to find prev day
                 # This simulates loading daily data up to session_date - 1 year, but it's empty or doesn't contain prev_day_date
                empty_daily_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                empty_daily_df.index = pd.DatetimeIndex([], name='timestamp', tz='UTC')
                return empty_daily_df
            return original_get_bars(symbol, timeframe, start_time, end_time)

        self.mock_data_manager.get_bars.side_effect = no_daily_bars_for_prev_lookup

        simulator_no_prev = MarketSimulator(
            symbol=TEST_SYMBOL, data_manager=self.mock_data_manager,
            market_config=self.market_config, model_config=self.model_config,
            start_time=SESSION_START_ET_STR, end_time=SESSION_END_ET_STR, logger=MagicMock()
        )

        self.assertIsNone(simulator_no_prev.prev_day_date) # _find_previous_valid_trading_day returns None
        prev_data = simulator_no_prev.get_previous_day_data()
        self.assertEqual(prev_data, {}) #

        # Simulator should still initialize and precompute states, using None for prev_day_close
        first_tick_utc = self._create_sample_datetime(SESSION_DATE_STR, MARKET_HOURS["PREMARKET_START"]).astimezone(UTC_TZ)
        state_at_4am = simulator_no_prev.get_state_at_time(first_tick_utc)
        self.assertIsNotNone(state_at_4am)
        self.assertIsNone(state_at_4am['previous_day_close']) #
        # Initial price might be NaN or based on first available trade/quote of the session day
        # _precompute_timeline_states: last_price, last_bid, last_ask are None initially
        # _initialize_locf_values will try to set them from raw_trades_df.head(10)
        # Our sample_trades_df starts at 4:00:10 ET with price 120.1
        # So for 4:00:00, current_price should still be None or what _initialize_locf_values set it to
        # if it could read ahead.
        # Let's check a slightly later time after first trade
        state_after_first_trade = simulator_no_prev.get_state_at_time(self.sample_trades_df.index[0])
        self.assertEqual(state_after_first_trade['current_price'], self.sample_trades_df['price'].iloc[0])

        self.mock_data_manager.get_bars.side_effect = original_get_bars # Restore mock

    def test_09_ensure_utc_datetime_parsing(self):
        """Test the _ensure_utc_datetime utility for various inputs."""
        # String, no timezone (assumed market_tz)
        dt_str_naive = f"{SESSION_DATE_STR} 10:00:00"
        expected_dt_utc = datetime.combine(self.session_date, time(10,0,0), tzinfo=MARKET_TZ).astimezone(UTC_TZ)
        self.assertEqual(self.simulator._ensure_utc_datetime(dt_str_naive), expected_dt_utc) #

        # String, with timezone ET
        dt_str_et = f"{SESSION_DATE_STR} 10:00:00 America/New_York" # Valid IANA for ET
        self.assertEqual(self.simulator._ensure_utc_datetime(dt_str_et), expected_dt_utc)

        # String, with UTC timezone
        dt_str_utc = expected_dt_utc.isoformat()
        self.assertEqual(self.simulator._ensure_utc_datetime(dt_str_utc), expected_dt_utc)

        # Datetime, naive
        dt_naive = datetime.combine(self.session_date, time(10,0,0))
        self.assertEqual(self.simulator._ensure_utc_datetime(dt_naive), expected_dt_utc) #

        # Datetime, with market_tz
        dt_market = datetime.combine(self.session_date, time(10,0,0), tzinfo=MARKET_TZ)
        self.assertEqual(self.simulator._ensure_utc_datetime(dt_market), expected_dt_utc)

        # Datetime, with UTC
        dt_utc = expected_dt_utc
        self.assertEqual(self.simulator._ensure_utc_datetime(dt_utc), expected_dt_utc) #

        # None input
        self.assertIsNone(self.simulator._ensure_utc_datetime(None)) #

    def test_10_intraday_high_low_tracking(self):
        """Test tracking of intraday high and low prices."""
        # State at 4:00:10 (after trades 120.1, 120.2)
        ts1 = self.session_day_premarket_start_utc + timedelta(seconds=10)
        state1 = self.simulator.get_state_at_time(ts1)
        self.assertEqual(state1['intraday_high'], 120.2) #
        self.assertEqual(state1['intraday_low'], 120.1)  #

        # State at 4:01:05 (after trade 120.3)
        ts2 = self.session_day_premarket_start_utc + timedelta(minutes=1, seconds=5)
        state2 = self.simulator.get_state_at_time(ts2)
        self.assertEqual(state2['intraday_high'], 120.3)
        self.assertEqual(state2['intraday_low'], 120.1)

        # State at 9:30:30 ET (after trade 122.0)
        ts3 = self.session_day_regular_start_utc + timedelta(seconds=30)
        state3 = self.simulator.get_state_at_time(ts3)
        self.assertEqual(state3['intraday_high'], 122.0)
        self.assertEqual(state3['intraday_low'], 120.1) # Low from premarket still holds

        # Test a state before any trades on the session day (e.g. 4:00:00)
        ts0 = self.session_day_premarket_start_utc
        state0 = self.simulator.get_state_at_time(ts0)
        # Intraday H/L are None until first bar/trade of the day
        self.assertIsNone(state0['intraday_high'])
        self.assertIsNone(state0['intraday_low'])

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.simulator, 'close'):
            self.simulator.close() #

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)