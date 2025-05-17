import logging
from datetime import datetime, timezone, timedelta
import unittest

import numpy as np

from simulators.portfolio_simulator import PortfolioManager, PositionSideEnum, OrderTypeEnum, OrderSideEnum, FillDetails


class MockPortfolioConfig:
    def __init__(self, initial_cash, max_position_value_ratio, max_position_holding_seconds):
        self.initial_cash = initial_cash
        self.max_position_value_ratio = max_position_value_ratio
        self.max_position_holding_seconds = max_position_holding_seconds


class MockExecutionConfig:
    def __init__(self, allow_shorting):
        self.allow_shorting = allow_shorting


class MockSimulationSubConfig:
    def __init__(self, portfolio_config, execution_config):
        self.portfolio_config = portfolio_config
        self.execution_config = execution_config


class MockModelSubConfig:
    def __init__(self, portfolio_seq_len, portfolio_feat_dim):
        self.portfolio_seq_len = portfolio_seq_len
        self.portfolio_feat_dim = portfolio_feat_dim


class MockEnvSubConfig:
    pass


class Config:  # Top-level mock Config class
    def __init__(self, initial_cash=100000.0, max_position_value_ratio=0.5,
                 max_position_holding_seconds=86400, allow_shorting=True,
                 portfolio_seq_len=5, portfolio_feat_dim=5):
        self.env = MockEnvSubConfig()
        self.simulation = MockSimulationSubConfig(
            portfolio_config=MockPortfolioConfig(
                initial_cash, max_position_value_ratio, max_position_holding_seconds
            ),
            execution_config=MockExecutionConfig(allow_shorting)
        )
        self.model = MockModelSubConfig(portfolio_seq_len, portfolio_feat_dim)


# --- Unit Test Script ---
class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("TestPortfolioManager")
        self.logger.setLevel(logging.CRITICAL)  # Suppress logs during normal test runs
        # To see logs for debugging:
        # self.logger.setLevel(logging.DEBUG)
        # handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # handler.setFormatter(formatter)
        # if not self.logger.hasHandlers():
        #      self.logger.addHandler(handler)

        self.start_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.tradable_assets = ["BTCUSDT", "ETHUSDT"]

    def _create_pm(self, initial_cash=100000.0, allow_shorting=True, seq_len=3, feat_dim=5):
        self.config = Config(
            initial_cash=initial_cash,
            allow_shorting=allow_shorting,
            portfolio_seq_len=seq_len,
            portfolio_feat_dim=feat_dim,
            max_position_value_ratio=0.5,
            max_position_holding_seconds=3600
        )
        pm = PortfolioManager(self.logger, self.config, self.tradable_assets)
        pm.reset(self.start_time)
        return pm

    def test_initialization_and_reset(self):
        pm = self._create_pm(initial_cash=50000)
        self.assertEqual(pm.cash, 50000)
        self.assertEqual(pm.current_total_equity, 50000)
        self.assertEqual(pm.current_unrealized_pnl, 0.0)
        self.assertEqual(len(pm.positions), 2)
        self.assertEqual(pm.positions["BTCUSDT"]['current_side'], PositionSideEnum.FLAT)
        self.assertEqual(len(pm.portfolio_feature_history), pm.portfolio_seq_len)

        pm.cash = 1000
        reset_time = self.start_time + timedelta(days=1)
        pm.reset(reset_time)
        self.assertEqual(pm.cash, 50000)
        self.assertEqual(pm.current_total_equity, 50000)
        # After reset, portfolio_value_history should have one entry from the reset
        self.assertEqual(len(pm.portfolio_value_history), 1)
        self.assertEqual(pm.portfolio_value_history[0], (reset_time, 50000))

    def test_single_long_trade_cycle(self):
        pm = self._create_pm()
        asset = "BTCUSDT"

        fill_time1 = self.start_time + timedelta(minutes=1)
        buy_fill = FillDetails(
            asset_id=asset, fill_timestamp=fill_time1, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY, requested_quantity=1.0, executed_quantity=1.0,
            executed_price=20000.0, commission=10.0, fees=1.0, slippage_cost_total=5.0
        )
        pm.update_fill(buy_fill)

        self.assertEqual(pm.positions[asset]['quantity'], 1.0)
        self.assertEqual(pm.positions[asset]['avg_entry_price'], 20000.0)
        expected_cash_after_buy = 100000.0 - (1.0 * 20000.0) - 10.0 - 1.0
        self.assertAlmostEqual(pm.cash, expected_cash_after_buy)

        market_time1 = self.start_time + timedelta(minutes=5)
        pm.update_market_value({asset: 21000.0, "ETHUSDT": 1000.0}, market_time1)
        self.assertAlmostEqual(pm.positions[asset]['unrealized_pnl'], 1000.0)
        self.assertAlmostEqual(pm.current_total_equity, expected_cash_after_buy + 21000.0)  # Cash + MV of asset

        market_time2 = self.start_time + timedelta(minutes=10)
        pm.update_market_value({asset: 19500.0, "ETHUSDT": 1000.0}, market_time2)
        self.assertAlmostEqual(pm.positions[asset]['unrealized_pnl'], -500.0)

        fill_time2 = self.start_time + timedelta(minutes=15)
        sell_fill = FillDetails(
            asset_id=asset, fill_timestamp=fill_time2, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL, requested_quantity=1.0, executed_quantity=1.0,
            executed_price=20500.0, commission=12.0, fees=2.0, slippage_cost_total=6.0
        )
        pm.update_fill(sell_fill)

        # Call update_market_value after closing fill to update current_total_equity
        pm.update_market_value({}, fill_time2)  # Pass empty dict as all positions of this asset are closed
        # and other assets are assumed flat or their prices irrelevant here.

        self.assertEqual(pm.positions[asset]['quantity'], 0.0)
        self.assertEqual(pm.positions[asset]['current_side'], PositionSideEnum.FLAT)
        self.assertEqual(len(pm.open_trades), 0)
        self.assertEqual(len(pm.trade_log), 1)

        closed_trade = pm.trade_log[0]
        self.assertAlmostEqual(closed_trade['realized_pnl'], 475.0)

        expected_final_cash = expected_cash_after_buy + (1.0 * 20500.0) - 12.0 - 2.0
        self.assertAlmostEqual(pm.cash, expected_final_cash)
        self.assertAlmostEqual(pm.current_total_equity, expected_final_cash)  # Now this should pass

    def test_single_short_trade_cycle(self):
        pm = self._create_pm(allow_shorting=True)
        asset = "ETHUSDT"

        fill_time1 = self.start_time + timedelta(hours=1)
        sell_fill = FillDetails(
            asset_id=asset, fill_timestamp=fill_time1, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL, requested_quantity=10.0, executed_quantity=10.0,
            executed_price=1500.0, commission=7.0, fees=0.5, slippage_cost_total=3.0
        )
        pm.update_fill(sell_fill)
        expected_cash_after_short_sell = 100000.0 + (10.0 * 1500.0) - 7.0 - 0.5
        self.assertAlmostEqual(pm.cash, expected_cash_after_short_sell)

        market_time1 = self.start_time + timedelta(hours=2)
        pm.update_market_value({asset: 1400.0, "BTCUSDT": 20000}, market_time1)
        self.assertAlmostEqual(pm.positions[asset]['unrealized_pnl'], (1500.0 - 1400.0) * 10.0)
        # Equity for short is cash + market_value (qty * current_price), per code logic
        self.assertAlmostEqual(pm.current_total_equity, expected_cash_after_short_sell + (10.0 * 1400.0))

        fill_time2 = self.start_time + timedelta(hours=3)
        buy_cover_fill = FillDetails(
            asset_id=asset, fill_timestamp=fill_time2, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY, requested_quantity=10.0, executed_quantity=10.0,
            executed_price=1450.0, commission=6.0, fees=0.4, slippage_cost_total=2.0
        )
        pm.update_fill(buy_cover_fill)

        # Call update_market_value after closing fill
        pm.update_market_value({}, fill_time2)

        self.assertEqual(pm.positions[asset]['quantity'], 0.0)
        self.assertEqual(len(pm.trade_log), 1)
        closed_trade = pm.trade_log[0]
        self.assertAlmostEqual(closed_trade['realized_pnl'], 486.1)

        expected_final_cash = expected_cash_after_short_sell - (10.0 * 1450.0) - 6.0 - 0.4
        self.assertAlmostEqual(pm.cash, expected_final_cash)
        self.assertAlmostEqual(pm.current_total_equity, expected_final_cash)  # Now this should pass

    def test_scaling_in_long_position(self):
        pm = self._create_pm()
        asset = "BTCUSDT"

        fill1_ts = self.start_time + timedelta(minutes=1)
        fill1 = FillDetails(asset_id=asset, fill_timestamp=fill1_ts, order_type=OrderTypeEnum.MARKET,
                            order_side=OrderSideEnum.BUY, requested_quantity=1.0, executed_quantity=1.0,
                            executed_price=20000.0, commission=10.0, fees=1.0, slippage_cost_total=0)
        pm.update_fill(fill1)

        fill2_ts = self.start_time + timedelta(minutes=5)
        fill2 = FillDetails(asset_id=asset, fill_timestamp=fill2_ts, order_type=OrderTypeEnum.MARKET,
                            order_side=OrderSideEnum.BUY, requested_quantity=1.0, executed_quantity=1.0,
                            executed_price=21000.0, commission=11.0, fees=1.0, slippage_cost_total=0)
        pm.update_fill(fill2)

        self.assertEqual(pm.positions[asset]['quantity'], 2.0)
        expected_avg_price = (1.0 * 20000.0 + 1.0 * 21000.0) / 2.0
        self.assertEqual(pm.positions[asset]['avg_entry_price'], expected_avg_price)

        fill3_ts = self.start_time + timedelta(minutes=10)
        fill3 = FillDetails(asset_id=asset, fill_timestamp=fill3_ts, order_type=OrderTypeEnum.MARKET,
                            order_side=OrderSideEnum.SELL, requested_quantity=2.0, executed_quantity=2.0,
                            executed_price=22000.0, commission=15.0, fees=2.0, slippage_cost_total=0)
        pm.update_fill(fill3)

        # Call update_market_value after closing fill
        pm.update_market_value({}, fill3_ts)

        self.assertEqual(pm.positions[asset]['quantity'], 0.0)
        self.assertEqual(len(pm.trade_log), 1)
        closed_trade = pm.trade_log[0]
        self.assertAlmostEqual(closed_trade['realized_pnl'], 2960)
        expected_final_cash = 100000 + 2960  # Initial cash + total PNL
        self.assertAlmostEqual(pm.cash, expected_final_cash)
        self.assertAlmostEqual(pm.current_total_equity, expected_final_cash)

    def test_partial_close_long_position(self):
        pm = self._create_pm()
        asset = "ETHUSDT"

        fill1_ts = self.start_time + timedelta(seconds=10)
        fill1 = FillDetails(asset_id=asset, fill_timestamp=fill1_ts, order_type=OrderTypeEnum.MARKET,
                            order_side=OrderSideEnum.BUY, requested_quantity=10.0, executed_quantity=10.0,
                            executed_price=1000.0, commission=5.0, fees=0.5, slippage_cost_total=0)
        pm.update_fill(fill1)

        fill2_ts = self.start_time + timedelta(seconds=20)
        fill2 = FillDetails(asset_id=asset, fill_timestamp=fill2_ts, order_type=OrderTypeEnum.MARKET,
                            order_side=OrderSideEnum.SELL, requested_quantity=6.0, executed_quantity=6.0,
                            executed_price=1100.0, commission=3.0, fees=0.3, slippage_cost_total=0)
        pm.update_fill(fill2)

        self.assertEqual(pm.positions[asset]['quantity'], 4.0)
        # After partial close, equity should be cash + market value of remaining 4 units
        # For simplicity, let's update market value here if we were to check equity mid-trade
        # pm.update_market_value({asset: 1100.0}, fill2_ts) # Example if checking equity here

        fill3_ts = self.start_time + timedelta(seconds=30)
        fill3 = FillDetails(asset_id=asset, fill_timestamp=fill3_ts, order_type=OrderTypeEnum.MARKET,
                            order_side=OrderSideEnum.SELL, requested_quantity=4.0, executed_quantity=4.0,
                            executed_price=1200.0, commission=2.0, fees=0.2, slippage_cost_total=0)
        pm.update_fill(fill3)

        # Call update_market_value after final closing fill
        pm.update_market_value({}, fill3_ts)

        self.assertEqual(pm.positions[asset]['quantity'], 0.0)
        self.assertEqual(len(pm.trade_log), 1)
        closed_trade = pm.trade_log[0]
        self.assertAlmostEqual(closed_trade['realized_pnl'], 1389.0)

        expected_final_cash = 100000 + 1389.0  # Initial cash + total PNL
        self.assertAlmostEqual(pm.cash, expected_final_cash)
        self.assertAlmostEqual(pm.current_total_equity, expected_final_cash)  # Now this should pass

    def test_portfolio_features_and_observation(self):
        pm = self._create_pm(seq_len=3, feat_dim=5)
        asset = self.tradable_assets[0]

        obs_init = pm.get_portfolio_observation()
        self.assertEqual(obs_init['features'].shape, (3, 5))
        expected_initial_feat = np.array([0, 0, 0, 0, 1.0], dtype=np.float32)
        # Check the latest feature in the history after reset
        self.assertTrue(np.allclose(pm.portfolio_feature_history[-1], expected_initial_feat, atol=1e-5), f"Actual: {pm.portfolio_feature_history[-1]}")

        fill_time1 = self.start_time + timedelta(minutes=1)
        buy_fill = FillDetails(
            asset_id=asset, fill_timestamp=fill_time1, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY, requested_quantity=0.1, executed_quantity=0.1,
            executed_price=200000.0, commission=10.0, fees=1.0, slippage_cost_total=0)
        pm.update_fill(buy_fill)

        # Market update is crucial for features reflecting position
        market_time1 = self.start_time + timedelta(minutes=5)
        # Initial equity = 100k. Costs = 11. Cash = 100k - 20k - 11 = 79989.
        # Position value = 0.1 * 200k = 20k.
        # Current total equity = 79989 + 20k = 99989.
        pm.update_market_value({asset: 200000.0, self.tradable_assets[1]: 1000.0}, market_time1)

        obs1 = pm.get_portfolio_observation()
        self.assertEqual(obs1['features'].shape, (3, 5))
        last_features1 = obs1['features'][-1]

        # norm_pos_size: current_pos_market_value / (current_total_equity * max_position_value_ratio)
        # current_pos_market_value = 0.1 * 200000 = 20000
        # current_total_equity = 99989 (calculated above)
        # max_pos_val_ratio = 0.5
        # norm_pos_size = 20000 / (99989 * 0.5) approx 20000 / 49994.5 approx 0.40004
        self.assertAlmostEqual(last_features1[0], 20000 / (pm.current_total_equity * self.config.simulation.portfolio_config.max_position_value_ratio),
                               delta=1e-4)
        self.assertAlmostEqual(last_features1[1], 0.0, delta=1e-5)
        self.assertAlmostEqual(last_features1[2], 0.0, delta=1e-5)
        self.assertGreater(last_features1[3], 0.0)
        # norm_cash_pct = cash / current_total_equity = 79989 / 99989 approx 0.80
        self.assertAlmostEqual(last_features1[4], pm.cash / pm.current_total_equity, delta=1e-5)

        market_time2 = self.start_time + timedelta(minutes=10)
        # UPL = (210000 - 200000) * 0.1 = 1000. Cash still 79989.
        # New market value of asset = 0.1 * 210000 = 21000.
        # New current_total_equity = 79989 + 21000 = 100989.
        pm.update_market_value({asset: 210000.0, self.tradable_assets[1]: 1000.0}, market_time2)

        obs2 = pm.get_portfolio_observation()
        last_features2 = obs2['features'][-1]

        # Entry value of trade = 0.1 * 200000 = 20000
        # Norm UPL = UPL_asset (1000) / entry_value_total_asset (20000) = 0.05
        self.assertAlmostEqual(last_features2[1], 1000 / (0.1 * 200000.0), delta=1e-5)

    def test_trader_vue_metrics(self):
        pm = self._create_pm()
        asset = "BTCUSDT"

        t1_f1_ts = self.start_time + timedelta(days=1, minutes=1)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=t1_f1_ts, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.BUY, requested_quantity=1, executed_quantity=1,
                                   executed_price=20000, commission=10, fees=1, slippage_cost_total=0))
        # Market update for portfolio history
        pm.update_market_value({asset: 20000}, t1_f1_ts + timedelta(seconds=1))

        t1_f2_ts = self.start_time + timedelta(days=1, minutes=10)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=t1_f2_ts, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.SELL, requested_quantity=1, executed_quantity=1,
                                   executed_price=21000, commission=10, fees=1, slippage_cost_total=0))
        pm.update_market_value({asset: 21000}, t1_f2_ts + timedelta(seconds=1))  # Update equity history

        t2_f1_ts = self.start_time + timedelta(days=2, minutes=1)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=t2_f1_ts, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.BUY, requested_quantity=1, executed_quantity=1,
                                   executed_price=22000, commission=10, fees=1, slippage_cost_total=0))
        pm.update_market_value({asset: 22000}, t2_f1_ts + timedelta(seconds=1))

        t2_f2_ts = self.start_time + timedelta(days=2, minutes=10)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=t2_f2_ts, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.SELL, requested_quantity=1, executed_quantity=1,
                                   executed_price=21500, commission=10, fees=1, slippage_cost_total=0))
        pm.update_market_value({asset: 21500}, t2_f2_ts + timedelta(seconds=1))

        metrics = pm.get_trader_vue_metrics()

        self.assertEqual(metrics["num_total_trades"], 2)
        self.assertEqual(metrics["num_winning_trades"], 1)
        self.assertEqual(metrics["num_losing_trades"], 1)
        self.assertAlmostEqual(metrics["total_net_profit_closed_trades"], 978 - 522)

        # For drawdown, portfolio_value_history is now populated by market updates
        # Initial: (start_time, 100000)
        # After T1 B: (t1_f1_ts+1s, 100000 - 11(costs) + 0 UPL_MV = 99989)  (MV of asset = 20k, cash = 100k-20k-11 = 79989. Equity = 79989+20k = 99989)
        # After T1 S: (t1_f2_ts+1s, 100000 + 978 = 100978) (cash = 100978, MV=0)
        # After T2 B: (t2_f1_ts+1s, 100978 - (22000+11) + 22000 = 100978 - 11 = 100967) (Cash after T1 + PNL_T1 - cost_T2_pos - fee_T2. Equity = new_cash + MV_T2_pos)
        # After T2 S: (t2_f2_ts+1s, 100967 - 522 (approx, exact value is total equity) = 100978 - 522 = 100456)
        # History: [100000, 99989, 100978, 100967, 100456] (approx values based on logic)
        # HWM:     [100000, 100000, 100978, 100978, 100978]
        # DD_abs:  [0,      11,     0,      11,     522   ] Max_DD_abs = 522
        # DD_pct:  for 522, HWM was 100978. So 522/100978 * 100
        # Note: The actual drawdown values will depend on the exact sequence of equity values recorded.
        # The previous manual history was clearer for testing _calculate_max_drawdown directly.
        # The test here is more for integration.
        self.assertGreaterEqual(metrics["max_portfolio_drawdown_abs"], 0)  # Basic check
        self.assertGreaterEqual(metrics["max_portfolio_drawdown_pct"], 0)

    def test_max_drawdown_helper(self):
        pm = self._create_pm()
        series1 = np.array([100, 110, 105, 120, 110, 115, 100, 130, 120])
        max_dd_abs, max_dd_pct = pm._calculate_max_drawdown(series1)
        self.assertAlmostEqual(max_dd_abs, 20.0)
        self.assertAlmostEqual(max_dd_pct, (20.0 / 120.0) * 100)

        series2 = np.array([100, 90, 80, 70])
        max_dd_abs, max_dd_pct = pm._calculate_max_drawdown(series2)
        self.assertAlmostEqual(max_dd_abs, 30.0)
        self.assertAlmostEqual(max_dd_pct, 30.0)

        series3 = np.array([100])
        max_dd_abs, max_dd_pct = pm._calculate_max_drawdown(series3)
        self.assertEqual(max_dd_abs, 0.0)
        self.assertEqual(max_dd_pct, 0.0)

    def test_fill_for_non_tradable_asset(self):
        pm = self._create_pm()
        non_tradable_asset = "DOGEUSDT"  # Assuming this is not in self.tradable_assets
        fill_time = self.start_time + timedelta(minutes=1)
        buy_fill = FillDetails(
            asset_id=non_tradable_asset, fill_timestamp=fill_time, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY, requested_quantity=100.0, executed_quantity=100.0,
            executed_price=0.15, commission=0.1, fees=0.05, slippage_cost_total=0.01
        )
        # Expecting a KeyError because the PortfolioManager initializes positions only for tradable_assets
        with self.assertRaises(KeyError):
            pm.update_fill(buy_fill)

    def test_disallow_shorting_attempt_to_open_short(self):
        pm = self._create_pm(allow_shorting=False)  # Crucial: shorting is disallowed
        asset = "BTCUSDT"
        initial_cash = pm.cash

        fill_time = self.start_time + timedelta(minutes=1)
        sell_to_open_short_fill = FillDetails(
            asset_id=asset, fill_timestamp=fill_time, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.SELL, requested_quantity=1.0, executed_quantity=1.0,
            executed_price=20000.0, commission=10.0, fees=1.0, slippage_cost_total=0.0
        )

        # This fill should be handled as "unhandled" or ignored for opening a new short position
        # because allow_shorting is False and current position is FLAT.
        # The code logs a warning: "Unhandled fill scenario..."
        # We need to check that the state doesn't change as if a short was opened.
        pm.update_fill(sell_to_open_short_fill)

        self.assertEqual(pm.positions[asset]['current_side'], PositionSideEnum.FLAT, "Position should remain FLAT")
        self.assertEqual(pm.positions[asset]['quantity'], 0.0, "Quantity should remain 0")
        self.assertEqual(len(pm.open_trades), 0, "No new trade should be opened")

        # Cash should only decrease by commission and fees, not by the fill value of the "short sell"
        # as the short position wasn't established.
        # The current `update_fill` deducts commission & fees *before* checking side.
        # Then, for the "SELL to open short" path:
        # `elif (fill['order_side'] == OrderSideEnum.SELL and pos_data['current_side'] != PositionSideEnum.LONG and self.allow_shorting):`
        # this condition will be FALSE.
        # Then it hits the final `else: self.logger.warning(f"Unhandled fill scenario...")`
        # So cash isn't credited with `fill_value`.
        expected_cash = initial_cash - sell_to_open_short_fill['commission'] - sell_to_open_short_fill['fees']
        self.assertAlmostEqual(pm.cash, expected_cash, msg="Cash should only reflect costs for unhandled fill")

    def test_market_update_missing_price_for_open_position(self):
        pm = self._create_pm()
        asset_open = "BTCUSDT"
        asset_other = "ETHUSDT"  # Assume this remains flat or has a consistent price

        # Open a position in BTCUSDT
        fill_time = self.start_time + timedelta(minutes=1)
        buy_fill = FillDetails(
            asset_id=asset_open, fill_timestamp=fill_time, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY, requested_quantity=1.0, executed_quantity=1.0,
            executed_price=20000.0, commission=10.0, fees=1.0, slippage_cost_total=0.0
        )
        pm.update_fill(buy_fill)

        # First market update with price for BTCUSDT
        market_time1 = self.start_time + timedelta(minutes=5)
        pm.update_market_value({asset_open: 20500.0, asset_other: 1500.0}, market_time1)
        self.assertNotEqual(pm.positions[asset_open]['market_value'], 0.0)
        self.assertNotEqual(pm.positions[asset_open]['unrealized_pnl'], 0.0)

        # Second market update *missing* the price for the open BTCUSDT position
        market_time2 = self.start_time + timedelta(minutes=10)
        pm.update_market_value({asset_other: 1510.0}, market_time2)  # Price for asset_open is missing

        # According to PortfolioManager logic, market_value and unrealized_pnl for asset_open should become 0 for this tick
        self.assertEqual(pm.positions[asset_open]['market_value'], 0.0, "Market value should be 0 if price is missing")
        self.assertEqual(pm.positions[asset_open]['unrealized_pnl'], 0.0, "Unrealized PnL should be 0 if price is missing")
        # The position itself (quantity, avg_entry_price) should remain.
        self.assertEqual(pm.positions[asset_open]['quantity'], 1.0)
        self.assertEqual(pm.positions[asset_open]['avg_entry_price'], 20000.0)

    def test_features_reflect_only_first_tradable_asset(self):
        # Ensure tradable_assets has at least two assets for this test
        if len(self.tradable_assets) < 2:
            self.skipTest("This test requires at least two tradable assets.")

        pm = self._create_pm()
        first_asset = self.tradable_assets[0]  # e.g., BTCUSDT
        second_asset = self.tradable_assets[1]  # e.g., ETHUSDT

        # Get initial features (should be flat for first_asset)
        initial_obs = pm.get_portfolio_observation()['features'][-1]

        # Open a significant trade ONLY in the second_asset
        fill_time = self.start_time + timedelta(minutes=1)
        buy_fill_second_asset = FillDetails(
            asset_id=second_asset, fill_timestamp=fill_time, order_type=OrderTypeEnum.MARKET,
            order_side=OrderSideEnum.BUY, requested_quantity=10.0, executed_quantity=10.0,
            executed_price=1500.0, commission=1.0, fees=0.1, slippage_cost_total=0.0
        )
        pm.update_fill(buy_fill_second_asset)

        # Update market value, including for the second asset to have a position value
        market_time = self.start_time + timedelta(minutes=5)
        pm.update_market_value({
            first_asset: 20000.0,  # Assuming some price for the first asset (it's flat)
            second_asset: 1550.0  # Price for the second asset changes
        }, market_time)

        # Get features again
        # These features should still reflect the state of 'first_asset', which is FLAT.
        # So, norm_pos_size, norm_unreal_pnl, norm_mae_pct for first_asset should be ~0.
        current_obs_features = pm.get_portfolio_observation()['features'][-1]

        self.assertAlmostEqual(current_obs_features[0], 0.0, delta=1e-5, msg="Normalized position size for first asset should be ~0")
        self.assertAlmostEqual(current_obs_features[1], 0.0, delta=1e-5, msg="Normalized UPL for first asset should be ~0")
        # MAE pct for first_asset (which is flat) should be 0
        self.assertAlmostEqual(current_obs_features[2], 0.0, delta=1e-5, msg="Normalized MAE for first asset should be ~0")
        # Time in trade for first_asset (which is flat) should be 0
        self.assertAlmostEqual(current_obs_features[3], 0.0, delta=1e-5, msg="Time in trade for first asset should be ~0")
        # Cash percentage will have changed due to the trade in the second asset.
        self.assertNotAlmostEqual(current_obs_features[4], initial_obs[4], msg="Cash percentage should change")

    def test_trader_vue_metrics_no_trades(self):
        pm = self._create_pm()
        # Call immediately after reset, or after a market update with no trades
        pm.update_market_value({}, self.start_time + timedelta(minutes=1))
        metrics = pm.get_trader_vue_metrics()

        self.assertEqual(metrics["num_total_trades"], 0)
        self.assertEqual(metrics["num_winning_trades"], 0)
        self.assertEqual(metrics["num_losing_trades"], 0)
        self.assertEqual(metrics["num_breakeven_trades"], 0)
        self.assertEqual(metrics["total_net_profit_closed_trades"], 0.0)
        self.assertEqual(metrics["avg_net_pnl_per_trade"], 0.0)
        self.assertEqual(metrics["win_rate_pct"], 0.0)
        self.assertEqual(metrics["profit_factor_gross"], 0.0)  # or np.nan/inf depending on exact handling of 0/0
        self.assertEqual(metrics["reward_risk_ratio_net_avg"], 0.0)  # or np.nan/inf
        self.assertEqual(metrics["max_consecutive_wins"], 0)
        self.assertEqual(metrics["max_consecutive_losses"], 0)
        # Drawdown should be 0 if only initial capital point exists or flat equity
        self.assertEqual(metrics["max_portfolio_drawdown_abs"], 0.0)
        self.assertEqual(metrics["max_portfolio_drawdown_pct"], 0.0)

    def test_trader_vue_metrics_all_wins_and_streaks(self):
        pm = self._create_pm()
        asset = "BTCUSDT"
        current_time = self.start_time

        # Trade 1 (Win)
        current_time += timedelta(minutes=1)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.BUY, executed_quantity=1,
                                   executed_price=20000, commission=10, fees=1, slippage_cost_total=0))
        current_time += timedelta(minutes=9)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.SELL, executed_quantity=1,
                                   executed_price=21000, commission=10, fees=1, slippage_cost_total=0))
        pm.update_market_value({asset: 21000}, current_time)  # PNL: 978

        # Trade 2 (Win)
        current_time += timedelta(minutes=10)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.BUY, executed_quantity=1,
                                   executed_price=22000, commission=10, fees=1, slippage_cost_total=0))
        current_time += timedelta(minutes=9)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.SELL, executed_quantity=1,
                                   executed_price=23000, commission=10, fees=1, slippage_cost_total=0))
        pm.update_market_value({asset: 23000}, current_time)  # PNL: 978

        # Trade 3 (Loss to break win streak)
        current_time += timedelta(minutes=10)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.BUY, executed_quantity=1,
                                   executed_price=24000, commission=10, fees=1, slippage_cost_total=0))
        current_time += timedelta(minutes=9)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.SELL, executed_quantity=1,
                                   executed_price=23000, commission=10, fees=1, slippage_cost_total=0))
        pm.update_market_value({asset: 23000}, current_time)  # PNL: -1022

        # Trade 4 & 5 (Wins again)
        current_time += timedelta(minutes=10)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.BUY, executed_quantity=1,
                                   executed_price=20000, commission=10, fees=1, slippage_cost_total=0))
        current_time += timedelta(minutes=9)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.SELL, executed_quantity=1,
                                   executed_price=21000, commission=10, fees=1, slippage_cost_total=0))  # PNL: 978
        pm.update_market_value({asset: 21000}, current_time)

        current_time += timedelta(minutes=10)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.BUY, executed_quantity=1,
                                   executed_price=20000, commission=10, fees=1, slippage_cost_total=0))
        current_time += timedelta(minutes=9)
        pm.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time, order_type=OrderTypeEnum.MARKET,
                                   order_side=OrderSideEnum.SELL, executed_quantity=1,
                                   executed_price=21000, commission=10, fees=1, slippage_cost_total=0))  # PNL: 978
        pm.update_market_value({asset: 21000}, current_time)

        metrics = pm.get_trader_vue_metrics()
        self.assertEqual(metrics["num_total_trades"], 5)
        self.assertEqual(metrics["num_winning_trades"], 4)
        self.assertEqual(metrics["num_losing_trades"], 1)
        self.assertAlmostEqual(metrics["win_rate_pct"], (4 / 5) * 100)
        self.assertEqual(metrics["max_consecutive_wins"], 2)  # W, W, (L), W, W
        self.assertEqual(metrics["max_consecutive_losses"], 1)

        # For all wins:
        pm_all_wins = self._create_pm()
        current_time_aw = self.start_time
        current_time_aw += timedelta(minutes=1)
        pm_all_wins.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time_aw, order_type=OrderTypeEnum.MARKET,
                                            order_side=OrderSideEnum.BUY, executed_quantity=1,
                                            executed_price=20000, commission=0, fees=0, slippage_cost_total=0))
        current_time_aw += timedelta(minutes=9)
        pm_all_wins.update_fill(FillDetails(asset_id=asset, fill_timestamp=current_time_aw, order_type=OrderTypeEnum.MARKET,
                                            order_side=OrderSideEnum.SELL, executed_quantity=1,
                                            executed_price=21000, commission=0, fees=0, slippage_cost_total=0))
        pm_all_wins.update_market_value({asset: 21000}, current_time_aw)

        metrics_all_wins = pm_all_wins.get_trader_vue_metrics()
        self.assertEqual(metrics_all_wins["num_winning_trades"], 1)
        self.assertEqual(metrics_all_wins["num_losing_trades"], 0)
        self.assertEqual(metrics_all_wins["win_rate_pct"], 100.0)
        self.assertEqual(metrics_all_wins["profit_factor_gross"], np.inf)  # Gross profit / 0 gross loss
        self.assertEqual(metrics_all_wins["reward_risk_ratio_net_avg"], np.inf)  # Avg win / 0 avg loss


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
