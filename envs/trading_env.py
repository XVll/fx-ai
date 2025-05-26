# envs/trading_env.py - IMPROVED: Better episode logging and metrics integration

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config.schemas import Config
from data.data_manager import DataManager
from envs.reward import RewardCalculator
from rewards.calculator import RewardSystemV2
from feature.feature_extractor import FeatureExtractor
from simulators.execution_simulator import ExecutionSimulator
from simulators.market_simulator import MarketSimulator
from simulators.portfolio_simulator import (
    PortfolioSimulator, PortfolioState, OrderTypeEnum, OrderSideEnum,
    PositionSideEnum, FillDetails
)


class ActionTypeEnum(Enum):
    """Defines the type of action the agent can take."""
    HOLD = 0
    BUY = 1
    SELL = 2


class PositionSizeTypeEnum(Enum):
    """Defines the relative size of the position for an action."""
    SIZE_25 = 0  # 25%
    SIZE_50 = 1  # 50%
    SIZE_75 = 2  # 75%
    SIZE_100 = 3  # 100%

    @property
    def value_float(self) -> float:
        """Returns the float multiplier for the size (0.25, 0.50, 0.75, 1.0)."""
        return (self.value + 1) * 0.25


class TerminationReasonEnum(Enum):
    """Reasons for episode termination for the info dict."""
    END_OF_SESSION_DATA = "END_OF_SESSION_DATA"
    MAX_LOSS_REACHED = "MAX_LOSS_REACHED"
    BANKRUPTCY = "BANKRUPTCY"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"
    SETUP_FAILURE = "SETUP_FAILURE"
    INVALID_ACTION_LIMIT_REACHED = "INVALID_ACTION_LIMIT_REACHED"


class TradingEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'logs', 'none'], 'render_fps': 10}

    def __init__(self, config: Config, data_manager: DataManager, logger: Optional[logging.Logger] = None,
                 metrics_integrator=None):
        super().__init__()
        self.config = config

        # Use standard logging with Rich formatting
        if logger is None:
            self.logger = logging.getLogger(f"{__name__}.TradingEnv")
        else:
            self.logger = logger

        # Environment Configuration
        env_cfg = self.config.env
        self.primary_asset: Optional[str] = None

        # Metrics integration
        self.metrics_integrator = metrics_integrator

        self.max_steps_per_episode: int = env_cfg.max_episode_steps or 57600  # Default: 16 hours
        self.random_reset_within_session: bool = config.simulation.random_start_prob > 0
        self.max_session_loss_percentage: float = 1.0 - env_cfg.early_stop_loss_threshold
        self.bankruptcy_threshold_factor: float = 0.1  # Stop if equity < 10% of initial
        self.max_invalid_actions_per_episode: int = env_cfg.invalid_action_limit or 1000

        # Position sizing configuration
        self.default_position_value = 10000.0  # Default position value

        self.data_manager = data_manager
        self.market_simulator: Optional[MarketSimulator] = None
        self.execution_manager: Optional[ExecutionSimulator] = None
        self.portfolio_manager: Optional[PortfolioSimulator] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.reward_calculator: Optional[Union[RewardCalculator, RewardSystemV2]] = None
        self.use_reward_v2 = True  # Always use RewardSystemV2

        # Action Space
        self.action_types = list(ActionTypeEnum)
        self.position_size_types = list(PositionSizeTypeEnum)
        self.action_space = spaces.MultiDiscrete([len(self.action_types), len(self.position_size_types)])

        # Action tracking for metrics and logging
        self.action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
        self.step_count_for_debug = 0
        self.win_loss_counts = {"wins": 0, "losses": 0}

        # Observation Space
        model_cfg = self.config.model
        self.observation_space: spaces.Dict = spaces.Dict({
            'hf': spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.hf_seq_len, model_cfg.hf_feat_dim),
                             dtype=np.float32),
            'mf': spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.mf_seq_len, model_cfg.mf_feat_dim),
                             dtype=np.float32),
            'lf': spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.lf_seq_len, model_cfg.lf_feat_dim),
                             dtype=np.float32),
            'portfolio': spaces.Box(low=-np.inf, high=np.inf,
                                    shape=(model_cfg.portfolio_seq_len, model_cfg.portfolio_feat_dim),
                                    dtype=np.float32),
            'static': spaces.Box(low=-np.inf, high=np.inf, shape=(1, model_cfg.static_feat_dim), dtype=np.float32),
        })

        # Episode State
        self.current_session_start_time_utc: Optional[datetime] = None
        self.current_session_end_time_utc: Optional[datetime] = None
        self.current_step: int = 0
        self.invalid_action_count_episode: int = 0
        self.episode_total_reward: float = 0.0
        self._last_observation: Optional[Dict[str, np.ndarray]] = None
        self._last_portfolio_state_before_action: Optional[PortfolioState] = None
        self._last_decoded_action: Optional[Dict[str, Any]] = None
        self.initial_capital_for_session: float = 0.0

        # Enhanced episode tracking
        self.episode_number: int = 0
        self.total_episodes: int = 0
        self.total_steps: int = 0
        self.update_count: int = 0
        self.episode_start_time: float = 0.0

        # Episode performance tracking
        self.episode_fills: List[FillDetails] = []
        self.episode_peak_equity: float = 0.0
        self.episode_max_drawdown: float = 0.0

        # Current state tracking
        self.current_termination_reason: Optional[str] = None
        self.is_terminated: bool = False
        self.is_truncated: bool = False

        self.render_mode = None  # No rendering by default

    def setup_session(self, symbol: str, start_time: Union[str, datetime], end_time: Union[str, datetime]):
        """Configures the environment for a specific trading session."""
        if not symbol or not isinstance(symbol, str):
            self.logger.error("A valid symbol (string) must be provided to setup_session.")
            raise ValueError("A valid symbol (string) must be provided to setup_session.")

        self.primary_asset = symbol

        try:
            self.current_session_start_time_utc = pd.Timestamp(start_time, tz='UTC').to_pydatetime()
            self.current_session_end_time_utc = pd.Timestamp(end_time, tz='UTC').to_pydatetime()
        except Exception as e:
            self.logger.error(f"Error parsing session start/end times: {start_time}, {end_time}. Error: {e}")
            raise ValueError(f"Invalid session start/end times: {e}")

        self.logger.info(f"🎯 Session setup: {self.primary_asset} | "
                         f"{self.current_session_start_time_utc.strftime('%Y-%m-%d %H:%M')} to "
                         f"{self.current_session_end_time_utc.strftime('%Y-%m-%d %H:%M')} UTC")

        if self.np_random is None:
            _, _ = super().reset(seed=None)

        # Initialize components
        self.market_simulator = MarketSimulator(
            symbol=self.primary_asset,
            data_manager=self.data_manager,
            simulation_config=self.config.simulation,
            model_config=self.config.model,
            mode="backtesting",  # Use backtesting mode for training
            np_random=self.np_random,
            start_time=self.current_session_start_time_utc,
            end_time=self.current_session_end_time_utc,
            logger=logging.getLogger(f"{__name__}.MarketSim")
        )

        self.portfolio_manager = PortfolioSimulator(
            logger=logging.getLogger(f"{__name__}.PortfolioMgr"),
            config=self.config,
            tradable_assets=[self.primary_asset],
            trade_callback=self._on_trade_completed
        )

        self.feature_extractor = FeatureExtractor(
            symbol=self.primary_asset,
            market_simulator=self.market_simulator,
            config=self.config.model,
            logger=logging.getLogger(f"{__name__}.FeatureExt")
        )

        # Initialize reward system based on configuration
        if self.use_reward_v2:
            self.reward_calculator = RewardSystemV2(
                config=self.config,
                metrics_integrator=self.metrics_integrator,
                logger=logging.getLogger(f"{__name__}.RewardV2")
            )
            self.logger.info("Using RewardSystemV2 with comprehensive component tracking")
            
            # Register reward components with metrics integrator
            if self.metrics_integrator and hasattr(self.reward_calculator, 'components'):
                for component in self.reward_calculator.components:
                    self.metrics_integrator.register_reward_component(
                        component.metadata.name,
                        component.metadata.type.value
                    )
        else:
            self.reward_calculator = RewardCalculator(
                config=self.config,
                metrics_integrator=self.metrics_integrator,
                logger=logging.getLogger(f"{__name__}.RewardCalc")
            )

        self.execution_manager = ExecutionSimulator(
            logger=logging.getLogger(f"{__name__}.ExecSim"),
            simulation_config=self.config.simulation,  # Pass the whole simulation config
            np_random=self.np_random,
            market_simulator=self.market_simulator
        )

        self.logger.info("✅ All simulators and managers initialized")

    def set_training_info(self, episode_num: int = 0, total_episodes: int = 0,
                          total_steps: int = 0, update_count: int = 0):
        """Set training information for metrics tracking"""
        if episode_num > self.episode_number:
            self.episode_number = episode_num

        self.total_episodes = total_episodes
        self.total_steps = total_steps
        self.update_count = update_count

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if not self.primary_asset or not self.market_simulator or \
                not self.portfolio_manager or not self.feature_extractor or \
                not self.reward_calculator or not self.execution_manager:
            self.logger.error("Session not properly set up. Call `setup_session(symbol, start, end)` before `reset()`.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Session not set up. Call setup_session first.",
                               "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        # Reset episode counters and state tracking
        self.current_step = 0
        self.invalid_action_count_episode = 0
        self.episode_total_reward = 0.0
        self._last_decoded_action = None
        self.episode_start_time = datetime.now().timestamp()

        # Reset episode performance tracking
        self.episode_fills = []
        self.episode_peak_equity = 0.0
        self.episode_max_drawdown = 0.0

        # Reset current state tracking
        self.current_termination_reason = None
        self.is_terminated = False
        self.is_truncated = False

        # Reset action debug counts
        self.action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
        self.step_count_for_debug = 0
        self.win_loss_counts = {"wins": 0, "losses": 0}

        # Increment episode number
        self.episode_number += 1

        # Start episode metrics tracking
        if self.metrics_integrator:
            self.metrics_integrator.start_episode()

        # Sync episode number with training info (for dashboard)
        self.set_training_info(episode_num=self.episode_number, 
                              total_episodes=self.total_episodes,
                              total_steps=self.total_steps, 
                              update_count=self.update_count)

        # Reset simulators
        self.execution_manager.reset(np_random_seed_source=self.np_random)

        market_reset_options = {'random_start': self.random_reset_within_session}
        if 'start_time_offset_seconds' in options:
            market_reset_options['start_time_offset_seconds'] = options['start_time_offset_seconds']

        initial_market_state = self.market_simulator.reset(options=market_reset_options)
        if initial_market_state is None or 'timestamp_utc' not in initial_market_state:
            self.logger.error("Market simulator failed to reset or provide initial state.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Market simulator reset failed.",
                               "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        current_sim_time = initial_market_state['timestamp_utc']
        
        # Start visualization tracking after we have the market state
        if self.metrics_integrator and hasattr(self.metrics_integrator, 'metrics_manager'):
            episode_date = current_sim_time.strftime('%Y-%m-%d')
            self.metrics_integrator.metrics_manager.start_episode_visualization(
                self.episode_number, self.primary_asset, episode_date
            )
            
            # Start dashboard episode tracking with the correct episode number
            if hasattr(self.metrics_integrator.metrics_manager, 'dashboard_collector') and self.metrics_integrator.metrics_manager.dashboard_collector:
                self.metrics_integrator.metrics_manager.dashboard_collector.on_episode_start(self.episode_number)
                
            # Update metrics manager state with current episode
            self.metrics_integrator.metrics_manager.update_state(
                episode=self.episode_number,
                is_training=True
            )

        self.portfolio_manager.reset(episode_start_timestamp=current_sim_time)
        self.initial_capital_for_session = self.portfolio_manager.initial_capital
        self.episode_peak_equity = self.initial_capital_for_session

        if hasattr(self.reward_calculator, 'reset'):
            self.reward_calculator.reset()
        if hasattr(self.feature_extractor, 'reset'):
            self.feature_extractor.reset()

        self._last_observation = self._get_observation(initial_market_state, current_sim_time)
        if self._last_observation is None:
            self.logger.error("Failed to get initial observation.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Initial observation failed.",
                               "termination_reason": TerminationReasonEnum.OBSERVATION_FAILURE.value}

        self._last_portfolio_state_before_action = self.portfolio_manager.get_portfolio_state(current_sim_time)
        initial_info = self._get_current_info(reward=0.0,
                                              current_portfolio_state_for_info=self._last_portfolio_state_before_action)

        # Update metrics with initial portfolio state
        if self.metrics_integrator and self._last_portfolio_state_before_action:
            self._update_portfolio_metrics(self._last_portfolio_state_before_action)

        return self._last_observation, initial_info

    def _get_dummy_observation(self) -> Dict[str, np.ndarray]:
        dummy_obs = {}
        if isinstance(self.observation_space, spaces.Dict):
            for key in self.observation_space.keys():
                space_item = self.observation_space[key]
                dummy_obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)
        else:
            self.logger.error("Observation space is not a gymnasium.spaces.Dict. Cannot create dummy observation.")
        return dummy_obs

    def _get_observation(self, market_state_now: Dict[str, Any], current_sim_time: datetime) -> Optional[
        Dict[str, np.ndarray]]:
        if market_state_now is None:
            self.logger.warning(f"Market state is None at {current_sim_time} during observation generation.")
            return None
        try:
            current_portfolio_state = self.portfolio_manager.get_portfolio_state(current_sim_time)
            market_features_dict = self.feature_extractor.extract_features()
            if market_features_dict is None:
                self.logger.warning(f"FeatureExtractor returned None at {current_sim_time}.")
                return None
        except Exception as e:
            self.logger.error(f"Error during feature extraction at {current_sim_time}: {e}")
            return None

        portfolio_obs_component = self.portfolio_manager.get_portfolio_observation()
        portfolio_features_array = portfolio_obs_component['features']

        obs = {
            'hf': market_features_dict.get('hf'),
            'mf': market_features_dict.get('mf'),
            'lf': market_features_dict.get('lf'),
            'static': market_features_dict.get('static'),
            'portfolio': portfolio_features_array
        }

        # Handle NaN values
        for key, arr in obs.items():
            if arr is not None:
                nan_count = np.isnan(arr).sum()
                if nan_count > 0:
                    obs[key] = np.nan_to_num(arr, nan=0.0)

        if not isinstance(self.observation_space, spaces.Dict):
            self.logger.error("Observation space is not a gymnasium.spaces.Dict.")
            return None

        for key in self.observation_space.keys():
            space_item = self.observation_space[key]
            if obs.get(key) is None:
                obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)

            if key == 'static' and obs[key].ndim == 1:
                obs[key] = obs[key].reshape(1, -1)

            if obs[key].shape != space_item.shape:
                self.logger.error(f"Shape mismatch for observation key '{key}'. "
                                  f"Expected {space_item.shape}, Got {obs[key].shape}.")
                return None
        return obs

    def _decode_action(self, raw_action) -> Dict[str, Any]:
        """Decode the agent's action into a structured format."""
        # Extract action components, handling both tuples and arrays
        if isinstance(raw_action, (tuple, list)):
            action_type_idx, size_type_idx = raw_action
            raw_action_list = list(raw_action)
        elif hasattr(raw_action, 'tolist'):  # NumPy array or PyTorch tensor
            action_type_idx, size_type_idx = raw_action
            raw_action_list = raw_action.tolist()
        else:
            self.logger.error(f"Unexpected action type: {type(raw_action)}")
            action_type_idx, size_type_idx = 0, 0
            raw_action_list = [0, 0]

        # Ensure indices are integers and within valid range
        action_type_idx = int(action_type_idx) % len(self.action_types)
        size_type_idx = int(size_type_idx) % len(self.position_size_types)

        action_type = self.action_types[action_type_idx]
        size_type = self.position_size_types[size_type_idx]

        # Track debug counts
        self.action_counts[action_type.name] += 1
        self.step_count_for_debug += 1

        return {
            "type": action_type,
            "size_enum": size_type,
            "size_float": size_type.value_float,
            "raw_action": raw_action_list,
            "invalid_reason": None
        }

    def _translate_agent_action_to_order(self, decoded_action: Dict[str, Any], portfolio_state: PortfolioState,
                                         market_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Translate agent action using FIXED dollar amounts with better validation"""
        action_type = decoded_action['type']
        size_float = decoded_action['size_float']

        if not self.primary_asset:
            decoded_action['invalid_reason'] = "Primary asset not set in environment."
            self.invalid_action_count_episode += 1
            return None
        asset_id = self.primary_asset

        ideal_ask = market_state.get('best_ask_price')
        ideal_bid = market_state.get('best_bid_price')
        current_price_fallback = market_state.get('current_price')

        if ideal_ask is None or ideal_bid is None:
            if current_price_fallback is not None and current_price_fallback > 0:
                ideal_ask = current_price_fallback * 1.0002
                ideal_bid = current_price_fallback * 0.9998
            else:
                decoded_action['invalid_reason'] = "Missing market prices (BBO and current) for order."
                self.invalid_action_count_episode += 1
                return None

        # Better price validation
        if ideal_ask <= 0 or ideal_bid <= 0 or ideal_ask <= ideal_bid:
            decoded_action['invalid_reason'] = f"Invalid BBO prices: Ask ${ideal_ask:.4f}, Bid ${ideal_bid:.4f}."
            self.invalid_action_count_episode += 1
            return None

        pos_data = portfolio_state['positions'].get(asset_id)
        if not pos_data:
            decoded_action['invalid_reason'] = f"Position data for asset {asset_id} not found."
            self.invalid_action_count_episode += 1
            return None

        current_qty = pos_data['quantity']
        current_pos_side = pos_data['current_side']
        cash = portfolio_state['cash']
        total_equity = portfolio_state['total_equity']

        max_pos_value_abs = total_equity * self.portfolio_manager.max_position_value_ratio
        allow_shorting = self.portfolio_manager.allow_shorting

        quantity_to_trade = 0.0
        order_side: Optional[OrderSideEnum] = None

        if action_type == ActionTypeEnum.HOLD:
            return None

        elif action_type == ActionTypeEnum.BUY:
            # Use FIXED dollar amount instead of equity percentage
            target_buy_value = self.default_position_value * size_float

            # Better cash and risk validation
            available_cash = min(cash, target_buy_value)
            target_buy_value = min(available_cash, max_pos_value_abs)

            if target_buy_value > 10.0 and ideal_ask > 0.01:
                quantity_to_trade = target_buy_value / ideal_ask
                order_side = OrderSideEnum.BUY

                # Handle covering short positions
                if current_pos_side == PositionSideEnum.SHORT:
                    quantity_to_trade += abs(current_qty)

            else:
                if target_buy_value <= 10.0:
                    decoded_action['invalid_reason'] = f"Insufficient buying power: ${target_buy_value:.2f} < $10.00"
                else:
                    decoded_action['invalid_reason'] = f"Invalid ask price: ${ideal_ask:.4f}"

        elif action_type == ActionTypeEnum.SELL:
            if current_pos_side == PositionSideEnum.LONG and current_qty > 0:
                # Sell percentage of current position
                quantity_to_trade = size_float * current_qty
                order_side = OrderSideEnum.SELL

            elif allow_shorting:
                # Use FIXED dollar amount for shorting
                target_short_value = self.default_position_value * size_float
                target_short_value = min(target_short_value, max_pos_value_abs)

                if target_short_value > 10.0 and ideal_bid > 0.01:
                    quantity_to_trade = target_short_value / ideal_bid
                    order_side = OrderSideEnum.SELL

                    # Handle covering long positions when initiating short
                    if current_pos_side == PositionSideEnum.LONG:
                        quantity_to_trade += current_qty

                else:
                    if target_short_value <= 10.0:
                        decoded_action['invalid_reason'] = f"Insufficient shorting power: ${target_short_value:.2f} < $10.00"
                    else:
                        decoded_action['invalid_reason'] = f"Invalid bid price: ${ideal_bid:.4f}"
            else:
                decoded_action['invalid_reason'] = "SELL: No long position and shorting disabled."

        # Final order validation
        if quantity_to_trade > 0.01 and order_side is not None:
            quantity_to_trade = abs(quantity_to_trade)
            order_params = {
                'asset_id': asset_id,
                'order_type': OrderTypeEnum.MARKET,
                'order_side': order_side,
                'quantity': quantity_to_trade,
                'ideal_decision_price_ask': ideal_ask,
                'ideal_decision_price_bid': ideal_bid
            }
            decoded_action['translated_order'] = {
                k: v.value if isinstance(v, Enum) else v
                for k, v in order_params.items()
            }

            return order_params
        elif decoded_action['invalid_reason'] is None and action_type != ActionTypeEnum.HOLD:
            decoded_action['invalid_reason'] = f"Quantity too small: {quantity_to_trade:.6f} (min=0.01)"

        if decoded_action['invalid_reason']:
            self.invalid_action_count_episode += 1

        return None

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._last_observation is None or self.primary_asset is None:
            self.logger.error("Step called with invalid state.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, 0.0, True, False, {
                "error": "Critical: invalid state.",
                "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        self.current_step += 1
        market_state_at_decision = self.market_simulator.get_current_market_state()
        if market_state_at_decision is None or 'timestamp_utc' not in market_state_at_decision:
            self.logger.error("Market simulator returned invalid state. Terminating.")
            return self._last_observation, 0.0, True, False, {"error": "Market state unavailable.",
                                                              "termination_reason": TerminationReasonEnum.OBSERVATION_FAILURE.value}

        current_sim_time_decision = market_state_at_decision['timestamp_utc']
        self._last_portfolio_state_before_action = self.portfolio_manager.get_portfolio_state(current_sim_time_decision)
        self._last_decoded_action = self._decode_action(action)
        order_request = self._translate_agent_action_to_order(
            self._last_decoded_action, self._last_portfolio_state_before_action, market_state_at_decision
        )

        fill_details_list: List[FillDetails] = []
        if order_request:
            fill = self.execution_manager.execute_order(
                asset_id=order_request['asset_id'], order_type=order_request['order_type'],
                order_side=order_request['order_side'], requested_quantity=order_request['quantity'],
                ideal_decision_price_ask=order_request['ideal_decision_price_ask'],
                ideal_decision_price_bid=order_request['ideal_decision_price_bid'],
                decision_timestamp=current_sim_time_decision
            )
            if fill:
                fill_details_list.append(fill)
                self.episode_fills.append(fill)
                self.portfolio_manager.update_fill(fill)

                # Record fill in metrics
                if self.metrics_integrator:
                    self.metrics_integrator.record_fill({
                        'executed_quantity': fill['executed_quantity'],
                        'executed_price': fill['executed_price'],
                        'commission': fill['commission'],
                        'fees': fill['fees'],
                        'slippage_cost_total': fill['slippage_cost_total']
                    })

        time_for_pf_update_after_fill = fill_details_list[-1]['fill_timestamp'] if fill_details_list else current_sim_time_decision

        price_for_decision_val = market_state_at_decision.get('current_price')
        if price_for_decision_val is None or price_for_decision_val <= 0:
            ask = market_state_at_decision.get('best_ask_price')
            bid = market_state_at_decision.get('best_bid_price')
            if ask is not None and bid is not None and ask > 0 and bid > 0:
                price_for_decision_val = (ask + bid) / 2
            else:
                price_for_decision_val = 0.0

        prices_at_decision_time = {self.primary_asset: price_for_decision_val}
        self.portfolio_manager.update_market_value(prices_at_decision_time, time_for_pf_update_after_fill)
        portfolio_state_after_action_fills = self.portfolio_manager.get_portfolio_state(time_for_pf_update_after_fill)

        market_advanced = self.market_simulator.step()
        market_state_next_t: Optional[Dict[str, Any]] = None
        next_sim_time: Optional[datetime] = None

        if market_advanced:
            market_state_next_t = self.market_simulator.get_current_market_state()
            if market_state_next_t and 'timestamp_utc' in market_state_next_t:
                next_sim_time = market_state_next_t['timestamp_utc']
            else:
                market_advanced = False

        if market_state_next_t and next_sim_time:
            price_for_next_val = market_state_next_t.get('current_price')
            if price_for_next_val is None or price_for_next_val <= 0:
                ask = market_state_next_t.get('best_ask_price')
                bid = market_state_next_t.get('best_bid_price')
                if ask is not None and bid is not None and ask > 0 and bid > 0:
                    price_for_next_val = (ask + bid) / 2
                else:
                    price_for_next_val = 0.0

            prices_at_next_time = {self.primary_asset: price_for_next_val}
            self.portfolio_manager.update_market_value(prices_at_next_time, next_sim_time)

        portfolio_state_next_t = self.portfolio_manager.get_portfolio_state(
            next_sim_time or time_for_pf_update_after_fill)

        # Track episode performance
        current_equity = portfolio_state_next_t['total_equity']
        if current_equity > self.episode_peak_equity:
            self.episode_peak_equity = current_equity

        current_drawdown = (self.episode_peak_equity - current_equity) / self.episode_peak_equity if self.episode_peak_equity > 0 else 0
        if current_drawdown > self.episode_max_drawdown:
            self.episode_max_drawdown = current_drawdown

        observation_next_t: Optional[Dict[str, np.ndarray]] = None
        terminated_by_obs_failure = False
        if market_state_next_t and next_sim_time:
            observation_next_t = self._get_observation(market_state_next_t, next_sim_time)
            if observation_next_t is None:
                observation_next_t = self._last_observation
                terminated_by_obs_failure = True
        else:
            observation_next_t = self._last_observation

        if observation_next_t is None:
            observation_next_t = self._get_dummy_observation()
            if not terminated_by_obs_failure:
                terminated_by_obs_failure = True

        self._last_observation = observation_next_t

        terminated = False
        truncated = False
        termination_reason: Optional[TerminationReasonEnum] = None

        if current_equity <= self.initial_capital_for_session * self.bankruptcy_threshold_factor:
            terminated = True
            termination_reason = TerminationReasonEnum.BANKRUPTCY
        elif current_equity <= self.initial_capital_for_session * (1 - self.max_session_loss_percentage):
            terminated = True
            termination_reason = TerminationReasonEnum.MAX_LOSS_REACHED

        if terminated_by_obs_failure and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.OBSERVATION_FAILURE

        if not market_advanced and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.END_OF_SESSION_DATA

        if self.invalid_action_count_episode >= self.max_invalid_actions_per_episode and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.INVALID_ACTION_LIMIT_REACHED

        if not terminated and self.current_step >= self.max_steps_per_episode:
            truncated = True

        # Update current state tracking
        self.is_terminated = terminated
        self.is_truncated = truncated
        if termination_reason:
            self.current_termination_reason = termination_reason.value

        reward = self.reward_calculator.calculate(
            portfolio_state_before_action=self._last_portfolio_state_before_action,
            portfolio_state_after_action_fills=portfolio_state_after_action_fills,
            portfolio_state_next_t=portfolio_state_next_t,
            market_state_at_decision=market_state_at_decision,
            market_state_next_t=market_state_next_t,
            decoded_action=self._last_decoded_action,
            fill_details_list=fill_details_list,
            terminated=terminated, truncated=truncated,
            termination_reason=termination_reason
        )
        self.episode_total_reward += reward

        info = self._get_current_info(
            reward=reward,
            fill_details_list=fill_details_list,
            current_portfolio_state_for_info=portfolio_state_next_t,
            termination_reason_enum=termination_reason,
            is_terminated=terminated,
            is_truncated=truncated
        )

        # Update metrics
        if self.metrics_integrator:
            # Record environment step
            action_name = self._last_decoded_action['type'].name if self._last_decoded_action else 'UNKNOWN'
            is_invalid = bool(self._last_decoded_action.get('invalid_reason')) if self._last_decoded_action else False
            
            # Collect visualization data
            if hasattr(self.metrics_integrator, 'metrics_manager'):
                # Use the most recent price available
                viz_price = price_for_decision_val  # Always defined
                if market_state_next_t:
                    viz_price = market_state_next_t.get('current_price', price_for_decision_val)
                
                viz_data = {
                    'price': viz_price,
                    'volume': market_state_next_t.get('total_volume', 0) if market_state_next_t else 0,
                    'position': portfolio_state_next_t['positions'].get(self.primary_asset, {}).get('quantity', 0),
                    'reward': reward,
                    'action': action_name,
                    'vwap': market_state_next_t.get('vwap', 0) if market_state_next_t else 0,
                }
                
                # Add feature data if available
                if hasattr(self.feature_extractor, 'last_features'):
                    features = self.feature_extractor.last_features
                    if features:
                        viz_data.update({
                            'rsi': features.get('rsi', 0),
                            'volatility': features.get('volatility_1m', 0),
                            'sma_fast': features.get('sma_20', 0),
                            'sma_slow': features.get('sma_50', 0),
                            'momentum': features.get('momentum_1m', 0),
                            'volume_ratio': features.get('volume_ratio', 0),
                        })
                
                self.metrics_integrator.metrics_manager.collect_step_visualization(viz_data)
                
                # Send to dashboard with market state for OHLC data
                pos_data = portfolio_state_next_t['positions'].get(self.primary_asset, {})
                dashboard_data = {
                    'step': self.current_step,
                    'symbol': self.primary_asset,
                    'price': viz_price,
                    'bid': market_state_next_t.get('best_bid_price', viz_price * 0.9999) if market_state_next_t else viz_price * 0.9999,
                    'ask': market_state_next_t.get('best_ask_price', viz_price * 1.0001) if market_state_next_t else viz_price * 1.0001,
                    'volume': market_state_next_t.get('total_volume', 0) if market_state_next_t else 0,
                    'position': pos_data.get('quantity', 0),
                    'avg_entry_price': pos_data.get('avg_entry_price', viz_price),
                    'reward': reward,
                    'equity': portfolio_state_next_t['total_equity'],
                    'cash': portfolio_state_next_t['cash'],
                    'realized_pnl': portfolio_state_next_t['realized_pnl_session'],
                    'unrealized_pnl': portfolio_state_next_t['unrealized_pnl'],
                    'total_commission': portfolio_state_next_t.get('total_commissions_session', 0.0),
                    'total_slippage': portfolio_state_next_t.get('total_slippage_cost_session', 0.0),
                    'total_fees': portfolio_state_next_t.get('total_fees_session', 0.0),
                    'action': action_name,
                    'size': self._last_decoded_action.get('size_float', 1.0) if self._last_decoded_action else 1.0,
                    'invalid_action': is_invalid,  # Track invalid actions
                    'market_state': market_state_next_t,  # Pass full market state for OHLC
                    'reward_components': self.reward_calculator.get_last_reward_components()  # Include reward components
                }
                self.metrics_integrator.metrics_manager.update_dashboard_step(dashboard_data)
                
                # Send executions to dashboard (not trades - trades are handled by portfolio callback)
                if fill_details_list:
                    for fill in fill_details_list:
                        # Only send execution data to dashboard
                        dashboard_execution = {
                            'order_side': fill['order_side'].value,
                            'executed_quantity': fill['executed_quantity'],
                            'asset_id': fill['asset_id'],
                            'executed_price': fill['executed_price'],
                            'commission': fill['commission'],
                            'fees': fill['fees'],
                            'slippage_cost_total': fill['slippage_cost_total']
                        }
                        
                        # Add timestamp from market state
                        if market_state_next_t and 'timestamp_utc' in market_state_next_t:
                            dashboard_execution['timestamp'] = market_state_next_t['timestamp_utc']
                        if hasattr(self.metrics_integrator.metrics_manager, 'dashboard_collector') and self.metrics_integrator.metrics_manager.dashboard_collector:
                            self.metrics_integrator.metrics_manager.dashboard_collector.on_execution(dashboard_execution)

            self.metrics_integrator.record_environment_step(
                reward=reward,
                action=action_name,
                is_invalid=is_invalid,
                reward_components=self.reward_calculator.get_last_reward_components(),
                episode_reward=self.episode_total_reward
            )

            # Update portfolio metrics
            self._update_portfolio_metrics(portfolio_state_next_t)

            # Update position metrics
            # Use the most recent price available for position metrics
            position_price = price_for_decision_val
            if market_state_next_t:
                position_price = market_state_next_t.get('current_price', price_for_decision_val)
            self._update_position_metrics(portfolio_state_next_t, position_price)

        if terminated or truncated:
            # Comprehensive episode summary
            episode_duration = datetime.now().timestamp() - self.episode_start_time
            final_metrics = self.portfolio_manager.get_trader_vue_metrics()

            info['episode_summary'] = {
                "total_reward": self.episode_total_reward,
                "steps": self.current_step,
                "duration_seconds": episode_duration,
                "final_equity": portfolio_state_next_t['total_equity'],
                "peak_equity": self.episode_peak_equity,
                "max_drawdown_pct": self.episode_max_drawdown * 100,
                "session_realized_pnl_net": portfolio_state_next_t['realized_pnl_session'],
                "session_net_profit_equity_change": portfolio_state_next_t['total_equity'] - self.initial_capital_for_session,
                "session_total_commissions": portfolio_state_next_t['total_commissions_session'],
                "session_total_fees": portfolio_state_next_t['total_fees_session'],
                "session_total_slippage_cost": portfolio_state_next_t['total_slippage_cost_session'],
                "termination_reason": termination_reason.value if termination_reason else ("TRUNCATED" if truncated else "UNKNOWN"),
                "invalid_actions_in_episode": self.invalid_action_count_episode,
                "total_fills": len(self.episode_fills),
                **final_metrics
            }

            # End episode metrics tracking
            if self.metrics_integrator:
                self.metrics_integrator.end_episode(self.episode_total_reward, self.current_step)
                self.metrics_integrator.record_episode_end(self.episode_total_reward)
                
                # Generate and send episode visualizations
                if hasattr(self.metrics_integrator, 'metrics_manager'):
                    self.metrics_integrator.metrics_manager.end_episode_visualization()
                    
                    # Send episode summary to dashboard
                    episode_pnl = info['episode_summary']['session_net_profit_equity_change']
                    episode_data = {
                        'episode': self.episode_number,
                        'total_reward': self.episode_total_reward,
                        'total_pnl': episode_pnl,
                        'steps': self.current_step,
                        'win_rate': (self.win_loss_counts['wins'] / max(1, self.win_loss_counts['wins'] + self.win_loss_counts['losses'])) * 100,
                        'termination_reason': info['episode_summary']['termination_reason'],
                        'truncated': truncated,
                        'reset': True
                    }
                    self.metrics_integrator.metrics_manager.update_dashboard_episode(episode_data)

            # Enhanced episode summary logging
            total_actions = sum(self.action_counts.values())
            action_dist = {}
            if total_actions > 0:
                for action, count in self.action_counts.items():
                    action_dist[action] = (count / total_actions) * 100

            pnl = info['episode_summary']['session_net_profit_equity_change']
            pnl_pct = (pnl / self.initial_capital_for_session) * 100

            self.logger.info(f"🏁 EPISODE {self.episode_number} COMPLETE ({self.primary_asset})")
            self.logger.info(f"   💰 P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Reward: {self.episode_total_reward:.4f}")
            self.logger.info(f"   📊 Steps: {self.current_step} | Duration: {episode_duration:.1f}s")
            self.logger.info(f"   📈 Peak: ${self.episode_peak_equity:.2f} | Max DD: {self.episode_max_drawdown * 100:.1f}%")
            self.logger.info(f"   🎯 Actions: B{action_dist.get('BUY', 0):.0f}% S{action_dist.get('SELL', 0):.0f}% H{action_dist.get('HOLD', 0):.0f}%")
            self.logger.info(f"   🔄 Fills: {len(self.episode_fills)} | Invalid: {self.invalid_action_count_episode}")
            self.logger.info(f"   🏁 Reason: {info['episode_summary']['termination_reason']}")

        return observation_next_t, reward, terminated, truncated, info

    def _update_portfolio_metrics(self, portfolio_state: PortfolioState):
        """Update portfolio metrics"""
        if self.metrics_integrator:
            self.metrics_integrator.update_portfolio(
                equity=portfolio_state['total_equity'],
                cash=portfolio_state['cash'],
                unrealized_pnl=portfolio_state['unrealized_pnl'],
                realized_pnl=portfolio_state['realized_pnl_session'],
                total_commission=portfolio_state.get('total_commissions_session', 0.0),
                total_slippage=portfolio_state.get('total_slippage_cost_session', 0.0),
                total_fees=portfolio_state.get('total_fees_session', 0.0)
            )

    def _update_position_metrics(self, portfolio_state: PortfolioState, current_price: float):
        """Update position metrics"""
        if self.metrics_integrator and self.primary_asset:
            pos_data = portfolio_state['positions'].get(self.primary_asset, {})

            self.metrics_integrator.update_position(
                quantity=pos_data.get('quantity', 0.0),
                side=pos_data.get('current_side', PositionSideEnum.FLAT).value,
                avg_entry_price=pos_data.get('avg_entry_price', 0.0),
                market_value=pos_data.get('market_value', 0.0),
                unrealized_pnl=pos_data.get('unrealized_pnl', 0.0),
                current_price=current_price
            )

    def _get_current_info(self, reward: float, current_portfolio_state_for_info: PortfolioState,
                          fill_details_list: Optional[List[FillDetails]] = None,
                          termination_reason_enum: Optional[TerminationReasonEnum] = None,
                          is_terminated: bool = False, is_truncated: bool = False) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            'timestamp_iso': current_portfolio_state_for_info['timestamp'].isoformat(),
            'step': self.current_step,
            'reward_step': reward,
            'episode_cumulative_reward': self.episode_total_reward,
            'action_decoded': self._last_decoded_action,
            'fills_step': fill_details_list if fill_details_list else [],
            'portfolio_equity': current_portfolio_state_for_info['total_equity'],
            'portfolio_cash': current_portfolio_state_for_info['cash'],
            'portfolio_unrealized_pnl': current_portfolio_state_for_info['unrealized_pnl'],
            'portfolio_realized_pnl_session_net': current_portfolio_state_for_info['realized_pnl_session'],
            'invalid_action_in_step': bool(self._last_decoded_action.get('invalid_reason')) if self._last_decoded_action else False,
            'invalid_actions_total_episode': self.invalid_action_count_episode,
            'global_step_counter': self.total_steps,
            'episode_number': self.episode_number
        }

        if self.primary_asset:
            pos_detail = current_portfolio_state_for_info['positions'].get(self.primary_asset, {})
            info[f'position_{self.primary_asset}_qty'] = pos_detail.get('quantity', 0.0)
            info[f'position_{self.primary_asset}_side'] = pos_detail.get('current_side', PositionSideEnum.FLAT).value
            info[f'position_{self.primary_asset}_avg_entry'] = pos_detail.get('avg_entry_price', 0.0)

        if is_terminated and termination_reason_enum:
            info['termination_reason'] = termination_reason_enum.value
        if is_truncated:
            info['TimeLimit.truncated'] = True
        return info

    def _on_trade_completed(self, trade: Dict[str, Any]):
        """Callback for when a trade is completed"""
        try:
            # Update win/loss counts based on PnL
            pnl = trade.get('realized_pnl', 0.0)
            if pnl > 0:
                self.win_loss_counts['wins'] += 1
            elif pnl < 0:
                self.win_loss_counts['losses'] += 1
            
            # Convert trade data to dashboard format
            dashboard_trade = {
                'action': 'LONG' if trade['side'] == PositionSideEnum.LONG else 'SHORT',
                'quantity': abs(trade['entry_quantity_total']),
                'symbol': trade.get('asset_id', self.primary_asset),
                'entry_price': trade['avg_entry_price'],
                'exit_price': trade.get('avg_exit_price'),
                'pnl': pnl,
                'fees': trade.get('commission_total', 0.0) + trade.get('fees_total', 0.0),
                'commission': trade.get('commission_total', 0.0),
                'slippage': trade.get('slippage_total_trade_usd', 0.0)
            }
            
            # Send to dashboard if available
            if hasattr(self.metrics_integrator.metrics_manager, 'dashboard_collector') and self.metrics_integrator.metrics_manager.dashboard_collector:
                self.metrics_integrator.metrics_manager.dashboard_collector.on_trade(dashboard_trade)
                
        except Exception as e:
            self.logger.warning(f"Error in trade callback: {e}")

    def render(self, info_dict: Optional[Dict[str, Any]] = None):
        """Render method - basic console output"""
        if self.render_mode in ['human', 'logs'] and info_dict:
            print(f"Step {info_dict.get('step', 'N/A')}: "
                  f"Reward {info_dict.get('reward_step', 0.0):.4f}, "
                  f"Equity ${info_dict.get('portfolio_equity', 0.0):.2f}")

    def close(self):
        if self.market_simulator and hasattr(self.market_simulator, 'close'):
            self.market_simulator.close()
        self.logger.info("🔒 TradingEnvironment closed")