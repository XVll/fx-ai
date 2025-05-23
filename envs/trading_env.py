# envs/trading_env.py - FIXED: Position sizing and dashboard updates

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union
import time

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config.config import Config
from data.data_manager import DataManager
from envs.env_dashboard import TradingDashboard
from envs.reward import RewardCalculator
from feature.feature_extractor import FeatureExtractor
from simulators.execution_simulator import ExecutionSimulator
from simulators.market_simulator import MarketSimulator
from simulators.portfolio_simulator import (
    PortfolioManager, PortfolioState, OrderTypeEnum, OrderSideEnum,
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
    metadata = {'render_modes': ['human', 'logs', 'dashboard', 'none'], 'render_fps': 10}

    def __init__(self, config: Config, data_manager: DataManager, logger: Optional[logging.Logger] = None,
                 dashboard: Optional[TradingDashboard] = None):
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

        # Dashboard setup
        self.dashboard = dashboard
        self.use_dashboard = self.dashboard is not None

        # FIXED: Reduced dashboard update throttling for responsive updates
        self._last_dashboard_update = 0.0
        self._dashboard_update_interval = 0.5  # FIXED: Reduced to 0.5 seconds for faster updates
        self._step_update_counter = 0
        self._dashboard_update_frequency = 10  # FIXED: Update every 10 steps for more responsive updates

        if self.use_dashboard:
            logging.info("Trading dashboard mode enabled - will start after session setup")

        self.max_steps_per_episode: int = env_cfg.max_steps
        self.random_reset_within_session: bool = env_cfg.random_reset
        self.max_session_loss_percentage: float = env_cfg.max_episode_loss_percent
        self.bankruptcy_threshold_factor: float = env_cfg.bankruptcy_threshold_factor
        self.max_invalid_actions_per_episode: int = env_cfg.max_invalid_actions_per_episode

        # FIXED: Configurable position sizing with fixed dollar amounts
        self.default_position_value = getattr(config.simulation.portfolio_config, 'default_position_value', 10000.0)
        self.logger.info(
            f"🔧 Using FIXED position sizing: Base=${self.default_position_value:.2f} (SIZE_25=${self.default_position_value * 0.25:.2f}, SIZE_50=${self.default_position_value * 0.5:.2f}, SIZE_75=${self.default_position_value * 0.75:.2f}, SIZE_100=${self.default_position_value:.2f})")

        self.data_manager = data_manager
        self.market_simulator: Optional[MarketSimulator] = None
        self.execution_manager: Optional[ExecutionSimulator] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.reward_calculator: Optional[RewardCalculator] = None

        # Action Space
        self.action_types = list(ActionTypeEnum)
        self.position_size_types = list(PositionSizeTypeEnum)
        self.action_space = spaces.MultiDiscrete([len(self.action_types), len(self.position_size_types)])

        # FIXED: Enhanced action distribution tracking for episode-wide statistics
        self.action_debug_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
        self.step_count_for_debug = 0
        self.episode_action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}

        logging.info(f"Action space: {self.action_space} "
                     f"(ActionTypes: {[a.name for a in self.action_types]}, "
                     f"PositionSizes: {[s.name for s in self.position_size_types]})")

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

        # FIXED: Enhanced training state tracking for dashboard
        self.episode_number: int = 0
        self.total_episodes: int = 0
        self.total_steps: int = 0
        self.update_count: int = 0

        # FIXED: Current state tracking for footer
        self.current_termination_reason: Optional[str] = None
        self.is_terminated: bool = False
        self.is_truncated: bool = False

        self.render_mode = env_cfg.render_mode

    def setup_session(self, symbol: str, start_time: Union[str, datetime], end_time: Union[str, datetime]):
        """Configures the environment for a specific trading session."""
        if not symbol or not isinstance(symbol, str):
            logging.error("A valid symbol (string) must be provided to setup_session.")
            raise ValueError("A valid symbol (string) must be provided to setup_session.")

        self.primary_asset = symbol

        try:
            self.current_session_start_time_utc = pd.Timestamp(start_time, tz='UTC').to_pydatetime()
            self.current_session_end_time_utc = pd.Timestamp(end_time, tz='UTC').to_pydatetime()
        except Exception as e:
            logging.error(f"Error parsing session start/end times: {start_time}, {end_time}. Error: {e}")
            raise ValueError(f"Invalid session start/end times: {e}")

        logging.info(f"Setting up session for symbol '{self.primary_asset}' "
                     f"from {self.current_session_start_time_utc} to {self.current_session_end_time_utc}.")

        if self.np_random is None:
            _, _ = super().reset(seed=None)

        # Initialize components with standard logging
        self.market_simulator = MarketSimulator(
            symbol=self.primary_asset,
            data_manager=self.data_manager,
            market_config=self.config.simulation.market_config,
            model_config=self.config.model,
            mode=self.config.env.training_mode,
            np_random=self.np_random,
            start_time=self.current_session_start_time_utc,
            end_time=self.current_session_end_time_utc,
            logger=logging.getLogger(f"{__name__}.MarketSim")
        )

        self.portfolio_manager = PortfolioManager(
            logger=logging.getLogger(f"{__name__}.PortfolioMgr"),
            config=self.config,
            tradable_assets=[self.primary_asset]
        )

        self.feature_extractor = FeatureExtractor(
            symbol=self.primary_asset,
            market_simulator=self.market_simulator,
            config=self.config.model,
            logger=logging.getLogger(f"{__name__}.FeatureExt")
        )

        self.reward_calculator = RewardCalculator(
            config=self.config,
            logger=logging.getLogger(f"{__name__}.RewardCalc")
        )

        self.execution_manager = ExecutionSimulator(
            logger=logging.getLogger(f"{__name__}.ExecSim"),
            config_exec=self.config.simulation.execution_config,
            np_random=self.np_random,
            market_simulator=self.market_simulator
        )

        if self.use_dashboard:
            self.dashboard.set_symbol(symbol)
            self.dashboard.set_initial_capital(self.config.simulation.portfolio_config.initial_cash)

        logging.info("All simulators and managers initialized for the session.")

    def set_training_info(self, episode_num: int = 0, total_episodes: int = 0,
                          total_steps: int = 0, update_count: int = 0,
                          buffer_size: int = 0, is_training: bool = True,
                          is_evaluating: bool = False, learning_rate: float = 0.0):
        """Set training information for dashboard display"""
        # FIXED: Proper episode number tracking
        if episode_num > self.episode_number:
            self.episode_number = episode_num

        self.total_episodes = total_episodes
        self.total_steps = total_steps
        self.update_count = update_count

        # Update dashboard if available (but only occasionally)
        if self.dashboard and self._should_update_dashboard_training():
            self.dashboard.set_training_info(
                episode_num=self.episode_number,
                total_episodes=total_episodes,
                total_steps=total_steps,
                update_count=update_count,
                buffer_size=buffer_size,
                is_training=is_training,
                is_evaluating=is_evaluating,
                learning_rate=learning_rate
            )

    def _should_update_dashboard_training(self) -> bool:
        """Check if enough time has passed to update training info on dashboard"""
        current_time = time.time()
        if current_time - self._last_dashboard_update >= 1.0:  # FIXED: Update training info every 1 second
            self._last_dashboard_update = current_time
            return True
        return False

    def _should_update_dashboard_step(self) -> bool:
        """Check if we should update dashboard for this step (reduced frequency during training)"""
        # Always update on reset, termination, or truncation
        # During training, update every N steps or every few seconds
        self._step_update_counter += 1

        current_time = time.time()
        time_based_update = current_time - self._last_dashboard_update >= self._dashboard_update_interval
        step_based_update = self._step_update_counter % self._dashboard_update_frequency == 0

        if time_based_update or step_based_update:
            self._last_dashboard_update = current_time
            return True
        return False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[
        Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if not self.primary_asset or not self.market_simulator or \
                not self.portfolio_manager or not self.feature_extractor or \
                not self.reward_calculator or not self.execution_manager:
            logging.error("Session not properly set up. Call `setup_session(symbol, start, end)` before `reset()`.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Session not set up. Call setup_session first.",
                               "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        # FIXED: Reset episode counters and state tracking
        self.current_step = 0
        self.invalid_action_count_episode = 0
        self.episode_total_reward = 0.0
        self._last_decoded_action = None
        self._step_update_counter = 0

        # FIXED: Reset current state tracking
        self.current_termination_reason = None
        self.is_terminated = False
        self.is_truncated = False

        # FIXED: Reset episode-wide action counts
        self.episode_action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
        self.action_debug_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
        self.step_count_for_debug = 0

        # FIXED: Increment episode number properly
        self.episode_number += 1

        logging.info(f"Resetting environment for episode {self.episode_number}")

        # Reset simulators
        self.execution_manager.reset(np_random_seed_source=self.np_random)

        market_reset_options = {'random_start': self.random_reset_within_session}
        if 'start_time_offset_seconds' in options:
            market_reset_options['start_time_offset_seconds'] = options['start_time_offset_seconds']

        initial_market_state = self.market_simulator.reset(options=market_reset_options)
        if initial_market_state is None or 'timestamp_utc' not in initial_market_state:
            logging.error(
                "Market simulator failed to reset or provide initial state. Check data availability for the session.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Market simulator reset failed.",
                               "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        current_sim_time = initial_market_state['timestamp_utc']

        self.portfolio_manager.reset(episode_start_timestamp=current_sim_time)
        self.initial_capital_for_session = self.portfolio_manager.initial_capital

        if hasattr(self.reward_calculator, 'reset'):
            self.reward_calculator.reset()
        if hasattr(self.feature_extractor, 'reset'):
            self.feature_extractor.reset()

        self._last_observation = self._get_observation(initial_market_state, current_sim_time)
        if self._last_observation is None:
            logging.error("Failed to get initial observation. Ensure sufficient data/lookback at episode start.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Initial observation failed.",
                               "termination_reason": TerminationReasonEnum.OBSERVATION_FAILURE.value}

        self._last_portfolio_state_before_action = self.portfolio_manager.get_portfolio_state(current_sim_time)
        initial_info = self._get_current_info(reward=0.0,
                                              current_portfolio_state_for_info=self._last_portfolio_state_before_action)

        # Start dashboard if using it
        if self.use_dashboard and self.dashboard and not self.dashboard._running:
            logging.info("Starting trading dashboard...")
            self.dashboard.start()
            logging.info("Trading dashboard started successfully")

        # FIXED: Always update dashboard on reset (important initial state)
        if self.use_dashboard and self.dashboard and self.dashboard._running:
            try:
                market_state = self._get_current_market_state_safe()
                self.dashboard.update_state(initial_info, market_state)
            except Exception as e:
                logging.error(f"Error updating dashboard on reset: {e}")

        logging.info(f"Environment reset complete for {self.primary_asset}. "
                     f"Agent Start Time: {current_sim_time}, "
                     f"Initial Equity: ${self.initial_capital_for_session:.2f}")
        return self._last_observation, initial_info

    def _get_dummy_observation(self) -> Dict[str, np.ndarray]:
        dummy_obs = {}
        if isinstance(self.observation_space, spaces.Dict):
            for key in self.observation_space.keys():
                space_item = self.observation_space[key]
                dummy_obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)
        else:
            logging.error("Observation space is not a gymnasium.spaces.Dict. Cannot create dummy observation.")
        return dummy_obs

    def _get_observation(self, market_state_now: Dict[str, Any], current_sim_time: datetime) -> Optional[
        Dict[str, np.ndarray]]:
        if market_state_now is None:
            logging.warning(f"Market state is None at {current_sim_time} during observation generation.")
            return None
        try:
            current_portfolio_state = self.portfolio_manager.get_portfolio_state(current_sim_time)
            market_features_dict = self.feature_extractor.extract_features()
            if market_features_dict is None:
                logging.warning(f"FeatureExtractor returned None at {current_sim_time}. Not enough data for lookbacks?")
                return None
        except Exception as e:
            logging.error(f"Error during feature extraction at {current_sim_time}: {e}")
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
                    logging.warning(f"NaN values detected in {key} features: {nan_count} values")
                    obs[key] = np.nan_to_num(arr, nan=0.0)

        if not isinstance(self.observation_space, spaces.Dict):
            logging.error("Observation space is not a gymnasium.spaces.Dict. Cannot validate observation.")
            return None

        for key in self.observation_space.keys():
            space_item = self.observation_space[key]
            if obs.get(key) is None:
                logging.error(f"Observation missing key '{key}'. Filling with zeros.")
                obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)

            if key == 'static' and obs[key].ndim == 1:
                obs[key] = obs[key].reshape(1, -1)

            if obs[key].shape != space_item.shape:
                logging.error(f"Shape mismatch for observation key '{key}'. "
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
            logging.error(f"Unexpected action type: {type(raw_action)}")
            action_type_idx, size_type_idx = 0, 0
            raw_action_list = [0, 0]

        # Ensure indices are integers and within valid range
        action_type_idx = int(action_type_idx) % len(self.action_types)
        size_type_idx = int(size_type_idx) % len(self.position_size_types)

        action_type = self.action_types[action_type_idx]
        size_type = self.position_size_types[size_type_idx]

        # FIXED: Track both debug counts and episode-wide counts
        self.action_debug_counts[action_type.name] += 1
        self.episode_action_counts[action_type.name] += 1
        self.step_count_for_debug += 1

        # FIXED: Reduced logging frequency to prevent spam
        if self.step_count_for_debug % 500 == 0:  # Every 500 steps instead of 200
            total_actions = sum(self.action_debug_counts.values())
            if total_actions > 0:
                buy_pct = (self.action_debug_counts["BUY"] / total_actions) * 100
                sell_pct = (self.action_debug_counts["SELL"] / total_actions) * 100
                hold_pct = (self.action_debug_counts["HOLD"] / total_actions) * 100
                logging.debug(  # FIXED: Changed to debug level
                    f"ACTION DISTRIBUTION (Last 500 steps): BUY {buy_pct:.1f}% | SELL {sell_pct:.1f}% | HOLD {hold_pct:.1f}%")

        return {
            "type": action_type,
            "size_enum": size_type,
            "size_float": size_type.value_float,
            "raw_action": raw_action_list,
            "invalid_reason": None
        }

    def _translate_agent_action_to_order(self, decoded_action: Dict[str, Any], portfolio_state: PortfolioState,
                                         market_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """FIXED: Translate agent action using FIXED dollar amounts instead of equity percentages"""
        action_type = decoded_action['type']
        size_float = decoded_action['size_float']

        if not self.primary_asset:
            decoded_action['invalid_reason'] = "Primary asset not set in environment."
            logging.error("Attempted to translate action to order but primary_asset is not set.")
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

        if ideal_ask <= 0 or ideal_bid <= 0 or ideal_ask <= ideal_bid:
            decoded_action['invalid_reason'] = f"Invalid BBO prices: Ask ${ideal_ask:.2f}, Bid ${ideal_bid:.2f}."
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
            # FIXED: Use FIXED dollar amount instead of equity percentage
            target_buy_value = self.default_position_value * size_float  # size_float is 0.25, 0.50, 0.75, 1.0

            # FIXED: Check available cash first, then apply risk limits
            available_cash = min(cash, target_buy_value)
            target_buy_value = min(available_cash, max_pos_value_abs)

            if target_buy_value > 10.0 and ideal_ask > 0.01:  # FIXED: Minimum $10 order size
                quantity_to_trade = target_buy_value / ideal_ask
                order_side = OrderSideEnum.BUY

                # FIXED: Handle covering short positions
                if current_pos_side == PositionSideEnum.SHORT:
                    quantity_to_trade += abs(current_qty)  # Add quantity needed to cover short

                # FIXED: Enhanced logging for position sizing clarity
                logging.info(f"🔧 BUY Order Calculation: "
                             f"action_size={decoded_action['size_enum'].name}, "
                             f"size_float={size_float:.2f}, "
                             f"base_value=${self.default_position_value:.2f}, "
                             f"target_value=${self.default_position_value * size_float:.2f}, "
                             f"available_cash=${cash:.2f}, "
                             f"final_target=${target_buy_value:.2f}, "
                             f"price=${ideal_ask:.4f}, "
                             f"quantity={quantity_to_trade:.2f}")
            else:
                if target_buy_value <= 10.0:
                    decoded_action[
                        'invalid_reason'] = f"Insufficient buying power: target=${target_buy_value:.2f} (min=$10.00)"
                else:
                    decoded_action['invalid_reason'] = f"Invalid ask price: ${ideal_ask:.4f}"

        elif action_type == ActionTypeEnum.SELL:
            if current_pos_side == PositionSideEnum.LONG and current_qty > 0:
                # FIXED: Sell percentage of current position (correct behavior)
                quantity_to_trade = size_float * current_qty
                order_side = OrderSideEnum.SELL

                logging.info(f"🔧 SELL Order (Close Long): "
                             f"action_size={decoded_action['size_enum'].name}, "
                             f"size_float={size_float:.2f}, "
                             f"current_qty={current_qty:.2f}, "
                             f"quantity_to_sell={quantity_to_trade:.2f}")

            elif allow_shorting:
                # FIXED: Use FIXED dollar amount for shorting (like buying)
                target_short_value = self.default_position_value * size_float
                target_short_value = min(target_short_value, max_pos_value_abs)

                if target_short_value > 10.0 and ideal_bid > 0.01:
                    quantity_to_trade = target_short_value / ideal_bid
                    order_side = OrderSideEnum.SELL

                    # FIXED: Handle covering long positions when initiating short
                    if current_pos_side == PositionSideEnum.LONG:
                        quantity_to_trade += current_qty  # Add quantity needed to close long

                    logging.info(f"🔧 SELL Order (Short): "
                                 f"action_size={decoded_action['size_enum'].name}, "
                                 f"size_float={size_float:.2f}, "
                                 f"base_value=${self.default_position_value:.2f}, "
                                 f"target_value=${target_short_value:.2f}, "
                                 f"price=${ideal_bid:.4f}, "
                                 f"quantity={quantity_to_trade:.2f}")
                else:
                    if target_short_value <= 10.0:
                        decoded_action[
                            'invalid_reason'] = f"Insufficient shorting power: target=${target_short_value:.2f} (min=$10.00)"
                    else:
                        decoded_action['invalid_reason'] = f"Invalid bid price: ${ideal_bid:.4f}"
            else:
                decoded_action[
                    'invalid_reason'] = "SELL action invalid: No long position to close and shorting is disabled."

        # FIXED: Validate final order parameters
        if quantity_to_trade > 0.01 and order_side is not None:  # FIXED: Minimum 0.01 shares
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

            # FIXED: Log successful order translation
            estimated_value = quantity_to_trade * (ideal_ask if order_side == OrderSideEnum.BUY else ideal_bid)
            logging.info(
                f"✅ Order Translated: {order_side.value} {quantity_to_trade:.2f} shares @ ${ideal_ask if order_side == OrderSideEnum.BUY else ideal_bid:.4f} (≈${estimated_value:.2f})")

            return order_params
        elif decoded_action['invalid_reason'] is None and action_type != ActionTypeEnum.HOLD:
            decoded_action[
                'invalid_reason'] = f"Calculated quantity too small: {quantity_to_trade:.6f} shares (min=0.01)"

        if decoded_action['invalid_reason']:
            self.invalid_action_count_episode += 1
            logging.warning(f"❌ Invalid Action: {decoded_action['invalid_reason']}")
        return None

    def _get_current_market_state_safe(self) -> Optional[Dict[str, Any]]:
        """Safely get current market state without throwing exceptions"""
        try:
            if self.market_simulator:
                return self.market_simulator.get_current_market_state()
        except Exception as e:
            logging.debug(f"Error getting market state: {e}")
        return None

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._last_observation is None or self.primary_asset is None:
            logging.error(
                "Step called with _last_observation as None or primary_asset not set. Resetting or critical error.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, 0.0, True, False, {
                "error": "Critical: _last_observation was None or primary_asset not set.",
                "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        self.current_step += 1
        market_state_at_decision = self.market_simulator.get_current_market_state()
        if market_state_at_decision is None or 'timestamp_utc' not in market_state_at_decision:
            logging.error("Market simulator returned None or invalid state for current market state. Terminating.")
            return self._last_observation, 0.0, True, False, {"error": "Market state unavailable at decision.",
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
                self.portfolio_manager.update_fill(fill)
                # FIXED: Log fills with more detail
                logging.info(
                    f"💰 FILL: {fill['order_side'].value} {fill['executed_quantity']:.2f} @ ${fill['executed_price']:.4f} "
                    f"(Value: ${fill['executed_quantity'] * fill['executed_price']:.2f}, "
                    f"Fees: ${fill['commission'] + fill['fees']:.4f})")

        time_for_pf_update_after_fill = fill_details_list[-1][
            'fill_timestamp'] if fill_details_list else current_sim_time_decision

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

        current_equity = portfolio_state_next_t['total_equity']
        if current_equity <= self.initial_capital_for_session * self.bankruptcy_threshold_factor:
            terminated = True
            termination_reason = TerminationReasonEnum.BANKRUPTCY
            logging.warning(f"Episode terminated: BANKRUPTCY (Equity: ${current_equity:.2f})")
        elif current_equity <= self.initial_capital_for_session * (1 - self.max_session_loss_percentage):
            terminated = True
            termination_reason = TerminationReasonEnum.MAX_LOSS_REACHED
            logging.warning(f"Episode terminated: MAX_LOSS_REACHED (Equity: ${current_equity:.2f})")

        if terminated_by_obs_failure and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.OBSERVATION_FAILURE
            logging.warning("Episode terminated: OBSERVATION_FAILURE")

        if not market_advanced and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.END_OF_SESSION_DATA
            logging.info("Episode terminated: END_OF_SESSION_DATA")

        if self.invalid_action_count_episode >= self.max_invalid_actions_per_episode and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.INVALID_ACTION_LIMIT_REACHED
            logging.warning(
                f"Episode terminated: INVALID_ACTION_LIMIT_REACHED ({self.invalid_action_count_episode} invalid actions)")

        if not terminated and self.current_step >= self.max_steps_per_episode:
            truncated = True
            logging.info(f"Episode truncated: MAX_STEPS_REACHED ({self.current_step} steps)")

        # FIXED: Update current state tracking
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

        # FIXED: More frequent dashboard updates with less throttling
        if self.use_dashboard and self.dashboard and self.dashboard._running:
            should_update = (
                    terminated or truncated or  # Always update on episode end
                    self._should_update_dashboard_step()  # More frequent updates during training
            )

            if should_update:
                try:
                    market_state = self._get_current_market_state_safe()
                    self.dashboard.update_state(info, market_state)
                except Exception as e:
                    logging.error(f"Error updating dashboard: {e}")

        if terminated or truncated:
            final_metrics = self.portfolio_manager.get_trader_vue_metrics()
            info['episode_summary'] = {
                "total_reward": self.episode_total_reward,
                "steps": self.current_step,
                "final_equity": portfolio_state_next_t['total_equity'],
                "session_realized_pnl_net": portfolio_state_next_t['realized_pnl_session'],
                "session_net_profit_equity_change": portfolio_state_next_t[
                                                        'total_equity'] - self.initial_capital_for_session,
                "session_total_commissions": portfolio_state_next_t['total_commissions_session'],
                "session_total_fees": portfolio_state_next_t['total_fees_session'],
                "session_total_slippage_cost": portfolio_state_next_t['total_slippage_cost_session'],
                "termination_reason": termination_reason.value if termination_reason else (
                    "TRUNCATED" if truncated else "UNKNOWN"),
                "invalid_actions_in_episode": self.invalid_action_count_episode,
                **final_metrics
            }

            # FIXED: Log final episode info with less verbosity
            total_actions = sum(self.episode_action_counts.values())
            if total_actions > 0:
                buy_pct = (self.episode_action_counts["BUY"] / total_actions) * 100
                sell_pct = (self.episode_action_counts["SELL"] / total_actions) * 100
                hold_pct = (self.episode_action_counts["HOLD"] / total_actions) * 100

                logging.info(
                    f"🏁 Episode {self.episode_number} END ({self.primary_asset}). "
                    f"Reason: {info['episode_summary']['termination_reason']}. "
                    f"Net P&L: ${info['episode_summary']['session_net_profit_equity_change']:.2f}. "
                    f"Reward: {self.episode_total_reward:.4f}. Steps: {self.current_step}. "
                    f"Actions: B{buy_pct:.0f}% S{sell_pct:.0f}% H{hold_pct:.0f}%")

        return observation_next_t, reward, terminated, truncated, info

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
            'invalid_action_in_step': bool(
                self._last_decoded_action.get('invalid_reason')) if self._last_decoded_action else False,
            'invalid_actions_total_episode': self.invalid_action_count_episode,
            'episode_action_counts': self.episode_action_counts.copy(),
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

    def render(self, info_dict: Optional[Dict[str, Any]] = None):
        """Render method - delegates to dashboard if using dashboard mode"""
        if self.use_dashboard and self.dashboard:
            # Dashboard handles its own rendering via Rich Live
            return

        # For other render modes, could implement basic console output here
        if self.render_mode in ['human', 'logs'] and info_dict:
            print(f"Step {info_dict.get('step', 'N/A')}: "
                  f"Reward {info_dict.get('reward_step', 0.0):.4f}, "
                  f"Equity ${info_dict.get('portfolio_equity', 0.0):.2f}")

    def close(self):
        # Stop dashboard if running
        if self.dashboard and self.dashboard._running:
            logging.info("Stopping trading dashboard...")
            self.dashboard.stop()
        if self.market_simulator and hasattr(self.market_simulator, 'close'):
            self.market_simulator.close()
        logging.info("TradingEnvironment closed.")