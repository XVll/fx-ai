import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd  # For pd.Timestamp
import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config.config import Config  # Assuming your Config class is here
from data.data_manager import DataManager
from envs.reward import RewardCalculator  # Ensure this imports types correctly
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
    BUY = 1  # Enter a new long position or flip from short to long
    SELL = 2  # Enter a new short position (if allowed) or flip from long to short / exit long


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
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"  # Truncation
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"
    SETUP_FAILURE = "SETUP_FAILURE"
    INVALID_ACTION_LIMIT_REACHED = "INVALID_ACTION_LIMIT_REACHED"


class TradingEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'logs', 'none'], 'render_fps': 10}

    def __init__(self, config: Config, data_manager: DataManager, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # --- Environment Configuration ---
        env_cfg = self.config.env
        self.primary_asset: Optional[str] = None

        self.max_steps_per_episode: int = env_cfg.max_steps
        self.random_reset_within_session: bool = env_cfg.random_reset
        self.max_session_loss_percentage: float = env_cfg.max_episode_loss_percent
        self.bankruptcy_threshold_factor: float = env_cfg.bankruptcy_threshold_factor
        self.max_invalid_actions_per_episode: int = env_cfg.max_invalid_actions_per_episode

        self.data_manager = data_manager
        self.market_simulator: Optional[MarketSimulator] = None
        self.execution_manager: Optional[ExecutionSimulator] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.reward_calculator: Optional[RewardCalculator] = None

        # --- Action Space ---
        self.action_types = list(ActionTypeEnum)
        self.position_size_types = list(PositionSizeTypeEnum)
        self.action_space = spaces.MultiDiscrete([len(self.action_types), len(self.position_size_types)])
        self.logger.info(f"Action space: {self.action_space} "
                         f"(ActionTypes: {[a.name for a in self.action_types]}, "
                         f"PositionSizes: {[s.name for s in self.position_size_types]})")

        # --- Observation Space (from config.model) ---
        model_cfg = self.config.model
        # Explicitly type self.observation_space to help linters
        self.observation_space: spaces.Dict = spaces.Dict({
            'hf': spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.hf_seq_len, model_cfg.hf_feat_dim), dtype=np.float32),
            'mf': spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.mf_seq_len, model_cfg.mf_feat_dim), dtype=np.float32),
            'lf': spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.lf_seq_len, model_cfg.lf_feat_dim), dtype=np.float32),
            'portfolio': spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.portfolio_seq_len, model_cfg.portfolio_feat_dim), dtype=np.float32),
            'static': spaces.Box(low=-np.inf, high=np.inf, shape=(1, model_cfg.static_feat_dim), dtype=np.float32),
        })
        self.logger.info(f"Observation space defined with shapes: "
                         f"HF({model_cfg.hf_seq_len},{model_cfg.hf_feat_dim}), "
                         f"MF({model_cfg.mf_seq_len},{model_cfg.mf_feat_dim}), "
                         f"LF({model_cfg.lf_seq_len},{model_cfg.lf_feat_dim}), "
                         f"Portfolio({model_cfg.portfolio_seq_len},{model_cfg.portfolio_feat_dim}), "
                         f"Static(1,{model_cfg.static_feat_dim})")

        # --- Episode State ---
        self.current_session_start_time_utc: Optional[datetime] = None
        self.current_session_end_time_utc: Optional[datetime] = None
        self.current_step: int = 0
        self.invalid_action_count_episode: int = 0
        self.episode_total_reward: float = 0.0
        self._last_observation: Optional[Dict[str, np.ndarray]] = None
        self._last_portfolio_state_before_action: Optional[PortfolioState] = None
        self._last_decoded_action: Optional[Dict[str, Any]] = None
        self.initial_capital_for_session: float = 0.0

        self.render_mode = env_cfg.render_mode

    def setup_session(self, symbol: str, start_time: Union[str, datetime], end_time: Union[str, datetime]):
        """Configures the environment for a specific trading session (e.g., one day)."""
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

        self.logger.info(f"Setting up session for symbol '{self.primary_asset}' "
                         f"from {self.current_session_start_time_utc} to {self.current_session_end_time_utc}.")

        if self.np_random is None:
            _, _ = super().reset(seed=None)

        self.market_simulator = MarketSimulator(
            symbol=self.primary_asset,
            data_manager=self.data_manager,
            market_config=self.config.simulation.market_config,
            model_config=self.config.model,
            mode=self.config.env.training_mode,
            np_random=self.np_random,
            start_time=self.current_session_start_time_utc,
            end_time=self.current_session_end_time_utc,
            logger=self.logger.getChild("MarketSim")
        )
        self.portfolio_manager = PortfolioManager(
            logger=self.logger.getChild("PortfolioMgr"),
            config=self.config,
            tradable_assets=[self.primary_asset]
        )
        self.feature_extractor = FeatureExtractor(
            symbol=self.primary_asset,
            market_simulator=self.market_simulator,
            config=self.config.model,
            logger=self.logger.getChild("FeatureExt")
        )
        self.reward_calculator = RewardCalculator(
            config=self.config,
            logger=self.logger.getChild("RewardCalc")
        )
        self.execution_manager = ExecutionSimulator(
            logger=self.logger.getChild("ExecSim"),
            config_exec=self.config.simulation.execution_config,
            np_random=self.np_random,
            market_simulator=self.market_simulator
        )
        self.logger.info("All simulators and managers initialized for the session.")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if not self.primary_asset or not self.market_simulator or \
                not self.portfolio_manager or not self.feature_extractor or \
                not self.reward_calculator or not self.execution_manager:
            self.logger.error("Session not properly set up. Call `setup_session(symbol, start, end)` before `reset()`.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Session not set up. Call setup_session first.", "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        self.current_step = 0
        self.invalid_action_count_episode = 0
        self.episode_total_reward = 0.0
        self._last_decoded_action = None

        self.execution_manager.reset(np_random_seed_source=self.np_random)

        market_reset_options = {'random_start': self.random_reset_within_session}
        if 'start_time_offset_seconds' in options:
            market_reset_options['start_time_offset_seconds'] = options['start_time_offset_seconds']

        initial_market_state = self.market_simulator.reset(options=market_reset_options)
        if initial_market_state is None or 'timestamp_utc' not in initial_market_state:
            self.logger.critical("Market simulator failed to reset or provide initial state. Check data availability for the session.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Market simulator reset failed.", "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        current_sim_time = initial_market_state['timestamp_utc']

        self.portfolio_manager.reset(episode_start_timestamp=current_sim_time)
        self.initial_capital_for_session = self.portfolio_manager.initial_capital
        if hasattr(self.reward_calculator, 'reset'): self.reward_calculator.reset()
        if hasattr(self.feature_extractor, 'reset'): self.feature_extractor.reset()

        if hasattr(self.market_simulator, 'raw_1d_bars') and self.market_simulator.raw_1d_bars_df is not None:
            if hasattr(self.feature_extractor, 'update_daily_levels'):
                self.feature_extractor.update_daily_levels(self.market_simulator.raw_1d_bars, current_sim_time)
        else:
            self.logger.warning("raw_1d_bars not available from market_simulator for feature_extractor.update_daily_levels.")

        self._last_observation = self._get_observation(initial_market_state, current_sim_time)
        if self._last_observation is None:
            self.logger.critical("Failed to get initial observation. Ensure sufficient data/lookback at episode start.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, {"error": "Initial observation failed.", "termination_reason": TerminationReasonEnum.OBSERVATION_FAILURE.value}

        self._last_portfolio_state_before_action = self.portfolio_manager.get_portfolio_state(current_sim_time)
        initial_info = self._get_current_info(reward=0.0, current_portfolio_state_for_info=self._last_portfolio_state_before_action)

        if self.render_mode in ['human', 'logs']: self.render(info_dict=initial_info)
        self.logger.info(
            f"Environment reset for {self.primary_asset}. Agent Start Time: {current_sim_time}, Initial Equity: ${self.initial_capital_for_session:.2f}")
        return self._last_observation, initial_info

    def _get_dummy_observation(self) -> Dict[str, np.ndarray]:
        dummy_obs = {}
        if isinstance(self.observation_space, spaces.Dict):  # Type guard
            for key in self.observation_space.keys():  # Iterate using keys()
                space_item = self.observation_space[key]  # Access sub-space by key
                dummy_obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)
        else:
            self.logger.error("Observation space is not a gymnasium.spaces.Dict. Cannot create dummy observation.")
            # Fallback: create based on a predefined structure if possible, or raise error
            # This part depends on how you want to handle an unexpected observation_space type
            # For now, returning an empty dict, which will likely cause issues downstream if this path is hit.
        return dummy_obs

    def _get_observation(self, market_state_now: Dict[str, Any], current_sim_time: datetime) -> Optional[Dict[str, np.ndarray]]:
        if market_state_now is None:
            self.logger.warning(f"Market state is None at {current_sim_time} during observation generation.")
            return None
        try:
            current_portfolio_state = self.portfolio_manager.get_portfolio_state(current_sim_time)
            market_features_dict = self.feature_extractor.extract_features()
            if market_features_dict is None:
                self.logger.warning(f"FeatureExtractor returned None at {current_sim_time}. Not enough data for lookbacks?")
                return None
        except Exception as e:
            self.logger.error(f"Error during feature extraction at {current_sim_time}: {e}", exc_info=True)
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
        # Todo : Diagnostic logging for NaN values, can be removed.
        for key, arr in obs.items():
            if arr is not None:
                nan_count = np.isnan(arr).sum()
                if nan_count > 0:
                    self.logger.warning(f"NaN values detected in {key} features: {nan_count} values")
                    # Log specific indices where NaNs occur to help with debugging
                    nan_indices = np.where(np.isnan(arr))
                    self.logger.debug(f"NaN indices in {key}: {nan_indices}")

                    # Replace NaNs with zeros for safety
                    obs[key] = np.nan_to_num(arr, nan=0.0)

        if not isinstance(self.observation_space, spaces.Dict):
            self.logger.error("Observation space is not a gymnasium.spaces.Dict. Cannot validate observation.")
            return None  # Or handle error appropriately

        for key in self.observation_space.keys():  # Iterate using keys()
            space_item = self.observation_space[key]  # Access sub-space by key
            if obs.get(key) is None:
                self.logger.error(f"Observation missing key '{key}'. Filling with zeros.")
                obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)

            if key == 'static' and obs[key].ndim == 1:
                obs[key] = obs[key].reshape(1, -1)

            if obs[key].shape != space_item.shape:
                self.logger.error(
                    f"Shape mismatch for observation key '{key}'. "
                    f"Expected {space_item.shape}, Got {obs[key].shape}."
                )
                return None
        return obs

    def _decode_action(self, raw_action) -> Dict[str, Any]:
        """
        Decode the agent's action into a structured format the environment can use.
        Now handles both numpy arrays and tuples.
        """
        # Extract action components, handling both tuples and arrays
        if isinstance(raw_action, (tuple, list)):
            action_type_idx, size_type_idx = raw_action
            raw_action_list = list(raw_action)  # Convert to list for consistent handling
        elif hasattr(raw_action, 'tolist'):  # NumPy array or PyTorch tensor
            action_type_idx, size_type_idx = raw_action
            raw_action_list = raw_action.tolist()
        else:
            self.logger.error(f"Unexpected action type: {type(raw_action)}")
            # Fallback to sensible defaults
            action_type_idx, size_type_idx = 0, 0  # HOLD action, smallest size
            raw_action_list = [0, 0]

        # Ensure indices are integers and within valid range
        action_type_idx = int(action_type_idx) % len(self.action_types)
        size_type_idx = int(size_type_idx) % len(self.position_size_types)

        action_type = self.action_types[action_type_idx]
        size_type = self.position_size_types[size_type_idx]

        return {
            "type": action_type,
            "size_enum": size_type,
            "size_float": size_type.value_float,
            "raw_action": raw_action_list,
            "invalid_reason": None
        }

    def _translate_agent_action_to_order(self, decoded_action: Dict[str, Any], portfolio_state: PortfolioState, market_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        action_type = decoded_action['type']
        size_float = decoded_action['size_float']

        if not self.primary_asset:
            decoded_action['invalid_reason'] = "Primary asset not set in environment."
            self.logger.error("Attempted to translate action to order but primary_asset is not set.")
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
                self.logger.warning(f"Invalid prices for order at {market_state.get('timestamp_utc', 'N/A')}. No trade.")
                self.invalid_action_count_episode += 1
                return None

        if ideal_ask <= 0 or ideal_bid <= 0 or ideal_ask <= ideal_bid:
            decoded_action['invalid_reason'] = f"Invalid BBO prices: Ask ${ideal_ask:.2f}, Bid ${ideal_bid:.2f}."
            self.logger.warning(f"Invalid BBO for order at {market_state.get('timestamp_utc', 'N/A')}: Ask {ideal_ask}, Bid {ideal_bid}.")
            self.invalid_action_count_episode += 1
            return None

        pos_data = portfolio_state['positions'].get(asset_id)
        if not pos_data:
            decoded_action['invalid_reason'] = f"Position data for asset {asset_id} not found."
            self.logger.error(f"Position data for asset {asset_id} not found in portfolio_state.")
            self.invalid_action_count_episode += 1
            return None

        current_qty = pos_data['quantity']
        current_pos_side = pos_data['current_side']
        cash = portfolio_state['cash']
        total_equity = portfolio_state['total_equity']

        max_pos_value_abs = total_equity * self.portfolio_manager.max_position_value_ratio
        available_buying_power = cash
        allow_shorting = self.portfolio_manager.allow_shorting
        default_pos_value = self.portfolio_manager.default_position_value

        quantity_to_trade = 0.0
        order_side: Optional[OrderSideEnum] = None

        if action_type == ActionTypeEnum.HOLD:
            return None

        elif action_type == ActionTypeEnum.BUY:
            target_buy_value = round(size_float * max(default_pos_value,cash) / 100)
            target_buy_value = min(target_buy_value, max_pos_value_abs, cash)
            if target_buy_value > 1e-9 and ideal_ask > 1e-9:
                quantity_to_trade = target_buy_value / ideal_ask
                order_side = OrderSideEnum.BUY
                if current_pos_side == PositionSideEnum.SHORT:
                    quantity_to_trade += current_qty
            else:
                decoded_action['invalid_reason'] = "Insufficient buying power or invalid price for BUY."

        elif action_type == ActionTypeEnum.SELL:
            if current_pos_side == PositionSideEnum.LONG:
                quantity_to_trade = size_float * current_qty / 100
                order_side = OrderSideEnum.SELL
            elif allow_shorting:
                target_short_value = size_float * total_equity / 100
                target_short_value = min(target_short_value, max_pos_value_abs)
                if target_short_value > 1e-9 and ideal_bid > 1e-9:
                    quantity_to_trade = target_short_value / ideal_bid
                    order_side = OrderSideEnum.SELL
                    if current_pos_side == PositionSideEnum.LONG:
                        quantity_to_trade += current_qty
                else:
                    decoded_action['invalid_reason'] = "Insufficient shorting power or invalid price for SELL (short)."
            else:
                decoded_action['invalid_reason'] = "SELL action invalid: Not holding long and shorting disallowed."


        if quantity_to_trade > 1e-9 and order_side is not None:
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
            decoded_action['invalid_reason'] = "Calculated quantity was zero or order side not determined."

        if decoded_action['invalid_reason']:
            self.invalid_action_count_episode += 1
        return None

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._last_observation is None or self.primary_asset is None:
            self.logger.critical("Step called with _last_observation as None or primary_asset not set. Resetting or critical error.")
            dummy_obs = self._get_dummy_observation()
            return dummy_obs, 0.0, True, False, {"error": "Critical: _last_observation was None or primary_asset not set.",
                                                 "termination_reason": TerminationReasonEnum.SETUP_FAILURE.value}

        self.current_step += 1
        market_state_at_decision = self.market_simulator.get_current_market_state()
        if market_state_at_decision is None or 'timestamp_utc' not in market_state_at_decision:
            self.logger.error("Market simulator returned None or invalid state for current market state. Terminating.")
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
        if prices_at_decision_time[self.primary_asset] <= 0.0:
            self.logger.warning(
                f"Could not determine a valid price for {self.primary_asset} at decision time {current_sim_time_decision} for portfolio valuation after fills.")

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
                self.logger.warning("Market advanced but get_current_market_state returned None or invalid state.")

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
            if prices_at_next_time[self.primary_asset] <= 0.0:
                self.logger.warning(f"Could not determine a valid price for {self.primary_asset} at next time {next_sim_time} for portfolio valuation.")
            self.portfolio_manager.update_market_value(prices_at_next_time, next_sim_time)

        portfolio_state_next_t = self.portfolio_manager.get_portfolio_state(next_sim_time or time_for_pf_update_after_fill)

        observation_next_t: Optional[Dict[str, np.ndarray]] = None
        terminated_by_obs_failure = False
        if market_state_next_t and next_sim_time:
            observation_next_t = self._get_observation(market_state_next_t, next_sim_time)
            if observation_next_t is None:
                self.logger.warning(f"Failed to get observation o_{{t+1}} at sim time {next_sim_time}. Using last valid observation and terminating.")
                observation_next_t = self._last_observation
                terminated_by_obs_failure = True
        else:
            self.logger.info(f"No further market state for new observation at step {self.current_step}. Using last valid observation.")
            observation_next_t = self._last_observation

        if observation_next_t is None:
            self.logger.error("Observation is critically None even after fallbacks. Using dummy observation.")
            observation_next_t = self._get_dummy_observation()
            if not terminated_by_obs_failure: terminated_by_obs_failure = True

        self._last_observation = observation_next_t

        terminated = False
        truncated = False
        termination_reason: Optional[TerminationReasonEnum] = None

        current_equity = portfolio_state_next_t['total_equity']
        if current_equity <= self.initial_capital_for_session * self.bankruptcy_threshold_factor:
            terminated = True
            termination_reason = TerminationReasonEnum.BANKRUPTCY
            self.logger.info(f"Episode terminated: Bankruptcy. Equity ${current_equity:.2f} <= Threshold.")
        elif current_equity <= self.initial_capital_for_session * (1 - self.max_session_loss_percentage):
            terminated = True
            termination_reason = TerminationReasonEnum.MAX_LOSS_REACHED
            self.logger.info(f"Episode terminated: Max session loss. Equity ${current_equity:.2f}")

        if terminated_by_obs_failure and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.OBSERVATION_FAILURE

        if not market_advanced and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.END_OF_SESSION_DATA
            self.logger.info(f"Episode terminated: End of market data at step {self.current_step}.")

        if self.invalid_action_count_episode >= self.max_invalid_actions_per_episode and not terminated:
            terminated = True
            termination_reason = TerminationReasonEnum.INVALID_ACTION_LIMIT_REACHED
            self.logger.info(f"Episode terminated: Reached max invalid actions ({self.invalid_action_count_episode}).")

        if not terminated and self.current_step >= self.max_steps_per_episode:
            truncated = True
            self.logger.info(f"Episode truncated: MAX_STEPS_REACHED ({self.current_step}).")

        reward = self.reward_calculator.calculate(
            portfolio_state_before_action=self._last_portfolio_state_before_action,
            portfolio_state_after_action_fills=portfolio_state_after_action_fills,
            portfolio_state_next_t=portfolio_state_next_t,
            market_state_at_decision=market_state_at_decision, market_state_next_t=market_state_next_t,
            decoded_action=self._last_decoded_action, fill_details_list=fill_details_list,
            terminated=terminated, truncated=truncated, termination_reason=termination_reason
        )
        self.episode_total_reward += reward

        info = self._get_current_info(
            reward=reward, fill_details_list=fill_details_list,
            current_portfolio_state_for_info=portfolio_state_next_t,
            termination_reason_enum=termination_reason,
            is_terminated=terminated, is_truncated=truncated
        )

        if terminated or truncated:
            final_metrics = self.portfolio_manager.get_trader_vue_metrics()
            info['episode_summary'] = {
                "total_reward": self.episode_total_reward, "steps": self.current_step,
                "final_equity": portfolio_state_next_t['total_equity'],
                "session_realized_pnl_net": portfolio_state_next_t['realized_pnl_session'],
                "session_net_profit_equity_change": portfolio_state_next_t['total_equity'] - self.initial_capital_for_session,
                "session_total_commissions": portfolio_state_next_t['total_commissions_session'],
                "session_total_fees": portfolio_state_next_t['total_fees_session'],
                "session_total_slippage_cost": portfolio_state_next_t['total_slippage_cost_session'],
                "termination_reason": termination_reason.value if termination_reason else ("TRUNCATED" if truncated else "UNKNOWN"),
                "invalid_actions_in_episode": self.invalid_action_count_episode,
                **final_metrics
            }
            self.logger.info(f"EPISODE END ({self.primary_asset}). Reason: {info['episode_summary']['termination_reason']}. "
                             f"Net Profit (Equity Change): ${info['episode_summary']['session_net_profit_equity_change']:.2f}. "
                             f"Total Reward: {self.episode_total_reward:.4f}. Steps: {self.current_step}.")

        if self.render_mode in ['human', 'logs'] and (self.current_step % self.config.env.render_interval == 0 or terminated or truncated):
            self.render(info_dict=info)

        return observation_next_t, reward, terminated, truncated, info

    def _get_current_info(self, reward: float, current_portfolio_state_for_info: PortfolioState,
                          fill_details_list: Optional[List[FillDetails]] = None,
                          termination_reason_enum: Optional[TerminationReasonEnum] = None,
                          is_terminated: bool = False, is_truncated: bool = False) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            'timestamp_iso': current_portfolio_state_for_info['timestamp'].isoformat(),
            'step': self.current_step, 'reward_step': reward,
            'episode_cumulative_reward': self.episode_total_reward,
            'action_decoded': self._last_decoded_action,
            'fills_step': fill_details_list if fill_details_list else [],
            'portfolio_equity': current_portfolio_state_for_info['total_equity'],
            'portfolio_cash': current_portfolio_state_for_info['cash'],
            'portfolio_unrealized_pnl': current_portfolio_state_for_info['unrealized_pnl'],
            'portfolio_realized_pnl_session_net': current_portfolio_state_for_info['realized_pnl_session'],
            'invalid_action_in_step': bool(self._last_decoded_action.get('invalid_reason')) if self._last_decoded_action else False,
            'invalid_actions_total_episode': self.invalid_action_count_episode,
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
        """
        Enhanced rendering method using Rich library for terminal-friendly display.
        Fills are integrated into the main panel with color coding and consistent sizing.

        Args:
            info_dict: Dictionary containing step information
        """
        if self.render_mode not in ['human', 'logs'] or info_dict is None or not self.primary_asset:
            return

        console = Console()

        try:
            # Create a consistent layout with fixed heights
            layout_grid = Table.grid(expand=True)

            # Top row: Header with step info and main indicators
            header = Table.grid(padding=(0, 1))

            # Step info in a stylish format
            timestamp_iso = info_dict.get('timestamp_iso')
            time_str = 'N/A'
            if timestamp_iso and isinstance(timestamp_iso, str):
                try:
                    time_str = timestamp_iso.split('T')[1].split('.')[0]
                except IndexError:
                    time_str = 'N/A'  # Handle cases where split doesn't work as expected

            step_info = Text.assemble(
                ("STEP ", "bold cyan"),
                (f"{info_dict.get('step', 'N/A')}", "cyan"),
                " | ",
                ("TIME ", "bold cyan"),
                (time_str, "cyan")
            )

            # Current price and PnL indicators (key metrics)
            current_price = None
            if self.market_simulator:  # Check if market_simulator exists
                try:
                    market_state = self.market_simulator.get_current_market_state()
                    if market_state:  # Ensure market_state is not None
                        current_price = market_state.get('current_price')
                        if current_price is None:
                            current_price = market_state.get('best_bid_price') or market_state.get('best_ask_price')
                except Exception as e:
                    self.logger.warning(f"Error getting market state: {e}")
            else:
                self.logger.warning("Market simulator not available for price lookup.")

            unreal_pnl = info_dict.get('portfolio_unrealized_pnl', 0.0)
            real_pnl = info_dict.get('portfolio_realized_pnl_session_net', 0.0)

            price_text = Text.assemble(
                ("PRICE ", "bold yellow"),
                (f"${current_price:.2f}" if current_price is not None else "N/A", "yellow"),
                " | ",
                ("UNREAL PNL ", "bold"),
                (f"${unreal_pnl:.2f}", "green" if unreal_pnl > 0 else "red" if unreal_pnl < 0 else "white"),
                " | ",
                ("REAL PNL ", "bold"),
                (f"${real_pnl:.2f}", "green" if real_pnl > 0 else "red" if real_pnl < 0 else "white")
            )

            header.add_row(step_info, price_text)
            layout_grid.add_row(Panel(header, border_style="cyan", padding=(0, 0)))

            # Main content in a 2x2 grid
            content_grid = Table.grid(expand=True)

            # 1. Action panel
            action_panel = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            action_text_display = 'N/A'
            size_info = 'N/A'
            invalid_reason_text = ""  # Default to empty string

            decoded_action = info_dict.get('action_decoded', {})  # Ensure it's a dict, or empty dict
            if not isinstance(decoded_action, dict):  # Additional safety
                decoded_action = {}

            try:
                action_type_obj = decoded_action.get('type')
                if action_type_obj:
                    if hasattr(action_type_obj, 'name'):
                        name_val = action_type_obj.name
                        action_text_display = str(name_val) if name_val is not None else 'N/A'
                    else:
                        action_text_display = str(action_type_obj)

                size_enum = decoded_action.get('size_enum')
                if size_enum:
                    size_name_val = getattr(size_enum, 'name', None)  # Get name if exists
                    size_name = str(size_name_val) if size_name_val is not None else str(size_enum)
                    size_pct = decoded_action.get('size_float', 0.0) * 100
                    size_info = f"{size_name} ({size_pct:.0f}%)"

                raw_invalid_reason = decoded_action.get('invalid_reason')
                invalid_reason_text = str(raw_invalid_reason) if raw_invalid_reason is not None else ""

            except Exception as e:
                self.logger.warning(f"Error processing action info: {e}")

            action_panel.add_row("Type", Text(action_text_display, style="bold magenta"))
            action_panel.add_row("Size", Text(size_info, style="magenta"))
            action_panel.add_row("Invalid", Text(invalid_reason_text, style="bold red" if invalid_reason_text else ""))

            # 2. Position panel
            pos_panel = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            pos_qty = info_dict.get(f'position_{self.primary_asset}_qty', 0.0)
            pos_side = info_dict.get(f'position_{self.primary_asset}_side', 'FLAT')
            pos_avg_entry = info_dict.get(f'position_{self.primary_asset}_avg_entry', 0.0)

            price_diff = 0.0
            price_diff_pct = 0.0
            if current_price is not None and pos_qty != 0 and pos_avg_entry != 0:  # Check pos_qty != 0 and pos_avg_entry != 0
                price_diff = current_price - pos_avg_entry
                price_diff_pct = (price_diff / pos_avg_entry) * 100

            pos_side_text = str(pos_side) if pos_side is not None else 'N/A'

            pos_panel.add_row("Symbol", Text(f"{self.primary_asset} {pos_side_text}", style="bold blue"))
            pos_panel.add_row("Quantity", Text(f"{pos_qty:.2f}", style="blue"))
            pos_panel.add_row("Avg Entry", Text(f"${pos_avg_entry:.2f}", style="blue"))

            diff_style = "green" if price_diff > 0 else "red" if price_diff < 0 else ""
            diff_text = f"${price_diff:.2f} ({price_diff_pct:+.2f}%)" if pos_qty != 0 and current_price is not None else "N/A"
            pos_panel.add_row("P/L vs Entry", Text(diff_text, style=diff_style))

            # 3. Portfolio panel
            portfolio_panel = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            portfolio_equity = info_dict.get('portfolio_equity', 0.0)
            portfolio_cash = info_dict.get('portfolio_cash', 0.0)
            reward_step = info_dict.get('reward_step', 0.0)
            episode_cumulative_reward = info_dict.get('episode_cumulative_reward', 0.0)

            portfolio_panel.add_row("Total Equity", Text(f"${portfolio_equity:.2f}", style="bold green"))
            portfolio_panel.add_row("Cash", Text(f"${portfolio_cash:.2f}", style="green"))
            portfolio_panel.add_row("Step Reward", Text(f"{reward_step:.4f}",
                                                        style="green" if reward_step > 0 else "red" if reward_step < 0 else ""))
            portfolio_panel.add_row("Total Reward", Text(f"{episode_cumulative_reward:.4f}",
                                                         style="bold green" if episode_cumulative_reward > 0 else "bold red" if episode_cumulative_reward < 0 else "bold"))

            # 4. Fills panel
            fills_panel = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            fills_info = info_dict.get('fills_step', [])
            if not isinstance(fills_info, list):  # Ensure fills_info is a list
                fills_info = []

            if not fills_info:
                fills_panel.add_row("Status", Text("No fills this step", style="dim"))
                fills_panel.add_row("", "")  # Empty row for spacing
                fills_panel.add_row("", "")  # Empty row for spacing
                fills_panel.add_row("", "")  # Empty row for spacing
            else:
                # Assuming OrderSideEnum is imported (e.g., from simulators.portfolio_simulator import OrderSideEnum)
                # and is available in the current scope.
                # The line "OrderSideEnum = self.OrderSideEnum" was removed to address the AttributeError.
                # Ensure `OrderSideEnum` is properly imported at the module level.

                total_commission = 0.0
                total_fees = 0.0
                total_value = 0.0
                fills_display_texts = []

                for fill in fills_info:
                    if not isinstance(fill, dict):  # Skip if fill is not a dict
                        self.logger.warning(f"Skipping non-dict fill item: {fill}")
                        continue

                    side = fill.get('order_side')
                    # Handle both string and enum comparison for side
                    # This now relies on OrderSideEnum being available from an import.
                    is_buy = (isinstance(side, str) and side.upper() == OrderSideEnum.BUY) or \
                             (hasattr(side, 'value') and side.value == OrderSideEnum.BUY) or \
                             (side == OrderSideEnum.BUY)  # Direct enum comparison

                    color = "green" if is_buy else "red"
                    side_text = "BUY" if is_buy else "SELL"

                    qty = fill.get('executed_quantity', 0.0)
                    price = fill.get('executed_price', 0.0)
                    commission = fill.get('commission', 0.0)
                    fees = fill.get('fees', 0.0)

                    # Ensure numeric types before calculation
                    if not all(isinstance(v, (int, float)) for v in [qty, price, commission, fees]):
                        self.logger.warning(f"Non-numeric value in fill data: {fill}")
                        continue  # Skip this fill if data is not numeric

                    value = qty * price
                    total_commission += commission
                    total_fees += fees
                    total_value += value

                    fills_display_texts.append(f"[{color}]{side_text}[/{color}] {qty:.2f} @ ${price:.2f}")

                for i, text_markup in enumerate(fills_display_texts):
                    fills_panel.add_row("Trade" if i == 0 and fills_display_texts else "", Text.from_markup(text_markup))

                if not fills_display_texts:
                    fills_panel.add_row("Status", Text("No valid fills this step", style="dim"))
                    for _ in range(3): fills_panel.add_row("", "")

                if fills_display_texts or total_commission or total_fees or total_value:
                    fills_panel.add_row("Commission", Text(f"${total_commission:.2f}", style="red"))
                    fills_panel.add_row("Fees", Text(f"${total_fees:.2f}", style="red"))
                    fills_panel.add_row("Value", Text(f"${total_value:.2f}", style="bold"))

                min_fill_rows = 4
                while fills_panel.row_count < min_fill_rows and fills_display_texts:
                    fills_panel.add_row("", "")

            # Add all panels to the content grid
            content_grid.add_row(
                Panel(action_panel, title="[bold]Action", border_style="magenta", padding=(0, 0)),
                Panel(pos_panel, title="[bold]Position", border_style="blue", padding=(0, 0))
            )
            content_grid.add_row(
                Panel(portfolio_panel, title="[bold]Portfolio", border_style="green", padding=(0, 0)),
                Panel(fills_panel, title="[bold]Fills", border_style="yellow", padding=(0, 0))
            )

            # Add content grid to main layout
            layout_grid.add_row(content_grid)

            # Add footer for status messages
            footer_text_content = ""
            termination_reason = info_dict.get('termination_reason')
            time_limit_truncated = info_dict.get('TimeLimit.truncated')

            if termination_reason:
                footer_text_content = f"TERMINATED: {str(termination_reason)}"
            elif time_limit_truncated:
                footer_text_content = "TRUNCATED by Max Steps"

            if footer_text_content:
                style = "bold red" if "TERMINATED" in footer_text_content else "bold yellow"
                layout_grid.add_row(Panel(
                    Text(footer_text_content, style=style),
                    border_style="red" if "TERMINATED" in footer_text_content else "yellow",
                    padding=(0, 0)
                ))

            # Render the complete layout
            console.print(Panel(
                layout_grid,
                title=f"[bold]Trading Environment: {str(self.primary_asset)}[/bold]",
                border_style="white",
                padding=(0, 1)
            ))

        except Exception as e:
            try:
                console.print(Panel(f"Rendering Error: {str(e)}\nTraceback available in logs.",
                                    title="[bold red]Trading Environment - Render Error[/bold red]",
                                    style="bold red",
                                    border_style="red"))
                self.logger.exception("Critical rendering error in render method:")
            except Exception as fallback_e:
                print(f"CRITICAL RENDERING ERROR: {e}")
                print(f"FALLBACK RENDERER FAILED: {fallback_e}")
                import traceback
                traceback.print_exc()

    def close(self):
        if self.market_simulator and hasattr(self.market_simulator, 'close'):
            self.market_simulator.close()
        self.logger.info("TradingEnvironment closed.")
