import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional, Callable, TypedDict, Type
import numpy as np
import pandas as pd  # For pd.Timestamp
import gymnasium as gym  # type: ignore
from gymnasium import spaces  # type: ignore

import config.config
from config.config import Config
from data.data_manager import DataManager
from envs.reward import RewardCalculator
from feature.feature_extractor import FeatureExtractor
from simulators.execution_simulator import ExecutionSimulator
from simulators.market_simulator import MarketSimulator, TrainingMode
from simulators.portfolio_simulator import PortfolioManager, PortfolioState, OrderTypeEnum, OrderSideEnum, PositionSideEnum, FillDetails


class ActionTypeEnumForEnv(Enum):  # Example, align with your strategy
    HOLD = 0
    BUY_MARKET_SMALL = 1
    BUY_MARKET_LARGE = 2
    SELL_MARKET_SMALL = 3
    SELL_MARKET_LARGE = 4
    EXIT_ALL = 5


class TerminationReasonEnumForEnv(Enum):  # For info dict
    END_OF_SESSION_DATA = "END_OF_SESSION_DATA"
    MAX_LOSS_REACHED = "MAX_LOSS_REACHED"
    BANKRUPTCY = "BANKRUPTCY"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"  # Truncation
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"
    # Add other reasons


class TradingEnvironment(gym.Env):
    metadata = {'render_modes': ['human', 'logs', 'none'], 'render_fps': 10}

    def __init__(self, config: Config, data_manager: DataManager, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.tradable_assets = self.config['env'].get('tradable_assets', ['DEFAULT_ASSET'])
        if not self.tradable_assets: raise ValueError("tradable_assets must be configured in env config.")
        self.primary_asset = self.tradable_assets[0]

        self.data_manager = data_manager
        self.market_simulator: Optional[MarketSimulator] = None
        self.execution_manager: Optional[ExecutionSimulator] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.reward_calculator: Optional[RewardCalculator] = None

        # --- Action Space (Example: discrete actions) ---
        self.action_enums = list(ActionTypeEnumForEnv)
        self.action_space = spaces.Discrete(len(self.action_enums))
        self.logger.info(f"Action space: {self.action_space} (Actions: {[a.name for a in self.action_enums]})")

        # --- Observation Space (from config.model) ---
        m_cfg = self.config.model
        self.observation_space = spaces.Dict({
            'hf': spaces.Box(low=-np.inf, high=np.inf, shape=(m_cfg.hf_seq_len, m_cfg.hf_feat_dim), dtype=np.float32),
            'mf': spaces.Box(low=-np.inf, high=np.inf, shape=(m_cfg.mf_seq_len, m_cfg.mf_feat_dim), dtype=np.float32),
            'lf': spaces.Box(low=-np.inf, high=np.inf, shape=(m_cfg.lf_seq_len, m_cfg.lf_feat_dim), dtype=np.float32),
            'portfolio': spaces.Box(low=-np.inf, high=np.inf, shape=(m_cfg.portfolio_seq_len, m_cfg.portfolio_feat_dim), dtype=np.float32),
            'static': spaces.Box(low=-np.inf, high=np.inf, shape=(1, m_cfg.static_feat_dim), dtype=np.float32),
        })
        self.logger.info(f"Observation space: {self.observation_space}")

        # Episode state

        self.current_episode_start_time: Optional[datetime] = None
        self.current_episode_end_time: Optional[datetime] = None
        self.current_step: int = 0
        self.episode_total_reward: float = 0.0
        self._last_observation: Optional[Dict[str, np.ndarray]] = None
        self._last_portfolio_state_before_action: Optional[PortfolioState] = None
        self._last_decoded_action: Optional[Dict[str, Any]] = None

        self.render_mode = self.config.env.render_mode

    def setup_session(self, symbol: str, start_time: datetime, end_time: datetime):
        """Configures the environment for a specific trading session (e.g., one day)."""
        self.primary_asset = symbol  # Update primary asset if it changes per session
        if symbol not in self.tradable_assets: self.tradable_assets.append(symbol)  # Ensure it's tracked

        self.current_episode_start_time = pd.Timestamp(start_time).tz_localize('UTC').to_pydatetime() \
            if start_time.tzinfo is None else start_time.astimezone(timezone.utc)
        self.current_episode_end_time = pd.Timestamp(end_time).tz_localize('UTC').to_pydatetime() \
            if end_time.tzinfo is None else end_time.astimezone(timezone.utc)

        self.market_simulator = MarketSimulator(
            symbol=self.primary_asset,
            data_manager=self.data_manager,  # If your provider uses it
            config=self.config['simulation'].get('market_config', {}),
            mode=self.config['simulation'].get('market_mode', 'backtesting'),
            start_time=self.current_episode_start_time,
            end_time=self.current_episode_end_time,
            logger=self.logger.getChild("MarketSim")
        )
        self.portfolio_manager = PortfolioManager(self.logger, self.config, self.tradable_assets)
        self.feature_extractor = FeatureExtractor(symbol, self.config.model, self.logger)
        self.reward_calculator = RewardCalculator(self.config)
        self.execution_manager = ExecutionSimulator(self.logger, self.config.simulation.execution_config, self.np_random, self.market_simulator)

        self.logger.info(f"Session configured for {self.primary_asset} from {self.current_episode_start_time} to {self.current_episode_end_time}.")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)  # Important for self.np_random seeding via Gymnasium
        options = options or {}

        if not self.market_simulator or not self.current_episode_start_time:
            raise RuntimeError("Market simulator or session not set up. Call `setup_session()` before `reset()`.")

        self.current_step = 0
        self.episode_total_reward = 0.0
        self._last_decoded_action = None

        # Reset managers, propagating the (potentially new) RNG
        if hasattr(self.execution_manager, 'reset'): self.execution_manager.reset(np_random_seed_source=self.np_random)  # type: ignore

        # MarketSim reset options (e.g., random start within a session)
        market_reset_options = {'random_start': self.config.env.random_reset}
        if options.get('start_time_offset_seconds'):  # For curriculum
            market_reset_options['start_time_offset_seconds'] = options['start_time_offset_seconds']  # type: ignore

        initial_market_state = self.market_simulator.reset(options=market_reset_options)
        current_sim_time = initial_market_state['timestamp_utc']

        self.portfolio_manager.reset(episode_start_timestamp=current_sim_time)
        if hasattr(self.reward_calculator, 'reset'): self.reward_calculator.reset()  # type: ignore
        if hasattr(self.feature_extractor, 'reset'): self.feature_extractor.reset()  # type: ignore

        self._last_observation = self._get_observation(initial_market_state, current_sim_time)
        if self._last_observation is None:
            self.logger.critical("Failed to get initial observation. Ensure sufficient data/lookback at episode start.")
            # Return a dummy observation and mark for immediate termination if this happens
            dummy_obs_shape = self.observation_space['portfolio'].shape  # type: ignore
            self._last_observation = {
                'market_hf': np.zeros(self.observation_space['market_hf'].shape, dtype=np.float32),  # type: ignore
                'portfolio': np.zeros(dummy_obs_shape, dtype=np.float32)
            }
            # This scenario implies a critical setup error that should ideally not occur.
            # Or, the environment needs a mechanism to "fast-forward" until a valid observation can be made.
            raise RuntimeError("Initial observation failed. Check data and feature extraction settings.")

        self._last_portfolio_state_before_action = self.portfolio_manager.get_portfolio_state(current_sim_time)
        initial_info = self._get_current_info(reward=0.0, current_portfolio_state_for_info=self._last_portfolio_state_before_action)

        if self.render_mode in ['human', 'logs']: self.render(info_dict=initial_info)
        self.logger.info(f"Environment reset. Agent Start Time: {current_sim_time}, Initial Equity: ${self.portfolio_manager.initial_capital:.2f}")
        return self._last_observation, initial_info

    def _get_observation(self, market_state_now: Dict[str, Any], current_sim_time: datetime) -> Optional[Dict[str, np.ndarray]]:
        try:
            # Market features from FeatureExtractor
            market_features_dict = self.feature_extractor.extract_features(current_market_state=market_state_now)  # type: ignore
            if market_features_dict is None:
                self.logger.warning(f"FeatureExtractor returned None at {current_sim_time}. Not enough data for lookbacks?")
                return None
        except Exception as e:
            self.logger.error(f"Error during feature extraction at {current_sim_time}: {e}", exc_info=True)
            return None

        # Portfolio features from PortfolioManager
        portfolio_obs_component = self.portfolio_manager.get_portfolio_observation()  # PortfolioObservationFeatures
        portfolio_features_array = portfolio_obs_component['features']

        # Construct the full observation dictionary
        observation = {
            # Assuming market_hf is the primary key for market features for this example
            'market_hf': market_features_dict.get('hf_features', np.zeros(self.observation_space['market_hf'].shape, dtype=np.float32)),  # type: ignore
            # Add other market feature keys ('mf_features', etc.) as defined in observation_space
            'portfolio': portfolio_features_array
        }

        # Validate shapes (crucial for model input)
        for key, space in self.observation_space.spaces.items():  # type: ignore
            if key not in observation:
                self.logger.error(f"Observation missing key '{key}' defined in observation_space.")
                return None
            if observation[key].shape != space.shape:
                self.logger.error(
                    f"Shape mismatch for observation key '{key}'. "
                    f"Expected {space.shape}, Got {observation[key].shape}. "
                    f"Check FeatureExtractor/PortfolioManager output shapes against config.model."
                )
                return None
        return observation

    def _decode_action(self, raw_action: int) -> Dict[str, Any]:  # Assuming discrete action space
        selected_action_enum = self.action_enums[raw_action]
        # Example: Can add more details like size if action space is more complex (e.g., Tuple)
        return {"type": selected_action_enum, "raw_action": raw_action}

    def _translate_agent_action_to_order(self,
                                         decoded_action: Dict[str, Any],
                                         portfolio_state_before_action: PortfolioState,
                                         market_state_at_decision: Dict[str, Any]
                                         ) -> Optional[Dict[str, Any]]:
        """Translates agent's abstract action into concrete order parameters."""
        action_type = decoded_action['type']
        asset_id = self.primary_asset  # Assuming single asset trading for this translation logic

        # Ideal prices from market state at decision time (t)
        # For simplicity, using 'current_price' as a basis if bid/ask not directly available
        ideal_ask = market_state_at_decision.get('ask_price', market_state_at_decision.get('current_price', 0) * 1.0001)
        ideal_bid = market_state_at_decision.get('bid_price', market_state_at_decision.get('current_price', 0) * 0.9999)
        if ideal_ask <= 0 or ideal_bid <= 0:
            self.logger.warning(f"Invalid ideal prices ({ideal_ask}, {ideal_bid}) for order at {market_state_at_decision['timestamp_utc']}. No trade.")
            decoded_action['invalid_reason'] = "Missing or invalid market prices for order"
            return None

        order_params: Optional[Dict[str, Any]] = None
        current_pos_data = portfolio_state_before_action['positions'][asset_id]
        current_qty = current_pos_data['quantity']

        # Example sizing: 25% of current equity for BUY_MARKET_LARGE
        # More sophisticated sizing would come from config or action space itself
        equity = portfolio_state_before_action['total_equity']
        cash = portfolio_state_before_action['cash']

        qty_small = (equity * 0.10) / ideal_ask if ideal_ask > 0 else 0
        qty_large = (equity * 0.25) / ideal_ask if ideal_ask > 0 else 0
        qty_small = min(qty_small, cash / ideal_ask if ideal_ask > 0 else 0)  # Cap by cash
        qty_large = min(qty_large, cash / ideal_ask if ideal_ask > 0 else 0)  # Cap by cash

        if action_type == ActionTypeEnumForEnv.HOLD:
            return None
        elif action_type == ActionTypeEnumForEnv.BUY_MARKET_SMALL and qty_small > 1e-9:
            order_params = {'asset_id': asset_id, 'order_type': OrderTypeEnum.MARKET, 'order_side': OrderSideEnum.BUY,
                            'quantity': qty_small, 'ideal_price_ask': ideal_ask, 'ideal_price_bid': ideal_bid}
        elif action_type == ActionTypeEnumForEnv.BUY_MARKET_LARGE and qty_large > 1e-9:
            order_params = {'asset_id': asset_id, 'order_type': OrderTypeEnum.MARKET, 'order_side': OrderSideEnum.BUY,
                            'quantity': qty_large, 'ideal_price_ask': ideal_ask, 'ideal_price_bid': ideal_bid}
        elif action_type == ActionTypeEnumForEnv.SELL_MARKET_SMALL and current_qty > 1e-9:
            order_params = {'asset_id': asset_id, 'order_type': OrderTypeEnum.MARKET, 'order_side': OrderSideEnum.SELL,
                            'quantity': min(current_qty, qty_small), 'ideal_price_ask': ideal_ask, 'ideal_price_bid': ideal_bid}  # Sell a portion
        elif action_type == ActionTypeEnumForEnv.SELL_MARKET_LARGE and current_qty > 1e-9:
            order_params = {'asset_id': asset_id, 'order_type': OrderTypeEnum.MARKET, 'order_side': OrderSideEnum.SELL,
                            'quantity': min(current_qty, qty_large), 'ideal_price_ask': ideal_ask, 'ideal_price_bid': ideal_bid}
        elif action_type == ActionTypeEnumForEnv.EXIT_ALL and current_qty > 1e-9:
            order_params = {'asset_id': asset_id, 'order_type': OrderTypeEnum.MARKET,
                            'order_side': OrderSideEnum.SELL if current_pos_data['current_side'] == PositionSideEnum.LONG else OrderSideEnum.BUY,
                            'quantity': current_qty, 'ideal_price_ask': ideal_ask, 'ideal_price_bid': ideal_bid}


        if order_params: decoded_action['translated_order'] = order_params.copy()  # Log the translated order
        return order_params

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._last_observation is None:  # Should be caught by reset, defensive check
            self.logger.critical("Step called with _last_observation as None. This indicates a severe issue.")
            # Return a dummy observation and terminate
            dummy_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.observation_space.spaces.items()}  # type: ignore
            return dummy_obs, 0.0, True, False, {"error": "Critical: _last_observation was None.",
                                                 "termination_reason": TerminationReasonEnumForEnv.OBSERVATION_FAILURE.value}

        self.current_step += 1

        # 1. Get market state at decision time (m_t)
        market_state_at_decision = self.market_simulator.get_current_market_state()  # type: ignore
        if market_state_at_decision is None:  # Should not happen if is_done() is checked
            self.logger.error("Market simulator returned None for current state. Terminating.")
            return self._last_observation, 0.0, True, False, {"error": "Market state unavailable at decision.",
                                                              "termination_reason": TerminationReasonEnumForEnv.OBSERVATION_FAILURE.value}
        current_sim_time_decision = market_state_at_decision['timestamp_utc']

        # 2. Store portfolio state before action (p_t)
        self._last_portfolio_state_before_action = self.portfolio_manager.get_portfolio_state(current_sim_time_decision)

        # 3. Decode action and translate to order request
        self._last_decoded_action = self._decode_action(action)
        order_request = self._translate_agent_action_to_order(
            self._last_decoded_action,
            self._last_portfolio_state_before_action,
            market_state_at_decision
        )

        # 4. Execute order (if any) and update portfolio with fills
        fill_details_list: List[FillDetails] = []
        if order_request:
            # Pass ideal prices from decision time to ExecutionManager
            fill = self.execution_manager.execute_order(  # type: ignore
                asset_id=order_request['asset_id'],
                order_type=order_request['order_type'],
                order_side=order_request['order_side'],
                requested_quantity=order_request['quantity'],
                ideal_price_ask=order_request['ideal_price_ask'],  # From m_t
                ideal_price_bid=order_request['ideal_price_bid'],  # From m_t
                current_market_timestamp=current_sim_time_decision  # Time of decision
            )
            if fill:
                fill_details_list.append(fill)
                self.portfolio_manager.update_fill(fill)

        # 5. Portfolio state after fills, market values based on m_t (p'_t)
        # Use the latest fill timestamp if fills occurred, else current decision time.
        time_for_pf_update_after_fill = fill_details_list[-1]['fill_timestamp'] if fill_details_list else current_sim_time_decision
        # Market prices for this valuation are from m_t because market hasn't moved yet.
        prices_at_decision_time = {self.primary_asset: market_state_at_decision.get('current_price', 0.0)}
        self.portfolio_manager.update_market_value(prices_at_decision_time, time_for_pf_update_after_fill)
        portfolio_state_after_action_fills = self.portfolio_manager.get_portfolio_state(time_for_pf_update_after_fill)

        # 6. Advance Market Simulator to next state (m_{t+1})
        market_advanced = self.market_simulator.step()  # type: ignore
        market_state_next_t: Optional[Dict[str, Any]] = None
        next_sim_time: Optional[datetime] = None

        if market_advanced:
            market_state_next_t = self.market_simulator.get_current_market_state()  # type: ignore
            if market_state_next_t:
                next_sim_time = market_state_next_t['timestamp_utc']
            else:  # Should ideally not happen if market_advanced is True unless at the very end
                market_advanced = False  # Treat as end of data

        # 7. Update Portfolio Market Value with new prices from m_{t+1} (results in p_{t+1})
        # This is crucial for updating MFE/MAE for open trades with the latest market price.
        if market_state_next_t and next_sim_time:
            prices_at_next_time = {self.primary_asset: market_state_next_t.get('current_price', 0.0)}
            self.portfolio_manager.update_market_value(prices_at_next_time, next_sim_time)
        # If market didn't advance, portfolio features for obs will be based on last known state (p'_t)
        portfolio_state_next_t = self.portfolio_manager.get_portfolio_state(next_sim_time or time_for_pf_update_after_fill)

        # 8. Get New Observation (o_{t+1})
        observation_next_t: Optional[Dict[str, np.ndarray]] = None
        terminated_by_obs_failure = False
        if market_state_next_t and next_sim_time:  # If market advanced and new state is valid
            observation_next_t = self._get_observation(market_state_next_t, next_sim_time)
            if observation_next_t is None:
                self.logger.warning(f"Failed to get observation o_{{t+1}} at sim time {next_sim_time}. Terminating.")
                observation_next_t = self._last_observation  # Fallback to last valid observation
                terminated_by_obs_failure = True
        else:  # Market didn't advance or next state is None (end of data for observation)
            self.logger.info(f"No further market state for new observation at step {self.current_step}.")
            observation_next_t = self._last_observation  # Use last observation

        if observation_next_t is None:  # Should only happen if _last_observation was also None (e.g. reset failed)
            dummy_obs_shape = self.observation_space['portfolio'].shape  # type: ignore
            observation_next_t = {
                'market_hf': np.zeros(self.observation_space['market_hf'].shape, dtype=np.float32),  # type: ignore
                'portfolio': np.zeros(dummy_obs_shape, dtype=np.float32)
            }
            self.logger.error("Observation is critically None even after fallbacks.")
            if not terminated_by_obs_failure: terminated_by_obs_failure = True

        self._last_observation = observation_next_t  # Store o_{t+1}

        # 9. Check Terminations & Truncations
        terminated = False
        truncated = False
        termination_reason: Optional[TerminationReasonEnumForEnv] = None

        if terminated_by_obs_failure:
            terminated = True
            termination_reason = TerminationReasonEnumForEnv.OBSERVATION_FAILURE
        elif not market_advanced or (self.market_simulator and self.market_simulator.is_done()):  # type: ignore
            terminated = True
            termination_reason = TerminationReasonEnumForEnv.END_OF_SESSION_DATA
            self.logger.info(f"Episode terminated: End of market data at step {self.current_step}.")
        # Add other termination checks (bankruptcy, max loss) using PortfolioManager
        # Example:
        # elif self.portfolio_manager.check_max_loss(self.config['env'].get('max_loss_percentage_session', 0.2)):
        #     terminated = True; termination_reason = TerminationReasonEnumForEnv.MAX_LOSS_REACHED

        if not terminated and self.current_step >= self.config.env.max_steps:
            truncated = True
            termination_reason = TerminationReasonEnumForEnv.MAX_STEPS_REACHED
            self.logger.info(f"Episode truncated: MAX_STEPS_REACHED ({self.current_step}).")

        # 10. Calculate Reward (r_t)
        reward = self.reward_calculator.calculate(  # type: ignore
            portfolio_state_before_action=self._last_portfolio_state_before_action,  # p_t
            portfolio_state_after_action_fills=portfolio_state_after_action_fills,  # p'_t
            portfolio_state_next_t=portfolio_state_next_t,  # p_{t+1}
            market_state_at_decision=market_state_at_decision,
            market_state_next_t=market_state_next_t,  # Can be None if at end
            decoded_action=self._last_decoded_action,
            fill_details_list=fill_details_list,
            terminated=terminated, truncated=truncated, termination_reason=termination_reason
        )
        self.episode_total_reward += reward

        # 11. Get Info for this step (info_t)
        info = self._get_current_info(
            reward=reward,
            fill_details_list=fill_details_list,
            current_portfolio_state_for_info=portfolio_state_next_t,  # Log state p_{t+1}
            termination_reason_enum=termination_reason,
            is_terminated=terminated, is_truncated=truncated
        )

        if terminated or truncated:
            final_metrics = self.portfolio_manager.get_trader_vue_metrics()
            info['episode_summary'] = {
                "total_reward": self.episode_total_reward,
                "steps": self.current_step,
                "final_equity": portfolio_state_next_t['total_equity'],
                "session_realized_pnl_net": portfolio_state_next_t['realized_pnl_session'],
                "session_net_profit_equity_change": portfolio_state_next_t['total_equity'] - self.portfolio_manager.initial_capital,
                # Session totals from PortfolioState (which are from PortfolioManager)
                "session_total_commissions": portfolio_state_next_t['total_commissions_session'],
                "session_total_fees": portfolio_state_next_t['total_fees_session'],
                "session_total_slippage_cost": portfolio_state_next_t['total_slippage_cost_session'],
                "session_total_volume_traded": portfolio_state_next_t['total_volume_traded_session'],
                "session_total_turnover": portfolio_state_next_t['total_turnover_session'],
                "termination_reason": termination_reason.value if termination_reason else ("TRUNCATED" if truncated else "UNKNOWN"),
                **final_metrics  # Spread all detailed TraderVue metrics
            }
            self.logger.info(
                f"EPISODE END. Reason: {info['episode_summary']['termination_reason']}. Net Profit (Equity Change): ${info['episode_summary']['session_net_profit_equity_change']:.2f}")

        if self.render_mode in ['human', 'logs'] and (self.current_step % 10 == 0 or terminated or truncated):
            self.render(info_dict=info)

        return observation_next_t, reward, terminated, truncated, info

    def _get_current_info(self, reward: float, current_portfolio_state_for_info: PortfolioState,
                          fill_details_list: Optional[List[FillDetails]] = None,
                          termination_reason_enum: Optional[TerminationReasonEnumForEnv] = None,
                          is_terminated: bool = False, is_truncated: bool = False) -> Dict[str, Any]:
        """Helper to construct the info dictionary for logging for the current step t."""
        info: Dict[str, Any] = {
            'timestamp_iso': current_portfolio_state_for_info['timestamp'].isoformat(),
            'step': self.current_step,
            'reward_step': reward,
            'episode_cumulative_reward': self.episode_total_reward,
            'action_decoded': self._last_decoded_action,
            'fills_step': fill_details_list if fill_details_list else [],
            # Log key portfolio state items for step-wise W&B tracking if needed
            'portfolio_equity': current_portfolio_state_for_info['total_equity'],
            'portfolio_cash': current_portfolio_state_for_info['cash'],
            'portfolio_unrealized_pnl': current_portfolio_state_for_info['unrealized_pnl'],
            'portfolio_realized_pnl_session_net': current_portfolio_state_for_info['realized_pnl_session'],
        }
        # Add current position details for the primary asset
        pos_detail = current_portfolio_state_for_info['positions'].get(self.primary_asset, {})
        info[f'position_{self.primary_asset}_qty'] = pos_detail.get('quantity', 0.0)
        info[f'position_{self.primary_asset}_side'] = pos_detail.get('current_side', PositionSideEnum.FLAT).value

        if self._last_decoded_action and self._last_decoded_action.get('invalid_reason'):
            info['action_invalid_reason'] = self._last_decoded_action['invalid_reason']
        if is_terminated and termination_reason_enum:
            info['termination_reason'] = termination_reason_enum.value
        if is_truncated:
            info['TimeLimit.truncated'] = True  # Standard Gymnasium key for SB3 compatibility
        return info

    def render(self, info_dict: Optional[Dict[str, Any]] = None):
        if self.render_mode == 'none' or info_dict is None: return

        log_lines = [f"--- Step: {info_dict.get('step', self.current_step)} | Time: {info_dict.get('timestamp_iso', 'N/A')} ---"]
        if 'action_decoded' in info_dict and info_dict['action_decoded']:
            log_lines.append(f"Action: {info_dict['action_decoded'].get('type', {}).get('name', 'N/A')}")
            if 'translated_order' in info_dict['action_decoded']: log_lines.append(f"  Order: {info_dict['action_decoded']['translated_order']}")
            if 'invalid_reason' in info_dict['action_decoded']: log_lines.append(f"  Invalid: {info_dict['action_decoded']['invalid_reason']}")
        if 'fills_step' in info_dict and info_dict['fills_step']:
            for fill in info_dict['fills_step']: log_lines.append(
                f"  Fill: {fill['order_side'].value} {fill['executed_quantity']:.2f}@{fill['executed_price']:.2f} Comm:{fill['commission']:.2f} Fees:{fill['fees']:.2f}")

        log_lines.append(
            f"Portfolio: Equity ${info_dict.get('portfolio_equity', 0):.2f}, Cash ${info_dict.get('portfolio_cash', 0):.2f}, UnrealPnL ${info_dict.get('portfolio_unrealized_pnl', 0):.2f}")
        log_lines.append(f"Reward: Step {info_dict.get('reward_step', 0):.4f}, Episode Total {info_dict.get('episode_cumulative_reward', 0):.4f}")

        if info_dict.get('termination_reason'): log_lines.append(f"TERMINATED: {info_dict['termination_reason']}")
        if info_dict.get('TimeLimit.truncated'): log_lines.append("TRUNCATED by Max Steps")

        output_str = "\n".join(log_lines)
        if self.render_mode == 'human':
            print(output_str)
        elif self.render_mode == 'logs':
            self.logger.info(output_str)

        if (info_dict.get('termination_reason') or info_dict.get('TimeLimit.truncated')) and 'episode_summary' in info_dict:
            summary_log = ["--- EPISODE SUMMARY ---"]
            for k, v in info_dict['episode_summary'].items():
                summary_log.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            if self.render_mode == 'human':
                print("\n".join(summary_log))
            elif self.render_mode == 'logs':
                self.logger.info("\n".join(summary_log))

    def close(self):
        if self.market_simulator and hasattr(self.market_simulator, 'close'):
            self.market_simulator.close()
        self.logger.info("TradingEnvironment closed.")
