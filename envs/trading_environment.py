"""
Trading Environment V2 - Clean Stateless Architecture

A stateless trading environment that coordinates simulators without managing
training state or episode progression. Designed for clean separation of concerns.

Key principles:
- No state about training progress
- Simple coordinator role only
- Delegates all data concerns to simulators
- Minimal interface for training
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
from numpy.typing import NDArray

from config import Config
from data import DataManager
from simulators import MarketSimulator, ExecutionSimulator, PortfolioSimulator
from simulators.portfolio_simulator import PortfolioState, FillDetails
from rewards.calculator import RewardSystem
from core.types import Symbol
from .action_mask import ActionMask


@dataclass
class ResetPoint:
    """Typed reset point information for episode setup."""
    timestamp: pd.Timestamp
    quality_score: float = 0.0
    roc_score: float = 0.0
    activity_score: float = 0.0
    price: float = 0.0
    index: int = 0


class TradingEnvironment(gym.Env):
    """
    Clean stateless trading environment for V2 architecture.
    
    This environment:
    - Coordinates simulators for market, execution, and portfolio
    - Provides gymnasium-compatible interface
    - Has NO knowledge of training progress or episode management
    - Delegates all data/caching concerns to MarketSimulator
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
            self,
            config: Config,
            data_manager: DataManager,
            callback_manager=None,
    ):
        """Initialize the environment with minimal dependencies."""
        super().__init__()

        self.config = config
        self.data_manager = data_manager
        self.callback_manager = callback_manager
        self.logger = logging.getLogger(__name__)

        # Simulators (initialized when the environment is set up)
        self.market_simulator: Optional[MarketSimulator] = None
        self.execution_simulator: Optional[ExecutionSimulator] = None
        self.portfolio_simulator: Optional[PortfolioSimulator] = None
        self.reward_calculator: Optional[RewardSystem] = None
        self.action_mask: Optional[ActionMask] = None

        # Current episode info (minimal tracking)
        self.symbol: Optional[Symbol] = None
        self.current_date: Optional[datetime] = None

        # Define action and observation spaces
        self._setup_spaces()

        self.logger.info("TradingEnvironment initialized (stateless)")

    def _setup_spaces(self):
        """Set up gymnasium action and observation spaces."""
        action_dim = self.config.model.action_dim
        self.action_space = spaces.Discrete(action_dim[0] * action_dim[1])

        # Observation space: multi-branch features
        model_cfg = self.config.model
        self.observation_space = spaces.Dict({
            "hf": spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.hf_seq_len, model_cfg.hf_feat_dim), dtype=np.float32),
            "mf": spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.mf_seq_len, model_cfg.mf_feat_dim), dtype=np.float32),
            "lf": spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.lf_seq_len, model_cfg.lf_feat_dim), dtype=np.float32),
            "portfolio": spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.portfolio_seq_len, model_cfg.portfolio_feat_dim), dtype=np.float32),
        })

    def initialize(self, symbol: str, date: datetime) -> bool:
        """
        Initialize environment for a trading session (symbol, date).
        
        Called ONCE when training starts. Sets up simulators and loads day data.
        Heavy operations happen here (data loading, cache preparation).
        
        Args:
            symbol: Trading symbol
            date: Trading date
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.symbol = symbol
            self.current_date = date

            self.logger.info(f"Initializing environment: {symbol} {date.strftime('%Y-%m-%d')}")

            # Initialize market simulator
            self.market_simulator = MarketSimulator(
                symbol=symbol,
                data_manager=self.data_manager,
                model_config=self.config.model,
                simulation_config=self.config.simulation,
                logger=self.logger
            )

            # Initialize day data (heavy operation)
            if not self.market_simulator.initialize_day(date):
                self.logger.error(f"Failed to initialize market data for {date}")
                return False

            # Initialize other simulators
            self.execution_simulator = ExecutionSimulator(
                simulation_config=self.config.simulation,
                market_simulator=self.market_simulator,
                logger=self.logger,
                np_random=self.np_random
            )

            self.portfolio_simulator = PortfolioSimulator(
                env_config=self.config.env,
                simulation_config=self.config.simulation,
                model_config=self.config.model,
                tradable_assets=[symbol],
                logger=self.logger
            )

            self.reward_calculator = RewardSystem(
                config=self.config,
                callback_manager=self.callback_manager,
                logger=self.logger
            )

            self.action_mask = ActionMask(
                config=self.config.simulation,
                logger=self.logger
            )

            # Get time range info
            start_time, end_time = self.market_simulator.get_time_range()
            self.logger.info(
                f"Environment initialized: {start_time.strftime('%H:%M:%S')} - "
                f"{end_time.strftime('%H:%M:%S')} ({len(self.market_simulator.df_market_state)} timesteps)"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}")
            return False

    def reset(
            self,
            symbol: str,
            date: datetime,
            reset_point: ResetPoint,
            seed: Optional[int] = None
    ) -> Tuple[Dict[str, NDArray], Dict[str, Any]]:
        """
        Reset environment for a new episode with episode setup and initialization.
        
        This method handles both day initialization (if needed) and episode reset
        in one place, providing a clean interface for the training manager.
        
        Args:
            symbol: Trading symbol
            date: Trading date
            reset_point: Reset point information with timestamp and metadata
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (observation, info)
        """
        try:
            # Set random seed if provided
            if seed is not None:
                super().reset(seed=seed)

            # Check if we need to initialize for a new day
            needs_initialization = (
                    not self.market_simulator or
                    self.symbol != symbol or
                    self.current_date != date
            )

            if needs_initialization:
                self.logger.info(f"Initializing environment for new session: {symbol} {date.strftime('%Y-%m-%d')}")

                # Initialize environment for this day (heavy operation)
                if not self.initialize(symbol, date):
                    raise RuntimeError(f"Failed to initialize environment for {symbol} {date}")

                self.logger.debug(f"✅ Environment initialized for {symbol} {date}")

            # Validate reset point
            if not reset_point:
                raise ValueError("reset_point is required")

            if not self.market_simulator:
                raise RuntimeError("Environment not initialized")

            reset_timestamp = reset_point.timestamp

            # Set market simulator to reset point (fast operation)
            if not self.market_simulator.set_time(reset_timestamp):
                raise RuntimeError(f"Failed to set time to {reset_timestamp}")

            # Reset simulators to this timestamp
            self.execution_simulator.reset(np_random_seed_source=self.np_random)
            self.portfolio_simulator.reset(session_start=reset_timestamp)
            self.reward_calculator.reset()

            # Get initial observation
            observation = self._get_observation()

            # Convert timestamp string to datetime for isoformat if needed
            if isinstance(reset_timestamp, str):
                from core.utils.time_utils import to_datetime
                reset_timestamp_dt = to_datetime(reset_timestamp)
                reset_timestamp_iso = reset_timestamp_dt.isoformat() if reset_timestamp_dt else reset_timestamp
            else:
                reset_timestamp_iso = reset_timestamp.isoformat()

            # Build comprehensive info
            info = {
                'symbol': str(self.symbol),
                'date': self.current_date.strftime('%Y-%m-%d'),
                'reset_timestamp': reset_timestamp_iso,
                'reset_point': {
                    'timestamp': reset_point.timestamp,
                    'quality_score': reset_point.quality_score,
                    'roc_score': reset_point.roc_score,
                    'activity_score': reset_point.activity_score,
                    'price': reset_point.price,
                    'index': reset_point.index
                },
                'setup_type': 'new_session' if needs_initialization else 'episode_reset',
                'initialized': needs_initialization
            }

            return observation, info

        except Exception as e:
            self.logger.error(f"Failed to setup episode: {e}")
            raise

    def step(self, action: Union[int, NDArray[np.int32]]) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute (linear index 0-11)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current market state
        market_state = self.market_simulator.get_current_market_data()

        if not market_state:
            # End of data
            obs = self._get_observation()
            return obs, 0.0, True, False, {'termination_reason': 'no_market_data'}

        current_time = market_state["timestamp"]

        # Get portfolio state before action
        portfolio_before = self.portfolio_simulator.get_portfolio_state(current_time)

        # Decode and validate action
        action = self._validate_action(action, portfolio_before, market_state)

        # Execute action through execution simulator
        execution_result = self.execution_simulator.execute_action(
            raw_action=action,
            market_state=market_state,
            portfolio_state=portfolio_before,
            primary_asset=self.symbol,
            portfolio_manager=self.portfolio_simulator
        )

        # Process fills if any
        fill_details = []
        if execution_result.fill_details:
            enriched_fill = self.portfolio_simulator.process_fill(
                execution_result.fill_details
            )
            fill_details.append(enriched_fill)

        # Update portfolio with current prices
        self._update_portfolio(market_state, fill_details)
        portfolio_after = self.portfolio_simulator.get_portfolio_state(current_time)

        # Advance market one step
        if not self.market_simulator.step():
            # End of data
            obs = self._get_observation()
            reward = self._calculate_terminal_reward(portfolio_after)
            return obs, reward, True, False, {'termination_reason': 'end_of_data'}

        # Get next market state and update portfolio
        next_market_state = self.market_simulator.get_current_market_data()
        if next_market_state:
            self._update_portfolio(next_market_state, [])
            portfolio_next = self.portfolio_simulator.get_portfolio_state(
                next_market_state["timestamp"]
            )
        else:
            portfolio_next = portfolio_after

        # Check termination conditions
        terminated, term_reason = self._check_termination(portfolio_next)

        # Calculate reward
        reward = self.reward_calculator.calculate(
            portfolio_state_before_action=portfolio_before,
            portfolio_state_after_action_fills=portfolio_after,
            portfolio_state_next_t=portfolio_next,
            market_state_at_decision=market_state,
            market_state_next_t=next_market_state,
            decoded_action=execution_result.action_decode_result,
            fill_details_list=fill_details,
            terminated=terminated,
            truncated=False,
            termination_reason=term_reason
        )

        # Get next observation
        observation = self._get_observation()

        # Build minimal info
        info = {
            'timestamp': current_time.isoformat(),
            'action_taken': execution_result.action_decode_result.to_dict()
        }
        if terminated:
            info['termination_reason'] = term_reason

        return observation, reward, terminated, False, info

    def _get_observation(self) -> Dict[str, NDArray]:
        """Get current observation from simulators."""
        # Get features from market simulator
        features = self.market_simulator.get_current_features()
        if features is None:
            raise ValueError("No features available from market simulator")

        # Get portfolio observation
        portfolio_obs = self.portfolio_simulator.get_portfolio_observation()

        # Build observation dict
        obs_dict = {
            "hf": features.get("hf"),
            "mf": features.get("mf"),
            "lf": features.get("lf"),
            "portfolio": portfolio_obs["features"]
        }

        return obs_dict

    def _validate_action(self, action: Union[int, NDArray[np.int32]], portfolio_state: PortfolioState, market_state: Dict[str, Any]) -> int:
        """Validate and potentially override invalid actions."""
        # Convert to int
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        # Validate bounds
        if not 0 <= action < 12:
            self.logger.warning(f"Invalid action {action}, defaulting to HOLD")
            return 0  # HOLD_25

        # Apply action masking if enabled
        if self.config.simulation.use_action_masking and self.action_mask:
            mask = self.action_mask.get_action_mask(portfolio_state, market_state)
            if not mask[action]:
                # Find first valid action (prefer HOLD actions)
                for hold_action in [0, 1, 2, 3]:  # HOLD actions
                    if mask[hold_action]:
                        return hold_action
                # If no HOLD valid, find any valid action
                for i in range(12):
                    if mask[i]:
                        return i
                # Should not reach here if mask is properly constructed
                return 0

        return action

    def _update_portfolio(self, market_state: Dict[str, Any], fill_details: list[FillDetails]):
        """Update portfolio with current market prices."""
        timestamp = market_state["timestamp"]

        # Extract price
        current_price = market_state.get("current_price", 0.0)
        if current_price <= 0:
            # Try mid price
            bid = market_state.get("best_bid", 0)
            ask = market_state.get("best_ask", 0)
            if bid > 0 and ask > 0:
                current_price = (bid + ask) / 2

        if current_price > 0:
            prices = {self.symbol: current_price}
            self.portfolio_simulator.update_market_values(prices, timestamp)

    def _check_termination(self, portfolio_state: PortfolioState) -> Tuple[bool, Optional[str]]:
        """Check basic termination conditions."""
        equity = portfolio_state.get("total_equity", 0.0)
        initial = self.portfolio_simulator.initial_capital

        # Bankruptcy
        if equity <= initial * 0.1:  # 90% loss
            return True, "bankruptcy"

        # Max loss
        if equity <= initial * 0.75:  # 25% loss
            return True, "max_loss"

        return False, None

    def _calculate_terminal_reward(self, portfolio_state: PortfolioState) -> float:
        """Calculate reward for terminal state."""
        # Simple PnL-based terminal reward
        equity = portfolio_state.get("total_equity", 0.0)
        initial = self.portfolio_simulator.initial_capital
        pnl = equity - initial
        return pnl / initial  # Normalized PnL

    def get_action_mask(self) -> Optional[NDArray[np.bool_]]:
        """Get current valid action mask."""
        if not self.action_mask or not self.market_simulator:
            return None

        current_time = self.market_simulator.df_market_state.index[self.market_simulator.current_index]
        portfolio_state = self.portfolio_simulator.get_portfolio_state(current_time)
        market_state = self.market_simulator.get_current_market_data()

        if not portfolio_state or not market_state:
            return None

        return self.action_mask.get_action_mask(portfolio_state, market_state)

    def close(self):
        pass
