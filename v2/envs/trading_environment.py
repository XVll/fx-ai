"""
V2 Trading Environment - Clean Architecture Implementation

This environment focuses purely on single episode execution:
- Market simulation and feature extraction
- Portfolio management and order execution
- Reward calculation
- State observation generation

Episode management, day selection, and reset point cycling are handled
by EpisodeManager in the v2 architecture.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime, time, timedelta
from dataclasses import dataclass

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from numpy.typing import NDArray

from data.data_manager import DataManager
from simulators.market_simulator import MarketSimulator
from simulators.execution_simulator import ExecutionSimulator
from simulators.portfolio_simulator import (
    PortfolioSimulator, 
    PortfolioState,
    FillDetails,
)
from rewards.calculator import RewardSystem

from v2.config import Config
from v2.core.types import (
    Symbol, Timestamp, TerminationReasonEnum,
    ActionType, PositionSizeType, ObservationArray
)
from v2.callbacks import CallbackManager
from .action_mask import ActionMask


@dataclass
class EpisodeConfig:
    """Configuration for a single episode."""
    symbol: Symbol
    date: datetime
    start_time: datetime
    end_time: datetime
    max_steps: Optional[int] = None
    reset_point_info: Optional[Dict[str, Any]] = None


class TradingEnvironment(gym.Env):
    """
    V2 Trading Environment - Clean single episode execution.
    
    Design principles:
    - Focus on single episode mechanics
    - Clean separation from episode management
    - Type-safe interfaces
    - Efficient state management
    - Clear termination logic
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        config: Config,
        data_manager: DataManager,
        callback_manager: Optional[CallbackManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize environment with clean dependencies."""
        super().__init__()
        
        self.config = config
        self.data_manager = data_manager
        self.callback_manager = callback_manager
        self.logger = logger or logging.getLogger(f"{__name__}.TradingEnvironment")
        
        # Core components (initialized per episode)
        self.market_simulator: Optional[MarketSimulator] = None
        self.execution_simulator: Optional[ExecutionSimulator] = None
        self.portfolio_simulator: Optional[PortfolioSimulator] = None
        self.reward_system: Optional[RewardSystem] = None
        self.action_mask: Optional[ActionMask] = None
        
        # Episode configuration
        self.episode_config: Optional[EpisodeConfig] = None
        
        # Episode state
        self.current_step: int = 0
        self.episode_reward: float = 0.0
        self._last_observation: Optional[Dict[str, ObservationArray]] = None
        self._current_state: Optional[Dict[str, ObservationArray]] = None
        
        # Define action and observation spaces
        self._setup_spaces()
        
        self.logger.info("🏗️ V2 TradingEnvironment initialized")
    
    def _setup_spaces(self):
        """Setup gymnasium action and observation spaces."""
        # Action space: [action_type, position_size]
        self.action_space = spaces.MultiDiscrete([3, 4])
        
        # Observation space: multi-branch features
        model_cfg = self.config.model
        self.observation_space = spaces.Dict({
            "hf": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(model_cfg.hf_seq_len, model_cfg.hf_feat_dim),
                dtype=np.float32
            ),
            "mf": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(model_cfg.mf_seq_len, model_cfg.mf_feat_dim),
                dtype=np.float32
            ),
            "lf": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(model_cfg.lf_seq_len, model_cfg.lf_feat_dim),
                dtype=np.float32
            ),
            "portfolio": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(model_cfg.portfolio_seq_len, model_cfg.portfolio_feat_dim),
                dtype=np.float32
            ),
        })
    
    def setup_episode(self, episode_config: EpisodeConfig) -> bool:
        """
        Setup environment for a new episode.
        
        This replaces the old setup_session method with cleaner separation.
        Episode selection is handled by EpisodeManager.
        
        Args:
            episode_config: Configuration for the episode
            
        Returns:
            True if setup successful, False otherwise
        """
        self.episode_config = episode_config
        self.logger.info(
            f"🎯 Setting up episode: {episode_config.symbol} on "
            f"{episode_config.date.strftime('%Y-%m-%d')} "
            f"[{episode_config.start_time.strftime('%H:%M:%S')} - "
            f"{episode_config.end_time.strftime('%H:%M:%S')}]"
        )
        
        # Initialize market simulator
        self.market_simulator = MarketSimulator(
            symbol=episode_config.symbol,
            data_manager=self.data_manager,
            model_config=self.config.model,
            simulation_config=self.config.simulation,
        )
        
        # Initialize day data
        if not self.market_simulator.initialize_day(episode_config.date):
            self.logger.error(f"Failed to initialize market data for {episode_config.date}")
            return False
        
        # Initialize other simulators
        self._initialize_simulators()
        
        # Log episode setup info
        stats = self.market_simulator.get_stats()
        self.logger.info(
            f"✅ Episode ready: {stats['total_seconds']} seconds of data, "
            f"warmup: {stats['warmup_info']['has_warmup']}"
        )
        
        return True
    
    def _initialize_simulators(self):
        """Initialize all simulator components for the episode."""
        if self.np_random is None:
            self.reset(seed=None)
        
        # Portfolio simulator
        self.portfolio_simulator = PortfolioSimulator(
            logger=logging.getLogger(f"{__name__}.PortfolioSimulator"),
            env_config=self.config.env,
            simulation_config=self.config.simulation,
            model_config=self.config.model,
            tradable_assets=[self.episode_config.symbol],
        )
        
        # Execution simulator
        self.execution_simulator = ExecutionSimulator(
            logger=logging.getLogger(f"{__name__}.ExecutionSimulator"),
            simulation_config=self.config.simulation,
            np_random=self.np_random,
            market_simulator=self.market_simulator,
        )
        
        # Reward system
        self.reward_system = RewardSystem(
            config=self.config.env.reward,
            callback_manager=self.callback_manager,
            logger=logging.getLogger(f"{__name__}.RewardSystem"),
        )
        
        # Action masking
        self.action_mask = ActionMask(
            config=self.config.simulation,
            logger=logging.getLogger(f"{__name__}.ActionMask")
        )
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, ObservationArray], Dict[str, Any]]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            Tuple of (initial_observation, info)
        """
        super().reset(seed=seed)
        
        if not self.episode_config:
            raise RuntimeError("Episode not configured. Call setup_episode first.")
        
        # Reset episode state
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Reset market simulator to episode start time
        if not self.market_simulator.reset():
            raise RuntimeError("Failed to reset market simulator")
        
        if not self.market_simulator.set_time(self.episode_config.start_time):
            raise RuntimeError(f"Failed to set time to {self.episode_config.start_time}")
        
        # Get initial market state
        initial_market_state = self.market_simulator.get_market_state()
        if not initial_market_state:
            raise RuntimeError("Failed to get initial market state")
        
        # Reset simulators
        current_time = initial_market_state.timestamp
        self.execution_simulator.reset(np_random_seed_source=self.np_random)
        self.portfolio_simulator.reset(session_start=current_time)
        
        if hasattr(self.reward_system, 'reset'):
            self.reward_system.reset()
        
        # Get initial observation
        self._last_observation = self._get_observation()
        self._current_state = self._last_observation.copy() if self._last_observation else None
        
        if not self._last_observation:
            raise RuntimeError("Failed to get initial observation")
        
        # Build info dict
        info = {
            'episode_config': {
                'symbol': str(self.episode_config.symbol),
                'date': self.episode_config.date.strftime('%Y-%m-%d'),
                'start_time': self.episode_config.start_time.strftime('%H:%M:%S'),
                'end_time': self.episode_config.end_time.strftime('%H:%M:%S'),
            },
            'reset_point_info': self.episode_config.reset_point_info or {},
            'initial_cash': self.portfolio_simulator.initial_capital,
        }
        
        self.logger.debug(f"Episode reset complete at {current_time}")
        return self._last_observation, info
    
    def step(
        self, 
        action: Union[int, NDArray[np.int32], Tuple[int, int]]
    ) -> Tuple[Dict[str, ObservationArray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute (various formats supported)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Get current market state
        market_state = self.market_simulator.get_current_market_data()
        if not market_state:
            return self._handle_step_error("Invalid market state")
        
        current_time = market_state["timestamp"]
        
        # Check episode time limit
        if current_time >= self.episode_config.end_time:
            return self._terminate_episode(
                TerminationReasonEnum.MAX_DURATION,
                "Episode time limit reached"
            )
        
        # Get portfolio state before action
        portfolio_state_before = self.portfolio_simulator.get_portfolio_state(current_time)
        
        # Apply action masking if configured
        action = self._apply_action_masking(action, portfolio_state_before, market_state)
        
        # Execute action
        execution_result = self.execution_simulator.execute_action(
            raw_action=action,
            market_state=market_state,
            portfolio_state=portfolio_state_before,
            primary_asset=self.episode_config.symbol,
            portfolio_manager=self.portfolio_simulator,
        )
        
        # Process any fills
        fill_details_list = []
        if execution_result.fill_details:
            enriched_fill = self.portfolio_simulator.process_fill(
                execution_result.fill_details
            )
            fill_details_list.append(enriched_fill)
        
        # Update portfolio with current prices
        self._update_portfolio_prices(market_state, fill_details_list)
        portfolio_state_after = self.portfolio_simulator.get_portfolio_state(current_time)
        
        # Advance market simulation
        market_advanced = self.market_simulator.step()
        
        # Get next observation
        if market_advanced:
            next_observation = self._get_observation()
            if next_observation:
                self._last_observation = next_observation
                self._current_state = next_observation.copy()
            else:
                return self._terminate_episode(
                    TerminationReasonEnum.OBSERVATION_FAILURE,
                    "Failed to generate observation"
                )
        else:
            # End of market data
            return self._terminate_episode(
                TerminationReasonEnum.END_OF_SESSION_DATA,
                "Market data exhausted"
            )
        
        # Update portfolio at next timestep
        next_market_state = self.market_simulator.get_current_market_data()
        if next_market_state:
            self._update_portfolio_prices(next_market_state, [])
            portfolio_state_next = self.portfolio_simulator.get_portfolio_state(
                next_market_state["timestamp"]
            )
        else:
            portfolio_state_next = portfolio_state_after
        
        # Check termination conditions
        terminated, termination_reason = self._check_termination(
            portfolio_state_next, 
            market_advanced
        )
        
        # Calculate reward
        reward = self.reward_system.calculate(
            portfolio_state_before_action=portfolio_state_before,
            portfolio_state_after_action_fills=portfolio_state_after,
            portfolio_state_next_t=portfolio_state_next,
            market_state_at_decision=market_state,
            market_state_next_t=next_market_state,
            decoded_action=execution_result.action_decode_result.to_dict(),
            fill_details_list=fill_details_list,
            terminated=terminated,
            truncated=False,
            termination_reason=termination_reason,
        )
        
        self.episode_reward += reward
        
        # Build info dict
        info = self._build_step_info(
            reward=reward,
            portfolio_state=portfolio_state_next,
            execution_result=execution_result,
            termination_reason=termination_reason,
        )
        
        # Log progress periodically
        if self.current_step % 100 == 0:
            self._log_progress(current_time, portfolio_state_next)
        
        return self._last_observation, reward, terminated, False, info
    
    def get_current_state(self) -> Optional[Dict[str, ObservationArray]]:
        """
        Get current environment state for PPOTrainer.
        
        This is the main interface method for v2 architecture.
        
        Returns:
            Current observation state or None if not available
        """
        return self._current_state.copy() if self._current_state else None
    
    def get_action_mask(self) -> Optional[NDArray[np.bool_]]:
        """
        Get current valid action mask.
        
        Returns:
            Boolean array of valid actions or None if masking disabled
        """
        if not self.action_mask or not self.market_simulator:
            return None
        
        current_time = self.market_simulator.get_current_time()
        portfolio_state = self.portfolio_simulator.get_portfolio_state(current_time)
        market_state = self.market_simulator.get_current_market_data()
        
        if not portfolio_state or not market_state:
            return None
        
        return self.action_mask.get_action_mask(portfolio_state, market_state)
    
    def _get_observation(self) -> Optional[Dict[str, ObservationArray]]:
        """Generate observation from current state."""
        try:
            # Get pre-calculated features from market simulator
            features = self.market_simulator.get_current_features()
            if not features:
                return None
            
            # Get portfolio features
            portfolio_obs = self.portfolio_simulator.get_portfolio_observation()
            portfolio_features = portfolio_obs["features"]
            
            # Build observation dict
            obs = {
                "hf": features.get("hf"),
                "mf": features.get("mf"),
                "lf": features.get("lf"),
                "portfolio": portfolio_features,
            }
            
            # Validate and clean
            for key, arr in obs.items():
                if arr is None:
                    # Use zeros if missing
                    space_item = self.observation_space[key]
                    obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)
                else:
                    # Replace NaN with 0
                    obs[key] = np.nan_to_num(arr, nan=0.0)
                
                # Validate shape
                expected_shape = self.observation_space[key].shape
                if obs[key].shape != expected_shape:
                    self.logger.error(
                        f"Shape mismatch for '{key}': "
                        f"expected {expected_shape}, got {obs[key].shape}"
                    )
                    return None
            
            return obs
            
        except Exception as e:
            self.logger.error(f"Error generating observation: {e}")
            return None
    
    def _apply_action_masking(
        self, 
        action: Union[int, NDArray[np.int32], Tuple[int, int]],
        portfolio_state: PortfolioState,
        market_state: Dict[str, Any]
    ) -> Union[int, NDArray[np.int32], Tuple[int, int]]:
        """Apply action masking to validate and potentially override action."""
        if not self.action_mask:
            return action
        
        # Convert to tuple format if needed
        if isinstance(action, (int, np.integer)):
            # Assume it's a linear index
            action_type_idx = int(action) // 4
            size_idx = int(action) % 4
            action = (action_type_idx, size_idx)
        elif hasattr(action, '__len__') and len(action) >= 2:
            action = (int(action[0]), int(action[1]))
        
        # Check if action is valid
        linear_idx = action[0] * 4 + action[1]
        mask = self.action_mask.get_action_mask(portfolio_state, market_state)
        
        if not mask[linear_idx]:
            # Invalid action - force to HOLD
            self.logger.debug(
                f"Invalid action {self.action_mask.get_action_description(linear_idx)} "
                f"masked to HOLD"
            )
            return (0, 0)  # HOLD with 25% size
        
        return action
    
    def _update_portfolio_prices(
        self, 
        market_state: Dict[str, Any], 
        fill_details: list[FillDetails]
    ):
        """Update portfolio with current market prices."""
        timestamp = fill_details[-1].fill_timestamp if fill_details else market_state["timestamp"]
        
        # Extract price
        current_price = market_state.get("current_price", 0.0)
        if current_price <= 0:
            bid = market_state.get("best_bid_price", 0)
            ask = market_state.get("best_ask_price", 0)
            if bid > 0 and ask > 0:
                current_price = (bid + ask) / 2
        
        if current_price > 0:
            prices = {self.episode_config.symbol: current_price}
            self.portfolio_simulator.update_market_values(prices, timestamp)
    
    def _check_termination(
        self, 
        portfolio_state: PortfolioState, 
        market_advanced: bool
    ) -> Tuple[bool, Optional[TerminationReasonEnum]]:
        """Check if episode should terminate."""
        # Get current equity
        current_equity = portfolio_state.get("total_equity", 0.0)
        initial_capital = self.portfolio_simulator.initial_capital
        
        # Bankruptcy check
        bankruptcy_threshold = initial_capital * 0.1  # 10% of initial
        if current_equity <= bankruptcy_threshold:
            return True, TerminationReasonEnum.BANKRUPTCY
        
        # Max loss check
        max_loss_threshold = initial_capital * 0.75  # 25% loss
        if current_equity <= max_loss_threshold:
            return True, TerminationReasonEnum.MAX_LOSS_REACHED
        
        # Max steps check
        if self.episode_config.max_steps and self.current_step >= self.episode_config.max_steps:
            return True, TerminationReasonEnum.MAX_STEPS_REACHED
        
        # End of data
        if not market_advanced:
            return True, TerminationReasonEnum.END_OF_SESSION_DATA
        
        return False, None
    
    def _terminate_episode(
        self, 
        reason: TerminationReasonEnum, 
        message: str
    ) -> Tuple[Dict[str, ObservationArray], float, bool, bool, Dict[str, Any]]:
        """Handle episode termination."""
        self.logger.info(f"Episode terminated: {message}")
        
        # Return last valid observation or dummy
        obs = self._last_observation or self._get_dummy_observation()
        
        info = {
            "termination_reason": reason.value,
            "episode_reward": self.episode_reward,
            "episode_steps": self.current_step,
        }
        
        return obs, 0.0, True, False, info
    
    def _handle_step_error(
        self, 
        error_msg: str
    ) -> Tuple[Dict[str, ObservationArray], float, bool, bool, Dict[str, Any]]:
        """Handle errors during step execution."""
        self.logger.error(f"Step error: {error_msg}")
        return self._terminate_episode(TerminationReasonEnum.ERROR, error_msg)
    
    def _get_dummy_observation(self) -> Dict[str, ObservationArray]:
        """Generate dummy observation matching observation space."""
        dummy_obs = {}
        for key, space in self.observation_space.items():
            dummy_obs[key] = np.zeros(space.shape, dtype=space.dtype)
        return dummy_obs
    
    def _build_step_info(
        self,
        reward: float,
        portfolio_state: PortfolioState,
        execution_result: Any,
        termination_reason: Optional[TerminationReasonEnum],
    ) -> Dict[str, Any]:
        """Build info dictionary for step return."""
        info = {
            "step": self.current_step,
            "reward": reward,
            "episode_reward": self.episode_reward,
            "action_taken": execution_result.action_decode_result.to_dict(),
        }
        
        if termination_reason:
            info["termination_reason"] = termination_reason.value
        
        return info
    
    def _log_progress(self, current_time: datetime, portfolio_state: PortfolioState):
        """Log episode progress periodically."""
        equity = portfolio_state.get("total_equity", 0.0)
        pnl = equity - self.portfolio_simulator.initial_capital
        pnl_pct = (pnl / self.portfolio_simulator.initial_capital * 100 
                   if self.portfolio_simulator.initial_capital > 0 else 0)
        
        self.logger.info(
            f"Step {self.current_step} | "
            f"Time: {current_time.strftime('%H:%M:%S')} | "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | "
            f"Reward: {self.episode_reward:.4f}"
        )
    
    def setup_session(self, symbol: str, date: Union[str, datetime]) -> None:
        """
        Setup a trading session - adapter method for v2 compatibility.
        
        Note: In v2, actual episode configuration happens separately.
        This just prepares the base session.
        
        Args:
            symbol: Trading symbol
            date: Trading date
        """
        # Store session info for later episode setup
        self._session_symbol = symbol
        self._session_date = pd.Timestamp(date).to_pydatetime() if isinstance(date, str) else date
        self.logger.debug(f"Session info stored: {symbol} on {self._session_date.strftime('%Y-%m-%d')}")
    
    def reset_at_point(self, reset_point_idx: int = 0) -> Tuple[Dict[str, ObservationArray], Dict[str, Any]]:
        """
        Reset to a specific reset point - adapter method for v2 compatibility.
        
        Note: In v2, reset points are managed by EpisodeManager.
        This method creates an episode config and delegates to the new setup.
        
        Args:
            reset_point_idx: Index of reset point (managed by EpisodeManager)
            
        Returns:
            Tuple of (initial_observation, info)
        """
        if not hasattr(self, '_session_symbol') or not hasattr(self, '_session_date'):
            raise RuntimeError("Session not configured. Call setup_session first.")
        
        # Create episode config from session info
        # Note: In production, reset point details would come from EpisodeManager
        session_start = datetime.combine(self._session_date.date(), time(4, 0))  # 4 AM ET
        session_end = datetime.combine(self._session_date.date(), time(20, 0))   # 8 PM ET
        
        # Convert to UTC (simple offset for now, production would use proper timezone handling)
        session_start_utc = session_start + timedelta(hours=5)  # Assume EST
        session_end_utc = session_end + timedelta(hours=5)
        
        # For reset points, we'd normally get timing from EpisodeManager
        # For now, use simple logic based on index
        reset_offsets = [
            timedelta(hours=5, minutes=30),   # 9:30 AM ET
            timedelta(hours=6, minutes=30),   # 10:30 AM ET  
            timedelta(hours=10),              # 2:00 PM ET
            timedelta(hours=11, minutes=30),  # 3:30 PM ET
        ]
        
        reset_offset = reset_offsets[reset_point_idx % len(reset_offsets)]
        episode_start = session_start_utc + reset_offset
        episode_end = min(episode_start + timedelta(hours=2), session_end_utc)
        
        # Create episode config
        episode_config = EpisodeConfig(
            symbol=self._session_symbol,
            date=self._session_date,
            start_time=episode_start,
            end_time=episode_end,
            reset_point_info={'index': reset_point_idx}
        )
        
        # Setup and reset
        if not self.setup_episode(episode_config):
            raise RuntimeError("Failed to setup episode")
        
        return self.reset()
    
    def close(self):
        """Clean up environment resources."""
        if self.market_simulator:
            self.market_simulator.close()
        
        self.logger.info("🔒 TradingEnvironment closed")