"""
TradingEnvironment implementation following TDD principles.

This is a clean, minimal implementation of the ITradingEnvironment interface.
"""

from datetime import datetime
from typing import Optional, Union
import logging
import numpy as np
import gymnasium as gym
import pandas as pd
from gymnasium import spaces

from v2.config import Config
from v2.data.interfaces import IDataManager
from v2.envs import IActionMask
from v2.envs.interfaces import (
    ITradingEnvironment,
    ObservationDict,
    InfoDict,
    ActionArray,
    ActionMask,
    ActionProbabilities,
    ResetPoint,
    MomentumDay,
    SessionOptions,
)
from v2.simulation.interfaces import IMarketSimulator, IPortfolioSimulator, IExecutionSimulator
from v2.simulation.rewards.interfaces import IRewardCalculator


class TradingEnvironment(gym.Env):
    """
    Clean implementation of trading environment following ITradingEnvironment interface.
    
    Uses dependency injection for all components to enable testing and modularity.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
            self,
            config: Config,
            market_simulator: IMarketSimulator,
            portfolio_simulator: IPortfolioSimulator,
            execution_simulator: IExecutionSimulator,
            reward_calculator: IRewardCalculator,
            action_mask: IActionMask,
            data_manager: IDataManager,
            logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize trading environment with injected dependencies.
        
        Args:
            config: Environment configuration
            market_simulator: Market simulation component
            portfolio_simulator: Portfolio management component
            execution_simulator: Trade execution component
            reward_calculator: Reward calculation component
            action_mask: Action masking component
            data_manager: Data management component
            logger: Optional logger
        """
        super().__init__()

        self.config = config
        self.market_simulator = market_simulator
        self.portfolio_simulator = portfolio_simulator
        self.execution_simulator = execution_simulator
        self.reward_calculator = reward_calculator
        self.action_mask = action_mask
        self.data_manager = data_manager
        self.logger = logger or logging.getLogger(__name__)

        # Environment state
        self.current_symbol: Optional[str] = None
        self.current_date: Optional[datetime] = None
        self.reset_points: list[ResetPoint] = []
        self.current_reset_idx: int = 0

        model_cfg = self.config.model

        # Action space: [action_type, position_size] - discrete
        self.action_space = spaces.MultiDiscrete([3, 4])  # 3 actions x 4 sizes = 12 total

        # Observation space: multi-frequency features
        self.observation_space = spaces.Dict({
            "hf": spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.hf_seq_len, model_cfg.hf_feat_dim), dtype=np.float32),
            "mf": spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.mf_seq_len, model_cfg.mf_feat_dim), dtype=np.float32),
            "lf": spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.lf_seq_len, model_cfg.lf_feat_dim), dtype=np.float32),
            "portfolio": spaces.Box(low=-np.inf, high=np.inf, shape=(model_cfg.portfolio_seq_len, model_cfg.portfolio_feat_dim), dtype=np.float32),
        })

    def setup_session(self, symbol: str, date: Union[str, datetime]) -> None:
        """Setup trading session for symbol and date."""
        if not symbol or not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("A valid symbol (string) must be provided.")

        self.current_symbol = symbol

        if isinstance(date, str):
            self.current_date = pd.Timestamp(date).to_pydatetime()
        else:
            self.current_date = date

        self.logger.info(
            f"├── 🎯 Session setup: {self.current_symbol} on {self.current_date.strftime('%Y-%m-%d')}"
        )

        if not self.market_simulator.initialize_day(self.current_date):
            raise ValueError(f"Failed to initialize market simulator for {symbol} on {self.current_date.strftime('%Y-%m-%d')}")

        # Load reset points for this session
        self.reset_points = self.data_manager.get_reset_points(symbol, self.current_date)
        self.current_reset_idx = 0

        # Handle case where data manager returns None
        reset_points_count = len(self.reset_points) if self.reset_points is not None else 0
        self.logger.info(f"🔄 {reset_points_count} reset points available for training")
        self.logger.info(f"🔄 Session setup complete: {symbol} on {self.current_date.strftime('%Y-%m-%d')}")

    def reset(self, seed: Optional[int] = None, options: Optional[SessionOptions] = None) -> tuple[ObservationDict, InfoDict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        if not self.current_symbol or not self.current_date:
            raise ValueError("Session not setup. Call setup_session first.")

        # Use first reset point by default
        reset_point_idx = (options or {}).get("reset_point_idx", 0)
        return self.reset_at_point(reset_point_idx)

    def step(self, action: ActionArray) -> tuple[ObservationDict, float, bool, bool, InfoDict]:
        """Execute one environment step."""
        # Placeholder implementation - will be filled in later iterations
        obs = self._get_dummy_observation()
        reward = 0.0
        terminated = False
        truncated = False
        info = {"step": "placeholder"}

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close environment and cleanup resources."""
        self.logger.info("Environment closed")

    def render(self, info_dict: Optional[InfoDict] = None) -> None:
        """Render environment state."""
        if self.render_mode == "human":
            print(f"TradingEnvironment: {self.current_symbol} on {self.current_date}")

    def get_action_mask(self) -> Optional[ActionMask]:
        """Get current action mask for valid actions."""
        # Placeholder - will implement when action masking is needed
        return None

    def mask_action_probabilities(self, action_probs: ActionProbabilities) -> ActionProbabilities:
        """Apply action masking to action probabilities."""
        # Placeholder - return unchanged for now
        return action_probs

    def reset_at_point(self, reset_point_idx: Optional[int] = None) -> tuple[ObservationDict, InfoDict]:
        """Reset to specific reset point within session."""
        if reset_point_idx is None:
            reset_point_idx = self.current_reset_idx

        if not self.reset_points or reset_point_idx >= len(self.reset_points):
            raise ValueError( f"Reset point index {reset_point_idx} out of range (max: {len(self.reset_points) - 1})")

        reset_point = self.reset_points[reset_point_idx]
        self.current_reset_idx = reset_point_idx

        # Todo : This is problematic, how do we handle reward calculation for open positions?
        # position_close_pnl = self._handle_open_positions_at_reset()

        # Randomized offset
        self.episode_start_time = reset_point["timestamp"]
        offset = np.random.Generator.uniform(-self.config.reset_offset_backward, self.config.reset_offset_forward)
        randomized_start = self.episode_start_time + pd.Timedelta(seconds=offset)

        # Reset all components
        self.market_simulator.reset()
        self.portfolio_simulator.reset() # Session Start Time.
        self.execution_simulator.reset()
        self.reward_calculator.reset()

        # Get initial observation
        obs = self._get_observation()
        info = {"reset_point": reset_point, "reset_point_idx": reset_point_idx}

        return obs, info

    def get_next_reset_point(self) -> Optional[ResetPoint]:
        """Get next available reset point."""
        next_idx = self.current_reset_idx + 1
        if next_idx < len(self.reset_points):
            return self.reset_points[next_idx]
        return None

    def has_more_reset_points(self) -> bool:
        """Check if more reset points are available."""
        return self.current_reset_idx + 1 < len(self.reset_points)

    def select_next_momentum_day(self, exclude_dates: Optional[list[datetime]] = None) -> Optional[MomentumDay]:
        """Select next momentum day for training."""
        momentum_days = self.data_manager.get_momentum_days(self.current_symbol or "", min_activity=0.0)

        if exclude_dates:
            # Filter out excluded dates
            momentum_days = [day for day in momentum_days if day["date"] not in exclude_dates]

        return momentum_days[0] if momentum_days else None

    def prepare_next_session(self, symbol: str, date: Union[str, datetime]) -> None:
        """Prepare next session in background."""
        # Placeholder - implement when session switching is needed
        self.logger.info(f"Preparing next session: {symbol} on {date}")

    def switch_to_next_session(self) -> None:
        """Switch to prepared next session."""
        # Placeholder - implement when session switching is needed
        self.logger.info("Switching to next session")

    def set_training_info(self, episode_num: int = 0, total_episodes: int = 0,
                          total_steps: int = 0, update_count: int = 0) -> None:
        """Set training information for metrics tracking."""
        # Placeholder - implement when training metrics are needed
        pass

    # Helper methods

    def _get_observation(self) -> ObservationDict:
        """Get current observation from market and portfolio simulators."""
        # Get features from market simulator
        features = self.market_simulator.get_current_features()
        if not features:
            return self._get_dummy_observation()

        # Get portfolio observation
        portfolio_obs = self.portfolio_simulator.get_portfolio_observation()

        # Combine observations
