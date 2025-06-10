"""
Trading Environment - Redesigned for momentum-based low-float trading

This implementation uses the new architecture:
- Pre-calculated features from MarketSimulator
- Momentum-based episode selection
- Day-based training with reset points
- Position handling across episode boundaries
"""

import logging
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config.schemas import Config
from data.data_manager import DataManager
from envs.action_masking import ActionMask
from rewards.calculator import RewardSystem
from simulators.execution_simulator import ExecutionSimulator
from simulators.market_simulator import MarketSimulator
from simulators.portfolio_simulator import (
    PortfolioSimulator,
    PortfolioState,
    FillDetails,
)
from core.types import TerminationReasonEnum




class TradingEnvironment(gym.Env):
    """
    New trading environment designed for momentum-based training.

    Key features:
    - Uses pre-calculated features from MarketSimulator
    - Day-based episodes with multiple reset points
    - Momentum-aware episode selection
    - Position handling across episode boundaries
    - Sniper trading focused (quick in/out)
    """

    metadata = {"render_modes": ["human", "logs", "none"], "render_fps": 10}

    def _safe_date_format(self, date_obj) -> str:
        """Safely format a date object to YYYY-MM-DD string"""
        if isinstance(date_obj, str):
            return date_obj  # Already a string
        elif hasattr(date_obj, 'strftime'):
            return date_obj.strftime('%Y-%m-%d')
        elif hasattr(date_obj, 'date'):
            return date_obj.date().strftime('%Y-%m-%d')
        else:
            return str(date_obj)

    def __init__(
        self,
        config: Config,
        data_manager: DataManager,
        logger: Optional[logging.Logger] = None,
        callback_manager=None,
    ):
        super().__init__()
        self.config = config
        self.data_manager = data_manager
        self.callback_manager = callback_manager

        # Logger setup
        if logger is None:
            self.logger = logging.getLogger(f"{__name__}.TradingEnv")
        else:
            self.logger = logger

        # Environment configuration
        env_cfg = self.config.env
        self.primary_asset: Optional[str] = None
        # Action masking eliminates invalid actions, so no limit needed
        self.bankruptcy_threshold_factor: float = 0.1
        # Fixed max loss threshold - 25% loss (6.25k out of 25k)
        self.max_session_loss_percentage: float = 0.25

        # Action Space - execution simulator handles decoding now
        self.action_space = spaces.MultiDiscrete(
            [3, 4]
        )  # [action_types, position_sizes]

        # Observation Space - same as before
        model_cfg = self.config.model
        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "hf": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(model_cfg.hf_seq_len, model_cfg.hf_feat_dim),
                    dtype=np.float32,
                ),
                "mf": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(model_cfg.mf_seq_len, model_cfg.mf_feat_dim),
                    dtype=np.float32,
                ),
                "lf": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(model_cfg.lf_seq_len, model_cfg.lf_feat_dim),
                    dtype=np.float32,
                ),
                "portfolio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(model_cfg.portfolio_seq_len, model_cfg.portfolio_feat_dim),
                    dtype=np.float32,
                ),
            }
        )

        # Core components - initialized in setup_session
        self.market_simulator: Optional[MarketSimulator] = None
        self.next_market_simulator: Optional[MarketSimulator] = (
            None  # For background preparation
        )
        self.execution_manager: Optional[ExecutionSimulator] = None
        self.portfolio_manager: Optional[PortfolioSimulator] = None
        self.reward_calculator: Optional[RewardSystem] = None
        self.action_mask: Optional[ActionMask] = None

        # Episode state
        self.current_step: int = 0
        self.max_steps: int = config.env.max_steps  # Maximum steps per episode
        self.max_training_steps: Optional[int] = (
            config.env.max_training_steps
        )  # Training limit (with penalty)
        self.episode_total_reward: float = 0.0
        self.initial_capital_for_session: float = (
            self.config.simulation.initial_capital
        )  # Initialize with config value
        self.episode_number: int = 0

        # Episode boundaries and reset points
        self.current_session_date: Optional[datetime] = None
        self.episode_start_time_utc: Optional[datetime] = None
        self.episode_end_time_utc: Optional[datetime] = None
        self.reset_points: List[Dict] = []
        self.current_reset_idx: int = 0


        # State tracking
        self._last_observation: Optional[Dict[str, np.ndarray]] = None
        self._last_portfolio_state_before_action: Optional[PortfolioState] = None
        self._last_decoded_action: Optional[Dict[str, Any]] = None
        self.current_termination_reason: Optional[str] = None
        self.is_terminated: bool = False
        self.is_truncated: bool = False


        self.render_mode = None

    def setup_session(self, symbol: str, date: Union[str, datetime]):
        """
        Setup a new trading session for a specific date.
        Uses the new MarketSimulator with pre-calculated features.
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("A valid symbol (string) must be provided.")

        self.primary_asset = symbol

        # Parse date
        if isinstance(date, str):
            self.current_session_date = pd.Timestamp(date).to_pydatetime()
        else:
            self.current_session_date = date

        self.logger.info(
            f"├── 🎯 Session setup: {self.primary_asset} on {self._safe_date_format(self.current_session_date)}"
        )

        # Create MarketSimulator for this session
        self.market_simulator = MarketSimulator(
            symbol=self.primary_asset,
            data_manager=self.data_manager,
            model_config=self.config.model,
            simulation_config=self.config.simulation,
        )

        # Initialize day - this pre-calculates ALL features for the entire day
        success = self.market_simulator.initialize_day(self.current_session_date)
        if not success:
            raise ValueError(
                f"Failed to initialize {symbol} on {self.current_session_date}"
            )

        # Get session stats
        stats = self.market_simulator.get_stats()
        self.logger.info(
            f"✅ Session ready: {stats['total_seconds']} seconds, "
            f"warmup: {stats['warmup_info']['has_warmup']}"
        )

        # Load reset points from momentum indices (if available) or fallback to fixed points
        self.reset_points = self._generate_reset_points()
        self.current_reset_idx = 0

        # Initialize other components
        self._initialize_simulators()

        self.logger.info(
            f"🔄 {len(self.reset_points)} reset points available for training"
        )

    def _generate_fixed_reset_points(self) -> List[Dict]:
        """Generate fixed reset points based on market hours."""
        reset_points = []
        base_date = self.current_session_date.date()

        # Fixed reset times (ET) - convert to UTC
        fixed_times = [
            time(9, 30),  # Market open
            time(10, 30),  # Post-open settlement
            time(14, 0),  # Afternoon session
            time(15, 30),  # Power hour
        ]

        for reset_time in fixed_times:
            reset_dt = datetime.combine(base_date, reset_time)
            # Convert ET to UTC (assuming EST/EDT handling is done elsewhere)
            reset_dt_utc = (
                pd.Timestamp(reset_dt, tz="US/Eastern")
                .tz_convert("UTC")
                .to_pydatetime()
            )

            reset_points.append(
                {
                    "timestamp": reset_dt_utc,
                    "activity_score": 0.5,  # Default activity score
                    "combined_score": 0.5,  # Default combined score
                    "max_duration_hours": 4,
                    "reset_type": "fixed",
                }
            )

        return reset_points

    def _generate_reset_points(self) -> List[Dict]:
        """Generate reset points using momentum indices with intelligent fallback for gaps."""
        # Try to get momentum-based reset points from data manager
        momentum_reset_points = self.data_manager.get_reset_points(
            self.primary_asset, self.current_session_date
        )

        if not momentum_reset_points.empty:
            # Use momentum index reset points directly
            reset_points = []
            for _, row in momentum_reset_points.iterrows():
                reset_points.append(
                    {
                        "timestamp": row["timestamp"],
                        "activity_score": row.get("activity_score", 0.5),
                        "combined_score": row.get("combined_score", 0.5),
                        "day_activity_score": row.get("day_activity_score", 0.5),
                        # Add 2-component scores for adaptive data system
                        "roc_score": row.get("roc_score", 0.0),
                        "max_duration_hours": self._get_duration_for_activity(
                            row.get("activity_score", 0.5)
                        ),
                        "reset_type": "momentum",
                        "volume_ratio": row.get("volume_ratio", 1.0),
                        "price_change": row.get("price_change", 0.0),
                        # Additional fields from scanner for completeness
                        "price": row.get("price", 0.0),
                        "volume": row.get("volume", 0),
                        "session": row.get("session", "regular"),
                    }
                )

            # Check for early trading hour gaps and supplement with fixed points if needed
            reset_points = self._supplement_with_early_fixed_points(reset_points)

            self.logger.info(
                f"Using {len(reset_points)} reset points (momentum + early fixed supplements)"
            )
            return reset_points

        else:
            # Fallback to fixed reset points
            self.logger.info("No momentum reset points found, using fixed schedule")
            return self._generate_fixed_reset_points()

    def _supplement_with_early_fixed_points(
        self, momentum_reset_points: List[Dict]
    ) -> List[Dict]:
        """Supplement momentum reset points with fixed early points if there are gaps."""
        if not momentum_reset_points:
            return momentum_reset_points

        # Find the earliest momentum reset point
        earliest_momentum = min(rp["timestamp"] for rp in momentum_reset_points)

        # Trading session starts at 4 AM ET, check if we have coverage
        base_date = self.current_session_date.date()
        session_start_et = datetime.combine(base_date, time(4, 0))
        session_start_utc = (
            pd.Timestamp(session_start_et, tz="US/Eastern")
            .tz_convert("UTC")
            .to_pydatetime()
        )

        # If momentum points start after 10 AM ET, add early fixed points
        cutoff_et = datetime.combine(base_date, time(10, 0))
        cutoff_utc = (
            pd.Timestamp(cutoff_et, tz="US/Eastern").tz_convert("UTC").to_pydatetime()
        )

        if earliest_momentum > cutoff_utc:
            # Add early fixed reset points to cover the gap
            early_fixed_times = [
                time(6, 0),  # Pre-market
                time(9, 30),  # Market open
            ]

            early_points = []
            for reset_time in early_fixed_times:
                reset_dt = datetime.combine(base_date, reset_time)
                reset_dt_utc = (
                    pd.Timestamp(reset_dt, tz="US/Eastern")
                    .tz_convert("UTC")
                    .to_pydatetime()
                )

                # Only add if it's before the earliest momentum point
                if reset_dt_utc < earliest_momentum:
                    # Try to get a price for this timestamp from market data
                    price = self._get_price_at_timestamp(reset_dt_utc)

                    early_points.append(
                        {
                            "timestamp": reset_dt_utc,
                            "price": price,
                            "activity_score": 0.3,  # Lower activity for early supplemental points
                            "combined_score": 0.3,
                            "max_duration_hours": 4,
                            "reset_type": "early_fixed_supplement",
                        }
                    )

            if early_points:
                self.logger.info(
                    f"Added {len(early_points)} early fixed reset points to supplement momentum data"
                )
                # Combine and sort by timestamp
                all_points = early_points + momentum_reset_points
                all_points.sort(key=lambda x: x["timestamp"])
                return all_points

        return momentum_reset_points

    def _get_price_at_timestamp(self, timestamp: datetime) -> float:
        """Get price at a specific timestamp from market data, with fallback."""
        try:
            # For early fixed points, we don't have exact price data
            # Use a reasonable estimate based on typical MLGO trading ranges
            # In production, this could query the market data more sophisticated
            return 3.0  # Default price for MLGO early session
        except Exception:
            return 3.0  # Safe fallback

    def _get_adaptive_randomization_window(self, reset_point: Dict) -> int:
        """Get adaptive randomization window in minutes based on activity score."""
        activity_score = reset_point.get("activity_score", 0.5)
        combined_score = reset_point.get("combined_score", 0.5)
        reset_type = reset_point.get("reset_type", "momentum")

        # Base window depends on activity level
        # With 5-minute reset points, use tighter windows for better precision
        if activity_score >= 0.8:
            base_window = 2  # Very high activity - tight window (±2 min)
        elif activity_score >= 0.6:
            base_window = 3  # High activity - moderate window (±3 min)
        elif activity_score >= 0.4:
            base_window = 6  # Medium activity - wider window (±6 min)
        else:
            base_window = 10  # Low activity - wide window (±10 min)

        # Adjust based on combined score (includes day quality)
        # Higher combined score = more important point
        score_multiplier = 1.0 - (combined_score - 0.5) * 0.3  # 0.85 to 1.15 range
        score_multiplier = max(0.7, min(1.3, score_multiplier))

        # Adjust based on reset type
        type_multipliers = {
            "momentum": 1.0,  # Standard for momentum-based points
            "fixed": 2.0,  # Wider for fixed points
        }

        type_multiplier = type_multipliers.get(reset_type, 1.0)

        # Calculate final window
        final_window = int(base_window * score_multiplier * type_multiplier)

        # Ensure reasonable bounds (1-30 minutes)
        return max(1, min(30, final_window))

    def _get_duration_for_activity(self, activity_score: float) -> float:
        """Get episode duration hours based on activity score.

        Higher activity periods typically have more concentrated action,
        so we can use shorter episodes. Lower activity periods need more
        time to capture meaningful movements.
        """
        if activity_score >= 0.8:
            return 1.5  # Very high activity - shorter episodes
        elif activity_score >= 0.6:
            return 2.0  # High activity
        elif activity_score >= 0.4:
            return 3.0  # Medium activity
        else:
            return 4.0  # Low activity - longer episodes

    def _initialize_simulators(self):
        """Initialize all simulator components."""
        if self.np_random is None:
            _, _ = super().reset(seed=None)

        # Portfolio simulator
        self.portfolio_manager = PortfolioSimulator(
            logger=logging.getLogger(f"{__name__}.PortfolioMgr"),
            env_config=self.config.env,
            simulation_config=self.config.simulation,
            model_config=self.config.model,
            tradable_assets=[self.primary_asset],
        )

        # Reward system
        self.reward_calculator = RewardSystem(
            config=self.config.env.reward,
            callback_manager=self.callback_manager,
            logger=logging.getLogger(f"{__name__}.RewardSystem"),
        )

        # Execution simulator
        self.execution_manager = ExecutionSimulator(
            logger=logging.getLogger(f"{__name__}.ExecSim"),
            simulation_config=self.config.simulation,
            np_random=self.np_random,
            market_simulator=self.market_simulator,
        )

        # Action masking system
        self.action_mask = ActionMask(
            config=self.config, logger=logging.getLogger(f"{__name__}.ActionMask")
        )

        self.logger.info("✅ All simulators initialized")

    def prepare_next_session(self, symbol: str, date: Union[str, datetime]):
        """Prepare next session in background for fast switching."""
        if isinstance(date, str):
            next_date = pd.Timestamp(date).to_pydatetime()
        else:
            next_date = date

        self.logger.info(
            f"🔄 Preparing next session: {symbol} on {self._safe_date_format(next_date)}"
        )

        # Create MarketSimulator for next session
        self.next_market_simulator = MarketSimulator(
            symbol=symbol,
            data_manager=self.data_manager,
            model_config=self.config.model,
            simulation_config=self.config.simulation,
        )

        # Initialize in background
        success = self.next_market_simulator.initialize_day(next_date)
        if success:
            self.logger.info(
                f"✅ Next session ready: {symbol} {self._safe_date_format(next_date)}"
            )
        else:
            self.logger.error(
                f"❌ Failed to prepare next session: {symbol} {self._safe_date_format(next_date)}"
            )

    def switch_to_next_session(self):
        """Switch to the prepared next session."""
        if self.next_market_simulator is None:
            raise ValueError(
                "No next session prepared. Call prepare_next_session first."
            )

        # Switch simulators
        self.market_simulator = self.next_market_simulator
        self.next_market_simulator = None

        # Update execution manager to use new market simulator
        self.execution_manager.market_simulator = self.market_simulator

        # Generate new reset points
        self.reset_points = self._generate_reset_points()
        self.current_reset_idx = 0

        self.logger.info("🔄 Switched to prepared session")

    def get_momentum_days(self, min_activity: float = 0.0) -> pd.DataFrame:
        """Get available momentum days for the current symbol."""
        return self.data_manager.get_momentum_days(self.primary_asset, min_activity)

    def select_next_momentum_day(
        self, exclude_dates: Optional[List[datetime]] = None
    ) -> Optional[Dict]:
        """Select next momentum day based on quality and adaptive data criteria."""
        momentum_days = self.get_momentum_days(min_activity=0.0)

        if momentum_days.empty:
            return None

        # Apply exclusions
        if exclude_dates:
            exclude_dates_only = [
                d.date() if isinstance(d, datetime) else d for d in exclude_dates
            ]
            momentum_days = momentum_days[
                ~momentum_days["date"].dt.date.isin(exclude_dates_only)
            ]

        if momentum_days.empty:
            return None

        # Simple selection by quality score (highest first)
        best_day = momentum_days.iloc[0]

        return {
            "symbol": best_day["symbol"],
            "date": best_day["date"],
            "quality_score": best_day[
                "activity_score"
            ],  # Map activity_score to quality_score
            "max_intraday_move": best_day.get("max_intraday_move", 0.0),
            "volume_multiplier": best_day.get("volume_multiplier", 1.0),
        }

    def reset_at_point(
        self, reset_point_idx: int = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset to a specific reset point within the loaded day.
        This is the main reset method for momentum-based training.
        """
        if reset_point_idx is None:
            reset_point_idx = self.current_reset_idx

        if reset_point_idx >= len(self.reset_points):
            self.logger.error(
                f"Reset point index {reset_point_idx} out of range (max: {len(self.reset_points) - 1})"
            )
            return self._get_dummy_observation(), {"error": "Invalid reset point"}

        reset_point = self.reset_points[reset_point_idx]
        self.current_reset_idx = reset_point_idx


        # Reset episode state
        self.current_step = 0
        self.episode_total_reward = 0.0

        # Reset state tracking
        self.current_termination_reason = None
        self.is_terminated = False
        self.is_truncated = False
        self._last_decoded_action = None

        # Increment episode number
        self.episode_number += 1

        # Set episode boundaries
        self.episode_start_time_utc = reset_point["timestamp"]
        max_duration = timedelta(hours=reset_point.get("max_duration_hours", 4))
        market_close = datetime.combine(
            self.current_session_date.date(), time(20, 0)
        )  # 8 PM ET -> UTC
        market_close_utc = (
            pd.Timestamp(market_close, tz="US/Eastern")
            .tz_convert("UTC")
            .to_pydatetime()
        )

        self.episode_end_time_utc = min(
            self.episode_start_time_utc + max_duration, market_close_utc
        )



        # Reset market simulator with adaptive randomization
        # Adjust randomization window based on pattern type and quality
        max_offset_minutes = self._get_adaptive_randomization_window(reset_point)
        max_offset_seconds = max_offset_minutes * 60

        random_offset_seconds = self.np_random.integers(
            -max_offset_seconds, max_offset_seconds + 1
        )
        randomized_start = self.episode_start_time_utc + timedelta(
            seconds=int(random_offset_seconds)
        )

        # Log episode reset info
        original_time = reset_point["timestamp"].strftime("%H:%M:%S") if hasattr(reset_point["timestamp"], 'strftime') else str(reset_point["timestamp"])
        randomized_time = randomized_start.strftime("%H:%M:%S") if hasattr(randomized_start, 'strftime') else str(randomized_start)
        offset_minutes = random_offset_seconds // 60
        window_minutes = max_offset_minutes

        self.logger.info(
            f"🎯 Episode {self.episode_number} reset: {original_time} → {randomized_time} "
            f"({offset_minutes:+d}m/±{window_minutes}m) | Activity: {reset_point.get('activity_score', 0.5):.2f} "
            f"| Combined: {reset_point.get('combined_score', 0.5):.2f} | ROC: {reset_point.get('roc_score', 0.0):.2f} | {reset_point.get('reset_type', 'momentum')}"
        )

        # Ensure we don't go before 4 AM or after 8 PM
        market_open = datetime.combine(self.current_session_date.date(), time(4, 0))
        market_open_utc = (
            pd.Timestamp(market_open, tz="US/Eastern").tz_convert("UTC").to_pydatetime()
        )
        market_close_utc = (
            pd.Timestamp(market_open, tz="US/Eastern")
            .tz_convert("UTC")
            .to_pydatetime()
            .replace(hour=20)
        )

        randomized_start = max(randomized_start, market_open_utc)
        randomized_start = min(
            randomized_start, market_close_utc - timedelta(hours=1)
        )  # At least 1 hour before close

        # Update episode end time based on randomized start
        max_duration = timedelta(hours=reset_point.get("max_duration_hours", 4))
        self.episode_end_time_utc = min(
            randomized_start + max_duration, market_close_utc
        )

        # Update episode start time to randomized time
        self.episode_start_time_utc = randomized_start

        # Reset market simulator and set to randomized start time
        if not self.market_simulator.reset():
            self.logger.error("Market simulator failed to reset")
            return self._get_dummy_observation(), {
                "error": "Market simulator reset failed"
            }

        if not self.market_simulator.set_time(randomized_start):
            self.logger.error(
                f"Failed to set market simulator time to {randomized_start}"
            )
            return self._get_dummy_observation(), {"error": "Failed to set market time"}

        initial_market_state = self.market_simulator.get_market_state()

        if initial_market_state is None:
            self.logger.error("Market simulator failed to reset")
            return self._get_dummy_observation(), {
                "error": "Market simulator reset failed"
            }

        # Reset simulators
        current_sim_time = initial_market_state.timestamp
        # self.logger.debug(f"DEBUG: About to reset execution manager")
        self.execution_manager.reset(np_random_seed_source=self.np_random)
        # self.logger.debug(f"DEBUG: Execution manager reset completed")

        # self.logger.debug(f"DEBUG: About to reset portfolio manager at time {current_sim_time}")
        self.portfolio_manager.reset(session_start=current_sim_time)
        # self.logger.debug(f"DEBUG: Portfolio manager reset completed")

        # self.logger.debug(f"DEBUG: Getting initial capital from portfolio manager")
        self.initial_capital_for_session = self.portfolio_manager.initial_capital
        # self.logger.debug(f"DEBUG: Set initial capital to {self.initial_capital_for_session}")
        # Ensure initial_capital_for_session is not None
        if self.initial_capital_for_session is None:
            # self.logger.debug(f"DEBUG: Initial capital was None, using config value")
            self.initial_capital_for_session = self.config.simulation.initial_capital

        if hasattr(self.reward_calculator, "reset"):
            # self.logger.debug(f"DEBUG: About to reset reward calculator")
            self.reward_calculator.reset()
            # self.logger.debug(f"DEBUG: Reward calculator reset completed")

        # Get initial observation using pre-calculated features
        # self.logger.debug(f"DEBUG: About to get initial observation")
        self._last_observation = self._get_observation()
        # self.logger.debug(f"DEBUG: Got initial observation: {self._last_observation is not None}")
        if self._last_observation is None:
            self.logger.error("Failed to get initial observation")
            return self._get_dummy_observation(), {
                "error": "Initial observation failed"
            }

        self._last_portfolio_state_before_action = (
            self.portfolio_manager.get_portfolio_state(current_sim_time)
        )


        initial_info = self._get_current_info(
            reward=0.0,
            current_portfolio_state_for_info=self._last_portfolio_state_before_action,
        )



        return self._last_observation, initial_info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Standard gym reset - defaults to first reset point.
        For momentum training, use reset_at_point instead.
        """
        super().reset(seed=seed)
        options = options or {}

        if not self.market_simulator:
            self.logger.error("Session not set up. Call setup_session first.")
            return self._get_dummy_observation(), {"error": "Session not set up"}

        # Use first reset point by default
        reset_point_idx = options.get("reset_point_idx", 0)
        return self.reset_at_point(reset_point_idx)

    def _get_observation(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get observation using pre-calculated features from MarketSimulator.
        This is much faster than the old on-the-fly feature calculation.
        """
        try:
            # Get current time from market simulator
            # self.logger.debug(f"DEBUG: Getting current market data in _get_observation")
            market_state = self.market_simulator.get_current_market_data()
            if market_state is None:
                # self.logger.debug(f"DEBUG: market_state is None")
                return None

            current_sim_time = market_state["timestamp"]
            # self.logger.debug(f"DEBUG: Current sim time: {current_sim_time}")

            # Get pre-calculated features from MarketSimulator - O(1) lookup!
            # self.logger.debug(f"DEBUG: About to get current features")
            features = self.market_simulator.get_current_features()
            # self.logger.debug(f"DEBUG: Got features: {features is not None}")
            if features is None:
                self.logger.warning(f"No features available at {current_sim_time}")
                return None

            # Get portfolio observation
            # self.logger.debug(f"DEBUG: About to get portfolio state")
            current_portfolio_state = self.portfolio_manager.get_portfolio_state(
                current_sim_time
            )
            # self.logger.debug(f"DEBUG: Got portfolio state")

            # self.logger.debug(f"DEBUG: About to get portfolio observation")
            portfolio_obs_component = self.portfolio_manager.get_portfolio_observation()
            # self.logger.debug(f"DEBUG: Got portfolio observation component")

            portfolio_features_array = portfolio_obs_component["features"]
            # self.logger.debug(f"DEBUG: Got portfolio features array with shape {portfolio_features_array.shape if hasattr(portfolio_features_array, 'shape') else 'unknown'}")

            # Construct observation dict
            obs = {
                "hf": features.get("hf"),
                "mf": features.get("mf"),
                "lf": features.get("lf"),
                "portfolio": portfolio_features_array,
            }

            # Handle NaN values and shape validation
            # self.logger.debug(f"DEBUG: About to handle NaN values")
            for key, arr in obs.items():
                if arr is not None:
                    # Replace NaN with 0
                    nan_count = np.isnan(arr).sum()
                    if nan_count > 0:
                        # self.logger.debug(f"DEBUG: Found {nan_count} NaN values in {key}")
                        obs[key] = np.nan_to_num(arr, nan=0.0)
                else:
                    # Use zeros if feature is missing
                    space_item = self.observation_space[key]
                    obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)
                    # self.logger.debug(f"DEBUG: {key} was None, using zeros")
            # self.logger.debug(f"DEBUG: NaN handling completed")

            # Validate shape
            for key in obs:
                expected_shape = self.observation_space[key].shape
                if obs[key].shape != expected_shape:
                    self.logger.error(
                        f"Shape mismatch for observation key '{key}'. "
                        f"Expected {expected_shape}, Got {obs[key].shape}"
                    )
                    return None

            return obs

        except Exception as e:
            self.logger.error(f"Error during observation generation: {e}")
            return None

    def _get_dummy_observation(self) -> Dict[str, np.ndarray]:
        """Generate dummy observation matching observation space."""
        dummy_obs = {}
        for key in self.observation_space.keys():
            space_item = self.observation_space[key]
            dummy_obs[key] = np.zeros(space_item.shape, dtype=space_item.dtype)
        return dummy_obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step - simplified to use ExecutionSimulator."""
        # Check for training interruption at the beginning of each step
        if (
            hasattr(__import__("main"), "training_interrupted")
            and __import__("main").training_interrupted
        ):
            self.logger.warning("Environment step interrupted by training shutdown")
            return (
                self._last_observation or self._get_dummy_observation(),
                0.0,
                True,
                False,
                {"interrupted": True, "termination_reason": "TRAINING_INTERRUPTED"},
            )

        if self._last_observation is None or self.primary_asset is None:
            self.logger.error("Step called with invalid state")
            return (
                self._get_dummy_observation(),
                0.0,
                True,
                False,
                {"error": "Invalid state"},
            )

        self.current_step += 1

        # Get current market state
        market_state_at_decision = self.market_simulator.get_current_market_data()

        # Log progress every 50 steps
        if self.current_step % 50 == 0:
            if market_state_at_decision:
                current_time = market_state_at_decision["timestamp"]
                elapsed_seconds = self.current_step  # Each step is 1 second
                # For episode progress, use max_steps if set, otherwise show unbounded progress
                if self.max_steps:
                    progress_pct = self.current_step / self.max_steps * 100
                    self.logger.info(
                        f"📈 Episode progress: Step {self.current_step}/{self.max_steps} ({progress_pct:.1f}%) | "
                        f"Episode {self.episode_number} | Sim time: {current_time.strftime('%H:%M:%S') if hasattr(current_time, 'strftime') else str(current_time)} | "
                        f"Elapsed: {elapsed_seconds // 60}m {elapsed_seconds % 60}s"
                    )
                else:
                    self.logger.info(
                        f"📈 Episode progress: Step {self.current_step} | "
                        f"Episode {self.episode_number} | Sim time: {current_time.strftime('%H:%M:%S') if hasattr(current_time, 'strftime') else str(current_time)} | "
                        f"Elapsed: {elapsed_seconds // 60}m {elapsed_seconds % 60}s"
                    )
        if market_state_at_decision is None:
            self.logger.error("Market simulator returned invalid state")
            return (
                self._last_observation,
                0.0,
                True,
                False,
                {"error": "Market state unavailable"},
            )

        current_sim_time_decision = market_state_at_decision["timestamp"]

        # Check if we've reached episode end time
        if current_sim_time_decision >= self.episode_end_time_utc:
            self.logger.info(
                f"Episode end time reached: {current_sim_time_decision} >= {self.episode_end_time_utc}"
            )
            return (
                self._last_observation,
                0.0,
                True,
                False,
                {"termination_reason": TerminationReasonEnum.MAX_DURATION.value},
            )

        # Get portfolio state before action
        self._last_portfolio_state_before_action = (
            self.portfolio_manager.get_portfolio_state(current_sim_time_decision)
        )

        # Apply action masking to validate action before execution
        if self.action_mask:
            # Convert raw action to linear index for validation
            if hasattr(action, "__len__") and len(action) >= 2:
                action_type_idx = int(action[0]) % 3
                size_idx = int(action[1]) % 4
                linear_action_idx = action_type_idx * 4 + size_idx

                # Check if action is valid
                is_valid = self.action_mask.is_action_valid(
                    linear_action_idx,
                    self._last_portfolio_state_before_action,
                    market_state_at_decision,
                )

                if not is_valid:
                    self.logger.debug(
                        f"Invalid action masked: {self.action_mask.get_action_description(linear_action_idx)} "
                        f"Valid actions: {self.action_mask.get_valid_actions(self._last_portfolio_state_before_action, market_state_at_decision)}"
                    )
                    # Force to HOLD action (always valid)
                    action = [0, 0]  # HOLD with 25% size (ignored anyway)

        # Execute action through ExecutionSimulator (handles decode -> validate -> execute)
        execution_result = self.execution_manager.execute_action(
            raw_action=action,
            market_state=market_state_at_decision,
            portfolio_state=self._last_portfolio_state_before_action,
            primary_asset=self.primary_asset,
            portfolio_manager=self.portfolio_manager,
        )

        # Store decoded action for metrics and reward calculation
        self._last_decoded_action = execution_result.action_decode_result.to_dict()



        # Handle fill if order was executed
        fill_details_list: List[FillDetails] = []
        if execution_result.fill_details:
            # Process fill and get enriched details for reward system
            enriched_fill = self.portfolio_manager.process_fill(
                execution_result.fill_details
            )
            fill_details_list.append(enriched_fill)


        # Update portfolio with current market prices
        time_for_pf_update = (
            fill_details_list[-1].fill_timestamp
            if fill_details_list
            else current_sim_time_decision
        )
        current_price = market_state_at_decision.get("current_price", 0.0)
        if current_price <= 0:
            ask = market_state_at_decision.get("best_ask_price", 0)
            bid = market_state_at_decision.get("best_bid_price", 0)
            if ask > 0 and bid > 0:
                current_price = (ask + bid) / 2

        prices_at_decision = {self.primary_asset: current_price}
        self.portfolio_manager.update_market_values(
            prices_at_decision, time_for_pf_update
        )
        portfolio_state_after_action = self.portfolio_manager.get_portfolio_state(
            time_for_pf_update
        )

        # Advance market simulator
        market_advanced = self.market_simulator.step()
        market_state_next_t = None
        next_sim_time = None

        if market_advanced:
            market_state_next_t = self.market_simulator.get_current_market_data()
            if market_state_next_t and "timestamp" in market_state_next_t:
                next_sim_time = market_state_next_t["timestamp"]
            else:
                market_advanced = False

        # Update portfolio with next market prices
        if market_state_next_t and next_sim_time:
            next_price = market_state_next_t.get("current_price", 0.0)
            if next_price <= 0:
                ask = market_state_next_t.get("best_ask_price", 0)
                bid = market_state_next_t.get("best_bid_price", 0)
                if ask > 0 and bid > 0:
                    next_price = (ask + bid) / 2

            prices_at_next_time = {self.primary_asset: next_price}
            self.portfolio_manager.update_market_values(
                prices_at_next_time, next_sim_time
            )

        portfolio_state_next_t = self.portfolio_manager.get_portfolio_state(
            next_sim_time or time_for_pf_update
        )


        # Get next observation
        observation_next_t = None
        terminated_by_obs_failure = False

        if market_state_next_t and next_sim_time:
            observation_next_t = self._get_observation()
            if observation_next_t is None:
                observation_next_t = self._last_observation
                terminated_by_obs_failure = True
        else:
            observation_next_t = self._last_observation

        if observation_next_t is None:
            observation_next_t = self._get_dummy_observation()
            terminated_by_obs_failure = True

        self._last_observation = observation_next_t

        # Check termination conditions
        terminated = False
        truncated = False
        termination_reason: Optional[TerminationReasonEnum] = None

        # Bankruptcy check (skip if initial capital is zero to avoid division by zero)
        if (
            self.initial_capital_for_session is not None
            and self.initial_capital_for_session > 0
            and current_equity
            <= self.initial_capital_for_session * self.bankruptcy_threshold_factor
        ):
            terminated = True
            termination_reason = TerminationReasonEnum.BANKRUPTCY

        # Max loss check (skip if initial capital is zero to avoid division by zero)
        elif (
            self.initial_capital_for_session is not None
            and self.initial_capital_for_session > 0
            and current_equity
            <= self.initial_capital_for_session * (1 - self.max_session_loss_percentage)
        ):
            terminated = True
            termination_reason = TerminationReasonEnum.MAX_LOSS_REACHED

        # Observation failure
        elif terminated_by_obs_failure:
            terminated = True
            termination_reason = TerminationReasonEnum.OBSERVATION_FAILURE

        # End of data
        elif not market_advanced:
            terminated = True
            termination_reason = TerminationReasonEnum.END_OF_SESSION_DATA

        # Natural episode end (no penalty)
        elif (
            self.max_steps is not None
            and self.max_steps > 0
            and self.current_step >= self.max_steps
        ):
            terminated = True
            termination_reason = (
                TerminationReasonEnum.MAX_DURATION
            )  # Changed to MAX_DURATION (no penalty)

        # Training step limit reached (with penalty)
        elif (
            self.max_training_steps is not None
            and self.max_training_steps > 0
            and self.current_step >= self.max_training_steps
        ):
            terminated = True
            termination_reason = (
                TerminationReasonEnum.MAX_STEPS_REACHED
            )  # This gets penalty

        # Episode time limit reached
        elif next_sim_time and next_sim_time >= self.episode_end_time_utc:
            terminated = True
            termination_reason = TerminationReasonEnum.MAX_DURATION

        # Update state tracking
        self.is_terminated = terminated
        self.is_truncated = truncated
        if termination_reason:
            self.current_termination_reason = termination_reason.value

        # Episode completion logging is done in _log_episode_completion method to avoid duplication

        # Calculate reward
        reward = self.reward_calculator.calculate(
            portfolio_state_before_action=self._last_portfolio_state_before_action,
            portfolio_state_after_action_fills=portfolio_state_after_action,
            portfolio_state_next_t=portfolio_state_next_t,
            market_state_at_decision=market_state_at_decision,
            market_state_next_t=market_state_next_t,
            decoded_action=self._last_decoded_action,
            fill_details_list=fill_details_list,
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
        )
        self.episode_total_reward += reward

        # Create info dict
        info = self._get_current_info(
            reward=reward,
            fill_details_list=fill_details_list,
            current_portfolio_state_for_info=portfolio_state_next_t,
            termination_reason_enum=termination_reason,
            is_terminated=terminated,
            is_truncated=truncated,
        )


        # Episode end handling
        if terminated or truncated:
            self._handle_episode_end(portfolio_state_next_t, info)

        return observation_next_t, reward, terminated, truncated, info



    def _handle_episode_end(
        self, portfolio_state: PortfolioState, info: Dict[str, Any]
    ):
        """Handle episode termination."""
        # Minimal episode end logging
        final_equity = (
            portfolio_state.get("total_equity", 0.0)
            if hasattr(portfolio_state, "get")
            else 0.0
        )
        pnl = final_equity - (self.initial_capital_for_session or 0.0)
        pnl_pct = (
            (pnl / self.initial_capital_for_session) * 100
            if self.initial_capital_for_session and self.initial_capital_for_session > 0
            else 0
        )

        self.logger.info(
            f"🏁 Episode {self.episode_number}: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Reward: {self.episode_total_reward:.4f} | Steps: {self.current_step}"
        )

    def get_next_reset_point(self) -> Optional[Dict]:
        """Get the next available reset point."""
        if self.current_reset_idx + 1 < len(self.reset_points):
            return self.reset_points[self.current_reset_idx + 1]
        return None

    def has_more_reset_points(self) -> bool:
        """Check if there are more reset points available."""
        return self.current_reset_idx + 1 < len(self.reset_points)

    def _get_current_info(
        self,
        reward: float,
        current_portfolio_state_for_info: PortfolioState,
        fill_details_list: Optional[List[FillDetails]] = None,
        termination_reason_enum: Optional[TerminationReasonEnum] = None,
        is_terminated: bool = False,
        is_truncated: bool = False,
    ) -> Dict[str, Any]:
        """Create minimal info dictionary for step."""
        # Build minimal info dictionary with only essential fields
        info = {
            "step": self.current_step,
            "episode_number": self.episode_number,
            "reward_step": reward,
            "episode_cumulative_reward": self.episode_total_reward,
        }

        # Termination info
        if is_terminated and termination_reason_enum:
            info["termination_reason"] = termination_reason_enum.value
        if is_truncated:
            info["TimeLimit.truncated"] = True

        return info







    def render(self, info_dict: Optional[Dict[str, Any]] = None):
        """Basic render method."""
        if self.render_mode in ["human", "logs"] and info_dict:
            print(
                f"Step {info_dict.get('step', 'N/A')}: "
                f"Reward {info_dict.get('reward_step', 0.0):.4f}, "
                f"Equity ${info_dict.get('portfolio_equity', 0.0):.2f}"
            )

    def get_action_mask(self) -> Optional[np.ndarray]:
        """
        Get current action mask for the agent.

        Returns:
            Boolean array of shape (12,) where True = valid action, or None if masking disabled
        """
        if (
            not self.action_mask
            or not self.portfolio_manager
            or not self.market_simulator
        ):
            return None

        # Get current states
        current_time = self.market_simulator.get_current_time()
        portfolio_state = self.portfolio_manager.get_portfolio_state(current_time)
        market_state = self.market_simulator.get_current_market_data()

        if not portfolio_state or not market_state:
            return None

        return self.action_mask.get_action_mask(portfolio_state, market_state)

    def mask_action_probabilities(self, action_probs: np.ndarray) -> np.ndarray:
        """
        Apply action masking to action probabilities.

        Args:
            action_probs: Raw action probabilities from policy network

        Returns:
            Masked and renormalized action probabilities
        """
        if (
            not self.action_mask
            or not self.portfolio_manager
            or not self.market_simulator
        ):
            return action_probs

        # Get current states
        current_time = self.market_simulator.get_current_time()
        portfolio_state = self.portfolio_manager.get_portfolio_state(current_time)
        market_state = self.market_simulator.get_current_market_data()

        if not portfolio_state or not market_state:
            return action_probs

        return self.action_mask.mask_action_probabilities(
            action_probs, portfolio_state, market_state
        )

    def close(self):
        """Close environment and cleanup resources."""
        if self.market_simulator and hasattr(self.market_simulator, "close"):
            self.market_simulator.close()
        if self.next_market_simulator and hasattr(self.next_market_simulator, "close"):
            self.next_market_simulator.close()
        self.logger.info("🔒 TradingEnvironment closed")