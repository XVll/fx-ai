"""
Trading Environment - Redesigned for momentum-based low-float trading

This implementation uses the new architecture:
- Pre-calculated features from MarketSimulator
- Momentum-based episode selection
- Day-based training with reset points
- Position handling across episode boundaries
"""

import logging
from datetime import datetime, timedelta, time, timezone
from enum import Enum
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
    """Reasons for episode termination."""
    END_OF_SESSION_DATA = "END_OF_SESSION_DATA"
    MAX_LOSS_REACHED = "MAX_LOSS_REACHED"
    BANKRUPTCY = "BANKRUPTCY"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"
    SETUP_FAILURE = "SETUP_FAILURE"
    INVALID_ACTION_LIMIT_REACHED = "INVALID_ACTION_LIMIT_REACHED"
    MARKET_CLOSE = "MARKET_CLOSE"
    MAX_DURATION = "MAX_DURATION"


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
    
    metadata = {'render_modes': ['human', 'logs', 'none'], 'render_fps': 10}

    def __init__(self, config: Config, data_manager: DataManager, logger: Optional[logging.Logger] = None,
                 callback_manager=None):
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
        self.action_space = spaces.MultiDiscrete([3, 4])  # [action_types, position_sizes]

        # Observation Space - same as before
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
        })

        # Core components - initialized in setup_session
        self.market_simulator: Optional[MarketSimulator] = None
        self.next_market_simulator: Optional[MarketSimulator] = None  # For background preparation
        self.execution_manager: Optional[ExecutionSimulator] = None
        self.portfolio_manager: Optional[PortfolioSimulator] = None
        self.reward_calculator: Optional[RewardSystem] = None
        self.action_mask: Optional[ActionMask] = None

        # Episode state
        self.current_step: int = 0
        self.max_episode_steps: int = config.env.max_episode_steps  # Natural episode length (no penalty)
        self.max_training_steps: Optional[int] = config.env.max_training_steps  # Training limit (with penalty)
        self.max_steps: int = config.env.max_episode_steps  # Legacy compatibility
        self.invalid_action_count_episode: int = 0
        self.episode_total_reward: float = 0.0
        self.initial_capital_for_session: float = self.config.simulation.initial_capital  # Initialize with config value
        self.episode_number: int = 0
        self.episode_start_time: float = 0.0

        # Episode boundaries and reset points
        self.current_session_date: Optional[datetime] = None
        self.episode_start_time_utc: Optional[datetime] = None
        self.episode_end_time_utc: Optional[datetime] = None
        self.reset_points: List[Dict] = []
        self.current_reset_idx: int = 0

        # Performance tracking
        self.episode_fills: List[FillDetails] = []
        self.episode_peak_equity: float = 0.0
        self.episode_max_drawdown: float = 0.0
        self.action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
        self.win_loss_counts = {"wins": 0, "losses": 0}

        # State tracking
        self._last_observation: Optional[Dict[str, np.ndarray]] = None
        self._last_portfolio_state_before_action: Optional[PortfolioState] = None
        self._last_decoded_action: Optional[Dict[str, Any]] = None
        self.current_termination_reason: Optional[str] = None
        self.is_terminated: bool = False
        self.is_truncated: bool = False

        # Training info
        self.total_episodes: int = 0
        self.total_steps: int = 0
        self.update_count: int = 0

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

        self.logger.info(f"🎯 Setting up session: {self.primary_asset} on {self.current_session_date.strftime('%Y-%m-%d')}")

        # Create MarketSimulator for this session
        self.market_simulator = MarketSimulator(
            symbol=self.primary_asset,
            data_manager=self.data_manager,
            model_config=self.config.model,
            simulation_config=self.config.simulation
        )

        # Initialize day - this pre-calculates ALL features for the entire day
        success = self.market_simulator.initialize_day(self.current_session_date)
        if not success:
            raise ValueError(f"Failed to initialize {symbol} on {self.current_session_date}")

        # Get session stats
        stats = self.market_simulator.get_stats()
        self.logger.info(f"✅ Session ready: {stats['total_seconds']} seconds, "
                        f"warmup: {stats['warmup_info']['has_warmup']}")

        # Load reset points from momentum indices (if available) or fallback to fixed points
        self.reset_points = self._generate_reset_points()
        self.current_reset_idx = 0

        # Initialize other components
        self._initialize_simulators()

        self.logger.info(f"🔄 {len(self.reset_points)} reset points available for training")

    def _generate_fixed_reset_points(self) -> List[Dict]:
        """Generate fixed reset points based on market hours."""
        reset_points = []
        base_date = self.current_session_date.date()
        
        # Fixed reset times (ET) - convert to UTC
        fixed_times = [
            time(9, 30),   # Market open
            time(10, 30),  # Post-open settlement  
            time(14, 0),   # Afternoon session
            time(15, 30),  # Power hour
        ]
        
        for reset_time in fixed_times:
            reset_dt = datetime.combine(base_date, reset_time)
            # Convert ET to UTC (assuming EST/EDT handling is done elsewhere)
            reset_dt_utc = pd.Timestamp(reset_dt, tz='US/Eastern').tz_convert('UTC').to_pydatetime()
            
            reset_points.append({
                'timestamp': reset_dt_utc,
                'activity_score': 0.5,  # Default activity score
                'combined_score': 0.5,  # Default combined score
                'max_duration_hours': 4,
                'reset_type': 'fixed'
            })
        
        return reset_points
        
    def _generate_reset_points(self) -> List[Dict]:
        """Generate reset points using momentum indices with intelligent fallback for gaps."""
        # Try to get momentum-based reset points from data manager
        momentum_reset_points = self.data_manager.get_reset_points(
            self.primary_asset, 
            self.current_session_date
        )
        
        if not momentum_reset_points.empty:
            # Use momentum index reset points directly
            reset_points = []
            for _, row in momentum_reset_points.iterrows():
                reset_points.append({
                    'timestamp': row['timestamp'],
                    'activity_score': row.get('activity_score', 0.5),
                    'combined_score': row.get('combined_score', 0.5),
                    'day_activity_score': row.get('day_activity_score', 0.5),
                    # Add 2-component scores for curriculum system
                    'roc_score': row.get('roc_score', 0.0),
                    'max_duration_hours': self._get_duration_for_activity(row.get('activity_score', 0.5)),
                    'reset_type': 'momentum',
                    'volume_ratio': row.get('volume_ratio', 1.0),
                    'price_change': row.get('price_change', 0.0),
                    # Additional fields from scanner for completeness
                    'price': row.get('price', 0.0),
                    'volume': row.get('volume', 0),
                    'session': row.get('session', 'regular')
                })
            
            # Check for early trading hour gaps and supplement with fixed points if needed
            reset_points = self._supplement_with_early_fixed_points(reset_points)
            
            self.logger.info(f"Using {len(reset_points)} reset points (momentum + early fixed supplements)")
            return reset_points
            
        else:
            # Fallback to fixed reset points
            self.logger.info("No momentum reset points found, using fixed schedule")
            return self._generate_fixed_reset_points()
    
    def _supplement_with_early_fixed_points(self, momentum_reset_points: List[Dict]) -> List[Dict]:
        """Supplement momentum reset points with fixed early points if there are gaps."""
        if not momentum_reset_points:
            return momentum_reset_points
            
        # Find the earliest momentum reset point
        earliest_momentum = min(rp['timestamp'] for rp in momentum_reset_points)
        
        # Trading session starts at 4 AM ET, check if we have coverage
        base_date = self.current_session_date.date()
        session_start_et = datetime.combine(base_date, time(4, 0))
        session_start_utc = pd.Timestamp(session_start_et, tz='US/Eastern').tz_convert('UTC').to_pydatetime()
        
        # If momentum points start after 10 AM ET, add early fixed points
        cutoff_et = datetime.combine(base_date, time(10, 0))
        cutoff_utc = pd.Timestamp(cutoff_et, tz='US/Eastern').tz_convert('UTC').to_pydatetime()
        
        if earliest_momentum > cutoff_utc:
            # Add early fixed reset points to cover the gap
            early_fixed_times = [
                time(6, 0),    # Pre-market
                time(9, 30),   # Market open
            ]
            
            early_points = []
            for reset_time in early_fixed_times:
                reset_dt = datetime.combine(base_date, reset_time)
                reset_dt_utc = pd.Timestamp(reset_dt, tz='US/Eastern').tz_convert('UTC').to_pydatetime()
                
                # Only add if it's before the earliest momentum point
                if reset_dt_utc < earliest_momentum:
                    # Try to get a price for this timestamp from market data
                    price = self._get_price_at_timestamp(reset_dt_utc)
                    
                    early_points.append({
                        'timestamp': reset_dt_utc,
                        'price': price,
                        'activity_score': 0.3,  # Lower activity for early supplemental points
                        'combined_score': 0.3,
                        'max_duration_hours': 4,
                        'reset_type': 'early_fixed_supplement'
                    })
            
            if early_points:
                self.logger.info(f"Added {len(early_points)} early fixed reset points to supplement momentum data")
                # Combine and sort by timestamp
                all_points = early_points + momentum_reset_points
                all_points.sort(key=lambda x: x['timestamp'])
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
        activity_score = reset_point.get('activity_score', 0.5)
        combined_score = reset_point.get('combined_score', 0.5)
        reset_type = reset_point.get('reset_type', 'momentum')
        
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
            'momentum': 1.0,      # Standard for momentum-based points
            'fixed': 2.0          # Wider for fixed points
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
            trade_callback=self._on_trade_completed
        )

        # Reward system
        self.reward_calculator = RewardSystem(
            config=self.config.env.reward,
            callback_manager=self.callback_manager,
            logger=logging.getLogger(f"{__name__}.RewardSystem")
        )

        # Execution simulator
        self.execution_manager = ExecutionSimulator(
            logger=logging.getLogger(f"{__name__}.ExecSim"),
            simulation_config=self.config.simulation,
            np_random=self.np_random,
            market_simulator=self.market_simulator
        )

        # Action masking system
        self.action_mask = ActionMask(
            config=self.config,
            logger=logging.getLogger(f"{__name__}.ActionMask")
        )

        self.logger.info("✅ All simulators initialized")

    def prepare_next_session(self, symbol: str, date: Union[str, datetime]):
        """Prepare next session in background for fast switching."""
        if isinstance(date, str):
            next_date = pd.Timestamp(date).to_pydatetime()
        else:
            next_date = date

        self.logger.info(f"🔄 Preparing next session: {symbol} on {next_date.strftime('%Y-%m-%d')}")

        # Create MarketSimulator for next session
        self.next_market_simulator = MarketSimulator(
            symbol=symbol,
            data_manager=self.data_manager,
            model_config=self.config.model,
            simulation_config=self.config.simulation
        )

        # Initialize in background
        success = self.next_market_simulator.initialize_day(next_date)
        if success:
            self.logger.info(f"✅ Next session ready: {symbol} {next_date.strftime('%Y-%m-%d')}")
        else:
            self.logger.error(f"❌ Failed to prepare next session: {symbol} {next_date.strftime('%Y-%m-%d')}")

    def switch_to_next_session(self):
        """Switch to the prepared next session."""
        if self.next_market_simulator is None:
            raise ValueError("No next session prepared. Call prepare_next_session first.")
        
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

    def select_next_momentum_day(self, exclude_dates: Optional[List[datetime]] = None) -> Optional[Dict]:
        """Select next momentum day based on quality and curriculum."""
        momentum_days = self.get_momentum_days(min_activity=0.0)
        
        if momentum_days.empty:
            return None
            
        # Apply exclusions
        if exclude_dates:
            exclude_dates_only = [d.date() if isinstance(d, datetime) else d for d in exclude_dates]
            momentum_days = momentum_days[~momentum_days['date'].dt.date.isin(exclude_dates_only)]
            
        if momentum_days.empty:
            return None
            
        # Simple selection by quality score (highest first)
        best_day = momentum_days.iloc[0]
        
        return {
            'symbol': best_day['symbol'],
            'date': best_day['date'],
            'quality_score': best_day['activity_score'],  # Map activity_score to quality_score
            'max_intraday_move': best_day.get('max_intraday_move', 0.0),
            'volume_multiplier': best_day.get('volume_multiplier', 1.0)
        }

    def reset_at_point(self, reset_point_idx: int = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset to a specific reset point within the loaded day.
        This is the main reset method for momentum-based training.
        """
        if reset_point_idx is None:
            reset_point_idx = self.current_reset_idx

        if reset_point_idx >= len(self.reset_points):
            self.logger.error(f"Reset point index {reset_point_idx} out of range (max: {len(self.reset_points)-1})")
            return self._get_dummy_observation(), {"error": "Invalid reset point"}

        reset_point = self.reset_points[reset_point_idx]
        self.current_reset_idx = reset_point_idx
        
        # Handle any open positions before reset
        position_close_pnl = self._handle_open_positions_at_reset()

        # Reset episode state
        self.current_step = 0
        self.invalid_action_count_episode = 0
        self.episode_total_reward = 0.0
        self.episode_start_time = datetime.now().timestamp()
        
        # Reset performance tracking
        self.episode_fills = []
        self.episode_peak_equity = 0.0
        self.episode_max_drawdown = 0.0
        self.action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
        self.win_loss_counts = {"wins": 0, "losses": 0}
        
        # Reset state tracking
        self.current_termination_reason = None
        self.is_terminated = False
        self.is_truncated = False
        self._last_decoded_action = None

        # Increment episode number
        self.episode_number += 1

        # Set episode boundaries
        self.episode_start_time_utc = reset_point['timestamp']
        max_duration = timedelta(hours=reset_point.get('max_duration_hours', 4))
        market_close = datetime.combine(self.current_session_date.date(), time(20, 0))  # 8 PM ET -> UTC
        market_close_utc = pd.Timestamp(market_close, tz='US/Eastern').tz_convert('UTC').to_pydatetime()
        
        self.episode_end_time_utc = min(
            self.episode_start_time_utc + max_duration,
            market_close_utc
        )

        # Start episode tracking
        if self.callback_manager:
            self.callback_manager.trigger('on_episode_start', 
                                        self.episode_number,
                                        {
                                            'symbol': self.primary_asset,
                                            'date': self.current_session_date,
                                            'start_time': self.episode_start_time_utc,
                                            'reset_point': reset_point
                                        })
            
        # Reset dashboard episode counters
        try:
            from dashboard.shared_state import dashboard_state
            dashboard_state.reset_episode_counters()
        except ImportError:
            pass  # Dashboard not available

        # Reset market simulator with adaptive randomization
        # Adjust randomization window based on pattern type and quality
        max_offset_minutes = self._get_adaptive_randomization_window(reset_point)
        max_offset_seconds = max_offset_minutes * 60
        
        random_offset_seconds = self.np_random.integers(-max_offset_seconds, max_offset_seconds + 1)
        randomized_start = self.episode_start_time_utc + timedelta(seconds=int(random_offset_seconds))
        
        # Log episode reset info
        original_time = reset_point['timestamp'].strftime('%H:%M:%S')
        randomized_time = randomized_start.strftime('%H:%M:%S') 
        offset_minutes = random_offset_seconds // 60
        window_minutes = max_offset_minutes
        
        self.logger.info(f"🎯 Episode {self.episode_number} reset: {original_time} → {randomized_time} "
                        f"({offset_minutes:+d}m/±{window_minutes}m) | Activity: {reset_point.get('activity_score', 0.5):.2f} "
                        f"| Combined: {reset_point.get('combined_score', 0.5):.2f} | ROC: {reset_point.get('roc_score', 0.0):.2f} | {reset_point.get('reset_type', 'momentum')}")
        
        # Ensure we don't go before 4 AM or after 8 PM
        market_open = datetime.combine(self.current_session_date.date(), time(4, 0))
        market_open_utc = pd.Timestamp(market_open, tz='US/Eastern').tz_convert('UTC').to_pydatetime()
        market_close_utc = pd.Timestamp(market_open, tz='US/Eastern').tz_convert('UTC').to_pydatetime().replace(hour=20)
        
        randomized_start = max(randomized_start, market_open_utc)
        randomized_start = min(randomized_start, market_close_utc - timedelta(hours=1))  # At least 1 hour before close
        
        # Update episode end time based on randomized start
        max_duration = timedelta(hours=reset_point.get('max_duration_hours', 4))
        self.episode_end_time_utc = min(
            randomized_start + max_duration,
            market_close_utc
        )
        
        # Update episode start time to randomized time
        self.episode_start_time_utc = randomized_start
        
        # Reset market simulator and set to randomized start time
        if not self.market_simulator.reset():
            self.logger.error("Market simulator failed to reset")
            return self._get_dummy_observation(), {"error": "Market simulator reset failed"}
            
        if not self.market_simulator.set_time(randomized_start):
            self.logger.error(f"Failed to set market simulator time to {randomized_start}")
            return self._get_dummy_observation(), {"error": "Failed to set market time"}
            
        initial_market_state = self.market_simulator.get_market_state()
        
        if initial_market_state is None:
            self.logger.error("Market simulator failed to reset")
            return self._get_dummy_observation(), {"error": "Market simulator reset failed"}

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
        # self.logger.debug(f"DEBUG: Setting episode peak equity to {self.initial_capital_for_session}")
        self.episode_peak_equity = self.initial_capital_for_session
        # self.logger.debug(f"DEBUG: Episode peak equity set")

        if hasattr(self.reward_calculator, 'reset'):
            # self.logger.debug(f"DEBUG: About to reset reward calculator")
            self.reward_calculator.reset()
            # self.logger.debug(f"DEBUG: Reward calculator reset completed")

        # Get initial observation using pre-calculated features
        # self.logger.debug(f"DEBUG: About to get initial observation")
        self._last_observation = self._get_observation()
        # self.logger.debug(f"DEBUG: Got initial observation: {self._last_observation is not None}")
        if self._last_observation is None:
            self.logger.error("Failed to get initial observation")
            return self._get_dummy_observation(), {"error": "Initial observation failed"}

        self._last_portfolio_state_before_action = self.portfolio_manager.get_portfolio_state(current_sim_time)
        
        # Trigger episode visualization start
        if self.callback_manager:
            episode_date = current_sim_time.strftime('%Y-%m-%d')
            self.callback_manager.trigger('on_custom_event',
                                        'episode_visualization_start',
                                        {
                                            'episode_number': self.episode_number,
                                            'symbol': self.primary_asset,
                                            'date': episode_date
                                        })

        initial_info = self._get_current_info(
            reward=0.0,
            current_portfolio_state_for_info=self._last_portfolio_state_before_action
        )
        
        # Include position close P&L in initial info
        if position_close_pnl is not None:
            initial_info['position_close_pnl'] = position_close_pnl
            initial_info['had_open_position_at_reset'] = True
        else:
            initial_info['had_open_position_at_reset'] = False

        # Update dashboard with quality metrics from reset point
        self._update_dashboard_quality_metrics(reset_point)
        
        # Send initial chart data immediately at episode start to reduce display delay
        self._send_initial_chart_data()

        return self._last_observation, initial_info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
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
        reset_point_idx = options.get('reset_point_idx', 0)
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
                
            current_sim_time = market_state['timestamp']
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
            current_portfolio_state = self.portfolio_manager.get_portfolio_state(current_sim_time)
            # self.logger.debug(f"DEBUG: Got portfolio state")
            
            # self.logger.debug(f"DEBUG: About to get portfolio observation")
            portfolio_obs_component = self.portfolio_manager.get_portfolio_observation()
            # self.logger.debug(f"DEBUG: Got portfolio observation component")
            
            portfolio_features_array = portfolio_obs_component['features']
            # self.logger.debug(f"DEBUG: Got portfolio features array with shape {portfolio_features_array.shape if hasattr(portfolio_features_array, 'shape') else 'unknown'}")

            # Construct observation dict
            obs = {
                'hf': features.get('hf'),
                'mf': features.get('mf'), 
                'lf': features.get('lf'),
                'portfolio': portfolio_features_array
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
                    self.logger.error(f"Shape mismatch for observation key '{key}'. "
                                     f"Expected {expected_shape}, Got {obs[key].shape}")
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



    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one environment step - simplified to use ExecutionSimulator."""
        if self._last_observation is None or self.primary_asset is None:
            self.logger.error("Step called with invalid state")
            return self._get_dummy_observation(), 0.0, True, False, {"error": "Invalid state"}

        self.current_step += 1

        # Get current market state
        market_state_at_decision = self.market_simulator.get_current_market_data()
        
        # Log progress every 50 steps
        if self.current_step % 50 == 0:
            if market_state_at_decision:
                current_time = market_state_at_decision['timestamp']
                elapsed_seconds = self.current_step  # Each step is 1 second
                # For episode progress, use max_episode_steps if set, otherwise show unbounded progress
                if self.max_episode_steps:
                    progress_pct = (self.current_step / self.max_episode_steps * 100)
                    self.logger.info(f"📈 Episode progress: Step {self.current_step}/{self.max_episode_steps} ({progress_pct:.1f}%) | "
                                   f"Episode {self.episode_number} | Sim time: {current_time.strftime('%H:%M:%S')} | "
                                   f"Elapsed: {elapsed_seconds//60}m {elapsed_seconds%60}s")
                else:
                    self.logger.info(f"📈 Episode progress: Step {self.current_step} | "
                                   f"Episode {self.episode_number} | Sim time: {current_time.strftime('%H:%M:%S')} | "
                                   f"Elapsed: {elapsed_seconds//60}m {elapsed_seconds%60}s")
        if market_state_at_decision is None:
            self.logger.error("Market simulator returned invalid state")
            return self._last_observation, 0.0, True, False, {"error": "Market state unavailable"}

        current_sim_time_decision = market_state_at_decision['timestamp']
        
        # Check if we've reached episode end time
        if current_sim_time_decision >= self.episode_end_time_utc:
            self.logger.info(f"Episode end time reached: {current_sim_time_decision} >= {self.episode_end_time_utc}")
            return self._last_observation, 0.0, True, False, {
                "termination_reason": TerminationReasonEnum.MAX_DURATION.value
            }

        # Get portfolio state before action
        self._last_portfolio_state_before_action = self.portfolio_manager.get_portfolio_state(current_sim_time_decision)

        # Apply action masking to validate action before execution
        if self.action_mask:
            # Convert raw action to linear index for validation
            if hasattr(action, '__len__') and len(action) >= 2:
                action_type_idx = int(action[0]) % 3
                size_idx = int(action[1]) % 4
                linear_action_idx = action_type_idx * 4 + size_idx
                
                # Check if action is valid
                is_valid = self.action_mask.is_action_valid(
                    linear_action_idx, 
                    self._last_portfolio_state_before_action, 
                    market_state_at_decision
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
            portfolio_manager=self.portfolio_manager
        )
        
        # Store decoded action for metrics and reward calculation
        self._last_decoded_action = execution_result.action_decode_result.to_dict()
        
        # Track action counts and invalid actions
        action_name = execution_result.action_decode_result.action_type
        self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
        if not execution_result.action_decode_result.is_valid:
            self.invalid_action_count_episode += 1
            
        # Emit action to dashboard (invalid actions no longer tracked due to action masking)
        from dashboard.event_stream import event_stream
        event_stream.emit_action_decision(
            action=action_name,
            confidence=1.0,  # Default confidence
            reasoning={},
            features={},
            is_invalid=False  # Action masking prevents invalid actions
        )

        # Handle fill if order was executed
        fill_details_list: List[FillDetails] = []
        if execution_result.fill_details:
            # Process fill and get enriched details for reward system
            enriched_fill = self.portfolio_manager.process_fill(execution_result.fill_details)
            fill_details_list.append(enriched_fill)
            self.episode_fills.append(enriched_fill)
            
            # Emit enriched fill event to dashboard with correct P&L
            from dashboard.event_stream import event_stream, EventType
            event_stream.emit_trade(
                side=enriched_fill.order_side.name,
                quantity=int(enriched_fill.executed_quantity),
                price=enriched_fill.executed_price,  # Use actual execution price
                fill_price=enriched_fill.executed_price,
                pnl=enriched_fill.realized_pnl or 0.0,  # Use portfolio-calculated P&L
                commission=enriched_fill.commission,
                order_id=f"fill_{enriched_fill.fill_timestamp.timestamp():.0f}",
                slippage_cost=enriched_fill.slippage_cost_total,
                timestamp=enriched_fill.fill_timestamp,
                closes_position=enriched_fill.closes_position,
                holding_time_minutes=enriched_fill.holding_time_minutes
            )

        # Update portfolio with current market prices
        time_for_pf_update = fill_details_list[-1].fill_timestamp if fill_details_list else current_sim_time_decision
        current_price = market_state_at_decision.get('current_price', 0.0)
        if current_price <= 0:
            ask = market_state_at_decision.get('best_ask_price', 0)
            bid = market_state_at_decision.get('best_bid_price', 0)
            if ask > 0 and bid > 0:
                current_price = (ask + bid) / 2

        prices_at_decision = {self.primary_asset: current_price}
        self.portfolio_manager.update_market_values(prices_at_decision, time_for_pf_update)
        portfolio_state_after_action = self.portfolio_manager.get_portfolio_state(time_for_pf_update)

        # Advance market simulator
        market_advanced = self.market_simulator.step()
        market_state_next_t = None
        next_sim_time = None

        if market_advanced:
            market_state_next_t = self.market_simulator.get_current_market_data()
            if market_state_next_t and 'timestamp' in market_state_next_t:
                next_sim_time = market_state_next_t['timestamp']
            else:
                market_advanced = False

        # Update portfolio with next market prices
        if market_state_next_t and next_sim_time:
            next_price = market_state_next_t.get('current_price', 0.0)
            if next_price <= 0:
                ask = market_state_next_t.get('best_ask_price', 0)
                bid = market_state_next_t.get('best_bid_price', 0)
                if ask > 0 and bid > 0:
                    next_price = (ask + bid) / 2

            prices_at_next_time = {self.primary_asset: next_price}
            self.portfolio_manager.update_market_values(prices_at_next_time, next_sim_time)

        portfolio_state_next_t = self.portfolio_manager.get_portfolio_state(
            next_sim_time or time_for_pf_update
        )

        # Track episode performance
        current_equity = portfolio_state_next_t.get('total_equity', 0.0)
        if current_equity is not None and current_equity > self.episode_peak_equity:
            self.episode_peak_equity = current_equity

        current_drawdown = 0.0
        if self.episode_peak_equity > 0 and current_equity is not None:
            current_drawdown = (self.episode_peak_equity - current_equity) / self.episode_peak_equity
        if current_drawdown > self.episode_max_drawdown:
            self.episode_max_drawdown = current_drawdown

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
        if (self.initial_capital_for_session is not None and 
            self.initial_capital_for_session > 0 and
            current_equity <= self.initial_capital_for_session * self.bankruptcy_threshold_factor):
            terminated = True
            termination_reason = TerminationReasonEnum.BANKRUPTCY

        # Max loss check (skip if initial capital is zero to avoid division by zero)
        elif (self.initial_capital_for_session is not None and 
              self.initial_capital_for_session > 0 and
              current_equity <= self.initial_capital_for_session * (1 - self.max_session_loss_percentage)):
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
        elif self.max_episode_steps is not None and self.max_episode_steps > 0 and self.current_step >= self.max_episode_steps:
            terminated = True
            termination_reason = TerminationReasonEnum.MAX_DURATION  # Changed to MAX_DURATION (no penalty)
            
        # Training step limit reached (with penalty)
        elif (self.max_training_steps is not None and 
              self.max_training_steps > 0 and 
              self.current_step >= self.max_training_steps):
            terminated = True
            termination_reason = TerminationReasonEnum.MAX_STEPS_REACHED  # This gets penalty
            
        # Episode time limit reached
        elif next_sim_time and next_sim_time >= self.episode_end_time_utc:
            terminated = True
            termination_reason = TerminationReasonEnum.MAX_DURATION

        # Update state tracking
        self.is_terminated = terminated
        self.is_truncated = truncated
        if termination_reason:
            self.current_termination_reason = termination_reason.value
            
        # Log episode completion
        if terminated or truncated:
            if current_sim_time_decision and self.episode_start_time_utc:
                episode_duration = (current_sim_time_decision - self.episode_start_time_utc).total_seconds()
                duration_str = f"Duration: {episode_duration/60:.1f}m | "
            else:
                duration_str = ""
            
            self.logger.info(f"🏁 Episode {self.episode_number} completed | "
                           f"Steps: {self.current_step} | {duration_str}"
                           f"Reason: {termination_reason.value if termination_reason else 'truncated'} | "
                           f"Total reward: {self.episode_total_reward:.2f}")

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
            termination_reason=termination_reason
        )
        self.episode_total_reward += reward

        # Create info dict
        info = self._get_current_info(
            reward=reward,
            fill_details_list=fill_details_list,
            current_portfolio_state_for_info=portfolio_state_next_t,
            termination_reason_enum=termination_reason,
            is_terminated=terminated,
            is_truncated=truncated
        )

        # Update callbacks with step data
        if self.callback_manager:
            self._trigger_step_callbacks(portfolio_state_next_t, market_state_next_t, reward, fill_details_list)

        # Episode end handling
        if terminated or truncated:
            self._handle_episode_end(portfolio_state_next_t, info)

        return observation_next_t, reward, terminated, truncated, info

    def _trigger_step_callbacks(self, portfolio_state: PortfolioState, market_state: Optional[Dict], 
                               reward: float, fill_details_list: List[FillDetails]):
        """Trigger callbacks for environment step."""
        if not self.callback_manager:
            return

        # Get action info
        action_name = self._last_decoded_action.get('action_type', 'UNKNOWN') if self._last_decoded_action else 'UNKNOWN'
        
        # Trigger episode step callback
        self.callback_manager.trigger('on_episode_step', {
            'state': self._last_observation,
            'action': action_name,
            'reward': reward,
            'next_state': None,  # Not needed for current callbacks
            'info': {
                'reward_components': self.reward_calculator.get_last_reward_components(),
                'episode_reward': self.episode_total_reward,
                'current_step': self.current_step,
                'max_steps': self.max_steps,
                'episode_number': self.episode_number
            },
            'step_num': self.current_step
        })

        # Trigger portfolio update callback
        self.callback_manager.trigger('on_portfolio_update', {
            'timestamp': portfolio_state.get('timestamp'),
            'equity': portfolio_state['total_equity'],
            'cash': portfolio_state['cash'],
            'unrealized_pnl': portfolio_state['unrealized_pnl'],
            'realized_pnl': portfolio_state['realized_pnl_session'],
            'session_metrics': portfolio_state['session_metrics'],
            'positions': portfolio_state['positions']
        })

        # Trigger fill callbacks
        for fill in fill_details_list:
            self.callback_manager.trigger('on_order_filled', {
                'symbol': fill.asset_id,
                'executed_quantity': fill.executed_quantity,
                'executed_price': fill.executed_price,
                'commission': fill.commission,
                'fees': fill.fees,
                'slippage_cost_total': fill.slippage_cost_total,
                'fill_timestamp': fill.fill_timestamp,
                'order_side': fill.order_side.name,
                'order_type': fill.order_type.name
            })
            
        # Update dashboard with candle data every 5 steps for more responsive charts
        if self.current_step % 5 == 0 and self.market_simulator:
            # Get ALL 1-minute bars for the CURRENT trading day only (not warmup)
            if hasattr(self.market_simulator, 'combined_bars_1m') and self.market_simulator.combined_bars_1m is not None:
                # Filter to only include current day data (4 AM to 8 PM ET)
                current_date = self.current_session_date.date()
                
                # Create time boundaries in ET (NY time)
                market_open_et = pd.Timestamp(f"{current_date} 04:00:00", tz='America/New_York')
                market_close_et = pd.Timestamp(f"{current_date} 20:00:00", tz='America/New_York')
                
                # Convert the entire day's 1m bars to list format for dashboard
                candle_list = []
                for timestamp, row in self.market_simulator.combined_bars_1m.iterrows():
                    # Convert timestamp to ET for comparison
                    ts = pd.Timestamp(timestamp)
                    if ts.tz is None:
                        # If no timezone, assume UTC
                        ts_et = ts.tz_localize('UTC').tz_convert('America/New_York')
                    else:
                        # Already has timezone, just convert
                        ts_et = ts.tz_convert('America/New_York')
                    
                    # Only include bars within the current trading day
                    if market_open_et <= ts_et <= market_close_et:
                        # Store timestamp in ET (NY time) for display, remove timezone
                        candle_list.append({
                            'timestamp': ts_et.replace(tzinfo=None).isoformat(),  # Remove tz for consistency
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row.get('volume', 0))
                        })
                
                if candle_list:
                    from dashboard.shared_state import dashboard_state
                    dashboard_state.update_candle_data(candle_list)

    def _handle_episode_end(self, portfolio_state: PortfolioState, info: Dict[str, Any]):
        """Handle episode termination."""
        episode_duration = datetime.now().timestamp() - self.episode_start_time
        try:
            final_metrics = self.portfolio_manager.get_trading_metrics()
            # Check if final_metrics is a real dict by trying to iterate its keys
            if not isinstance(final_metrics, dict):
                try:
                    list(final_metrics.keys())
                except (TypeError, AttributeError):
                    final_metrics = {}
        except (AttributeError, TypeError):
            final_metrics = {}

        # Episode summary - safely access portfolio state in case it's a mock
        final_equity = portfolio_state.get('total_equity', 0.0) if hasattr(portfolio_state, 'get') else 0.0
        realized_pnl = portfolio_state.get('realized_pnl_session', 0.0) if hasattr(portfolio_state, 'get') else 0.0
        session_metrics = portfolio_state.get('session_metrics', {}) if hasattr(portfolio_state, 'get') else {}
        
        # Only update info if it's a real dict, not a mock
        episode_summary = {
            "total_reward": self.episode_total_reward,
            "steps": self.current_step,
            "duration_seconds": episode_duration,
            "final_equity": final_equity,
            "peak_equity": self.episode_peak_equity,
            "max_drawdown_pct": self.episode_max_drawdown * 100,
            "session_realized_pnl_net": realized_pnl,
            "session_net_profit_equity_change": final_equity - (self.initial_capital_for_session or 0.0),
            "session_total_commissions": session_metrics.get('total_commissions_session', 0.0),
            "session_total_fees": session_metrics.get('total_fees_session', 0.0),
            "session_total_slippage_cost": session_metrics.get('total_slippage_cost_session', 0.0),
            "termination_reason": self.current_termination_reason or "UNKNOWN",
            "invalid_actions_in_episode": self.invalid_action_count_episode,
            "total_fills": len(self.episode_fills),
            **final_metrics
        }
        
        # Safely assign episode summary to info
        if hasattr(info, 'keys') and callable(getattr(info, 'keys')):
            try:
                info['episode_summary'] = episode_summary
            except (TypeError, AttributeError):
                pass  # Skip if info is a mock

        # End episode tracking
        if self.callback_manager:
            self.callback_manager.trigger('on_episode_end',
                                        self.episode_number,
                                        {
                                            'total_reward': self.episode_total_reward,
                                            'episode_length': self.current_step,
                                            'action_counts': self.action_counts,
                                            'final_portfolio_state': portfolio_state,
                                            'episode_summary': episode_summary
                                        })

        # Episode summary logging
        pnl = episode_summary.get('session_net_profit_equity_change', 0.0)
        pnl_pct = (pnl / self.initial_capital_for_session) * 100 if self.initial_capital_for_session > 0 else 0

        self.logger.info(f"🏁 EPISODE {self.episode_number} COMPLETE ({self.primary_asset})")
        self.logger.info(f"   💰 P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Reward: {self.episode_total_reward:.4f}")
        self.logger.info(f"   📊 Steps: {self.current_step} | Duration: {episode_duration:.1f}s")
        self.logger.info(f"   🔄 Reset Point: {self.current_reset_idx+1}/{len(self.reset_points)}")

    def get_next_reset_point(self) -> Optional[Dict]:
        """Get the next available reset point."""
        if self.current_reset_idx + 1 < len(self.reset_points):
            return self.reset_points[self.current_reset_idx + 1]
        return None

    def has_more_reset_points(self) -> bool:
        """Check if there are more reset points available."""
        return self.current_reset_idx + 1 < len(self.reset_points)

    def _get_current_info(self, reward: float, current_portfolio_state_for_info: PortfolioState,
                         fill_details_list: Optional[List[FillDetails]] = None,
                         termination_reason_enum: Optional[TerminationReasonEnum] = None,
                         is_terminated: bool = False, is_truncated: bool = False) -> Dict[str, Any]:
        """Create info dictionary for step."""
        # Get current market data
        current_price = 0.0
        bid_price = 0.0
        ask_price = 0.0
        volume = 0
        timestamp = current_portfolio_state_for_info.get('timestamp', datetime.now(timezone.utc))
        
        # Extract market data from market simulator if available
        if hasattr(self, 'market_simulator') and self.market_simulator:
            current_market_data = self.market_simulator.get_current_market_data()
            if current_market_data:
                current_price = current_market_data.get('close', current_market_data.get('price', 0.0))
                bid_price = current_market_data.get('bid', current_price)
                ask_price = current_market_data.get('ask', current_price)
                volume = current_market_data.get('volume', 0)
                # Use market simulator timestamp if available
                if 'timestamp' in current_market_data:
                    timestamp = current_market_data['timestamp']
        
        # Get position details for dashboard
        position_qty = 0.0
        position_side = "FLAT"
        avg_entry_price = 0.0
        
        if self.primary_asset:
            pos_detail = current_portfolio_state_for_info['positions'].get(self.primary_asset, {})
            position_qty = pos_detail.get('quantity', 0.0)
            position_side_enum = pos_detail.get('current_side', PositionSideEnum.FLAT)
            if hasattr(position_side_enum, 'value'):
                position_side = position_side_enum.value
            else:
                position_side = str(position_side_enum)
            avg_entry_price = pos_detail.get('avg_entry_price', 0.0)
        
        # Build comprehensive info dictionary
        info = {
            # Episode/Step tracking
            'timestamp': timestamp,
            'timestamp_iso': timestamp.isoformat(),
            'step': self.current_step,
            'episode_number': self.episode_number,
            'reset_point_idx': self.current_reset_idx,
            'reset_points_total': len(self.reset_points),
            
            # Rewards and actions
            'reward_step': reward,
            'episode_cumulative_reward': self.episode_total_reward,
            'action_decoded': self._last_decoded_action,
            'invalid_action_in_step': not self._last_decoded_action.get('is_valid', True) if self._last_decoded_action else False,
            'invalid_actions_total_episode': self.invalid_action_count_episode,
            
            # Market data (crucial for dashboard)
            'current_price': current_price,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'volume': volume,
            
            # Portfolio state (renamed for dashboard compatibility)
            'total_equity': current_portfolio_state_for_info.get('total_equity', 0.0),
            'portfolio_equity': current_portfolio_state_for_info.get('total_equity', 0.0),
            'cash': current_portfolio_state_for_info.get('cash', 0.0),
            'portfolio_cash': current_portfolio_state_for_info.get('cash', 0.0),
            'unrealized_pnl': current_portfolio_state_for_info.get('unrealized_pnl', 0.0),
            'portfolio_unrealized_pnl': current_portfolio_state_for_info.get('unrealized_pnl', 0.0),
            'realized_pnl': current_portfolio_state_for_info.get('realized_pnl_session', 0.0),
            'portfolio_realized_pnl_session_net': current_portfolio_state_for_info.get('realized_pnl_session', 0.0),
            'total_pnl': current_portfolio_state_for_info.get('realized_pnl_session', 0.0) + current_portfolio_state_for_info.get('unrealized_pnl', 0.0),
            
            # Position data (both formats for compatibility)
            'position': position_qty,
            'position_qty': position_qty,
            'position_side': position_side,
            'avg_entry_price': avg_entry_price,
            
            # Trading statistics
            'total_trades': getattr(self, 'win_loss_counts', {}).get('wins', 0) + getattr(self, 'win_loss_counts', {}).get('losses', 0),
            'win_rate': self._calculate_win_rate(),
            'num_trades': getattr(self, 'win_loss_counts', {}).get('wins', 0) + getattr(self, 'win_loss_counts', {}).get('losses', 0),
            
            # Trade execution details
            'fills_step': fill_details_list if fill_details_list else [],
        }

        # Legacy position details (maintain backward compatibility)
        if self.primary_asset:
            info[f'position_{self.primary_asset}_qty'] = position_qty
            info[f'position_{self.primary_asset}_side'] = position_side
            info[f'position_{self.primary_asset}_avg_entry'] = avg_entry_price

        # Termination info
        if is_terminated and termination_reason_enum:
            info['termination_reason'] = termination_reason_enum.value
        if is_truncated:
            info['TimeLimit.truncated'] = True

        return info
    
    def _calculate_win_rate(self) -> float:
        """Calculate current win rate based on completed trades."""
        if not hasattr(self, 'win_loss_counts'):
            return 0.0
        
        wins = self.win_loss_counts.get('wins', 0)
        losses = self.win_loss_counts.get('losses', 0)
        total_trades = wins + losses
        
        if total_trades == 0:
            return 0.0
        
        return (wins / total_trades) * 100.0

    def _update_dashboard_quality_metrics(self, reset_point: Dict):
        """Update dashboard with quality metrics from the current reset point."""
        try:
            from dashboard.shared_state import dashboard_state
            
            # Extract quality metrics from reset point
            quality_metrics = {
                'day_activity_score': reset_point.get('day_activity_score', 0.0),
                'volume_ratio': reset_point.get('volume_ratio', 1.0),
                'reset_point_quality': reset_point.get('combined_score', 0.0),
                # 2-component scores from reset point
                'current_roc_score': reset_point.get('roc_score', 0.0),
                'current_activity_score': reset_point.get('activity_score', 0.0),
            }
            
            # Get day-level metrics from data manager momentum days
            if self.data_manager and self.primary_asset and self.current_session_date:
                momentum_days = self.data_manager.get_momentum_days(self.primary_asset, min_activity=0.0)
                if not momentum_days.empty:
                    # Find the current day's data
                    current_day_data = momentum_days[
                        momentum_days['date'].dt.date == self.current_session_date.date()
                    ]
                    if not current_day_data.empty:
                        day_data = current_day_data.iloc[0]
                        quality_metrics.update({
                            'halt_count': day_data.get('halt_count', 0),
                            'max_intraday_move': day_data.get('max_intraday_move', 0.0)
                        })
            
            # Calculate average spread from current market data if available
            if self.market_simulator:
                market_state = self.market_simulator.get_current_market_data()
                if market_state:
                    bid = market_state.get('best_bid_price', 0)
                    ask = market_state.get('best_ask_price', 0)
                    if bid > 0 and ask > 0:
                        quality_metrics['avg_spread'] = ask - bid
            
            # Calculate curriculum information if data manager has curriculum support
            curriculum_metrics = {}
            if hasattr(self.data_manager, 'index_utils') and self.data_manager.index_utils:
                # Get curriculum stage from the index utils
                index_utils = self.data_manager.index_utils
                total_episodes = getattr(index_utils, 'total_episodes', self.total_episodes)
                
                # Calculate curriculum stage based on episode count
                if total_episodes < 10000:
                    stage = 'early'
                    min_quality = 0.8
                    next_threshold = 10000
                elif total_episodes < 50000:
                    stage = 'intermediate'
                    min_quality = 0.6
                    next_threshold = 50000
                else:
                    stage = 'advanced'
                    min_quality = 0.0
                    next_threshold = None
                
                curriculum_metrics.update({
                    'curriculum_stage': stage,
                    'curriculum_min_quality': min_quality,
                    'total_episodes_for_curriculum': total_episodes,
                    'curriculum_progress': (total_episodes / next_threshold * 100) if next_threshold else 100.0
                })
                
                # Update the quality metrics dict to include curriculum info
                quality_metrics.update(curriculum_metrics)
            
            # Update dashboard state
            dashboard_state.update_quality_metrics(quality_metrics)
            
        except Exception as e:
            self.logger.warning(f"Failed to update dashboard quality metrics: {e}")

    def _handle_open_positions_at_reset(self) -> Optional[float]:
        """Handle any open positions before episode reset.
        
        Returns:
            The P&L from closing positions, or None if no positions were open
        """
        if not self.portfolio_manager:
            return None
            
        # Get current portfolio state
        market_data = self.market_simulator.get_current_market_data()
        if market_data is None:
            return None
        current_time = market_data['timestamp']
        portfolio_state = self.portfolio_manager.get_portfolio_state(current_time)
        
        # Check if we have any open positions
        has_positions = False
        total_close_pnl = 0.0
        
        for asset_id, pos_data in portfolio_state['positions'].items():
            if pos_data['quantity'] != 0:
                has_positions = True
                
                # Get current market prices
                market_state = self.market_simulator.get_current_market_data()
                current_price = market_state.get('current_price', 0.0) if market_state else 0.0
                if current_price <= 0:
                    ask = market_state.get('best_ask', 0) if market_state else 0
                    bid = market_state.get('best_bid', 0) if market_state else 0
                    if ask > 0 and bid > 0:
                        current_price = (ask + bid) / 2
                
                # Calculate the P&L if we were to close this position
                # This includes unrealized P&L + any trading costs
                position_value = abs(pos_data['quantity']) * current_price
                unrealized_pnl = pos_data['unrealized_pnl']
                
                # Estimate trading costs for closing (commission + slippage)
                est_commission = max(
                    self.config.simulation.min_commission_per_order,
                    position_value * self.config.simulation.commission_per_share * 0.01  # Rough estimate
                )
                est_slippage = position_value * 0.001  # 0.1% slippage estimate
                
                close_pnl = unrealized_pnl - est_commission - est_slippage
                total_close_pnl += close_pnl
                
                self.logger.info(f"🔄 Open position at reset: {asset_id} {pos_data['quantity']:.2f} @ avg ${pos_data['avg_entry_price']:.4f}"
                               f" | Current ${current_price:.4f} | Unrealized P&L: ${unrealized_pnl:+.2f}"
                               f" | Est. close P&L: ${close_pnl:+.2f}")
        
        if has_positions:
            # Apply a "reset penalty" to the reward system
            # This ensures the model experiences the consequence of holding positions
            if hasattr(self.reward_calculator, 'apply_position_close_penalty'):
                self.reward_calculator.apply_position_close_penalty(total_close_pnl)
            
            # Track this in episode metrics
            self.episode_total_reward += total_close_pnl * 0.1  # Scale down to match reward scaling
            
            return total_close_pnl
        
        return None
    
    def _send_initial_chart_data(self):
        """Send initial chart data to dashboard immediately to reduce display delay."""
        try:
            if not self.market_simulator or not hasattr(self.market_simulator, 'combined_bars_1m'):
                return
                
            # Get available 1m bars, even if limited
            combined_bars_1m = self.market_simulator.combined_bars_1m
            if combined_bars_1m is None or combined_bars_1m.empty:
                return
                
            # Filter to current day data (4 AM to 8 PM ET)
            current_date = self.current_session_date.date()
            market_open_et = pd.Timestamp(f"{current_date} 04:00:00", tz='America/New_York')
            market_close_et = pd.Timestamp(f"{current_date} 20:00:00", tz='America/New_York')
            
            # Convert available bars to list format
            candle_list = []
            for timestamp, row in combined_bars_1m.iterrows():
                ts = pd.Timestamp(timestamp)
                if ts.tz is None:
                    ts_et = ts.tz_localize('UTC').tz_convert('America/New_York')
                else:
                    ts_et = ts.tz_convert('America/New_York')
                
                # Only include bars within the current trading day
                if market_open_et <= ts_et <= market_close_et:
                    candle_list.append({
                        'timestamp': ts_et.replace(tzinfo=None).isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row.get('volume', 0))
                    })
            
            # Send to dashboard if we have any data
            if candle_list:
                from dashboard.shared_state import dashboard_state
                dashboard_state.update_candle_data(candle_list)
                self.logger.info(f"📊 Sent initial chart data: {len(candle_list)} candles")
                
        except Exception as e:
            self.logger.warning(f"Failed to send initial chart data: {e}")
    
    def _on_trade_completed(self, trade: Dict[str, Any]):
        """Callback for completed trades."""
        try:
            # Update win/loss counts
            pnl = trade.get('realized_pnl', 0.0)
            if pnl > 0:
                self.win_loss_counts['wins'] += 1
            elif pnl < 0:
                self.win_loss_counts['losses'] += 1
                
            # Trigger position closed callback for completed trades
            if self.callback_manager:
                try:
                    # Map asset_id to symbol for callback compatibility
                    trade_data = dict(trade)
                    if 'asset_id' in trade_data and 'symbol' not in trade_data:
                        trade_data['symbol'] = trade_data['asset_id']
                    self.callback_manager.trigger('on_position_closed', trade_data)
                except Exception as e:
                    self.logger.warning(f"Error in position closed callback: {e}")
                
            # Emit completed trade to dashboard
            from dashboard.event_stream import event_stream
            event_stream.emit_trade(
                side=trade.get('side', 'UNKNOWN'),
                quantity=int(trade.get('entry_quantity', 0)),
                price=trade.get('avg_entry_price', 0),
                fill_price=trade.get('avg_exit_price', 0),
                pnl=pnl,
                commission=trade.get('total_commission', 0),
                order_id=trade.get('trade_id', ''),
                # Additional trade-specific data
                entry_timestamp=trade.get('entry_timestamp'),
                exit_timestamp=trade.get('exit_timestamp'),
                holding_time_seconds=trade.get('holding_period_seconds', 0),
                is_completed_trade=True  # Flag to distinguish from executions
            )
        except Exception as e:
            self.logger.warning(f"Error in trade callback: {e}")

    def set_training_info(self, episode_num: int = 0, total_episodes: int = 0,
                         total_steps: int = 0, update_count: int = 0):
        """Set training information for metrics tracking."""
        if episode_num > self.episode_number:
            self.episode_number = episode_num
        self.total_episodes = total_episodes
        self.total_steps = total_steps
        self.update_count = update_count

    def render(self, info_dict: Optional[Dict[str, Any]] = None):
        """Basic render method."""
        if self.render_mode in ['human', 'logs'] and info_dict:
            print(f"Step {info_dict.get('step', 'N/A')}: "
                  f"Reward {info_dict.get('reward_step', 0.0):.4f}, "
                  f"Equity ${info_dict.get('portfolio_equity', 0.0):.2f}")

    def get_action_mask(self) -> Optional[np.ndarray]:
        """
        Get current action mask for the agent.
        
        Returns:
            Boolean array of shape (12,) where True = valid action, or None if masking disabled
        """
        if not self.action_mask or not self.portfolio_manager or not self.market_simulator:
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
        if not self.action_mask or not self.portfolio_manager or not self.market_simulator:
            return action_probs
            
        # Get current states
        current_time = self.market_simulator.get_current_time()
        portfolio_state = self.portfolio_manager.get_portfolio_state(current_time)
        market_state = self.market_simulator.get_current_market_data()
        
        if not portfolio_state or not market_state:
            return action_probs
            
        return self.action_mask.mask_action_probabilities(action_probs, portfolio_state, market_state)

    def close(self):
        """Close environment and cleanup resources."""
        if self.market_simulator and hasattr(self.market_simulator, 'close'):
            self.market_simulator.close()
        if self.next_market_simulator and hasattr(self.next_market_simulator, 'close'):
            self.next_market_simulator.close()
        self.logger.info("🔒 TradingEnvironment closed")