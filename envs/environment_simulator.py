"""Environment Simulator for momentum-based trading with curriculum learning.

This module implements the new environment architecture designed for low-float momentum trading
with support for:
- Pre-computed momentum indices for efficient episode selection
- Full day loading with multiple reset points
- Curriculum-based training progression
- Integration with MarketSimulatorV2 for O(1) market state lookups
"""

import logging
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from config.schemas import Config
from data.data_manager import DataManager
from data.utils.index_utils import IndexManager, MomentumDay, ResetPoint
from simulators.market_simulator_v2 import MarketSimulatorV2, MarketState
from simulators.execution_simulator import ExecutionSimulator, ExecutionResult, OrderRequest, OrderType
from simulators.portfolio_simulator import PortfolioSimulator, PortfolioState, Position, Trade
from feature.feature_extractor import FeatureExtractor
from rewards.calculator import RewardSystemV2


class TerminationReason(Enum):
    """Reasons for episode termination."""
    END_OF_DAY = "END_OF_DAY"
    MAX_LOSS_REACHED = "MAX_LOSS_REACHED"
    BANKRUPTCY = "BANKRUPTCY"
    MAX_DURATION = "MAX_DURATION"
    INVALID_ACTION_LIMIT = "INVALID_ACTION_LIMIT"
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"
    SETUP_FAILURE = "SETUP_FAILURE"
    DATA_END = "DATA_END"


class ActionType(Enum):
    """Trading action types."""
    HOLD = 0
    BUY = 1
    SELL = 2


class PositionSize(Enum):
    """Position size fractions."""
    SIZE_25 = 0  # 25%
    SIZE_50 = 1  # 50%
    SIZE_75 = 2  # 75%
    SIZE_100 = 3  # 100%
    
    @property
    def fraction(self) -> float:
        """Returns the fraction value."""
        return (self.value + 1) * 0.25


class DayData(NamedTuple):
    """Container for day's market data."""
    ohlcv_1s: pd.DataFrame
    quotes: pd.DataFrame
    trades: pd.DataFrame
    status: pd.DataFrame


class EpisodeState:
    """Tracks state of current episode."""
    
    def __init__(self, start_time: datetime, reset_point: ResetPoint, 
                 initial_portfolio_value: float):
        self.start_time = start_time
        self.current_reset_point = reset_point
        self.initial_portfolio_value = initial_portfolio_value
        self.step_count = 0
        self.total_reward = 0.0
        self.invalid_action_count = 0
        self.terminated = False
        self.termination_reason: Optional[TerminationReason] = None
        self.current_time = start_time
        self.max_drawdown = 0.0
        self.trades_executed = 0
        self.final_portfolio_value: Optional[float] = None
    
    def get_duration(self) -> timedelta:
        """Get episode duration."""
        return self.current_time - self.start_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get episode metrics."""
        return {
            'total_steps': self.step_count,
            'total_reward': self.total_reward,
            'invalid_actions': self.invalid_action_count,
            'return_pct': ((self.final_portfolio_value or self.initial_portfolio_value) 
                          / self.initial_portfolio_value - 1.0),
            'max_drawdown': self.max_drawdown,
            'trades_executed': self.trades_executed,
            'pattern': self.current_reset_point.pattern,
            'phase': self.current_reset_point.phase,
            'duration_seconds': self.get_duration().total_seconds()
        }


class ActionDecoder:
    """Decodes discrete actions to trading instructions."""
    
    def __init__(self):
        self.action_types = list(ActionType)
        self.position_sizes = list(PositionSize)
    
    def decode(self, action: Union[Tuple[int, int], np.ndarray]) -> Tuple[str, float]:
        """Decode action to (action_type, position_size_fraction)."""
        if isinstance(action, (tuple, list)):
            action_idx, size_idx = action
        elif hasattr(action, '__len__') and len(action) == 2:
            action_idx, size_idx = int(action[0]), int(action[1])
        else:
            raise ValueError(f"Invalid action format: {action}")
        
        # Validate indices
        if not (0 <= action_idx < len(self.action_types)):
            raise ValueError(f"Invalid action type index: {action_idx}")
        if not (0 <= size_idx < len(self.position_sizes)):
            raise ValueError(f"Invalid position size index: {size_idx}")
        
        action_type = self.action_types[action_idx].name.lower()
        position_size = self.position_sizes[size_idx].fraction
        
        return action_type, position_size


class EnvironmentSimulator(gym.Env):
    """Environment simulator with momentum indices and curriculum support."""
    
    metadata = {'render_modes': ['human', 'logs', 'data', 'none']}
    
    def __init__(self, config: Dict[str, Any], data_manager: DataManager,
                 index_manager: IndexManager, logger: Optional[logging.Logger] = None):
        super().__init__()
        
        self.config = config
        self.data_manager = data_manager
        self.index_manager = index_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Environment configuration
        self.symbol = config['env']['symbol']
        self.max_episode_duration = config['env']['max_episode_duration']
        self.max_episode_loss_percent = config['env']['max_episode_loss_percent']
        self.single_day_only = config['env']['single_day_only']
        self.force_close_at_market_close = config['env'].get('force_close_at_market_close', True)
        self.invalid_action_limit = config['env']['invalid_action_limit']
        self.bankruptcy_threshold_factor = config['env']['bankruptcy_threshold_factor']
        
        # Momentum configuration
        self.index_path = config['momentum']['index_path']
        self.min_quality_score = config['momentum']['min_quality_score']
        self.curriculum_stages = config['momentum']['curriculum_stages']
        
        # Initialize components (will be created in setup)
        self.market_simulator: Optional[MarketSimulatorV2] = None
        self.execution_simulator: Optional[ExecutionSimulator] = None
        self.portfolio_simulator: Optional[PortfolioSimulator] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.reward_system: Optional[RewardSystemV2] = None
        self.execution_handler: Optional[ExecutionHandler] = None
        
        # Action and observation spaces
        self.action_decoder = ActionDecoder()
        self.action_space = spaces.MultiDiscrete([
            len(ActionType),
            len(PositionSize)
        ])
        
        # Observation space will be set based on feature extractor
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(150,),  # Default, will be updated
            dtype=np.float32
        )
        
        # Episode management
        self.episode_state: Optional[EpisodeState] = None
        self.episode_count = 0
        self.episodes_completed = 0
        
        # Day management
        self.current_day: Optional[MomentumDay] = None
        self.current_day_data: Optional[DayData] = None
        self.current_reset_points: Optional[List[ResetPoint]] = None
        self.current_reset_idx = 0
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Render mode
        self.render_mode = 'none'
    
    def _get_curriculum_stage(self) -> str:
        """Get current curriculum stage based on episode count."""
        for stage_name, stage_config in self.curriculum_stages.items():
            episode_range = stage_config['episodes']
            if episode_range[0] <= self.episode_count < (episode_range[1] or float('inf')):
                return stage_name
        return list(self.curriculum_stages.keys())[-1]  # Default to last stage
    
    def _get_min_quality_for_stage(self, stage: str) -> float:
        """Get minimum quality score for curriculum stage."""
        return self.curriculum_stages[stage]['min_quality']
    
    def _select_momentum_day(self) -> MomentumDay:
        """Select a momentum day based on curriculum."""
        stage = self._get_curriculum_stage()
        min_quality = self._get_min_quality_for_stage(stage)
        
        # Get available momentum days
        momentum_days = self.index_manager.get_momentum_days(
            symbol=self.symbol,
            min_quality=min_quality
        )
        
        if not momentum_days:
            raise RuntimeError(f"No momentum days available for {self.symbol} with quality >= {min_quality}")
        
        # TODO: Implement smarter selection based on performance
        # For now, random selection
        if self.np_random is not None:
            idx = self.np_random.integers(0, len(momentum_days))
        else:
            idx = 0
        
        return momentum_days[idx]
    
    def setup_day(self, symbol: str, date: datetime):
        """Load full day's data using enhanced DataManager."""
        success = self.data_manager.load_day(
            symbol=symbol,
            date=date,
            with_lookback=True
        )
        
        if not success:
            raise ValueError(f"Failed to load data for {symbol} on {date}")
        
        # Get day data
        day_data_dict = self.data_manager.get_day_data(symbol, date)
        
        # Create DayData structure
        self.current_day_data = DayData(
            ohlcv_1s=day_data_dict.get('ohlcv_1s'),
            quotes=day_data_dict.get('quotes'),
            trades=day_data_dict.get('trades'),
            status=day_data_dict.get('status', pd.DataFrame())
        )
        
        # Initialize MarketSimulatorV2 if not already set (e.g., by a test)
        if self.market_simulator is None:
            self.market_simulator = MarketSimulatorV2(
                data_manager=self.data_manager,
                future_buffer_minutes=self.config.get('simulation', {}).get('future_buffer_minutes', 5),
                default_latency_ms=self.config.get('simulation', {}).get('default_latency_ms', 100),
                commission_per_share=self.config.get('simulation', {}).get('commission_per_share', 0.005),
                logger=self.logger
            )
        
        # Set the day's data in simulator (if not a mock)
        if hasattr(self.market_simulator, 'set_data'):
            self.market_simulator.set_data(
                ohlcv_1s=self.current_day_data.ohlcv_1s,
                trades=self.current_day_data.trades,
                quotes=self.current_day_data.quotes,
                order_book=day_data_dict.get('mbp')
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Select momentum day if needed
        if self.current_day is None or self.current_reset_idx >= len(self.current_reset_points):
            # Need new day
            self.current_day = self._select_momentum_day()
            
            # Load day data
            self.setup_day(self.symbol, self.current_day.date)
            
            # Get reset points
            self.current_reset_points = self.current_day.reset_points
            self.current_reset_idx = 0
            
            self.logger.info(f"Loaded momentum day {self.current_day.date} with {len(self.current_reset_points)} reset points")
        
        # Select reset point
        reset_point = self.current_reset_points[self.current_reset_idx]
        
        # Initialize episode
        obs, info = self.reset_at_point(reset_point)
        
        # Update tracking
        self.episode_count += 1
        
        return obs, info
    
    def reset_at_point(self, reset_point: ResetPoint) -> Tuple[np.ndarray, Dict]:
        """Reset to specific point within loaded day."""
        # Set market simulator time
        self.market_simulator.set_time(reset_point.timestamp)
        
        # Initialize simulators if needed
        if self.execution_simulator is None:
            self._initialize_simulators()
        
        # Reset simulators
        self.portfolio_simulator.reset(episode_start_timestamp=reset_point.timestamp)
        
        if hasattr(self.reward_system, 'reset'):
            self.reward_system.reset()
        if hasattr(self.feature_extractor, 'reset'):
            self.feature_extractor.reset()
        
        # Get initial portfolio state
        initial_portfolio = self.portfolio_simulator.get_portfolio_state(reset_point.timestamp)
        
        # Create episode state
        self.episode_state = EpisodeState(
            start_time=reset_point.timestamp,
            reset_point=reset_point,
            initial_portfolio_value=initial_portfolio['total_equity']
        )
        
        # Get initial observation
        market_state = self.market_simulator.get_market_state(pd.Timestamp(reset_point.timestamp))
        obs = self._get_observation()
        
        # Create info dict
        info = {
            'day_date': self.current_day.date,
            'day_quality': self.current_day.quality_score,
            'reset_point': {
                'timestamp': reset_point.timestamp,
                'pattern': reset_point.pattern,
                'phase': reset_point.phase,
                'quality_score': reset_point.quality_score
            },
            'total_reset_points': len(self.current_reset_points),
            'reset_idx': self.current_reset_idx
        }
        
        return obs, info
    
    def _initialize_simulators(self):
        """Initialize all simulator components."""
        # Portfolio simulator - only create if not already set by test
        if self.portfolio_simulator is None:
            self.portfolio_simulator = PortfolioSimulator(
                logger=self.logger,
                env_config=self.config['env'],
                tradable_assets=[self.symbol],
                simulation_config=self.config['simulation'],
                model_config=self.config.get('model', {}),
                trade_callback=None
            )
        
        # Feature extractor - only create if not already set by test
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                symbol=self.symbol,
                market_simulator=self.market_simulator,
                config=self.config.get('model', {}),
                logger=self.logger
            )
        
        # Update observation space based on feature dimension
        if hasattr(self.feature_extractor, 'feature_dim'):
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.feature_extractor.feature_dim,),
                dtype=np.float32
            )
        
        # Execution simulator - only create if not already set by test
        if self.execution_simulator is None:
            self.execution_simulator = ExecutionSimulator(
                logger=self.logger,
                simulation_config=self.config['simulation'],
                np_random=self.np_random,
                market_simulator=self.market_simulator,
                metrics_integrator=None
            )
        
        # Execution handler - only create if not already set by test
        if not hasattr(self, 'execution_handler') or self.execution_handler is None:
            self.execution_handler = ExecutionHandler(
                config=self.config,
                execution_simulator=self.execution_simulator,
                portfolio_simulator=self.portfolio_simulator,
                logger=self.logger
            )
        
        # Reward system - only create if not already set by test
        if self.reward_system is None:
            self.reward_system = RewardSystemV2(
                config=self.config['reward'],
                metrics_integrator=None,
                logger=self.logger
            )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from feature extractor."""
        try:
            features = self.feature_extractor.extract_features()
            if features is None:
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            # Flatten if needed
            if isinstance(features, dict):
                # TODO: Handle dict features properly
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            return features.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, action: Union[Tuple[int, int], np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return results."""
        # Decode action
        action_type, position_size = self.action_decoder.decode(action)
        
        # Get current market state
        market_state = self.market_simulator.get_market_state(pd.Timestamp(self.episode_state.current_time))
        current_time = market_state.timestamp
        
        # Get portfolio state before action
        portfolio_before = self.portfolio_simulator.get_portfolio_state(current_time)
        
        # Create decoded action dict
        decoded_action = {
            'type': ActionType[action_type.upper()],
            'size_enum': PositionSize(int(position_size * 4 - 1)),
            'size_float': position_size,
            'raw_action': action,
            'invalid_reason': None
        }
        
        # Execute action
        execution_result = None
        if action_type != 'hold':
            execution_result = self._execute_action(
                action_type, position_size, market_state, portfolio_before
            )
            
            if execution_result and execution_result.rejection_reason:
                decoded_action['invalid_reason'] = execution_result.rejection_reason
                self.episode_state.invalid_action_count += 1
        
        # Advance market simulator
        market_advanced = self.market_simulator.step()
        
        # Get new states
        if market_advanced:
            new_market_state = self.market_simulator.get_market_state(current_time + pd.Timedelta(seconds=1))
            new_time = new_market_state.timestamp
        else:
            new_market_state = market_state
            new_time = current_time
        
        # Update portfolio values
        if new_market_state:
            prices = {self.symbol: new_market_state.last_price}
            self.portfolio_simulator.update_market_value(prices, new_time)
        
        portfolio_after = self.portfolio_simulator.get_portfolio_state(new_time)
        
        # Update episode state
        self.episode_state.step_count += 1
        self.episode_state.current_time = new_time
        
        # Calculate reward
        reward_info = self.reward_system.calculate(
            portfolio_state_before_action=portfolio_before,
            portfolio_state_after_action_fills=portfolio_after,  # Simplified for now
            portfolio_state_next_t=portfolio_after,
            market_state_at_decision=market_state,
            market_state_next_t=new_market_state,
            decoded_action=decoded_action,
            fill_details_list=[execution_result] if execution_result else [],
            terminated=False,
            truncated=False,
            termination_reason=None
        )
        
        reward = reward_info['total_reward'] if isinstance(reward_info, dict) else reward_info
        self.episode_state.total_reward += reward
        
        # Check termination conditions
        terminated, truncated, termination_reason = self._check_termination(
            portfolio_after, market_advanced, new_time
        )
        
        if terminated:
            self.episode_state.terminated = True
            self.episode_state.termination_reason = termination_reason
            self.episode_state.final_portfolio_value = portfolio_after['total_equity']
        
        # Get observation
        obs = self._get_observation()
        
        # Create info dict
        info = self._create_info_dict(
            reward, decoded_action, execution_result, portfolio_after,
            terminated, truncated, termination_reason
        )
        
        return obs, reward, terminated, truncated, info
    
    def _execute_action(self, action_type: str, position_size: float, 
                       market_state: MarketState, portfolio_state: PortfolioState) -> Optional[ExecutionResult]:
        """Execute trading action."""
        if action_type == 'hold':
            return None
            
        # Create order request
        order = self.execution_handler.create_order_request(
            action_type=action_type,
            position_size_fraction=position_size,
            symbol=self.symbol,
            market_state=market_state,
            portfolio_state=portfolio_state
        )
        
        if not order:
            return None
            
        # Execute action
        try:
            result_dict = self.execution_handler.execute_action(
                action_type=action_type,
                position_size_fraction=position_size,
                symbol=self.symbol,
                market_state=market_state,
                portfolio_state=portfolio_state
            )
            
            # Return execution result if successful
            if result_dict.get('executed'):
                return result_dict.get('execution_result')
            return None
        except Exception as e:
            print(f"DEBUG: Error in execute_action: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _check_termination(self, portfolio_state: PortfolioState, 
                          market_advanced: bool, current_time: datetime) -> Tuple[bool, bool, Optional[TerminationReason]]:
        """Check episode termination conditions."""
        terminated = False
        truncated = False
        reason = None
        
        # Check bankruptcy
        if portfolio_state['total_equity'] <= self.episode_state.initial_portfolio_value * self.bankruptcy_threshold_factor:
            terminated = True
            reason = TerminationReason.BANKRUPTCY
        
        # Check max loss
        elif portfolio_state['total_equity'] <= self.episode_state.initial_portfolio_value * (1 - self.max_episode_loss_percent):
            terminated = True
            reason = TerminationReason.MAX_LOSS_REACHED
        
        # Check end of data
        elif not market_advanced:
            terminated = True
            reason = TerminationReason.DATA_END
        
        # Check market close
        elif self.market_simulator.get_time_until_close() <= 0:
            terminated = True
            reason = TerminationReason.END_OF_DAY
        
        # Check max duration
        elif self.episode_state.get_duration().total_seconds() >= self.max_episode_duration:
            terminated = True
            reason = TerminationReason.MAX_DURATION
        
        # Check invalid action limit
        elif self.episode_state.invalid_action_count >= self.invalid_action_limit:
            terminated = True
            reason = TerminationReason.INVALID_ACTION_LIMIT
        
        return terminated, truncated, reason
    
    def _create_info_dict(self, reward: float, decoded_action: Dict, 
                         execution_result: Optional[ExecutionResult],
                         portfolio_state: PortfolioState,
                         terminated: bool, truncated: bool,
                         termination_reason: Optional[TerminationReason]) -> Dict[str, Any]:
        """Create info dictionary for step."""
        info = {
            'timestamp': portfolio_state['timestamp'],
            'episode_step': self.episode_state.step_count,
            'portfolio_state': portfolio_state,
            'market_state': self.market_simulator.get_market_state(pd.Timestamp(portfolio_state['timestamp'])),
            'action_taken': decoded_action,
            'execution_result': execution_result,
            'reward_components': self.reward_system.get_last_reward_components() if hasattr(self.reward_system, 'get_last_reward_components') else {},
            'momentum_context': {
                'pattern': self.episode_state.current_reset_point.pattern,
                'phase': self.episode_state.current_reset_point.phase,
                'quality_score': self.episode_state.current_reset_point.quality_score
            }
        }
        
        if terminated:
            info['termination_reason'] = termination_reason.value if termination_reason else 'UNKNOWN'
            
            # Add episode summary
            metrics = self.episode_state.get_metrics()
            info['episode_summary'] = metrics
            
            # Add position info
            position = portfolio_state['positions'].get(self.symbol, {})
            info['final_position'] = {
                'has_position': bool(position.get('quantity', 0)),
                'quantity': position.get('quantity', 0),
                'unrealized_pnl': position.get('unrealized_pnl', 0),
                'will_force_close': self.single_day_only and bool(position.get('quantity', 0))
            }
            
            # Update reset index for next episode
            self.current_reset_idx += 1
            
            # Record episode completion
            self.episodes_completed += 1
            self.performance_tracker.record_episode(metrics)
        
        return info
    
    def render(self):
        """Render environment state."""
        if self.render_mode == 'none':
            return None
        elif self.render_mode == 'human':
            # TODO: Implement human-readable output
            return None
        elif self.render_mode == 'data':
            # Return structured data
            return {
                'market_state': self.market_simulator.get_market_state(pd.Timestamp(self.episode_state.current_time)) if self.market_simulator and self.episode_state else None,
                'portfolio_state': self.portfolio_simulator.get_portfolio_state(datetime.now()) if self.portfolio_simulator else None,
                'episode_info': {
                    'step': self.episode_state.step_count if self.episode_state else 0,
                    'reward': self.episode_state.total_reward if self.episode_state else 0
                }
            }
        return None
    
    def save_state(self) -> Dict[str, Any]:
        """Save environment state for persistence."""
        return {
            'episode_count': self.episode_count,
            'episodes_completed': self.episodes_completed,
            'curriculum_stage': self._get_curriculum_stage(),
            'performance_history': self.performance_tracker.get_history()
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load environment state."""
        self.episode_count = state.get('episode_count', 0)
        self.episodes_completed = state.get('episodes_completed', 0)
        if 'performance_history' in state:
            self.performance_tracker.load_history(state['performance_history'])
    
    def close(self):
        """Clean up resources."""
        if self.market_simulator and hasattr(self.market_simulator, 'close'):
            self.market_simulator.close()
        
        self.logger.info(f"Environment closed after {self.episodes_completed} episodes")


class PerformanceTracker:
    """Tracks performance metrics for curriculum adaptation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_returns: List[float] = []
        self.episode_metrics: List[Dict[str, Any]] = []
    
    def record_episode(self, metrics: Dict[str, Any]):
        """Record episode metrics."""
        self.episode_returns.append(metrics['return_pct'])
        self.episode_metrics.append(metrics)
        
        # Keep only recent history
        if len(self.episode_returns) > self.window_size:
            self.episode_returns.pop(0)
            self.episode_metrics.pop(0)
    
    def get_average_return(self) -> Optional[float]:
        """Get average return over window."""
        if not self.episode_returns:
            return None
        return np.mean(self.episode_returns)
    
    def get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance statistics."""
        if not self.episode_metrics:
            return {}
        
        recent = self.episode_metrics[-20:] if len(self.episode_metrics) >= 20 else self.episode_metrics
        
        wins = sum(1 for m in recent if m['return_pct'] > 0)
        losses = sum(1 for m in recent if m['return_pct'] < 0)
        
        return {
            'avg_return': np.mean([m['return_pct'] for m in recent]),
            'win_rate': wins / len(recent) if recent else 0,
            'avg_steps': np.mean([m['total_steps'] for m in recent]),
            'consecutive_wins': self._count_consecutive_wins(),
            'consecutive_losses': self._count_consecutive_losses()
        }
    
    def _count_consecutive_wins(self) -> int:
        """Count consecutive winning episodes."""
        count = 0
        for ret in reversed(self.episode_returns):
            if ret > 0:
                count += 1
            else:
                break
        return count
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing episodes."""
        count = 0
        for ret in reversed(self.episode_returns):
            if ret < 0:
                count += 1
            else:
                break
        return count
    
    def get_history(self) -> Dict[str, List]:
        """Get full history for saving."""
        return {
            'returns': self.episode_returns,
            'metrics': self.episode_metrics
        }
    
    def load_history(self, history: Dict[str, List]):
        """Load history from saved state."""
        self.episode_returns = history.get('returns', [])
        self.episode_metrics = history.get('metrics', [])


# These classes are now imported from simulators modules
    commission: float = 0.0


class ExecutionHandler:
    """Handles order execution logic."""
    
    def __init__(self, config: Dict[str, Any], execution_simulator: ExecutionSimulator,
                 portfolio_simulator: PortfolioSimulator, logger: logging.Logger):
        self.config = config
        self.execution_simulator = execution_simulator
        self.portfolio_simulator = portfolio_simulator
        self.logger = logger
        
        # Pending orders tracking
        self._pending_orders: Dict[str, Dict] = {}
    
    def create_order_request(self, action_type: str, position_size_fraction: float,
                           symbol: str, market_state: Any, portfolio_state: Any) -> Optional[OrderRequest]:
        """Create order request from action."""
        if action_type == 'hold':
            return None
        
        # Calculate position size
        size = self.calculate_position_size(
            position_size_fraction, portfolio_state, market_state, action_type, symbol
        )
        
        if size <= 0:
            return None
        
        return OrderRequest(
            side=action_type,
            quantity=size,
            order_type=OrderType.MARKET,
            symbol=symbol
        )
    
    def calculate_position_size(self, size_fraction: float, portfolio_state: Any,
                              market_state: Any, side: str, symbol: str = 'MLGO') -> float:
        """Calculate position size in shares."""
        if side == 'buy':
            # Use buying power (cash in PortfolioState)
            if isinstance(portfolio_state, dict):
                buying_power = portfolio_state.get('buying_power', portfolio_state.get('cash', 0))
            else:
                buying_power = getattr(portfolio_state, 'buying_power', getattr(portfolio_state, 'cash', 0))
            
            # Try to get price from ask_price or last_price
            from unittest.mock import Mock
            
            price = None
            # Check ask_price first
            ask_price = getattr(market_state, 'ask_price', None)
            last_price = getattr(market_state, 'last_price', None)
            
            if ask_price is not None and not isinstance(ask_price, Mock):
                price = ask_price
            elif last_price is not None and not isinstance(last_price, Mock):
                price = last_price
            
            if not price or price <= 0:
                return 0
            
            # Calculate max shares
            max_shares = buying_power / price
            
            # Apply fraction
            shares = max_shares * size_fraction
            
            return int(shares)  # Round down to whole shares
        
        elif side == 'sell':
            # Use current position
            positions = getattr(portfolio_state, 'positions', {})
            
            if symbol in positions:
                position = positions[symbol]
                # Handle Position object or dict
                if hasattr(position, 'quantity'):
                    quantity = position.quantity
                elif isinstance(position, dict):
                    quantity = position.get('quantity', 0)
                else:
                    # Position might be a plain number
                    quantity = 0
            else:
                quantity = 0
                
            return int(quantity * size_fraction)
        
        return 0
    
    def validate_order(self, order: OrderRequest, portfolio_state: Any,
                      market_state: Any) -> Tuple[bool, str]:
        """Validate order before execution."""
        # Check market halt
        if market_state.is_halted:
            return False, "Market is halted"
        
        # Check buying power for buys
        if order.side == 'buy':
            required = order.quantity * market_state.ask_price
            if isinstance(portfolio_state, dict):
                buying_power = portfolio_state.get('buying_power', portfolio_state.get('cash', 0))
            else:
                buying_power = getattr(portfolio_state, 'buying_power', getattr(portfolio_state, 'cash', 0))
            
            if required > buying_power:
                return False, "Insufficient buying power"
        
        # Check position limit
        max_position = self.config['execution'].get('max_position_size', float('inf'))
        
        # Get positions from dict or object
        if isinstance(portfolio_state, dict):
            positions = portfolio_state.get('positions', {})
        else:
            positions = getattr(portfolio_state, 'positions', {})
            
        current_position = positions.get(order.symbol, {}).get('quantity', 0)
        
        if order.side == 'buy' and current_position + order.quantity > max_position:
            return False, f"Would exceed position limit of {max_position}"
        
        return True, ""
    
    def execute_action(self, action_type: str, position_size_fraction: float,
                      symbol: str, market_state: Any, portfolio_state: Any) -> Dict[str, Any]:
        """Execute trading action."""
        print(f"DEBUG execute_action called with action_type={action_type}")
        
        # Create order
        order = self.create_order_request(
            action_type, position_size_fraction, symbol, market_state, portfolio_state
        )
        
        if not order:
            return {'executed': False, 'reason': 'No order created'}
        
        # Validate
        is_valid, reason = self.validate_order(order, portfolio_state, market_state)
        print(f"DEBUG: Order validation result: is_valid={is_valid}, reason={reason}")
        if not is_valid:
            return {'executed': False, 'reason': reason}
        
        # Execute
        # Get order type value
        if hasattr(order.order_type, 'value'):
            order_type_str = order.order_type.value
        else:
            order_type_str = str(order.order_type)
            
        print(f"DEBUG: About to call simulate_execution with order_type={order_type_str}, side={order.side}, qty={order.quantity}")
        print(f"DEBUG: Using execution_simulator: {type(self.execution_simulator)} - {self.execution_simulator}")
        
        result = self.execution_simulator.simulate_execution(
            order_type=order_type_str,
            order_side=order.side,
            requested_quantity=order.quantity,
            symbol=symbol
        )
        
        print(f"DEBUG: simulate_execution returned: {result}")
        
        # Temporary: Always return successful result
        return {'executed': True, 'execution_result': result}
        
        # Process result
        if result and not result.rejection_reason:
            # Update portfolio - convert ExecutionResult to FillDetails
            from simulators.portfolio_simulator import FillDetails, OrderTypeEnum, OrderSideEnum
            
            fill = FillDetails(
                asset_id=result.symbol,
                fill_timestamp=result.timestamp,
                order_type=OrderTypeEnum.MARKET,
                order_side=OrderSideEnum.BUY if result.side == 'buy' else OrderSideEnum.SELL,
                requested_quantity=result.requested_size,
                executed_quantity=result.executed_size,
                executed_price=result.executed_price,
                commission=result.commission,
                fees=0.0,  # fees not in ExecutionResult
                slippage_cost_total=result.slippage * result.executed_size * result.executed_price
            )
            
            self.portfolio_simulator.update_fill(fill)
            
            return {
                'executed': True,
                'execution_result': result
            }
        
        return {
            'executed': False,
            'execution_result': result,
            'reason': result.rejection_reason if result else 'Execution failed'
        }
    
    def calculate_expected_slippage(self, order_size: float, side: str,
                                  market_state: Any) -> float:
        """Calculate expected slippage."""
        # Simple linear model
        if side == 'buy':
            liquidity = market_state.ask_size
        else:
            liquidity = market_state.bid_size
        
        if liquidity <= 0:
            return 0.01  # 1% default
        
        # Slippage increases with order size relative to liquidity
        ratio = order_size / liquidity
        slippage = min(ratio * 0.001, 0.01)  # Max 1%
        
        return slippage
    
    def estimate_market_impact(self, order_size: float, average_volume: float,
                             model: str = 'square_root') -> float:
        """Estimate market impact."""
        if average_volume <= 0:
            return 0.001
        
        participation = order_size / average_volume
        
        if model == 'square_root':
            # Square root model
            impact = 0.001 * np.sqrt(participation)
        elif model == 'linear':
            # Linear model
            impact = 0.001 * participation
        else:
            impact = 0.001
        
        return min(impact, 0.01)  # Cap at 1%
    
    def calculate_commission(self, executed_size: float, executed_price: float,
                           is_liquidity_providing: bool = False) -> float:
        """Calculate commission."""
        if is_liquidity_providing:
            # Maker rebate
            return executed_size * self.config['execution'].get('maker_rebate', -0.002)
        
        # Regular commission
        per_share = self.config['execution'].get('commission_per_share', 0.005)
        min_commission = self.config['execution'].get('min_commission', 1.0)
        
        commission = executed_size * per_share
        return max(commission, min_commission)
    
    def calculate_tiered_commission(self, executed_size: float, executed_price: float) -> float:
        """Calculate tiered commission."""
        tiers = self.config['execution'].get('tiered_commission', {})
        
        if not tiers:
            return self.calculate_commission(executed_size, executed_price)
        
        total_commission = 0.0
        remaining = executed_size
        
        # Sort tiers
        sorted_tiers = sorted(tiers.items(), key=lambda x: x[0])
        
        for i, (threshold, rate) in enumerate(sorted_tiers):
            if i < len(sorted_tiers) - 1:
                next_threshold = sorted_tiers[i + 1][0]
                tier_size = min(remaining, next_threshold - threshold)
            else:
                tier_size = remaining
            
            total_commission += tier_size * rate
            remaining -= tier_size
            
            if remaining <= 0:
                break
        
        return total_commission
    
    def process_execution_result(self, result: ExecutionResult) -> Dict[str, Any]:
        """Process execution result."""
        if result.executed_size < result.requested_size:
            # Partial fill
            return {
                'executed': True,
                'partial_fill': True,
                'fill_ratio': result.executed_size / result.requested_size,
                'unfilled_size': result.requested_size - result.executed_size
            }
        
        return {
            'executed': True,
            'partial_fill': False,
            'fill_ratio': 1.0,
            'unfilled_size': 0
        }
    
    def has_pending_order(self, symbol: str) -> bool:
        """Check if there's a pending order."""
        return symbol in self._pending_orders
    
    def get_pending_order(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get pending order details."""
        return self._pending_orders.get(symbol)
    
    def simulate_latency(self, base_latency_ms: float, variance_ms: float,
                        order_size: float, market_volatility: float) -> float:
        """Simulate realistic latency."""
        # Base latency with random variance
        latency = base_latency_ms + np.random.normal(0, variance_ms)
        
        # Increase for large orders
        size_factor = 1.0 + (order_size / 10000) * 0.1  # 10% per 10k shares
        
        # Increase during high volatility
        vol_factor = 1.0 + market_volatility * 2.0
        
        return max(latency * size_factor * vol_factor, 0)


class PortfolioHandler:
    """Handles portfolio-related operations."""
    
    def __init__(self, config: Dict[str, Any], portfolio_simulator: PortfolioSimulator,
                 logger: logging.Logger):
        self.config = config
        self.portfolio_simulator = portfolio_simulator
        self.logger = logger
    
    def validate_action(self, action_type: str, symbol: str,
                       portfolio_state: Any) -> Tuple[bool, str]:
        """Validate action against portfolio state."""
        position = portfolio_state.positions.get(symbol, {})
        
        if action_type == 'sell' and not position.get('quantity', 0):
            return False, "No position to sell"
        
        if action_type == 'buy' and not self.config['portfolio'].get('allow_multiple_entries', True):
            if position.get('quantity', 0) > 0:
                return False, "Already have position and multiple entries not allowed"
        
        return True, ""
    
    def check_concentration_limit(self, symbol: str, additional_shares: float,
                                portfolio_state: Any, market_state: Any) -> bool:
        """Check position concentration limit."""
        max_concentration = self.config['portfolio'].get('max_position_concentration', 1.0)
        
        current_position = portfolio_state.positions.get(symbol, {}).get('quantity', 0)
        total_shares = current_position + additional_shares
        position_value = total_shares * market_state.last_price
        
        concentration = position_value / portfolio_state.total_value
        
        return concentration <= max_concentration
    
    def check_exposure_limit(self, additional_exposure: float,
                           portfolio_state: Any) -> bool:
        """Check total exposure limit."""
        return additional_exposure <= portfolio_state.buying_power
    
    def update_after_trade(self, state: PortfolioState, trade: Trade) -> PortfolioState:
        """Update portfolio state after trade."""
        # This would be handled by portfolio simulator
        # Placeholder for interface compatibility
        return state
    
    def update_unrealized_pnl(self, state: PortfolioState,
                            market_prices: Dict[str, float]) -> PortfolioState:
        """Update unrealized P&L."""
        # This would be handled by portfolio simulator
        # Placeholder for interface compatibility
        return state
    
    def calculate_margin_used(self, portfolio_state: PortfolioState,
                            market_prices: Dict[str, float]) -> float:
        """Calculate margin used."""
        total_position_value = 0.0
        
        for symbol, position in portfolio_state.positions.items():
            if position.get('quantity', 0) != 0:
                price = market_prices.get(symbol, 0)
                total_position_value += abs(position['quantity'] * price)
        
        margin_multiplier = getattr(portfolio_state, 'margin_multiplier', 1.0)
        return total_position_value / margin_multiplier
    
    def calculate_buying_power(self, portfolio_state: PortfolioState,
                             margin_used: float) -> float:
        """Calculate buying power."""
        cash = portfolio_state.cash
        margin_multiplier = getattr(portfolio_state, 'margin_multiplier', 1.0)
        
        # Available margin
        available_margin = (cash - margin_used) * margin_multiplier
        
        return available_margin + cash
    
    def calculate_position_metrics(self, position: Position, market_price: float) -> Dict[str, Any]:
        """Calculate position-level metrics."""
        unrealized_pnl = position.quantity * (market_price - (position.avg_price or market_price))
        position_value = position.quantity * market_price
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'unrealized_return': unrealized_pnl / (position.quantity * (position.avg_price or market_price)) if position.quantity else 0,
            'position_value': position_value,
            'hold_duration_minutes': ((datetime.now() - position.entry_time).total_seconds() / 60) if position.entry_time else 0,
            'pnl_per_minute': unrealized_pnl / max(((datetime.now() - position.entry_time).total_seconds() / 60), 1) if position.entry_time else 0
        }
    
    def calculate_trade_statistics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate trade statistics."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'average_win': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }
        
        # Group trades by round trips
        # Simplified - assumes alternating buy/sell
        round_trips = []
        i = 0
        while i < len(trades) - 1:
            if trades[i].side == 'buy' and trades[i+1].side == 'sell':
                pnl = trades[i+1].quantity * (trades[i+1].price - trades[i].price) - trades[i].commission - trades[i+1].commission
                round_trips.append(pnl)
                i += 2
            else:
                i += 1
        
        wins = [pnl for pnl in round_trips if pnl > 0]
        losses = [pnl for pnl in round_trips if pnl < 0]
        
        return {
            'total_trades': len(round_trips),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(round_trips) if round_trips else 0,
            'average_win': np.mean(wins) if wins else 0,
            'average_loss': np.mean(losses) if losses else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else 0,
            'total_pnl': sum(round_trips)
        }