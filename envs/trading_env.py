# env/trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import logging

from simulation.simulator import Simulator
from data.data_manager import DataManager


class TradingEnv(gym.Env):
    """
    OpenAI Gym-compatible environment for financial trading.
    Designed for reinforcement learning models to learn trading strategies.
    
    Features:
    - Compatible with standard RL libraries (Stable Baselines, RLlib, etc.)
    - Configurable reward function for momentum trading
    - Normalized state representation for RL models
    - Handles continuous actions for position sizing
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 1}
    
    def __init__(self, 
                 simulator: Simulator,
                 config: Dict = None,
                 reward_function: Optional[callable] = None,
                 logger: logging.Logger = None):
        """
        Initialize the trading environment.
        
        Args:
            simulator: Configured Simulator instance
            config: Environment configuration
            reward_function: Custom reward function (optional)
            logger: Optional logger
        """
        self.simulator = simulator
        self.config = config or {}
        self.custom_reward_fn = reward_function
        self.logger = logger or logging.getLogger(__name__)
        
        # Environment configuration
        self.random_reset = self.config.get('random_reset', False)
        self.state_dim = self.config.get('state_dim', 300)  # Dimension of state vector
        self.window_size = self.config.get('window_size', 30)  # Number of timesteps in state
        self.max_steps = self.config.get('max_steps', 1000)  # Maximum steps per episode
        self.normalize_state = self.config.get('normalize_state', True)  # Whether to normalize state
        self.normalize_reward = self.config.get('normalize_reward', True)  # Whether to normalize reward
        self.reward_scaling = self.config.get('reward_scaling', 1.0)  # Reward scaling factor
        self.early_stop_pct = self.config.get('early_stop_pct', -0.05)  # Stop if portfolio drops by this percentage
        
        # Penalty hyperparameters for reward
        self.trading_penalty = self.config.get('trading_penalty', 1.0)  # Fixed penalty per trade
        self.holding_penalty = self.config.get('holding_penalty', 0.001)  # Per-step penalty for holding
        self.inactivity_penalty = self.config.get('inactivity_penalty', 0.0001)  # Penalty for not trading
        self.exposure_penalty = self.config.get('exposure_penalty', 0.001)  # Penalty for large positions
        self.drawdown_penalty = self.config.get('drawdown_penalty', 5.0)  # Penalty for exceeding drawdown
        self.winning_reward_bonus = self.config.get('winning_reward_bonus', 2.0)  # Bonus for profitable trades
        
        # Environment state
        self.current_step = 0
        self.done = False
        self.info = {}
        self.current_state = None
        self.current_position = 0.0
        self.last_trade_time = None
        self.last_trade_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.initial_portfolio_value = 0.0
        self.last_portfolio_value = 0.0
        self.max_portfolio_value = 0.0
        self.current_drawdown_pct = 0.0
        self.steps_since_trade = 0
        
        # Define action and observation space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Returns:
            Tuple of (initial_state, info)
        """
        super().reset(seed=seed)
        
        # Reset simulator state
        state_dict = self.simulator.reset(random_day=self.random_reset)
        
        # Reset environment state
        self.current_step = 0
        self.done = False
        self.info = {}
        self.current_position = 0.0
        self.last_trade_time = None
        self.last_trade_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.steps_since_trade = 0
        
        # Get portfolio value
        portfolio_state = self.simulator.get_portfolio_state()
        self.initial_portfolio_value = portfolio_state.get('total_value', 100000.0)
        self.last_portfolio_value = self.initial_portfolio_value
        self.max_portfolio_value = self.initial_portfolio_value
        
        # Get normalized state
        norm_state = self._get_normalized_state()
        self.current_state = norm_state
        
        return norm_state, self.info
    
    def step(self, action):
        """
        Take a step in the environment by executing an action.
        
        Args:
            action: Action to take (normalized -1.0 to 1.0)
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Ensure action is in correct format
        action_value = float(action[0]) if hasattr(action, "__len__") else float(action)
        
        # Execute action in simulator
        next_state, raw_reward, done, info = self.simulator.step(action_value)
        
        # Update environment state
        self.current_step += 1
        self.steps_since_trade += 1
        
        # Get portfolio state
        portfolio_state = self.simulator.get_portfolio_state()
        current_portfolio_value = portfolio_state.get('total_value', self.last_portfolio_value)
        
        # Calculate portfolio change
        portfolio_change = current_portfolio_value - self.last_portfolio_value
        portfolio_change_pct = portfolio_change / self.last_portfolio_value if self.last_portfolio_value > 0 else 0.0
        
        # Update max portfolio value
        if current_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = current_portfolio_value
        
        # Calculate drawdown
        drawdown = self.max_portfolio_value - current_portfolio_value
        drawdown_pct = drawdown / self.max_portfolio_value if self.max_portfolio_value > 0 else 0.0
        self.current_drawdown_pct = drawdown_pct
        
        # Check for early stopping based on drawdown
        if drawdown_pct > abs(self.early_stop_pct) and self.early_stop_pct != 0:
            done = True
            info['early_stop'] = f"Max drawdown exceeded: {drawdown_pct:.2%}"
        
        # Check for maximum steps
        if self.current_step >= self.max_steps:
            info['truncated'] = True
            done = True
        
        # Get position information
        position = self.simulator.portfolio_simulator.get_position(self.simulator.current_symbol)
        self.current_position = position['quantity'] if position else 0.0
        
        # Track trade information
        trade_executed = False
        if info.get('action_result', {}).get('success') and info.get('action_result', {}).get('action') != 'hold':
            self.last_trade_time = self.simulator.current_timestamp
            self.last_trade_price = info.get('action_result', {}).get('fill_price', 0.0)
            trade_executed = True
            self.steps_since_trade = 0
            
            # Track realized P&L
            self.realized_pnl += info.get('action_result', {}).get('realized_pnl', 0.0)
        
        # Calculate unrealized P&L
        if position:
            self.unrealized_pnl = position.get('unrealized_pnl', 0.0)
        else:
            self.unrealized_pnl = 0.0
        
        # Update state
        self.last_portfolio_value = current_portfolio_value
        
        # Calculate reward using custom function if provided
        if self.custom_reward_fn:
            reward = self.custom_reward_fn(
                self, action_value, portfolio_change, portfolio_change_pct, 
                trade_executed, info
            )
        else:
            reward = self._calculate_reward(
                action_value, portfolio_change, portfolio_change_pct, 
                trade_executed, info
            )
        
        # Get normalized state
        norm_state = self._get_normalized_state()
        self.current_state = norm_state
        
        # Update info dictionary
        info.update({
            'step': self.current_step,
            'position': self.current_position,
            'portfolio_value': current_portfolio_value,
            'portfolio_change': portfolio_change,
            'portfolio_change_pct': portfolio_change_pct,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'max_drawdown_pct': drawdown_pct,
            'reward': reward,
            'timestamp': self.simulator.current_timestamp
        })
        
        # Set info for episode end
        if done:
            total_pnl = current_portfolio_value - self.initial_portfolio_value
            total_pnl_pct = total_pnl / self.initial_portfolio_value
            
            info['episode'] = {
                'steps': self.current_step,
                'reward': self.simulator.total_reward,
                'portfolio_value': current_portfolio_value,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'max_drawdown_pct': self.simulator.portfolio_simulator.get_statistics().get('max_drawdown_pct', 0.0),
                'trade_count': len(self.simulator.portfolio_simulator.get_trade_history()),
                'win_rate': self.simulator.portfolio_simulator.get_statistics().get('win_rate', 0.0)
            }
        
        return norm_state, reward, done, False, info
    
    def _get_normalized_state(self) -> np.ndarray:
        """
        Get normalized state representation for RL model.
        
        Returns:
            NumPy array with normalized state
        """
        # Get raw state array from the simulator
        raw_state = self.simulator.get_current_state_array()
        
        # If state is empty, return zeros
        if raw_state is None or len(raw_state) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Normalize state if required
        if self.normalize_state:
            # Clip extreme values
            raw_state = np.clip(raw_state, -1e6, 1e6)
            
            # Handle NaNs
            raw_state = np.nan_to_num(raw_state, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure state has correct dimensions
        if len(raw_state) > self.state_dim:
            # Truncate if too long
            norm_state = raw_state[:self.state_dim]
        elif len(raw_state) < self.state_dim:
            # Pad with zeros if too short
            norm_state = np.pad(raw_state, (0, self.state_dim - len(raw_state)))
        else:
            norm_state = raw_state
        
        return norm_state.astype(np.float32)
    
    def _calculate_reward(self, action: float, portfolio_change: float, 
                         portfolio_change_pct: float, trade_executed: bool,
                         info: Dict[str, Any]) -> float:
        """
        Calculate reward based on portfolio change and trading activity.
        
        Args:
            action: Executed action value
            portfolio_change: Absolute change in portfolio value
            portfolio_change_pct: Percentage change in portfolio value
            trade_executed: Whether a trade was executed
            info: Information dictionary
            
        Returns:
            Calculated reward
        """
        # Base reward is the portfolio change percentage
        # Using percentage change makes reward scale-invariant
        reward = portfolio_change_pct * 100.0  # Convert to basis points for better scaling
        
        # Add penalty or bonus based on action
        action_result = info.get('action_result', {})
        
        # If a trade was executed
        if trade_executed:
            # Apply trading penalty (transaction costs, etc.)
            reward -= self.trading_penalty
            
            # Add bonus for profitable trades
            realized_pnl = action_result.get('realized_pnl', 0.0)
            if realized_pnl > 0:
                reward += self.winning_reward_bonus
        else:
            # Small penalty for holding positions (encourages efficient capital use)
            if abs(self.current_position) > 0:
                reward -= self.holding_penalty * abs(self.current_position)
            
            # Very small penalty for inactivity (encourages action when appropriate)
            if self.steps_since_trade > 10:  # Only after some inactivity
                reward -= self.inactivity_penalty * (self.steps_since_trade - 10)
        
        # Penalize large drawdowns
        if self.current_drawdown_pct > 0.03:  # Only penalize significant drawdowns
            reward -= self.drawdown_penalty * (self.current_drawdown_pct - 0.03)
        
        # Scale reward if needed
        if self.normalize_reward:
            reward *= self.reward_scaling
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Rendered frame
        """
        # Simple text rendering for now
        if mode == 'human':
            portfolio_state = self.simulator.get_portfolio_state()
            market_state = self.simulator.get_market_state()
            
            print(f"Step: {self.current_step}")
            print(f"Timestamp: {self.simulator.current_timestamp}")
            print(f"Position: {self.current_position}")
            print(f"Portfolio Value: ${portfolio_state.get('total_value', 0.0):.2f}")
            print(f"Unrealized P&L: ${self.unrealized_pnl:.2f}")
            print(f"Realized P&L: ${self.realized_pnl:.2f}")
            print(f"Market Price: ${market_state.get('price', 0.0):.4f}")
            print(f"Drawdown: {self.current_drawdown_pct:.2%}")
            print("-" * 40)
            
            return None
        
        # For rgb_array mode, would return visualization of the trading chart, positions, etc.
        # This would require a more complex visualization module
        return None
    
    def close(self):
        """Clean up resources."""
        pass

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        return [seed]


class MomentumTradingReward:
    """
    Specialized reward function for momentum trading strategy.
    Emphasizes:
    - Quick profit capture
    - Early exit on trend reversal
    - Pattern recognition
    - Tape reading
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the reward function.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Reward weights
        self.profit_weight = self.config.get('profit_weight', 1.0)
        self.trend_weight = self.config.get('trend_weight', 0.5)
        self.speed_weight = self.config.get('speed_weight', 0.3)
        self.timing_weight = self.config.get('timing_weight', 0.2)
        self.risk_weight = self.config.get('risk_weight', 0.8)
        
        # Momentum-specific parameters
        self.quick_profit_bonus = self.config.get('quick_profit_bonus', 2.0)  # Bonus for fast profits
        self.trend_continuation_bonus = self.config.get('trend_continuation_bonus', 1.5)  # Bonus for riding momentum
        self.exit_bonus = self.config.get('exit_bonus', 1.0)  # Bonus for exiting declining trends
        self.entry_timing_bonus = self.config.get('entry_timing_bonus', 0.5)  # Bonus for good entry timing
        self.reversal_penalty = self.config.get('reversal_penalty', 2.0)  # Penalty for not exiting reversals
        self.false_signal_penalty = self.config.get('false_signal_penalty', 1.0)  # Penalty for acting on false signals
        
        # Thresholds
        self.quick_profit_threshold = self.config.get('quick_profit_threshold', 0.01)  # 1% profit in short time
        self.quick_time_threshold = self.config.get('quick_time_threshold', 10)  # Steps for "quick" profit
        self.trend_threshold = self.config.get('trend_threshold', 0.005)  # 0.5% move to define trend
    
    def __call__(self, env: TradingEnv, action: float, portfolio_change: float, 
                portfolio_change_pct: float, trade_executed: bool, info: Dict[str, Any]) -> float:
        """
        Calculate reward for the momentum trading strategy.
        
        Args:
            env: Trading environment
            action: Executed action value
            portfolio_change: Absolute change in portfolio value
            portfolio_change_pct: Percentage change in portfolio value
            trade_executed: Whether a trade was executed
            info: Information dictionary
            
        Returns:
            Calculated reward
        """
        # Base reward from portfolio change
        reward = portfolio_change_pct * 100.0 * self.profit_weight
        
        # Get market state
        market_state = env.simulator.get_market_state()
        price = market_state.get('price', 0.0)
        tape_imbalance = market_state.get('tape_imbalance', 0.0)
        
        # Get position and trade info
        position = env.current_position
        action_result = info.get('action_result', {})
        
        # Entry reward components
        if trade_executed and action > 0 and env.current_position > 0:
            # Entry into a long position
            
            # Check tape imbalance (positive = more buying)
            if tape_imbalance > 0.3:
                # Bonus for entering with strong buying momentum
                reward += self.entry_timing_bonus * tape_imbalance
            elif tape_imbalance < -0.2:
                # Penalty for entering against selling pressure
                reward -= self.false_signal_penalty * abs(tape_imbalance)
        
        # Exit reward components
        if trade_executed and action < 0 and env.current_position == 0:
            # Exit from a position
            
            # Check if this was a profitable trade
            realized_pnl = action_result.get('realized_pnl', 0.0)
            
            if realized_pnl > 0:
                # Profitable exit, add bonus based on how quickly the profit was made
                if env.steps_since_trade <= self.quick_time_threshold:
                    quick_bonus = self.quick_profit_bonus * (1.0 - env.steps_since_trade / self.quick_time_threshold)
                    reward += quick_bonus
                
                # Add bonus for exiting at right time (if tape has turned negative)
                if tape_imbalance < -0.1:
                    reward += self.exit_bonus * abs(tape_imbalance)
            else:
                # Loss-making exit
                # If tape is still positive, might be exiting too early
                if tape_imbalance > 0.2:
                    reward -= self.false_signal_penalty * 0.5
                else:
                    # Good decision to cut losses, especially with negative tape
                    reward += self.exit_bonus * (1.0 - abs(tape_imbalance))
        
        # Holding reward components for open positions
        if position > 0 and not trade_executed:
            # Holding a long position
            
            # If tape is turning negative, penalize for not exiting
            if tape_imbalance < -0.3:
                reward -= self.reversal_penalty * abs(tape_imbalance)
            
            # If tape is strongly positive, reward riding the momentum
            if tape_imbalance > 0.4:
                reward += self.trend_continuation_bonus * tape_imbalance
        
        # Penalize large drawdowns more heavily for momentum strategy
        if env.current_drawdown_pct > 0.02:  # Lower threshold for momentum
            reward -= env.drawdown_penalty * (env.current_drawdown_pct - 0.02) * self.risk_weight
        
        return reward