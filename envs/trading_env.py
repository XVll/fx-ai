# envs/trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
import logging

from simulation.simulator import Simulator


class TradingEnv(gym.Env):
    """
    Simplified OpenAI Gym-compatible environment for financial trading.
    """

    metadata = {'render_modes': ['human']}

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
        self.state_dim = self.config.get('state_dim', 20)  # Dimension of state vector
        self.max_steps = self.config.get('max_steps', 1000)  # Maximum steps per episode
        self.normalize_state = self.config.get('normalize_state', True)

        # Define action and observation space
        # Action space: continuous value between -1 and 1
        # -1 = full sell, 0 = hold, 1 = full buy
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space: vector of state variables
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Environment state
        self.current_step = 0
        self.total_reward = 0.0

    # envs/trading_env.py (modify the _get_normalized_state method)

    def _get_normalized_state(self) -> np.ndarray:
        """
        Get normalized state representation for RL model.

        Returns:
            NumPy array with normalized state
        """
        # Get state from simulator - this now uses the tensor dict format
        # but we'll convert it back to flat array for compatibility
        raw_state_dict = self.simulator.get_current_state_tensor_dict()

        # For backward compatibility, flatten the tensors
        flattened_state = []

        # Add high-frequency features (truncated if too large)
        hf = raw_state_dict['hf_features'].cpu().numpy().flatten()
        if len(hf) > 0:
            flattened_state.append(hf[:min(len(hf), 600)])  # Limit to 600 elements

        # Add medium-frequency features
        mf = raw_state_dict['mf_features'].cpu().numpy().flatten()
        if len(mf) > 0:
            flattened_state.append(mf[:min(len(mf), 300)])  # Limit to 300 elements

        # Add low-frequency features
        lf = raw_state_dict['lf_features'].cpu().numpy().flatten()
        if len(lf) > 0:
            flattened_state.append(lf[:min(len(lf), 300)])  # Limit to 300 elements

        # Add static features
        sf = raw_state_dict['static_features'].cpu().numpy().flatten()
        if len(sf) > 0:
            flattened_state.append(sf)

        # Concatenate all features
        flat_state = np.concatenate(flattened_state) if flattened_state else np.array([])

        # If state is empty, return zeros
        if len(flat_state) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        # Ensure right length
        if len(flat_state) > self.state_dim:
            # Truncate if too long
            norm_state = flat_state[:self.state_dim]
        elif len(flat_state) < self.state_dim:
            # Pad with zeros if too short
            norm_state = np.pad(flat_state, (0, self.state_dim - len(flat_state)))
        else:
            norm_state = flat_state

        return norm_state.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        Returns:
            Tuple of (initial_state, info)
        """
        super().reset(seed=seed)

        # Reset simulator
        state_dict = self.simulator.reset()

        # Reset environment state
        self.current_step = 0
        self.total_reward = 0.0

        # Get normalized state
        norm_state = self._get_normalized_state()

        return norm_state, {}

    def step(self, action):
        """
        Take a step in the environment by executing an action.

        Args:
            action: Action to take (normalized -1.0 to 1.0)

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Ensure action is in correct format
        action_value = float(action[0].item() if hasattr(action[0], "item") else action[0]) if hasattr(action, "__len__") else float(action)

        # Execute in simulator
        simulator_result = self.simulator.step(action_value)
        next_state, reward, done, info = simulator_result

        # Update step count
        self.current_step += 1
        self.total_reward += reward

        # Check for maximum steps
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            done = True

        # Get normalized state
        norm_state = self._get_normalized_state()

        # Update info with step information
        info.update({
            'step': self.current_step,
            'total_reward': self.total_reward,
        })

        self.logger.info(f"Step {self.current_step}: Reward={reward:.4f}, Total={self.total_reward:.4f}")

        if done:
            self.logger.info(f"Episode finished: Steps={self.current_step}, Total Reward={self.total_reward:.4f}")

        return norm_state, reward, done, truncated, info

    def _get_normalized_state(self) -> np.ndarray:
        """
        Get normalized state representation for RL model.

        Returns:
            NumPy array with normalized state
        """
        # Get state from simulator
        raw_state = self.simulator.get_current_state_array()

        # If state is empty, return zeros
        if raw_state is None or len(raw_state) == 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        # Simple normalization - just ensure right length
        if len(raw_state) > self.state_dim:
            # Truncate if too long
            norm_state = raw_state[:self.state_dim]
        elif len(raw_state) < self.state_dim:
            # Pad with zeros if too short
            norm_state = np.pad(raw_state, (0, self.state_dim - len(raw_state)))
        else:
            norm_state = raw_state

        return norm_state.astype(np.float32)

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode: Rendering mode

        Returns:
            Rendered frame (None for human mode)
        """
        # Simple text rendering
        if mode == 'human':
            portfolio_state = self.simulator.get_portfolio_state()
            market_state = self.simulator.get_market_state()

            print(f"Step: {self.current_step}")
            print(f"Timestamp: {market_state.get('timestamp')}")
            print(f"Price: ${market_state.get('price', 0.0):.4f}")
            print(f"Portfolio Value: ${portfolio_state.get('total_value', 0.0):.2f}")
            print("-" * 40)

        return None

    def close(self):
        """Clean up resources."""
        pass


class MomentumTradingReward:
    """
    Minimal reward function for momentum trading.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the reward function.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def __call__(self, env: TradingEnv, action: float, portfolio_change: float,
                 portfolio_change_pct: float, trade_executed: bool, info: Dict[str, Any]) -> float:
        """
        Calculate reward for momentum trading.

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
        # Simplified reward - just use portfolio change
        reward = portfolio_change

        # Small penalty for trading
        if trade_executed:
            reward -= 0.1

        return reward
