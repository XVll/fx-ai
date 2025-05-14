from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Union, Optional
import logging

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from config.config import Config
from envs.trading_reward import TradingReward
from envs.trading_simulator import TradingSimulator


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
            self,
            trading_simulator: TradingSimulator,
            cfg: Config,
            reward_function: Optional[callable] = None,
            logger: logging.Logger = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.simulator = trading_simulator

        if reward_function is None:
            reward_config = cfg if not hasattr(cfg, 'reward') else cfg.reward
            reward_type = cfg.get('reward_type', 'momentum')


            if reward_type == 'momentum':
                self.reward_fn = TradingReward(reward_config)
            else:
                # Default simple reward function
                self.reward_fn = lambda env, action, change, pct, traded, info: change
        else:
            self.reward_fn = reward_function

        # Look for parameters in either top-level or nested format
        general = cfg.get('general', cfg)  # Try general section, fallback to top-level
        reward = cfg.get('reward', cfg)  # Try reward section, fallback to top-level

        # Environment configuration
        self.state_dim = cfg.get('state_dim', 20)  # Dimension of state vector
        self.max_steps = cfg.get('max_steps', 1000)  # Maximum steps per episode
        self.normalize_state = cfg.get('normalize_state', True)
        self.random_reset = cfg.get('random_reset', True)

        # Reward parameters from config
        self.reward_type = reward.get('type', reward.get('reward_type', 'momentum'))
        self.reward_scaling = reward.get('scaling', reward.get('reward_scaling', 1.0))
        self.trade_penalty = reward.get('trade_penalty', 0.1)
        self.hold_penalty = reward.get('hold_penalty', 0.0)
        self.early_exit_bonus = reward.get('early_exit_bonus', 0.5)
        self.flush_prediction_bonus = reward.get('flush_prediction_bonus', 2.0)

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
        """
        # Convert action if needed
        action_value = float(action[0].item() if hasattr(action[0], "item") else action[0]) if hasattr(action,
                                                                                                       "__len__") else float(
            action)

        # Execute in simulator
        simulator_result = self.simulator.step(action_value)
        next_state, raw_reward, done, info = simulator_result

        # Calculate portfolio metrics for reward
        portfolio_value = self.simulator.get_portfolio_state().get('total_value', 0)
        prev_portfolio_value = info.get('prev_portfolio_value', portfolio_value)
        portfolio_change = portfolio_value - prev_portfolio_value
        portfolio_change_pct = portfolio_change / prev_portfolio_value if prev_portfolio_value > 0 else 0

        # Use custom reward function
        reward = self.reward_fn(
            self,
            action_value,
            portfolio_change,
            portfolio_change_pct,
            info.get('action_result', {}).get('action', '') != 'hold',  # True if trade executed
            info
        )

        # Update step count and info
        self.current_step += 1

        # Record portfolio value for next step
        info['prev_portfolio_value'] = portfolio_value
        info['total_reward'] = self.total_reward + reward
        self.total_reward += reward

        # Check for maximum steps
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            done = True

        # Get normalized state
        norm_state = self._get_normalized_state()

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
