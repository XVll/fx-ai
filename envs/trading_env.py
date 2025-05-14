import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
import logging

# Import typed Config classes directly
from config.config import Config, EnvConfig
from envs.trading_reward import TradingReward
from envs.trading_simulator import TradingSimulator


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(
            self,
            trading_simulator: TradingSimulator,
            cfg: EnvConfig,  # Use the typed EnvConfig directly
            reward_function: Optional[callable] = None,
            logger: logging.Logger = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.simulator = trading_simulator

        # Initialize reward function
        if reward_function is None:
            self.reward_fn = TradingReward(cfg.reward)
        else:
            self.reward_fn = reward_function

        # Environment configuration - direct access to typed fields
        self.state_dim = cfg.state_dim
        self.max_steps = cfg.max_steps
        self.normalize_state = cfg.normalize_state
        self.random_reset = cfg.random_reset

        # Get reward config parameters
        self.reward_type = cfg.reward.type
        self.reward_scaling = cfg.reward.scaling
        self.trade_penalty = cfg.reward.trade_penalty
        self.hold_penalty = cfg.reward.hold_penalty
        self.early_exit_bonus = cfg.reward.early_exit_bonus
        self.flush_prediction_bonus = cfg.reward.flush_prediction_bonus

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