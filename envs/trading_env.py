# trading_env.py
import gymnasium as gym
import numpy as np
import torch
import logging

from data.data_manager import DataManager
from envs.RewardCalculator import RewardCalculator
from feature.feature_extractor import FeatureExtractor
from simulators.execution_simulator import ExecutionSimulator
from simulators.market_simulator import MarketSimulator
from simulators.portfolio_simulator import PortfolioSimulator


class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning.

    This environment simulates a high-frequency trading scenario for momentum/squeeze
    trading with 1-second decision frequency on volatile, low-float stocks.

    Implements the standard gymnasium (gym) interface with custom trading components.
    """

    def __init__(self, data_manager: DataManager, cfg=None, logger=None):
        """
        Initialize the trading environment.

        Args:
            cfg: Configuration object or dictionary
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.cfg = cfg or {}

        # Data manager for historical data
        self.data_manager = data_manager
        # Extract configuration parameters
        if hasattr(cfg, 'state_dim'):
            self.state_dim = cfg.state_dim
        else:
            self.state_dim = self.cfg.get('state_dim', 1000)

        if hasattr(cfg, 'max_steps'):
            self.max_steps = cfg.max_steps
        else:
            self.max_steps = self.cfg.get('max_steps', 500)

        if hasattr(cfg, 'normalize_state'):
            self.normalize_state = cfg.normalize_state
        else:
            self.normalize_state = self.cfg.get('normalize_state', True)

        if hasattr(cfg, 'random_reset'):
            self.random_reset = cfg.random_reset
        else:
            self.random_reset = self.cfg.get('random_reset', True)

        if hasattr(cfg, 'max_position'):
            self.max_position = cfg.max_position
        else:
            self.max_position = self.cfg.get('max_position', 1.0)

        # Rendering mode (for visualization if needed)
        self.render_mode = 'human'

        # === REQUIRED BY GYM.ENV - Action and observation spaces ===
        # Action space: continuous [-1, 1] for position sizing
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space: continuous state vector
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Initialize simulation components
        self.market_simulator = None
        self.execution_simulator = None
        self.portfolio_simulator = None
        self.feature_extractor = None
        self.reward_calculator = None

        # Current state tracking
        self.current_step = 0
        self.current_state = None
        self.current_state_dict = None  # For structured state representation
        self.done = False
        self.info = {}

        # Episode statistics
        self.episode_stats = {
            'total_reward': 0.0,
            'total_pnl': 0.0,
            'trades_executed': 0,
            'position_changes': 0,
        }

        # Feature normalization
        self.feature_means = None
        self.feature_stds = None

        self.logger.info("TradingEnv initialized with state_dim={}, max_steps={}".format(
            self.state_dim, self.max_steps))

    # === REQUIRED BY GYM.ENV - Reset method ===
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        Args:
            seed: Optional random seed
            options: Additional options for reset

        Returns:
            tuple: (observation, info) as required by Gymnasium API
        """
        # Set a random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Process options
        options = options or {}
        random_start = options.get('random_start', self.random_reset)

        # Reset step counter and episode stats
        self.current_step = 0
        self.done = False
        self.episode_stats = {
            'total_reward': 0.0,
            'total_pnl': 0.0,
            'trades_executed': 0,
            'position_changes': 0,
        }

        # Reset simulation components
        if self.market_simulator:
            self.market_simulator.reset({
                'random_start': random_start,
                'max_steps': self.max_steps
            })

        if self.execution_simulator:
            self.execution_simulator.reset()

        if self.portfolio_simulator:
            self.portfolio_simulator.reset()

        if self.feature_extractor:
            self.feature_extractor.reset()

        if self.reward_calculator:
            self.reward_calculator.reset()

        # Get initial state
        self.current_state, self.current_state_dict = self._get_observation()

        # Reset info dict
        self.info = {}

        # IMPORTANT: Return EXACTLY two values - the observation and info
        result= self.current_state, self.info
        print(
            f"Reset returning: {result}, type: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'not a tuple'}")
        return result

    # === REQUIRED BY GYM.ENV - Step method ===
    def step(self, action):
        """
        Take a step in the environment with the given action.

        Args:
            action: Action to take (position size adjustment)
                - Shape: (1,)
                - Range: [-1.0, 1.0] where:
                  * -1.0 = maximum short position
                  * 0.0 = flat (no position)
                  * 1.0 = maximum long position

        Returns:
            tuple: (observation, reward, terminated, truncated, info) as required by Gymnasium API
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Increment step counter
        self.current_step += 1

        # Process action and update market/portfolio
        self._process_action(action)

        # Get the new state
        observation, state_dict = self._get_observation()
        self.current_state = observation
        self.current_state_dict = state_dict

        # Calculate reward
        reward = self._calculate_reward()

        # Update episode statistics
        self.episode_stats['total_reward'] += reward
        if self.portfolio_simulator:
            portfolio_state = self.portfolio_simulator.get_portfolio_state()
            self.episode_stats['total_pnl'] = portfolio_state.get('total_pnl', 0.0)

        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Update info dict with current state information
        self.info = self._get_info()

        # Add episode stats to info if episode is ending
        if terminated or truncated:
            self.info['episode'] = self.episode_stats

        return observation, reward, terminated, truncated, self.info

    # === OPTIONAL GYM.ENV METHOD - Render method ===
    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode: Rendering mode (default is 'human')
        """
        if self.render_mode == 'human':
            # Simple text output for now
            if self.portfolio_simulator:
                portfolio_state = self.portfolio_simulator.get_portfolio_state()
                print(f"Step: {self.current_step}, Position: {portfolio_state['position']:.2f}, "
                      f"Cash: ${portfolio_state['cash']:.2f}, Value: ${portfolio_state['total_value']:.2f}")

    # === OPTIONAL GYM.ENV METHOD - Close method ===
    def close(self):
        """
        Clean up environment resources.

        This method is part of the gym.Env interface.
        """
        if self.market_simulator:
            self.market_simulator.close()

    # === HELPER METHODS ===
    def _get_observation(self):
        """
        Build the state representation (observation) from market data.

        Returns:
            tuple: (observation, state_dict)
                - observation: Flattened numpy array with shape (state_dim,)
                - state_dict: Structured dictionary for model input
        """
        if not self.market_simulator or not self.feature_extractor:
            # Return zeros if components aren't initialized
            flat_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            return flat_state, {}

        # Get current market state from simulator
        market_state = self.market_simulator.get_current_market_state()

        # Get current portfolio state
        portfolio_state = None
        if self.portfolio_simulator:
            portfolio_state = self.portfolio_simulator.get_portfolio_state()

        # Extract features using the feature extractor
        features = self.feature_extractor.extract_features(
            market_state=market_state,
            portfolio_state=portfolio_state
        )

        # Apply normalization if configured
        if self.normalize_state:
            normalized_features = self.feature_extractor.normalize_features(features)
        else:
            normalized_features = features

        # For transformer model, prepare structured state dict
        state_dict = {
            'hf_features': normalized_features.get('hf_features'),  # Shape: [batch, seq_len, feat_dim]
            'mf_features': normalized_features.get('mf_features'),
            'lf_features': normalized_features.get('lf_features'),
            'static_features': normalized_features.get('static_features'),
            # Todo: Add position and cash features
        }

        # ALSO create a flattened version for the gym API
        # We need to flatten all the features into a single array
        flat_features = []

        # Flatten and add static features if available
        if state_dict.get('static_features') is not None:
            flat_features.append(state_dict['static_features'].flatten())

        # Flatten and add sequence features
        for key in ['hf_features', 'mf_features', 'lf_features']:
            if state_dict.get(key) is not None:
                # Reshape to flatten while preserving batch dim
                shape = state_dict[key].shape
                flat_features.append(state_dict[key].reshape(1, -1).flatten())

        # Combine all flattened features
        if flat_features:
            flat_state = np.concatenate(flat_features)

            # Ensure it matches the expected observation space size
            if len(flat_state) > self.observation_space.shape[0]:
                self.logger.warning(
                    f"Flattened state size ({len(flat_state)}) exceeds observation_space "
                    f"({self.observation_space.shape[0]}). Truncating.")
                flat_state = flat_state[:self.observation_space.shape[0]]
            elif len(flat_state) < self.observation_space.shape[0]:
                # Pad with zeros if needed
                padding = np.zeros(self.observation_space.shape[0] - len(flat_state), dtype=np.float32)
                flat_state = np.concatenate([flat_state, padding])
        else:
            # Fallback to zeros if no features are available
            flat_state = np.zeros(self.observation_space.shape, dtype=np.float32)

        return flat_state, state_dict

    def _process_action(self, action):
        """
        Process and execute the trading action.

        Args:
            action: Action value from the agent (-1.0 to 1.0)
        """
        if not self.market_simulator or not self.portfolio_simulator or not self.execution_simulator:
            self.logger.warning("Cannot process action: simulators not initialized")
            return

        # Convert normalized action (-1 to 1) to a target position size
        # -1.0 = maximum short, 0.0 = flat, 1.0 = maximum long
        target_position = float(action[0]) * self.max_position

        # Get current market data
        market_state = self.market_simulator.get_current_market_state()
        current_price = market_state.get('current_price')
        current_timestamp = market_state.get('timestamp')

        if current_price is None:
            self.logger.warning("No current price available, cannot execute action")
            return

        # 1. Get current portfolio state
        portfolio_state = self.portfolio_simulator.get_portfolio_state()
        current_position = portfolio_state['position']

        # 2. Calculate position delta
        delta = target_position - current_position

        # 3. Execute the order through the execution simulator
        if abs(delta) > 0.0001:  # Only execute if delta is significant
            order_type = 'buy' if delta > 0 else 'sell'

            execution_result = self.execution_simulator.execute_order(
                order_type=order_type,
                size=abs(delta),
                timestamp=current_timestamp
            )

            # 4. Update portfolio with execution result
            if execution_result['status'] == 'executed':
                self.portfolio_simulator.update_portfolio(execution_result)

                # Update episode stats
                self.episode_stats['trades_executed'] += 1
                self.episode_stats['position_changes'] += 1
        else:
            execution_result = {
                'status': 'no_action',
                'reason': 'position_delta_too_small',
                'timestamp': current_timestamp
            }

        # 5. Advance the market simulator to the next time step
        self.market_simulator.step()

        # 6. Store order result in info
        self.info['action_result'] = execution_result

    def _calculate_reward(self):
        """
        Calculate the reward for the current step.

        Returns:
            float: Reward value
        """
        if not self.reward_calculator:
            return 0.0

        # Get required states
        market_state = None
        portfolio_state = None

        if self.market_simulator:
            market_state = self.market_simulator.get_current_market_state()

        if self.portfolio_simulator:
            portfolio_state = self.portfolio_simulator.get_portfolio_state()

        # Calculate reward using the reward calculator
        reward = self.reward_calculator.calculate_reward(
            market_state=market_state,
            portfolio_state=portfolio_state,
            info=self.info
        )

        return reward

    def _is_terminated(self):
        """
        Check if the episode has terminated (reached a terminal state).

        Returns:
            bool: True if terminated, False otherwise
        """
        # Check if we've reached the end of market data
        if self.market_simulator and self.market_simulator.is_done():
            return True

        # Check for other termination conditions (bankruptcy, etc.)
        if self.portfolio_simulator:
            portfolio_state = self.portfolio_simulator.get_portfolio_state()

            # Terminate if account value drops below threshold
            min_account_value = self.cfg.get('min_account_value', 0.0)
            if portfolio_state.get('total_value', 0) <= min_account_value:
                self.logger.info(f"Episode terminated due to account value below threshold: "
                                 f"${portfolio_state.get('total_value', 0):.2f} <= ${min_account_value:.2f}")
                return True

        return False

    def _is_truncated(self):
        """
        Check if the episode should be truncated (e.g., reached max steps).

        Returns:
            bool: True if truncated, False otherwise
        """
        # Check if we've reached max steps
        if self.current_step >= self.max_steps:
            return True

        return False

    def _get_info(self):
        """
        Get diagnostic information about the current state.

        Returns:
            dict: Information dictionary
        """
        info = {
            'step': self.current_step,
        }

        # Add market information
        if self.market_simulator:
            market_state = self.market_simulator.get_current_market_state()
            if market_state:
                info['timestamp'] = market_state.get('timestamp')
                info['price'] = market_state.get('current_price')
                info['volume'] = market_state.get('current_volume', 0)

        # Add portfolio information
        if self.portfolio_simulator:
            portfolio_state = self.portfolio_simulator.get_portfolio_state()
            if portfolio_state:
                info['position'] = portfolio_state.get('position', 0)
                info['cash'] = portfolio_state.get('cash', 0)
                info['equity'] = portfolio_state.get('total_value', 0)
                info['unrealized_pnl'] = portfolio_state.get('unrealized_pnl', 0)
                info['realized_pnl'] = portfolio_state.get('realized_pnl', 0)
                info['total_pnl'] = portfolio_state.get('total_pnl', 0)

        # Add episode stats to info
        for k, v in self.episode_stats.items():
            info[f'episode_{k}'] = v

        return info

    # === CUSTOM METHOD - Not part of gym.Env ===
    def initialize_for_symbol(self, symbol, mode='backtesting',
                              start_time=None, end_time=None, timeframes=None):
        """
        Initialize the environment for trading a specific symbol.

        Args:
            symbol: Trading symbol
            mode: 'backtesting' or 'live'
            start_time: Start time for historical data (backtesting)
            end_time: End time for historical data (backtesting)
            timeframes: List of timeframes to load (e.g., ['1s', '1m', '5m'])

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info(f"Initializing environment for {symbol} in {mode} mode, from {start_time} to {end_time}")

        # Default timeframes if not provided
        if timeframes is None:
            timeframes = ['1s', '1m', '5m', '1d']

        try:
            # 1. Initialize market simulator with updated parameters
            # Todo: Reset market from random point every episode
            self.market_simulator = MarketSimulator(
                symbol=symbol,
                data_manager=self.data_manager,
                mode=mode,
                start_time=start_time,
                end_time=end_time,
                config=self.cfg.get('market_config', {}),
                logger=self.logger
            )

            # 2. Initialize execution simulator
            execution_config = getattr(self.cfg, 'execution_config', {})
            self.execution_simulator = ExecutionSimulator(
                market_simulator=self.market_simulator,
                config=execution_config,
                logger=self.logger
            )

            # 3. Initialize portfolio simulator
            portfolio_config = getattr(self.cfg, 'portfolio_config', {})
            self.portfolio_simulator = PortfolioSimulator(
                config=portfolio_config,
                logger=self.logger
            )

            # 4. Initialize feature extractor
            feature_config = getattr(self.cfg, 'feature_config', {})
            self.feature_extractor = FeatureExtractor(
                symbol=symbol,
                config=feature_config,
                logger=self.logger
            )

            # 5. Initialize reward calculator
            reward_config = getattr(self.cfg, 'reward_config', {})
            self.reward_calculator = RewardCalculator(
                config=reward_config,
                logger=self.logger
            )

            # Reset the environment
            self.reset()

            self.logger.info(f"Environment initialized successfully for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing environment: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def get_portfolio_state(self):
        """
        Get the current portfolio state.

        Returns:
            dict: Portfolio state or None if not available
        """
        if self.portfolio_simulator:
            return self.portfolio_simulator.get_portfolio_state()
        return None

    def get_trade_history(self):
        """
        Get the trade history.

        Returns:
            list: List of trade dictionaries or empty list if not available
        """
        if self.portfolio_simulator:
            return self.portfolio_simulator.get_trade_history()
        return []

    def get_model_state_dict(self):
        """
        Get the current state in the format expected by the model.

        This is useful for external model inference without stepping the environment.

        Returns:
            dict: State dictionary for model input
        """
        return self.current_state_dict
