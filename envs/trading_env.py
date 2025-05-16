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

        # Define explicit action types
        self.ACTION_HOLD = 0  # Do nothing, maintain current position
        self.ACTION_ENTER_LONG = 1  # Enter initial long position
        self.ACTION_SCALE_IN = 2  # Add to existing long position
        self.ACTION_SCALE_OUT = 3  # Reduce long position (partial take profit)
        self.ACTION_EXIT = 4  # Close entire long position
        self.position_sizes = [0.25, 0.50, 0.75, 1.00]

        # === REQUIRED BY GYM.ENV - Action and observation spaces ===
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(5),
                gym.spaces.Discrete(len(self.position_sizes)),  # Action space for position size
            )
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
    # In trading_env.py - step method

    # envs/trading_env.py - Fixed version for step method
    def step(self, action):
        """
        Take a step in the environment with the given action.
        Fixed to handle end of data and None market state.

        Args:
            action: Action to take (action_type, size_idx)
                - Can be a tuple, list, or two-element array

        Returns:
            tuple: (observation, reward, terminated, truncated, info) as required by Gymnasium API
        """
        # Ensure action is in the correct format
        if isinstance(action, (list, np.ndarray)):
            action = tuple(map(int, action[:2]))  # Convert to tuple of integers

        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Increment step counter
        self.current_step += 1

        # Process action and update market/portfolio
        self._process_action(action)

        # Get the new state
        observation, state_dict = self._get_observation()

        # Check if we've reached the end of data - critical fix!
        if observation is None or self.market_simulator is None or self.market_simulator.is_done():
            # Set terminated flag when we've reached end of data
            self.logger.info("End of data reached, terminating episode")
            terminated = True
            truncated = False

            # Use last valid state or zeros as observation
            if self.current_state is not None:
                observation = self.current_state
            else:
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)

            # Use empty dict for state_dict
            state_dict = {}

            # Use zero reward at end of data
            reward = 0.0

            # Set info with reason
            self.info = self._get_info()
            self.info['end_reason'] = 'end_of_data'

            return observation, reward, terminated, truncated, self.info

        # Continue with normal flow
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

    # In trading_env.py - _process_action method

    # Fix for _process_action method in trading_env.py

    def _process_action(self, action):
        """
        Process and execute the trading action.

        Args:
            action: Tuple of (action_type, size_idx)
                - action_type: Integer code for action type (0=HOLD, 1=ENTER, etc.)
                - size_idx: Index for self.position_sizes
        """
        if not self.market_simulator or not self.portfolio_simulator or not self.execution_simulator:
            self.logger.warning("Cannot process action: simulators not initialized")
            return

        # Unpack action
        action_type, size_idx = action

        # Initialize action_name with a default value to prevent UnboundLocalError
        action_name = f"UNKNOWN({action_type})"

        # Map size_idx to actual position size
        position_size = self.position_sizes[size_idx]

        # Get current portfolio state
        portfolio_state = self.portfolio_simulator.get_portfolio_state()
        current_position = portfolio_state['position']

        # Get current market data
        market_state = self.market_simulator.get_current_market_state()

        # Check for timestamp in market state and add it if missing
        if 'timestamp_utc' not in market_state and 'timestamp' not in market_state:
            # Use current simulator timestamp as fallback
            if hasattr(self.market_simulator, 'current_timestamp_utc') and self.market_simulator.current_timestamp_utc:
                market_state['timestamp'] = self.market_simulator.current_timestamp_utc
                self.logger.debug(
                    f"Added missing timestamp to market state: {self.market_simulator.current_timestamp_utc}")

        # Get timestamp from market state, using either key (timestamp_utc or timestamp)
        current_timestamp = market_state.get('timestamp_utc', market_state.get('timestamp'))

        # Get current price from 1s bar if available
        current_price = 0.0
        if 'current_1s_bar' in market_state and market_state['current_1s_bar']:
            current_price = market_state['current_1s_bar'].get('close', 0.0)
        elif 'current_price' in market_state:
            current_price = market_state.get('current_price', 0.0)

        if not current_timestamp:
            self.logger.warning("Cannot process action: missing timestamp in market state")
            return

        # Default values - no execution
        delta = 0.0
        order_type = None

        # Process based on action type
        if action_type == self.ACTION_HOLD:
            # No action needed
            action_name = "HOLD"

        elif action_type == self.ACTION_ENTER_LONG:
            if current_position <= 0:  # Only if not already long
                # Calculate target position
                target_position = position_size * self.max_position
                delta = target_position - current_position
                order_type = 'buy'
                action_name = f"ENTER_LONG({position_size})"
            else:
                # Already in position but tried to enter again
                action_name = f"ENTER_LONG_IGNORED(already in position)"

        elif action_type == self.ACTION_SCALE_IN:
            if current_position > 0:  # Only if already long
                # Calculate new target position
                new_target = min(self.max_position,
                                 current_position + (position_size * self.max_position))
                delta = new_target - current_position
                if delta > 0:
                    order_type = 'buy'
                    action_name = f"SCALE_IN({position_size})"
                else:
                    action_name = f"SCALE_IN_IGNORED(no additional position possible)"
            else:
                action_name = f"SCALE_IN_IGNORED(no existing position)"

        elif action_type == self.ACTION_SCALE_OUT:
            if current_position > 0:  # Only if long
                # Calculate reduction amount
                reduction = position_size * current_position
                delta = -reduction  # Negative delta for selling
                order_type = 'sell'
                action_name = f"SCALE_OUT({position_size})"
            else:
                action_name = f"SCALE_OUT_IGNORED(no position to reduce)"

        elif action_type == self.ACTION_EXIT:
            if current_position > 0:  # Only if long
                delta = -current_position  # Close entire position
                order_type = 'sell'
                action_name = "EXIT"
            else:
                action_name = f"EXIT_IGNORED(no position to exit)"

        else:
            action_name = f"UNKNOWN({action_type})"
            self.logger.warning(f"Unknown action type: {action_type}")

        # Execute the order if needed
        execution_result = None
        if order_type and abs(delta) > 0.0001:
            execution_result = self.execution_simulator.execute_order(
                order_type=order_type,
                size=abs(delta),
                timestamp=current_timestamp
            )

            # Update portfolio with execution result
            if execution_result['status'] == 'executed':
                self.portfolio_simulator.update_portfolio(execution_result)

                # Update episode stats
                self.episode_stats['trades_executed'] += 1
                self.episode_stats['position_changes'] += 1

                self.logger.info(f"Executed {action_name}: {order_type} {abs(delta):.4f} @ {current_price:.4f}")

        # Store result in info
        if execution_result:
            execution_result['action_name'] = action_name
            self.info['action_result'] = execution_result
        else:
            self.info['action_result'] = {
                'status': 'no_action',
                'reason': action_name,
                'timestamp': current_timestamp
            }

        # Advance the market simulator to the next time step
        self.market_simulator.step()

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
