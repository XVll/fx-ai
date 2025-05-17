from datetime import datetime
from enum import Enum
from typing import TypedDict
import gymnasium as gym
import numpy as np
import logging
from config.config import Config
from data.data_manager import DataManager
from envs.reward import RewardCalculator
from feature.feature_extractor import FeatureExtractor
from simulators.execution_simulator import ExecutionSimulator
from simulators.market_simulator import MarketSimulator
from simulators.portfolio_simulator import PortfolioSimulator

class EpisodeStats(TypedDict, total=False):
    total_reward: float
    total_pnl: float
    trades_executed: int
    position_changes: int
class StepStats(TypedDict, total=False):
    step: int
    position_changes: int
    position: float
    cash: float
    equity: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    timestamp: datetime
    price: float
    volume: float
    action_result: dict
    end_reason: str
    episode_stats_at_step: EpisodeStats
class EnvironmentStats(TypedDict, total=False):
    step: StepStats
    episode: EpisodeStats
class ActionType(Enum):
    HOLD = 0  # Do nothing, maintain current position
    ENTER_LONG = 1  # Enter the initial long position
    SCALE_IN = 2  # Add to the existing long position
    SCALE_OUT = 3  # Reduce long position (partial take profit)
    EXIT = 4  # Close the entire long position
class TrainingMode(Enum):
    BACKTESTING = 0
    LIVE = 1

class TradingEnv(gym.Env):
    def __init__(self, data_manager: DataManager, config: Config, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config

        self.data_manager = data_manager

        self.render_mode = 'human'

        # Action space: discrete actions
        self.position_sizes = [0.25, 0.50, 0.75, 1.00]

        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(len(ActionType)),
                gym.spaces.Discrete(len(self.position_sizes)),
            )
        )

        # Observation space: continuous state vector
        # Low and High is the range of the feature values, shape is the dimension of the array (seq_len * feat_dim)
        self.observation_space = gym.spaces.Dict(
            {
                'hf_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.model.hf_seq_len, self.config.model.hf_feat_dim), dtype=np.float32),
                'mf_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.model.mf_seq_len, self.config.model.mf_feat_dim), dtype=np.float32),
                'lf_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.model.lf_seq_len, self.config.model.lf_feat_dim), dtype=np.float32),
                'static_features': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.model.static_feat_dim,), dtype=np.float32),
            }
        )

        # Initialize simulation components
        self.market_simulator = None
        self.execution_simulator = None
        self.portfolio_simulator = None
        self.feature_extractor = None
        self.reward = None

        # Current state tracking
        self.obs = None

        self.current_step = 0
        self.environment_stats: EnvironmentStats = {}
        self.episode_stats: EpisodeStats = {}

        self.logger.info(f"Trading environment initialized with state_dim={config.env.state_dimension}, max_steps={config.env.max_steps}")

    def _is_terminated(self):
        # Check if we've reached the end of market data
        if self.market_simulator.is_done(): return True

        # Check for other termination conditions (bankruptcy, etc.)
        # if self.portfolio_simulator:
        #     portfolio_state = self.portfolio_simulator.get_portfolio_state()
        #
        #     # Terminate if account value drops below threshold
        #     min_account_value = self.config.get('min_account_value', 0.0)
        #     if portfolio_state.get('total_value', 0) <= min_account_value:
        #         self.logger.info(f"Episode terminated due to account value below threshold: "
        #                          f"${portfolio_state.get('total_value', 0):.2f} <= ${min_account_value:.2f}")
        #         return True
        #
        return False

    def _is_truncated(self):
        if self.current_step >= self.config.env.max_steps:
            return True

        return False

    def _calculate_reward(self):
        """Calculate the reward based on the current step"""

        market_state = self.market_simulator.get_current_market_state()
        portfolio_state = self.portfolio_simulator.get_portfolio_state()

        # Calculate reward using the reward calculator
        # reward = self.reward.calculate(market_state=market_state, portfolio_state=portfolio_state, info=self.info)
        # Todo : Implement better reward calculation
        # Todo : return reward
        return 0

    def _get_step_stats(self):
        """ Get diagnostic information about the current state. Especially useful for debugging and w&b."""
        si: StepStats = {'step': self.current_step}

        market_state = self.market_simulator.get_current_market_state()
        if market_state:
            si['timestamp'] = market_state.get('timestamp')
            si['price'] = market_state.get('current_price')
            si['volume'] = market_state.get('current_volume', 0)

        portfolio_state = self.portfolio_simulator.get_portfolio_state()
        if portfolio_state:
            si['position'] = portfolio_state.get('position', 0)
            si['cash'] = portfolio_state.get('cash', 0)
            si['equity'] = portfolio_state.get('total_value', 0)
            si['unrealized_pnl'] = portfolio_state.get('unrealized_pnl', 0)
            si['realized_pnl'] = portfolio_state.get('realized_pnl', 0)
            si['total_pnl'] = portfolio_state.get('total_pnl', 0)

        si['episode_stats_at_step'] = self.episode_stats

        return si

    def _get_observation(self):
        """Get the current observation from the environment."""
        if not self.market_simulator or not self.feature_extractor or not self.portfolio_simulator:
            self.logger.warning("Cannot get observation: simulators not initialized")
            return None

        market_state = self.market_simulator.get_current_market_state()
        portfolio_state = self.portfolio_simulator.get_portfolio_state()

        # Extract features using the feature extractor
        features = self.feature_extractor.extract_features(
            market_state=market_state,
            portfolio_state=portfolio_state
        )

        # Apply normalization
        # Todo : Do we need none normalization?
        if self.config.env.normalize_state:
            normalized_features = self.feature_extractor.normalize_features(features)
        else:
            normalized_features = features

        # For transformer model and gym, prepare a structured state dict
        # Todo : Implement better normalization
        obs = {
            'hf_features': normalized_features.get('hf_features'),  # Shape: [seq_len, feat_dim]
            'mf_features': normalized_features.get('mf_features'),
            'lf_features': normalized_features.get('lf_features'),
            'static_features': normalized_features.get('static_features'),
            # Todo: Add position and cash features
        }

        return obs

    def render(self, mode='human'):
        """  """
        # Todo: Learn how to use this
        if self.render_mode == 'human':
            # Simple text output for now
            if self.portfolio_simulator:
                portfolio_state = self.portfolio_simulator.get_portfolio_state()
                print(f"Step: {self.current_step}, Position: {portfolio_state['position']:.2f}, "
                      f"Cash: ${portfolio_state['cash']:.2f}, Value: ${portfolio_state['total_value']:.2f}")

    def initialize(self, symbol, mode: TrainingMode = TrainingMode.BACKTESTING, start_time=None, end_time=None) -> None:
        """Initialize the environment with a specific symbol and mode"""
        self.logger.info(f"Initializing environment for {symbol} in {mode} mode, from {start_time} to {end_time}")

        try:
            self.market_simulator = MarketSimulator(symbol=symbol, data_manager=self.data_manager, rng=self.np_random, mode=mode, start_time=start_time,
                                                    end_time=end_time, config=self.config.simulation.market_config, logger=self.logger)
            self.execution_simulator = ExecutionSimulator(market_simulator=self.market_simulator, rng=self.np_random,
                                                          config=self.config.simulation.execution_config, logger=self.logger)
            self.portfolio_simulator = PortfolioSimulator(config=self.config.simulation.portfolio_config, logger=self.logger)
            self.feature_extractor = FeatureExtractor(symbol=symbol, config=self.config.simulation.feature_config, logger=self.logger)
            self.reward = RewardCalculator(config=self.config.env.reward, logger=self.logger)

            self.reset()

            self.logger.info(f"Environment initialized successfully for {symbol}")
        except Exception as e:
            self.logger.error(f"Error initializing environment: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def close(self):
        if self.market_simulator:
            self.market_simulator.close()

    def reset(self, seed=None, options=None):
        """Called by the agent's collect_rollouts method, also provides observation"""
        super().reset(seed=seed)

        random_start = options.get('random_start', self.config.env.random_reset)

        self.market_simulator.reset({'random_start': random_start, 'max_steps': self.config.env.max_steps})
        self.execution_simulator.reset()
        self.portfolio_simulator.reset()
        self.feature_extractor.reset()
        self.reward.reset()

        self.environment_stats = {}
        self.episode_stats = {}
        self.current_step = 0

        # Get initial state
        # Todo : Check if the 1st tick obs is ready after resetting everything.
        # Todo : If we start at tick 3000, this should be returning (3000 - lookback window)
        # Todo : We should handle starting at pre-market 4:00 AM, at that point market wont have any previous data
        self.obs = self._get_observation()

        print(f"Resetting environment should we keep this.: {self.obs}")
        return self.obs, self.episode_stats

    def step(self, action):
        """"""

        # Action is a tuple of (action_type, size)
        if isinstance(action, (list, np.ndarray)):
            action = tuple(map(int, action[:2]))

        self.current_step += 1

        # Process action and update
        self._process_action(action)

        # Get the new state
        observation = self._get_observation()

        # Check if we've reached the end of the data-critical fix!
        if observation is None or self.market_simulator.is_done():
            # Set the terminated flag when we've reached end of data
            self.logger.info("End of data reached, terminating episode")
            terminated = True
            truncated = False

            # Use last valid state or zeros as observation
            if self.obs is not None:
                observation = self.obs
            else:
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)

            reward = 0.0  # No reward at end of data

            # Set info with reason
            step_stats = self._get_step_stats()
            step_stats['end_reason'] = "END_OF_DATA"
            self.info = step_stats

            return observation, reward, terminated, truncated, self.info

        self.obs = observation

        reward = self._calculate_reward()

        portfolio_state = self.portfolio_simulator.get_portfolio_state()

        self.episode_stats['total_pnl'] = portfolio_state.get('total_pnl', 0.0)
        self.episode_stats['total_reward'] += reward

        terminated = self._is_terminated()
        truncated = self._is_truncated()

        self.environment_stats['step'] = self._get_step_stats()

        if terminated or truncated:
            self.environment_stats['episode'] = self.episode_stats

        return observation, reward, terminated, truncated, self.environment_stats

    # --- o ---
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
        if action_type == ActionType.HOLD:
            # No action needed
            action_name = "HOLD"

        elif action_type == ActionType.ENTER_LONG:
            if current_position <= 0:  # Only if not already long
                # Calculate target position
                target_position = position_size * self.config.env.max_position
                delta = target_position - current_position
                order_type = 'buy'
                action_name = f"ENTER_LONG({position_size})"
            else:
                # Already in position but tried to enter again
                action_name = f"ENTER_LONG_IGNORED(already in position)"

        elif action_type == ActionType.SCALE_IN:
            if current_position > 0:  # Only if already long
                # Calculate new target position
                new_target = min(self.config.env.max_position, current_position + (position_size * self.config.env.max_position))
                delta = new_target - current_position
                if delta > 0:
                    order_type = 'buy'
                    action_name = f"SCALE_IN({position_size})"
                else:
                    action_name = f"SCALE_IN_IGNORED(no additional position possible)"
            else:
                action_name = f"SCALE_IN_IGNORED(no existing position)"

        elif action_type == ActionType.SCALE_OUT:
            if current_position > 0:  # Only if long
                # Calculate reduction amount
                reduction = position_size * current_position
                delta = -reduction  # Negative delta for selling
                order_type = 'sell'
                action_name = f"SCALE_OUT({position_size})"
            else:
                action_name = f"SCALE_OUT_IGNORED(no position to reduce)"

        elif action_type == ActionType.EXIT:
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

                #   self.logger.info(f"Executed {action_name}: {order_type} {abs(delta):.4f} @ {current_price:.4f}")
                self.logger.info("Action executed: %s, Order type: %s, Size: %.4f, Price: %.4f", )

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
