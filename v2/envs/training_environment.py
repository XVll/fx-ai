import gymnasium as gym
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime
import pandas as pd

# TODO: During full porting, import actual trading environment
# from envs.trading_environment_original import TradingEnvironment as OldTradingEnvironment


class TrainingEnvironment(gym.Env):
    """
    V2 TrainingEnvironment - Bridge to old trading environment
    
    Responsibilities:
    - Provide clean interface to PPOTrainer via get_current_state()
    - Handle session setup and reset point management
    - Delegate actual trading logic to wrapped environment
    - Bridge between v2 architecture and old implementation
    
    NOTE: Currently wraps/stubs the old TradingEnvironment.
    Full porting will happen in a separate phase.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, data_manager=None):
        """Initialize TrainingEnvironment with configuration."""
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.TrainingEnvironment")
        self.config = config or {}
        self.data_manager = data_manager
        
        # Current state tracking
        self._current_state: Optional[Dict[str, np.ndarray]] = None
        self._current_symbol: Optional[str] = None
        self._current_date: Optional[str] = None
        self._current_reset_point: Optional[int] = None
        
        # TODO: During full porting, initialize actual TradingEnvironment
        # self._wrapped_env = OldTradingEnvironment(config, data_manager)
        self._wrapped_env = None
        
        # Stub action/observation spaces (will be properly defined during porting)
        self.action_space = gym.spaces.Discrete(12)  # Placeholder
        self.observation_space = gym.spaces.Dict({})  # Placeholder
        
        self.logger.info("🏗️ V2 TrainingEnvironment initialized (bridging to old implementation)")
    
    def get_current_state(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get current environment state for PPOTrainer.
        
        This is the main interface method that PPOTrainer calls.
        
        Returns:
            Current environment state as dict of numpy arrays, or None if not ready
        """
        if self._current_state is None:
            self.logger.warning("No current state available - environment may not be properly setup")
            return None
        
        self.logger.debug(f"Returning current state for {self._current_symbol} {self._current_date}")
        return self._current_state.copy()
    
    def setup_session(self, symbol: str, date: Union[str, datetime]) -> None:
        """
        Setup a new trading session for a specific symbol and date.
        Ported from old TradingEnvironment.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            date: Trading date (string or datetime)
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("A valid symbol (string) must be provided.")

        self._current_symbol = symbol

        # Parse date
        if isinstance(date, str):
            self._current_date = pd.Timestamp(date).to_pydatetime()
        else:
            self._current_date = date

        self.logger.info(
            f"🎯 Setting up session: {self._current_symbol} on {self._safe_date_format(self._current_date)}"
        )

        # TODO: During full porting, delegate to wrapped environment
        # self._wrapped_env.setup_session(symbol, date)
        
        # For now, create dummy state to make system functional
        self._current_state = self._create_dummy_state()
        
        self.logger.debug(f"✅ Session setup complete: {symbol} {self._safe_date_format(self._current_date)}")
    
    def reset_at_point(self, reset_point_idx: int = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset to a specific reset point within the loaded day.
        This is the main reset method for momentum-based training.
        Ported from old TradingEnvironment.
        
        Args:
            reset_point_idx: Index of reset point to start from
            
        Returns:
            Tuple of (initial_state, info)
        """
        if reset_point_idx is None:
            reset_point_idx = 0  # Default to first reset point
            
        self.logger.info(f"🔄 Resetting at reset point {reset_point_idx}")
        
        self._current_reset_point = reset_point_idx
        
        # TODO: During full porting, delegate to wrapped environment
        # return self._wrapped_env.reset_at_point(reset_point_idx)
        
        # For now, return dummy state to make system functional
        initial_state = self._create_dummy_state()
        self._current_state = initial_state
        
        info = {
            'reset_point_index': reset_point_idx,
            'symbol': self._current_symbol,
            'date': self._safe_date_format(self._current_date),
            'reset_type': 'point_reset'
        }
        
        self.logger.debug(f"✅ Reset complete at point {reset_point_idx}")
        return initial_state, info
    
    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take environment step with given action.
        
        Args:
            action: Action to take (in environment format, already converted by PPOTrainer)
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # TODO: During full porting, delegate to wrapped environment
        # return self._wrapped_env.step(action)
        
        # For now, return dummy data to make system functional
        next_state = self._create_dummy_state()
        self._current_state = next_state
        
        reward = np.random.randn() * 0.1  # Small random reward for testing
        terminated = np.random.random() < 0.01  # 1% chance of episode end
        truncated = False
        info = {
            'step': 'stub_implementation',
            'action_taken': action
        }
        
        return next_state, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Standard gym environment reset.
        
        Returns:
            Tuple of (initial_state, info)
        """
        self.logger.info("🔄 Standard environment reset")
        
        # TODO: During full porting, delegate to wrapped environment
        # return self._wrapped_env.reset(**kwargs)
        
        initial_state = self._create_dummy_state()
        self._current_state = initial_state
        
        info = {
            'reset_type': 'standard',
            'symbol': self._current_symbol,
            'date': self._safe_date_format(self._current_date)
        }
        return initial_state, info
    
    def _create_dummy_state(self) -> Dict[str, np.ndarray]:
        """Create dummy state for testing purposes."""
        # TODO: Remove this during full porting - replace with real state extraction
        
        return {
            'hf': np.random.randn(60, 10).astype(np.float32),    # High-frequency features
            'mf': np.random.randn(12, 8).astype(np.float32),     # Medium-frequency features  
            'lf': np.random.randn(1, 6).astype(np.float32),      # Low-frequency features
            'portfolio': np.random.randn(1, 4).astype(np.float32), # Portfolio features
            'static': np.random.randn(3).astype(np.float32),     # Static features
        }
    
    def _safe_date_format(self, date_obj) -> str:
        """Safely format date object to string."""
        if date_obj is None:
            return "None"
        elif isinstance(date_obj, str):
            return date_obj
        elif hasattr(date_obj, 'strftime'):
            return date_obj.strftime('%Y-%m-%d')
        else:
            return str(date_obj)
    
    def close(self) -> None:
        """Clean up environment resources."""
        self.logger.info("🔚 TrainingEnvironment closed")
        
        # TODO: During full porting, close wrapped environment
        # if self._wrapped_env:
        #     self._wrapped_env.close()
        
        self._current_state = None