"""Feature Extractor - Works with pre-calculated features from MarketSimulatorV3

This feature extractor is designed to work with MarketSimulatorV3 which pre-calculates
all features. It simply retrieves the pre-calculated features and adds portfolio features.
"""

import logging
from typing import Dict, Optional
import numpy as np

from config.schemas import ModelConfig
from simulators.market_simulator import MarketSimulator


class FeatureExtractor:
    """Feature extractor that retrieves pre-calculated features from MarketSimulatorV3"""

    def __init__(self,
                 symbol: str,
                 market_simulator: MarketSimulator,
                 config: ModelConfig,
                 logger: Optional[logging.Logger] = None):
        """Initialize the feature extractor
        
        Args:
            symbol: Trading symbol
            market_simulator: MarketSimulatorV3 instance with pre-calculated features
            config: Model configuration
            logger: Optional logger
        """
        self.symbol = symbol
        self.market_simulator = market_simulator
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Feature dimensions from config
        self.hf_seq_len = config.hf_seq_len
        self.hf_feat_dim = config.hf_feat_dim
        self.mf_seq_len = config.mf_seq_len
        self.mf_feat_dim = config.mf_feat_dim
        self.lf_seq_len = config.lf_seq_len
        self.lf_feat_dim = config.lf_feat_dim
        self.static_feat_dim = config.static_feat_dim
        self.portfolio_seq_len = config.portfolio_seq_len
        self.portfolio_feat_dim = config.portfolio_feat_dim
        
        # Portfolio state for feature extraction
        self.portfolio_state = {
            'position': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'cash_balance': 100000.0,  # Default starting cash
            'total_equity': 100000.0,
            'buying_power': 100000.0,
            'avg_entry_price': 0.0,
            'position_value': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        # Track portfolio history for sequence features
        self.portfolio_history = []
        self.max_history_length = 100  # Keep last 100 states

    def extract_features(self) -> Optional[Dict[str, np.ndarray]]:
        """Extract features for current market state
        
        Since MarketSimulatorV3 pre-calculates features, this method simply
        retrieves them and adds portfolio features.
        
        Returns:
            Dictionary with feature arrays or None if unavailable
        """
        try:
            # Get pre-calculated features from market simulator
            features = self.market_simulator.get_current_features()
            
            if features is None:
                self.logger.warning("No pre-calculated features available")
                return None
                
            # Add portfolio features
            features['portfolio'] = self._extract_portfolio_features()
            
            # Validate feature dimensions
            if not self._validate_features(features):
                self.logger.error("Feature validation failed")
                return None
            
            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def update_portfolio_state(self, new_state: Dict[str, float]):
        """Update portfolio state for feature extraction
        
        Args:
            new_state: Dictionary with portfolio state values
        """
        # Update current state
        self.portfolio_state.update(new_state)
        
        # Add to history
        self.portfolio_history.append(self.portfolio_state.copy())
        
        # Maintain history size
        if len(self.portfolio_history) > self.max_history_length:
            self.portfolio_history.pop(0)

    def _extract_portfolio_features(self) -> np.ndarray:
        """Extract portfolio features as a sequence
        
        Returns:
            Array of shape (portfolio_seq_len, portfolio_feat_dim)
        """
        try:
            features = np.zeros((self.portfolio_seq_len, self.portfolio_feat_dim), dtype=np.float32)
            
            # Get market data for normalization
            market_data = self.market_simulator.get_current_market_data()
            current_price = market_data['current_price'] if market_data else 1.0
            
            # Use portfolio history to fill sequence
            history_len = len(self.portfolio_history)
            
            for i in range(self.portfolio_seq_len):
                # Get state from history (most recent last)
                if i < history_len:
                    state = self.portfolio_history[-(history_len - i)]
                else:
                    state = self.portfolio_state
                    
                # Basic portfolio features
                features[i, 0] = float(state.get('position', 0.0)) / 1000.0  # Normalize by 1000 shares
                features[i, 1] = float(state.get('unrealized_pnl', 0.0)) / 10000.0  # Normalize by $10k
                features[i, 2] = float(state.get('realized_pnl', 0.0)) / 10000.0
                features[i, 3] = float(state.get('cash_balance', 0.0)) / 100000.0  # Normalize by $100k
                
                # Additional features if dimension allows
                if self.portfolio_feat_dim > 4:
                    features[i, 4] = float(state.get('total_equity', 0.0)) / 100000.0
                    
                if self.portfolio_feat_dim > 5:
                    features[i, 5] = float(state.get('buying_power', 0.0)) / 100000.0
                    
                if self.portfolio_feat_dim > 6:
                    # Position as percentage of equity
                    equity = state.get('total_equity', 100000.0)
                    position_value = state.get('position', 0.0) * current_price
                    features[i, 6] = position_value / max(1000, equity)
                    
                if self.portfolio_feat_dim > 7:
                    # Win rate
                    total_trades = state.get('total_trades', 0)
                    winning_trades = state.get('winning_trades', 0)
                    features[i, 7] = winning_trades / max(1, total_trades)
                    
                if self.portfolio_feat_dim > 8:
                    # Average entry vs current price (if in position)
                    avg_entry = state.get('avg_entry_price', 0.0)
                    if avg_entry > 0 and current_price > 0:
                        features[i, 8] = (current_price - avg_entry) / avg_entry
                    else:
                        features[i, 8] = 0.0
                        
                if self.portfolio_feat_dim > 9:
                    # Drawdown from peak equity
                    peak_equity = max(s.get('total_equity', 100000.0) for s in self.portfolio_history[-20:]) if self.portfolio_history else 100000.0
                    current_equity = state.get('total_equity', 100000.0)
                    features[i, 9] = (peak_equity - current_equity) / max(1000, peak_equity)
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Portfolio feature extraction failed: {e}")
            return np.zeros((self.portfolio_seq_len, self.portfolio_feat_dim), dtype=np.float32)

    def _validate_features(self, features: Dict[str, np.ndarray]) -> bool:
        """Validate feature dimensions
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            True if valid, False otherwise
        """
        expected_shapes = {
            'hf': (self.hf_seq_len, self.hf_feat_dim),
            'mf': (self.mf_seq_len, self.mf_feat_dim),
            'lf': (self.lf_seq_len, self.lf_feat_dim),
            'static': (self.static_feat_dim,),
            'portfolio': (self.portfolio_seq_len, self.portfolio_feat_dim)
        }
        
        for key, expected_shape in expected_shapes.items():
            if key not in features:
                self.logger.error(f"Missing feature key: {key}")
                return False
                
            actual_shape = features[key].shape
            if actual_shape != expected_shape:
                self.logger.error(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
                return False
                
        return True

    def reset(self):
        """Reset feature extractor state"""
        # Reset portfolio state
        self.portfolio_state = {
            'position': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'cash_balance': 100000.0,
            'total_equity': 100000.0,
            'buying_power': 100000.0,
            'avg_entry_price': 0.0,
            'position_value': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        # Clear history
        self.portfolio_history.clear()
        
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about feature dimensions and structure
        
        Returns:
            Dictionary with feature information
        """
        return {
            'hf': {
                'shape': (self.hf_seq_len, self.hf_feat_dim),
                'description': 'High-frequency features (1-second resolution)'
            },
            'mf': {
                'shape': (self.mf_seq_len, self.mf_feat_dim),
                'description': 'Medium-frequency features (1-minute bars)'
            },
            'lf': {
                'shape': (self.lf_seq_len, self.lf_feat_dim),
                'description': 'Low-frequency features (5-minute bars)'
            },
            'static': {
                'shape': (self.static_feat_dim,),
                'description': 'Static features (time encodings, market context)'
            },
            'portfolio': {
                'shape': (self.portfolio_seq_len, self.portfolio_feat_dim),
                'description': 'Portfolio state features'
            }
        }
        
    def get_portfolio_state(self) -> Dict[str, float]:
        """Get current portfolio state
        
        Returns:
            Copy of current portfolio state
        """
        return self.portfolio_state.copy()