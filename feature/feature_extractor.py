# feature/feature_extractor.py - Feature extraction using the new modular system

import logging
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime

from config.schemas import ModelConfig
from simulators.market_simulator import MarketSimulator
from feature.feature_manager import FeatureManager
from feature.contexts import MarketContext


class FeatureExtractor:
    """Feature extractor using the new modular feature system"""

    def __init__(self, symbol: str, market_simulator: MarketSimulator,
                 config: ModelConfig, logger: Optional[logging.Logger] = None):
        self.symbol = symbol
        self.market_simulator = market_simulator
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize feature manager
        self.feature_manager = FeatureManager(
            symbol=symbol,
            config=config,
            logger=self.logger
        )
        
        # Feature dimensions
        self.hf_seq_len = config.hf_seq_len
        self.hf_feat_dim = config.hf_feat_dim
        self.mf_seq_len = config.mf_seq_len
        self.mf_feat_dim = config.mf_feat_dim
        self.lf_seq_len = config.lf_seq_len
        self.lf_feat_dim = config.lf_feat_dim
        self.static_feat_dim = config.static_feat_dim
        self.portfolio_seq_len = config.portfolio_seq_len
        self.portfolio_feat_dim = config.portfolio_feat_dim
        
        # Cache for efficiency
        self._last_features = None
        self._last_timestamp = None

    def reset(self):
        """Reset feature extractor state"""
        self._last_features = None
        self._last_timestamp = None
        self.feature_manager.reset()

    def extract_features(self) -> Optional[Dict[str, np.ndarray]]:
        """Extract features for current market state"""
        try:
            current_market_state = self.market_simulator.get_current_market_state()
            if not current_market_state:
                return None

            current_timestamp = current_market_state.get('timestamp_utc')

            # Use cache if same timestamp
            if (self._last_features is not None and
                    current_timestamp == self._last_timestamp):
                return self._last_features

            # Create MarketContext from current state
            context = self._create_market_context(current_market_state)
            if context is None:
                return None

            # Extract features using the modular system
            features = self.feature_manager.extract_features(context)
            
            if features is None:
                return None
                
            # Add portfolio features if not included
            if 'portfolio' not in features:
                features['portfolio'] = self._extract_portfolio_features(current_market_state)

            # Cache results
            self._last_features = features
            self._last_timestamp = current_timestamp

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _create_market_context(self, market_state: Dict[str, Any]) -> Optional[MarketContext]:
        """Convert market simulator state to MarketContext"""
        try:
            # Get basic market data
            timestamp = market_state.get('timestamp_utc')
            if not timestamp:
                return None
                
            current_price = market_state.get('current_price', 0.0)
            
            # Get windows
            hf_window = market_state.get('hf_data_window', [])
            bars_1m = market_state.get('1m_bars_window', [])
            bars_5m = market_state.get('5m_bars_window', [])
            
            # Get previous day data
            prev_day_data = market_state.get('previous_day_data', {})
            prev_day_close = market_state.get('previous_day_close', 0.0)
            
            # Get session information
            session = market_state.get('market_session', 'UNKNOWN')
            
            # Get intraday stats
            intraday_high = market_state.get('intraday_high', current_price)
            intraday_low = market_state.get('intraday_low', current_price)
            
            # Create context
            context = MarketContext(
                timestamp=timestamp,
                current_price=current_price,
                hf_window=hf_window,
                mf_1m_window=bars_1m,
                lf_5m_window=bars_5m,
                prev_day_close=prev_day_close,
                prev_day_high=prev_day_data.get('high', 0.0),
                prev_day_low=prev_day_data.get('low', 0.0),
                session_high=intraday_high,
                session_low=intraday_low,
                session=session,
                market_cap=market_state.get('market_cap', 1e9),  # Default 1B if not provided
                session_volume=market_state.get('session_volume', 0.0),
                session_trades=market_state.get('session_trades', 0),
                session_vwap=market_state.get('session_vwap', current_price)
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to create market context: {e}")
            return None

    def _extract_portfolio_features(self, market_state: Dict[str, Any]) -> np.ndarray:
        """Extract portfolio features from market state"""
        try:
            # Portfolio features are a sequence but typically just current state repeated
            features = np.zeros((self.portfolio_seq_len, self.portfolio_feat_dim), dtype=np.float32)
            
            # Get portfolio state from market simulator if available
            portfolio_state = market_state.get('portfolio_state', {})
            
            # Fill all timesteps with current portfolio state
            for i in range(self.portfolio_seq_len):
                # Basic portfolio features
                features[i, 0] = float(portfolio_state.get('position', 0.0))
                features[i, 1] = float(portfolio_state.get('unrealized_pnl', 0.0))
                features[i, 2] = float(portfolio_state.get('realized_pnl', 0.0))
                features[i, 3] = float(portfolio_state.get('cash_balance', 0.0))
                
                # Additional features if dimension allows
                if self.portfolio_feat_dim > 4:
                    features[i, 4] = float(portfolio_state.get('total_equity', 0.0))
                if self.portfolio_feat_dim > 5:
                    features[i, 5] = float(portfolio_state.get('buying_power', 0.0))
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Portfolio feature extraction failed: {e}")
            return np.zeros((self.portfolio_seq_len, self.portfolio_feat_dim), dtype=np.float32)