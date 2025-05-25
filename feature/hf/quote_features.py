"""High-frequency quote-based features"""
import numpy as np
from typing import Dict, Any, List
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry


@feature_registry.register("spread_compression", category="hf")
class SpreadCompressionFeature(BaseFeature):
    """Bid-ask spread compression over 1 second"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="spread_compression", normalize=False)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate spread compression over 1 second"""
        hf_window = market_data.get('hf_data_window', [])
        
        if len(hf_window) < 2:
            return 0.0
        
        # Get quotes from last two windows
        prev_window = hf_window[-2]
        curr_window = hf_window[-1]
        
        # Check if both windows have quotes
        if 'quotes' not in prev_window or 'quotes' not in curr_window:
            return 0.0
        
        prev_quotes = prev_window['quotes']
        curr_quotes = curr_window['quotes']
        
        if not prev_quotes or not curr_quotes:
            return 0.0
        
        # Get last quote from each window
        prev_quote = prev_quotes[-1]
        curr_quote = curr_quotes[-1]
        
        # Check required fields
        required_fields = ['bid_price', 'ask_price']
        for quote in [prev_quote, curr_quote]:
            for field in required_fields:
                if field not in quote or quote[field] is None:
                    return 0.0
        
        # Calculate spreads
        prev_spread = prev_quote['ask_price'] - prev_quote['bid_price']
        curr_spread = curr_quote['ask_price'] - curr_quote['bid_price']
        
        # Compression = old_spread - new_spread (positive = compression)
        compression = prev_spread - curr_spread
        
        return compression
    
    def get_default_value(self) -> float:
        """Default to no compression"""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Raw values for spread compression"""
        return {}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "quotes",
            "lookback": 2,
            "fields": ["hf_data_window"]
        }


@feature_registry.register("quote_velocity", category="hf")
class QuoteVelocityFeature(BaseFeature):
    """Mid-price velocity from quotes"""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="quote_velocity", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate mid-price velocity from quotes"""
        hf_window = market_data.get('hf_data_window', [])
        
        if len(hf_window) < 2:
            return 0.0
        
        # Get quotes from last two windows
        prev_window = hf_window[-2]
        curr_window = hf_window[-1]
        
        # Check if both windows have quotes
        if 'quotes' not in prev_window or 'quotes' not in curr_window:
            return 0.0
        
        prev_quotes = prev_window['quotes']
        curr_quotes = curr_window['quotes']
        
        if not prev_quotes or not curr_quotes:
            return 0.0
        
        # Get last quote from each window
        prev_quote = prev_quotes[-1]
        curr_quote = curr_quotes[-1]
        
        # Check required fields
        required_fields = ['bid_price', 'ask_price']
        for quote in [prev_quote, curr_quote]:
            for field in required_fields:
                if field not in quote or quote[field] is None:
                    return 0.0
        
        # Calculate mid-prices
        prev_mid = (prev_quote['bid_price'] + prev_quote['ask_price']) / 2
        curr_mid = (curr_quote['bid_price'] + curr_quote['ask_price']) / 2
        
        # Avoid division by zero
        if prev_mid == 0:
            return 0.0
        
        # Calculate velocity as percentage change
        velocity = (curr_mid - prev_mid) / prev_mid
        
        return velocity
    
    def get_default_value(self) -> float:
        """Default to no movement"""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Normalize to [-1, 1] for ±10% per second max"""
        return {
            "min": -0.1,  # -10% per second
            "max": 0.1    # +10% per second
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "data_type": "quotes",
            "lookback": 2,
            "fields": ["hf_data_window"]
        }