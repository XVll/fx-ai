"""Portfolio-related features"""
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime, timezone
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry


@feature_registry.register("portfolio_position_size", category="portfolio")
class PortfolioPositionSizeFeature(BaseFeature):
    """Current position size as percentage of portfolio"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="portfolio_position_size",
            normalize=False  # Already normalized to [0, 1]
        ))
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate position size as percentage"""
        portfolio_state = market_data.get('portfolio_state', {})
        
        position = float(portfolio_state.get('position', 0.0))
        total_equity = float(portfolio_state.get('total_equity', 0.0))
        position_value = float(portfolio_state.get('current_position_value', 0.0))
        
        # Handle no position
        if position == 0 or total_equity <= 0:
            return 0.0
        
        # Use absolute position value for size
        if position_value != 0:
            position_pct = abs(position_value) / total_equity
        else:
            # Fallback: estimate from position and price
            current_price = market_data.get('current_price', 0.0)
            if current_price > 0:
                position_pct = abs(position * current_price) / total_equity
            else:
                return 0.0
        
        # Clamp to [0, 1] (100% max)
        return min(1.0, position_pct)
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "fields": ["portfolio_state"]
        }


@feature_registry.register("portfolio_average_price", category="portfolio")
class PortfolioAveragePriceFeature(BaseFeature):
    """Average entry price distance from current price"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="portfolio_average_price",
            normalize=True
        ))
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate normalized distance from average entry price"""
        current_price = float(market_data.get('current_price', 0.0))
        portfolio_state = market_data.get('portfolio_state', {})
        
        position = float(portfolio_state.get('position', 0.0))
        avg_price = float(portfolio_state.get('average_entry_price', 0.0))
        
        # No position or invalid prices
        if position == 0 or avg_price <= 0 or current_price <= 0:
            return 0.0
        
        # Calculate percentage distance
        distance_pct = (current_price - avg_price) / avg_price
        
        # For long positions, positive = profit, negative = loss
        # For short positions, negative = profit, positive = loss
        if position < 0:
            distance_pct = -distance_pct
        
        return distance_pct
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {
            "min": -0.2,  # -20% loss
            "max": 0.2    # +20% profit
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "fields": ["current_price", "portfolio_state"]
        }


@feature_registry.register("portfolio_unrealized_pnl", category="portfolio")
class PortfolioUnrealizedPnLFeature(BaseFeature):
    """Unrealized P&L as percentage of equity"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="portfolio_unrealized_pnl",
            normalize=True
        ))
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate unrealized P&L percentage"""
        portfolio_state = market_data.get('portfolio_state', {})
        
        unrealized_pnl = float(portfolio_state.get('unrealized_pnl', 0.0))
        total_equity = float(portfolio_state.get('total_equity', 0.0))
        
        if total_equity <= 0:
            return 0.0
        
        # P&L as percentage of equity
        pnl_pct = unrealized_pnl / total_equity
        
        return pnl_pct
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {
            "min": -0.1,  # -10% loss
            "max": 0.1    # +10% profit
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "fields": ["portfolio_state"]
        }


@feature_registry.register("portfolio_time_in_position", category="portfolio")
class PortfolioTimeInPositionFeature(BaseFeature):
    """Time in position normalized to trading session"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="portfolio_time_in_position",
            normalize=False  # Already normalized
        ))
        self.max_holding_seconds = 3600  # 1 hour max for normalization
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate normalized time in position"""
        timestamp = market_data.get('timestamp')
        portfolio_state = market_data.get('portfolio_state', {})
        
        position = float(portfolio_state.get('position', 0.0))
        
        # No position
        if position == 0:
            return 0.0
        
        # Get time in position
        time_in_position = portfolio_state.get('time_in_position_seconds')
        if time_in_position is not None:
            seconds = float(time_in_position)
        else:
            # Try to calculate from entry time
            entry_time = portfolio_state.get('position_entry_time')
            if entry_time and timestamp:
                # Convert to datetime if needed
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                delta = timestamp - entry_time
                seconds = delta.total_seconds()
            else:
                return 0.0
        
        # Normalize to [0, 1] where 1 = max holding time
        normalized = min(seconds / self.max_holding_seconds, 1.0)
        
        return normalized
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "fields": ["timestamp", "portfolio_state"]
        }


@feature_registry.register("portfolio_max_adverse_excursion", category="portfolio")
class PortfolioMaxAdverseExcursionFeature(BaseFeature):
    """Maximum adverse excursion during trade"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__(config or FeatureConfig(
            name="portfolio_max_adverse_excursion",
            normalize=False  # Already normalized
        ))
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate normalized MAE"""
        portfolio_state = market_data.get('portfolio_state', {})
        
        position = float(portfolio_state.get('position', 0.0))
        
        # No position
        if position == 0:
            return 0.0
        
        # Get MAE
        mae = portfolio_state.get('max_adverse_excursion', 0.0)
        
        if mae is None:
            return 0.0
        
        # MAE is typically negative (drawdown)
        mae_pct = abs(float(mae))
        
        # Normalize to [0, 1] where 0 = no drawdown, 1 = 10% drawdown
        normalized_mae = min(mae_pct / 0.1, 1.0)
        
        return normalized_mae
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {}  # Already [0, 1]
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            "fields": ["portfolio_state"]
        }