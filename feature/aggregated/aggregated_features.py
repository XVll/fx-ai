"""Aggregated features that efficiently summarize entire sequence windows"""
import numpy as np
from typing import Dict, Any, List
from ..feature_base import BaseFeature


class HFMomentumSummaryFeature(BaseFeature):
    """Comprehensive momentum analysis using entire HF window"""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Aggregate momentum metrics from entire HF window"""
        try:
            hf_window = market_data.get('hf_data_window', [])
            if len(hf_window) < 3:
                return 0.0
            
            # Extract prices from the window
            prices = []
            for entry in hf_window:
                if entry and entry.get('1s_bar'):
                    prices.append(entry['1s_bar'].get('close', 0.0))
            
            if len(prices) < 3:
                return 0.0
            
            prices = np.array(prices)
            
            # Calculate various momentum metrics
            returns = np.diff(prices) / prices[:-1]
            
            # Exponentially weighted momentum (recent changes matter more)
            weights = np.exp(np.linspace(-1, 0, len(returns)))
            weights = weights / weights.sum()
            weighted_momentum = np.sum(returns * weights)
            
            # Momentum acceleration (change in momentum)
            if len(returns) >= 3:
                momentum_window = 3
                early_momentum = np.mean(returns[:momentum_window])
                late_momentum = np.mean(returns[-momentum_window:])
                momentum_acceleration = late_momentum - early_momentum
            else:
                momentum_acceleration = 0.0
            
            # Combine metrics with momentum acceleration having higher weight
            summary = weighted_momentum * 0.4 + momentum_acceleration * 0.6
            
            return float(np.clip(summary, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'hf_data', 'lookback': 60, 'fields': ['1s_bar']}


class HFVolumeDynamicsFeature(BaseFeature):
    """Volume pattern analysis using entire HF window"""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Analyze volume patterns across HF window"""
        try:
            hf_window = market_data.get('hf_data_window', [])
            if len(hf_window) < 5:
                return 0.0
            
            # Extract volumes from the window
            volumes = []
            for entry in hf_window:
                if entry and entry.get('1s_bar'):
                    volumes.append(entry['1s_bar'].get('volume', 0.0))
            
            if len(volumes) < 5:
                return 0.0
            
            volumes = np.array(volumes)
            
            # Volume trend (increasing vs decreasing)  
            try:
                if len(volumes) > 1:
                    volumes_std = np.std(volumes)
                    indices = np.arange(len(volumes))
                    indices_std = np.std(indices)
                    
                    # Only calculate correlation if both arrays have meaningful variation
                    if volumes_std > 1e-8 and indices_std > 1e-8:
                        volume_trend = np.corrcoef(indices, volumes)[0, 1]
                        if np.isnan(volume_trend) or np.isinf(volume_trend):
                            volume_trend = 0.0
                    else:
                        volume_trend = 0.0  # No variation, no trend
                else:
                    volume_trend = 0.0
            except Exception as e:
                volume_trend = 0.0
            
            # Volume acceleration (recent vs earlier)
            mid_point = len(volumes) // 2
            early_avg = np.mean(volumes[:mid_point])
            late_avg = np.mean(volumes[mid_point:])
            volume_acceleration = (late_avg - early_avg) / (early_avg + 1e-8)
            
            # Volume volatility (consistency vs spikes)
            volume_cv = np.std(volumes) / (np.mean(volumes) + 1e-8)
            
            # Combine metrics
            dynamics = volume_trend * 0.4 + volume_acceleration * 0.4 - volume_cv * 0.2
            
            return float(np.clip(dynamics, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'hf_data', 'lookback': 60, 'fields': ['1s_bar']}


class HFMicrostructureQualityFeature(BaseFeature):
    """Microstructure quality using spread and quote data"""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Assess market quality from entire HF window"""
        try:
            hf_window = market_data.get('hf_data_window', [])
            if len(hf_window) < 3:
                return 0.0
            
            spreads = []
            quote_counts = []
            
            for entry in hf_window:
                if entry:
                    quotes = entry.get('quotes', [])
                    quote_counts.append(len(quotes))
                    
                    if quotes:
                        # Calculate spread from latest quote
                        latest_quote = quotes[-1]
                        bid = latest_quote.get('bid_px', 0.0)
                        ask = latest_quote.get('ask_px', 0.0)
                        if bid > 0 and ask > 0:
                            spread = (ask - bid) / ((ask + bid) / 2)
                            spreads.append(spread)
            
            if not spreads:
                return 0.0
            
            spreads = np.array(spreads)
            quote_counts = np.array(quote_counts)
            
            # Average spread quality (lower is better)
            avg_spread = np.mean(spreads)
            
            # Spread stability (less volatile is better)
            spread_stability = 1.0 / (1.0 + np.std(spreads))
            
            # Quote activity (more quotes generally better)
            quote_activity = np.mean(quote_counts) / 10.0  # Normalize
            
            # Combined quality score
            quality = spread_stability * 0.5 + quote_activity * 0.3 - avg_spread * 0.2
            
            return float(np.clip(quality, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'hf_data', 'lookback': 60, 'fields': ['quotes']}


class MFTrendConsistencyFeature(BaseFeature):
    """Trend consistency analysis using entire MF window"""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Measure trend consistency across MF timeframe"""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            if len(bars_1m) < 5:
                return 0.0
            
            # Extract closes
            closes = [bar.get('close', 0.0) for bar in bars_1m]
            closes = np.array(closes)
            
            if len(closes) < 5:
                return 0.0
            
            # Calculate returns
            returns = np.diff(closes) / closes[:-1]
            
            # Trend direction consistency
            positive_returns = returns > 0
            negative_returns = returns < 0
            
            # Consecutive move consistency
            consistency_score = 0.0
            if len(returns) > 0:
                pos_ratio = np.sum(positive_returns) / len(returns)
                neg_ratio = np.sum(negative_returns) / len(returns)
                
                # High consistency if mostly positive or mostly negative
                consistency_score = max(pos_ratio, neg_ratio) * 2 - 1
            
            # Trend magnitude consistency (avoid choppy markets)
            return_magnitudes = np.abs(returns)
            magnitude_consistency = 1.0 / (1.0 + np.std(return_magnitudes))
            
            # Combined consistency
            overall_consistency = consistency_score * 0.7 + magnitude_consistency * 0.3
            
            return float(np.clip(overall_consistency, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 30, 'fields': ['close']}


class MFVolumePriceDivergenceFeature(BaseFeature):
    """Volume-price divergence analysis using MF window"""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Detect volume-price divergences"""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            if len(bars_1m) < 6:
                return 0.0
            
            # Extract price and volume data
            closes = [bar.get('close', 0.0) for bar in bars_1m]
            volumes = [bar.get('volume', 0.0) for bar in bars_1m]
            
            closes = np.array(closes)
            volumes = np.array(volumes)
            
            if len(closes) < 6:
                return 0.0
            
            # Calculate trends for recent vs earlier periods
            mid_point = len(closes) // 2
            
            # Price trend
            early_price = np.mean(closes[:mid_point])
            late_price = np.mean(closes[mid_point:])
            price_direction = 1.0 if late_price > early_price else -1.0
            
            # Volume trend  
            early_volume = np.mean(volumes[:mid_point])
            late_volume = np.mean(volumes[mid_point:])
            volume_direction = 1.0 if late_volume > early_volume else -1.0
            
            # Calculate correlation between price and volume
            try:
                if len(closes) > 1 and len(closes) == len(volumes):
                    closes_std = np.std(closes)
                    volumes_std = np.std(volumes)
                    
                    # Only calculate correlation if both arrays have meaningful variation
                    if closes_std > 1e-8 and volumes_std > 1e-8:
                        price_volume_corr = np.corrcoef(closes, volumes)[0, 1]
                        if np.isnan(price_volume_corr) or np.isinf(price_volume_corr):
                            price_volume_corr = 0.0
                    else:
                        price_volume_corr = 0.0  # No variation, no correlation
                else:
                    price_volume_corr = 0.0
            except Exception as e:
                price_volume_corr = 0.0
            
            # Divergence score (positive when volume confirms price, negative when divergent)
            confirmation_score = price_direction * volume_direction
            
            # Weight by correlation strength
            divergence_signal = confirmation_score * abs(price_volume_corr)
            
            return float(np.clip(divergence_signal, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 30, 'fields': ['close', 'volume']}


class MFMomentumPersistenceFeature(BaseFeature):
    """Momentum persistence analysis across MF window"""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Measure how well momentum persists"""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            if len(bars_1m) < 8:
                return 0.0
            
            # Extract closes
            closes = [bar.get('close', 0.0) for bar in bars_1m]
            closes = np.array(closes)
            
            if len(closes) < 8:
                return 0.0
            
            # Calculate rolling momentum (3-bar windows)
            window_size = 3
            momentum_values = []
            
            for i in range(window_size, len(closes)):
                recent = closes[i-window_size+1:i+1]
                momentum = (recent[-1] - recent[0]) / recent[0]
                momentum_values.append(momentum)
            
            if len(momentum_values) < 3:
                return 0.0
            
            momentum_values = np.array(momentum_values)
            
            # Measure momentum persistence
            # Positive if momentum maintains direction, negative if it reverses
            direction_changes = 0
            for i in range(1, len(momentum_values)):
                if (momentum_values[i] > 0) != (momentum_values[i-1] > 0):
                    direction_changes += 1
            
            # Persistence score (fewer direction changes = higher persistence)
            max_changes = len(momentum_values) - 1
            if max_changes > 0:
                persistence = 1.0 - (direction_changes / max_changes)
            else:
                persistence = 0.0
            
            # Weight by magnitude of momentum
            avg_momentum_magnitude = np.mean(np.abs(momentum_values))
            
            # Final score
            persistence_score = persistence * min(avg_momentum_magnitude * 10, 1.0)
            
            return float(np.clip(persistence_score, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 30, 'fields': ['close']}