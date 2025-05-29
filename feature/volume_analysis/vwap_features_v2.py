"""Sequence-aware VWAP features for professional momentum trading."""

import numpy as np
from typing import Dict, Any, List
from ..feature_base import BaseFeature


class VWAPInteractionDynamicsFeature(BaseFeature):
    """Analyzes how price interacts with VWAP over time sequences.
    
    Professional traders use VWAP as dynamic support/resistance.
    This feature captures:
    - Support/resistance bounces vs breaks
    - Duration of price above/below VWAP  
    - Multiple touch points and their outcomes
    """
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate VWAP interaction dynamics."""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            if len(bars_1m) < 10:
                return 0.0
            
            # Extract prices and VWAPs
            prices = []
            vwaps = []
            
            for bar in bars_1m[-20:]:  # Last 20 minutes
                close = bar.get('close', 0)
                vwap = bar.get('vwap', close)
                if close > 0 and vwap > 0:
                    prices.append(close)
                    vwaps.append(vwap)
            
            if len(prices) < 10:
                return 0.0
            
            prices = np.array(prices)
            vwaps = np.array(vwaps)
            
            # Calculate relative positions (above/below VWAP)
            relative_positions = (prices - vwaps) / vwaps
            above_vwap = relative_positions > 0
            
            # Measure support/resistance strength
            # Count touches within 0.2% of VWAP
            touches = np.abs(relative_positions) < 0.002
            num_touches = np.sum(touches)
            
            # Measure persistence (how long price stays on one side)
            direction_changes = 0
            for i in range(1, len(above_vwap)):
                if above_vwap[i] != above_vwap[i-1]:
                    direction_changes += 1
            
            persistence_score = 1.0 - (direction_changes / max(1, len(above_vwap) - 1))
            
            # Recent trend relative to VWAP
            recent_trend = np.mean(relative_positions[-5:]) - np.mean(relative_positions[:5])
            
            # Combine metrics
            # High touches + high persistence + positive trend = strong support/resistance
            interaction_strength = (
                min(num_touches / 5.0, 1.0) * 0.3 +  # Touch frequency
                persistence_score * 0.4 +             # Persistence 
                np.clip(recent_trend * 10, -1, 1) * 0.3  # Trend direction
            )
            
            return float(np.clip(interaction_strength, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 20, 'fields': ['close', 'vwap']}


class VWAPBreakoutQualityFeature(BaseFeature):
    """Measures quality of VWAP breakouts for momentum trading.
    
    Professional momentum traders distinguish between:
    - Failed breaks (weak momentum)
    - Sustained breaks (strong momentum)
    - Volume-confirmed breaks (highest quality)
    """
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate VWAP breakout quality."""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            if len(bars_1m) < 15:
                return 0.0
            
            # Extract data from recent window
            prices = []
            vwaps = []
            volumes = []
            
            for bar in bars_1m[-15:]:  # Last 15 minutes
                close = bar.get('close', 0)
                vwap = bar.get('vwap', close)
                volume = bar.get('volume', 0)
                
                if close > 0 and vwap > 0:
                    prices.append(close)
                    vwaps.append(vwap)
                    volumes.append(volume)
            
            if len(prices) < 10:
                return 0.0
            
            prices = np.array(prices)
            vwaps = np.array(vwaps)
            volumes = np.array(volumes)
            
            # Find recent breakout attempts (price crossing VWAP)
            relative_pos = (prices - vwaps) / vwaps
            above_vwap = relative_pos > 0.001  # Small threshold to avoid noise
            
            # Look for recent breakout (transition from below to above or vice versa)
            breakout_detected = False
            breakout_direction = 0
            breakout_start_idx = -1
            
            for i in range(5, len(above_vwap)):  # Start from index 5 to have history
                if above_vwap[i] != above_vwap[i-1]:
                    # Direction change detected
                    # Check if it's sustained (at least 3 bars)
                    if i + 2 < len(above_vwap):
                        direction_sustained = all(above_vwap[i:i+3] == above_vwap[i])
                        if direction_sustained:
                            breakout_detected = True
                            breakout_direction = 1 if above_vwap[i] else -1
                            breakout_start_idx = i
                            break
            
            if not breakout_detected:
                return 0.0
            
            # Measure breakout quality factors
            # 1. Magnitude of breakout
            breakout_magnitude = np.mean(np.abs(relative_pos[breakout_start_idx:]))
            
            # 2. Volume confirmation
            pre_breakout_volume = np.mean(volumes[:breakout_start_idx])
            breakout_volume = np.mean(volumes[breakout_start_idx:])
            volume_confirmation = breakout_volume / (pre_breakout_volume + 1e-8)
            
            # 3. Persistence (how well breakout is maintained)
            post_breakout_positions = relative_pos[breakout_start_idx:]
            if breakout_direction > 0:
                persistence = np.mean(post_breakout_positions > 0)
            else:
                persistence = np.mean(post_breakout_positions < 0)
            
            # Combine quality factors
            quality_score = (
                min(breakout_magnitude * 20, 1.0) * 0.3 +    # Magnitude (capped)
                min(volume_confirmation / 2.0, 1.0) * 0.4 +  # Volume confirmation
                persistence * 0.3                            # Persistence
            )
            
            # Apply direction
            final_score = quality_score * breakout_direction
            
            return float(np.clip(final_score, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 15, 'fields': ['close', 'vwap', 'volume']}


class VWAPMeanReversionTendencyFeature(BaseFeature):
    """Measures how strongly price tends to revert to VWAP.
    
    High mean reversion = choppy, range-bound conditions
    Low mean reversion = trending conditions suitable for momentum
    """
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate VWAP mean reversion tendency."""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            if len(bars_1m) < 12:
                return 0.5  # Neutral
            
            # Extract data
            prices = []
            vwaps = []
            
            for bar in bars_1m[-12:]:  # Last 12 minutes
                close = bar.get('close', 0)
                vwap = bar.get('vwap', close)
                if close > 0 and vwap > 0:
                    prices.append(close)
                    vwaps.append(vwap)
            
            if len(prices) < 8:
                return 0.5
            
            prices = np.array(prices)
            vwaps = np.array(vwaps)
            
            # Calculate distance from VWAP over time
            distances = np.abs(prices - vwaps) / vwaps
            
            # Look for mean reversion patterns
            # Count instances where price moves away then back to VWAP
            reversion_events = 0
            for i in range(2, len(distances) - 1):
                # Check if distance increased then decreased
                if distances[i-1] < distances[i] and distances[i] > distances[i+1]:
                    reversion_events += 1
            
            # Calculate autocorrelation of distances (persistence vs reversion)
            if len(distances) > 4:
                # Simple autocorrelation at lag 1
                distances_normalized = (distances - np.mean(distances)) / (np.std(distances) + 1e-8)
                autocorr = np.corrcoef(distances_normalized[:-1], distances_normalized[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0.0
            else:
                autocorr = 0.0
            
            # High autocorrelation = trending (low reversion)
            # Low autocorrelation = mean reverting
            # Negative autocorrelation = strong mean reversion
            
            # Convert to reversion tendency score
            # High positive = strong mean reversion (bad for momentum)
            # High negative = strong trending (good for momentum)
            reversion_tendency = -autocorr  # Flip sign so positive = reversion
            
            # Add reversion event frequency
            event_frequency = reversion_events / max(1, len(distances) - 3)
            
            # Combine metrics
            final_tendency = reversion_tendency * 0.7 + event_frequency * 0.3
            
            return float(np.clip(final_tendency, -1.0, 1.0))
            
        except Exception:
            return 0.5  # Neutral mean reversion
    
    def get_default_value(self) -> float:
        return 0.5
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 12, 'fields': ['close', 'vwap']}