"""Sequence-aware EMA features for professional momentum trading."""

import numpy as np
from typing import Dict, Any, List
from ..feature_base import BaseFeature


class EMAInteractionPatternFeature(BaseFeature):
    """Analyzes price interaction patterns with EMAs over time.
    
    Professional traders look for:
    - EMA respect vs rejection patterns
    - Multiple timeframe EMA alignment
    - Breakout vs pullback behavior around EMAs
    """
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate EMA interaction patterns."""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            if len(bars_1m) < 20:
                return 0.0
            
            # Calculate EMAs for recent window
            closes = [bar.get('close', 0) for bar in bars_1m[-20:]]
            closes = [c for c in closes if c > 0]
            
            if len(closes) < 15:
                return 0.0
            
            closes = np.array(closes)
            
            # Calculate EMA9 and EMA20
            ema9 = self._calculate_ema(closes, 9)
            ema20 = self._calculate_ema(closes, 20)
            
            if len(ema9) < 10 or len(ema20) < 10:
                return 0.0
            
            # Analyze price-EMA interactions in recent period
            recent_closes = closes[-10:]
            recent_ema9 = ema9[-10:]
            recent_ema20 = ema20[-10:]
            
            # 1. EMA alignment (trending vs choppy)
            ema9_above_ema20 = recent_ema9 > recent_ema20
            alignment_consistency = np.sum(ema9_above_ema20 == ema9_above_ema20[0]) / len(ema9_above_ema20)
            alignment_direction = 1 if ema9_above_ema20[0] else -1
            
            # 2. Price position relative to EMAs
            price_above_ema9 = recent_closes > recent_ema9
            price_above_ema20 = recent_closes > recent_ema20
            
            # 3. EMA respect/rejection analysis
            # Count bounces off EMAs (price touches then moves away)
            ema9_touches = self._count_ema_touches(recent_closes, recent_ema9)
            ema20_touches = self._count_ema_touches(recent_closes, recent_ema20)
            
            # 4. Trend strength (EMAs sloping in same direction)
            ema9_slope = (recent_ema9[-1] - recent_ema9[0]) / recent_ema9[0]
            ema20_slope = (recent_ema20[-1] - recent_ema20[0]) / recent_ema20[0]
            slope_alignment = 1 if ema9_slope * ema20_slope > 0 else 0
            
            # Combine pattern metrics
            pattern_strength = (
                alignment_consistency * 0.3 +           # EMA alignment consistency
                slope_alignment * 0.3 +                 # Slope alignment  
                min((ema9_touches + ema20_touches) / 4, 1.0) * 0.2 +  # EMA respect
                np.clip(abs(ema9_slope) * 50, 0, 1) * 0.2  # Trend strength
            )
            
            # Apply directional bias
            final_pattern = pattern_strength * alignment_direction
            
            return float(np.clip(final_pattern, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA for given period."""
        if len(prices) < period:
            return np.array([])
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _count_ema_touches(self, prices: np.ndarray, ema: np.ndarray, threshold: float = 0.003) -> int:
        """Count how many times price touches EMA (within threshold)."""
        distances = np.abs(prices - ema) / ema
        touches = np.sum(distances < threshold)
        return int(touches)
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 20, 'fields': ['close']}


class EMACrossoverDynamicsFeature(BaseFeature):
    """Analyzes EMA crossover behavior and momentum quality.
    
    Professional momentum traders use EMA crossovers for:
    - Trend change confirmation
    - Entry/exit timing
    - Momentum strength assessment
    """
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate EMA crossover dynamics."""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            if len(bars_1m) < 25:
                return 0.0
            
            # Get price data
            closes = [bar.get('close', 0) for bar in bars_1m[-25:]]
            volumes = [bar.get('volume', 0) for bar in bars_1m[-25:]]
            closes = [c for c in closes if c > 0]
            
            if len(closes) < 20:
                return 0.0
            
            closes = np.array(closes)
            volumes = np.array(volumes[:len(closes)])
            
            # Calculate EMAs
            ema9 = self._calculate_ema(closes, 9)
            ema20 = self._calculate_ema(closes, 20)
            
            if len(ema9) < 15 or len(ema20) < 15:
                return 0.0
            
            # Look for recent crossovers
            ema9_above = ema9 > ema20
            crossover_points = []
            
            for i in range(1, len(ema9_above)):
                if ema9_above[i] != ema9_above[i-1]:
                    crossover_points.append((i, ema9_above[i]))
            
            if not crossover_points:
                return 0.0  # No recent crossovers
            
            # Analyze most recent crossover
            last_cross_idx, is_bullish_cross = crossover_points[-1]
            
            # Only analyze if crossover is recent (last 10 bars)
            if len(ema9) - last_cross_idx > 10:
                return 0.0
            
            # Measure crossover quality
            # 1. Speed of crossover (sharp vs gradual)
            cross_speed = abs(ema9[last_cross_idx] - ema20[last_cross_idx]) / ema20[last_cross_idx]
            
            # 2. Follow-through after crossover
            bars_after_cross = len(ema9) - last_cross_idx
            if bars_after_cross > 1:
                if is_bullish_cross:
                    follow_through = np.mean(ema9[last_cross_idx:] > ema20[last_cross_idx:])
                else:
                    follow_through = np.mean(ema9[last_cross_idx:] < ema20[last_cross_idx:])
            else:
                follow_through = 1.0
            
            # 3. Volume confirmation around crossover
            pre_cross_volume = np.mean(volumes[max(0, last_cross_idx-3):last_cross_idx])
            post_cross_volume = np.mean(volumes[last_cross_idx:])
            volume_confirmation = post_cross_volume / (pre_cross_volume + 1e-8)
            
            # 4. Price momentum around crossover
            pre_cross_momentum = (closes[last_cross_idx] - closes[max(0, last_cross_idx-3)]) / closes[max(0, last_cross_idx-3)]
            post_cross_momentum = (closes[-1] - closes[last_cross_idx]) / closes[last_cross_idx]
            
            momentum_consistency = 1.0 if pre_cross_momentum * post_cross_momentum > 0 else 0.0
            
            # Combine quality metrics
            crossover_quality = (
                min(cross_speed * 100, 1.0) * 0.25 +      # Speed
                follow_through * 0.25 +                   # Follow-through
                min(volume_confirmation / 2.0, 1.0) * 0.25 +  # Volume
                momentum_consistency * 0.25               # Momentum consistency
            )
            
            # Apply direction
            direction = 1 if is_bullish_cross else -1
            final_score = crossover_quality * direction
            
            return float(np.clip(final_score, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA for given period."""
        if len(prices) < period:
            return np.array([])
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {'data_type': 'mf_data', 'lookback': 25, 'fields': ['close', 'volume']}


class EMATrendAlignmentFeature(BaseFeature):
    """Measures multi-timeframe EMA trend alignment for momentum confirmation.
    
    Strong momentum occurs when EMAs across timeframes align in same direction.
    Professional traders use this for trend confirmation and strength assessment.
    """
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate EMA trend alignment."""
        try:
            bars_1m = market_data.get('1m_bars_window', [])
            bars_5m = market_data.get('5m_bars_window', [])
            
            if len(bars_1m) < 20 or len(bars_5m) < 10:
                return 0.0
            
            # Get 1m data
            closes_1m = [bar.get('close', 0) for bar in bars_1m[-20:]]
            closes_1m = [c for c in closes_1m if c > 0]
            
            # Get 5m data
            closes_5m = [bar.get('close', 0) for bar in bars_5m[-10:]]
            closes_5m = [c for c in closes_5m if c > 0]
            
            if len(closes_1m) < 15 or len(closes_5m) < 8:
                return 0.0
            
            closes_1m = np.array(closes_1m)
            closes_5m = np.array(closes_5m)
            
            # Calculate EMAs for 1m timeframe
            ema9_1m = self._calculate_ema(closes_1m, 9)
            ema20_1m = self._calculate_ema(closes_1m, 20)
            
            # Calculate EMAs for 5m timeframe
            ema9_5m = self._calculate_ema(closes_5m, 9)
            ema20_5m = self._calculate_ema(closes_5m, 20)
            
            if len(ema9_1m) < 10 or len(ema20_1m) < 10 or len(ema9_5m) < 5 or len(ema20_5m) < 5:
                return 0.0
            
            # Check EMA ordering for each timeframe
            # Bullish: EMA9 > EMA20, Bearish: EMA9 < EMA20
            
            # 1m timeframe alignment
            tf1_bullish = ema9_1m[-1] > ema20_1m[-1]
            tf1_strength = abs(ema9_1m[-1] - ema20_1m[-1]) / ema20_1m[-1]
            
            # 5m timeframe alignment  
            tf5_bullish = ema9_5m[-1] > ema20_5m[-1]
            tf5_strength = abs(ema9_5m[-1] - ema20_5m[-1]) / ema20_5m[-1]
            
            # Check if both timeframes agree
            alignment_agreement = 1.0 if tf1_bullish == tf5_bullish else 0.0
            
            # Overall direction
            overall_bullish = tf1_bullish and tf5_bullish
            overall_bearish = not tf1_bullish and not tf5_bullish
            
            # Calculate slope alignment (EMAs trending in same direction)
            ema9_1m_slope = (ema9_1m[-1] - ema9_1m[-5]) / ema9_1m[-5] if len(ema9_1m) >= 5 else 0
            ema20_1m_slope = (ema20_1m[-1] - ema20_1m[-5]) / ema20_1m[-5] if len(ema20_1m) >= 5 else 0
            ema9_5m_slope = (ema9_5m[-1] - ema9_5m[-3]) / ema9_5m[-3] if len(ema9_5m) >= 3 else 0
            ema20_5m_slope = (ema20_5m[-1] - ema20_5m[-3]) / ema20_5m[-3] if len(ema20_5m) >= 3 else 0
            
            # All slopes pointing same direction = strong alignment
            all_slopes = [ema9_1m_slope, ema20_1m_slope, ema9_5m_slope, ema20_5m_slope]
            positive_slopes = sum(1 for slope in all_slopes if slope > 0.001)
            negative_slopes = sum(1 for slope in all_slopes if slope < -0.001)
            
            slope_alignment = max(positive_slopes, negative_slopes) / 4.0
            
            # Combine alignment metrics
            alignment_strength = (
                alignment_agreement * 0.4 +                    # Timeframe agreement
                slope_alignment * 0.3 +                        # Slope alignment
                min((tf1_strength + tf5_strength) * 25, 1.0) * 0.3  # EMA separation strength
            )
            
            # Apply directional bias
            if overall_bullish:
                direction = 1
            elif overall_bearish:
                direction = -1
            else:
                direction = 0
            
            final_alignment = alignment_strength * direction
            
            return float(np.clip(final_alignment, -1.0, 1.0))
            
        except Exception:
            return 0.0
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA for given period."""
        if len(prices) < period:
            return np.array([])
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def get_default_value(self) -> float:
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        return {'min': -1.0, 'max': 1.0, 'range_type': 'symmetric'}
    
    def get_requirements(self) -> Dict[str, Any]:
        return {
            'data_type': 'mf_data', 
            'lookback': 25, 
            'fields': ['close'],
            '5m_bars_window': {'lookback': 10, 'fields': ['close']}
        }