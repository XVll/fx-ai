# Feature System Improvement Recommendations

## Executive Summary

After analyzing the current feature system, I've identified several opportunities for improvement that will enhance the model's ability to detect and trade squeeze setups effectively.

## Immediate Actions (High Priority)

### 1. **Consolidate Redundant Swing Features**

**Problem**: Swing features exist in both `feature/mf/swing_features.py` and `feature/pattern/` causing confusion and redundancy.

**Solution**:
```python
# Remove from feature_manager.py available_features['mf']:
'1m_swing_high_distance', '1m_swing_low_distance',
'5m_swing_high_distance', '5m_swing_low_distance'

# Keep only pattern versions which are more comprehensive:
'swing_high_distance', 'swing_low_distance',
'swing_high_price_pct', 'swing_low_price_pct',
'bars_since_swing_high', 'bars_since_swing_low'
```

**Impact**: Reduces feature count from 42 to 38 in MF, eliminates confusion, maintains all functionality.

### 2. **Add Critical Missing Features**

#### 2.1 Halt State Feature (TODO from docs)
```python
# Add to feature/market_structure/halt_features.py
@feature_registry.register("is_halted", category="static")
class HaltStateFeature(BaseFeature):
    """Trading halt status indicator."""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        return 1.0 if market_data.get('is_halted', False) else 0.0
```

#### 2.2 LULD (Limit Up/Limit Down) Features
```python
# Add to feature/market_structure/luld_features.py
@feature_registry.register("distance_to_luld_up", category="lf")
class DistanceToLULDUp(BaseFeature):
    """Distance to limit up band as percentage."""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        current_price = market_data.get('current_price', 0)
        luld_up = self._calculate_luld_up(market_data)
        if current_price <= 0 or luld_up <= 0:
            return 1.0  # Max distance if no valid data
        return (luld_up - current_price) / current_price
```

#### 2.3 VWAP Features
```python
# Add to feature/mf/vwap_features.py
@feature_registry.register("distance_to_vwap", category="mf")
class DistanceToVWAP(BaseFeature):
    """Distance from current price to VWAP as percentage."""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        current_price = market_data.get('current_price', 0)
        vwap = market_data.get('session_vwap', current_price)
        if current_price <= 0:
            return 0.0
        return (current_price - vwap) / vwap
```

#### 2.4 Relative Volume
```python
# Add to feature/mf/relative_volume_features.py
@feature_registry.register("relative_volume_at_time", category="mf")
class RelativeVolumeAtTime(BaseFeature):
    """Current volume compared to average at this time of day."""
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        current_volume = market_data.get('session_volume', 0)
        time_of_day = market_data.get('timestamp').time()
        avg_volume_at_time = self._get_historical_avg_volume(time_of_day)
        
        if avg_volume_at_time <= 0:
            return 1.0
        return current_volume / avg_volume_at_time
```

### 3. **Reorganize Feature Structure**

Create new specialized directories:

```
feature/
├── hf/                    # Keep as-is
├── mf/                    # Remove pattern features
├── lf/                    # Keep as-is
├── static/                # Keep as-is
├── portfolio/             # Keep as-is
├── pattern/               # All pattern detection
├── market_structure/      # NEW: Halts, LULD, circuit breakers
├── volume_analysis/       # NEW: VWAP, relative volume, cumulative delta
└── microstructure/        # NEW: Depth, order flow (future)
```

## Medium Priority Improvements

### 1. **Feature Naming Convention**

Standardize all feature names:
```python
# Current (inconsistent):
'price_velocity'           # No timeframe prefix
'1m_price_velocity'        # Timeframe at start
'swing_high_distance'      # No timeframe

# Proposed (consistent):
'hf_1s_price_velocity'     # branch_timeframe_feature
'mf_1m_price_velocity'
'mf_5m_swing_high_distance'
```

### 2. **Dynamic Feature Configuration**

Add configuration to enable/disable feature groups:
```yaml
# config/features.yaml
features:
  enabled_groups:
    - core           # Always on
    - pattern        # Squeeze patterns
    - volume         # Volume analysis
    - market_struct  # Halts, LULD
  
  disabled_groups:
    - microstructure # Not ready yet
    - experimental   # Testing only
```

### 3. **Feature Importance Tracking**

Add mechanism to track which features the model uses most:
```python
class FeatureImportanceTracker:
    def __init__(self):
        self.feature_attention_weights = {}
        
    def update(self, feature_name: str, attention_weight: float):
        if feature_name not in self.feature_attention_weights:
            self.feature_attention_weights[feature_name] = []
        self.feature_attention_weights[feature_name].append(attention_weight)
    
    def get_importance_ranking(self):
        return sorted(
            [(name, np.mean(weights)) for name, weights in self.feature_attention_weights.items()],
            key=lambda x: x[1],
            reverse=True
        )
```

## Low Priority (Future Enhancements)

### 1. **Advanced Market Microstructure**
- Order book imbalance (requires L2 data)
- Trade size distribution
- Quote stability metrics
- Maker/taker ratio

### 2. **Inter-Symbol Features**
- Sector momentum
- Market breadth
- Correlation with SPY/QQQ
- Relative strength

### 3. **Adaptive Features**
- Volatility-adjusted thresholds
- Regime-dependent calculations
- Self-calibrating normalization

## Implementation Priorities

### Phase 1 (1 week)
1. Consolidate swing features
2. Add halt state feature
3. Add LULD features
4. Add VWAP distance feature

### Phase 2 (2 weeks)
1. Add relative volume features
2. Reorganize feature directories
3. Implement feature naming convention
4. Add cumulative delta

### Phase 3 (1 month)
1. Feature importance tracking
2. Dynamic configuration
3. Advanced volume analysis
4. Performance optimization

## Expected Impact

### Performance Improvements
- **Reduced redundancy**: 10% faster feature calculation
- **Better organization**: Easier maintenance and debugging
- **Critical features**: Better squeeze detection and risk management

### Trading Improvements
- **Halt/LULD awareness**: Avoid trading into halts
- **VWAP as S/R**: Better entry/exit levels
- **Relative volume**: Identify unusual activity
- **Pattern clarity**: Cleaner squeeze detection

## Migration Guide

### Step 1: Update Feature Manager
```python
# In feature/feature_manager.py
available_features = {
    'mf': [
        # Remove these:
        # '1m_swing_high_distance', '1m_swing_low_distance',
        # '5m_swing_high_distance', '5m_swing_low_distance',
        
        # Keep pattern versions...
    ]
}
```

### Step 2: Update Model Config
```python
# In config/schemas.py
mf_feat_dim: int = 38  # Reduced from 42
```

### Step 3: Add New Features
Follow the feature engineering guide to add:
1. Halt state feature
2. LULD features  
3. VWAP features
4. Relative volume

### Step 4: Test Everything
```bash
poetry run pytest tests/test_features.py -v
poetry run python scripts/verify_features.py
```

## Conclusion

The current feature system is well-designed but can be improved by:
1. Eliminating redundancy (swing features)
2. Adding critical missing features (halt, LULD, VWAP)
3. Better organization (new directories)
4. Consistent naming conventions

These improvements will make the system more maintainable and improve trading performance, especially for squeeze setups where volume, VWAP, and halt awareness are critical.