# Feature Simplification Plan

## Problem Statement

Current features are inefficient because:
1. Build expensive 60-timestep sequences
2. Most features only use 1-3 timesteps from sequence
3. Transformer already handles temporal patterns
4. Redundant calculation at every timestep

## Recommended Solution: Aggregated Features

Instead of point-in-time features, use **aggregated statistics** from sequences:

### Current (Inefficient):
```python
# Calculated at each timestep independently
price_velocity[t] = (price[t] - price[t-1]) / price[t-1]  # Only uses 2 points
price_acceleration[t] = velocity[t] - velocity[t-1]        # Only uses 3 points
```

### Proposed (Efficient):
```python
# Calculated once for entire sequence window
price_momentum = {
    'mean_return': mean(all_returns_in_window),
    'return_volatility': std(all_returns_in_window), 
    'trend_strength': correlation(returns, time),
    'max_drawdown': worst_decline_in_window
}
```

## Features to Remove/Replace

### Remove These HF Features:
- `price_velocity` → Replace with `price_momentum_stats`
- `price_acceleration` → Included in momentum stats
- `hf_volume_velocity` → Replace with `volume_profile_stats`
- `hf_volume_acceleration` → Included in volume stats

### Remove These MF Features:
- `1m_price_velocity`, `5m_price_velocity` → Replace with trend features
- `1m_price_acceleration`, `5m_price_acceleration` → Redundant
- `1m_volume_velocity`, `5m_volume_velocity` → Replace with volume features

### Keep These (Already Efficient):
- Pattern features (use full sequences to find patterns)
- VWAP features (use session data)
- Portfolio features (track state changes)

## New Aggregated Features

### HF Aggregated (Replace 4 features with 2):
```python
class PriceMomentumStats:
    def calculate_raw(self, market_data):
        prices = extract_prices_from_hf_window(market_data)
        returns = calculate_returns(prices)
        
        return {
            'mean_return': np.mean(returns),
            'return_std': np.std(returns),
            'trend_strength': calculate_trend_strength(returns),
            'momentum_persistence': calculate_persistence(returns)
        }

class VolumeProfileStats:
    def calculate_raw(self, market_data):
        volumes = extract_volumes_from_hf_window(market_data)
        
        return {
            'volume_mean': np.mean(volumes),
            'volume_spike_count': count_spikes(volumes),
            'volume_trend': calculate_volume_trend(volumes),
            'volume_clustering': calculate_clustering(volumes)
        }
```

### MF Aggregated (Replace 8 features with 2):
```python
class CandlePatternStats:
    def calculate_raw(self, market_data):
        bars = market_data.get('mf_bars_1m', [])
        
        return {
            'trend_consistency': calculate_trend_consistency(bars),
            'volatility_regime': calculate_volatility_regime(bars),
            'momentum_quality': calculate_momentum_quality(bars),
            'range_behavior': calculate_range_behavior(bars)
        }
```

## Benefits of This Approach:

1. **Efficiency**: Calculate once per window instead of per timestep
2. **Information Rich**: Capture statistical properties of sequences
3. **Transformer Friendly**: Let model focus on cross-timeframe patterns
4. **Reduces Redundancy**: No more velocity + acceleration pairs
5. **Better Features**: Statistics are more stable than point measurements

## Implementation Plan:

### Phase 1: Replace Velocity Features
- Remove: `price_velocity`, `price_acceleration`, `volume_velocity`, `volume_acceleration`
- Add: `momentum_stats`, `volume_stats`
- Test: Verify model performance

### Phase 2: Replace MF Velocity Features  
- Remove: All MF velocity/acceleration features
- Add: `trend_stats`, `volatility_stats`
- Test: Compare performance

### Phase 3: Optimize Pattern Features
- Keep swing/pattern features but optimize calculation
- Ensure they use full sequences efficiently

## Expected Results:

- **Faster Training**: ~40% fewer features to calculate
- **Better Performance**: Richer statistical features
- **Cleaner Code**: Less redundant feature calculations
- **More Stable**: Aggregated features less noisy than point measurements