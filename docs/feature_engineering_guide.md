# Feature Engineering Guide for FxAIv2

## Overview

This guide provides best practices and guidelines for adding new features to the FxAIv2 trading system. Features are the foundation of our AI's decision-making, so proper design and implementation are critical.

## Feature Design Philosophy

### 1. **Purpose-Driven Design**
Every feature should answer a specific question about market state:
- **What** is happening? (price, volume, spread)
- **How fast** is it happening? (velocity, acceleration)
- **Where** are we in the range? (position, distance to levels)
- **When** did it happen? (time since event, bars since pattern)
- **Why** might it matter? (pattern detection, squeeze indicators)

### 2. **Temporal Alignment**
Features must be placed in the correct temporal branch based on:
- **Update Frequency**: How often does the underlying data change?
- **Decision Timeframe**: At what speed do we need this information?
- **Computational Cost**: Can we afford to calculate it every second?

## Feature Categories and When to Use Them

### High-Frequency (HF) Features - 1-second updates
**Use for:**
- Immediate market dynamics (tape reading)
- Order flow analysis
- Micro price movements
- Quote dynamics

**Examples:**
- Price velocity/acceleration over 1s
- Tape imbalance (buy vs sell volume)
- Spread compression
- Quote update velocity

**Don't use for:**
- Technical indicators (too noisy at 1s)
- Pattern recognition (needs more history)
- Support/resistance (too granular)

### Medium-Frequency (MF) Features - 1-5 minute updates
**Use for:**
- Technical indicators
- Candle patterns
- Short-term trends
- Entry/exit signals

**Examples:**
- EMA distances
- Swing points
- Candle analysis
- Pattern detection

**Don't use for:**
- Tick-level analysis (too slow)
- Daily statistics (too fast)
- Session-wide metrics

### Low-Frequency (LF) Features - Daily/session updates
**Use for:**
- Market context
- Major levels
- Session statistics
- Long-term trends

**Examples:**
- Support/resistance levels
- Daily range position
- Previous day comparisons
- Psychological levels

**Don't use for:**
- Intraday patterns (too slow)
- Entry timing (too coarse)

### Static Features - Rarely change
**Use for:**
- Time encoding
- Market state
- Symbol characteristics
- Session type

**Examples:**
- Time of day encoding
- Market cap
- Sector classification
- Holiday indicators

### Portfolio Features - Position-dependent
**Use for:**
- Risk management
- Position tracking
- P&L monitoring
- Trade analysis

**Examples:**
- Current position size
- Unrealized P&L
- Time in position
- Maximum adverse excursion

## Step-by-Step Guide to Adding a New Feature

### Step 1: Define the Feature

Ask yourself:
1. **What market behavior am I trying to capture?**
2. **What timeframe is most relevant?**
3. **What data do I need?**
4. **How will this help the model make better decisions?**

### Step 2: Choose the Right Branch

```python
# Decision tree for feature placement
if updates_every_second and needs_tick_data:
    branch = "HF"
elif updates_every_minute or needs_candle_patterns:
    branch = "MF"
elif updates_daily or uses_session_data:
    branch = "LF"
elif rarely_changes:
    branch = "Static"
elif depends_on_position:
    branch = "Portfolio"
```

### Step 3: Implement the Feature

#### 3.1: Create Feature Class

```python
from feature.feature_base import BaseFeature, FeatureConfig
from feature.feature_registry import feature_registry

@feature_registry.register("your_feature_name", category="mf")
class YourFeatureName(BaseFeature):
    """Brief description of what this feature captures."""
    
    def __init__(self, config: FeatureConfig = None):
        if config is None:
            config = FeatureConfig(name="your_feature_name", normalize=True)
        super().__init__(config)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate the raw feature value."""
        # Your calculation logic here
        return value
    
    def get_default_value(self) -> float:
        """Return default when calculation fails."""
        return 0.0
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Define how to normalize this feature."""
        return {
            'min': -1.0,  # Expected minimum
            'max': 1.0,   # Expected maximum  
            'range_type': 'symmetric'  # or 'positive'
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Specify data requirements."""
        return {
            'mf_bars_1m': {
                'lookback': 20,
                'fields': ['high', 'low', 'close', 'volume']
            }
        }
```

#### 3.2: Handle Edge Cases

```python
def calculate_raw(self, market_data: Dict[str, Any]) -> float:
    # Get required data
    bars = market_data.get('mf_bars_1m', [])
    
    # Handle missing data
    if not bars or len(bars) < self.min_required:
        return self.get_default_value()
    
    # Handle invalid values
    try:
        values = [bar['close'] for bar in bars if bar['close'] > 0]
        if not values:
            return self.get_default_value()
            
        # Your calculation
        result = your_calculation(values)
        
        # Sanity check
        if np.isnan(result) or np.isinf(result):
            return self.get_default_value()
            
        return float(result)
        
    except Exception as e:
        self.logger.warning(f"Feature calculation failed: {e}")
        return self.get_default_value()
```

### Step 4: Register the Feature

1. **Add to feature list** in `feature/feature_manager.py`:
```python
'mf': [
    # ... existing features ...
    'your_feature_name',  # Add your feature
]
```

2. **Update model config** if adding to new category or changing counts:
```python
# In config/schemas.py
mf_feat_dim: int = 43  # Increment if adding to MF
```

3. **Import in loader** in `feature/load_features.py`:
```python
from feature.mf import your_feature_module
```

### Step 5: Test the Feature

Create comprehensive tests:

```python
def test_your_feature():
    """Test your feature implementation."""
    feature = YourFeatureName()
    
    # Test normal case
    market_data = {
        'mf_bars_1m': [
            {'close': 100, 'volume': 1000},
            {'close': 101, 'volume': 1500},
        ]
    }
    result = feature.calculate(market_data)
    assert 0 <= result <= 1  # If normalized to [0,1]
    
    # Test edge cases
    assert feature.calculate({}) == feature.get_default_value()
    assert feature.calculate({'mf_bars_1m': []}) == feature.get_default_value()
```

### Step 6: Document the Feature

Update `docs/feature_system_plan.md`:
```markdown
| MF_Your_Feature | `your_feature_name` | [0, 1] | Brief description |
```

## Feature Engineering Best Practices

### 1. **Normalization is Critical**
- Always normalize features to bounded ranges
- Use [-1, 1] for symmetric features (velocity, momentum)
- Use [0, 1] for positive features (position, percentage)
- Clip extreme values to prevent outliers

### 2. **Handle Missing Data Gracefully**
- Never return NaN or infinity
- Use sensible defaults (usually 0.0)
- Log warnings for debugging
- Consider forward-filling when appropriate

### 3. **Design for Efficiency**
```python
# Good: Calculate once, use multiple times
ema_values = calculate_ema(prices, period=20)
distance = (current_price - ema_values[-1]) / current_price

# Bad: Recalculate every time
distance1 = (current_price - calculate_ema(prices, 20)[-1]) / current_price
distance2 = (current_price - calculate_ema(prices, 20)[-1]) / atr  # Recalculated!
```

### 4. **Think in Vectors**
```python
# Good: Vectorized operations
prices = np.array([bar['close'] for bar in bars])
returns = np.diff(prices) / prices[:-1]

# Bad: Loop-based calculations
returns = []
for i in range(1, len(bars)):
    ret = (bars[i]['close'] - bars[i-1]['close']) / bars[i-1]['close']
    returns.append(ret)
```

### 5. **Consider Feature Interactions**
Some features work better together:
- Velocity + Acceleration = Momentum quality
- Volume + Price movement = Conviction
- Pattern + Volume = Breakout probability

### 6. **Avoid Lookahead Bias**
```python
# Bad: Uses future information
swing_high = max([bar['high'] for bar in all_bars])  # Includes future!

# Good: Only uses past data
past_bars = [bar for bar in bars if bar['timestamp'] <= current_time]
swing_high = max([bar['high'] for bar in past_bars[-20:]])  # Last 20 bars only
```

## Common Patterns for Squeeze Trading Features

### 1. **Range Compression Detection**
```python
def calculate_range_compression(bars, lookback=20):
    recent_ranges = [(b['high'] - b['low']) for b in bars[-5:]]
    historical_ranges = [(b['high'] - b['low']) for b in bars[-lookback:]]
    
    recent_avg = np.mean(recent_ranges)
    historical_avg = np.mean(historical_ranges)
    
    # Lower ratio = tighter range = potential squeeze
    return recent_avg / historical_avg if historical_avg > 0 else 1.0
```

### 2. **Volume Surge Detection**
```python
def calculate_volume_surge(bars, lookback=20):
    recent_volume = sum(b['volume'] for b in bars[-3:])
    average_volume = np.mean([b['volume'] for b in bars[-lookback:]])
    
    # Higher ratio = volume surge = potential breakout
    return recent_volume / (3 * average_volume) if average_volume > 0 else 1.0
```

### 3. **Pattern Quality Score**
```python
def calculate_pattern_quality(highs, lows):
    # Higher highs and higher lows = uptrend
    hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
    
    # Normalize to [-1, 1]
    trend_score = (hh_count + hl_count) / (2 * (len(highs) - 1))
    return 2 * trend_score - 1  # Convert [0,1] to [-1,1]
```

## Debugging Features

### 1. **Add Logging**
```python
def calculate_raw(self, market_data: Dict[str, Any]) -> float:
    value = your_calculation(market_data)
    
    if self.config.debug:
        self.logger.info(f"{self.name}: {value:.4f}")
    
    return value
```

### 2. **Visualize Features**
```python
import matplotlib.pyplot as plt

def visualize_feature(timestamps, feature_values, prices):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    ax1.plot(timestamps, prices, label='Price')
    ax1.legend()
    
    ax2.plot(timestamps, feature_values, label='Feature')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.legend()
    
    plt.show()
```

### 3. **Check Feature Distribution**
```python
def analyze_feature_distribution(feature_values):
    print(f"Mean: {np.mean(feature_values):.4f}")
    print(f"Std: {np.std(feature_values):.4f}")
    print(f"Min: {np.min(feature_values):.4f}")
    print(f"Max: {np.max(feature_values):.4f}")
    print(f"% zeros: {(np.array(feature_values) == 0).sum() / len(feature_values) * 100:.1f}%")
```

## Advanced Topics

### 1. **Cross-Timeframe Features**
For features that need multiple timeframes:
```python
def calculate_momentum_alignment(market_data):
    # Check if all timeframes agree
    hf_trend = market_data['hf_velocity'] > 0
    mf_trend = market_data['mf_velocity'] > 0  
    lf_trend = market_data['lf_velocity'] > 0
    
    # Score from -1 (all bearish) to +1 (all bullish)
    alignment = (hf_trend + mf_trend + lf_trend - 1.5) / 1.5
    return alignment
```

### 2. **Adaptive Features**
Features that adjust to market conditions:
```python
def calculate_adaptive_threshold(values, market_regime):
    if market_regime == 'high_volatility':
        threshold = np.percentile(values, 90)
    else:
        threshold = np.percentile(values, 75)
    
    return (values[-1] - threshold) / threshold
```

### 3. **Composite Features**
Combining multiple signals:
```python
def calculate_squeeze_score(market_data):
    # Combine multiple indicators
    range_compression = calculate_range_compression(market_data)
    volume_buildup = calculate_volume_buildup(market_data)
    pattern_quality = calculate_pattern_quality(market_data)
    
    # Weighted combination
    weights = [0.4, 0.3, 0.3]
    score = sum(w * v for w, v in zip(weights, 
                [range_compression, volume_buildup, pattern_quality]))
    
    return np.clip(score, 0, 1)
```

## Checklist for New Features

Before submitting a new feature:

- [ ] Feature has clear purpose and documentation
- [ ] Placed in correct temporal branch
- [ ] Implements all required methods
- [ ] Handles edge cases gracefully  
- [ ] Returns normalized values in expected range
- [ ] No NaN or infinity values possible
- [ ] Efficient calculation (no redundant operations)
- [ ] Added to feature manager configuration
- [ ] Updated model dimensions if needed
- [ ] Comprehensive tests written
- [ ] Documentation updated
- [ ] No lookahead bias
- [ ] Reasonable computational cost

## Conclusion

Good features are the foundation of successful trading models. Take time to:
1. Understand what market behavior you're capturing
2. Place features in the appropriate temporal branch
3. Handle edge cases robustly
4. Test thoroughly
5. Document clearly

Remember: A few well-designed features beat many poorly-designed ones. Quality over quantity!