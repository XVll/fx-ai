# Feature System Design

## Core Principles

### 1. No NaN Values
- Every feature MUST return a valid float value
- Features implement `get_default_value()` for missing data scenarios
- Graceful fallbacks for all edge cases (division by zero, missing data, etc.)

### 2. Normalization
- All features are normalized to bounded ranges
- Most features use [-1, 1] or [0, 1] ranges
- Normalization parameters are feature-specific

### 3. Modular Architecture
- Each feature is a self-contained class
- Features inherit from `BaseFeature`
- Registry pattern for dynamic feature discovery
- Easy to add/remove features via configuration

## Feature Categories

### Static Features (1 timestep)
- **time_of_day_sin/cos**: [-1, 1] - Natural sine/cosine encoding
- **market_session_type**: [0, 1] - 0=closed, 0.25=pre, 1.0=regular, 0.75=post

### HF Features (60-second window, 1s bars)
- **price_velocity**: [-1, 1] - Normalized to ±10% per second max
- **price_acceleration**: [-1, 1] - Change in velocity
- **tape_imbalance**: [-1, 1] - Buy/sell volume ratio
- **tape_aggression_ratio**: [-1, 1] - Market orders hitting bid vs ask

### MF Features (30-minute window, 1m/5m bars)
- **price_velocity_1m/5m**: [-1, 1] - Normalized to reasonable % changes
- **volume_velocity_1m/5m**: [-1, 1] - Volume change rate
- **distance_to_ema9/20**: [-1, 1] - Normalized to ±50% distance
- **position_in_candle**: [0, 1] - Position within current candle range
- **body_size_relative**: [0, 1] - Body size relative to range

### LF Features (10-day window, daily bars)
- **position_in_daily_range**: [0, 1] - Position in today's range
- **position_in_prev_day_range**: [0, 1] - Position relative to yesterday
- **price_change_from_prev_close**: [-1, 1] - Normalized to ±20% daily moves
- **distance_to_support/resistance**: [0, 1] - 0=at level, 1=far away
- **whole/half_dollar_proximity**: [0, 1] - 0=at level, 1=max distance

## Data Handling

### Missing Data Strategies

1. **No Trades/Quotes**
   - Use last known values (LOCF)
   - Default to neutral values (0.0 for imbalance, 0.5 for positions)

2. **Insufficient History**
   - Use available data with appropriate defaults
   - EMA calculations adapt to available periods
   - Support/resistance use fewer points if needed

3. **Invalid Values**
   - Filter out negative sizes, NaN prices
   - Clamp extreme values to normalization bounds
   - Use tick rule for trade classification if conditions missing

### Edge Cases

1. **Market Gaps**
   - Position features clamp to [0, 1] even if price outside range
   - Velocity features clip at maximum normalized values

2. **Halted/Zero Price**
   - Return default values
   - Avoid division by zero with explicit checks

3. **Penny Stocks**
   - Percentage-based calculations handle appropriately
   - Absolute dollar distances normalized by price

## Implementation Requirements

### Base Feature Interface
```python
class BaseFeature:
    def calculate(self, market_data: Dict[str, Any]) -> float:
        """Calculate normalized feature value"""
        raw_value = self.calculate_raw(market_data)
        return self.normalize(raw_value)
    
    def calculate_raw(self, market_data: Dict[str, Any]) -> float:
        """Calculate raw feature value"""
        raise NotImplementedError
    
    def get_default_value(self) -> float:
        """Return default normalized value for missing data"""
        raise NotImplementedError
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Return parameters for normalization"""
        raise NotImplementedError
    
    def get_requirements(self) -> Dict[str, Any]:
        """Return data requirements"""
        raise NotImplementedError
```

### Feature Manager
- Handles batch feature calculation
- Aggregates data requirements
- Manages feature enable/disable
- Provides vectorization for model input

### Normalization Classes
- `MinMaxNormalizer`: Maps to [0, 1] with clipping
- `StandardNormalizer`: Z-score with clipping at ±3 std
- Both handle NaN/inf gracefully

## Testing Strategy

### Unit Tests
- Each feature tested independently
- Edge cases explicitly tested
- Normalization bounds verified

### Integration Tests
- Multiple features from same data
- Cross-timeframe consistency
- Sparse data handling
- Extreme market conditions

### Performance Tests
- Caching effectiveness
- Batch calculation efficiency
- Memory usage with large windows

## Output Format

### Model Input Structure
```python
{
    "static": np.array((1, n_static_features), dtype=np.float32),
    "hf": np.array((hf_seq_len, n_hf_features), dtype=np.float32),
    "mf": np.array((mf_seq_len, n_mf_features), dtype=np.float32),
    "lf": np.array((lf_seq_len, n_lf_features), dtype=np.float32)
}
```

### Guarantees
- All values are float32
- No NaN or infinite values
- All values within normalized bounds
- Consistent shapes based on configuration