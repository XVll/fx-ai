# Feature System Documentation

## Overview

The FxAIv2 feature system is a modular, extensible framework for extracting and normalizing market microstructure features across multiple timeframes. The system is designed to handle sparse data, edge cases, and provide consistent normalized values for the reinforcement learning model.

## Architecture

### Core Components

1. **BaseFeature**: Abstract base class for all features
   - Handles normalization, default values, and data requirements
   - Ensures no NaN values through robust error handling

2. **FeatureRegistry**: Dynamic feature discovery and registration
   - Decorator-based registration system
   - Automatic loading of all feature modules

3. **FeatureManager**: Orchestrates feature calculation
   - Batch processing for efficiency
   - Aggregates data requirements across features
   - Handles missing features gracefully

4. **MarketContext**: Unified data structure for feature calculation
   - Contains all market state information
   - Provides consistent interface for features

## Implemented Features

### Static Features (3 implemented)
Features that change slowly or represent time/session information.

| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| S_Time_Of_Day_Sin | `time_of_day_sin` | [-1, 1] | Sine encoding of time (captures morning vs afternoon) |
| S_Time_Of_Day_Cos | `time_of_day_cos` | [-1, 1] | Cosine encoding of time (complement to sine) |
| S_Market_Session_Type | `market_session_type` | [0, 1] | 0=closed, 0.25=pre, 1.0=regular, 0.75=post |

### High-Frequency Features (6 implemented)
Features calculated from 1-second data over a 60-second window.

| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| HF_1s_Price_Velocity | `price_velocity` | [-1, 1] | Rate of price change (normalized to ±10%/sec) |
| HF_1s_Price_Acceleration | `price_acceleration` | [-1, 1] | Change in price velocity |
| HF_Tape_1s_Imbalance | `tape_imbalance` | [-1, 1] | Buy vs sell volume ratio |
| HF_Tape_1s_Aggression_Ratio | `tape_aggression_ratio` | [-1, 1] | Orders hitting bid vs ask |
| HF_Quote_1s_Spread_Compression | `spread_compression` | [-1, 1] | Change in bid-ask spread |
| - | `quote_velocity` | [-1, 1] | Mid-price velocity from quotes |

**Not Implemented:**
- HF_1s_Volume_Velocity
- HF_1s_Volume_Acceleration
- HF_Quote_1s_Quote_Imbalance

### Medium-Frequency Features (12 implemented)
Features calculated from 1-minute and 5-minute bars over 30-minute windows.

| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| MF_1m_Price_Velocity | `1m_price_velocity` | [-1, 1] | 1-minute price change rate |
| MF_5m_Price_Velocity | `5m_price_velocity` | [-1, 1] | 5-minute price change rate |
| MF_1m_Volume_Velocity | `1m_volume_velocity` | [-1, 1] | 1-minute volume change rate |
| MF_5m_Volume_Velocity | `5m_volume_velocity` | [-1, 1] | 5-minute volume change rate |
| MF_1m_Dist_To_EMA9 | `1m_ema9_distance` | [-1, 1] | Distance to 9-period EMA (±50% max) |
| MF_1m_Dist_To_EMA20 | `1m_ema20_distance` | [-1, 1] | Distance to 20-period EMA |
| MF_5m_Dist_To_EMA9 | `5m_ema9_distance` | [-1, 1] | 5m 9-period EMA distance |
| MF_5m_Dist_To_EMA20 | `5m_ema20_distance` | [-1, 1] | 5m 20-period EMA distance |
| MF_1m_Position_Current_Candle | `1m_position_in_current_candle` | [0, 1] | Position in current 1m candle |
| MF_5m_Position_In_CurrentCandle | `5m_position_in_current_candle` | [0, 1] | Position in current 5m candle |
| MF_1m_BodySize_Rel | `1m_body_size_relative` | [0, 1] | Body size relative to range |
| MF_5m_BodySize_Rel | `5m_body_size_relative` | [0, 1] | 5m body size relative |

**Not Implemented:**
- MF_1m/5m_Price_Acceleration
- MF_1m/5m_Volume_Acceleration
- MF_1m/5m_Position_In_PreviousCandle
- MF_1m/5m_UpperWick_Rel
- MF_1m/5m_LowerWick_Rel
- MF_1m/5m_Swing_High/Low_Dist

### Low-Frequency Features (7 implemented)
Features calculated from daily data and intraday ranges.

| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| LF_Position_In_Daily_Range | `daily_range_position` | [0, 1] | Position in today's high-low range |
| LF_Position_In_PrevDay_Range | `position_in_prev_day_range` | [0, 1] | Position relative to yesterday's range |
| LF_Price_Change_From_Prev_Close | `price_change_from_prev_close` | [-1, 1] | % change from previous close (±20% max) |
| LF_Dist_To_Closest_LT_Support | `support_distance` | [0, 1] | Distance to nearest support (0=at level) |
| LF_Dist_To_Closest_LT_Resistance | `resistance_distance` | [0, 1] | Distance to nearest resistance |
| LF_Whole_Dollar_Proximity | `whole_dollar_proximity` | [0, 1] | Distance to nearest $1.00 level |
| LF_Half_Dollar_Proximity | `half_dollar_proximity` | [0, 1] | Distance to nearest $0.50 level |

### Portfolio Features (5 documented, handled separately)
Portfolio features are extracted directly in the FeatureExtractor, not through the modular system.

- Portfolio_Current_Position_Size
- Portfolio_Average_Price
- Portfolio_Unrealized_PnL
- Portfolio_Time_In_Position
- Portfolio_Max_Adverse_Excursion

## Data Flow

```
MarketSimulator → MarketContext → FeatureManager → Feature Classes → Normalized Values
                                         ↓
                                  FeatureExtractor → Model Input Tensors
```

## Key Design Principles

### 1. No NaN Values
- Every feature MUST return a valid float value
- Features implement `get_default_value()` for missing data
- Graceful fallbacks for edge cases (division by zero, missing data)

### 2. Normalization
- All features normalized to bounded ranges (typically [-1, 1] or [0, 1])
- Normalization parameters are feature-specific
- Extreme values are clipped to maintain bounds

### 3. Missing Data Handling
- **No Trades/Quotes**: Use last known values (LOCF)
- **Insufficient History**: Adapt calculations to available data
- **Invalid Values**: Filter and use defaults

### 4. Edge Cases
- **Market Gaps**: Position features clamp to [0, 1]
- **Halted/Zero Price**: Return default values
- **Penny Stocks**: Percentage-based calculations

## Feature Calculation Details

### Trade Classification
Features that analyze trade flow use multiple methods:
1. Trade conditions (["BUY"], ["SELL"])
2. Price vs bid/ask comparison
3. Tick rule as fallback

### Support/Resistance Detection
- Uses local minima/maxima over 20-day window
- Filters levels by minimum separation (1% default)
- Returns distance as percentage of price

### EMA Calculation
- Adapts to available data periods
- Uses exponential weighting
- Returns normalized distance from price

### Time Encoding
- Converts market hours (9:30 AM - 4:00 PM ET) to radians
- Sine/cosine encoding preserves cyclical nature
- Handles extended hours appropriately

## Configuration

Features are automatically loaded by the FeatureManager. The system currently loads all available features in each category. Future enhancements could include:
- Configuration-based feature selection
- Dynamic feature enabling/disabling
- Custom feature parameters

## Performance Considerations

1. **Caching**: FeatureExtractor caches results per timestamp
2. **Vectorization**: Features process data in batches when possible
3. **Memory**: Rolling windows limit memory usage
4. **Computation**: Simple calculations prioritized over complex algorithms

## Testing

Each feature includes comprehensive tests covering:
- Normal operation
- Edge cases (missing data, extreme values)
- Normalization bounds
- Default value behavior

## Future Enhancements

1. **Missing Features** (24 total):
   - Volume acceleration features
   - Candle wick analysis
   - Swing high/low distances
   - Previous candle positions
   - Quote imbalance

2. **System Improvements**:
   - Configuration-based feature selection
   - Feature importance tracking
   - Online normalization parameter updates
   - Cross-feature dependencies

3. **Data Enhancements**:
   - Multi-day technical indicators
   - Market microstructure metrics
   - Cross-asset features
   - Sentiment indicators