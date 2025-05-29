# Todo :
* Add a halt state feature and LULD levels.
# Feature System Documentation

## Overview

The FxAIv2 feature system is a modular, extensible framework for extracting and normalizing market microstructure features across multiple timeframes. The system is designed to handle sparse data, edge cases, and provide consistent normalized values for the reinforcement learning model. **Now enhanced with pattern recognition features and cross-timeframe analysis capabilities for squeeze trading.**

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

5. **PatternFeatures**: Advanced pattern detection for squeeze setups
   - Swing point identification
   - Consolidation and range compression detection
   - Multi-timeframe momentum alignment

### Model Architecture Updates

**Multi-Branch Transformer with Pattern Recognition**
- **Temporal Pooling**: Exponentially weighted averaging preserves temporal patterns (was: last-timestep-only)
- **Cross-Timeframe Attention**: HF features attend to MF/LF patterns for entry timing within larger setups
- **Pattern Extraction**: Convolutional layers identify key patterns in sequences
- **6-way Fusion**: HF, MF, LF, Portfolio, Static, and Cross-Attention branches

## Implemented Features

Total: **66 features** across 5 categories (50 original + 16 new pattern features)

### Static Features (3 implemented)
Features that change slowly or represent time/session information.

| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| S_Time_Of_Day_Sin | `time_of_day_sin` | [-1, 1] | Sine encoding of time (captures morning vs afternoon) |
| S_Time_Of_Day_Cos | `time_of_day_cos` | [-1, 1] | Cosine encoding of time (complement to sine) |
| S_Market_Session_Type | `market_session_type` | [0, 1] | 0=closed, 0.25=pre, 1.0=regular, 0.75=post |

### High-Frequency Features (9 implemented)
Features calculated from 1-second data over a 60-second window.

| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| HF_1s_Price_Velocity | `price_velocity` | [-1, 1] | Rate of price change (normalized to ±10%/sec) |
| HF_1s_Price_Acceleration | `price_acceleration` | [-1, 1] | Change in price velocity |
| HF_1s_Volume_Velocity | `hf_volume_velocity` | [-1, 1] | Rate of volume change (normalized to ±1000%/sec) |
| HF_1s_Volume_Acceleration | `hf_volume_acceleration` | [-1, 1] | Change in volume velocity |
| HF_Tape_1s_Imbalance | `tape_imbalance` | [-1, 1] | Buy vs sell volume ratio |
| HF_Tape_1s_Aggression_Ratio | `tape_aggression_ratio` | [-1, 1] | Orders hitting bid vs ask |
| HF_Quote_1s_Spread_Compression | `spread_compression` | [-1, 1] | Change in bid-ask spread |
| HF_Quote_1s_Quote_Imbalance | `quote_imbalance` | [-1, 1] | Bid vs ask size imbalance |
| HF_Quote_Velocity | `quote_velocity` | [-1, 1] | Mid-price velocity from quotes |

### Medium-Frequency Features (42 implemented - 26 original + 16 pattern)
Features calculated from 1-minute and 5-minute bars over 30-minute windows.

#### Original MF Features (26)
| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| MF_1m_Price_Velocity | `1m_price_velocity` | [-1, 1] | 1-minute price change rate |
| MF_5m_Price_Velocity | `5m_price_velocity` | [-1, 1] | 5-minute price change rate |
| MF_1m_Price_Acceleration | `1m_price_acceleration` | [-1, 1] | 1-minute price acceleration |
| MF_5m_Price_Acceleration | `5m_price_acceleration` | [-1, 1] | 5-minute price acceleration |
| MF_1m_Volume_Velocity | `1m_volume_velocity` | [-1, 1] | 1-minute volume change rate |
| MF_5m_Volume_Velocity | `5m_volume_velocity` | [-1, 1] | 5-minute volume change rate |
| MF_1m_Volume_Acceleration | `1m_volume_acceleration` | [-1, 1] | 1-minute volume acceleration |
| MF_5m_Volume_Acceleration | `5m_volume_acceleration` | [-1, 1] | 5-minute volume acceleration |
| MF_1m_Dist_To_EMA9 | `1m_ema9_distance` | [-1, 1] | Distance to 9-period EMA (±50% max) |
| MF_1m_Dist_To_EMA20 | `1m_ema20_distance` | [-1, 1] | Distance to 20-period EMA |
| MF_5m_Dist_To_EMA9 | `5m_ema9_distance` | [-1, 1] | 5m 9-period EMA distance |
| MF_5m_Dist_To_EMA20 | `5m_ema20_distance` | [-1, 1] | 5m 20-period EMA distance |
| MF_1m_Position_Current_Candle | `1m_position_in_current_candle` | [0, 1] | Position in current 1m candle |
| MF_5m_Position_In_CurrentCandle | `5m_position_in_current_candle` | [0, 1] | Position in current 5m candle |
| MF_1m_Position_In_PreviousCandle | `1m_position_in_previous_candle` | [0, 1] | Position in previous 1m candle |
| MF_5m_Position_In_PreviousCandle | `5m_position_in_previous_candle` | [0, 1] | Position in previous 5m candle |
| MF_1m_BodySize_Rel | `1m_body_size_relative` | [0, 1] | Body size relative to range |
| MF_5m_BodySize_Rel | `5m_body_size_relative` | [0, 1] | 5m body size relative |
| MF_1m_UpperWick_Rel | `1m_upper_wick_relative` | [0, 1] | Upper wick size relative to range |
| MF_5m_UpperWick_Rel | `5m_upper_wick_relative` | [0, 1] | 5m upper wick relative |
| MF_1m_LowerWick_Rel | `1m_lower_wick_relative` | [0, 1] | Lower wick size relative to range |
| MF_5m_LowerWick_Rel | `5m_lower_wick_relative` | [0, 1] | 5m lower wick relative |
| MF_1m_Swing_High_Dist | `1m_swing_high_distance` | [-1, 1] | Distance to 1m swing high |
| MF_1m_Swing_Low_Dist | `1m_swing_low_distance` | [-1, 1] | Distance to 1m swing low |
| MF_5m_Swing_High_Dist | `5m_swing_high_distance` | [-1, 1] | Distance to 5m swing high |
| MF_5m_Swing_Low_Dist | `5m_swing_low_distance` | [-1, 1] | Distance to 5m swing low |

#### New Pattern Detection Features (16)
| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| MF_Swing_High_Distance | `swing_high_distance` | [-1, 1] | Distance to last major swing high |
| MF_Swing_Low_Distance | `swing_low_distance` | [-1, 1] | Distance to last major swing low |
| MF_Swing_High_Price_Pct | `swing_high_price_pct` | [-1, 1] | % from current price to swing high |
| MF_Swing_Low_Price_Pct | `swing_low_price_pct` | [-1, 1] | % from current price to swing low |
| MF_Bars_Since_Swing_High | `bars_since_swing_high` | [0, 1] | Time since last swing high |
| MF_Bars_Since_Swing_Low | `bars_since_swing_low` | [0, 1] | Time since last swing low |
| MF_Higher_Highs_Count | `higher_highs_count` | [0, 1] | Number of higher highs in window |
| MF_Higher_Lows_Count | `higher_lows_count` | [0, 1] | Number of higher lows (uptrend) |
| MF_Lower_Highs_Count | `lower_highs_count` | [0, 1] | Number of lower highs (downtrend) |
| MF_Lower_Lows_Count | `lower_lows_count` | [0, 1] | Number of lower lows in window |
| MF_Range_Compression | `range_compression` | [0, 1] | Current range vs average (squeeze) |
| MF_Consolidation_Score | `consolidation_score` | [0, 1] | Tightness of price consolidation |
| MF_Triangle_Apex_Distance | `triangle_apex_distance` | [0, 1] | Distance to triangle pattern apex |
| MF_Momentum_Alignment | `momentum_alignment` | [-1, 1] | Multi-timeframe momentum alignment |
| MF_Breakout_Potential | `breakout_potential` | [0, 1] | Likelihood of range breakout |
| MF_Squeeze_Intensity | `squeeze_intensity` | [0, 1] | Combined squeeze indicators |

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

### Portfolio Features (5 implemented)
Portfolio features track the agent's current position and trading performance.

| Feature | Implementation | Range | Description |
|---------|----------------|-------|-------------|
| Portfolio_Current_Position_Size | `portfolio_position_size` | [-1, 1] | Current position size (normalized) |
| Portfolio_Average_Price | `portfolio_average_price` | [-1, 1] | Average entry price relative to current price |
| Portfolio_Unrealized_PnL | `portfolio_unrealized_pnl` | [-1, 1] | Current P&L (±10% clipped) |
| Portfolio_Time_In_Position | `portfolio_time_in_position` | [0, 1] | Time holding position (1 hour max) |
| Portfolio_Max_Adverse_Excursion | `portfolio_max_adverse_excursion` | [-1, 0] | Maximum drawdown in position |

## Data Flow

```
MarketSimulator → MarketContext → FeatureManager → Feature Classes → Normalized Values
                                         ↓
                                  FeatureExtractor → Model Input Tensors
                                         ↓
                              Multi-Branch Transformer with:
                              - Temporal Pooling (all timesteps)
                              - Cross-Timeframe Attention
                              - Pattern Extraction Layers
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

### 5. Temporal Pattern Preservation
- **Full Sequence Utilization**: Model uses weighted pooling, not just last timestep
- **Cross-Scale Dependencies**: HF features can query MF/LF patterns
- **Pattern Memory**: Key patterns extracted and preserved

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

### Pattern Detection (NEW)
- **Swing Points**: Identifies local highs/lows with 3-bar confirmation
- **Trend Analysis**: Counts higher highs/lows for trend strength
- **Range Compression**: Compares recent vs average range for squeeze detection
- **Triangle Detection**: Identifies converging price patterns
- **Momentum Alignment**: Checks if multiple timeframes show same direction

### EMA Calculation
- Adapts to available data periods
- Uses exponential weighting
- Returns normalized distance from price

### Time Encoding
- Converts market hours (9:30 AM - 4:00 PM ET) to radians
- Sine/cosine encoding preserves cyclical nature
- Handles extended hours appropriately

## Configuration

Features are automatically loaded by the FeatureManager. The system loads all available features with these dimensions:
- **HF Features**: 9 features × 60 timesteps = 540 values → 256 dim projection
- **MF Features**: 42 features × 30 timesteps = 1,260 values → 256 dim projection  
- **LF Features**: 7 features × 30 timesteps = 210 values → 256 dim projection
- **Portfolio**: 5 features × 5 timesteps = 25 values → 256 dim projection
- **Static**: 3 features → 256 dim projection
- **Cross-Attention**: Derived from HF attending to MF/LF → 256 dim

Total raw features before projection: ~2,038 values
After transformer processing and fusion: 512 dimensional representation

## Performance Considerations

1. **Caching**: FeatureExtractor caches results per timestamp
2. **Vectorization**: Features process data in batches when possible
3. **Memory**: Rolling windows limit memory usage
4. **Computation**: Simple calculations prioritized over complex algorithms
5. **Temporal Efficiency**: Weighted pooling reduces sequence length while preserving patterns

## Testing

Each feature includes comprehensive tests covering:
- Normal operation
- Edge cases (missing data, extreme values)
- Normalization bounds
- Default value behavior
- Pattern detection accuracy

## Model Integration

The enhanced transformer architecture leverages features as follows:

1. **Branch Processing**: Each timeframe processed by dedicated transformer encoder
2. **Temporal Pooling**: Exponentially weighted average (recent data weighted more)
3. **Cross-Attention**: HF branch queries MF/LF for pattern context
4. **Pattern Extraction**: Conv1D layers identify key patterns in LF data
5. **Fusion**: 6-way attention fusion combines all information streams

This allows the model to:
- See developing squeeze patterns over full time windows
- Identify optimal entry points within larger patterns
- Maintain awareness of swing points and support/resistance
- React to tape acceleration while respecting chart structure

## Future Enhancements

1. **Completed**: All 66 features implemented (50 original + 16 pattern)

2. **System Improvements**:
   - Adaptive pattern detection thresholds
   - Volume profile features
   - Market regime detection
   - Inter-symbol correlation features
   - Options flow integration

3. **Potential New Features**:
   - VWAP deviation bands
   - Relative volume at time
   - Cumulative delta
   - Market depth imbalance
   - Time-weighted average price
   - Opening range breakout levels