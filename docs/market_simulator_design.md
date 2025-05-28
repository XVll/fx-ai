# Market Simulator Design

## Overview

MarketSimulator implements a uniform timeline approach with pre-calculated features for efficient training. Unlike previous versions that calculated features on-demand, pre-computes all market states and features for an entire trading day (4 AM - 8 PM ET) at initialization.

## Key Design Principles

### Handling Missing Previous Day Data
The simulator gracefully handles cases where previous trading day data is unavailable:
- Checks for cached previous day data from DataManager
- Searches up to 10 days back to find a valid trading day (handles holidays)
- Uses momentum index to verify data availability when possible
- Falls back to synthetic data with reasonable defaults if no previous day exists
- Ensures training can start even on the first day of available data

## Original Design Principles

### 1. Uniform Timeline
- Creates a complete 1-second resolution timeline for the entire trading day
- Every second has a market state, even if no trades occurred
- Uses forward-filling (LOCF - Last Observation Carried Forward) for missing data
- Synthetic bars are created for periods with no trading activity

### 2. Pre-calculated Features
- All features (HF, MF, LF, Static) are calculated once during initialization
- Features are stored in a DataFrame alongside market data
- O(1) lookup time during training - no recalculation needed
- Significant performance improvement for training

### 3. Data Flow
```
Raw Data (trades, quotes, status) 
    ↓
Build 1s Bars from Trades
    ↓
Aggregate to 1m, 5m Bars
    ↓
Create Uniform Timeline (every second)
    ↓
For each second:
    - Build HF window (60s of 1s data)
    - Build MF window (60 x 1m bars)
    - Build LF window (60 x 5m bars)
    - Calculate features via FeatureManager
    - Store in df_market_state
```

## Implementation Details

### Market State Structure
Each second in the timeline contains:
```python
{
    # Market Data
    'timestamp': pd.Timestamp,
    'current_price': float,
    'best_bid': float,
    'best_ask': float,
    'bid_size': int,
    'ask_size': int,
    'mid_price': float,
    'spread': float,
    'market_session': str,  # PREMARKET, REGULAR, POSTMARKET, CLOSED
    'is_halted': bool,
    
    # Session Statistics
    'intraday_high': float,
    'intraday_low': float,
    'session_volume': float,
    'session_trades': int,
    'session_vwap': float,
    
    # Pre-calculated Features
    'hf_features': np.ndarray,  # (hf_seq_len, hf_feat_dim)
    'mf_features': np.ndarray,  # (mf_seq_len, mf_feat_dim)
    'lf_features': np.ndarray,  # (lf_seq_len, lf_feat_dim)
    'static_features': np.ndarray  # (static_feat_dim,)
}
```

### Building Uniform Timeline

1. **Create 1s Bars from Trades**
   ```python
   # Group trades by second
   trades_df['timestamp_1s'] = trades_df.index.floor('1s')
   
   # Aggregate to OHLCV
   bars_1s = trades_df.groupby('timestamp_1s').agg({
       'price': ['first', 'max', 'min', 'last'],
       'size': 'sum'
   })
   ```

2. **Fill Missing Seconds**
   ```python
   # Create complete timeline
   timeline = pd.date_range(start='04:00', end='20:00', freq='1s')
   
   # Reindex bars to timeline
   bars_1s = bars_1s.reindex(timeline)
   
   # Forward fill prices
   bars_1s[['open', 'high', 'low', 'close']] = bars_1s[['open', 'high', 'low', 'close']].ffill()
   ```

3. **Aggregate to Higher Timeframes**
   ```python
   # 1-minute bars
   bars_1m = bars_1s.resample('1min').agg({
       'open': 'first',
       'high': 'max',
       'low': 'min',
       'close': 'last',
       'volume': 'sum'
   })
   
   # 5-minute bars
   bars_5m = bars_1s.resample('5min').agg(...)
   ```

### Feature Window Construction

For each second in the timeline, we build historical windows:

1. **HF Window (High-Frequency)**
   - 60 seconds of tick data
   - Includes trades, quotes, and 1s bars
   - Always has exactly 60 entries (uniform)

2. **MF Window (Medium-Frequency)**
   - 60 x 1-minute bars
   - Aligned to minute boundaries
   - Synthetic bars for missing data

3. **LF Window (Low-Frequency)**
   - 60 x 5-minute bars
   - Aligned to 5-minute boundaries
   - Synthetic bars for missing data

### Example: Synthetic Data Handling

When no trades occur:
```python
# If no trades in a second, create synthetic bar
if no_trades:
    synthetic_bar = {
        'open': last_known_price,
        'high': last_known_price,
        'low': last_known_price,
        'close': last_known_price,
        'volume': 0,
        'is_synthetic': True
    }
```

## Usage Example

```python
# Initialize simulator
market_sim = MarketSimulator(
    symbol="MLGO",
    data_manager=data_manager,
    model_config=config.model,
    simulation_config=config.simulation
)

# Initialize day - pre-calculates everything
market_sim.initialize_day(datetime(2025, 2, 10))

# Now you can query any second instantly
market_sim.set_time(pd.Timestamp("2025-02-10 09:30:00"))
state = market_sim.get_market_state()
features = market_sim.get_current_features()

# Step through time
for _ in range(3600):  # 1 hour
    state = market_sim.get_market_state()
    features = market_sim.get_current_features()
    # Use features for model inference
    market_sim.step()
```

## Performance Benefits

1. **Training Speed**: No feature calculation during training
2. **Consistency**: All episodes see identical features for same timestamp
3. **Memory Efficiency**: Features stored once, not recalculated
4. **Debugging**: Can inspect pre-calculated features easily

## Memory Considerations

For a typical trading day:
- Timeline: 57,600 seconds (16 hours)
- Features per second: ~15,000 floats
- Total memory: ~3.5 GB per symbol per day

This is a reasonable tradeoff for the significant performance gains during training.

## Future Enhancements

1. **Compression**: Store features in compressed format
2. **Lazy Loading**: Load features in chunks as needed
3. **Multi-Symbol**: Efficient handling of multiple symbols
4. **Live Trading**: Incremental feature updates for live data