# Market Simulator Adjustments for Feature System

## Current Market State Data

The MarketSimulator currently provides:
- `timestamp_utc` - Current timestamp
- `market_session` - PREMARKET, REGULAR, POSTMARKET, CLOSED
- `current_price` - Last known price
- `best_bid_price`, `best_ask_price` - L1 quotes
- `mid_price` - Calculated from bid/ask
- `best_bid_size`, `best_ask_size` - Quote sizes
- `intraday_high`, `intraday_low` - Day's high/low
- `previous_day_close` - Previous day's close
- `previous_day_data` - Dictionary with previous day OHLC
- `current_1s_bar` - Current 1-second bar
- `hf_data_window` - Rolling window of HF data (trades, quotes, 1s bars)
- `1m_bars_window` - Rolling window of 1-minute bars
- `5m_bars_window` - Rolling window of 5-minute bars

## Required Adjustments

### 1. Trade Data Enhancements
- **Trade Conditions**: Need to preserve trade conditions in the trades list within hf_data_window
- **Trade Side Classification**: Add buy/sell classification based on:
  - Trade conditions if available (e.g., ["BUY"], ["SELL"])
  - Price vs bid/ask comparison (at ask = buy, at bid = sell)
  - Tick rule as fallback (uptick = buy, downtick = sell)
- **Handle Missing Trades**: Ensure empty trades arrays when no activity

### 2. Historical Data
- **Multi-day History**: Add `daily_bars_window` with at least 20 days of daily bars for:
  - Support/resistance detection
  - Swing high/low calculations
  - Multi-day patterns
  
### 3. Session Data
- **Previous Session Data**: Enhance previous_day_data to include:
  - Post-market high/low/close
  - Pre-market high/low from current day
  - Gap calculations

### 4. Technical Levels
- **Support/Resistance**: Add automatic detection or allow passing in:
  - `support_levels` - List of recent support levels
  - `resistance_levels` - List of recent resistance levels
  
### 5. Additional Calculations
- **Cumulative Volume**: Track cumulative volume for the session
- **VWAP**: Session VWAP (already partially implemented)
- **Dollar Volume**: Track dollar volume in bars

### 6. Data Continuity
- **Fill Missing Seconds**: Ensure every second has an entry in hf_data_window
- **Synthetic Bars**: Create synthetic 1s bars using LOCF when no trades
- **Quote Continuity**: Carry forward last known bid/ask when no new quotes

## Implementation Priority

1. **Critical** (System won't work without these):
   - Ensure no None/NaN values in any data fields
   - Fill missing data with appropriate defaults
   - Consistent data structure even with sparse activity

2. **High Priority** (Required for basic features):
   - Trade conditions/classification in hf_data_window
   - daily_bars_window for LF features (20+ days)
   - Proper handling of pre-market data
   
3. **Medium Priority** (Enhances feature quality):
   - Enhanced previous session data
   - Support/resistance levels
   - Multi-day candle aggregations
   
4. **Low Priority** (Nice to have):
   - Additional cumulative metrics
   - Pre-calculated technical indicators

## Data Quality Requirements

1. **No Missing Values**:
   - Every timestamp must have complete data
   - Use LOCF (Last Observation Carried Forward) for prices
   - Use 0 or empty arrays for activity-based data (trades, quotes)

2. **Consistent Structure**:
   - hf_data_window always has exactly window_size entries
   - Each entry has: timestamp, 1s_bar (or None), trades[], quotes[]
   - Bar windows always sorted chronologically

3. **Edge Case Handling**:
   - First seconds of the day use previous day's close
   - Gaps in trading filled with synthetic data
   - Halts represented with unchanged prices, zero volume