# Data Layer and Market Simulator Redesign

## Overview

This document outlines the redesign of the data layer and market simulator to support efficient training on 1000+ symbols with 3-10 years of historical data. The design prioritizes:

1. **Fast data loading** with intelligent pre-fetching
2. **Efficient storage** using hierarchical caching
3. **Smart episode selection** based on momentum patterns
4. **Seamless transition** between historical and live trading

## Core Design Principles

### Data Organization Philosophy

The system organizes data hierarchically to optimize for both storage efficiency and access speed. At the top level, we maintain global indices that allow rapid identification of high-value training episodes without scanning individual files. Symbol-specific data is organized by data type and time granularity, with 1-second OHLCV bars serving as the primary data source.

The file structure reflects a balance between granularity and practicality. Daily files for tick data prevent individual files from becoming too large, while aggregated files for longer timeframes reduce file count. Each symbol maintains its own directory to enable parallel processing and independent data management.

### Data Type Selection Strategy

After analyzing the requirements for momentum trading, we've determined that 1-second OHLCV bars provide the optimal balance between granularity and storage efficiency. These bars capture sufficient detail for our 1-second decision intervals while being significantly smaller than raw tick data. 

Trades and quotes data serve as secondary sources, primarily used for realistic execution simulation and spread calculation. Status data, while small, is critical for identifying halted periods and ensuring we don't attempt to trade during restricted times. We intentionally skip hourly and daily bars as these can be computed on-demand from the 1-second data when needed for longer-term indicators.

## Momentum Scanner and Index System (Implemented)

### Scanning Strategy

The momentum scanner (`data/scanner/momentum_scanner.py`) operates as an offline process that pre-processes years of market data to identify high-value training days. Rather than randomly sampling from all available data, the scanner identifies days where significant price movement occurred, ensuring our model trains on the most relevant market conditions.

The implemented scanning process:

**Daily Momentum Identification**
- Processes Databento files directly (.dbn.zst format)
- Calculates daily price range: (high - low) / open
- Identifies days with 10-30% movement (configurable thresholds)
- Checks volume elevation compared to average
- Detects trading halts and squeeze patterns

**Intraday Analysis**
For each identified momentum day:
- Analyzes 1-second and 1-minute OHLCV data
- Identifies squeeze periods (rapid price/volume changes)
- Calculates quality scores based on movement and volume
- Determines best trading periods (market open, power hour, etc.)
- Stores comprehensive metrics for curriculum-based selection

### Quality Scoring System

Each identified momentum period receives a quality score based on multiple factors:

- **Price Movement**: The magnitude of the price change, with larger moves receiving higher scores
- **Volume Profile**: How volume compares to typical levels, with 2-5x normal volume being ideal
- **Pattern Clarity**: How clearly defined the momentum pattern is (breakout, flush, bounce, etc.)
- **Spread Quality**: Tighter spreads indicate better liquidity and more reliable price discovery
- **Time of Day**: Market open and power hour receive higher multipliers due to increased activity

### Index Storage and Retrieval

The scanner's output is stored in an efficient Parquet-format index:

**Day-Level Index:**
- Symbol and date
- Daily metrics (open, close, high, low, volume)
- Max intraday move percentage
- Number of halts
- Overall day quality score
- File paths to raw data

**Reset Point Index:**
- Parent day reference
- Timestamp of reset point
- Momentum phase classification
- Local quality score
- Volume and price velocity at point
- Pattern type (breakout, flush, etc.)

This dual-index structure allows:
1. Fast selection of high-value days
2. Quick access to reset points within selected days
3. Efficient filtering by pattern type or quality
4. Direct paths to raw data files

Total index size: ~100MB for 1000 symbols × 5 years

## Streamlined Data Management System (Implemented)

### Two-Tier Cache Architecture

The enhanced DataManager (`data/data_manager.py`) implements a simplified two-tier cache design optimized for single-day episodes with multiple reset points:

**Level 1 - Active Episode Cache (RAM)**
The active cache holds data for the current trading day being trained. Since episodes operate within a single day (4 AM - 8 PM ET) with multiple reset points, we load the entire day's data once and reuse it across resets. This includes:
- Full day's 1-second OHLCV data
- Associated trades and quotes for execution simulation
- Previous day's data for early morning lookback
- Approximately 2-4GB per symbol-day

**Level 2 - Pre-load Buffer (RAM)**
A small buffer holding the next 2-3 days ready for immediate loading. Since each day's training involves multiple reset points and can take 10-30 minutes depending on the number of resets, we only need a small number of days pre-loaded. The buffer size adapts based on training speed:
- Fast training (< 10 min/day): 3 days buffered
- Normal training (10-20 min/day): 2 days buffered
- Slow training (> 20 min/day): 1 day buffered

### Adaptive Pre-loading System

The pre-loading system is designed for efficiency with single-day episodes:

**Pre-loading Strategy:**
1. Monitor training progress on current day (number of resets completed)
2. When 75% of reset points are processed, queue next day for loading
3. Use momentum index to select next day based on curriculum requirements
4. Load in background while current day completes

Since we're working with pre-scanned momentum days from our index, selection is fast:
- Prime momentum days for early training
- Mixed quality days for intermediate training
- Full spectrum for advanced training

The system maintains a simple FIFO queue of upcoming days rather than complex prediction logic. This approach aligns with the episodic nature of training where we exhaust all reset points in a day before moving on.

### Lookback Data Handling

For momentum trading with intraday episodes, lookback handling is straightforward:

**Standard Lookback Requirements:**
- Previous trading day's full session (for daily indicators)
- Last 2 hours of previous day (for momentum context)
- Current day up to reset point

When a day is loaded for training:
1. Load current day's full data (4 AM - 8 PM)
2. Load previous trading day's data
3. Cache both in active memory
4. All reset points within the day share this data

**Edge Cases:**
- **Monday Training**: Automatically loads previous Friday
- **Post-Holiday**: Loads last valid trading day
- **First Day of Symbol**: Starts with available data
- **Weekend Gaps**: Uses last valid close for indicators

Since episodes don't span multiple days, we avoid complex data stitching and can maintain clean day boundaries.

## Enhanced Market Simulator Design (Implemented)

### Core Responsibilities

The MarketSimulatorV2 (`simulators/market_simulator_v2.py`) operates on single-day data loaded by the Data Manager:

1. **Timestamp-accurate market state** queries within the active day
2. **Future visibility** for realistic execution simulation
3. **Efficient O(1) lookups** using hash-based indices (NO precomputed 1s grids)
4. **Halt and trading status** awareness
5. **Sparse data handling** with on-demand interpolation

### State Management Architecture

The simulator maintains several key data structures for efficient operation:

**Hash-Based Time Indexing (Implemented)**
Rather than precomputing uniform 1-second grids, MarketSimulatorV2 builds hash indices mapping actual data timestamps to row indices. This enables O(1) constant-time lookups while using 90%+ less memory than grid-based approaches:

```python
def _build_indices(self):
    """Build hash indices for O(1) timestamp lookups."""
    if self.ohlcv_1s is not None and not self.ohlcv_1s.empty:
        for idx, timestamp in enumerate(self.ohlcv_1s.index):
            self.ohlcv_index[timestamp] = idx
```

**Future Buffer System**
To simulate realistic order execution with latency, the simulator maintains a rolling buffer of future data (typically 5 minutes ahead). This allows accurate simulation of:
- Execution latency (50-200ms typical)
- Slippage based on future price movement
- Order rejection due to halts
- Market impact on large orders

**State Interpolation**
For timestamps between available data points, the simulator performs intelligent interpolation:
- Linear interpolation for prices
- Volume distribution based on typical patterns
- Spread widening during low-activity periods
- Maintaining bid-ask relationship constraints

### Execution Simulation

The execution simulator models real-world trading mechanics:

**Latency Modeling**
Orders experience realistic delays from decision to execution:
- Network latency: 10-50ms
- Exchange processing: 20-100ms
- Total latency: 50-200ms typical

During this delay, the market continues moving, potentially resulting in different execution prices than expected.

**Slippage Calculation**
Slippage depends on multiple factors:
- Order size relative to typical volume
- Current spread and market depth
- Time of day (higher during low-volume periods)
- Recent volatility

The simulator uses the future buffer to calculate realistic slippage based on actual price movement during the execution window.

**Halt Detection**
The simulator continuously monitors trading status and automatically rejects orders during:
- Trading halts (volatility or news)
- Pre-market/after-hours restrictions
- Symbol-specific suspensions

## Intelligent Episode Selection System

### Day and Reset Point Selection

The selection system works with pre-scanned momentum days and their reset points:

**Prime Momentum Days** (Quality > 0.8)
Days with the cleanest momentum patterns:
- 20-50%+ intraday price movement
- Multiple high-quality reset points
- Clear phases: accumulation → breakout → momentum → distribution
- High volume throughout (3-5x average)
- Multiple trading opportunities

**Secondary Momentum Days** (Quality 0.6-0.8)
Good training days with moderate activity:
- 10-20% intraday movement
- Several usable reset points
- Some choppy periods mixed with trends
- Above-average volume (2-3x normal)

**Risk Training Days**
Days that teach risk management:
- Halted stocks with gap movements
- Failed breakout patterns
- Sudden reversals and flushes
- News-driven volatility

**Mixed Activity Days**
Realistic trading days with varied conditions:
- Periods of momentum and dead zones
- 5-10% movement with clear patterns
- Normal to elevated volume
- Good for patience training

Within each selected day, reset points are categorized by:
- **Time of day**: Market open, midday, power hour
- **Pattern type**: Breakout, flush, bounce, consolidation
- **Quality score**: Based on local momentum conditions

### Curriculum-Based Selection

The selection system implements a progressive curriculum that evolves with training progress:

**Early Training (0-10K episodes)**
Focus on high-quality setups to establish core patterns:
- 70% prime momentum episodes
- 20% secondary momentum
- 10% critical risk scenarios

This allows the model to first learn profitable patterns before dealing with complexity.

**Intermediate Training (10K-50K episodes)**
Broaden exposure while maintaining quality:
- 40% prime momentum
- 30% secondary momentum
- 20% risk scenarios
- 10% accumulation periods

The model learns to distinguish between different qualities of setups.

**Advanced Training (50K+ episodes)**
Full market exposure with equal weighting:
- Equal sampling from all categories
- Emphasis on diversity
- Inclusion of dead zones
- Real-world distribution

### Diversity and Anti-Overfitting Measures

The system tracks episode usage to prevent overfitting to specific examples:

- Usage counting per symbol-date combination
- Preference for less-used episodes
- Forced diversity after repeated sampling
- Symbol rotation to prevent bias

Performance-based adjustments dynamically modify selection:
- Poor performance on risk scenarios → increase their frequency
- Overtrading in dead zones → more patience training
- Strong momentum trading → introduce more complex scenarios

## Unified Data Provider Interface (Implemented)

### Design Philosophy

The UnifiedDataProvider (`data/provider/data_provider.py`) serves as an abstraction layer that makes the data source transparent to the rest of the system. Whether running historical backtests or live trading, the consuming components receive data through the same interface, ensuring consistency and simplifying the codebase.

### Historical Mode Operation

In historical mode, the provider:

1. Leverages the hierarchical cache system for fast data access
2. Uses the momentum index for intelligent episode selection
3. Handles all lookback requirements automatically
4. Provides consistent 1-second resolution data
5. Simulates real-time progression for training

### Live Mode Operation

In live mode, the provider:

1. Loads sufficient historical data for indicator calculation
2. Subscribes to real-time market data feeds
3. Merges historical and live data seamlessly
4. Maintains the same 1-second update frequency
5. Handles connection failures and data gaps

### Transition Handling

The system handles the transition from historical to live data carefully:

**Lookback Loading**
Before market open, the system loads the previous N days of data required for:
- Technical indicator calculation
- Pattern recognition context
- Volume profile comparison
- Support/resistance levels

**Real-time Integration**
As live data arrives:
- New bars are appended to the historical data
- Indicators update incrementally
- State remains consistent across the transition
- No gap between historical and live data

### Data Quality Assurance

The provider implements several quality checks:

- **Timestamp Validation**: Ensures chronological order
- **Gap Detection**: Identifies and handles missing data
- **Outlier Filtering**: Removes obviously erroneous prices
- **Halt Handling**: Properly manages trading suspensions
- **Corporate Actions**: Adjusts for splits and dividends

## Data Ownership and Responsibility

### Environment vs Market Simulator Responsibilities

**Environment Owns:**
- Episode lifecycle management (single day with multiple resets)
- Reset point selection within a day
- Action processing and validation
- Reward calculation
- Feature extraction orchestration
- Position handling across reset boundaries

**Market Simulator Owns:**
- Timestamp-accurate state queries within loaded day
- Execution simulation with realistic mechanics
- Future buffer for slippage calculation
- Spread and halt status tracking
- Intraday state interpolation

**Data Manager Owns:**
- Full day data loading from disk
- Two-tier caching (active day + pre-load buffer)
- Momentum index queries
- File system interaction
- Previous day lookback loading

**Momentum Scanner Owns:**
- Historical data scanning for high-value days
- Reset point identification within days
- Quality scoring and pattern classification
- Index creation and updates
- Curriculum-based day selection

This separation ensures each component has a clear, focused responsibility without overlap.

## Implementation Status

### Completed Components

✅ **Momentum Scanner** (`data/scanner/momentum_scanner.py`)
- Offline scanning of Databento files
- Quality scoring and momentum detection
- Parquet index generation
- CLI interface for batch processing

✅ **Enhanced DataManager** (`data/data_manager.py`)
- Two-tier caching system (L1 active, L2 pre-load)
- Single-day episode loading with lookback
- Background pre-loading support
- Efficient memory management

✅ **MarketSimulatorV2** (`simulators/market_simulator_v2.py`)
- O(1) hash-based lookups (no precomputed grids)
- Sparse data handling with interpolation
- Future buffer for execution simulation
- Integrated execution simulator

✅ **UnifiedDataProvider** (`data/provider/data_provider.py`)
- Consistent interface for historical/live data
- Timestamp-based data access
- Automatic data type selection

✅ **Supporting Utilities**
- Index management utilities (`data/utils/index_utils.py`)
- Test scripts (`scripts/test_new_data_layer.py`)
- Momentum scanning script (`scripts/scan_momentum_days.py`)

### Storage Optimizations Achieved

**Memory Efficiency:**
- No precomputed 1-second grids (90%+ memory savings)
- Sparse data representation with hash indices
- On-demand interpolation for missing timestamps
- Efficient caching of only high-value days

**Cache Storage:**
- L1 (RAM): 2-4GB per active trading day
- L2 (RAM): 6-12GB for 2-3 pre-loaded days
- Total RAM: ~15-20GB for smooth operation

### Key Implementation Changes from Original Design

**1. Offline vs Live Scanning**
- Originally planned: Live scanning during training
- Implemented: Offline pre-scanning with indexed results
- Benefit: No runtime overhead, instant episode selection

**2. Hash Indices vs Precomputed Grids**
- Originally planned: Uniform 1-second grids
- Implemented: Sparse data with hash-based O(1) lookups
- Benefit: 90%+ memory reduction, same performance

**3. Simplified Caching**
- Originally planned: Complex 3-tier cache
- Implemented: Simple 2-tier (active + pre-load)
- Benefit: Easier to manage, sufficient for single-day episodes

**4. Direct File Processing**
- Originally planned: Load through existing data pipeline
- Implemented: Direct Databento file processing in scanner
- Benefit: Faster scanning, no intermediate conversions

### System Integration Points

**Feature Extraction Integration:**
The feature extraction system will receive data through the market simulator's interface, with pre-computed rolling windows and efficient update mechanisms for incremental calculation.

**Metrics System Integration:**
The metrics system can query the market simulator for point-in-time market state, enabling accurate performance measurement and analysis.

**Reward System Integration:**
The reward calculator receives execution results from the market simulator, ensuring rewards reflect realistic trading outcomes including slippage and latency.

## Configuration Philosophy

The configuration system follows these principles:

1. **Sensible Defaults**: Works out-of-the-box for common cases
2. **Progressive Disclosure**: Advanced options available but not required
3. **Environment-Specific**: Separate configs for dev/staging/production
4. **Hot Reloading**: Key parameters adjustable without restart
5. **Validation**: Type-safe with clear error messages

## Monitoring and Observability

### Key Metrics to Track

**Data Layer Metrics:**
- Cache hit rates (L1/L2/L3)
- Data loading latencies
- Pre-load queue depth
- Memory usage by cache level

**Scanner Metrics:**
- Episodes scanned per second
- Quality score distribution
- Pattern type frequencies
- Symbol coverage

**Simulator Metrics:**
- State query latencies
- Execution simulation accuracy
- Interpolation frequency
- Future buffer utilization

### Debugging and Profiling

**Debug Modes:**
- Trace mode for data flow visualization
- Profiling mode for performance analysis
- Validation mode for data integrity checks
- Replay mode for issue reproduction

**Logging Strategy:**
- Structured logging with context
- Adjustable verbosity levels
- Performance-critical path sampling
- Automatic log rotation

## Next Implementation Steps

### Immediate Tasks

1. **Run Momentum Scanner**: Execute offline scanning to build indices
   ```bash
   poetry run python scripts/scan_momentum_days.py --symbol MLGO --start 2024-01-01 --end 2024-12-31
   ```

2. **Environment Integration**: Update TradingEnvironment to use MarketSimulatorV2

3. **Episode Scanner**: Implement runtime MomentumEpisodeScanner using indices

4. **Training Manager**: Add curriculum-based episode selection

5. **Position Handler**: Implement episode boundary handling

### Future Enhancements

1. **Distributed Training**: Redis for sharing momentum index across workers
2. **Incremental Scanning**: Update indices with new trading days
3. **Multi-Symbol Support**: Concurrent training across symbols
4. **Cloud Integration**: S3/GCS backend for large datasets
5. **Live Transition**: Seamless historical-to-live data handling

### Extensibility Points

The design includes several extension points for future enhancements:

- Pluggable data providers for new sources
- Custom scanner strategies for different markets
- Alternative caching backends
- Additional data types (options, L2, etc.)
- Multi-asset correlation support

## Conclusion

This design creates a robust, scalable data layer that can efficiently handle thousands of symbols while providing microsecond-precision market simulation. The hierarchical caching, intelligent pre-loading, and momentum-based episode selection ensure the system can scale to production requirements while maintaining the performance needed for effective reinforcement learning training.

The separation of concerns between the environment, market simulator, and data manager creates a clean architecture that's easy to understand, test, and extend. The unified interface between historical and live data ensures a smooth transition from development to production trading.