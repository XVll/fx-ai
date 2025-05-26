# Market Simulator Refactoring Design Plan

## Overview

This document outlines a comprehensive refactoring plan for the MarketSimulator to support both live and file-based data, optimize performance, handle sparse data efficiently, and integrate with the upcoming environment_redesign_v3 requirements.

## Core Design Principles

1. **Unified Interface**: Single interface for both live and historical data
2. **Sparse Data Native**: Efficient handling of sparse market data without unnecessary filling
3. **Performance First**: Optimize for speed with minimal memory overhead
4. **Future State Tracking**: Support for execution simulation with latency
5. **Warm-up Data**: Seamless handling of pre-session data requirements
6. **Modular Architecture**: Clean separation of concerns for maintainability

## Architecture Design

### 1. Data Source Abstraction

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Iterator
import asyncio

class MarketDataSource(ABC):
    """Abstract base class for market data sources"""
    
    @abstractmethod
    async def subscribe(self, symbols: List[str], data_types: List[str]) -> None:
        """Subscribe to real-time data streams"""
        pass
    
    @abstractmethod
    async def get_snapshot(self, symbol: str, timestamp: datetime) -> Dict:
        """Get point-in-time market snapshot"""
        pass
    
    @abstractmethod
    def get_historical_range(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Get historical data for a range"""
        pass

class FileBasedDataSource(MarketDataSource):
    """Implementation for Databento file-based data"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self._data_cache = {}
        self._sparse_indices = {}  # Pre-computed indices for fast lookup
        
    async def subscribe(self, symbols: List[str], data_types: List[str]) -> None:
        # Pre-load and index data for efficient access
        await self._build_sparse_indices(symbols, data_types)

class LiveDataSource(MarketDataSource):
    """Implementation for live market data"""
    
    def __init__(self, connection_config: Dict):
        self.connection = None
        self._buffer = asyncio.Queue(maxsize=10000)
        self._snapshot_cache = TTLCache(maxsize=1000, ttl=60)
```

### 2. Sparse Data Architecture

Instead of creating uniform timelines with synthetic fills, we'll use an event-driven sparse data model:

```python
class SparseMarketState:
    """Efficient sparse representation of market state"""
    
    def __init__(self):
        # Sparse storage with fast lookup
        self.trades = SortedDict()  # timestamp -> trade data
        self.quotes = SortedDict()  # timestamp -> quote data
        self.bars = {
            '1s': SortedDict(),
            '1m': SortedDict(),
            '5m': SortedDict()
        }
        
        # Last known values for carry-forward
        self._last_values = {
            'price': None,
            'bid': None,
            'ask': None,
            'bid_size': 0,
            'ask_size': 0
        }
        
        # Efficient range queries
        self._indices = {
            'trades': IntervalTree(),
            'quotes': IntervalTree(),
            'bars': {tf: IntervalTree() for tf in ['1s', '1m', '5m']}
        }
    
    def add_event(self, event_type: str, timestamp: datetime, data: Dict):
        """Add market event to sparse storage"""
        if event_type == 'trade':
            self.trades[timestamp] = data
            self._indices['trades'].add(Interval(timestamp, timestamp, data))
            self._update_last_values(data)
        # ... similar for quotes and bars
    
    def get_state_at(self, timestamp: datetime, lookback_window: Optional[int] = None) -> Dict:
        """Get market state at specific timestamp with optional lookback"""
        state = {
            'timestamp': timestamp,
            'last_values': self._last_values.copy()
        }
        
        if lookback_window:
            # Efficient range query using interval trees
            state['trades'] = self._get_events_in_window('trades', timestamp, lookback_window)
            state['quotes'] = self._get_events_in_window('quotes', timestamp, lookback_window)
            
        return state
    
    def get_next_event_time(self, after: datetime) -> Optional[datetime]:
        """Get timestamp of next market event - useful for sparse iteration"""
        next_times = []
        
        for storage in [self.trades, self.quotes] + list(self.bars.values()):
            idx = storage.bisect_right(after)
            if idx < len(storage):
                next_times.append(storage.iloc[idx])
                
        return min(next_times) if next_times else None
```

### 3. Hybrid Simulator Design

The new MarketSimulator will support both sparse event-driven processing and traditional time-based stepping:

```python
class MarketSimulatorV2:
    """Refactored market simulator with sparse data support"""
    
    def __init__(self, config: SimulationConfig, data_source: MarketDataSource):
        self.config = config
        self.data_source = data_source
        self.mode = config.mode  # 'sparse', 'uniform', or 'hybrid'
        
        # State management
        self.sparse_state = SparseMarketState()
        self.current_time = None
        
        # Feature extraction windows (configurable)
        self.feature_windows = {
            'hf': config.hf_window,  # e.g., 60 seconds
            'mf': config.mf_window,  # e.g., 20 x 1-minute bars
            'lf': config.lf_window   # e.g., 12 x 5-minute bars
        }
        
        # Execution lookahead for slippage simulation
        self.execution_buffer = deque(maxlen=config.execution_latency_seconds)
        
        # Warm-up data handling
        self.warmup_manager = WarmupDataManager(config.warmup_periods)
        
    async def initialize(self, symbol: str, start_time: datetime, end_time: datetime):
        """Initialize simulator with data"""
        # Load warmup data (previous day, pre-market)
        warmup_data = await self.warmup_manager.load_warmup_data(
            symbol, start_time, self.data_source
        )
        
        # Populate sparse state with historical data
        if self.data_source.__class__.__name__ == 'FileBasedDataSource':
            await self._load_historical_sparse(symbol, start_time, end_time)
        else:
            # For live mode, just initialize with warmup
            self._initialize_from_warmup(warmup_data)
            
    def step(self) -> Tuple[bool, MarketUpdate]:
        """Advance simulation by one step"""
        if self.mode == 'sparse':
            return self._sparse_step()
        elif self.mode == 'uniform':
            return self._uniform_step()
        else:  # hybrid
            return self._hybrid_step()
    
    def _sparse_step(self) -> Tuple[bool, MarketUpdate]:
        """Event-driven stepping - jump to next market event"""
        next_event_time = self.sparse_state.get_next_event_time(self.current_time)
        
        if not next_event_time:
            return False, None
            
        # Jump to next event
        self.current_time = next_event_time
        
        # Get state with appropriate lookback windows
        market_update = self._create_market_update(self.current_time)
        
        return True, market_update
    
    def _create_market_update(self, timestamp: datetime) -> MarketUpdate:
        """Create comprehensive market update for given timestamp"""
        update = MarketUpdate(timestamp=timestamp)
        
        # Get current state with LOCF values
        current_state = self.sparse_state.get_state_at(timestamp)
        update.current_price = current_state['last_values']['price']
        update.best_bid = current_state['last_values']['bid']
        update.best_ask = current_state['last_values']['ask']
        
        # Get feature windows efficiently
        update.hf_window = self._get_sparse_window('hf', timestamp)
        update.mf_window = self._get_sparse_window('mf', timestamp)
        update.lf_window = self._get_sparse_window('lf', timestamp)
        
        # Calculate derived features on-demand
        update.features = self._calculate_features_lazy(update)
        
        # Add execution lookahead for slippage calculation
        update.future_states = self._get_future_states(timestamp)
        
        return update
    
    def _get_sparse_window(self, window_type: str, timestamp: datetime) -> SparseWindow:
        """Get sparse data window for feature extraction"""
        window_config = self.feature_windows[window_type]
        
        # Create sparse window that only materializes data when accessed
        return SparseWindow(
            sparse_state=self.sparse_state,
            end_time=timestamp,
            window_size=window_config['size'],
            data_types=window_config['data_types']
        )
```

### 4. Performance Optimizations

```python
class PerformanceOptimizations:
    """Key optimizations for speed"""
    
    @staticmethod
    def create_indexed_cache(df: pd.DataFrame) -> IndexedCache:
        """Create indexed cache for O(log n) lookups"""
        return IndexedCache(
            data=df,
            indices={
                'time': df.index,
                'price_levels': create_price_level_index(df),
                'volume_buckets': create_volume_bucket_index(df)
            }
        )
    
    @staticmethod
    @numba.jit(nopython=True)
    def calculate_microstructure_features(trades: np.ndarray, quotes: np.ndarray) -> np.ndarray:
        """JIT-compiled feature calculation for speed"""
        # Implement core calculations in numba for 10-100x speedup
        pass
    
    @staticmethod
    def create_memory_mapped_cache(file_path: str) -> np.memmap:
        """Memory-mapped files for large datasets"""
        return np.memmap(file_path, dtype='float32', mode='r')
```

### 5. Live/Backtest Unified Interface

```python
class UnifiedMarketInterface:
    """Single interface for both live and backtest modes"""
    
    def __init__(self, mode: str, config: Dict):
        self.mode = mode
        
        if mode == 'live':
            self.data_source = LiveDataSource(config['live_connection'])
            self.simulator = MarketSimulatorV2(config, self.data_source)
            self.is_async = True
        else:
            self.data_source = FileBasedDataSource(DataManager(config))
            self.simulator = MarketSimulatorV2(config, self.data_source)
            self.is_async = False
    
    async def get_market_update(self) -> MarketUpdate:
        """Get next market update - works for both modes"""
        if self.mode == 'live':
            return await self.simulator.get_live_update()
        else:
            success, update = self.simulator.step()
            return update if success else None
    
    def supports_lookahead(self) -> bool:
        """Check if future state queries are supported"""
        return self.mode == 'backtest'
```

### 6. Integration with Environment V3

Support for momentum phase tracking and advanced features:

```python
class MomentumAwareMarketSimulator(MarketSimulatorV2):
    """Extended simulator for environment v3 requirements"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum_tracker = MomentumPhaseTracker()
        self.tape_analyzer = TapeAnalyzer()
        
    def _create_market_update(self, timestamp: datetime) -> MarketUpdate:
        """Enhanced update with momentum features"""
        update = super()._create_market_update(timestamp)
        
        # Add momentum phase detection
        update.momentum_phase = self.momentum_tracker.identify_phase(
            self.sparse_state, timestamp
        )
        
        # Add tape reading features
        update.tape_features = self.tape_analyzer.analyze_window(
            self.sparse_state.get_state_at(timestamp, lookback_window=60)
        )
        
        # Add setup quality metrics
        update.setup_quality = self._evaluate_setup_quality(timestamp)
        
        return update
```

### 7. Efficient Data Pipeline

```python
class EfficientDataPipeline:
    """Optimized data flow from source to features"""
    
    def __init__(self):
        # Lazy evaluation of features
        self.feature_cache = LRUCache(maxsize=10000)
        
        # Vectorized operations where possible
        self.vectorized_ops = VectorizedOperations()
        
        # Parallel processing for independent calculations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def process_market_data(self, raw_data: Dict) -> ProcessedData:
        """Process raw market data efficiently"""
        
        # Check cache first
        cache_key = self._create_cache_key(raw_data)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Parallel feature extraction
        futures = []
        
        if 'trades' in raw_data:
            futures.append(
                self.thread_pool.submit(self.vectorized_ops.process_trades, raw_data['trades'])
            )
            
        if 'quotes' in raw_data:
            futures.append(
                self.thread_pool.submit(self.vectorized_ops.process_quotes, raw_data['quotes'])
            )
        
        # Wait for all calculations
        results = await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
        
        # Combine results
        processed = ProcessedData.combine(results)
        self.feature_cache[cache_key] = processed
        
        return processed
```

## Implementation Benefits

### 1. **Performance Improvements**
- **Sparse data handling**: 10-100x reduction in memory usage and iteration time
- **Indexed lookups**: O(log n) instead of O(n) for time-based queries
- **Vectorized operations**: 10-50x speedup for feature calculations
- **Lazy evaluation**: Only compute features when needed

### 2. **Flexibility**
- **Mode switching**: Easy transition between sparse/uniform/hybrid modes
- **Live/backtest unity**: Same code paths for both modes
- **Configurable windows**: Adjust feature extraction windows without code changes

### 3. **Advanced Features**
- **Execution simulation**: Built-in support for latency and slippage
- **Momentum tracking**: Native support for environment v3 requirements
- **Warm-up handling**: Seamless previous day and pre-market data

### 4. **Maintainability**
- **Clean interfaces**: Clear separation between data sources and processing
- **Testability**: Mock data sources for unit testing
- **Extensibility**: Easy to add new data sources or processing pipelines

## Migration Strategy

1. **Phase 1**: Implement core sparse data structures and data source abstraction
2. **Phase 2**: Build unified interface and performance optimizations
3. **Phase 3**: Add momentum tracking and environment v3 features
4. **Phase 4**: Integrate with existing codebase with backward compatibility
5. **Phase 5**: Performance testing and optimization
6. **Phase 6**: Full migration and deprecation of old simulator

## Configuration Example

```yaml
market_simulator:
  mode: hybrid  # sparse, uniform, or hybrid
  
  data_source:
    type: file_based  # or live
    cache_size: 10000
    use_memory_mapping: true
    
  sparse_config:
    use_interval_trees: true
    compression: zstd
    
  performance:
    use_numba: true
    vectorize_operations: true
    parallel_features: true
    cache_features: true
    
  windows:
    hf:
      size: 60  # seconds
      data_types: ['trades', 'quotes']
    mf:
      size: 20  # 1-minute bars
      data_types: ['bars_1m']
    lf:
      size: 12  # 5-minute bars
      data_types: ['bars_5m']
      
  execution:
    latency_ms: 100
    lookahead_seconds: 5
    
  warmup:
    previous_days: 1
    premarket_hours: 1
```

## Expected Performance Metrics

- **Memory usage**: 80-90% reduction vs current implementation
- **Initialization time**: 5-10x faster
- **Step latency**: <1ms for sparse mode, <5ms for uniform mode
- **Feature calculation**: 10-50x faster with vectorization
- **Live mode latency**: <10ms from data arrival to feature availability

This refactored design addresses all requirements while providing significant performance improvements and better maintainability.