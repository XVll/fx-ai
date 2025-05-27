"""Test script to demonstrate the new data layer functionality.

This script shows how the enhanced data layer works with:
- Momentum index-based day selection
- 2-tier caching
- Efficient market simulation
- Single-day episode support
"""

import sys
from pathlib import Path
import logging
import time
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DatabentoFileProvider
from data.scanner.momentum_scanner import MomentumScanner
from data.utils.index_utils import IndexManager
from simulators.market_simulator_v2 import MarketSimulatorV2
from utils.logger import setup_logger


def main():
    """Demonstrate the new data layer capabilities."""
    # Setup logging
    logger = setup_logger('test_data_layer', level=logging.INFO)
    
    # Initialize components
    logger.info("🚀 Initializing new data layer components...")
    
    # 1. Create data provider
    provider = DatabentoFileProvider(
        data_dir='dnb/mlgo',
        verbose=False
    )
    
    # 2. Create momentum scanner
    scanner = MomentumScanner(
        data_dir='dnb/mlgo',
        output_dir='outputs/indices',
        logger=logger
    )
    
    # 3. Create enhanced data manager with scanner
    data_manager = DataManager(
        provider=provider,
        momentum_scanner=scanner,
        preload_days=2,
        logger=logger
    )
    
    # 4. Load momentum indices
    momentum_days, reset_points = scanner.load_index()
    
    if momentum_days.empty:
        logger.warning("No momentum index found. Run scan_momentum_days.py first!")
        return
        
    # 5. Create index manager for smart selection
    index_manager = IndexManager(momentum_days, reset_points, logger)
    
    # 6. Create market simulator
    market_sim = MarketSimulatorV2(
        data_manager=data_manager,
        future_buffer_minutes=5,
        logger=logger
    )
    
    logger.info(f"\n📊 Momentum Index Stats:")
    logger.info(f"   Total momentum days: {len(momentum_days)}")
    logger.info(f"   Total reset points: {len(reset_points)}")
    logger.info(f"   Unique symbols: {momentum_days['symbol'].nunique()}")
    
    # Test day selection and loading
    symbol = 'MLGO'
    
    # Select a high-quality momentum day
    logger.info(f"\n🎯 Selecting momentum day for {symbol}...")
    selected_day = index_manager.select_training_day(
        symbol=symbol,
        curriculum_stage='early'  # Focus on high-quality days
    )
    
    if selected_day is None:
        logger.error(f"No momentum days found for {symbol}")
        return
        
    logger.info(f"   Selected: {selected_day['date']} with quality={selected_day['quality_score']:.2f}")
    logger.info(f"   Max intraday move: {selected_day['max_intraday_move']:.1%}")
    logger.info(f"   Volume multiplier: {selected_day['volume_multiplier']:.1f}x")
    
    # Load the day using enhanced data manager
    logger.info(f"\n📥 Loading day data with 2-tier caching...")
    start_time = time.time()
    
    day_data = data_manager.load_day(
        symbol=symbol,
        date=selected_day['date'],
        data_types=['bars_1s', 'quotes', 'trades', 'status']
    )
    
    load_time = time.time() - start_time
    logger.info(f"   Load time: {load_time:.2f}s")
    
    # Display loaded data stats
    for data_type, df in day_data.items():
        if df is not None and not df.empty:
            logger.info(f"   {data_type}: {len(df):,} rows")
            
    # Initialize market simulator for the day
    logger.info(f"\n🏛️ Initializing market simulator...")
    market_sim.initialize_day(symbol, selected_day['date'])
    
    # Get reset points for the day
    reset_points_df = index_manager.get_day_reset_points(
        symbol=symbol,
        date=selected_day['date'],
        max_points=5  # Just test with first 5
    )
    
    logger.info(f"   Found {len(reset_points_df)} reset points")
    
    # Test market state queries at reset points
    logger.info(f"\n📈 Testing market state queries at reset points:")
    
    for idx, reset_point in reset_points_df.iterrows():
        timestamp = reset_point['timestamp']
        pattern = reset_point['pattern_type']
        quality = reset_point['combined_score']
        
        # Get market state with O(1) lookup
        state_start = time.time()
        market_state = market_sim.get_market_state(timestamp)
        state_time = (time.time() - state_start) * 1000  # Convert to ms
        
        logger.info(f"\n   Reset Point {idx + 1}:")
        logger.info(f"     Time: {timestamp}")
        logger.info(f"     Pattern: {pattern}, Quality: {quality:.2f}")
        logger.info(f"     Bid: ${market_state.bid_price:.2f}, Ask: ${market_state.ask_price:.2f}")
        logger.info(f"     Spread: ${market_state.spread:.2f} ({market_state.spread/market_state.ask_price*100:.2f}%)")
        logger.info(f"     Query time: {state_time:.1f}ms")
        
        # Test order execution simulation
        if idx == 0:  # Just test on first reset point
            logger.info(f"\n   📊 Testing order execution simulation:")
            
            # Simulate a buy order
            exec_result = market_sim.simulate_order_execution(
                timestamp=timestamp,
                side='buy',
                size=1000,
                order_type='market'
            )
            
            logger.info(f"     Order: BUY 1000 shares")
            logger.info(f"     Requested price: ${exec_result.requested_price:.2f}")
            logger.info(f"     Executed price: ${exec_result.executed_price:.2f}")
            logger.info(f"     Slippage: ${exec_result.slippage:.2f}")
            logger.info(f"     Latency: {exec_result.latency_ms:.1f}ms")
            logger.info(f"     Commission: ${exec_result.commission:.2f}")
            
    # Test L2 cache pre-loading
    logger.info(f"\n🔄 Testing L2 cache pre-loading...")
    
    # Wait a bit for background pre-loading
    time.sleep(2)
    
    # Check cache stats
    cache_stats = data_manager.get_session_stats()
    logger.info(f"   L1 hits: {cache_stats['l1_hits']}")
    logger.info(f"   L2 hits: {cache_stats['l2_hits']}")
    logger.info(f"   L2 cache size: {cache_stats['l2_cache_size']}")
    logger.info(f"   Pre-loaded days: {cache_stats['preload_count']}")
    
    # Load next day (should hit L2 cache if pre-loaded)
    next_days = index_manager._get_next_momentum_days(symbol, selected_day['date'], 1)
    if next_days:
        logger.info(f"\n📥 Loading next day (testing L2 cache)...")
        next_start = time.time()
        
        next_data = data_manager.load_day(
            symbol=symbol,
            date=next_days[0]
        )
        
        next_load_time = time.time() - next_start
        logger.info(f"   Load time: {next_load_time:.2f}s")
        
        # Check if it was a cache hit
        new_stats = data_manager.get_session_stats()
        if new_stats['l2_hits'] > cache_stats['l2_hits']:
            logger.info(f"   ✅ L2 cache hit!")
        else:
            logger.info(f"   ❌ L2 cache miss (loaded from disk)")
            
    # Display final statistics
    logger.info(f"\n📊 Final Statistics:")
    
    final_stats = data_manager.get_session_stats()
    logger.info(f"   Total cache hit rate: {final_stats['cache_hit_rate']:.1f}%")
    logger.info(f"   Days loaded: {final_stats['days_loaded']}")
    logger.info(f"   Total rows loaded: {final_stats['total_rows_loaded']:,}")
    
    exec_stats = market_sim.get_execution_stats()
    if exec_stats:
        logger.info(f"\n   Execution Stats:")
        logger.info(f"     Total executions: {exec_stats['total_executions']}")
        logger.info(f"     Avg slippage: ${exec_stats['avg_slippage']:.3f}")
        logger.info(f"     Avg latency: {exec_stats['avg_latency_ms']:.1f}ms")
        
    usage_stats = index_manager.get_usage_stats()
    logger.info(f"\n   Index Usage Stats:")
    logger.info(f"     Unique days used: {usage_stats['total_unique_days']}")
    logger.info(f"     Avg usage per day: {usage_stats['avg_usage_per_day']:.1f}")
    
    # Cleanup
    logger.info(f"\n🧹 Cleaning up...")
    data_manager.close()
    
    logger.info(f"\n✅ Test completed successfully!")
    

if __name__ == "__main__":
    main()