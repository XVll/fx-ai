"""Test script for MarketSimulatorV3 with uniform timeline and pre-calculated features"""

import pandas as pd
from datetime import datetime
import logging

from simulators.market_simulator import MarketSimulator
from feature.feature_extractor import FeatureExtractor
from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DatabentFileProvider
from config.loader import load_config


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config()
    
    # Initialize data provider and manager
    provider = DatabentFileProvider(
        base_path=config.data.base_path,
        logger=logger
    )
    
    data_manager = DataManager(
        provider=provider,
        logger=logger
    )
    
    # Test parameters
    symbol = "MLGO"
    test_date = datetime(2025, 2, 10)  # A Monday
    
    logger.info(f"Testing MarketSimulatorV3 for {symbol} on {test_date}")
    
    # Initialize market simulator
    market_sim = MarketSimulator(
        symbol=symbol,
        data_manager=data_manager,
        model_config=config.model,
        simulation_config=config.simulation,
        logger=logger
    )
    
    # Initialize day - this pre-calculates all features
    logger.info("Initializing day (pre-calculating all features)...")
    if not market_sim.initialize_day(test_date):
        logger.error("Failed to initialize day")
        return
        
    # Get time range
    start_time, end_time = market_sim.get_time_range()
    logger.info(f"Day initialized with uniform timeline: {start_time} to {end_time}")
    logger.info(f"Total seconds in timeline: {len(market_sim.df_market_state)}")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        symbol=symbol,
        market_simulator=market_sim,
        config=config.model,
        logger=logger
    )
    
    # Test at different times throughout the day
    test_times = [
        pd.Timestamp("2025-02-10 04:00:00", tz='America/New_York'),  # Pre-market start
        pd.Timestamp("2025-02-10 09:30:00", tz='America/New_York'),  # Market open
        pd.Timestamp("2025-02-10 12:00:00", tz='America/New_York'),  # Midday
        pd.Timestamp("2025-02-10 15:59:00", tz='America/New_York'),  # Near close
        pd.Timestamp("2025-02-10 19:00:00", tz='America/New_York'),  # Post-market
    ]
    
    logger.info("\n" + "="*80)
    logger.info("Testing uniform timeline at different times:")
    logger.info("="*80)
    
    for test_time in test_times:
        # Convert to UTC for lookup
        test_time_utc = test_time.tz_convert('UTC')
        
        # Jump to specific time
        if market_sim.set_time(test_time_utc):
            # Get market state
            state = market_sim.get_market_state()
            
            logger.info(f"\nTime: {test_time} ({test_time_utc} UTC)")
            logger.info(f"Market Session: {state.market_session}")
            logger.info(f"Current Price: ${state.current_price:.2f}")
            logger.info(f"Bid/Ask: ${state.best_bid:.2f} / ${state.best_ask:.2f} (spread: ${state.spread:.3f})")
            logger.info(f"Session Volume: {state.session_volume:,.0f}")
            logger.info(f"Intraday Range: ${state.intraday_low:.2f} - ${state.intraday_high:.2f}")
            
            # Get pre-calculated features
            features = market_sim.get_current_features()
            logger.info(f"Features shapes: HF={features['hf'].shape}, MF={features['mf'].shape}, "
                       f"LF={features['lf'].shape}, Static={features['static'].shape}")
    
    # Demonstrate stepping through time
    logger.info("\n" + "="*80)
    logger.info("Demonstrating uniform 1-second stepping:")
    logger.info("="*80)
    
    # Reset to a specific time
    start_demo = pd.Timestamp("2025-02-10 09:30:00", tz='America/New_York').tz_convert('UTC')
    market_sim.set_time(start_demo)
    
    # Step through 10 seconds
    for i in range(10):
        state = market_sim.get_market_state()
        logger.info(f"{state.timestamp}: Price=${state.current_price:.2f}, "
                   f"Volume={state.session_volume:,.0f}")
        
        if not market_sim.step():
            break
    
    # Show statistics
    logger.info("\n" + "="*80)
    logger.info("Day Statistics:")
    logger.info("="*80)
    stats = market_sim.get_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
        
    # Demonstrate feature extraction
    logger.info("\n" + "="*80)
    logger.info("Feature Extraction Test:")
    logger.info("="*80)
    
    # Reset to market open
    market_open = pd.Timestamp("2025-02-10 09:30:00", tz='America/New_York').tz_convert('UTC')
    market_sim.set_time(market_open)
    
    # Extract features using the V2 extractor
    features = feature_extractor.extract_features()
    if features:
        logger.info("Successfully extracted features:")
        for key, arr in features.items():
            logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            # Show sample values
            if arr.ndim == 1:
                logger.info(f"    Sample values: {arr[:5]}")
            else:
                logger.info(f"    Sample values (first row): {arr[0, :5]}")
    
    # Show memory efficiency
    logger.info("\n" + "="*80)
    logger.info("Memory Efficiency:")
    logger.info("="*80)
    
    # Calculate approximate memory usage
    total_seconds = len(market_sim.df_market_state)
    features_per_second = (
        config.model.hf_seq_len * config.model.hf_feat_dim +
        config.model.mf_seq_len * config.model.mf_feat_dim +
        config.model.lf_seq_len * config.model.lf_feat_dim +
        config.model.static_feat_dim
    )
    
    memory_mb = (total_seconds * features_per_second * 4) / (1024 * 1024)  # 4 bytes per float32
    logger.info(f"Approximate feature memory usage: {memory_mb:.1f} MB for {total_seconds:,} seconds")
    logger.info(f"Features per second: {features_per_second:,}")
    
    # Clean up
    market_sim.close()
    logger.info("\nTest completed successfully!")
    

if __name__ == "__main__":
    main()