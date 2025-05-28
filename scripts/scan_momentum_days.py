"""Script to scan for momentum days using the new data architecture.

Usage:
    python scripts/scan_momentum_days.py --symbol MLGO --min-quality 0.5
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DatabentoFileProvider
from data.scanner.momentum_scanner import MomentumScanner
from utils.logger import setup_rich_logging, get_logger


def main():
    parser = argparse.ArgumentParser(description="Scan for momentum days in available data")
    parser.add_argument('--symbol', type=str, required=True,
                       help='Symbol to scan')
    parser.add_argument('--min-quality', type=float, default=0.5,
                       help='Minimum quality score threshold')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_rich_logging(level=args.log_level)
    logger = get_logger('momentum_scanner')
    
    # Create momentum scanner
    momentum_scanner = MomentumScanner(
        data_dir="dnb/mlgo",
        output_dir="outputs/momentum_index",
        logger=logger
    )
    
    # Create data provider and manager
    data_provider = DatabentoFileProvider(
        data_dir="dnb/mlgo",
        verbose=False
    )
    
    data_manager = DataManager(
        provider=data_provider,
        momentum_scanner=momentum_scanner,
        logger=logger
    )
    
    # Run scan
    logger.info(f"🔍 Scanning momentum days for {args.symbol}")
    
    try:
        # Get momentum days from data manager
        momentum_days = data_manager.get_momentum_days(
            symbol=args.symbol,
            min_quality=args.min_quality
        )
        
        # Print summary
        if not momentum_days.empty:
            logger.info("\n📊 Momentum Days Summary:")
            logger.info(f"   Total days found: {len(momentum_days)}")
            logger.info(f"   Quality score range: {momentum_days['activity_score'].min():.2f} - {momentum_days['activity_score'].max():.2f}")
            logger.info(f"   Avg score: {momentum_days['activity_score'].mean():.2f}")
            
            # Top momentum days
            top_days = momentum_days.nlargest(10, 'activity_score')
            logger.info("\n🏆 Top 10 Momentum Days:")
            for _, row in top_days.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                logger.info(f"   {date_str}: quality={row['activity_score']:.2f}")
        else:
            logger.warning(f"❌ No momentum days found for {args.symbol} with quality >= {args.min_quality}")
            
        # Check for reset points on a specific day
        if not momentum_days.empty:
            sample_date = momentum_days.iloc[0]['date']
            reset_points = data_manager.get_reset_points(args.symbol, sample_date)
            
            if not reset_points.empty:
                logger.info(f"\n🎯 Sample Reset Points for {sample_date}:")
                logger.info(f"   Total reset points: {len(reset_points)}")
                for idx, point in reset_points.head(5).iterrows():
                    time_str = point['timestamp'].strftime('%H:%M:%S') if hasattr(point['timestamp'], 'strftime') else str(point['timestamp'])
                    logger.info(f"   {time_str}: activity={point.get('activity_score', 0):.2f}")
                
        logger.info("\n✅ Scan completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during scan: {e}", exc_info=True)
        sys.exit(1)
    

if __name__ == "__main__":
    main()