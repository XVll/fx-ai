"""Script to scan databento files and create momentum indices.

Usage:
    python scripts/scan_momentum_days.py --data-dir dnb/Mlgo --output-dir outputs/indices
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.scanner.momentum_scanner import MomentumScanner
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Scan for momentum days and create indices")
    parser.add_argument('--data-dir', type=str, default='dnb/Mlgo',
                       help='Directory containing Databento files')
    parser.add_argument('--output-dir', type=str, default='outputs/indices',
                       help='Directory to save index files')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to scan')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('momentum_scanner', level=getattr(logging, args.log_level))
    
    # Create scanner
    scanner = MomentumScanner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        logger=logger
    )
    
    # Run scan
    logger.info(f"Starting momentum scan in {args.data_dir}")
    
    try:
        day_df, reset_df = scanner.scan_all_symbols(
            symbols=args.symbols,
            max_workers=args.max_workers
        )
        
        # Print summary
        if not day_df.empty:
            logger.info("\n📊 Momentum Days Summary:")
            logger.info(f"   Total days found: {len(day_df)}")
            logger.info(f"   Unique symbols: {day_df['symbol'].nunique()}")
            logger.info(f"   Quality score range: {day_df['quality_score'].min():.2f} - {day_df['quality_score'].max():.2f}")
            logger.info(f"   Avg intraday move: {day_df['max_intraday_move'].mean():.1%}")
            
            # Top momentum days
            top_days = day_df.nlargest(5, 'quality_score')
            logger.info("\n🏆 Top 5 Momentum Days:")
            for _, row in top_days.iterrows():
                logger.info(f"   {row['symbol']} on {row['date'].strftime('%Y-%m-%d')}: "
                          f"{row['max_intraday_move']:.1%} move, quality={row['quality_score']:.2f}")
                
        if not reset_df.empty:
            logger.info(f"\n🎯 Reset Points Summary:")
            logger.info(f"   Total reset points: {len(reset_df)}")
            logger.info(f"   Avg per momentum day: {len(reset_df) / len(day_df):.1f}")
            
            # Pattern distribution
            pattern_counts = reset_df['pattern_type'].value_counts()
            logger.info("\n📈 Pattern Distribution:")
            for pattern, count in pattern_counts.items():
                logger.info(f"   {pattern}: {count} ({count/len(reset_df)*100:.1f}%)")
                
    except Exception as e:
        logger.error(f"Error during scan: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info("\n✅ Scan completed successfully!")
    

if __name__ == "__main__":
    main()