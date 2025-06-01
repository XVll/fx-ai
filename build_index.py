import sys
sys.path.append('.')

from data.scanner.momentum_scanner import MomentumScanner
from utils.logger import setup_rich_logging
from config.loader import load_config
import logging
import pandas as pd

# Load configuration
config = load_config()

# Setup logging
setup_rich_logging(level='INFO')
logger = logging.getLogger('build_index')

# Create scanner with rank-based scoring
data_dir = f"{config.data.data_dir}/{config.data.symbols[0].lower()}"
scanner = MomentumScanner(
    data_dir=data_dir,
    output_dir=f"{config.data.index_dir}/momentum_index",
    momentum_config=config.momentum_scanning,
    scoring_config=config.rank_based_scoring,
    session_config=config.session_volume,
    strategies_config=config.curriculum.strategies,
    logger=logger
)

print("Building momentum index...")

# Scan all symbols to build and save the index
day_df, reset_df = scanner.scan_all_symbols(symbols=config.data.symbols, max_workers=1)

print(f"\n✅ Index built and saved!")
print(f"Found {len(day_df)} momentum days")
print(f"Found {len(reset_df)} reset points")

if not day_df.empty:
    print("\n📊 Momentum days by quality:")
    print(day_df.groupby(pd.cut(day_df['activity_score'], 
                               bins=[0, 0.3, 0.5, 0.7, 1.0],
                               labels=['Low', 'Medium', 'High', 'Very High']))['date'].count())
    
    print("\n🏆 Top 5 momentum days:")
    for _, row in day_df.nlargest(5, 'activity_score').iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: "
              f"score={row['activity_score']:.3f}, "
              f"move={row['max_intraday_move']:.1%}, "
              f"vol={row['volume_multiplier']:.1f}x")

# Now test that we can read the saved index
print("\n📖 Testing index loading...")
loaded_days, loaded_resets = scanner.load_index()
print(f"Loaded {len(loaded_days)} days from saved index")

# Script completed successfully