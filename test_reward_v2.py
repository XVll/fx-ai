#!/usr/bin/env python3
"""
Test script for the new comprehensive reward system (v2).
Shows how to enable and configure the advanced reward components.
"""

import yaml
from pathlib import Path


def create_test_config():
    """Create a test configuration that uses the new reward system"""
    
    # Load the base config
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable reward v2 system
    config['env']['use_reward_v2'] = True
    
    # Add reward v2 configuration
    config['env']['reward_v2'] = {
        'components': {
            # Foundational rewards
            'realized_pnl': {
                'enabled': True,
                'weight': 1.0,
                'clip_min': -10.0,
                'clip_max': 10.0
            },
            'mark_to_market': {
                'enabled': True,
                'weight': 0.5,
                'max_leverage': 2.0,
                'clip_min': -5.0,
                'clip_max': 5.0
            },
            'differential_sharpe': {
                'enabled': True,
                'weight': 0.3,
                'window_size': 20,
                'min_periods': 5,
                'sharpe_scale': 0.1
            },
            
            # Shaping rewards
            'holding_time_penalty': {
                'enabled': True,
                'weight': 1.0,
                'max_holding_time': 60,
                'penalty_per_step': 0.001,
                'progressive_penalty': True
            },
            'overtrading_penalty': {
                'enabled': True,
                'weight': 1.0,
                'lookback_trades': 20,
                'frequency_window': 100,
                'max_trades_per_window': 5,
                'penalty_per_excess_trade': 0.01,
                'exponential_penalty': True
            },
            'quick_profit_incentive': {
                'enabled': True,
                'weight': 1.0,
                'quick_profit_time': 30,
                'bonus_rate': 0.5
            },
            'drawdown_penalty': {
                'enabled': True,
                'weight': 1.0,
                'warning_threshold': 0.02,
                'severe_threshold': 0.05,
                'base_penalty': 0.01
            },
            
            # Trade-specific penalties
            'mae_penalty': {
                'enabled': True,
                'weight': 1.0,
                'mae_threshold': 0.02,
                'base_penalty': 0.1,
                'loss_multiplier': 1.5
            },
            'mfe_penalty': {
                'enabled': True,
                'weight': 1.0,
                'give_back_threshold': 0.5,
                'base_penalty': 0.05,
                'reversal_multiplier': 2.0
            },
            
            # Terminal penalties
            'terminal_penalty': {
                'enabled': True,
                'weight': 1.0,
                'bankruptcy_penalty': 100.0,
                'max_loss_penalty': 50.0,
                'default_penalty': 10.0
            }
        },
        
        'aggregator': {
            'global_scale': 0.01,
            'use_smoothing': True,
            'smoothing_window': 10
        }
    }
    
    # Save test config
    test_config_path = Path("config/test_reward_v2.yaml")
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Test configuration saved to: {test_config_path}")
    return config


def run_quick_test():
    """Run a quick test with the new reward system"""
    # This would normally be integrated into your training script
    print("\n=== Testing Reward System V2 ===\n")
    
    # Key changes to make in your training script:
    print("To use the new reward system in your training:")
    print("1. Set env.use_reward_v2: true in your config")
    print("2. Add the reward_v2 configuration section")
    print("3. The environment will automatically use RewardSystemV2")
    print("4. Reward components will be tracked and sent to W&B")
    print("5. Dashboard will display component breakdown")
    
    print("\nKey benefits:")
    print("- Comprehensive reward shaping with 10+ components")
    print("- Anti-hacking measures (clipping, smoothing, decay)")
    print("- Detailed metrics for each component")
    print("- MAE/MFE tracking for trade quality")
    print("- Progressive penalties to shape behavior")
    
    print("\nComponent types:")
    print("- Foundational: PnL, MTM, Sharpe")
    print("- Shaping: Holding time, overtrading, quick profit, drawdown")
    print("- Trade-specific: MAE, MFE penalties")
    print("- Terminal: Bankruptcy, max loss")


if __name__ == "__main__":
    # Create test configuration
    config = create_test_config()
    
    # Run quick test
    run_quick_test()
    
    print("\n\nTo run training with the new reward system:")
    print("poetry run python scripts/run.py train --config config/test_reward_v2.yaml")