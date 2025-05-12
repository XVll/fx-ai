#!/usr/bin/env python
# main.py - Main entry point for the trading environment
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse

# Import our components
from data.data_manager import DataManager
from data.provider.data_bento.databento_file_provider import DabentoFileProvider
from feature.feature_extractor import FeatureExtractor
from simulation.simulator import Simulator
from simulation.market_simulator import MarketSimulator
from simulation.execution_simulator import ExecutionSimulator
from simulation.portfolio_simulator import PortfolioSimulator
from envs.trading_env import TradingEnv, MomentumTradingReward
from visualization.trade_visualizer import TradeVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_test.log")
    ]
)
logger = logging.getLogger(__name__)


def state_update_callback(state):
    """Example callback for state updates."""
    position = state.get('current_position', 0)
    pnl = state.get('unrealized_pnl', 0)
    logger.debug(f"State update: Position={position:.2f}, PnL={pnl:.2f}%")


def trade_callback(trade):
    """Example callback for completed trades."""
    logger.info(f"Trade completed: {trade.get('realized_pnl', 0):.2f}% PnL, Side: {trade.get('action', '')}")


def portfolio_update_callback(portfolio_state):
    """Example callback for portfolio updates."""
    cash = portfolio_state.get('cash', 0)
    total_value = portfolio_state.get('total_value', 0)
    logger.debug(f"Portfolio update: Cash=${cash:.2f}, Value=${total_value:.2f}")


def run_random_agent_test(env, num_episodes=5, max_steps=100):
    """
    Run a random agent to test the environment.

    Args:
        env: Trading environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        List of episode results
    """
    episode_results = []

    for episode in range(num_episodes):
        logger.info(f"Starting episode {episode + 1}/{num_episodes}")

        # Reset environment
        state, info = env.reset()

        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # Take random action
            action = np.random.uniform(-1, 1, size=(1,))

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            logger.debug(f"Step {step_count + 1}: Action={action[0]:.4f}, Reward={reward:.4f}")

            total_reward += reward
            step_count += 1

            # Print progress every 20 steps
            if step_count % 20 == 0:
                logger.info(f"Episode {episode + 1}, Step {step_count}: Total Reward={total_reward:.4f}")

        # Episode summary
        logger.info(f"Episode {episode + 1} finished: Steps={step_count}, Total Reward={total_reward:.4f}")

        # Get episode metrics
        if 'episode' in info:
            episode_info = info['episode']
            logger.info(f"Episode metrics: PnL=${episode_info['total_pnl']:.2f}, "
                        f"Win Rate={episode_info['win_rate']:.1%}, "
                        f"Trade Count={episode_info['trade_count']}")

            episode_results.append({
                'episode': episode + 1,
                'steps': step_count,
                'reward': total_reward,
                'pnl': episode_info['total_pnl'],
                'win_rate': episode_info['win_rate'],
                'trade_count': episode_info['trade_count'],
                'max_drawdown': episode_info['max_drawdown_pct']
            })

    return episode_results


def run_momentum_strategy_test(env, num_episodes=5, max_steps=100):
    """
    Run a simple momentum strategy to test the environment.

    Args:
        env: Trading environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode

    Returns:
        List of episode results
    """
    episode_results = []

    for episode in range(num_episodes):
        logger.info(f"Starting episode {episode + 1}/{num_episodes}")

        # Reset environment
        state, info = env.reset()

        total_reward = 0
        done = False
        step_count = 0

        # Strategy state
        in_position = False
        entry_price = 0
        position_steps = 0
        prev_prices = []

        while not done and step_count < max_steps:
            # Get current market state
            market_state = env.simulator.get_market_state()
            current_price = market_state.get('price', 0)
            tape_imbalance = market_state.get('tape_imbalance', 0)

            # Keep track of recent prices
            prev_prices.append(current_price)
            if len(prev_prices) > 10:
                prev_prices.pop(0)

            # Simple momentum strategy
            if len(prev_prices) >= 10:
                # Calculate short-term momentum
                short_term_return = (current_price / prev_prices[-5] - 1) * 100  # 5-step return

                # Calculate trend strength
                trend = (current_price - prev_prices[0]) / prev_prices[0] * 100  # Full window trend

                # Determine action based on momentum and tape
                if not in_position:
                    # Entry logic
                    if short_term_return > 0.2 and tape_imbalance > 0.3 and trend > 0:
                        # Strong momentum, positive tape, uptrend - enter long
                        action = np.array([0.75])  # 75% of max position
                        in_position = True
                        entry_price = current_price
                        position_steps = 0
                        logger.info(
                            f"ENTRY at ${current_price:.4f}, Momentum={short_term_return:.2f}%, Tape={tape_imbalance:.2f}")
                    else:
                        action = np.array([0.0])  # Stay flat
                else:
                    # Exit logic
                    position_steps += 1

                    # Calculate position P&L
                    pnl_pct = (current_price / entry_price - 1) * 100

                    # Exit conditions
                    if pnl_pct > 1.0:
                        # Take profit at 1%
                        action = np.array([0.0])  # Flat
                        in_position = False
                        logger.info(f"PROFIT TAKE at ${current_price:.4f}, P&L={pnl_pct:.2f}%")
                    elif pnl_pct < -0.5:
                        # Stop loss at -0.5%
                        action = np.array([0.0])  # Flat
                        in_position = False
                        logger.info(f"STOP LOSS at ${current_price:.4f}, P&L={pnl_pct:.2f}%")
                    elif tape_imbalance < -0.3 and short_term_return < 0:
                        # Momentum reversal - exit
                        action = np.array([0.0])  # Flat
                        in_position = False
                        logger.info(f"REVERSAL EXIT at ${current_price:.4f}, P&L={pnl_pct:.2f}%")
                    elif position_steps > 20:
                        # Time-based exit
                        action = np.array([0.0])  # Flat
                        in_position = False
                        logger.info(f"TIME EXIT at ${current_price:.4f}, P&L={pnl_pct:.2f}%")
                    else:
                        # Hold position
                        action = np.array([0.75])  # Maintain position
            else:
                # Not enough price history
                action = np.array([0.0])  # Stay flat

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            # Print progress every 20 steps
            if step_count % 20 == 0:
                logger.info(f"Episode {episode + 1}, Step {step_count}: Total Reward={total_reward:.4f}")

        # Episode summary
        logger.info(f"Episode {episode + 1} finished: Steps={step_count}, Total Reward={total_reward:.4f}")

        # Get episode metrics
        if 'episode' in info:
            episode_info = info['episode']
            logger.info(f"Episode metrics: PnL=${episode_info['total_pnl']:.2f}, "
                        f"Win Rate={episode_info['win_rate']:.1%}, "
                        f"Trade Count={episode_info['trade_count']}")

            episode_results.append({
                'episode': episode + 1,
                'steps': step_count,
                'reward': total_reward,
                'pnl': episode_info['total_pnl'],
                'win_rate': episode_info['win_rate'],
                'trade_count': episode_info['trade_count'],
                'max_drawdown': episode_info['max_drawdown_pct']
            })

    return episode_results


def visualize_results(simulator, visualizer, symbol, date_str):
    """
    Visualize the simulation results.

    Args:
        simulator: Simulator instance
        visualizer: TradeVisualizer instance
        symbol: Symbol being traded
        date_str: Date string
    """
    # Get trade history
    trades = simulator.portfolio_simulator.get_trade_history()

    # Get portfolio history
    portfolio_history = simulator.portfolio_simulator.get_portfolio_history()

    # Get price data - use 1m bars
    price_data = simulator.raw_data.get('bars_1m', pd.DataFrame())

    if price_data.empty:
        logger.warning("No price data available for visualization")
        return

    if trades:
        # Convert trades to format expected by visualizer
        vis_trades = []
        for trade in trades:
            if 'open_time' in trade and 'close_time' in trade:
                vis_trades.append(trade)
            elif 'symbol' in trade:
                # Portfolio simulator format
                vis_trade = {
                    'symbol': trade['symbol'],
                    'open_time': trade['open_time'],
                    'close_time': trade['close_time'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'realized_pnl': trade['realized_pnl'],
                    'quantity': trade['quantity']
                }
                vis_trades.append(vis_trade)

        # Plot price chart with trades
        visualizer.plot_price_chart_with_trades(
            price_data=price_data,
            trades=vis_trades,
            title=f"{symbol} Price Chart with Trades - {date_str}",
            save_filename=f"{symbol}_price_chart_{date_str}.png"
        )

        # Plot portfolio performance
        if not portfolio_history.empty:
            visualizer.plot_portfolio_performance(
                portfolio_history=portfolio_history,
                trades=vis_trades,
                title=f"Portfolio Performance - {date_str}",
                save_filename=f"{symbol}_portfolio_{date_str}.png"
            )

        # Plot trade analysis
        visualizer.plot_trade_analysis(
            trades=vis_trades,
            title=f"Trade Analysis - {symbol} {date_str}",
            save_filename=f"{symbol}_trade_analysis_{date_str}.png"
        )

        # Plot trade metrics
        visualizer.plot_trade_metrics(
            trades=vis_trades,
            title=f"Trade Metrics - {symbol} {date_str}",
            save_filename=f"{symbol}_trade_metrics_{date_str}.png"
        )

        # Multi-timeframe view if we have data for multiple timeframes
        data_dict = {}
        for tf in ['1s', '1m', '5m', '1d']:
            key = f'bars_{tf}'
            if key in simulator.raw_data and not simulator.raw_data[key].empty:
                data_dict[tf] = simulator.raw_data[key]

        if len(data_dict) > 1:
            timeframes = list(data_dict.keys())
            visualizer.plot_multi_timeframe_view(
                data_dict=data_dict,
                timeframes=timeframes,
                trades=vis_trades,
                title=f"Multi-Timeframe Analysis - {symbol} {date_str}",
                save_filename=f"{symbol}_multi_timeframe_{date_str}.png"
            )

        # Tape analysis if we have trade data
        if 'trades' in simulator.raw_data and not simulator.raw_data['trades'].empty:
            # Extract trade events from simulator
            trade_events = []
            for action_result in simulator.execution_simulator.order_history:
                if action_result.get('status') == 'filled':
                    event = {
                        'timestamp': action_result.get('update_timestamp'),
                        'action': action_result.get('side'),
                        'fill_price': action_result.get('avg_fill_price'),
                        'quantity': action_result.get('filled_quantity')
                    }
                    trade_events.append(event)

            visualizer.plot_tape_analysis(
                trades_df=simulator.raw_data['trades'],
                price_data=price_data,
                trade_events=trade_events,
                title=f"Tape Analysis - {symbol} {date_str}",
                save_filename=f"{symbol}_tape_analysis_{date_str}.png"
            )


def main():
    # Default configuration parameters
    config = {
        'data_dir': "./dnb/mlgo",
        'symbol': "MLGO",
        'date': "2025-03-27",
        'episodes': 3,
        'max_steps': 1000,
        'strategy': "momentum",
        'visualize': False,
        'debug': True  # Enable debug mode by default
    }

    # Optionally parse command line arguments
    try:
        parser = argparse.ArgumentParser(description="Test Trading Environment")
        parser.add_argument("--data_dir", type=str, default=config['data_dir'],
                            help=f"Path to data directory (default: {config['data_dir']})")
        parser.add_argument("--symbol", type=str, default=config['symbol'],
                            help=f"Symbol to test (default: {config['symbol']})")
        parser.add_argument("--date", type=str, default=config['date'],
                            help=f"Date to test (YYYY-MM-DD) (default: {config['date']})")
        parser.add_argument("--episodes", type=int, default=config['episodes'],
                            help=f"Number of episodes to run (default: {config['episodes']})")
        parser.add_argument("--max_steps", type=int, default=config['max_steps'],
                            help=f"Maximum steps per episode (default: {config['max_steps']})")
        parser.add_argument("--strategy", type=str, default=config['strategy'],
                            choices=["random", "momentum"],
                            help=f"Strategy to test (default: {config['strategy']})")
        parser.add_argument("--visualize", action="store_true", default=config['visualize'],
                            help="Visualize results")
        parser.add_argument("--debug", action="store_true", default=config['debug'],
                            help="Enable debug logging")

        args = parser.parse_args()

        # Update config with command line arguments
        config.update(vars(args))
    except:
        # If argument parsing fails, just use the defaults
        logger.info("Using default configuration")

    # Set logging level based on debug flag
    log_level = logging.DEBUG if config['debug'] else logging.INFO
    logger.setLevel(log_level)

    # Validate the data directory exists
    if not os.path.exists(config['data_dir']):
        logger.error(f"Data directory {config['data_dir']} does not exist. Please check the path.")
        return

    # Create provider, data manager, and simulator
    provider = DabentoFileProvider(config['data_dir'], verbose=config['debug'])
    data_manager = DataManager(provider, logger=logger)

    # Configure simulators
    market_config = {
        'min_spread_pct': 0.001,
        'slippage_factor': 0.5,
        'latency_ms': 250,
        'luld_enabled': True
    }

    execution_config = {
        'random_fill_failure_pct': 0.01
    }

    portfolio_config = {
        'initial_cash': 100000.0,
        'max_position_pct': 0.5,
        'max_drawdown_pct': 0.05,
        'position_size_limits': {config['symbol']: 1000}
    }

    sim_config = {
        'market_config': market_config,
        'execution_config': execution_config,
        'portfolio_config': portfolio_config
    }

    simulator = Simulator(data_manager, sim_config, logger=logger)

    # Add callbacks
    simulator.add_state_update_callback(state_update_callback)
    simulator.add_trade_callback(trade_callback)
    simulator.add_portfolio_update_callback(portfolio_update_callback)

    # Initialize simulator
    logger.info(f"Initializing simulator for {config['symbol']} on {config['date']}")
    success = simulator.initialize_for_symbol(
        config['symbol'], mode='backtesting',
        start_time=config['date'], end_time=config['date'],
        timeframes=["1s", "1m", "5m", "1d"]
    )

    if not success:
        logger.error("Failed to initialize simulator")
        return

    # Verify data was loaded properly
    if simulator.raw_data:
        for key, df in simulator.raw_data.items():
            if df.empty:
                logger.warning(f"No data loaded for {key}")
            else:
                logger.info(f"Loaded {len(df)} rows of {key} data")
                if config['debug']:
                    # Show first few rows for debugging
                    logger.debug(f"First 3 rows of {key} data:\n{df.head(3)}")
    else:
        logger.error("No data was loaded!")
        return

    # Verify features were extracted correctly
    if simulator.current_symbol in simulator.features_cache:
        features_df = simulator.features_cache[simulator.current_symbol]
        if features_df.empty:
            logger.warning("No features extracted!")
        else:
            logger.info(f"Extracted {len(features_df)} feature rows with {len(features_df.columns)} columns")
            if config['debug']:
                # Show feature timestamps for debugging
                logger.debug(f"First 5 feature timestamps: {features_df.index[:5].tolist()}")
                logger.debug(f"Last 5 feature timestamps: {features_df.index[-5:].tolist()}")
                logger.debug(f"Features include: {features_df.columns[:10].tolist()}")
    else:
        logger.error("No features cache for current symbol!")
        return

    # Configure environment
    env_config = {
        'random_reset': False,
        'state_dim': 300,
        'window_size': 30,
        'max_steps': config['max_steps'],
        'normalize_state': True,
        'normalize_reward': True,
        'early_stop_pct': -0.05
    }

    # Create custom reward function for momentum trading
    reward_fn = MomentumTradingReward()

    # Create environment
    env = TradingEnv(simulator, env_config, reward_function=reward_fn, logger=logger)

    # Run selected strategy
    if config['strategy'] == "random":
        logger.info("Running random agent test")
        results = run_random_agent_test(env, config['episodes'], config['max_steps'])
    else:
        logger.info("Running momentum strategy test")
        results = run_momentum_strategy_test(env, config['episodes'], config['max_steps'])

    # Print results summary
    logger.info("Results Summary:")
    for res in results:
        logger.info(f"Episode {res['episode']}: Reward={res['reward']:.2f}, PnL=${res['pnl']:.2f}, "
                    f"Win Rate={res['win_rate']:.1%}, Trades={res['trade_count']}")

    avg_pnl = sum(r['pnl'] for r in results) / len(results) if results else 0
    avg_win_rate = sum(r['win_rate'] for r in results) / len(results) if results else 0
    avg_trades = sum(r['trade_count'] for r in results) / len(results) if results else 0

    logger.info(f"Average PnL: ${avg_pnl:.2f}")
    logger.info(f"Average Win Rate: {avg_win_rate:.1%}")
    logger.info(f"Average Trades per Episode: {avg_trades:.1f}")

    # Visualize results if requested
    if config['visualize']:
        logger.info("Visualizing results")
        visualizer = TradeVisualizer(save_path="charts", logger=logger)
        visualize_results(simulator, visualizer, config['symbol'], config['date'])

    logger.info("Test completed successfully")


if __name__ == "__main__":
    main()