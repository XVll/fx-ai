from typing import Dict, Any, Optional
import logging
from .live_dashboard_v2 import LiveTradingDashboard

logger = logging.getLogger(__name__)


class DashboardMetricsCollector:
    """Integrates the live dashboard with the metrics system"""
    
    def __init__(self, dashboard: Optional[LiveTradingDashboard] = None):
        self.dashboard = dashboard or LiveTradingDashboard()
        self.is_started = False
        self.current_episode_num = 0
        
    def start(self, open_browser: bool = True):
        """Start the dashboard"""
        if not self.is_started:
            self.dashboard.start(open_browser=open_browser)
            self.is_started = True
            logger.info("Dashboard metrics collector started")
    
    def stop(self):
        """Stop the dashboard"""
        if self.is_started:
            self.dashboard.stop()
            self.is_started = False
    
    def on_step(self, step_data: Dict[str, Any]):
        """Handle environment step update"""
        if not self.is_started:
            return
        
        # Update market data
        market_update = {
            'symbol': step_data.get('symbol', 'N/A'),
            'price': step_data.get('price', 0),
            'bid': step_data.get('bid', step_data.get('price', 0) * 0.9999),
            'ask': step_data.get('ask', step_data.get('price', 0) * 1.0001),
            'volume': step_data.get('volume', 0),
        }
        
        # Add timestamp from market state if available
        if 'market_state' in step_data and step_data['market_state']:
            market_state = step_data['market_state']
            if isinstance(market_state, dict) and 'timestamp_utc' in market_state:
                market_update['timestamp'] = market_state['timestamp_utc']
                
        self.dashboard.update_market(market_update)
        
        # Load full day 1m bars data if available from market state (do this once per day)
        if 'market_state' in step_data and step_data['market_state']:
            market_state = step_data['market_state']
            symbol = step_data.get('symbol', 'N/A')
            current_date = market_update.get('timestamp')
            
            # Extract date for comparison
            date_key = None
            if current_date:
                from datetime import datetime
                if hasattr(current_date, 'date'):
                    date_key = current_date.date()
                elif isinstance(current_date, str):
                    try:
                        parsed_date = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
                        date_key = parsed_date.date()
                    except:
                        pass
            
            # Check if we need to load full day data (new day or new symbol)
            if (date_key and symbol and 
                (self.dashboard.state.full_day_date != date_key or 
                 self.dashboard.state.full_day_symbol != symbol)):
                
                # Get full day 1m bars from market state if available
                if 'full_day_1m_bars' in market_state and market_state['full_day_1m_bars']:
                    full_bars = market_state['full_day_1m_bars']
                    if isinstance(full_bars, list):
                        self.dashboard.state.full_day_1m_bars = []
                        for bar in full_bars:
                            if bar and isinstance(bar, dict) and all(k in bar for k in ['open', 'high', 'low', 'close']):
                                ohlc_bar = {
                                    'timestamp': bar.get('timestamp'),
                                    'open': float(bar['open']),
                                    'high': float(bar['high']),
                                    'low': float(bar['low']),
                                    'close': float(bar['close']),
                                    'volume': float(bar.get('volume', 0))
                                }
                                self.dashboard.state.full_day_1m_bars.append(ohlc_bar)
                        
                        # Replace incremental ohlc_data with full day data
                        self.dashboard.state.ohlc_data.clear()
                        self.dashboard.state.ohlc_data.extend(self.dashboard.state.full_day_1m_bars)
                        
                        # Update tracking variables
                        self.dashboard.state.full_day_date = date_key
                        self.dashboard.state.full_day_symbol = symbol
                        logger.info(f"Loaded {len(self.dashboard.state.full_day_1m_bars)} 1m bars for {symbol} on {date_key}")
                
                # Fallback to incremental update if full day data not available
                elif '1m_bars_window' in market_state:
                    bars_window = market_state.get('1m_bars_window', [])
                    if bars_window and isinstance(bars_window, list):
                        # Process all bars in the window (only add new ones)
                        for bar in bars_window:
                            if bar and isinstance(bar, dict) and all(k in bar for k in ['open', 'high', 'low', 'close']):
                                bar_timestamp = bar.get('timestamp')
                                if bar_timestamp and bar_timestamp != self.dashboard.state.last_bar_timestamp:
                                    ohlc_bar = {
                                        'timestamp': bar_timestamp,
                                        'open': float(bar['open']),
                                        'high': float(bar['high']),
                                        'low': float(bar['low']),
                                        'close': float(bar['close']),
                                        'volume': float(bar.get('volume', 0))
                                    }
                                    self.dashboard.state.ohlc_data.append(ohlc_bar)
                                    self.dashboard.state.last_bar_timestamp = bar_timestamp
        
        # Update position
        position_update = {
            'symbol': step_data.get('symbol', 'N/A'),
            'quantity': step_data.get('position', 0),
            'avg_entry_price': step_data.get('avg_entry_price', step_data.get('price', 0)),
        }
        self.dashboard.update_position(position_update)
        
        # Update portfolio
        portfolio_update = {
            'equity': step_data.get('equity', 25000),
            'cash': step_data.get('cash', 25000),
            'realized_pnl': step_data.get('realized_pnl', 0),
            'unrealized_pnl': step_data.get('unrealized_pnl', 0),
        }
        self.dashboard.update_portfolio(portfolio_update)
        
        # Update action with invalid action tracking
        if 'action' in step_data and 'reward' in step_data:
            self.dashboard.update_action(
                step=step_data.get('step', 0),
                action_type=step_data.get('action', 'HOLD').upper(),
                size=step_data.get('size', 1.0),
                reward=step_data.get('reward', 0.0)
            )
            
        # Track invalid actions for footer metrics
        if step_data.get('invalid_action', False):
            self.dashboard.state.action_analysis.invalid_actions_count += 1
            
        # Update reward components if present
        if 'reward_components' in step_data and step_data['reward_components']:
            self.dashboard.update_reward_components(step_data['reward_components'])
    
    def on_trade(self, trade_data: Dict[str, Any]):
        """Handle completed trade"""
        if not self.is_started:
            return
        
        trade_update = {
            'side': trade_data.get('action', 'UNKNOWN').upper(),
            'quantity': abs(trade_data.get('quantity', 0)),
            'symbol': trade_data.get('symbol', 'N/A'),
            'entry_price': trade_data.get('entry_price', 0),
            'exit_price': trade_data.get('exit_price'),
            'pnl': trade_data.get('pnl', 0),
            'fees': trade_data.get('fees', 0),
            'commission': trade_data.get('commission', 0),
            'slippage': trade_data.get('slippage', 0)
        }
        self.dashboard.update_trade(trade_update)
        
    def on_execution(self, execution_data: Dict[str, Any]):
        """Handle execution (fill)"""
        if not self.is_started:
            return
            
        execution_update = {
            'side': execution_data.get('order_side', 'UNKNOWN').upper(),
            'quantity': execution_data.get('executed_quantity', 0),
            'symbol': execution_data.get('asset_id', 'N/A'),
            'price': execution_data.get('executed_price', 0),
            'commission': execution_data.get('commission', 0),
            'fees': execution_data.get('fees', 0),
            'slippage': execution_data.get('slippage_cost_total', 0)
        }
        
        # Add timestamp from execution data if available
        if 'timestamp' in execution_data:
            execution_update['timestamp'] = execution_data['timestamp']
        elif 'market_state' in execution_data and execution_data['market_state']:
            market_state = execution_data['market_state']
            if isinstance(market_state, dict) and 'timestamp_utc' in market_state:
                execution_update['timestamp'] = market_state['timestamp_utc']
                
        self.dashboard.state.add_execution(execution_update)
    
    def on_episode_start(self, episode_num: int):
        """Handle episode start"""
        if not self.is_started:
            return
        
        # Don't maintain our own counter, use the provided episode number
        self.current_episode_num = episode_num
        self.dashboard.start_episode(episode_num)
    
    def on_episode_end(self, episode_data: Dict[str, Any]):
        """Handle episode end"""
        if not self.is_started:
            return
        
        reason = episode_data.get('termination_reason', 'Completed')
        truncated = episode_data.get('truncated', False)
        self.dashboard.state.end_current_episode(reason, truncated)
        self.dashboard.end_episode(reason)
    
    def on_training_update(self, training_data: Dict[str, Any]):
        """Handle training progress updates"""
        if not self.is_started:
            return
        
        self.dashboard.update_training_progress(training_data)
    
    def on_ppo_metrics(self, ppo_data: Dict[str, Any]):
        """Handle PPO metrics updates"""
        if not self.is_started:
            return
        
        self.dashboard.update_ppo_metrics(ppo_data)
    
    def set_model_info(self, model_name: str):
        """Set model information"""
        if not self.is_started:
            return
        
        self.dashboard.set_model_info(model_name)
    
    def on_reward_components(self, reward_data: Dict[str, Any]):
        """Handle reward component updates"""
        if not self.is_started:
            return
            
        # Update reward components display
        if hasattr(self.dashboard.state, 'reward_components'):
            self.dashboard.state.reward_components = reward_data
        
        # Update dashboard with reward breakdown
        self.dashboard.update_reward_components(reward_data)