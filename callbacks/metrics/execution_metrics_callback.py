"""
Execution metrics callback for WandB integration.

Tracks order execution metrics including fills, slippage, latency,
and execution quality statistics.
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Optional

try:
    import wandb
except ImportError:
    wandb = None

from callbacks.core.base import BaseCallback

logger = logging.getLogger(__name__)


class ExecutionMetricsCallback(BaseCallback):
    """
    Specialized callback for execution metrics with high-frequency data handling.
    
    Tracks:
    - Order fills: prices, sizes, slippage
    - Execution quality: latency, fill rates, market impact
    - Rolling calculations: average slippage, execution costs
    """
    
    def __init__(self, buffer_size: int = 2000, enabled: bool = True):
        """
        Initialize execution metrics callback.
        
        Args:
            buffer_size: Size of local buffers (higher for execution frequency)
            enabled: Whether callback is active
        """
        super().__init__(name="ExecutionMetrics", enabled=enabled)
        
        self.buffer_size = buffer_size
        
        # Execution buffers
        self.slippages = deque(maxlen=buffer_size)
        self.fill_prices = deque(maxlen=buffer_size)
        self.requested_prices = deque(maxlen=buffer_size)
        self.order_sizes = deque(maxlen=buffer_size)
        self.latencies = deque(maxlen=buffer_size)
        self.execution_costs = deque(maxlen=buffer_size)
        
        # Market impact tracking
        self.market_impacts = deque(maxlen=buffer_size)
        self.bid_ask_spreads = deque(maxlen=buffer_size)
        
        # Order status tracking
        self.fill_rates = deque(maxlen=1000)  # Track fill success
        self.partial_fills = deque(maxlen=1000)
        self.rejected_orders = deque(maxlen=1000)
        
        # Performance tracking
        self.fills_logged = 0
        self.orders_logged = 0
        self.total_volume = 0
        
        if wandb is None:
            self.logger.warning("wandb not installed - execution metrics will not be logged")
        
        self.logger.info(f"⚡ Execution metrics callback initialized (buffer_size={buffer_size})")
    
    def on_custom_event(self, event_name: str, context: Dict[str, Any]) -> None:
        """Handle execution-related custom events."""
        if not wandb or not wandb.run:
            return
        
        if event_name == 'execution_fill':
            self._handle_execution_fill(context)
        elif event_name == 'order_submitted':
            self._handle_order_submitted(context)
        elif event_name == 'order_rejected':
            self._handle_order_rejected(context)
        elif event_name == 'partial_fill':
            self._handle_partial_fill(context)
    
    def _handle_execution_fill(self, context: Dict[str, Any]) -> None:
        """Handle order fill events."""
        raw = context.get('raw', {})
        
        # Extract fill data
        fill_price = raw.get('fill_price', 0)
        requested_price = raw.get('requested_price', fill_price)
        order_size = raw.get('size', 0)
        latency_ms = raw.get('latency_ms', 0)
        execution_cost = raw.get('execution_cost', 0)
        bid_ask_spread = raw.get('bid_ask_spread', 0)
        
        # Calculate slippage
        slippage = abs(fill_price - requested_price) if requested_price != 0 else 0
        slippage_bps = (slippage / requested_price * 10000) if requested_price != 0 else 0
        
        # Calculate market impact
        market_impact = raw.get('market_impact', 0)
        
        # Add to buffers
        self.slippages.append(slippage_bps)
        self.fill_prices.append(fill_price)
        self.requested_prices.append(requested_price)
        self.order_sizes.append(order_size)
        self.latencies.append(latency_ms)
        self.execution_costs.append(execution_cost)
        self.market_impacts.append(market_impact)
        self.bid_ask_spreads.append(bid_ask_spread)
        
        # Update totals
        self.total_volume += order_size
        
        # Prepare metrics
        metrics = {
            # Raw execution metrics
            'execution/fill_price': fill_price,
            'execution/requested_price': requested_price,
            'execution/slippage_bps': slippage_bps,
            'execution/order_size': order_size,
            'execution/latency_ms': latency_ms,
            'execution/execution_cost': execution_cost,
            'execution/market_impact': market_impact,
            'execution/bid_ask_spread': bid_ask_spread,
            'execution/total_volume': self.total_volume,
        }
        
        # Add rolling metrics
        self._add_rolling_execution_metrics(metrics)
        
        # Log to WandB
        wandb.log(metrics)
        
        self.fills_logged += 1
        
        if self.fills_logged % 100 == 0:
            self.logger.debug(f"⚡ Logged {self.fills_logged} fills to WandB")
    
    def _handle_order_submitted(self, context: Dict[str, Any]) -> None:
        """Handle order submission events."""
        raw = context.get('raw', {})
        
        order_type = raw.get('order_type', 'unknown')
        order_side = raw.get('order_side', 'unknown')
        order_size = raw.get('size', 0)
        limit_price = raw.get('limit_price', 0)
        
        self.orders_logged += 1
        
        # Log order submission
        metrics = {
            'orders/submitted': 1,
            'orders/total_submitted': self.orders_logged,
            'orders/order_type': order_type,
            'orders/order_side': order_side,
            'orders/order_size': order_size,
            'orders/limit_price': limit_price
        }
        
        wandb.log(metrics)
    
    def _handle_order_rejected(self, context: Dict[str, Any]) -> None:
        """Handle order rejection events."""
        raw = context.get('raw', {})
        
        rejection_reason = raw.get('rejection_reason', 'unknown')
        order_size = raw.get('size', 0)
        
        self.rejected_orders.append(1)
        
        # Calculate rejection rate
        rejection_rate = 0
        if len(self.rejected_orders) >= 10:
            rejection_rate = np.mean(list(self.rejected_orders)[-100:])
        
        metrics = {
            'orders/rejected': 1,
            'orders/rejection_reason': rejection_reason,
            'orders/rejection_rate_100': rejection_rate,
            'orders/rejected_size': order_size
        }
        
        wandb.log(metrics)
    
    def _handle_partial_fill(self, context: Dict[str, Any]) -> None:
        """Handle partial fill events."""
        raw = context.get('raw', {})
        
        filled_size = raw.get('filled_size', 0)
        remaining_size = raw.get('remaining_size', 0)
        total_size = filled_size + remaining_size
        
        fill_percentage = (filled_size / total_size * 100) if total_size > 0 else 0
        
        self.partial_fills.append(fill_percentage)
        
        metrics = {
            'execution/partial_fill': 1,
            'execution/fill_percentage': fill_percentage,
            'execution/filled_size': filled_size,
            'execution/remaining_size': remaining_size
        }
        
        # Add average fill percentage
        if len(self.partial_fills) >= 10:
            metrics['execution/avg_fill_percentage_50'] = np.mean(list(self.partial_fills)[-50:])
        
        wandb.log(metrics)
    
    def _add_rolling_execution_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add rolling execution metrics to the metrics dict."""
        # 50-fill rolling metrics
        if len(self.slippages) >= 50:
            recent_50_slippage = list(self.slippages)[-50:]
            recent_50_latency = list(self.latencies)[-50:]
            recent_50_costs = list(self.execution_costs)[-50:]
            recent_50_sizes = list(self.order_sizes)[-50:]
            
            metrics.update({
                'rolling_50/avg_slippage_bps': np.mean(recent_50_slippage),
                'rolling_50/slippage_std_bps': np.std(recent_50_slippage),
                'rolling_50/avg_latency_ms': np.mean(recent_50_latency),
                'rolling_50/avg_execution_cost': np.mean(recent_50_costs),
                'rolling_50/avg_order_size': np.mean(recent_50_sizes),
                'rolling_50/total_volume_50': np.sum(recent_50_sizes)
            })
        
        # 100-fill rolling metrics
        if len(self.slippages) >= 100:
            recent_100_slippage = list(self.slippages)[-100:]
            recent_100_latency = list(self.latencies)[-100:]
            recent_100_impacts = list(self.market_impacts)[-100:]
            recent_100_spreads = list(self.bid_ask_spreads)[-100:]
            
            metrics.update({
                'rolling_100/avg_slippage_bps': np.mean(recent_100_slippage),
                'rolling_100/slippage_p95_bps': np.percentile(recent_100_slippage, 95),
                'rolling_100/slippage_p99_bps': np.percentile(recent_100_slippage, 99),
                'rolling_100/avg_latency_ms': np.mean(recent_100_latency),
                'rolling_100/latency_p95_ms': np.percentile(recent_100_latency, 95),
                'rolling_100/avg_market_impact': np.mean(recent_100_impacts),
                'rolling_100/avg_bid_ask_spread': np.mean(recent_100_spreads)
            })
        
        # 500-fill rolling metrics (execution quality)
        if len(self.slippages) >= 500:
            recent_500_slippage = list(self.slippages)[-500:]
            recent_500_costs = list(self.execution_costs)[-500:]
            
            # Calculate execution quality score (lower is better)
            execution_quality = np.mean(recent_500_slippage) + np.mean(recent_500_costs)
            
            metrics.update({
                'rolling_500/avg_slippage_bps': np.mean(recent_500_slippage),
                'rolling_500/total_execution_cost': np.sum(recent_500_costs),
                'rolling_500/execution_quality_score': execution_quality,
                'rolling_500/slippage_consistency': 1 / (1 + np.std(recent_500_slippage))  # Higher is better
            })
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Log episode-level execution summaries."""
        if not wandb or not wandb.run or len(self.slippages) == 0:
            return
        
        # Calculate episode execution summary
        episode_fills = context.get('episode_fills', len(self.slippages))
        episode_volume = context.get('episode_volume', 0)
        
        if episode_fills > 0:
            # Get execution metrics for this episode (approximate)
            recent_slippages = list(self.slippages)[-episode_fills:] if episode_fills <= len(self.slippages) else list(self.slippages)
            recent_latencies = list(self.latencies)[-episode_fills:] if episode_fills <= len(self.latencies) else list(self.latencies)
            
            episode_metrics = {
                'episode_execution/total_fills': episode_fills,
                'episode_execution/total_volume': episode_volume,
                'episode_execution/avg_slippage_bps': np.mean(recent_slippages) if recent_slippages else 0,
                'episode_execution/avg_latency_ms': np.mean(recent_latencies) if recent_latencies else 0,
                'episode_execution/max_slippage_bps': np.max(recent_slippages) if recent_slippages else 0
            }
            
            wandb.log(episode_metrics)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get callback statistics."""
        return {
            'buffer_size': self.buffer_size,
            'fills_logged': self.fills_logged,
            'orders_logged': self.orders_logged,
            'total_volume': self.total_volume,
            'slippages_buffer_size': len(self.slippages),
            'latencies_buffer_size': len(self.latencies),
            'avg_slippage_bps': np.mean(list(self.slippages)) if self.slippages else 0,
            'avg_latency_ms': np.mean(list(self.latencies)) if self.latencies else 0
        }