"""
Domain-specific WandB metrics callbacks for FxAI.

This module provides specialized callbacks for different system components:
- PPO/Training metrics (episodes, rewards, losses)
- Execution metrics (fills, slippage, latency)
- Portfolio metrics (positions, PnL, drawdowns)
- Model metrics (attention, gradients, activations)
- Session metrics (runtime, totals, performance)

Each callback maintains minimal local buffers for rolling calculations
while streaming all data to Weights & Biases for storage and visualization.
"""

from .ppo_metrics_callback import PPOMetricsCallback
from .execution_metrics_callback import ExecutionMetricsCallback
from .portfolio_metrics_callback import PortfolioMetricsCallback
from .model_metrics_callback import ModelMetricsCallback
from .session_metrics_callback import SessionMetricsCallback

__all__ = [
    'PPOMetricsCallback',
    'ExecutionMetricsCallback', 
    'PortfolioMetricsCallback',
    'ModelMetricsCallback',
    'SessionMetricsCallback'
]

def create_wandb_callbacks(
    ppo_buffer_size: int = 1000,
    execution_buffer_size: int = 2000,
    portfolio_buffer_size: int = 1000,
    model_buffer_size: int = 500,
    enable_ppo: bool = True,
    enable_execution: bool = True,
    enable_portfolio: bool = True,
    enable_model: bool = True,
    enable_session: bool = True
):
    """
    Create all WandB metric callbacks with specified configurations.
    
    Args:
        ppo_buffer_size: Buffer size for episode/training metrics
        execution_buffer_size: Buffer size for execution metrics (higher frequency)
        portfolio_buffer_size: Buffer size for portfolio metrics
        model_buffer_size: Buffer size for model internals
        enable_*: Whether to enable each callback type
        
    Returns:
        List of enabled callbacks
    """
    callbacks = []
    
    if enable_ppo:
        callbacks.append(PPOMetricsCallback(buffer_size=ppo_buffer_size))
    
    if enable_execution:
        callbacks.append(ExecutionMetricsCallback(buffer_size=execution_buffer_size))
    
    if enable_portfolio:
        callbacks.append(PortfolioMetricsCallback(buffer_size=portfolio_buffer_size))
    
    if enable_model:
        callbacks.append(ModelMetricsCallback(buffer_size=model_buffer_size))
    
    if enable_session:
        callbacks.append(SessionMetricsCallback())
    
    return callbacks