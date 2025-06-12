"""
Example usage of the WandB metrics callbacks system.

This demonstrates how to integrate the domain-specific callbacks
with your existing training loop and components.
"""

import wandb
import numpy as np
from callbacks.metrics import (
    PPOMetricsCallback,
    ExecutionMetricsCallback,
    PortfolioMetricsCallback,
    ModelMetricsCallback,
    SessionMetricsCallback,
    create_wandb_callbacks
)
from callbacks.core.callback_manager import CallbackManager


def example_training_integration():
    """Example of integrating WandB metrics with training loop."""
    
    # Initialize WandB run
    wandb.init(
        project="fx-trading-rl",
        name="ppo-momentum-training",
        config={
            "algorithm": "PPO",
            "learning_rate": 0.0003,
            "batch_size": 64,
            "total_episodes": 10000,
            "max_runtime_hours": 24
        }
    )
    
    # Create all metric callbacks
    metric_callbacks = create_wandb_callbacks(
        ppo_buffer_size=1000,
        execution_buffer_size=2000,
        portfolio_buffer_size=1000,
        model_buffer_size=500,
        enable_ppo=True,
        enable_execution=True,
        enable_portfolio=True,
        enable_model=True,
        enable_session=True
    )
    
    # Create callback manager and add metrics callbacks
    callback_manager = CallbackManager()
    for callback in metric_callbacks:
        callback_manager.add_callback(callback)
    
    # Start training
    callback_manager.trigger_training_start({
        'config': wandb.config,
        'total_episodes': 10000
    })
    
    # Simulate training loop
    for episode in range(1, 101):  # 100 episodes for demo
        
        # Start episode
        callback_manager.trigger_episode_start({
            'episode': episode,
            'symbol': 'AAPL',
            'current_episode': episode
        })
        
        # Simulate episode steps
        episode_reward = 0
        for step in range(50):  # 50 steps per episode
            
            # Simulate step
            step_reward = np.random.normal(0.1, 2.0)
            episode_reward += step_reward
            
            # Trigger step end
            callback_manager.trigger_step_end({
                'reward': step_reward,
                'action': np.random.choice([0, 1, 2]),
                'position': np.random.uniform(-1, 1),
                'unrealized_pnl': episode_reward,
                'portfolio_value': 100000 + episode_reward * 1000,
                'cash_balance': 50000,
                'margin_used': 25000
            })
            
            # Simulate custom events
            if step % 10 == 0:
                # Execution fill event
                callback_manager.trigger_custom_event('execution_fill', {
                    'raw': {
                        'fill_price': 100 + np.random.normal(0, 0.5),
                        'requested_price': 100,
                        'slippage': abs(np.random.normal(0, 0.01)),
                        'size': np.random.randint(100, 1000),
                        'latency_ms': np.random.gamma(2, 5),
                        'execution_cost': np.random.uniform(0.01, 0.05),
                        'bid_ask_spread': np.random.uniform(0.01, 0.03)
                    }
                })
            
            if step % 15 == 0:
                # Model forward event
                callback_manager.trigger_custom_event('model_forward', {
                    'raw': {
                        'attention_entropy': np.random.uniform(0.5, 2.0),
                        'attention_max_weight': np.random.uniform(0.1, 0.9),
                        'prediction_confidence': np.random.uniform(0.3, 0.9),
                        'action_entropy': np.random.uniform(0.2, 1.5),
                        'activation_magnitude': np.random.uniform(0.5, 3.0),
                        'activation_sparsity': np.random.uniform(0.1, 0.7),
                        'layer_activations': {
                            'encoder_0': np.random.uniform(0.5, 2.0),
                            'encoder_1': np.random.uniform(0.5, 2.0),
                            'decoder': np.random.uniform(0.5, 2.0)
                        }
                    }
                })
        
        # Simulate training update every 10 episodes
        if episode % 10 == 0:
            callback_manager.trigger_update_end({
                'policy_loss': np.random.uniform(0.01, 0.1),
                'value_loss': np.random.uniform(0.01, 0.1),
                'entropy_loss': np.random.uniform(0.001, 0.01),
                'loss': np.random.uniform(0.05, 0.2),
                'learning_rate': 0.0003 * (0.995 ** (episode // 10)),
                'kl_divergence': np.random.uniform(0.001, 0.01),
                'clip_fraction': np.random.uniform(0.1, 0.3),
                'grad_norm': np.random.uniform(0.5, 2.0),
                'explained_variance': np.random.uniform(0.1, 0.9)
            })
            
            # Model backward event
            callback_manager.trigger_custom_event('model_backward', {
                'raw': {
                    'gradient_norm': np.random.uniform(0.5, 2.0),
                    'gradient_variance': np.random.uniform(0.01, 0.1),
                    'gradient_max_value': np.random.uniform(1.0, 5.0),
                    'gradient_clip_ratio': np.random.uniform(0.0, 0.2),
                    'layer_gradients': {
                        'encoder_0': np.random.uniform(0.1, 1.0),
                        'encoder_1': np.random.uniform(0.1, 1.0),
                        'decoder': np.random.uniform(0.1, 1.0)
                    }
                }
            })
        
        # End episode
        pnl_component = episode_reward * 0.8
        penalty_component = episode_reward * 0.1
        efficiency_component = episode_reward * 0.1
        
        callback_manager.trigger_episode_end({
            'total_reward': episode_reward,
            'episode_length': 50,
            'num_trades': np.random.randint(1, 8),
            'reward_breakdown': {
                'pnl': pnl_component,
                'holding_penalty': -abs(penalty_component),
                'action_efficiency': efficiency_component,
                'total': episode_reward
            },
            'episode_volume': np.random.randint(5000, 25000)
        })
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}")
    
    # End training
    callback_manager.trigger_training_end({
        'final_episode': 100,
        'training_completed': True
    })
    
    # Print callback statistics
    print("\n📊 Callback Statistics:")
    for callback in metric_callbacks:
        stats = callback.get_stats()
        print(f"\n{callback.name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    wandb.finish()


def example_component_integration():
    """Example of how to integrate metrics in your existing components."""
    
    class ExecutionSimulator:
        def __init__(self, callback_manager):
            self.callback_manager = callback_manager
        
        def execute_order(self, order):
            """Execute order and publish metrics."""
            # ... execution logic ...
            
            # Simulate fill
            fill_price = order['price'] + np.random.normal(0, 0.01)
            slippage = abs(fill_price - order['price'])
            latency = np.random.gamma(2, 5)  # Realistic latency distribution
            
            # Publish execution metrics
            self.callback_manager.trigger_custom_event('execution_fill', {
                'raw': {
                    'fill_price': fill_price,
                    'requested_price': order['price'],
                    'slippage': slippage,
                    'size': order['size'],
                    'latency_ms': latency,
                    'execution_cost': slippage * order['size'],
                    'bid_ask_spread': 0.02,
                    'market_impact': slippage * 0.5
                }
            })
            
            return {
                'fill_price': fill_price,
                'size': order['size'],
                'slippage': slippage
            }
    
    class PortfolioManager:
        def __init__(self, callback_manager):
            self.callback_manager = callback_manager
            self.position = 0
            self.cash = 100000
            self.unrealized_pnl = 0
        
        def update_position(self, fill_info, current_price):
            """Update portfolio and publish metrics."""
            # ... portfolio logic ...
            
            old_position = self.position
            self.position += fill_info['size']
            
            # Calculate PnL
            self.unrealized_pnl = self.position * (current_price - fill_info['fill_price'])
            
            # Publish portfolio metrics
            self.callback_manager.trigger_custom_event('portfolio_update', {
                'raw': {
                    'position': self.position,
                    'unrealized_pnl': self.unrealized_pnl,
                    'cash_balance': self.cash,
                    'total_equity': self.cash + self.unrealized_pnl,
                    'margin_used': abs(self.position) * current_price * 0.25
                }
            })
            
            # Publish position change
            self.callback_manager.trigger_custom_event('position_change', {
                'raw': {
                    'old_position': old_position,
                    'new_position': self.position,
                    'position_delta': self.position - old_position
                }
            })
    
    class PPOModel:
        def __init__(self, callback_manager):
            self.callback_manager = callback_manager
        
        def forward(self, observation):
            """Forward pass with attention tracking."""
            # ... model forward logic ...
            
            # Simulate attention computation
            attention_weights = np.random.dirichlet([1] * 10)  # 10 attention heads
            attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8))
            
            # Publish model metrics
            self.callback_manager.trigger_custom_event('model_forward', {
                'raw': {
                    'attention_entropy': attention_entropy,
                    'attention_max_weight': np.max(attention_weights),
                    'prediction_confidence': np.random.uniform(0.5, 0.95),
                    'action_entropy': np.random.uniform(0.3, 1.2),
                    'layer_activations': {
                        'encoder_layer_0': np.random.uniform(0.5, 2.0),
                        'encoder_layer_1': np.random.uniform(0.5, 2.0),
                        'policy_head': np.random.uniform(0.5, 2.0),
                        'value_head': np.random.uniform(0.5, 2.0)
                    }
                }
            })
            
            return {'action': 1, 'value': 0.75}
    
    print("Component integration examples created!")


if __name__ == "__main__":
    print("🚀 Running WandB metrics integration example...")
    example_training_integration()
    
    print("\n💡 Component integration examples:")
    example_component_integration()