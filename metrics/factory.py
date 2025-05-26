# metrics/factory.py - Factory and integration helpers for metrics system

import logging
from typing import Dict, List, Optional, Any
import torch.nn as nn
import torch.optim as optim

from .manager import MetricsManager
from .transmitters.wandb_transmitter import WandBTransmitter, WandBConfig
from .collectors.model_metrics import ModelMetricsCollector, OptimizerMetricsCollector
from .collectors.training_metrics import TrainingMetricsCollector, EvaluationMetricsCollector
from .collectors.trading_metrics import PortfolioMetricsCollector, PositionMetricsCollector, TradeMetricsCollector
from .collectors.execution_metrics import ExecutionMetricsCollector, EnvironmentMetricsCollector, SystemMetricsCollector
from .collectors.visualization_metrics import VisualizationMetrics
from .collectors.reward_metrics import RewardMetricsCollector
from .collectors.model_internals_metrics import ModelInternalsCollector
from .core import MetricCategory, MetricFilter, MetricValue


class MetricsFactory:
    """Factory for creating and configuring metrics components"""

    @staticmethod
    def create_wandb_transmitter(config: WandBConfig) -> WandBTransmitter:
        """Create a W&B transmitter with configuration"""
        return WandBTransmitter(
            project_name=config.project_name,
            entity=config.entity,
            run_name=config.run_name,
            config=config.to_dict(),
            tags=config.tags,
            group=config.group,
            job_type=config.job_type,
            save_code=config.save_code,
            log_frequency=config.log_frequency
        )

    @staticmethod
    def create_model_collectors(model: Optional[nn.Module] = None,
                                optimizer: Optional[optim.Optimizer] = None) -> List:
        """Create model-related collectors"""
        collectors = []

        if model is not None:
            collectors.append(ModelMetricsCollector(model))

        if optimizer is not None:
            collectors.append(OptimizerMetricsCollector(optimizer))

        return collectors

    @staticmethod
    def create_training_collectors() -> List:
        """Create training-related collectors"""
        return [
            TrainingMetricsCollector(),
            EvaluationMetricsCollector()
        ]

    @staticmethod
    def create_trading_collectors(symbol: str, initial_capital: float = 25000.0) -> List:
        """Create trading-related collectors"""
        return [
            PortfolioMetricsCollector(initial_capital),
            PositionMetricsCollector(symbol),
            TradeMetricsCollector()
        ]

    @staticmethod
    def create_execution_collectors() -> List:
        """Create execution-related collectors"""
        return [
            ExecutionMetricsCollector(),
            EnvironmentMetricsCollector()
        ]

    @staticmethod
    def create_system_collectors() -> List:
        """Create system-related collectors"""
        return [SystemMetricsCollector()]
    
    @staticmethod
    def create_visualization_collectors() -> List:
        """Create visualization-related collectors"""
        return [VisualizationMetrics()]
    
    @staticmethod
    def create_reward_collectors() -> List:
        """Create reward system collectors"""
        return [RewardMetricsCollector()]

    @staticmethod
    def create_complete_metrics_system(
            wandb_config: WandBConfig,
            model: Optional[nn.Module] = None,
            optimizer: Optional[optim.Optimizer] = None,
            symbol: str = "UNKNOWN",
            initial_capital: float = 25000.0,
            include_system: bool = True
    ) -> MetricsManager:
        """Create a complete metrics system with all components"""

        # Create manager
        manager = MetricsManager(
            transmit_interval=1.0,
            auto_transmit=True,
            buffer_size=1000
        )

        # Add W&B transmitter
        wandb_transmitter = MetricsFactory.create_wandb_transmitter(wandb_config)
        manager.add_transmitter(wandb_transmitter)

        # Add all collectors
        collectors = []

        # Model collectors
        collectors.extend(MetricsFactory.create_model_collectors(model, optimizer))

        # Training collectors
        collectors.extend(MetricsFactory.create_training_collectors())

        # Trading collectors
        collectors.extend(MetricsFactory.create_trading_collectors(symbol, initial_capital))

        # Execution collectors
        collectors.extend(MetricsFactory.create_execution_collectors())
        
        # Reward collectors
        collectors.extend(MetricsFactory.create_reward_collectors())
        
        # Model internals collectors
        collectors.append(ModelInternalsCollector())

        # System collectors (optional)
        if include_system:
            collectors.extend(MetricsFactory.create_system_collectors())
            
        # Visualization collectors (always included for episode tracking)
        collectors.extend(MetricsFactory.create_visualization_collectors())

        # Register all collectors
        for collector in collectors:
            manager.register_collector(collector)

        return manager


class MetricsIntegrator:
    """Helper class for integrating metrics into existing components"""

    def __init__(self, metrics_manager: MetricsManager):
        self.metrics_manager = metrics_manager
        self.logger = logging.getLogger(__name__)

        # Get references to collectors for easy access
        self._collectors = {}
        self._update_collector_references()

    def _update_collector_references(self):
        """Update references to collectors"""
        for collector_id, collector in self.metrics_manager.collectors.items():
            collector_type = collector.__class__.__name__
            self._collectors[collector_type] = collector

    def get_collector(self, collector_type: str):
        """Get a specific collector by type"""
        return self._collectors.get(collector_type)

    # Model integration methods
    def record_model_losses(self, actor_loss: float, critic_loss: float, entropy: float):
        """Record model loss metrics"""
        collector = self.get_collector('ModelMetricsCollector')
        if collector:
            metrics = collector.record_loss_metrics(actor_loss, critic_loss, entropy)
            self._transmit_metrics(metrics)

    def record_ppo_metrics(self, clip_fraction: float, approx_kl: float, explained_variance: float):
        """Record PPO-specific metrics"""
        collector = self.get_collector('ModelMetricsCollector')
        if collector:
            metrics = collector.record_ppo_metrics(clip_fraction, approx_kl, explained_variance)
            self._transmit_metrics(metrics)

    def record_learning_rate(self, learning_rate: float):
        """Record learning rate"""
        collector = self.get_collector('ModelMetricsCollector')
        if collector:
            metrics = collector.record_learning_rate(learning_rate)
            self._transmit_metrics(metrics)

    # Training integration methods
    def start_training(self):
        """Start training tracking"""
        collector = self.get_collector('TrainingMetricsCollector')
        if collector:
            collector.start_training()

    def start_episode(self):
        """Start episode tracking"""
        collector = self.get_collector('TrainingMetricsCollector')
        if collector:
            collector.start_episode()

    def end_episode(self, reward: float, length: int):
        """End episode tracking"""
        collector = self.get_collector('TrainingMetricsCollector')
        if collector:
            collector.end_episode(reward, length)

    def start_update(self):
        """Start update tracking"""
        collector = self.get_collector('TrainingMetricsCollector')
        if collector:
            collector.start_update()

    def end_update(self):
        """End update tracking"""
        collector = self.get_collector('TrainingMetricsCollector')
        if collector:
            collector.end_update()

    def record_rollout_time(self, duration: float):
        """Record rollout duration"""
        collector = self.get_collector('TrainingMetricsCollector')
        if collector:
            collector.record_rollout_time(duration)

    def update_step(self, step: int):
        """Update step count"""
        collector = self.get_collector('TrainingMetricsCollector')
        if collector:
            collector.update_step(step)

        # Update manager state
        self.metrics_manager.update_state(step=step)

    # Evaluation integration methods
    def start_evaluation(self):
        """Start evaluation"""
        collector = self.get_collector('EvaluationMetricsCollector')
        if collector:
            collector.start_evaluation()

        self.metrics_manager.update_state(is_evaluating=True, is_training=False)

    def end_evaluation(self, rewards: List[float], lengths: List[int]):
        """End evaluation"""
        collector = self.get_collector('EvaluationMetricsCollector')
        if collector:
            collector.end_evaluation(rewards, lengths)

        self.metrics_manager.update_state(is_evaluating=False, is_training=True)
    
    # Model internals integration methods
    def update_attention_weights(self, weights: Any):
        """Update attention weights from model"""
        collector = self.get_collector('ModelInternalsCollector')
        if collector:
            collector.update_attention_weights(weights)
    
    def update_action_probabilities(self, action_probs: Any):
        """Update action probability distributions"""
        collector = self.get_collector('ModelInternalsCollector')
        if collector:
            collector.update_action_probabilities(action_probs)
    
    def update_feature_statistics(self, features: Dict[str, Any]):
        """Update feature distribution statistics"""
        collector = self.get_collector('ModelInternalsCollector')
        if collector:
            collector.update_feature_statistics(features)

    # Portfolio integration methods
    def update_portfolio(self, equity: float, cash: float, unrealized_pnl: float, realized_pnl: float,
                        total_commission: float = 0.0, total_slippage: float = 0.0, total_fees: float = 0.0):
        """Update portfolio state"""
        collector = self.get_collector('PortfolioMetricsCollector')
        if collector:
            collector.update_portfolio_state(equity, cash, unrealized_pnl, realized_pnl)
        
        # Pass costs to dashboard if available through metrics manager
        if hasattr(self.metrics_manager, 'dashboard_collector') and self.metrics_manager.dashboard_collector:
            # Update dashboard with portfolio costs
            self.metrics_manager.update_dashboard_step({
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'total_fees': total_fees
            })

    def update_position(self, quantity: float, side: str, avg_entry_price: float,
                        market_value: float, unrealized_pnl: float, current_price: float):
        """Update position state"""
        collector = self.get_collector('PositionMetricsCollector')
        if collector:
            collector.update_position(quantity, side, avg_entry_price,
                                      market_value, unrealized_pnl, current_price)

    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade"""
        collector = self.get_collector('TradeMetricsCollector')
        if collector:
            collector.record_trade(trade_data)

    # Execution integration methods
    def record_fill(self, fill_data: Dict[str, Any]):
        """Record a fill execution"""
        collector = self.get_collector('ExecutionMetricsCollector')
        if collector:
            collector.record_fill(fill_data)

    # Environment integration methods
    def record_environment_step(self, reward: float, action: str, is_invalid: bool = False,
                                reward_components: Optional[Dict[str, float]] = None,
                                episode_reward: Optional[float] = None):
        """Record environment step"""
        collector = self.get_collector('EnvironmentMetricsCollector')
        if collector:
            collector.record_step(reward, action, is_invalid, reward_components, episode_reward)
            
        # Also update reward metrics if components provided
        if reward_components:
            reward_collector = self.get_collector('RewardMetricsCollector')
            if reward_collector:
                reward_collector.update_reward(reward, reward_components)

    def record_episode_end(self, episode_reward: float):
        """Record episode end in environment"""
        collector = self.get_collector('EnvironmentMetricsCollector')
        if collector:
            collector.record_episode_end(episode_reward)
            
        # Reset reward collector for new episode
        reward_collector = self.get_collector('RewardMetricsCollector')
        if reward_collector:
            reward_collector.reset_episode()
            
    # Reward integration methods
    def register_reward_component(self, component_name: str, component_type: str):
        """Register a reward component for tracking"""
        collector = self.get_collector('RewardMetricsCollector')
        if collector:
            collector.register_component(component_name, component_type)
            
    def update_reward_metrics(self, total_reward: float, component_rewards: Dict[str, float]):
        """Update reward metrics directly"""
        collector = self.get_collector('RewardMetricsCollector')
        if collector:
            collector.update_reward(total_reward, component_rewards)
            
    def get_reward_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get reward component statistics"""
        collector = self.get_collector('RewardMetricsCollector')
        if collector:
            return collector.get_component_statistics()
        return {}

    # System integration methods
    def start_system_tracking(self):
        """Start system tracking"""
        collector = self.get_collector('SystemMetricsCollector')
        if collector:
            collector.start_tracking()
            
    # Custom metrics
    def record_custom_metrics(self, metrics: Dict[str, Any]):
        """Record custom metrics directly to transmitters"""
        # Convert to metric values
        metric_values = {k: MetricValue(v) for k, v in metrics.items()}
        self.metrics_manager.transmit_metrics(metric_values)

    # Transmission control
    def _transmit_metrics(self, metrics: Dict[str, Any]):
        """Transmit specific metrics immediately"""
        if metrics:
            self.metrics_manager.transmit_metrics(metrics)

    def transmit_all_metrics(self, force: bool = False):
        """Transmit all metrics"""
        if force:
            self.metrics_manager.transmit_metrics()

    def transmit_category_metrics(self, categories: List[MetricCategory]):
        """Transmit metrics for specific categories"""
        filter_obj = MetricFilter().filter_categories(*categories)
        self.metrics_manager.transmit_metrics(filter_obj=filter_obj)


class MetricsConfig:
    """Configuration class for metrics system"""

    def __init__(self,
                 # W&B Configuration
                 wandb_project: str = "fx-ai",
                 wandb_entity: Optional[str] = None,
                 wandb_run_name: Optional[str] = None,
                 wandb_tags: Optional[List[str]] = None,
                 wandb_group: Optional[str] = None,

                 # Metrics Configuration
                 transmit_interval: float = 1.0,
                 auto_transmit: bool = True,
                 buffer_size: int = 1000,
                 include_system_metrics: bool = True,

                 # Trading Configuration
                 symbol: str = "UNKNOWN",
                 initial_capital: float = 25000.0,

                 # Log Frequencies (in steps)
                 log_frequencies: Optional[Dict[str, int]] = None):
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_tags = wandb_tags or []
        self.wandb_group = wandb_group

        self.transmit_interval = transmit_interval
        self.auto_transmit = auto_transmit
        self.buffer_size = buffer_size
        self.include_system_metrics = include_system_metrics

        self.symbol = symbol
        self.initial_capital = initial_capital

        self.log_frequencies = log_frequencies or {
            "model": 1,
            "training": 1,
            "trading": 1,
            "execution": 1,
            "environment": 1,
            "system": 10
        }

    def create_wandb_config(self, additional_config: Optional[Dict[str, Any]] = None) -> WandBConfig:
        """Create W&B configuration"""
        config = additional_config or {}

        return WandBConfig(
            project_name=self.wandb_project,
            entity=self.wandb_entity,
            run_name=self.wandb_run_name,
            tags=self.wandb_tags,
            group=self.wandb_group,
            log_frequency=self.log_frequencies
        )

    def create_metrics_system(self,
                              model: Optional[nn.Module] = None,
                              optimizer: Optional[optim.Optimizer] = None,
                              additional_config: Optional[Dict[str, Any]] = None) -> tuple[MetricsManager, MetricsIntegrator]:
        """Create complete metrics system"""

        wandb_config = self.create_wandb_config(additional_config)

        manager = MetricsFactory.create_complete_metrics_system(
            wandb_config=wandb_config,
            model=model,
            optimizer=optimizer,
            symbol=self.symbol,
            initial_capital=self.initial_capital,
            include_system=self.include_system_metrics
        )

        integrator = MetricsIntegrator(manager)

        return manager, integrator


# Convenience functions for quick setup
def create_simple_metrics_system(project_name: str,
                                 symbol: str,
                                 model: Optional[nn.Module] = None,
                                 optimizer: Optional[optim.Optimizer] = None,
                                 entity: Optional[str] = None) -> tuple[MetricsManager, MetricsIntegrator]:
    """Create a simple metrics system with minimal configuration"""

    config = MetricsConfig(
        wandb_project=project_name,
        wandb_entity=entity,
        symbol=symbol
    )

    return config.create_metrics_system(model, optimizer)


def create_trading_metrics_system(project_name: str,
                                  symbol: str,
                                  initial_capital: float,
                                  model: Optional[nn.Module] = None,
                                  optimizer: Optional[optim.Optimizer] = None,
                                  entity: Optional[str] = None,
                                  run_name: Optional[str] = None) -> tuple[MetricsManager, MetricsIntegrator]:
    """Create a metrics system optimized for trading"""

    config = MetricsConfig(
        wandb_project=project_name,
        wandb_entity=entity,
        wandb_run_name=run_name,
        symbol=symbol,
        initial_capital=initial_capital,
        wandb_tags=["trading", "ppo", symbol],
        transmit_interval=0.5,  # More frequent for trading
        log_frequencies={
            "model": 1,
            "training": 1,
            "trading": 1,
            "execution": 1,
            "environment": 1,
            "system": 20  # Less frequent for system metrics
        }
    )

    return config.create_metrics_system(model, optimizer)