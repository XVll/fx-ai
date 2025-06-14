# PyTorch Lightning Callback System Refactoring Guide

## Executive Summary

This document provides a comprehensive guide for refactoring FxAI's custom callback system to PyTorch Lightning's callback architecture. The refactoring will eliminate boilerplate code, improve maintainability, and provide access to Lightning's extensive callback ecosystem.

## Table of Contents

1. [Current Callback Architecture Analysis](#current-callback-architecture-analysis)
2. [Lightning Callback System Overview](#lightning-callback-system-overview)
3. [Callback Mapping Strategy](#callback-mapping-strategy)
4. [Detailed Refactoring Guide](#detailed-refactoring-guide)
5. [Advanced Callback Patterns](#advanced-callback-patterns)
6. [Testing and Validation](#testing-and-validation)
7. [Migration Checklist](#migration-checklist)

## 1. Current Callback Architecture Analysis

### 1.1 Current Structure

```
callbacks/
├── __init__.py
├── base.py                 # BaseCallback abstract class
├── callback_manager.py     # Manages callback lifecycle
├── checkpoint.py          # Model checkpointing
├── context.py            # Training context management
├── factory.py            # Callback creation factory
├── metrics.py            # Metrics tracking
├── wandb_callback.py     # W&B integration
├── evaluation.py         # Model evaluation
├── continuous_training.py # Continuous training logic
├── optuna_callback.py    # Hyperparameter optimization
└── old/                  # Legacy callbacks
```

### 1.2 Current Callback Events

The existing system defines 16 callback events:

```python
class BaseCallback:
    """Current callback interface."""
    
    def on_training_start(self): pass
    def on_training_end(self): pass
    def on_episode_start(self): pass
    def on_episode_end(self): pass
    def on_step_start(self): pass
    def on_step_end(self): pass
    def on_rollout_start(self): pass
    def on_rollout_end(self): pass
    def on_update_start(self): pass
    def on_update_end(self): pass
    def on_evaluation_start(self): pass
    def on_evaluation_end(self): pass
    def on_checkpoint_saved(self): pass
    def on_checkpoint_loaded(self): pass
    def on_metrics_computed(self): pass
    def on_cycle_complete(self): pass
```

### 1.3 Current Implementation Example

```python
# Current implementation
class MetricsCallback(BaseCallback):
    def __init__(self, window_size: int = 100):
        self.metrics = defaultdict(deque)
        self.window_size = window_size
        
    def on_episode_end(self, episode_info: Dict[str, Any]):
        reward = episode_info['reward']
        self.metrics['episode_reward'].append(reward)
        
        if len(self.metrics['episode_reward']) > self.window_size:
            self.metrics['episode_reward'].popleft()
            
    def on_metrics_computed(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            self.metrics[key].append(value)
```

## 2. Lightning Callback System Overview

### 2.1 Lightning Callback Hooks

PyTorch Lightning provides comprehensive hooks for all training stages:

```python
class Callback:
    """Lightning callback interface (key methods)."""
    
    # Setup and teardown
    def setup(self, trainer, pl_module, stage): pass
    def teardown(self, trainer, pl_module, stage): pass
    
    # Training hooks
    def on_train_start(self, trainer, pl_module): pass
    def on_train_end(self, trainer, pl_module): pass
    def on_train_epoch_start(self, trainer, pl_module): pass
    def on_train_epoch_end(self, trainer, pl_module): pass
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): pass
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): pass
    
    # Validation hooks
    def on_validation_start(self, trainer, pl_module): pass
    def on_validation_end(self, trainer, pl_module): pass
    def on_validation_epoch_start(self, trainer, pl_module): pass
    def on_validation_epoch_end(self, trainer, pl_module): pass
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx): pass
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx): pass
    
    # Optimizer hooks
    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx): pass
    def on_before_backward(self, trainer, pl_module, loss): pass
    def on_after_backward(self, trainer, pl_module): pass
    def on_before_zero_grad(self, trainer, pl_module, optimizer): pass
    
    # Checkpoint hooks
    def on_save_checkpoint(self, trainer, pl_module, checkpoint): pass
    def on_load_checkpoint(self, trainer, pl_module, checkpoint): pass
```

### 2.2 Key Differences

| Aspect | Current System | Lightning System |
|--------|---------------|------------------|
| Event Granularity | Episode-based | Batch/Epoch-based |
| State Management | Manual | Automatic via trainer |
| Logging | Custom implementation | Built-in logger support |
| Distributed Support | Manual | Automatic |
| Configuration | Custom factory | Direct instantiation |

## 3. Callback Mapping Strategy

### 3.1 Event Mapping Table

| Current Event | Lightning Hook | Notes |
|--------------|----------------|-------|
| on_training_start | on_train_start | Direct mapping |
| on_training_end | on_train_end | Direct mapping |
| on_episode_start | on_train_batch_start | Episodes map to batches in RL |
| on_episode_end | on_train_batch_end | Check for done flag in batch |
| on_step_start | on_train_batch_start | Environment steps within batch |
| on_step_end | on_train_batch_end | Aggregate step metrics |
| on_rollout_start | on_train_epoch_start | Rollouts collected per epoch |
| on_rollout_end | on_train_epoch_end | Process rollout buffer |
| on_update_start | on_before_optimizer_step | Policy update |
| on_update_end | on_train_batch_end | After optimizer step |
| on_evaluation_start | on_validation_start | Direct mapping |
| on_evaluation_end | on_validation_end | Direct mapping |
| on_checkpoint_saved | on_save_checkpoint | Direct mapping |
| on_checkpoint_loaded | on_load_checkpoint | Direct mapping |
| on_metrics_computed | Use pl_module.log() | Automatic aggregation |
| on_cycle_complete | on_train_epoch_end | Training cycle completion |

### 3.2 Design Principles

1. **Preserve Functionality**: Ensure all current callback features are maintained
2. **Leverage Lightning Features**: Use built-in logging, checkpointing, and distributed support
3. **Minimize Code Changes**: Create compatibility layers where needed
4. **Improve Performance**: Utilize Lightning's optimizations
5. **Enhance Testability**: Use Lightning's testing utilities

## 4. Detailed Refactoring Guide

### 4.1 Base Callback Refactoring

```python
# fxai/lightning/callbacks/base.py
from pytorch_lightning.callbacks import Callback
from abc import ABC
from typing import Dict, Any, Optional

class RLCallback(Callback, ABC):
    """Base callback for RL-specific functionality."""
    
    def __init__(self):
        super().__init__()
        self._episode_data = {}
        self._step_count = 0
        self._episode_count = 0
        
    # RL-specific hooks
    def on_episode_start(self, trainer, pl_module) -> None:
        """Called at the start of an episode."""
        pass
        
    def on_episode_end(self, trainer, pl_module, episode_info: Dict[str, Any]) -> None:
        """Called at the end of an episode."""
        pass
        
    def on_step(self, trainer, pl_module, step_info: Dict[str, Any]) -> None:
        """Called after each environment step."""
        pass
        
    # Lightning hook implementations
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Map batch end to episode/step events."""
        # Check for episode boundaries
        if 'done' in batch:
            done_indices = torch.where(batch['done'])[0]
            for idx in done_indices:
                episode_info = self._extract_episode_info(batch, idx)
                self.on_episode_end(trainer, pl_module, episode_info)
                self._episode_count += 1
                
        # Track steps
        self._step_count += batch['observations'].shape[0]
        step_info = {'total_steps': self._step_count}
        self.on_step(trainer, pl_module, step_info)
        
    def _extract_episode_info(self, batch: Dict[str, torch.Tensor], idx: int) -> Dict[str, Any]:
        """Extract episode information from batch."""
        return {
            'reward': batch.get('episode_rewards', [0])[idx].item(),
            'length': batch.get('episode_lengths', [0])[idx].item(),
            'info': batch.get('info', {})
        }
```

### 4.2 Metrics Callback Refactoring

```python
# Before: Custom metrics callback
class MetricsCallback(BaseCallback):
    def __init__(self, window_size: int = 100):
        self.metrics = defaultdict(deque)
        self.window_size = window_size
        
    def on_episode_end(self, episode_info: Dict[str, Any]):
        reward = episode_info['reward']
        self.metrics['episode_reward'].append(reward)
        
    def on_metrics_computed(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            wandb.log({key: value})

# After: Lightning metrics callback
from pytorch_lightning.callbacks import Callback
import torchmetrics

class MetricsCallback(Callback):
    def __init__(self, window_size: int = 100):
        super().__init__()
        self.window_size = window_size
        
        # Use torchmetrics for automatic aggregation
        self.episode_reward = torchmetrics.MeanMetric()
        self.episode_length = torchmetrics.MeanMetric()
        self.win_rate = torchmetrics.MeanMetric()
        
        # Window tracking
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Process metrics from batch."""
        # Check for completed episodes
        if 'episode_rewards' in outputs:
            for reward in outputs['episode_rewards']:
                self.episode_reward.update(reward)
                self.reward_window.append(reward)
                
                # Log to Lightning
                pl_module.log('metrics/episode_reward', reward)
                pl_module.log('metrics/episode_reward_mean', np.mean(self.reward_window))
                pl_module.log('metrics/episode_reward_std', np.std(self.reward_window))
                
        if 'episode_lengths' in outputs:
            for length in outputs['episode_lengths']:
                self.episode_length.update(length)
                self.length_window.append(length)
                pl_module.log('metrics/episode_length', length)
                
    def on_train_epoch_end(self, trainer, pl_module):
        """Compute epoch-level metrics."""
        # Log aggregated metrics
        pl_module.log('metrics/episode_reward_epoch', self.episode_reward.compute())
        pl_module.log('metrics/episode_length_epoch', self.episode_length.compute())
        
        # Reset metrics
        self.episode_reward.reset()
        self.episode_length.reset()
```

### 4.3 Checkpoint Callback Refactoring

```python
# Before: Custom checkpoint callback
class CheckpointCallback(BaseCallback):
    def __init__(self, save_dir: str, save_freq: int):
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.best_reward = -float('inf')
        
    def on_cycle_complete(self, trainer, metrics):
        if metrics['episode_reward_mean'] > self.best_reward:
            self.best_reward = metrics['episode_reward_mean']
            self.save_checkpoint(trainer.model, 'best_model.pt')
            
    def save_checkpoint(self, model, filename):
        torch.save(model.state_dict(), os.path.join(self.save_dir, filename))

# After: Lightning checkpoint callback
from pytorch_lightning.callbacks import ModelCheckpoint

def create_checkpoint_callback(config):
    """Create Lightning checkpoint callback."""
    return ModelCheckpoint(
        dirpath=config.save_dir,
        filename='model-{epoch:04d}-{val_episode_reward:.3f}',
        monitor='val/episode_reward',
        mode='max',
        save_top_k=5,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=config.save_freq,
        save_on_train_epoch_end=True
    )

# Custom checkpoint callback for RL-specific data
class RLCheckpointCallback(ModelCheckpoint):
    """Extended checkpoint callback with RL-specific data."""
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Add RL-specific data to checkpoint."""
        # Add rollout buffer state
        if hasattr(trainer.datamodule, 'rollout_buffer'):
            checkpoint['rollout_buffer'] = trainer.datamodule.rollout_buffer.get_state()
            
        # Add environment state
        if hasattr(trainer.datamodule, 'env'):
            checkpoint['env_state'] = trainer.datamodule.env.get_state()
            
        # Add episode statistics
        checkpoint['episode_stats'] = {
            'total_episodes': pl_module.total_episodes,
            'total_steps': pl_module.total_steps,
            'best_reward': pl_module.best_reward
        }
        
    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Load RL-specific data from checkpoint."""
        # Restore rollout buffer
        if 'rollout_buffer' in checkpoint:
            trainer.datamodule.rollout_buffer.set_state(checkpoint['rollout_buffer'])
            
        # Restore environment state
        if 'env_state' in checkpoint:
            trainer.datamodule.env.set_state(checkpoint['env_state'])
            
        # Restore episode statistics
        if 'episode_stats' in checkpoint:
            pl_module.total_episodes = checkpoint['episode_stats']['total_episodes']
            pl_module.total_steps = checkpoint['episode_stats']['total_steps']
            pl_module.best_reward = checkpoint['episode_stats']['best_reward']
```

### 4.4 W&B Callback Refactoring

```python
# Before: Custom W&B callback
class WandbCallback(BaseCallback):
    def __init__(self, project: str, config: dict):
        wandb.init(project=project, config=config)
        
    def on_metrics_computed(self, metrics: Dict[str, float]):
        wandb.log(metrics)
        
    def on_episode_end(self, episode_info):
        wandb.log({
            'episode/reward': episode_info['reward'],
            'episode/length': episode_info['length']
        })

# After: Use Lightning's W&B logger
from pytorch_lightning.loggers import WandbLogger

def create_wandb_logger(config):
    """Create W&B logger for Lightning."""
    return WandbLogger(
        project=config.project,
        name=config.name,
        config=config.to_dict(),
        save_dir=config.save_dir,
        log_model=True,  # Automatically log model checkpoints
        tags=config.tags,
        notes=config.notes
    )

# Custom W&B callback for additional functionality
class RLWandbCallback(Callback):
    """Additional W&B logging for RL-specific metrics."""
    
    def __init__(self):
        super().__init__()
        self.episode_videos = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log RL-specific visualizations."""
        # Log action distribution
        if 'action_probs' in outputs:
            wandb.log({
                'charts/action_distribution': wandb.Histogram(
                    outputs['action_probs'].cpu().numpy()
                )
            })
            
        # Log value estimates
        if 'values' in outputs:
            wandb.log({
                'charts/value_estimates': wandb.Histogram(
                    outputs['values'].cpu().numpy()
                )
            })
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation videos."""
        if len(self.episode_videos) > 0:
            wandb.log({
                'videos/episodes': wandb.Video(
                    np.array(self.episode_videos),
                    fps=30,
                    format='mp4'
                )
            })
            self.episode_videos = []
```

### 4.5 Continuous Training Callback Refactoring

```python
# Before: Custom continuous training callback
class ContinuousTrainingCallback(BaseCallback):
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.version = self._get_latest_version()
        
    def on_training_start(self, trainer):
        if self.version > 0:
            checkpoint_path = self._get_checkpoint_path(self.version)
            trainer.load_checkpoint(checkpoint_path)
            
    def on_checkpoint_saved(self, trainer, checkpoint_path):
        self.version += 1
        new_path = self._get_checkpoint_path(self.version)
        shutil.copy(checkpoint_path, new_path)

# After: Lightning continuous training callback
class ContinuousTrainingCallback(Callback):
    """Manage continuous training with automatic versioning."""
    
    def __init__(self, checkpoint_dir: str, resume_from_best: bool = True):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.resume_from_best = resume_from_best
        self.version_manager = VersionManager(checkpoint_dir)
        
    def setup(self, trainer, pl_module, stage):
        """Setup continuous training."""
        if stage == 'fit':
            # Find latest checkpoint
            latest_ckpt = self.version_manager.get_latest_checkpoint()
            
            if latest_ckpt and trainer.ckpt_path is None:
                print(f"Resuming from checkpoint: {latest_ckpt}")
                trainer.ckpt_path = latest_ckpt
                
                # Update version
                self.version_manager.increment_version()
                
                # Update logger version
                if hasattr(trainer.logger, 'version'):
                    trainer.logger._version = self.version_manager.current_version
                    
    def on_train_start(self, trainer, pl_module):
        """Log training continuation info."""
        pl_module.log('training/version', self.version_manager.current_version)
        pl_module.log('training/resumed', 1 if trainer.ckpt_path else 0)
        
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Add versioning info to checkpoint."""
        checkpoint['version_info'] = {
            'version': self.version_manager.current_version,
            'timestamp': time.time(),
            'total_steps': pl_module.global_step,
            'best_reward': getattr(pl_module, 'best_reward', None)
        }
```

### 4.6 Optuna Callback Refactoring

```python
# Before: Custom Optuna callback
class OptunaCallback(BaseCallback):
    def __init__(self, trial, monitor: str = 'episode_reward_mean'):
        self.trial = trial
        self.monitor = monitor
        self.best_value = -float('inf')
        
    def on_metrics_computed(self, metrics):
        value = metrics.get(self.monitor, 0)
        self.trial.report(value, step=metrics['timesteps_total'])
        
        if self.trial.should_prune():
            raise optuna.TrialPruned()

# After: Lightning Optuna callback
from pytorch_lightning.callbacks import Callback
import optuna

class OptunaCallback(Callback):
    """Optuna integration for hyperparameter optimization."""
    
    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = 'val/episode_reward',
        direction: str = 'maximize'
    ):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.direction = direction
        self.best_value = -float('inf') if direction == 'maximize' else float('inf')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Report metrics to Optuna."""
        # Get metric value
        metrics = trainer.callback_metrics
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            return
            
        # Report to Optuna
        self.trial.report(current_value, step=trainer.current_epoch)
        
        # Update best value
        if self.direction == 'maximize':
            self.best_value = max(self.best_value, current_value)
        else:
            self.best_value = min(self.best_value, current_value)
            
        # Check for pruning
        if self.trial.should_prune():
            raise optuna.TrialPruned()
            
    def on_train_end(self, trainer, pl_module):
        """Report final value."""
        self.trial.set_user_attr('best_value', self.best_value)
        self.trial.set_user_attr('final_epoch', trainer.current_epoch)
```

## 5. Advanced Callback Patterns

### 5.1 Composite Callback Pattern

```python
class CompositeRLCallback(Callback):
    """Combine multiple RL callbacks into one."""
    
    def __init__(self, callbacks: List[Callback]):
        super().__init__()
        self.callbacks = callbacks
        
    def setup(self, trainer, pl_module, stage):
        for callback in self.callbacks:
            callback.setup(trainer, pl_module, stage)
            
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for callback in self.callbacks:
            callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
            
    # Implement all other hooks similarly...
```

### 5.2 State Machine Callback

```python
class StateMachineCallback(Callback):
    """Manage training states with a state machine."""
    
    def __init__(self):
        super().__init__()
        self.state = 'warmup'
        self.state_transitions = {
            'warmup': self._check_warmup_complete,
            'exploration': self._check_exploration_complete,
            'exploitation': self._check_exploitation_complete,
            'fine_tuning': self._check_fine_tuning_complete
        }
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Check for state transitions."""
        if self.state in self.state_transitions:
            next_state = self.state_transitions[self.state](trainer, pl_module)
            if next_state != self.state:
                self._transition_to(next_state, trainer, pl_module)
                
    def _transition_to(self, new_state: str, trainer, pl_module):
        """Handle state transition."""
        print(f"Transitioning from {self.state} to {new_state}")
        self.state = new_state
        
        # Adjust training parameters based on state
        if new_state == 'exploitation':
            # Reduce exploration
            pl_module.exploration_rate *= 0.1
        elif new_state == 'fine_tuning':
            # Reduce learning rate
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
```

### 5.3 Distributed Training Callback

```python
class DistributedRLCallback(Callback):
    """Handle RL-specific distributed training logic."""
    
    def __init__(self):
        super().__init__()
        self.sync_frequency = 100
        self.step_count = 0
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Synchronize environments periodically."""
        self.step_count += 1
        
        if self.step_count % self.sync_frequency == 0 and trainer.world_size > 1:
            # Synchronize environment states
            if trainer.is_global_zero:
                env_state = trainer.datamodule.env.get_state()
            else:
                env_state = None
                
            # Broadcast to all ranks
            env_state = trainer.strategy.broadcast(env_state, src=0)
            
            # Update environments
            trainer.datamodule.env.set_state(env_state)
            
    def on_train_epoch_end(self, trainer, pl_module):
        """Aggregate metrics across all processes."""
        if trainer.world_size > 1:
            # Get local metrics
            local_rewards = pl_module.episode_rewards
            
            # Gather from all processes
            all_rewards = trainer.strategy.all_gather(local_rewards)
            
            # Compute global statistics
            if trainer.is_global_zero:
                global_mean = torch.mean(all_rewards)
                global_std = torch.std(all_rewards)
                pl_module.log('distributed/global_reward_mean', global_mean)
                pl_module.log('distributed/global_reward_std', global_std)
```

## 6. Testing and Validation

### 6.1 Unit Tests for Callbacks

```python
# tests/test_lightning_callbacks.py
import pytest
import pytorch_lightning as pl
from unittest.mock import Mock, MagicMock

class TestMetricsCallback:
    """Test metrics callback functionality."""
    
    def test_episode_tracking(self):
        # Create callback
        callback = MetricsCallback(window_size=10)
        
        # Create mock objects
        trainer = Mock()
        pl_module = Mock()
        
        # Create batch with episode data
        batch = {
            'done': torch.tensor([False, True, False]),
            'episode_rewards': torch.tensor([0, 100, 0]),
            'episode_lengths': torch.tensor([0, 50, 0])
        }
        outputs = {'episode_rewards': [100]}
        
        # Call callback
        callback.on_train_batch_end(trainer, pl_module, outputs, batch, 0)
        
        # Verify logging
        pl_module.log.assert_called_with('metrics/episode_reward', 100)
        
    def test_window_aggregation(self):
        callback = MetricsCallback(window_size=3)
        
        # Add rewards
        for reward in [10, 20, 30, 40]:
            callback.reward_window.append(reward)
            
        # Check window size
        assert len(callback.reward_window) == 3
        assert list(callback.reward_window) == [20, 30, 40]
```

### 6.2 Integration Tests

```python
# tests/test_callback_integration.py
class TestCallbackIntegration:
    """Test callback integration with Lightning."""
    
    def test_full_training_loop(self, tmp_path):
        # Create model and data
        model = PPOLightningModule(config)
        datamodule = RLDataModule(config)
        
        # Create callbacks
        callbacks = [
            MetricsCallback(),
            RLCheckpointCallback(dirpath=tmp_path),
            EpisodeMetricsCallback()
        ]
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=2,
            callbacks=callbacks,
            fast_dev_run=True
        )
        
        # Train
        trainer.fit(model, datamodule)
        
        # Verify callbacks were called
        assert trainer.current_epoch == 1
        assert len(list(tmp_path.glob("*.ckpt"))) > 0
```

### 6.3 Performance Tests

```python
# tests/test_callback_performance.py
import time

class TestCallbackPerformance:
    """Test callback performance impact."""
    
    def test_callback_overhead(self):
        # Baseline without callbacks
        start = time.time()
        trainer = pl.Trainer(max_epochs=1, callbacks=[])
        trainer.fit(model, datamodule)
        baseline_time = time.time() - start
        
        # With callbacks
        start = time.time()
        trainer = pl.Trainer(
            max_epochs=1,
            callbacks=[
                MetricsCallback(),
                EpisodeMetricsCallback(),
                RewardSystemCallback(reward_system)
            ]
        )
        trainer.fit(model, datamodule)
        callback_time = time.time() - start
        
        # Check overhead is reasonable (< 10%)
        overhead = (callback_time - baseline_time) / baseline_time
        assert overhead < 0.1
```

## 7. Migration Checklist

### 7.1 Pre-Migration Checklist

- [ ] Backup current callback implementations
- [ ] Document current callback usage patterns
- [ ] Identify custom callback dependencies
- [ ] Create test suite for current callbacks
- [ ] Review Lightning callback documentation

### 7.2 Migration Steps

1. **Phase 1: Setup**
   - [ ] Create `fxai/lightning/callbacks/` directory
   - [ ] Implement base `RLCallback` class
   - [ ] Set up callback testing framework

2. **Phase 2: Core Callbacks**
   - [ ] Migrate `MetricsCallback`
   - [ ] Migrate `CheckpointCallback`
   - [ ] Migrate `WandbCallback`
   - [ ] Test core functionality

3. **Phase 3: Advanced Callbacks**
   - [ ] Migrate `ContinuousTrainingCallback`
   - [ ] Migrate `OptunaCallback`
   - [ ] Migrate `EvaluationCallback`
   - [ ] Implement new Lightning-specific callbacks

4. **Phase 4: Integration**
   - [ ] Update training scripts to use new callbacks
   - [ ] Update configuration files
   - [ ] Run integration tests
   - [ ] Performance benchmarking

5. **Phase 5: Cleanup**
   - [ ] Remove old callback system
   - [ ] Update documentation
   - [ ] Update example scripts
   - [ ] Final testing

### 7.3 Post-Migration Validation

- [ ] Verify all callback events are captured
- [ ] Confirm metrics are logged correctly
- [ ] Test distributed training scenarios
- [ ] Validate checkpoint compatibility
- [ ] Check performance metrics

## Conclusion

The refactoring of FxAI's callback system to PyTorch Lightning represents a significant improvement in code organization, maintainability, and functionality. Key benefits include:

1. **Reduced Boilerplate**: Lightning handles callback lifecycle management
2. **Better Integration**: Native support for logging, checkpointing, and distributed training
3. **Enhanced Features**: Access to Lightning's callback ecosystem
4. **Improved Testing**: Better testing utilities and patterns
5. **Future-Proof**: Easier to add new functionality

The migration can be done incrementally, allowing for thorough testing at each phase. The compatibility layer ensures that existing functionality is preserved while gaining the benefits of Lightning's architecture.