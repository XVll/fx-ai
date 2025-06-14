# PyTorch Lightning Implementation Guide for FxAI

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Step-by-Step Implementation](#step-by-step-implementation)
3. [Component Templates](#component-templates)
4. [Migration Patterns](#migration-patterns)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## 1. Quick Start Guide

### Prerequisites

```bash
# Install PyTorch Lightning
poetry add pytorch-lightning

# Install additional dependencies
poetry add lightning-bolts  # For RL utilities
poetry add torchmetrics     # For metric computation
```

### Basic Setup

```python
# fxai/lightning/__init__.py
from .ppo_module import PPOLightningModule
from .data_module import RLDataModule
from .callbacks import (
    EpisodeMetricsCallback,
    RolloutCollectionCallback,
    RewardSystemCallback
)

__all__ = [
    'PPOLightningModule',
    'RLDataModule',
    'EpisodeMetricsCallback',
    'RolloutCollectionCallback',
    'RewardSystemCallback'
]
```

## 2. Step-by-Step Implementation

### Step 1: Create Lightning Module Structure

```python
# fxai/lightning/ppo_module.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
from torchmetrics import MeanMetric

from model.transformer import MultiBranchTransformer
from config.model.model_config import ModelConfig
from config.training.training_config import PPOConfig

class PPOLightningModule(pl.LightningModule):
    """
    Lightning implementation of PPO for algorithmic trading.
    
    This module handles:
    - Model initialization and forward passes
    - Loss computation (policy, value, entropy)
    - Optimizer configuration
    - Metric logging
    - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        ppo_config: PPOConfig,
        reward_system: Any = None
    ):
        super().__init__()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['reward_system'])
        
        # Initialize model
        self.model = MultiBranchTransformer(model_config)
        
        # Value and policy heads
        self.value_head = nn.Sequential(
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(model_config.hidden_dim, 1)
        )
        
        # PPO parameters
        self.clip_epsilon = ppo_config.clip_epsilon
        self.entropy_coef = ppo_config.entropy_coef
        self.value_loss_coef = ppo_config.value_loss_coef
        self.max_grad_norm = ppo_config.max_grad_norm
        
        # Reward system
        self.reward_system = reward_system
        
        # Metrics
        self.train_policy_loss = MeanMetric()
        self.train_value_loss = MeanMetric()
        self.train_entropy = MeanMetric()
        self.val_episode_reward = MeanMetric()
        
        # Episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
    def forward(
        self,
        observations: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            observations: Dictionary of observation tensors
            
        Returns:
            action_logits: Logits for action distribution
            values: Value estimates
        """
        # Get features from transformer
        features = self.model(observations)
        
        # Get action logits and values
        action_logits = self.model.action_head(features)
        values = self.value_head(features)
        
        return action_logits, values.squeeze(-1)
        
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Single training step for PPO.
        
        Computes:
        - Policy loss (clipped surrogate objective)
        - Value function loss
        - Entropy bonus
        """
        # Unpack batch
        obs = batch['observations']
        actions = batch['actions']
        advantages = batch['advantages']
        returns = batch['returns']
        old_log_probs = batch['old_log_probs']
        
        # Forward pass
        action_logits, values = self(obs)
        
        # Compute action distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        
        # Policy loss (PPO clip)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Entropy loss
        entropy = action_dist.entropy().mean()
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss - 
            self.entropy_coef * entropy
        )
        
        # Update metrics
        self.train_policy_loss.update(policy_loss)
        self.train_value_loss.update(value_loss)
        self.train_entropy.update(entropy)
        
        # Log metrics
        self.log('train/policy_loss', policy_loss, prog_bar=True)
        self.log('train/value_loss', value_loss, prog_bar=True)
        self.log('train/entropy', entropy, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        # Log additional metrics
        self.log('train/clip_fraction', (torch.abs(ratio - 1) > self.clip_epsilon).float().mean())
        self.log('train/approx_kl', ((ratio - 1) - torch.log(ratio)).mean())
        
        return total_loss
        
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        """Validation step for monitoring performance."""
        # Compute validation metrics
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        
        # Forward pass
        action_logits, values = self(obs)
        
        # Compute metrics
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action_accuracy = (action_dist.probs.argmax(dim=-1) == actions).float().mean()
        
        # Log metrics
        self.log('val/action_accuracy', action_accuracy)
        self.log('val/mean_reward', rewards.mean())
        self.log('val/mean_value', values.mean())
        
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log aggregated metrics
        self.log('train/policy_loss_epoch', self.train_policy_loss.compute())
        self.log('train/value_loss_epoch', self.train_value_loss.compute())
        self.log('train/entropy_epoch', self.train_entropy.compute())
        
        # Reset metrics
        self.train_policy_loss.reset()
        self.train_value_loss.reset()
        self.train_entropy.reset()
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Group parameters
        param_groups = [
            {'params': self.model.parameters(), 'lr': self.hparams.ppo_config.learning_rate},
            {'params': self.value_head.parameters(), 'lr': self.hparams.ppo_config.learning_rate * 0.5}
        ]
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            param_groups,
            eps=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 1000,
            eta_min=self.hparams.ppo_config.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val/episode_reward'
            }
        }
        
    def predict_action(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Predict action for given observations.
        
        Args:
            observations: Current observations
            deterministic: If True, return most likely action
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        with torch.no_grad():
            action_logits, value = self(observations)
            
            # Create action distribution
            action_dist = torch.distributions.Categorical(logits=action_logits)
            
            # Sample or select action
            if deterministic:
                action = action_logits.argmax(dim=-1)
                log_prob = action_dist.log_prob(action)
            else:
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                
        return action.item(), log_prob.item(), value.item()
```

### Step 2: Create Data Module

```python
# fxai/lightning/data_module.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, IterableDataset
import numpy as np
from typing import Optional, Dict, List, Callable
from collections import deque

from envs.trading_environment import TradingEnvironment
from data.data_manager import DataManager
from simulators.rollout_buffer import RolloutBuffer

class RolloutDataset(Dataset):
    """Dataset for PPO rollout data."""
    
    def __init__(self, buffer: RolloutBuffer):
        self.buffer = buffer
        self._prepare_data()
        
    def _prepare_data(self):
        """Compute advantages and prepare data."""
        self.buffer.compute_returns_and_advantages()
        self.indices = np.arange(len(self.buffer))
        np.random.shuffle(self.indices)
        
    def __len__(self):
        return len(self.buffer)
        
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {
            'observations': self.buffer.get_observations(real_idx),
            'actions': self.buffer.actions[real_idx],
            'advantages': self.buffer.advantages[real_idx],
            'returns': self.buffer.returns[real_idx],
            'old_log_probs': self.buffer.old_log_probs[real_idx],
            'rewards': self.buffer.rewards[real_idx]
        }

class StreamingRolloutDataset(IterableDataset):
    """Streaming dataset for continuous rollout collection."""
    
    def __init__(self, env: TradingEnvironment, model: PPOLightningModule, buffer_size: int):
        self.env = env
        self.model = model
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
    def __iter__(self):
        """Continuously collect and yield rollout data."""
        obs = self.env.reset()
        
        while True:
            # Collect step
            action, log_prob, value = self.model.predict_action(obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Store transition
            transition = {
                'observations': obs,
                'action': action,
                'reward': reward,
                'done': done,
                'log_prob': log_prob,
                'value': value
            }
            
            self.buffer.append(transition)
            
            # Yield batch when buffer is full
            if len(self.buffer) >= self.buffer_size:
                yield self._prepare_batch()
                
            # Reset if done
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
                
    def _prepare_batch(self):
        """Prepare batch from buffer."""
        batch = {
            key: torch.stack([t[key] for t in self.buffer])
            for key in self.buffer[0].keys()
        }
        return batch

class RLDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for RL training.
    
    Handles:
    - Environment creation and management
    - Rollout collection
    - Batch generation
    - Data preprocessing
    """
    
    def __init__(
        self,
        data_config: Dict,
        env_config: Dict,
        rollout_length: int = 2048,
        batch_size: int = 64,
        num_workers: int = 0,
        prefetch_factor: int = 2
    ):
        super().__init__()
        self.data_config = data_config
        self.env_config = env_config
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # Components
        self.data_manager = None
        self.env = None
        self.rollout_buffer = None
        self.model = None  # Set by trainer
        
    def setup(self, stage: str) -> None:
        """Initialize components based on stage."""
        if stage == 'fit':
            # Initialize data manager
            self.data_manager = DataManager(self.data_config)
            
            # Create environment
            self.env = TradingEnvironment(
                data_manager=self.data_manager,
                **self.env_config
            )
            
            # Initialize rollout buffer
            self.rollout_buffer = RolloutBuffer(
                buffer_size=self.rollout_length,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            )
            
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        # For episodic training
        if self.rollout_buffer and len(self.rollout_buffer) > 0:
            dataset = RolloutDataset(self.rollout_buffer)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor
            )
        else:
            # For streaming training
            dataset = StreamingRolloutDataset(
                self.env,
                self.model,
                self.rollout_length
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=0,  # Streaming requires single worker
                pin_memory=True
            )
            
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        # Use separate validation episodes
        val_dataset = RolloutDataset(self.collect_validation_rollouts())
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
    def collect_rollouts(self, model: PPOLightningModule, n_steps: int) -> None:
        """Collect rollouts using current policy."""
        self.model = model  # Store reference
        self.rollout_buffer.reset()
        
        obs = self.env.reset()
        
        for step in range(n_steps):
            # Get action from model
            with torch.no_grad():
                action, log_prob, value = model.predict_action(
                    self._to_tensor(obs)
                )
            
            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Store transition
            self.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob
            )
            
            # Handle episode end
            if done:
                obs = self.env.reset()
                # Log episode metrics
                if 'episode' in info:
                    model.log('episode/reward', info['episode']['reward'])
                    model.log('episode/length', info['episode']['length'])
            else:
                obs = next_obs
                
    def collect_validation_rollouts(self, n_episodes: int = 10) -> RolloutBuffer:
        """Collect validation rollouts."""
        val_buffer = RolloutBuffer(
            buffer_size=n_episodes * 1000,  # Approximate
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            
            while not done:
                with torch.no_grad():
                    action, log_prob, value = self.model.predict_action(
                        self._to_tensor(obs)
                    )
                
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                val_buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    value=value,
                    log_prob=log_prob
                )
                
                obs = next_obs
                
        return val_buffer
        
    def _to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation to tensor."""
        return {
            key: torch.from_numpy(value).float()
            for key, value in obs.items()
        }
```

### Step 3: Create Custom Callbacks

```python
# fxai/lightning/callbacks.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, deque

class EpisodeMetricsCallback(Callback):
    """
    Track and log episode-level metrics.
    
    Tracks:
    - Episode rewards
    - Episode lengths
    - Success rates
    - Trading metrics (Sharpe, win rate, etc.)
    """
    
    def __init__(self, window_size: int = 100):
        super().__init__()
        self.window_size = window_size
        
        # Metrics storage
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_returns = deque(maxlen=window_size)
        self.episode_sharpes = deque(maxlen=window_size)
        self.episode_win_rates = deque(maxlen=window_size)
        
        # Current episode tracking
        self.current_episode = {
            'reward': 0.0,
            'length': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0
        }
        
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int
    ) -> None:
        """Update episode metrics after each batch."""
        # Check for episode ends in batch
        if 'done' in batch:
            done_mask = batch['done']
            rewards = batch['rewards']
            
            # Process completed episodes
            for i, done in enumerate(done_mask):
                if done:
                    self.current_episode['reward'] += rewards[i].item()
                    self._log_episode_end(pl_module)
                    self._reset_episode()
                else:
                    self.current_episode['reward'] += rewards[i].item()
                    self.current_episode['length'] += 1
                    
    def _log_episode_end(self, pl_module: pl.LightningModule) -> None:
        """Log metrics at episode end."""
        # Store metrics
        self.episode_rewards.append(self.current_episode['reward'])
        self.episode_lengths.append(self.current_episode['length'])
        
        # Calculate additional metrics
        if self.current_episode['trades'] > 0:
            win_rate = self.current_episode['wins'] / self.current_episode['trades']
            self.episode_win_rates.append(win_rate)
        
        # Log individual episode metrics
        pl_module.log('episode/reward', self.current_episode['reward'])
        pl_module.log('episode/length', self.current_episode['length'])
        
        # Log aggregated metrics
        if len(self.episode_rewards) > 0:
            pl_module.log('episode/reward_mean', np.mean(self.episode_rewards))
            pl_module.log('episode/reward_std', np.std(self.episode_rewards))
            pl_module.log('episode/reward_max', np.max(self.episode_rewards))
            pl_module.log('episode/reward_min', np.min(self.episode_rewards))
            
        if len(self.episode_lengths) > 0:
            pl_module.log('episode/length_mean', np.mean(self.episode_lengths))
            
        if len(self.episode_win_rates) > 0:
            pl_module.log('episode/win_rate_mean', np.mean(self.episode_win_rates))
            
    def _reset_episode(self) -> None:
        """Reset current episode tracking."""
        self.current_episode = {
            'reward': 0.0,
            'length': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0
        }

class RolloutCollectionCallback(Callback):
    """
    Manage rollout collection between training epochs.
    
    This callback:
    - Collects new rollouts before each epoch
    - Updates the old policy for PPO
    - Manages the rollout buffer
    """
    
    def __init__(self, n_steps: int = 2048, n_envs: int = 1):
        super().__init__()
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.total_timesteps = 0
        
    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Collect rollouts before training epoch."""
        # Get data module
        datamodule = trainer.datamodule
        
        # Collect rollouts
        pl_module.eval()  # Set to eval mode for rollout collection
        datamodule.collect_rollouts(pl_module, self.n_steps)
        pl_module.train()  # Back to train mode
        
        # Update timestep counter
        self.total_timesteps += self.n_steps * self.n_envs
        pl_module.log('timesteps/total', self.total_timesteps)
        
        # Store old policy for PPO
        pl_module.old_policy = {
            name: param.clone()
            for name, param in pl_module.model.named_parameters()
        }

class RewardSystemCallback(Callback):
    """
    Track and analyze reward system components.
    
    Provides detailed breakdown of:
    - Individual reward components
    - Component contributions
    - Reward statistics
    """
    
    def __init__(self, reward_system):
        super().__init__()
        self.reward_system = reward_system
        self.component_history = defaultdict(deque)
        self.history_size = 1000
        
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int
    ) -> None:
        """Track reward components after each batch."""
        # Get reward breakdown if available
        if hasattr(pl_module, 'last_reward_breakdown'):
            breakdown = pl_module.last_reward_breakdown
            
            # Store component values
            for component, value in breakdown.items():
                self.component_history[component].append(value)
                if len(self.component_history[component]) > self.history_size:
                    self.component_history[component].popleft()
                    
                # Log component metrics
                pl_module.log(f'reward/{component}', value)
                
            # Log aggregated metrics
            total_reward = sum(breakdown.values())
            pl_module.log('reward/total', total_reward)
            
            # Log component statistics
            for component, history in self.component_history.items():
                if len(history) > 0:
                    pl_module.log(f'reward/{component}_mean', np.mean(history))
                    pl_module.log(f'reward/{component}_std', np.std(history))

class CurriculumLearningCallback(Callback):
    """
    Implement curriculum learning for progressive training.
    
    Features:
    - Dynamic difficulty adjustment
    - Performance-based progression
    - Multi-stage curriculum
    """
    
    def __init__(self, curriculum_config: Dict):
        super().__init__()
        self.curriculum_config = curriculum_config
        self.current_stage = 0
        self.stage_performance = []
        self.promotion_threshold = curriculum_config.get('promotion_threshold', 0.7)
        
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Check for curriculum progression."""
        # Get current performance
        current_performance = trainer.callback_metrics.get('val/episode_reward', 0)
        self.stage_performance.append(current_performance)
        
        # Check if ready to advance
        if len(self.stage_performance) >= 10:
            avg_performance = np.mean(self.stage_performance[-10:])
            stage_config = self.curriculum_config['stages'][self.current_stage]
            
            if avg_performance >= stage_config['target_performance'] * self.promotion_threshold:
                self._advance_curriculum(trainer, pl_module)
                
    def _advance_curriculum(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Advance to next curriculum stage."""
        self.current_stage += 1
        
        if self.current_stage < len(self.curriculum_config['stages']):
            # Update environment difficulty
            new_stage = self.curriculum_config['stages'][self.current_stage]
            trainer.datamodule.env.set_difficulty(new_stage['difficulty'])
            
            # Log progression
            pl_module.log('curriculum/stage', self.current_stage)
            pl_module.log('curriculum/difficulty', new_stage['difficulty'])
            
            # Reset performance tracking
            self.stage_performance = []
            
            print(f"Advanced to curriculum stage {self.current_stage}: {new_stage['name']}")

class DistributedSyncCallback(Callback):
    """
    Synchronize environments and metrics in distributed training.
    
    Ensures:
    - Consistent environment states across processes
    - Proper metric aggregation
    - Synchronized random seeds
    """
    
    def __init__(self):
        super().__init__()
        self.sync_frequency = 100  # Sync every N steps
        self.step_count = 0
        
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict,
        batch_idx: int
    ) -> None:
        """Synchronize at batch start if needed."""
        self.step_count += 1
        
        if self.step_count % self.sync_frequency == 0 and trainer.world_size > 1:
            self._sync_environments(trainer)
            
    def _sync_environments(self, trainer: pl.Trainer) -> None:
        """Synchronize environment states across processes."""
        if trainer.is_global_zero:
            # Get environment state from rank 0
            env_state = trainer.datamodule.env.get_state()
        else:
            env_state = None
            
        # Broadcast state to all processes
        env_state = trainer.strategy.broadcast(env_state, src=0)
        
        # Set environment state on all processes
        if not trainer.is_global_zero:
            trainer.datamodule.env.set_state(env_state)
```

### Step 4: Create Training Script

```python
# fxai/lightning/train.py
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
    DeviceStatsMonitor
)
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from fxai.lightning import (
    PPOLightningModule,
    RLDataModule,
    EpisodeMetricsCallback,
    RolloutCollectionCallback,
    RewardSystemCallback,
    CurriculumLearningCallback,
    DistributedSyncCallback
)
from rewards.reward_system_v2 import RewardSystemV2

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Main training function with Lightning."""
    
    # Set seeds for reproducibility
    pl.seed_everything(cfg.training.seed)
    
    # Initialize reward system
    reward_system = RewardSystemV2(cfg.rewards)
    
    # Create model
    model = PPOLightningModule(
        model_config=cfg.model,
        ppo_config=cfg.training.ppo,
        reward_system=reward_system
    )
    
    # Create data module
    datamodule = RLDataModule(
        data_config=cfg.data,
        env_config=cfg.environment,
        rollout_length=cfg.training.rollout_length,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers
    )
    
    # Setup callbacks
    callbacks = [
        # Core callbacks
        RolloutCollectionCallback(
            n_steps=cfg.training.rollout_length,
            n_envs=cfg.training.n_envs
        ),
        EpisodeMetricsCallback(window_size=100),
        RewardSystemCallback(reward_system),
        
        # Checkpointing
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoints,
            filename='ppo-{epoch:04d}-{val_episode_reward:.3f}',
            monitor='val/episode_reward',
            mode='max',
            save_top_k=5,
            save_last=True,
            auto_insert_metric_name=False
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val/episode_reward',
            patience=cfg.training.early_stopping_patience,
            mode='max',
            verbose=True
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='step'),
        
        # Progress bar
        RichProgressBar(),
        
        # Device stats
        DeviceStatsMonitor()
    ]
    
    # Add optional callbacks
    if cfg.training.use_curriculum:
        callbacks.append(CurriculumLearningCallback(cfg.curriculum))
        
    if cfg.training.distributed:
        callbacks.append(DistributedSyncCallback())
    
    # Setup loggers
    loggers = [
        WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name or cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=cfg.paths.logs,
            log_model=True
        ),
        TensorBoardLogger(
            save_dir=cfg.paths.logs,
            name=cfg.experiment_name,
            version=cfg.version
        )
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        # Basic training config
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        
        # Hardware config
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        strategy=cfg.hardware.strategy,
        
        # Precision and optimization
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        
        # Callbacks and logging
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.training.log_every_n_steps,
        
        # Checkpointing
        enable_checkpointing=True,
        
        # Debugging
        detect_anomaly=cfg.debug.detect_anomaly,
        profiler=cfg.debug.profiler if cfg.debug.profile else None,
        
        # Other options
        deterministic=cfg.training.deterministic,
        benchmark=True  # Enable cudNN benchmarking
    )
    
    # Load from checkpoint if continuing
    ckpt_path = None
    if cfg.training.continue_from_checkpoint:
        ckpt_path = cfg.training.checkpoint_path
        print(f"Continuing from checkpoint: {ckpt_path}")
    
    # Train model
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path
    )
    
    # Test model
    if cfg.training.run_test:
        trainer.test(model=model, datamodule=datamodule)
    
    # Save final model
    if trainer.is_global_zero:
        final_path = f"{cfg.paths.models}/final_model.ckpt"
        trainer.save_checkpoint(final_path)
        print(f"Saved final model to: {final_path}")

if __name__ == "__main__":
    train()
```

## 3. Component Templates

### 3.1 Custom Optimizer Template

```python
# fxai/lightning/optimizers.py
from torch.optim import Optimizer
import torch

class PPOOptimizer(Optimizer):
    """Custom optimizer for PPO with adaptive learning rate."""
    
    def __init__(
        self,
        params,
        lr: float = 3e-4,
        eps: float = 1e-5,
        clip_range: float = 0.2,
        target_kl: float = 0.01
    ):
        defaults = dict(lr=lr, eps=eps, clip_range=clip_range, target_kl=target_kl)
        super().__init__(params, defaults)
        self.kl_history = []
        
    def step(self, closure=None):
        """Perform optimization step with adaptive learning rate."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            # Adapt learning rate based on KL divergence
            if len(self.kl_history) > 0:
                recent_kl = np.mean(self.kl_history[-10:])
                if recent_kl > group['target_kl'] * 2:
                    group['lr'] *= 0.5  # Reduce learning rate
                elif recent_kl < group['target_kl'] * 0.5:
                    group['lr'] *= 1.5  # Increase learning rate
                    
            # Standard Adam update
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Update biased first and second moment estimates
                exp_avg.mul_(0.9).add_(grad, alpha=0.1)
                exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)
                
                # Compute bias correction
                bias_correction1 = 1 - 0.9 ** state['step']
                bias_correction2 = 1 - 0.999 ** state['step']
                
                # Update parameters
                p.data.addcdiv_(
                    exp_avg / bias_correction1,
                    (exp_avg_sq / bias_correction2).sqrt().add_(group['eps']),
                    value=-group['lr']
                )
                
        return loss
```

### 3.2 Mixed Precision Training Template

```python
# fxai/lightning/mixed_precision.py
from pytorch_lightning.plugins import MixedPrecisionPlugin
import torch

class CustomMixedPrecision(MixedPrecisionPlugin):
    """Custom mixed precision handling for RL training."""
    
    def __init__(self, precision: str = "16-mixed", device: str = "cuda"):
        super().__init__(precision=precision, device=device)
        
        # Custom GradScaler for RL
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=True
        )
        
    def backward(
        self,
        model: pl.LightningModule,
        closure_loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *args,
        **kwargs
    ):
        """Custom backward pass with gradient scaling."""
        # Scale loss
        scaled_loss = self.scaler.scale(closure_loss)
        
        # Backward pass
        model.manual_backward(scaled_loss)
        
        # Unscale gradients for gradient clipping
        self.scaler.unscale_(optimizer)
        
        # Clip gradients
        if model.trainer.gradient_clip_val is not None:
            model.clip_gradients(
                optimizer,
                gradient_clip_val=model.trainer.gradient_clip_val,
                gradient_clip_algorithm=model.trainer.gradient_clip_algorithm
            )
        
    def optimizer_step(
        self,
        model: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        closure: Callable,
        **kwargs
    ):
        """Custom optimizer step with gradient scaling."""
        # Step optimizer with scaler
        self.scaler.step(optimizer)
        self.scaler.update()
        
        # Clear gradients
        optimizer.zero_grad()
```

### 3.3 Custom Logger Template

```python
# fxai/lightning/loggers.py
from pytorch_lightning.loggers import Logger
from typing import Dict, Any, Optional
import json
import os

class RLExperimentLogger(Logger):
    """Custom logger for RL experiments with detailed tracking."""
    
    def __init__(
        self,
        save_dir: str,
        name: str = "rl_experiment",
        version: Optional[str] = None
    ):
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._version = version or self._get_next_version()
        self._experiment_dir = os.path.join(save_dir, name, f"version_{self._version}")
        os.makedirs(self._experiment_dir, exist_ok=True)
        
        # Tracking structures
        self.metrics = defaultdict(list)
        self.hyperparameters = {}
        self.episodes = []
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def version(self) -> str:
        return self._version
        
    @property
    def save_dir(self) -> str:
        return self._save_dir
        
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.hyperparameters.update(params)
        
        # Save to file
        with open(os.path.join(self._experiment_dir, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f, indent=2)
            
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics."""
        for key, value in metrics.items():
            self.metrics[key].append({
                'step': step,
                'value': value
            })
            
        # Save periodically
        if step % 100 == 0:
            self._save_metrics()
            
    def log_episode(
        self,
        episode_data: Dict[str, Any]
    ) -> None:
        """Log complete episode data."""
        self.episodes.append(episode_data)
        
        # Save episode data
        episode_file = os.path.join(
            self._experiment_dir,
            f"episode_{len(self.episodes)}.json"
        )
        with open(episode_file, "w") as f:
            json.dump(episode_data, f, indent=2)
            
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        metrics_file = os.path.join(self._experiment_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(dict(self.metrics), f, indent=2)
            
    def _get_next_version(self) -> str:
        """Get next available version number."""
        versions = []
        if os.path.exists(os.path.join(self._save_dir, self._name)):
            for d in os.listdir(os.path.join(self._save_dir, self._name)):
                if d.startswith("version_"):
                    versions.append(int(d.split("_")[1]))
        return str(max(versions) + 1 if versions else 0)
```

## 4. Migration Patterns

### 4.1 Callback Migration Pattern

```python
# Original callback
class OriginalCallback(BaseCallback):
    def on_training_start(self, locals_: Dict[str, Any]) -> None:
        self.model = locals_["self"]
        self.start_time = time.time()
        
    def on_rollout_end(self) -> None:
        self.logger.record("rollout/ep_rew_mean", safe_mean(self.model.ep_info_buffer))
        
# Lightning callback
class LightningCallback(pl.Callback):
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.start_time = time.time()
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Access episode rewards from data module or module storage
        episode_rewards = trainer.datamodule.get_episode_rewards()
        pl_module.log("rollout/ep_rew_mean", np.mean(episode_rewards))
```

### 4.2 Training Loop Migration Pattern

```python
# Original training loop
def train_original():
    for update in range(1, n_updates + 1):
        # Collect rollouts
        rollout_buffer = collect_rollouts(env, model, n_steps)
        
        # Update policy
        for epoch in range(n_epochs):
            for batch in rollout_buffer.get_batches(batch_size):
                loss = compute_loss(model, batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
        # Logging
        logger.log({"loss": loss.item(), "update": update})

# Lightning training
def train_lightning():
    trainer = pl.Trainer(
        max_epochs=n_updates * n_epochs,
        val_check_interval=n_epochs,
        callbacks=[RolloutCollectionCallback(n_steps=n_steps)],
        logger=True
    )
    trainer.fit(model, datamodule)
```

### 4.3 Model Checkpoint Migration

```python
# Original checkpoint
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_stats': get_training_stats()
    }, path)

# Lightning checkpoint (automatic)
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='model-{epoch:02d}-{val_reward:.2f}',
    save_top_k=3,
    monitor='val_reward',
    mode='max'
)

# Manual checkpoint conversion
def convert_checkpoint(old_path: str, new_path: str):
    """Convert old checkpoint to Lightning format."""
    old_ckpt = torch.load(old_path)
    
    # Create Lightning checkpoint
    lightning_ckpt = {
        'epoch': old_ckpt['epoch'],
        'global_step': old_ckpt.get('global_step', 0),
        'pytorch-lightning_version': pl.__version__,
        'state_dict': {
            f'model.{k}': v for k, v in old_ckpt['model_state_dict'].items()
        },
        'optimizer_states': [old_ckpt['optimizer_state_dict']],
        'lr_schedulers': [],
        'callbacks': {},
        'hyper_parameters': {}
    }
    
    torch.save(lightning_ckpt, new_path)
```

## 5. Best Practices

### 5.1 Configuration Management

```yaml
# config/lightning_defaults.yaml
defaults:
  - _self_
  - model: multi_branch_transformer
  - training: ppo_lightning
  - environment: trading_env
  - data: databento
  - callbacks: default
  - hardware: auto

experiment_name: ${hydra:job.name}
seed: 42

trainer:
  # Training duration
  max_epochs: 1000
  max_steps: -1
  
  # Validation
  val_check_interval: 1.0
  check_val_every_n_epoch: 1
  
  # Optimization
  gradient_clip_val: 0.5
  accumulate_grad_batches: 1
  
  # Hardware
  accelerator: auto
  devices: auto
  strategy: auto
  
  # Precision
  precision: 32
  
  # Logging
  log_every_n_steps: 50
  
  # Debugging
  detect_anomaly: false
  profiler: null
  
  # Performance
  benchmark: true
  deterministic: false
```

### 5.2 Error Handling

```python
class RobustPPOModule(PPOLightningModule):
    """PPO module with enhanced error handling."""
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        """Training step with error handling."""
        try:
            # Validate inputs
            self._validate_batch(batch)
            
            # Compute loss
            loss = super().training_step(batch, batch_idx)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                self.log('error/nan_loss', 1.0)
                return None  # Skip this batch
                
            return loss
            
        except Exception as e:
            self.log('error/training_step', 1.0)
            self.logger.experiment.log({'error': str(e)})
            return None
            
    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate batch inputs."""
        required_keys = ['observations', 'actions', 'rewards', 'advantages', 'returns']
        
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key: {key}")
                
            if torch.isnan(batch[key]).any():
                raise ValueError(f"NaN values in {key}")
```

### 5.3 Performance Optimization

```python
class OptimizedDataModule(RLDataModule):
    """Optimized data module for fast training."""
    
    def setup(self, stage: str) -> None:
        """Setup with performance optimizations."""
        super().setup(stage)
        
        if stage == 'fit':
            # Pre-allocate buffers
            self._preallocate_buffers()
            
            # Enable environment vectorization
            if self.num_envs > 1:
                self.env = VectorizedEnv(
                    env_fn=lambda: self.env,
                    n_envs=self.num_envs
                )
                
            # Setup memory pinning
            if torch.cuda.is_available():
                self._setup_pinned_memory()
                
    def _preallocate_buffers(self) -> None:
        """Pre-allocate memory for efficiency."""
        # Pre-allocate rollout buffer
        self.rollout_buffer.preallocate(self.rollout_length)
        
        # Pre-allocate tensor storage
        self.obs_buffer = torch.zeros(
            (self.rollout_length, *self.env.observation_space.shape),
            dtype=torch.float32
        )
        
    def _setup_pinned_memory(self) -> None:
        """Setup pinned memory for GPU transfer."""
        self.pinned_buffer = torch.zeros(
            (self.batch_size, *self.env.observation_space.shape),
            dtype=torch.float32,
            pin_memory=True
        )
```

## 6. Troubleshooting

### 6.1 Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM errors | Large rollout buffers | Reduce `rollout_length` or use gradient accumulation |
| Slow training | Inefficient data loading | Increase `num_workers`, use `pin_memory=True` |
| NaN losses | Exploding gradients | Reduce learning rate, increase gradient clipping |
| Poor convergence | Wrong hyperparameters | Use Optuna integration for hyperparameter search |
| Distributed training hangs | Environment sync issues | Use `DistributedSyncCallback` |

### 6.2 Debugging Tools

```python
# Enable debugging mode
trainer = pl.Trainer(
    detect_anomaly=True,  # Detect NaN/Inf
    profiler="simple",    # Profile training
    limit_train_batches=10,  # Quick testing
    fast_dev_run=True,    # Single batch test
    overfit_batches=1     # Overfit single batch
)

# Custom debugging callback
class DebugCallback(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Check inputs
        for key, value in batch.items():
            if torch.isnan(value).any():
                raise ValueError(f"NaN in {key}")
                
    def on_after_backward(self, trainer, pl_module):
        # Check gradients
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
```

### 6.3 Performance Profiling

```python
# Profile with PyTorch Profiler
from pytorch_lightning.profilers import PyTorchProfiler

profiler = PyTorchProfiler(
    filename="profile_report",
    export_to_chrome=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True
)

trainer = pl.Trainer(profiler=profiler)

# Analyze results
# View profile_report.json in Chrome://tracing
```

## Conclusion

This implementation guide provides a comprehensive roadmap for integrating PyTorch Lightning into the FxAI system. The modular approach allows for incremental adoption while maintaining backward compatibility. Key benefits include:

1. **Cleaner Architecture**: Separation of concerns with Lightning modules
2. **Better Performance**: Automatic optimizations and distributed training
3. **Improved Debugging**: Built-in profiling and error detection
4. **Easier Maintenance**: Standardized patterns and less boilerplate

Follow the step-by-step implementation, use the provided templates, and refer to the troubleshooting section when encountering issues. The migration will result in a more robust, scalable, and maintainable training infrastructure for FxAI.