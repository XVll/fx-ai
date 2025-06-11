"""
Typed model state for clear and consistent model management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


@dataclass
class TrainingState:
    """Training state information."""
    global_step: int = 0
    global_episode: int = 0
    global_update: int = 0
    global_cycle: int = 0
    
    # Learning rate tracking
    current_lr: float = 0.0
    lr_schedule_state: Optional[Dict[str, Any]] = None
    
    # Performance tracking
    best_reward: float = float('-inf')
    best_reward_update: int = 0
    recent_rewards: List[float] = field(default_factory=list)
    
    # Training statistics
    total_timesteps: int = 0
    wall_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'global_step': self.global_step,
            'global_episode': self.global_episode,
            'global_update': self.global_update,
            'global_cycle': self.global_cycle,
            'current_lr': self.current_lr,
            'lr_schedule_state': self.lr_schedule_state,
            'best_reward': self.best_reward,
            'best_reward_update': self.best_reward_update,
            'recent_rewards': self.recent_rewards,
            'total_timesteps': self.total_timesteps,
            'wall_time_seconds': self.wall_time_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        """Create from dictionary."""
        return cls(
            global_step=data.get('global_step', 0),
            global_episode=data.get('global_episode', 0),
            global_update=data.get('global_update', 0),
            global_cycle=data.get('global_cycle', 0),
            current_lr=data.get('current_lr', 0.0),
            lr_schedule_state=data.get('lr_schedule_state'),
            best_reward=data.get('best_reward', float('-inf')),
            best_reward_update=data.get('best_reward_update', 0),
            recent_rewards=data.get('recent_rewards', []),
            total_timesteps=data.get('total_timesteps', 0),
            wall_time_seconds=data.get('wall_time_seconds', 0.0),
        )


@dataclass
class ModelMetadata:
    """Metadata about a saved model."""
    version: int
    timestamp: datetime
    reward: float
    
    # Training context
    symbol: Optional[str] = None
    day_quality: Optional[float] = None
    episode_count: int = 0
    update_count: int = 0
    
    # Model architecture
    model_class: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    evaluation_results: Optional[Dict[str, Any]] = None
    
    # File information
    file_path: Optional[Path] = None
    file_size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    
    # Training configuration
    training_config: Optional[Dict[str, Any]] = None
    
    # Additional notes
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'reward': self.reward,
            'symbol': self.symbol,
            'day_quality': self.day_quality,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'model_class': self.model_class,
            'model_config': self.model_config,
            'metrics': self.metrics,
            'evaluation_results': self.evaluation_results,
            'file_path': str(self.file_path) if self.file_path else None,
            'file_size_bytes': self.file_size_bytes,
            'checksum': self.checksum,
            'training_config': self.training_config,
            'notes': self.notes,
            'tags': self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        timestamp = data.get('timestamp', '')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()
            
        file_path = data.get('file_path')
        if file_path and not isinstance(file_path, Path):
            file_path = Path(file_path)
            
        return cls(
            version=data.get('version', 0),
            timestamp=timestamp,
            reward=data.get('reward', 0.0),
            symbol=data.get('symbol'),
            day_quality=data.get('day_quality'),
            episode_count=data.get('episode_count', 0),
            update_count=data.get('update_count', 0),
            model_class=data.get('model_class'),
            model_config=data.get('model_config'),
            metrics=data.get('metrics', {}),
            evaluation_results=data.get('evaluation_results'),
            file_path=file_path,
            file_size_bytes=data.get('file_size_bytes'),
            checksum=data.get('checksum'),
            training_config=data.get('training_config'),
            notes=data.get('notes'),
            tags=data.get('tags', []),
        )


@dataclass
class ModelState:
    """Complete state of a model including weights, optimizer, training state, and metadata."""
    
    # Core model components
    model_state_dict: Optional[Dict[str, torch.Tensor]] = None
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    
    # Training state
    training_state: TrainingState = field(default_factory=TrainingState)
    
    # Metadata
    metadata: ModelMetadata = field(default_factory=lambda: ModelMetadata(
        version=0,
        timestamp=datetime.now(),
        reward=0.0
    ))
    
    # Validation flags
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def validate(self) -> bool:
        """Validate the model state."""
        self.validation_errors.clear()
        
        if self.model_state_dict is None:
            self.validation_errors.append("Model state dict is missing")
            
        if not self.metadata.version >= 0:
            self.validation_errors.append("Invalid version number")
            
        if self.training_state.global_step < 0:
            self.validation_errors.append("Invalid global step count")
            
        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid
    
    def to_checkpoint(self) -> Dict[str, Any]:
        """Convert to checkpoint format for saving."""
        return {
            'model_state_dict': self.model_state_dict,
            'optimizer_state_dict': self.optimizer_state_dict,
            'training_state': self.training_state.to_dict(),
            'metadata': self.metadata.to_dict(),
            'checkpoint_version': '2.0',  # Version of checkpoint format
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint: Dict[str, Any]) -> 'ModelState':
        """Create from checkpoint dictionary."""
        # Handle legacy checkpoint format
        if 'checkpoint_version' not in checkpoint:
            # Legacy format conversion
            training_state = TrainingState(
                global_step=checkpoint.get('global_step_counter', 0),
                global_episode=checkpoint.get('global_episode_counter', 0),
                global_update=checkpoint.get('global_update_counter', 0),
                global_cycle=checkpoint.get('global_cycle_counter', 0),
            )
            
            metadata_dict = checkpoint.get('metadata', {})
            metadata = ModelMetadata(
                version=metadata_dict.get('version', 0),
                timestamp=datetime.fromisoformat(checkpoint.get('timestamp', datetime.now().isoformat())),
                reward=metadata_dict.get('reward', 0.0),
                metrics=metadata_dict.get('metrics', {}),
            )
        else:
            # New format
            training_state = TrainingState.from_dict(checkpoint.get('training_state', {}))
            metadata = ModelMetadata.from_dict(checkpoint.get('metadata', {}))
            
        return cls(
            model_state_dict=checkpoint.get('model_state_dict'),
            optimizer_state_dict=checkpoint.get('optimizer_state_dict'),
            training_state=training_state,
            metadata=metadata,
        )
    
    def update_from_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update state from training metrics."""
        if 'mean_reward' in metrics:
            reward = metrics['mean_reward']
            self.training_state.recent_rewards.append(reward)
            
            # Keep only last 100 rewards
            if len(self.training_state.recent_rewards) > 100:
                self.training_state.recent_rewards = self.training_state.recent_rewards[-100:]
                
            # Update best reward
            if reward > self.training_state.best_reward:
                self.training_state.best_reward = reward
                self.training_state.best_reward_update = self.training_state.global_update
                
        # Update metadata metrics
        self.metadata.metrics.update(metrics)