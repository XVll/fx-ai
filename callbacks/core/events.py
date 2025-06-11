"""
Event system for callbacks - defines all available events and their metadata.

This module provides a comprehensive event system with 30+ event types covering
all aspects of the training lifecycle.
"""

from enum import Enum, auto
from typing import Optional, Set, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


class EventType(Enum):
    """Comprehensive enumeration of all callback event types."""
    
    # Training lifecycle events
    TRAINING_START = auto()          # Training initialization complete
    TRAINING_END = auto()            # Training finalized
    TRAINING_ERROR = auto()          # Training error occurred
    
    # Episode lifecycle events
    EPISODE_START = auto()           # Episode initialized
    EPISODE_END = auto()             # Episode completed
    EPISODE_RESET = auto()           # Environment reset
    EPISODE_TERMINATED = auto()      # Episode terminated (natural end)
    EPISODE_TRUNCATED = auto()       # Episode truncated (timeout/limit)
    
    # Step events
    STEP_START = auto()              # Before environment step
    STEP_END = auto()                # After environment step
    ACTION_SELECTED = auto()         # After action selection
    REWARD_COMPUTED = auto()         # After reward computation
    
    # Rollout events
    ROLLOUT_START = auto()           # Before rollout collection
    ROLLOUT_END = auto()             # After rollout collection
    BUFFER_ADD = auto()              # Data added to replay buffer
    BUFFER_READY = auto()            # Buffer ready for training
    
    # Update/Training events
    UPDATE_START = auto()            # Before policy update
    UPDATE_END = auto()              # After policy update
    GRADIENT_COMPUTED = auto()       # After gradient computation
    OPTIMIZER_STEP = auto()          # After optimizer step
    BATCH_START = auto()             # Before processing batch
    BATCH_END = auto()               # After processing batch
    EPOCH_START = auto()             # Before training epoch
    EPOCH_END = auto()               # After training epoch
    
    # Model events
    MODEL_SAVED = auto()             # Model checkpoint saved
    MODEL_LOADED = auto()            # Model loaded from checkpoint
    MODEL_IMPROVED = auto()          # Model performance improved
    LEARNING_RATE_UPDATED = auto()   # Learning rate changed
    
    # Evaluation events
    EVALUATION_START = auto()        # Evaluation phase started
    EVALUATION_END = auto()          # Evaluation phase ended
    EVALUATION_EPISODE = auto()      # Single evaluation episode
    
    # Data events
    DATA_LOADED = auto()             # New data loaded
    DAY_SWITCHED = auto()            # Switched to new trading day
    SYMBOL_SWITCHED = auto()         # Switched to new symbol
    
    # Performance events
    MEMORY_WARNING = auto()          # Memory usage high
    PERFORMANCE_LOG = auto()         # Performance metrics logged
    
    # Custom events
    CUSTOM = auto()                  # User-defined custom event


class EventPriority(Enum):
    """Event priority levels for callback execution order."""
    CRITICAL = 0     # Execute first (e.g., error handlers)
    HIGH = 1         # High priority (e.g., model saving)
    NORMAL = 2       # Default priority
    LOW = 3          # Low priority (e.g., logging)
    BACKGROUND = 4   # Execute last (e.g., cleanup)


@dataclass
class EventMetadata:
    """Metadata associated with an event."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None                    # Component that triggered event
    priority: EventPriority = EventPriority.NORMAL
    is_async: bool = False                          # Whether event can be handled asynchronously
    tags: Set[str] = field(default_factory=set)     # Custom tags for filtering
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass 
class EventFilter:
    """Filter criteria for event selection."""
    event_types: Optional[Set[EventType]] = None    # Only these event types
    exclude_types: Optional[Set[EventType]] = None  # Exclude these event types
    sources: Optional[Set[str]] = None              # Only from these sources
    tags: Optional[Set[str]] = None                 # Must have these tags
    min_priority: Optional[EventPriority] = None    # Minimum priority level


class EventRegistry:
    """Registry of event metadata and documentation."""
    
    # Event documentation
    EVENT_DOCS = {
        EventType.TRAINING_START: "Triggered once when training begins, after all components are initialized",
        EventType.TRAINING_END: "Triggered once when training completes or is terminated",
        EventType.TRAINING_ERROR: "Triggered when a training error occurs",
        
        EventType.EPISODE_START: "Triggered at the start of each episode, after environment reset",
        EventType.EPISODE_END: "Triggered at the end of each episode, before environment reset",
        EventType.EPISODE_RESET: "Triggered when environment is reset for new episode",
        EventType.EPISODE_TERMINATED: "Triggered when episode ends naturally (e.g., task completed)",
        EventType.EPISODE_TRUNCATED: "Triggered when episode is cut short (e.g., max steps)",
        
        EventType.STEP_START: "Triggered before each environment step",
        EventType.STEP_END: "Triggered after each environment step",
        EventType.ACTION_SELECTED: "Triggered after action is selected by policy",
        EventType.REWARD_COMPUTED: "Triggered after reward is calculated",
        
        EventType.ROLLOUT_START: "Triggered before collecting rollout data",
        EventType.ROLLOUT_END: "Triggered after rollout collection completes",
        EventType.BUFFER_ADD: "Triggered when experience is added to replay buffer",
        EventType.BUFFER_READY: "Triggered when buffer has enough data for training",
        
        EventType.UPDATE_START: "Triggered before policy update begins",
        EventType.UPDATE_END: "Triggered after policy update completes",
        EventType.GRADIENT_COMPUTED: "Triggered after gradients are computed",
        EventType.OPTIMIZER_STEP: "Triggered after optimizer step",
        EventType.BATCH_START: "Triggered before processing training batch",
        EventType.BATCH_END: "Triggered after processing training batch",
        EventType.EPOCH_START: "Triggered at start of training epoch",
        EventType.EPOCH_END: "Triggered at end of training epoch",
        
        EventType.MODEL_SAVED: "Triggered when model checkpoint is saved",
        EventType.MODEL_LOADED: "Triggered when model is loaded from checkpoint",
        EventType.MODEL_IMPROVED: "Triggered when model performance improves",
        EventType.LEARNING_RATE_UPDATED: "Triggered when learning rate is updated",
        
        EventType.EVALUATION_START: "Triggered when evaluation phase begins",
        EventType.EVALUATION_END: "Triggered when evaluation phase completes",
        EventType.EVALUATION_EPISODE: "Triggered for each evaluation episode",
        
        EventType.DATA_LOADED: "Triggered when new market data is loaded",
        EventType.DAY_SWITCHED: "Triggered when switching to new trading day",
        EventType.SYMBOL_SWITCHED: "Triggered when switching to new symbol",
        
        EventType.MEMORY_WARNING: "Triggered when memory usage exceeds threshold",
        EventType.PERFORMANCE_LOG: "Triggered periodically with performance metrics",
        
        EventType.CUSTOM: "User-defined custom event"
    }
    
    # Event categories for grouping
    EVENT_CATEGORIES = {
        "lifecycle": {
            EventType.TRAINING_START, EventType.TRAINING_END, EventType.TRAINING_ERROR
        },
        "episode": {
            EventType.EPISODE_START, EventType.EPISODE_END, EventType.EPISODE_RESET,
            EventType.EPISODE_TERMINATED, EventType.EPISODE_TRUNCATED
        },
        "step": {
            EventType.STEP_START, EventType.STEP_END, EventType.ACTION_SELECTED,
            EventType.REWARD_COMPUTED
        },
        "rollout": {
            EventType.ROLLOUT_START, EventType.ROLLOUT_END, EventType.BUFFER_ADD,
            EventType.BUFFER_READY
        },
        "update": {
            EventType.UPDATE_START, EventType.UPDATE_END, EventType.GRADIENT_COMPUTED,
            EventType.OPTIMIZER_STEP, EventType.BATCH_START, EventType.BATCH_END,
            EventType.EPOCH_START, EventType.EPOCH_END
        },
        "model": {
            EventType.MODEL_SAVED, EventType.MODEL_LOADED, EventType.MODEL_IMPROVED,
            EventType.LEARNING_RATE_UPDATED
        },
        "evaluation": {
            EventType.EVALUATION_START, EventType.EVALUATION_END, 
            EventType.EVALUATION_EPISODE
        },
        "data": {
            EventType.DATA_LOADED, EventType.DAY_SWITCHED, EventType.SYMBOL_SWITCHED
        },
        "performance": {
            EventType.MEMORY_WARNING, EventType.PERFORMANCE_LOG
        }
    }
    
    @classmethod
    def get_event_info(cls, event_type: EventType) -> Dict[str, Any]:
        """Get comprehensive information about an event type."""
        return {
            "type": event_type,
            "name": event_type.name,
            "description": cls.EVENT_DOCS.get(event_type, "No description available"),
            "categories": [cat for cat, events in cls.EVENT_CATEGORIES.items() if event_type in events]
        }
    
    @classmethod
    def get_events_by_category(cls, category: str) -> Set[EventType]:
        """Get all events in a category."""
        return cls.EVENT_CATEGORIES.get(category, set())
    
    @classmethod
    def validate_event_type(cls, event_type: EventType) -> bool:
        """Validate that an event type is registered."""
        return event_type in cls.EVENT_DOCS