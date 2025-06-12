"""
Simplified but powerful callback system.

Provides essential callback functionality without overwhelming complexity:
- Multiple event hooks (16 key events)
- Simple state management
- Error isolation
- Easy to understand and extend
"""

from typing import Optional, Dict, Any
from abc import ABC
import logging
from pathlib import Path
import json

from core.shutdown import IShutdownHandler, get_global_shutdown_manager


class BaseCallback(IShutdownHandler, ABC):
    """
    Simple but powerful callback base class.
    
    Provides 16 key events covering all important training aspects:
    - Training lifecycle: start, end
    - Episodes: start, end
    - Steps: start, end, action_selected
    - Updates: start, end
    - Model: saved, improved
    - Data: day_switched
    - Performance: memory_warning
    - Custom: custom_event
    """
    
    def __init__(self, name: Optional[str] = None, enabled: bool = True, config: Optional[Dict[str, Any]] = None):
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.config = config or {}
        self.logger = logging.getLogger(f"callback.{self.name}")
        
        # Simple state storage
        self._state: Dict[str, Any] = {}
        
        # Basic counters
        self.episodes_seen = 0
        self.updates_seen = 0
        self.steps_seen = 0
        
        # Component references (set by manager)
        self.trainer = None
        self.environment = None
        self.data_manager = None
        
        self.logger.debug(f"Initialized {self.name}")
    
    # Core event hooks (16 essential events)
    
    def on_training_start(self, context: Dict[str, Any]) -> None:
        """Called once when training starts."""
        pass
    
    def on_training_end(self, context: Dict[str, Any]) -> None:
        """Called once when training ends."""
        pass
    
    def on_episode_start(self, context: Dict[str, Any]) -> None:
        """Called at the start of each episode."""
        self.episodes_seen += 1
    
    def on_episode_end(self, context: Dict[str, Any]) -> None:
        """Called at the end of each episode."""
        pass
    
    def on_step_start(self, context: Dict[str, Any]) -> None:
        """Called before each environment step."""
        self.steps_seen += 1
    
    def on_step_end(self, context: Dict[str, Any]) -> None:
        """Called after each environment step."""
        pass
    
    def on_action_selected(self, context: Dict[str, Any]) -> None:
        """Called after action is selected by policy."""
        pass
    
    def on_rollout_start(self, context: Dict[str, Any]) -> None:
        """Called before rollout collection."""
        pass
    
    def on_rollout_end(self, context: Dict[str, Any]) -> None:
        """Called after rollout collection."""
        pass
    
    def on_update_start(self, context: Dict[str, Any]) -> None:
        """Called before policy update."""
        self.updates_seen += 1
    
    def on_update_end(self, context: Dict[str, Any]) -> None:
        """Called after policy update."""
        pass
    
    def on_model_saved(self, context: Dict[str, Any]) -> None:
        """Called when model is saved."""
        pass
    
    def on_model_improved(self, context: Dict[str, Any]) -> None:
        """Called when model performance improves."""
        pass
    
    def on_day_switched(self, context: Dict[str, Any]) -> None:
        """Called when switching to new trading day."""
        pass
    
    def on_evaluation_complete(self, context: Dict[str, Any]) -> None:
        """Called when model evaluation completes."""
        pass
    
    def on_custom_event(self, event_name: str, context: Dict[str, Any]) -> None:
        """Called for custom events."""
        pass
    
    # Simple state management
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set state value."""
        self._state[key] = value
    
    def increment(self, key: str, amount: float = 1) -> float:
        """Increment a numeric value."""
        current = self.get_state(key, 0)
        new_value = current + amount
        self.set_state(key, new_value)
        return new_value
    
    def save_state(self, filepath: Optional[Path] = None) -> None:
        """Save state to JSON file."""
        if not filepath:
            filepath = Path(f"outputs/{self.name}_state.json")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state_data = {
            'state': self._state,
            'counters': {
                'episodes_seen': self.episodes_seen,
                'updates_seen': self.updates_seen,
                'steps_seen': self.steps_seen
            },
            'config': self.config
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            self.logger.debug(f"State saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def load_state(self, filepath: Optional[Path] = None) -> bool:
        """Load state from JSON file."""
        if not filepath:
            filepath = Path(f"outputs/{self.name}_state.json")
        
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            self._state = state_data.get('state', {})
            counters = state_data.get('counters', {})
            self.episodes_seen = counters.get('episodes_seen', 0)
            self.updates_seen = counters.get('updates_seen', 0)
            self.steps_seen = counters.get('steps_seen', 0)
            
            self.logger.debug(f"State loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return False
    
    # Utility methods
    
    def enable(self) -> None:
        """Enable callback."""
        self.enabled = True
        self.logger.debug("Enabled")
    
    def disable(self) -> None:
        """Disable callback."""
        self.enabled = False
        self.logger.debug("Disabled")
    
    def register_shutdown(self) -> None:
        """Register with shutdown manager."""
        shutdown_manager = get_global_shutdown_manager()
        shutdown_manager.register_component(
            component=self,
            timeout=10.0,
            name=f"callback_{self.name}"
        )
    
    def shutdown(self) -> None:
        """Cleanup on shutdown."""
        self.save_state()
        self.logger.debug("Shutdown complete")
    

    def __repr__(self) -> str:
        return f"{self.name}(enabled={self.enabled}, episodes={self.episodes_seen})"


