"""Test script for training manager panel visualization"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from datetime import datetime
from dashboard.panels.training_manager_panel import TrainingManagerPanel

# Define dark theme colors for testing
DARK_THEME = {
    "bg_primary": "#0f1419",
    "bg_secondary": "#1a1f29",
    "bg_tertiary": "#242c3a",
    "text_primary": "#e6e8ec",
    "text_secondary": "#8b92a3",
    "text_muted": "#5a6374",
    "accent_blue": "#4a9eff",
    "accent_green": "#00d4aa",
    "accent_red": "#ff4757",
    "accent_orange": "#ff9f43",
    "accent_purple": "#9c88ff",
    "border": "#2a3441",
}

@dataclass
class MockDashboardState:
    """Mock dashboard state with all required fields"""
    # Continuous training
    continuous_training_active: bool = True
    training_mode: str = "production"
    training_intensity: str = "continuous"
    performance_trend: str = "improving"
    updates_since_improvement: int = 12
    best_performance_episode: int = 234
    mean_episode_reward: float = 0.0456
    best_reward: float = 0.0512
    evaluation_frequency: int = 100
    checkpoint_frequency: int = 50
    
    # Training lifecycle
    training_max_episodes: float = 1000
    training_max_updates: float = 5000
    training_max_cycles: float = 10
    total_episodes: int = 456
    total_updates: int = 2345
    updates: int = 2345
    cycles_completed: int = 4
    cycle_count: int = 4
    training_hours: float = 3.5
    session_start_time: datetime = datetime.now()
    overall_progress: float = 45.6
    episodes_to_next_stage: int = 544
    next_stage_name: str = "Episode Limit"
    episodes_until_termination: int = 544
    updates_until_termination: int = 2655
    cycles_until_termination: int = 6
    termination_progress_pct: float = 45.6
    termination_reason: str = None
    
    # Day lifecycle
    current_momentum_day_date: str = "2025-04-15"
    current_symbol: str = "AAPL"
    current_momentum_day_quality: float = 0.876
    data_lifecycle_stage: str = "adaptive"
    stage_progress: float = 67.8
    episodes_on_current_day: int = 123
    target_cycles_per_day: int = 10
    day_switch_progress_pct: float = 40.0
    day_switch_in_progress: bool = False
    next_day_date: str = "2025-04-16"
    next_day_quality: float = 0.923
    episodes_until_day_switch: int = 27
    updates_until_day_switch: int = 135
    cycles_remaining_for_day_switch: int = 6
    day_score_range: list = field(default_factory=lambda: [0.5, 0.95])
    selection_mode: str = "adaptive"
    
    # Reset points
    current_reset_point_index: int = 15
    total_reset_points: int = 30
    current_cycle: int = 2
    cycle_progress: float = 50.0
    total_available_points: int = 30
    points_used_in_cycle: int = 15
    points_remaining_in_cycle: int = 15
    current_roc_score: float = 0.234
    current_activity_score: float = 0.567
    selected_reset_point_timestamp: str = "2025-04-15T09:45:00"
    reset_point_reuse_count: int = 1
    max_reset_point_reuse: int = 3
    reset_points_exhausted: bool = False
    reset_points_low_warning: bool = False
    preload_in_progress: bool = False
    preload_ready: bool = True
    preload_progress_pct: float = 100.0
    next_stage_preloaded: str = "Next momentum day"
    roc_range: list = field(default_factory=lambda: [0.1, 0.8])
    activity_range: list = field(default_factory=lambda: [0.2, 0.9])


def test_training_manager_panel():
    """Test the training manager panel with mock data"""
    print("Testing Training Manager Panel...")
    
    # Create panel
    panel = TrainingManagerPanel(DARK_THEME)
    
    # Create mock state
    state = MockDashboardState()
    
    # Test different scenarios
    scenarios = [
        ("Normal Training", {}),
        ("Performance Declining", {"performance_trend": "declining", "updates_since_improvement": 75}),
        ("Day Switch In Progress", {"day_switch_in_progress": True, "episodes_until_day_switch": 2}),
        ("Reset Points Low", {"reset_points_low_warning": True, "points_remaining_in_cycle": 3}),
        ("Training Near Completion", {"overall_progress": 95.0, "episodes_to_next_stage": 50}),
        ("Preloading Next Stage", {"preload_in_progress": True, "preload_progress_pct": 45.0}),
    ]
    
    for scenario_name, updates in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print('='*60)
        
        # Update state for scenario
        test_state = MockDashboardState()
        for key, value in updates.items():
            setattr(test_state, key, value)
        
        # Generate content
        content = panel.create_content(test_state)
        
        # Print the structure (simplified representation)
        print_component_tree(content, indent=0)
        
    print("\n✅ All scenarios tested successfully!")


def print_component_tree(component, indent=0):
    """Print a simplified representation of the component tree"""
    prefix = "  " * indent
    
    if hasattr(component, 'children'):
        # It's a component with children
        comp_type = type(component).__name__
        print(f"{prefix}{comp_type}")
        
        children = component.children
        if isinstance(children, list):
            for child in children:
                print_component_tree(child, indent + 1)
        elif children is not None:
            print_component_tree(children, indent + 1)
    elif hasattr(component, 'children') and component.children is None:
        # Component with no children
        comp_type = type(component).__name__
        print(f"{prefix}{comp_type}")
    elif isinstance(component, str):
        # Text content
        print(f"{prefix}'{component}'")
    else:
        # Other types
        print(f"{prefix}{type(component).__name__}: {str(component)[:50]}")


if __name__ == "__main__":
    test_training_manager_panel()