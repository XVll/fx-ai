# Dashboard module for momentum-based trading
# Main components for the momentum trading dashboard

# Avoid circular imports by doing conditional imports
try:
    from .dashboard import MomentumDashboard
except ImportError:
    MomentumDashboard = None

try:
    from .dashboard_integration import DashboardMetricsCollector, MockDashboardCollector  
except ImportError:
    DashboardMetricsCollector = None
    MockDashboardCollector = None

try:
    from .dashboard_data import DashboardState, MomentumDay, TrainingState, RewardComponents
except ImportError:
    DashboardState = None
    MomentumDay = None
    TrainingState = None
    RewardComponents = None

__all__ = ['MomentumDashboard', 'DashboardMetricsCollector', 'MockDashboardCollector', 
          'DashboardState', 'MomentumDay', 'TrainingState', 'RewardComponents']