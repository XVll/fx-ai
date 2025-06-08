"""
Core interfaces and types for the v2 trading system.

This package provides all interfaces, protocols, and type definitions
needed to build a modular, testable trading system.
"""

# Version info
__version__ = "2.0.0"
__author__ = "FxAI Team"

# Make key interfaces easily accessible
from .types.common import (
    # Basic types
    Symbol, Timestamp, Price, Volume, Quantity, Cash, PnL, Reward,
    # Enums
    ActionType, PositionSizeType, OrderType, OrderSide, PositionSide,
    TerminationReason, RunMode, FeatureFrequency,
    # Structured types
    MarketDataPoint, ExecutionInfo, EpisodeMetrics, ModelCheckpoint,
    # Protocols
    Configurable, Resettable, Serializable
)

from .agent.interfaces import (
    IAgent, ITrainableAgent, IPPOAgent,
    IAgentFactory, IAgentCallback
)

from .data.interfaces import (
    IDataProvider, IDataManager, IFeatureExtractor,
    IDataCache, IMomentumScanner
)

from .features.interfaces import (
    IFeature, IFeatureGroup, IFeatureRegistry,
    IFeatureStore, IFeaturePipeline,
    FeatureType
)

from .training.interfaces import (
    ITrainingMode, IStandardTrainingMode, IContinuousTrainingMode,
    IOptunaMode, IBenchmarkMode, ITrainingManager, ITrainingMonitor
)

from .simulation.interfaces import (
    IMarketSimulator, IExecutionSimulator, IPortfolioSimulator,
    ISimulationOrchestrator, IBacktestEngine
)

from .rewards.interfaces import (
    IRewardComponent, IPnLComponent, IActionPenaltyComponent,
    IRiskComponent, IRewardCalculator, IRewardAnalyzer,
    IRewardComponentFactory
)

from .environment.interfaces import (
    ITradingEnvironment, IActionMaskableEnvironment,
    ICurriculumEnvironment, IMultiAssetEnvironment,
    IEnvironmentMetrics, IEnvironmentFactory,
    IEnvironmentWrapper, IRewardWrapper, IObservationWrapper
)

from .config.interfaces import (
    IConfigSchema, IConfigProvider, IConfigRegistry,
    IModeConfig, IConfigBuilder, IConfigValidator,
    IConfigManager
)

from .monitoring.interfaces import (
    IMetric, IMetricsCollector, ILogger, IEventLogger,
    IMonitor, IProfiler, IHealthChecker,
    MetricType, LogLevel
)

from .optimization import (
    OptimizationState, TrialState,
    IHyperparameterSpace, ITrial, IObjective, ISampler, IPruner, IStudy,
    IOptunaIntegration, IHyperparameterOptimizer
)

from .attribution import (
    AttributionMethod, AttributionTarget,
    IAttributionResult, IBaselineProvider, IFeatureAttributor, IAttributionAnalyzer,
    ICaptumIntegration, IAttributionCache, IAttributionVisualizer
)

__all__ = [
    # Types
    "Symbol", "Timestamp", "Price", "Volume", "Quantity", "Cash", "PnL", "Reward",
    "ActionType", "PositionSizeType", "OrderType", "OrderSide", "PositionSide",
    "TerminationReason", "RunMode", "FeatureFrequency",
    "MarketDataPoint", "ExecutionInfo", "EpisodeMetrics", "ModelCheckpoint",
    "Configurable", "Resettable", "Serializable",
    
    # Agent interfaces
    "IAgent", "ITrainableAgent", "IPPOAgent", "IAgentFactory", "IAgentCallback",
    
    # Data interfaces
    "IDataProvider", "IDataManager", "IFeatureExtractor", "IDataCache", "IMomentumScanner",
    
    # Feature interfaces
    "IFeature", "IFeatureGroup", "IFeatureRegistry", "IFeatureStore", "IFeaturePipeline",
    "FeatureType",
    
    # Training interfaces
    "ITrainingMode", "IStandardTrainingMode", "IContinuousTrainingMode",
    "IOptunaMode", "IBenchmarkMode", "ITrainingManager", "ITrainingMonitor",
    
    # Simulation interfaces
    "IMarketSimulator", "IExecutionSimulator", "IPortfolioSimulator",
    "ISimulationOrchestrator", "IBacktestEngine",
    
    # Reward interfaces
    "IRewardComponent", "IPnLComponent", "IActionPenaltyComponent",
    "IRiskComponent", "IRewardCalculator", "IRewardAnalyzer",
    "IRewardComponentFactory",
    
    # Environment interfaces
    "ITradingEnvironment", "IActionMaskableEnvironment",
    "ICurriculumEnvironment", "IMultiAssetEnvironment",
    "IEnvironmentMetrics", "IEnvironmentFactory",
    "IEnvironmentWrapper", "IRewardWrapper", "IObservationWrapper",
    
    # Config interfaces
    "IConfigSchema", "IConfigProvider", "IConfigRegistry",
    "IModeConfig", "IConfigBuilder", "IConfigValidator",
    "IConfigManager",
    
    # Monitoring interfaces
    "IMetric", "IMetricsCollector", "ILogger", "IEventLogger",
    "IMonitor", "IProfiler", "IHealthChecker",
    "MetricType", "LogLevel",
    
    # Optimization interfaces
    "OptimizationState", "TrialState",
    "IHyperparameterSpace", "ITrial", "IObjective", "ISampler", "IPruner", "IStudy",
    "IOptunaIntegration", "IHyperparameterOptimizer",
    
    # Attribution interfaces
    "AttributionMethod", "AttributionTarget",
    "IAttributionResult", "IBaselineProvider", "IFeatureAttributor", "IAttributionAnalyzer",
    "ICaptumIntegration", "IAttributionCache", "IAttributionVisualizer",
]
