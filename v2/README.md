# FxAI v2 Architecture

## Overview

The v2 architecture is a complete redesign focusing on:
- **Modularity**: Clear interfaces enable component swapping
- **Testability**: Dependency injection and mocking support
- **Flexibility**: Multiple training modes and configurations
- **Scalability**: Distributed training and deployment ready

## Core Design Principles

### 1. Interface-Driven Development
Every component has a clear interface (Protocol) that defines its contract. This enables:
- Easy testing with mocks
- Component substitution
- Clear boundaries
- Better documentation

### 2. Separation of Concerns
The architecture separates:
- **Interfaces** (`/core`): Contracts and types
- **Implementations** (`/impl`): Concrete implementations
- **Tests** (`/tests`): Comprehensive test suites
- **Apps** (`/apps`): User-facing applications

### 3. Dependency Injection
Components receive dependencies through constructors:
```python
class TradingEnvironment:
    def __init__(self, 
                 market_sim: IMarketSimulator,
                 portfolio_sim: IPortfolioSimulator,
                 ...):
        # Store injected dependencies
```

## Architecture Layers

### Core Layer (`/core`)

#### Types (`/core/types`)
- Common type definitions
- Enums for actions, positions, etc.
- Structured data types (TypedDict)
- Protocol definitions

#### Interfaces by Domain

**Agent** (`/core/agent`)
- `IAgent`: Base agent interface
- `ITrainableAgent`: Training capabilities
- `IPPOAgent`: PPO-specific interface
- `IAgentFactory`: Agent creation

**Data** (`/core/data`)
- `IDataProvider`: Market data access
- `IDataManager`: High-level data orchestration
- `IFeatureExtractor`: Feature engineering
- `IMomentumScanner`: Momentum detection

**Features** (`/core/features`)
- `IFeature`: Individual feature interface
- `IFeatureGroup`: Related features
- `IFeatureRegistry`: Feature discovery
- `IFeatureStore`: Feature persistence

**Training** (`/core/training`)
- `ITrainingMode`: Base training mode
- `IStandardTrainingMode`: Episode-based training
- `IContinuousTrainingMode`: Never-ending improvement
- `IOptunaMode`: Hyperparameter optimization
- `IBenchmarkMode`: Performance evaluation

**Simulation** (`/core/simulation`)
- `IMarketSimulator`: Market data replay
- `IExecutionSimulator`: Order execution
- `IPortfolioSimulator`: Portfolio tracking
- `IBacktestEngine`: Strategy backtesting

**Rewards** (`/core/rewards`)
- `IRewardComponent`: Individual reward aspects
- `IRewardCalculator`: Reward aggregation
- `IRewardAnalyzer`: Reward analysis

**Environment** (`/core/environment`)
- `ITradingEnvironment`: Gymnasium-compatible env
- `IActionMaskableEnvironment`: Action masking
- `ICurriculumEnvironment`: Progressive difficulty

**Configuration** (`/core/config`)
- `IConfigProvider`: Configuration loading
- `IConfigSchema`: Validation schemas
- `IModeConfig`: Mode-specific configs

**Monitoring** (`/core/monitoring`)
- `IMetricsCollector`: Metrics collection
- `ILogger`: Structured logging
- `IMonitor`: High-level monitoring
- `IProfiler`: Performance profiling

### Implementation Layer (`/impl`)

Concrete implementations of all interfaces:

```
/impl
├── agent/
│   ├── ppo_agent.py
│   └── sac_agent.py
├── data/
│   ├── databento_provider.py
│   └── data_manager.py
├── training/
│   ├── standard_mode.py
│   ├── continuous_mode.py
│   ├── optuna_mode.py
│   └── benchmark_mode.py
└── ...
```

### Application Layer (`/apps`)

User-facing applications:

```
/apps
├── train/          # Training application
├── backtest/       # Backtesting tools
├── optimize/       # Hyperparameter optimization
├── dashboard/      # Web dashboard
└── cli/           # Command-line interface
```

## Training Modes

### Standard Training Mode
Traditional RL training with fixed episodes:
```python
mode = StandardTrainingMode()
mode.set_training_schedule(total_episodes=1000)
results = mode.run()
```

### Continuous Training Mode
Never-ending improvement with model versioning:
```python
mode = ContinuousTrainingMode()
mode.set_curriculum(curriculum_stages)
mode.set_improvement_criteria(metric="sharpe_ratio")
mode.run()  # Runs indefinitely
```

### Optuna Mode
Hyperparameter optimization:
```python
mode = OptunaMode()
mode.set_search_space(param_specs)
mode.set_optimization_config(n_trials=100)
best_params = mode.get_best_params()
```

### Benchmark Mode
Standardized performance evaluation:
```python
mode = BenchmarkMode()
mode.set_benchmark_suite(test_episodes)
mode.add_baseline("v1_model", model_path)
results = mode.get_benchmark_results()
```

## Configuration System

Hierarchical configuration with Hydra:

```yaml
# config/modes/continuous.yaml
mode: continuous
agent:
  type: ppo
  learning_rate: 3e-4
environment:
  max_episode_steps: 1000
curriculum:
  stages:
    - name: easy
      symbols: [AAPL, MSFT]
      min_quality: 0.7
```

## Testing Strategy

### Unit Tests
Test individual components with mocked dependencies:
```python
def test_agent_prediction():
    mock_env = Mock(spec=ITradingEnvironment)
    agent = PPOAgent(config, mock_env)
    action = agent.predict(observation)
    assert action.shape == expected_shape
```

### Integration Tests
Test component interactions:
```python
def test_training_mode():
    agent = create_test_agent()
    env = create_test_environment()
    mode = StandardTrainingMode()
    mode.initialize(agent, env, config)
    results = mode.run()
    assert results['episodes_completed'] == 100
```

### End-to-End Tests
Test complete workflows:
```python
def test_continuous_training_workflow():
    # Test full training pipeline
    # with real data and components
```

## Key Benefits

1. **Modularity**: Swap components without changing code
2. **Testability**: Mock any component for isolated testing
3. **Flexibility**: Multiple training modes and strategies
4. **Maintainability**: Clear interfaces and contracts
5. **Scalability**: Ready for distributed training
6. **Extensibility**: Easy to add new components

## Getting Started

1. Define your configuration
2. Choose a training mode
3. Select or implement components
4. Run training

```python
# Example: Start continuous training
from v2.apps.train import TrainingApp

app = TrainingApp()
app.load_config("configs/continuous_momentum.yaml")
app.start(mode="continuous")
```

## Future Extensions

- **Multi-Agent**: Coordinated trading strategies
- **Live Trading**: Real-time execution
- **Cloud Deployment**: Kubernetes operators
- **Model Zoo**: Pre-trained model repository
- **AutoML**: Automated architecture search
