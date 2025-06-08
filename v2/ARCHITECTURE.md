# v2 Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │ Training │  │ Backtest │  │ Optimize │  │Dashboard│  │
│  │   App    │  │   App    │  │   App    │  │   App   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬─────┘  │
├────────┼───────────┼───────────┼───────────┼─────────┤
│                    Orchestration Layer                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Training Mode Manager                     │  │
│  │  ┌─────────┐ ┌──────────┐ ┌───────┐ ┌────────┐  │  │
│  │  │Standard │ │Continuous│ │Optuna │ │Benchmark│  │  │
│  │  │  Mode   │ │  Mode    │ │ Mode  │ │  Mode  │  │  │
│  │  └─────────┘ └──────────┘ └───────┘ └────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Core Components                         │
│  ┌─────────┐  ┌────────────┐  ┌────────┐  ┌─────────┐  │
│  │  Agent  │  │Environment │  │Rewards │  │ Monitor │  │
│  │  (PPO)  │  │  (Trading) │  │ System │  │  System │  │
│  └───┬─────┘  └─────┬──────┘  └───┬────┘  └───┬─────┘  │
│      │             │            │           │          │
│  ┌───┴───┐  ┌────┴────┐  ┌───┴───┐  ┌───┴───┐  ┌───┴───┐  │
│  │Feature│  │ Market  │  │ Port- │  │ Exec  │  │ Data  │  │
│  │Extract│  │Simulator│  │ folio │  │ Sim   │  │Manager│  │
│  └───────┘  └─────────┘  └───────┘  └───────┘  └───────┘  │
├─────────────────────────────────────────────────────────────┤
│                       Data Layer                             │
│  ┌────────────────────┐  ┌─────────────────────────┐  │
│  │   Market Data      │  │   Configuration Files   │  │
│  │ (Databento Files)  │  │   (YAML/JSON Configs)   │  │
│  └────────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Patterns

### 1. Dependency Injection
```python
class TradingEnvironment:
    def __init__(self, 
                 market_sim: IMarketSimulator,
                 portfolio_sim: IPortfolioSimulator,
                 execution_sim: IExecutionSimulator,
                 reward_calc: IRewardCalculator):
        # All dependencies injected, not created
        self.market_sim = market_sim
        self.portfolio_sim = portfolio_sim
        # ...
```

### 2. Factory Pattern
```python
# Centralized creation with validation
factory = AgentFactory()
agent = factory.create_agent("ppo", config)
```

### 3. Strategy Pattern (Training Modes)
```python
# Different algorithms, same interface
mode: ITrainingMode = ContinuousMode()  # or StandardMode, OptunaMode, etc.
mode.initialize(agent, env, config)
results = mode.run()
```

### 4. Composite Pattern (Rewards)
```python
# Combine multiple reward components
reward_calc = RewardCalculator()
reward_calc.add_component(PnLComponent(weight=1.0))
reward_calc.add_component(RiskComponent(weight=0.2))
reward_calc.add_component(ActionPenaltyComponent(weight=0.1))
```

### 5. Observer Pattern (Callbacks)
```python
# Multiple observers for training events
callbacks = [WandbCallback(), CheckpointCallback(), EarlyStoppingCallback()]
mode.run(callbacks=callbacks)
```

## Interface Hierarchy

```
Protocol (Base Interfaces)
├── Configurable
├── Resettable  
├── Serializable
│
├── IAgent
│   └── ITrainableAgent
│       └── IPPOAgent
│
├── ITradingEnvironment
│   ├── IActionMaskableEnvironment
│   ├── ICurriculumEnvironment
│   └── IMultiAssetEnvironment
│
├── ITrainingMode
│   ├── IStandardTrainingMode
│   ├── IContinuousTrainingMode
│   ├── IOptunaMode
│   └── IBenchmarkMode
│
└── ...
```

## Data Flow

```
1. Data Loading:
   DataProvider → DataManager → MarketSimulator

2. Feature Extraction:
   MarketSimulator → FeatureExtractor → Environment

3. Decision Making:
   Environment → Agent → Action

4. Execution:
   Action → ExecutionSimulator → PortfolioSimulator

5. Reward Calculation:
   PortfolioState → RewardCalculator → Reward

6. Learning:
   Experience → Agent.train_step() → Updated Policy
```

## Configuration Flow

```yaml
# Top-level mode configuration
mode: continuous
mode_config:
  curriculum:
    - stage: easy
      symbols: [AAPL, MSFT]
  improvement_threshold: 0.01

# Component configurations
agent:
  type: ppo
  learning_rate: 3e-4
  
environment:
  max_steps: 1000
  
reward:
  components:
    - type: pnl
      weight: 1.0
```

## Benefits of This Architecture

1. **Testability**: Mock any component for isolated testing
2. **Flexibility**: Swap implementations without changing code
3. **Modularity**: Clear boundaries between components
4. **Extensibility**: Add new modes/components easily
5. **Maintainability**: Changes isolated to specific modules
6. **Type Safety**: Interfaces provide compile-time checks
7. **Documentation**: Interfaces serve as contracts
