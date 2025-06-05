# Training Initialization Log Analysis & Refactoring

## Current Problematic Log Flow:

```
🛡️ Installed graceful shutdown signal handlers
📋 Loading config overrides from: config/overrides/training.yaml
📋 Config overrides applied successfully  
📊 Action space: 3 types × 4 sizes = 12 total actions
📊 Enabled reward components: pnl, holding_penalty, drawdown_penalty, etc.

🚀 Starting FX-AI Training System
📊 Experiment: momentum_training
📈 Symbols: ['MLGO']  
🧠 Model: d_model=128, layers=6
🎯 Action space: 3×4

🔧 Using MPS (Apple Silicon) device
📂 Initializing data provider: databento
📊 Scanning for metadata in 'dnb/mlgo'...
📊 Found 6 directories, indexed 183 files
📊 Loaded 42 momentum days from index
📊 Loaded 5035 reset points from index
📊 Loaded momentum index with 42 days
🎯 Momentum-based training enabled

✅ Model created successfully
🌐 Starting dashboard server on port 8051
✅ Callback system initialized with: WandBCallback, DashboardCallback
📂 Loading best model: cache/model/best/model_v45_reward0.0000_20250605_142742.pt
📂 Loaded model: model_v45_reward0.0000_20250605_142742.pt
✅ Model loaded: step=0
⏹️ Early stopping enabled (patience: 300)
🔄 Continuing from previous best mean_reward: 0.0000
🔄 Continuous training callback added
🎯 Momentum tracking callback added
📊 ReplayBuffer initialized with capacity 2048 on device mps
🤖 PPOTrainer initialized with callback system. Device: mps

🎯 TrainingManager will handle data lifecycle initialization...
🚀 Starting TrainingManager-controlled training
   Training will complete based on TrainingManager termination criteria
🎯 DataLifecycleManager initialized for adaptive data management
📊 Loaded best model: reward=0.0000  # DUPLICATE!
🔄 ContinuousTraining initialized in production mode (enabled: True)
🎯 TrainingManager initialized in production mode
🎯 Starting training with TrainingManager in production mode  # DUPLICATE!
🚀 Training started by TrainingManager  # DUPLICATE!

🎯 Initializing DataLifecycleManager...
🔄 Initialized 3 reset points for day 2025-03-26 00:00:00 in sequential mode
🔄 Advanced to reset point 1/3 (cycle 0) in day 2025-03-26 00:00:00
🏙️ Advanced to new day: 2025-03-26 00:00:00 (MLGO)
✅ Data lifecycle initialized with day: 2025-03-26 00:00:00
✅ DataLifecycleManager initialized successfully

🎯 ROLLOUT START: Collecting 2048 steps
   ℹ️ PPO collects data across multiple episodes before training
   ℹ️ Episodes will reset automatically when they complete
📊 ReplayBuffer cleared.
📅 Using training day: 2025-03-26 (quality: 0.997)
📍 Updated dashboard with 1 reset points
📅 Setting up session: MLGO on 2025-03-26 (quality: 0.997)
🎯 Setting up session: MLGO on 2025-03-26  # DUPLICATE MESSAGE!

🔧 FeatureCacheManager initialized with cache_dir: cache/features
🔧 Initializing market simulator for MLGO on 2025-03-26 00:00:00
🔧 Creating new feature cache for MLGO 2025-03-26
🔧 Feature cache session loaded for MLGO 2025-03-26
🔧 Cache miss for MLGO on 2025-03-26 00:00:00, loading from disk...

📊 DATA LOADING (48.11s):
   📊 Loaded 4205 1s bars for MLGO
   📊 Loaded 604 1m bars for MLGO
   📊 No 5m bar files found for MLGO
   📊 No direct 5m bars found for MLGO, building from 1m bars
   📊 Loaded 604 1m bars for MLGO  # DUPLICATE!
   📊 Built 159 5m bars from 1m bars for MLGO
   📊 Loaded 1 1d bars for MLGO
   📊 Dashboard available at: http://localhost:8051
   📊 Loaded 10692 trades for MLGO
   📊 Loaded 122665 quotes for MLGO
   📊 Loaded 5 status records for MLGO
   📈 Data Load Summary:
      ⏱️ Duration: 48.11s
      📊 Total rows: 138,331
      ✅ Successful: 7 data types
         • MLGO_quotes: 122,665 rows
         • MLGO_trades: 10,692 rows
         • MLGO_bars_1s: 4,205 rows

📊 MORE DATA LOADING (48.19s):  # DUPLICATE LOADING!
   [Same process repeats for next day warmup]

🔧 FEATURE PROCESSING:
   📊 Using cached previous day data for warmup
   📊 Pre-loading MLGO for 2025-03-27
   📊 Combined data spans 143836 seconds with warmup
   📊 Starting vectorized pre-computation of 57601 market states...
   📊 Processing quotes vectorized...
   [More loading...]
   📊 Skipping feature pre-computation for faster initialization...
   📊 Features will be calculated on-demand during training
   📊 Feature dimensions per timestamp (computed on-demand):
      - HF: 60x9 = 540 values
      - MF: 30x43 = 1290 values  
      - LF: 30x19 = 570 values
   📊 Placeholder memory usage: 22.0 MB for 57601 states
   📊 Completed FAST initialization of 57601 market states
   📊 Successfully initialized 57601 market states with warmup data
   ✅ Session ready: 57601 seconds, warmup: True

🎯 EPISODE SETUP:
   📊 Using 159 reset points (momentum + early fixed supplements)
   📊 Portfolio reset - Capital: $25,000.00
   📊 Enabled P&L reward (coefficient: 100.0)
   [All reward components listed...]
   📊 Initialized percentage-based reward system with 10 components
   📊 Action masking initialized - max_position_ratio: 1.0
   ✅ All simulators initialized
   🔄 159 reset points available for training
   🎯 Episode 1 reset: 15:00:00 → 14:59:31 (-1m/±1m) | Activity: 1.00 | Combined: 0.95 | ROC: 0.09 | momentum
   📊 Portfolio reset - Capital: $25,000.00  # DUPLICATE!
   📊 Sent initial chart data: 958 candles

❌ ERROR: Error in WandBCallback.on_model_forward: You must call wandb.init() before wandb.log()
```

## Issues Identified:

1. **Duplicate Messages**: Multiple "Starting training", "Setting up session", "Portfolio reset"
2. **Redundant Model Loading**: Best model loaded twice with same info
3. **Excessive Data Loading Logs**: Each file type logged multiple times
4. **WandB Not Initialized**: Callback trying to log before wandb.init()
5. **Confusing Flow**: TrainingManager and PPOTrainer both announcing starts
6. **Verbose Feature Processing**: Too many technical details during warmup

## Proposed Clean Log Flow:

```
🚀 SYSTEM STARTUP
├── 🛡️ Graceful shutdown handlers installed
├── 📋 Config loaded: training.yaml
├── 🧠 Model: d_model=128, layers=6, actions=12
├── 🔧 Device: MPS (Apple Silicon)
└── ✅ Core components ready

📂 DATA INITIALIZATION  
├── 📊 Momentum Index: 42 days, 5035 reset points
├── 📂 Model: v45 (reward: 0.0000) loaded
├── 🌐 Dashboard: http://localhost:8051
└── ✅ Infrastructure ready

🎯 TRAINING MANAGER STARTUP
├── 🔄 DataLifecycleManager: adaptive mode
├── 📅 Selected day: 2025-03-26 (quality: 0.997)
├── 🔄 Reset points: 3 available (sequential mode)
└── ✅ Training lifecycle ready

📊 SESSION SETUP: MLGO 2025-03-26
├── 🔄 Loading market data... (48s)
│   ├── 📊 OHLCV: 4,205×1s + 604×1m + 159×5m + 1×1d
│   ├── 📊 Trades: 10,692 records  
│   ├── 📊 Quotes: 122,665 records
│   └── 📊 Total: 138,331 rows
├── 🔄 Warmup data loaded (next day)
├── 🔧 Market states: 57,601 ready (features on-demand)
└── ✅ Session ready: 159 reset points available

🎯 EPISODE 1 START
├── 📊 Portfolio: $25,000 capital
├── 🎯 Reset: 15:00:00 → 14:59:31 (momentum, score: 0.95)
├── 🔄 Reward system: 10 components active
└── 🚀 Training begins...
```