# W&B Tools for AI Trading System: **User Guide**

## 1 · Overview of Available Tools

| Tool | Description | Usage |
|------|-------------|-------|
| **WandbCallback** | Tracks training, trading performance, and generates visualizations | Added during training |
| **Dashboard** | Real-time monitoring of experiments and trading performance | Launched via `dashboard.py` |
| **Hyperparameter Sweep** | Optimizes model parameters automatically | Configured and run via `run_sweep.py` |
| **Report Generator** | Creates performance and trade analysis reports | Run via `utils/wandb_reports.py` |
| **Model Analyzer** | Analyzes model behavior, feature importance, etc. | Run via `utils/model_analysis.py` |
| **Enhanced Logger** | Structured logging with W&B integration | Used throughout your codebase |

---

## 2 · Installation & Setup

### Initial setup
```bash
# Install required packages
pip install -r requirements.txt

# Log in to W&B
wandb login

# Set environment variables (optional)
export WANDB_PROJECT="ai-trading"
export WANDB_ENTITY="your-username"   # Optional
```

### Configuration
Edit **`config/wandb.yaml`** to customize:

- Project name  
- Logging frequency  
- Visualization settings  
- Model checkpoint saving  

---

## 3 · Using the Tools

### Training with W&B tracking
```bash
# Enable W&B tracking in a training run
python main.py wandb.enabled=true 

# With custom configurations
python main.py wandb.enabled=true wandb.log_frequency.steps=10 wandb.log_model=true
```

### Hyperparameter optimization
```bash
# Run a sweep with default settings
python run_sweep.py --config sweep_config.yaml

# Customize the sweep
python run_sweep.py --config sweep_config.yaml --name "lr_and_layers_sweep" --count 20
```
You can edit **`sweep_config.yaml`** to adjust:

- Learning-rate ranges  
- Model dimensions  
- Training durations  
- Reward-scaling factors  

### Real-time dashboard
```bash
# Launch with default settings
python dashboard.py

# Specify project and entity
python dashboard.py --project ai-trading --entity your-username

# Change port
python dashboard.py --port 8502
```
The dashboard provides:

- Training-metrics visualization  
- Trade-performance analytics  
- Model comparisons  
- Win/Loss analysis by time and duration  

### Generating reports
```bash
# Generate performance-comparison report
python -m utils.wandb_reports --project ai-trading --report-type performance --limit 5

# Generate trade analysis for a specific run
python -m utils.wandb_reports --project ai-trading --report-type trade --run-id abc123

# Custom output format
python -m utils.wandb_reports --project ai-trading --format md --output-dir ./my-reports
```

### Model analysis
```bash
# Comprehensive model analysis
python -m utils.model_analysis --model-path models/best_model.pt                                --config-path models/config.json                                --log-to-wandb

# Specific analysis types
python -m utils.model_analysis --model-path models/best_model.pt                                --config-path models/config.json                                --analysis-type feature
```
Available `--analysis-type` values:

- `feature` – Feature-importance visualization  
- `action` – Action-pattern analysis  
- `latent` – Latent-space visualization  
- `sensitivity` – Market-sensitivity analysis  
- `all` – Run every analysis  

---

## 4 · Common Workflows

### Initial model training
```bash
# Train with W&B tracking enabled
python main.py wandb.enabled=true

# View results on the dashboard
python dashboard.py
```

### Optimizing hyperparameters
```bash
# Run a sweep
python run_sweep.py --config sweep_config.yaml --count 20

# View sweep results
python dashboard.py

# Train with optimized parameters
python main.py model.d_model=96 training.lr=0.0002 env.reward_scaling=2.5
```

### Analyzing trading performance
```bash
# Train a model
python main.py wandb.enabled=true

# Generate trade-analysis report
python -m utils.wandb_reports --project ai-trading                               --report-type trade                               --run-id YOUR_RUN_ID

# Analyze model behavior
python -m utils.model_analysis --model-path outputs/models/best_model.pt                                --config-path outputs/config.json
```

### Comparing multiple models
```bash
# Train several models with different configurations
python main.py wandb.enabled=true model=transformer_small
python main.py wandb.enabled=true model=transformer_medium
python main.py wandb.enabled=true model=transformer_large

# Generate performance-comparison report
python -m utils.wandb_reports --project ai-trading --report-type performance --limit 3

# View comparison on the dashboard
python dashboard.py
```

---

## 5 · W&B Reports & Visualizations

### Trading performance
- **Cumulative P&L** – Profit growth over time  
- **Win/Loss Distribution** – Breakdown of gains vs. losses  
- **Trade Duration Analysis** – Impact of holding time on returns  
- **Win Rate by Hour** – Most profitable trading hours  

### Model behavior
- **Feature Importance** – Drivers behind trading decisions  
- **Action Patterns** – Reactions to market conditions  
- **Market Sensitivity** – Effect of price changes on position sizing  
- **Latent Space** – Internal representation of market states  

### Training progress
- **Reward Curves** – Learning trajectory  
- **Loss Components** – Policy vs. value-function losses  
- **Gradient Norms** – Detecting optimization instabilities  
- **Action Distributions** – Policy evolution over time  

---

## 6 · Troubleshooting

| Issue | Possible fixes |
|-------|----------------|
| **W&B not logging data** | • Check your internet connection<br>• Verify login via `wandb status`<br>• Ensure `wandb.enabled=true` in your config |
| **Missing visualizations** | • Confirm required data (trades, state data) is produced<br>• Verify settings in `config/wandb.yaml` |
| **Dashboard not showing data** | • Check project/entity names<br>• Confirm runs finished successfully |

For deeper issues, inspect logs under the `wandb/` directory or contact W&B support.

---

## 7 · Advanced Customization

### Custom metrics  
Extend **`WandbCallback`** and override `on_step` / `on_episode_end` to log additional metrics.

### Custom visualizations  
Edit **`visualization/dashboard.py`** to add bespoke components.

### Integration with a live-trading platform
```python
import wandb

wandb.init(project="ai-trading", job_type="live-trading")

def on_trade_completed(trade):
    wandb.log({
        "live/entry_price":      trade.entry_price,
        "live/exit_price":       trade.exit_price,
        "live/realized_pnl":     trade.realized_pnl,
        "live/duration_seconds": trade.duration_seconds
    })
```

---

**Happy experimenting & profitable trading!**
