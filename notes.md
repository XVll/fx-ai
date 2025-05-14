# W&B Tools for AI Trading System: **User Guide**
---

## 2 · Installation & Setup

### Initial setup

```bash
# Log in to W&B
wandb login
```

### Optimizing hyperparameters

```bash
# Run a sweep
python run_sweep.py --config sweep_config.yaml --count 20

# Train with optimized parameters
python main.py model.d_model=96 training.lr=0.0002 env.reward_scaling=2.5

# Customize the sweep
python run_sweep.py --config sweep_config.yaml --name "lr_and_layers_sweep" --count 20
```

### Analyzing trading performance

```bash
# Train a model
python main.py wandb.enabled=true

# With custom configurations
python main.py wandb.enabled=true wandb.log_frequency.steps=10 wandb.log_model=true

```

### Comparing multiple models

```bash
# Train several models with different configurations
python main.py wandb.enabled=true model=transformer_small
python main.py wandb.enabled=true model=transformer_medium
python main.py wandb.enabled=true model=transformer_large

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