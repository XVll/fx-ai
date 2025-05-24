# W&B Tools for AI Trading System: **User Guide**
Todo: 
- [ ] Apply proper normalization to features
---

## 2 · Installation & Setup

### Initial setup

```bash
# Log in to W&B
wandb login
```
 For this task we created files inside metrics folder to set foundation manager, transmitters, collectors,integrators etc, now we will continue to integrate rest of the project, ignore obselete files after these changes and list them so I will delete. While implementing changes adjust logging too, log important stuff, warning and errors since we do already keep track all metrics no need to log them again, I like to read console, when there is 100 log flowing every second it becomes unreadable. Also unneceassary log configuration and preperation since we do not use live dashboard they are not need ed just rich handler.
### Optimizing hyperparameters

```bash
# Run a sweep
python sweep.py --config default.yaml --count 20

# Train with optimized parameters
python main.py model.d_model=96 training.lr=0.0002 env.reward_scaling=2.5

# Customize the sweep
python sweep.py --config default.yaml --name "lr_and_layers_sweep" --count 20
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
# Train several ai with different configurations
python main.py wandb.enabled=true model=transformer_small
python main.py wandb.enabled=true model=transformer_medium
python main.py wandb.enabled=true model=transformer_large

```