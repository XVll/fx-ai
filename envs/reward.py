from typing import Dict


class MomentumTradingReward:
    """
    Reward function for momentum trading strategy.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the reward function with configuration.
        """
        self.config = config or {}

        # Extract config with fallback to defaults
        if hasattr(self.config, "_to_dict"):
            config = self.config._to_dict()
        else:
            config = self.config

        # Reward scaling
        self.reward_scaling = config.get('reward_scaling', 1.0)
        self.trade_penalty = config.get('trade_penalty', 0.1)
        self.hold_penalty = config.get('hold_penalty', 0.0)
        self.early_exit_bonus = config.get('early_exit_bonus', 0.5)
        self.flush_prediction_bonus = config.get('flush_prediction_bonus', 2.0)

        # Strategy parameters
        strategy = config.get('strategy', {})
        self.momentum_threshold = strategy.get('momentum_threshold', 0.5)
        self.volume_surge_threshold = strategy.get('volume_surge_threshold', 3.0)

        # Tape analysis parameters
        tape = strategy.get('tape_analysis', {})
        self.tape_speed_threshold = tape.get('speed_threshold', 3.0)
        self.tape_imbalance_threshold = tape.get('imbalance_threshold', 0.7)

    def __call__(self, env, action, portfolio_change, portfolio_change_pct,
                 trade_executed, info):
        """
        Calculate reward based on momentum trading strategy.
        """
        # Base reward is portfolio change
        reward = portfolio_change * self.reward_scaling

        # Apply trading penalties
        if trade_executed:
            reward -= self.trade_penalty
        else:
            reward -= self.hold_penalty

        # Bonus for predicting a flush
        if info.get('predicted_flush', False) and info.get('actual_flush', False):
            reward += self.flush_prediction_bonus

        # Bonus for early exit before a flush
        if info.get('early_exit', False) and info.get('subsequent_flush', False):
            reward += self.early_exit_bonus

        # Bonus for capturing momentum with high volume
        if (trade_executed and
                info.get('momentum_strength', 0) > self.momentum_threshold and
                info.get('relative_volume', 0) > self.volume_surge_threshold):
            reward *= 1.5  # 50% boost for capturing strong momentum

        # Penalty for missing obvious momentum
        if (not trade_executed and
                info.get('momentum_strength', 0) > self.momentum_threshold * 2 and
                info.get('relative_volume', 0) > self.volume_surge_threshold * 2):
            reward -= self.trade_penalty * 2

        return reward