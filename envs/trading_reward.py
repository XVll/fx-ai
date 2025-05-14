from config.config import RewardConfig


class TradingReward:
    """
    Reward function for momentum trading strategy.
    """

    def __init__(self, config: RewardConfig):
        """
        Initialize the reward function with a strongly-typed configuration.

        Args:
            config: Typed RewardConfig object
        """
        # Direct attribute access from the typed config
        self.reward_scaling = config.scaling
        self.trade_penalty = config.trade_penalty
        self.hold_penalty = config.hold_penalty
        self.early_exit_bonus = config.early_exit_bonus
        self.flush_prediction_bonus = config.flush_prediction_bonus
        self.momentum_threshold = config.momentum_threshold
        self.volume_surge_threshold = config.volume_surge_threshold
        self.tape_speed_threshold = config.tape_speed_threshold
        self.tape_imbalance_threshold = config.tape_imbalance_threshold

    def __call__(self, env, action, portfolio_change, portfolio_change_pct,
                 trade_executed, info):
        """
        Calculate reward based on momentum trading strategy.

        Args:
            env: Trading environment
            action: Action value taken
            portfolio_change: Absolute portfolio change in value
            portfolio_change_pct: Percentage portfolio change
            trade_executed: Whether a trade was executed (vs. hold)
            info: Additional information dict

        Returns:
            float: Calculated reward value
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