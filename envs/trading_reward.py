class TradingReward:
    """Minimal trading reward function."""

    def __init__(self,
                 reward_scaling=1.0,
                 trade_penalty=0.01,
                 hold_penalty=0.0):
        """Initialize reward parameters."""
        self.reward_scaling = reward_scaling
        self.trade_penalty = trade_penalty
        self.hold_penalty = hold_penalty

    def calculate(self,
                  portfolio_change,
                  trade_executed=False,
                  momentum_strength=0.0,
                  relative_volume=1.0):
        """Calculate reward based on portfolio change and trading decisions."""
        # Base reward is portfolio change
        reward = portfolio_change * self.reward_scaling

        # Apply trading penalties
        if trade_executed:
            reward -= self.trade_penalty
        else:
            reward -= self.hold_penalty

        # Simple momentum bonus
        if trade_executed and momentum_strength > 0.5 and relative_volume > 3.0:
            reward *= 1.5  # 50% boost for trading with momentum

        return reward