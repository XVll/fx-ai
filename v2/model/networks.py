# ai/networks.py
# NOTE: These networks are currently unused as the MultiBranchTransformer
# has built-in actor-critic heads. Kept for potential future use.

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Actor network for the policy, outputs action distribution parameters.
    For continuous actions: mean and log_std of a squashed Gaussian.
    For discrete actions: logits of categorical distribution.
    """

    def __init__(self, input_dim, action_dim, continuous_action=True, hidden_dim=256):
        super().__init__()
        self.continuous_action = continuous_action

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        if continuous_action:
            # For continuous actions (PPO with Gaussian)
            self.mean = nn.Linear(hidden_dim, action_dim)
            # Log standard deviation (separate parameter)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # For discrete actions (categorical)
            self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            For continuous: (mean, log_std) tuple
            For discrete: logits tensor
        """
        shared_out = self.shared(x)

        if self.continuous_action:
            # Output mean and use parameter for log_std
            action_mean = self.mean(shared_out)
            # Expand log_std to match the batch size
            batch_size = x.size(0)
            log_std = self.log_std.expand(batch_size, -1)
            return action_mean, log_std
        else:
            # Output logits for categorical distribution
            logits = self.logits(shared_out)
            return logits


class CriticNetwork(nn.Module):
    """
    Critic network that estimates the value function.
    """

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Value estimates of shape [batch_size, 1]
        """
        return self.critic(x)
