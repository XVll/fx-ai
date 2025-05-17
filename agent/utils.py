# agent/utils.py
import numpy as np
import torch
from typing import Dict, List, Any, Optional
import logging

# Configure a logger for this module (optional, but good practice)
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Replay buffer for PPO, storing transitions.
    It converts NumPy arrays from the environment to PyTorch tensors for storage.
    """

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer: List[Dict[str, Any]] = []
        self.position: int = 0  # Current position to insert new experience

        # These will be populated when prepare_data_for_training is called
        self.states: Optional[Dict[str, torch.Tensor]] = None
        self.actions: Optional[torch.Tensor] = None
        self.log_probs: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.rewards: Optional[torch.Tensor] = None
        self.dones: Optional[torch.Tensor] = None
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        logger.info(f"ReplayBuffer initialized with capacity {self.capacity} on device {self.device}")

    def _process_state_dict(self, state_dict_np: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Converts a dictionary of NumPy arrays to a dictionary of PyTorch tensors."""
        processed_tensors = {}
        for key, array_val in state_dict_np.items():
            # Ensure the numpy array is not an object array if it contains numerical data
            if array_val.dtype == np.object_:
                try:
                    array_val = np.stack(array_val.tolist())  # For safety if it's a list of arrays
                except Exception as e:
                    logger.error(f"Could not stack object array for key {key}: {e}. Array: {array_val}")
                    # Fallback: try to convert as is, or handle error more gracefully
                    pass

            # Convert to float32 tensor, assuming numerical data
            # The environment should already provide states with a batch-like dimension of 1
            # e.g., (1, seq_len, feat_dim) or (1, feat_dim)
            try:
                tensor_val = torch.from_numpy(array_val).float().to(self.device)
                processed_tensors[key] = tensor_val
            except TypeError as e:
                logger.error(f"TypeError converting key '{key}' to tensor: {e}. Value: {array_val}, Dtype: {array_val.dtype}")
                # Handle or re-raise depending on how critical this is
                raise
        return processed_tensors

    def add(self,
            state_np: Dict[str, np.ndarray],
            action: torch.Tensor,  # Assuming action from a model is already a tensor
            reward: float,
            next_state_np: Dict[str, np.ndarray],
            done: bool,
            action_info: Dict[str, torch.Tensor]):  # Contains 'value' and 'log_prob' as tensors
        """
        Add a transition to the buffer.
        State and next_state are expected as Dict[str, np.ndarray] from the environment.
        Action, value, and log_prob are expected as PyTorch tensors from the model/agent.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append({})  # Add a new slot if capacity not reached

        # Store raw NumPy states for now, convert to a batch of tensors later
        # Or convert on the fly if memory is not an issue, and for type consistency in buffer dicts
        # Let's convert on the fly to simplify prepare_data_for_training

        experience = {
            'state': self._process_state_dict(state_np),
            'action': action.detach().to(self.device),  # Ensure on a correct device and detached
            'reward': torch.tensor([reward], dtype=torch.float32, device=self.device),
            'next_state': self._process_state_dict(next_state_np),
            'done': torch.tensor([done], dtype=torch.bool, device=self.device),
            'value': action_info['value'].detach().to(self.device),
            'log_prob': action_info['log_prob'].detach().to(self.device)
        }

        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def prepare_data_for_training(self) -> None:
        """
        Converts the list of experiences into batched tensors for training.
        This should be called when the buffer is full or a rollout is complete.
        """
        if not self.buffer:
            logger.warning("Buffer is empty, cannot prepare data for training.")
            return

        # Initialize lists for each component
        all_states_components: Dict[str, List[torch.Tensor]] = {
            key: [] for key in self.buffer[0]['state'].keys()
        }
        all_actions: List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []
        all_values: List[torch.Tensor] = []
        all_rewards: List[torch.Tensor] = []
        all_dones: List[torch.Tensor] = []

        for exp in self.buffer:
            for key, tensor_val in exp['state'].items():
                all_states_components[key].append(tensor_val)
            all_actions.append(exp['action'])
            all_log_probs.append(exp['log_prob'])
            all_values.append(exp['value'])
            all_rewards.append(exp['reward'])
            all_dones.append(exp['done'])

        # Batch all components
        self.states = {
            key: torch.cat(tensors_list, dim=0)
            for key, tensors_list in all_states_components.items()
        }
        self.actions = torch.cat(all_actions, dim=0)
        self.log_probs = torch.cat(all_log_probs, dim=0)
        self.values = torch.cat(all_values, dim=0)
        self.rewards = torch.cat(all_rewards, dim=0)
        self.dones = torch.cat(all_dones, dim=0)

        # The PPO agent will compute advantages and returns after this step
        self.advantages = None
        self.returns = None
        logger.info(f"Buffer data prepared for training. Buffer size: {len(self.buffer)}")

    def get_training_data(self) -> Optional[Dict[str, Any]]:
        """Returns all necessary data for a PPO update epoch if prepared."""
        if self.states is None or self.actions is None or self.log_probs is None or \
                self.rewards is None or self.dones is None or self.values is None or \
                self.advantages is None or self.returns is None:  # Check if advantages and returns are computed
            logger.error("Training data not fully prepared (states, actions, log_probs, "
                         "rewards, dones, values, advantages, or returns are None). "
                         "Call prepare_data_for_training() and then compute_advantages() first.")
            return None

        return {
            "states": self.states,
            "actions": self.actions,
            "old_log_probs": self.log_probs,
            "advantages": self.advantages,
            "returns": self.returns,
            "values": self.values  # For KL divergence calculation or other diagnostics if needed
        }

    def get_size(self) -> int:
        """Get the current number of experiences in the buffer."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer and reset related tensors."""
        self.buffer.clear()
        self.position = 0
        self.states = None
        self.actions = None
        self.log_probs = None
        self.values = None
        self.rewards = None
        self.dones = None
        self.advantages = None
        self.returns = None
        logger.info("ReplayBuffer cleared.")

    def is_ready_for_training(self) -> bool:
        """Checks if the buffer has enough samples (e.g., is full or met a threshold)."""
        # This can be customized, e.g., return len(self.buffer) == self.capacity
        # For PPO, typically we collect a fixed number of steps/episodes, then train.
        return len(self.buffer) > 0  # A basic check, trainer will decide when to train


# Optional: State normalization utility (if needed globally)
# If normalization is part of the environment or feature extractor, this might not be needed here.
# The original `normalize_state_dict` and `preprocess_state_to_dict` are not directly
# used if the environment already returns Dict[str, np.ndarray] and the buffer/agent handles
# the np.ndarray -> torch.Tensor conversion.
# The `preprocess_state_to_dict` is not needed if your ` environment ` already
# produces the correct Dict[str, np.ndarray] structure.

def convert_state_dict_to_tensors(
        state_dict_np: Dict[str, np.ndarray],
        device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Converts a dictionary of NumPy arrays (features) to a dictionary of PyTorch tensors
    and moves them to the specified device.
    Assumes that the NumPy arrays from the environment already have the correct
    batch-like dimension (e.g., (1, seq_len, feat_dim) or (1, feat_dim)).
    """
    state_dict_torch = {}
    for key, np_array in state_dict_np.items():
        # Ensure the numpy array is not an object array if it contains numerical data
        if np_array.dtype == np.object_:
            try:  # Attempt to stack if it's a list of arrays, common for 'portfolio' if not pre-stacked
                np_array = np.stack(np_array.tolist())
            except Exception as e:
                logger.warning(f"Could not stack object array for key {key} during tensor conversion: {e}. Using as is.")

        try:
            state_dict_torch[key] = torch.from_numpy(np_array).float().to(device)
        except TypeError as e:
            logger.error(f"TypeError converting key '{key}' to tensor: {e}. Value: {np_array}, Dtype: {np_array.dtype}")
            raise
    return state_dict_torch