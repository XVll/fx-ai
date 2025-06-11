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
        logger.info(
            f"ReplayBuffer initialized with capacity {self.capacity} on device {self.device}"
        )

    def _process_state_dict(
        self, state_dict_np: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Converts a dictionary of NumPy arrays to a dictionary of PyTorch tensors,
        preserving the original tensor structure.
        """
        processed_tensors = {}
        for key, array_val in state_dict_np.items():
            # Debug logging for object arrays
            if array_val.dtype == np.object_:
                logger.warning(
                    f"Found object array for key '{key}', shape: {array_val.shape}, will attempt conversion"
                )

            # Ensure the numpy array is not an object array if it contains numerical data
            if array_val.dtype == np.object_:
                try:
                    # First attempt to convert to a proper numpy array
                    if isinstance(array_val, np.ndarray) and array_val.size > 0:
                        # Try to extract the first element to determine proper dtype
                        first_elem = array_val.flat[0]
                        if isinstance(first_elem, (int, float, np.number)):
                            array_val = np.array(array_val.tolist(), dtype=np.float32)
                        else:
                            array_val = np.stack(array_val.tolist())
                    else:
                        array_val = np.array(array_val.tolist(), dtype=np.float32)
                except Exception as e:
                    logger.error(
                        f"Could not convert object array for key {key}: {e}. Array shape: {array_val.shape}"
                    )
                    # Create a zero array as fallback to prevent corruption
                    if hasattr(array_val, "shape") and array_val.shape:
                        # Create proper shape with float32 dtype
                        shape = array_val.shape
                        array_val = np.zeros(shape, dtype=np.float32)
                    else:
                        raise ValueError(f"Cannot process object array for key {key}")

            # Ensure array is contiguous and has proper dtype before conversion
            if not array_val.flags["C_CONTIGUOUS"]:
                array_val = np.ascontiguousarray(array_val, dtype=np.float32)
            elif array_val.dtype != np.float32:
                array_val = array_val.astype(np.float32)

            # Convert to float32 tensor, preserving the original dimensions
            try:
                # Convert to PyTorch tensor, preserving the original shape
                tensor_val = torch.from_numpy(array_val).to(self.device)
                processed_tensors[key] = tensor_val

                # Log the shape for debugging
                logger.debug(f"Processed tensor '{key}' with shape: {tensor_val.shape}")
            except (TypeError, RuntimeError) as e:
                logger.error(
                    f"Error converting key '{key}' to tensor: {e}. Value shape: {array_val.shape}, Dtype: {array_val.dtype}"
                )
                # Handle or re-raise depending on how critical this is
                raise

        return processed_tensors

    def add(
        self,
        state_np: Dict[str, np.ndarray],
        action: torch.Tensor,  # Assuming action from a model is already a tensor
        reward: float,
        next_state_np: Dict[str, np.ndarray],
        done: bool,
        action_info: Dict[str, torch.Tensor],
    ):  # Contains 'value' and 'log_prob' as tensors
        """
        Add a transition to the buffer.
        State and next_state are expected as Dict[str, np.ndarray] from the environment.
        Action, value, and log_prob are expected as PyTorch tensors from the model/agent.
        """
        # Handle zero capacity buffer gracefully
        if self.capacity == 0:
            logger.warning("Cannot add experience to zero capacity buffer.")
            return

        if len(self.buffer) < self.capacity:
            self.buffer.append({})  # Add a new slot if capacity not reached

        # Store raw NumPy states for now, convert to a batch of tensors later
        # Or convert on the fly if memory is not an issue, and for type consistency in buffer dicts
        # Let's convert on the fly to simplify prepare_data_for_training

        experience = {
            "state": self._process_state_dict(state_np),
            "action": action.detach().to(self.device),
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "next_state": self._process_state_dict(next_state_np),
            "done": torch.tensor([done], dtype=torch.bool, device=self.device),
            "value": action_info["value"].detach().to(self.device),
            "log_prob": action_info["log_prob"].detach().to(self.device),
        }

        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def prepare_data_for_training(self) -> None:
        """
        Converts the list of experiences into batched tensors for training,
        preserving the original tensor dimensions for each component.
        """
        if not self.buffer:
            logger.warning("Buffer is empty, preparing empty tensors for training.")
            # Initialize empty tensors with proper shapes
            self.states = {}
            self.actions = torch.tensor([], dtype=torch.int32, device=self.device).reshape(0, 2)
            self.log_probs = torch.tensor([], dtype=torch.float32, device=self.device)
            self.values = torch.tensor([], dtype=torch.float32, device=self.device)
            self.rewards = torch.tensor([], dtype=torch.float32, device=self.device)
            self.dones = torch.tensor([], dtype=torch.bool, device=self.device)
            self.advantages = None
            self.returns = None
            return

        # Initialize structures for each component
        all_states_components: Dict[str, List[torch.Tensor]] = {
            key: [] for key in self.buffer[0]["state"].keys()
        }
        all_actions: List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []
        all_values: List[torch.Tensor] = []
        all_rewards: List[torch.Tensor] = []
        all_dones: List[torch.Tensor] = []

        for exp in self.buffer:
            for key, tensor_val in exp["state"].items():
                all_states_components[key].append(tensor_val)
            all_actions.append(exp["action"])
            all_log_probs.append(exp["log_prob"])
            all_values.append(exp["value"])
            all_rewards.append(exp["reward"])
            all_dones.append(exp["done"])

        # Batch all components, preserving dimensions
        self.states = {}

        # For each component type, properly concatenate along batch dimension
        for key, tensors_list in all_states_components.items():
            # Determine the dimensions for proper concatenation
            first_tensor = tensors_list[0]
            if key in ["hf", "mf", "lf", "portfolio"]:
                # These should be [seq_len, feat_dim] tensors stacked into [batch_size, seq_len, feat_dim]
                if first_tensor.ndim == 2:  # [seq_len, feat_dim]
                    self.states[key] = torch.stack(tensors_list, dim=0)
                    logger.debug(
                        f"Stacked {key} tensors to shape: {self.states[key].shape}"
                    )
                elif first_tensor.ndim == 3:  # Already [1, seq_len, feat_dim]
                    self.states[key] = torch.cat(tensors_list, dim=0)
                    logger.debug(
                        f"Concatenated {key} tensors to shape: {self.states[key].shape}"
                    )
                else:
                    logger.warning(
                        f"Unexpected shape for {key}: {first_tensor.shape}. Attempting default concatenation."
                    )
                    self.states[key] = torch.cat(tensors_list, dim=0)
            else:
                # Other components: default to concatenation
                logger.debug(f"Default concatenation for {key} tensors")
                self.states[key] = torch.cat(tensors_list, dim=0)

        # Process other components
        self.actions = torch.cat(all_actions, dim=0)
        self.log_probs = torch.cat(all_log_probs, dim=0)
        self.values = torch.cat(all_values, dim=0)
        self.rewards = torch.cat(all_rewards, dim=0)
        self.dones = torch.cat(all_dones, dim=0)

        # The PPO agent will compute advantages and returns after this step
        self.advantages = None
        self.returns = None
        logger.info(
            f"Buffer data prepared for training. Buffer size: {len(self.buffer)}"
        )

        # Log shapes for debugging
        for key, tensor in self.states.items():
            logger.debug(f"State component '{key}' shape: {tensor.shape}")
        logger.debug(f"Actions shape: {self.actions.shape}")
        logger.debug(f"Log_probs shape: {self.log_probs.shape}")
        logger.debug(f"Values shape: {self.values.shape}")
        logger.debug(f"Rewards shape: {self.rewards.shape}")
        logger.debug(f"Dones shape: {self.dones.shape}")

    def get_training_data(self) -> Optional[Dict[str, Any]]:
        """Returns all necessary data for a PPO update epoch if prepared."""
        # Check if basic data is prepared
        if (
            self.states is None
            or self.actions is None
            or self.log_probs is None
            or self.rewards is None
            or self.dones is None
            or self.values is None
        ):
            raise ValueError(
                "Training data not prepared. Call prepare_data_for_training() first."
            )
        
        # Check if advantages and returns are computed
        if self.advantages is None or self.returns is None:
            logger.error(
                "Advantages and returns not computed. "
                "Call compute_advantages() after prepare_data_for_training()."
            )
            return None

        return {
            "states": self.states,
            "actions": self.actions,
            "old_log_probs": self.log_probs,
            "advantages": self.advantages,
            "returns": self.returns,
            "values": self.values,  # For KL divergence calculation or other diagnostics if needed
        }

    def get_size(self) -> int:
        """Get the current number of experiences in the buffer."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer and reset related tensors."""
        # Properly clean up tensor references to avoid memory leaks
        for exp in self.buffer:
            for key in list(exp.keys()):
                if isinstance(exp[key], torch.Tensor):
                    # Explicitly detach and delete tensor references
                    exp[key] = exp[key].detach().cpu()
                    del exp[key]
                elif isinstance(exp[key], dict):
                    for sub_key in list(exp[key].keys()):
                        if isinstance(exp[key][sub_key], torch.Tensor):
                            exp[key][sub_key] = exp[key][sub_key].detach().cpu()
                            del exp[key][sub_key]

        self.buffer.clear()
        self.position = 0

        # Clear tensors and free memory
        if self.states is not None:
            for key in self.states:
                if isinstance(self.states[key], torch.Tensor):
                    self.states[key] = self.states[key].detach().cpu()

        # Detach and clear all tensors
        for attr in [
            "actions",
            "log_probs",
            "values",
            "rewards",
            "dones",
            "advantages",
            "returns",
        ]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor):
                    setattr(self, attr, tensor.detach().cpu())

        self.states = None
        self.actions = None
        self.log_probs = None
        self.values = None
        self.rewards = None
        self.dones = None
        self.advantages = None
        self.returns = None

        # Force garbage collection to free memory
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    state_dict_np: Dict[str, np.ndarray], device: torch.device
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
            try:
                # First attempt to convert to a proper numpy array
                if isinstance(np_array, np.ndarray) and np_array.size > 0:
                    # Try to extract the first element to determine proper dtype
                    first_elem = np_array.flat[0]
                    if isinstance(first_elem, (int, float, np.number)):
                        np_array = np.array(np_array.tolist(), dtype=np.float32)
                    else:
                        np_array = np.stack(np_array.tolist())
                else:
                    np_array = np.array(np_array.tolist(), dtype=np.float32)
            except Exception as e:
                logger.error(
                    f"Could not convert object array for key {key} during tensor conversion: {e}"
                )
                if hasattr(np_array, "shape") and np_array.shape:
                    shape = np_array.shape
                    np_array = np.zeros(shape, dtype=np.float32)
                else:
                    raise ValueError(f"Cannot process object array for key {key}")

        # Ensure array is contiguous and has proper dtype before conversion
        if not np_array.flags["C_CONTIGUOUS"]:
            np_array = np.ascontiguousarray(np_array, dtype=np.float32)
        elif np_array.dtype != np.float32:
            np_array = np.array(np_array, dtype=np.float32)

        try:
            state_dict_torch[key] = torch.from_numpy(np_array).to(device)
        except (TypeError, RuntimeError) as e:
            logger.error(
                f"Error converting key '{key}' to tensor: {e}. Shape: {np_array.shape}, Dtype: {np_array.dtype}"
            )
            raise
    return state_dict_torch
