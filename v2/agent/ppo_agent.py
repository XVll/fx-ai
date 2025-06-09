from torch import optim

from v2.agent.replay_buffer import ReplayBuffer


class PPOTrainer:

    def __init__(
            self,
            env: TradingEnvironment,
            model: MultiBranchTransformer,
            callback_manager: CallbackManager,
            config: Any,
            device: Optional[Union[str, torch.device]] = None,
            output_dir: str = "./ppo_output",
    ):
        self.logger = logging.getLogger(__name__)
        self.env = env
        self.model = model
        self.config = config  # Store full config for curriculum access
        self.callback_manager = callback_manager
        self.device = device

        # Output directories
        self.model_dir = os.path.join(output_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Extract training parameters from config
        training_config = config.training
        self.lr = training_config.learning_rate
        self.gamma = training_config.gamma
        self.gae_lambda = training_config.gae_lambda
        self.clip_eps = training_config.clip_epsilon
        self.critic_coef = training_config.value_coef
        self.entropy_coef = training_config.entropy_coef
        self.max_grad_norm = training_config.max_grad_norm
        self.ppo_epochs = training_config.n_epochs
        self.batch_size = training_config.batch_size
        self.rollout_steps = training_config.rollout_steps

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=self.rollout_steps, device=self.device)

        # Training state
        self.global_step_counter = 0
        self.global_episode_counter = 0
        self.global_update_counter = 0

        # Performance tracking
        self.is_evaluating = False

        # Momentum training state
        self.current_momentum_day = None
        self.used_momentum_days = set()
        self.current_reset_points = []
        self.used_reset_point_indices = set()
        # Data quality filtering (managed by TrainingManager)
        self.quality_range = [0.7, 1.0]  # Default quality range


        self.logger.info(f"🤖 PPOTrainer initialized with callback system. Device: {self.device}")

    def run_training_step(self) -> bool:
        """
        Run one training step for TrainingManager integration
        Returns True if training should continue, False to stop
        """

        # Collect rollout data
        rollout_info = self.collect_rollout_data()

        # Check buffer size
        if self.buffer.get_size() < self.batch_size:
            return True  # Continue training, skip this update

        # Update policy
        update_metrics = self.update_policy()

        return True  # Continue training