from envs.trading_env import BasicTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

# 1. Create a function that returns a fresh env instance
def make_env():
    def _init():
        env = BasicTradingEnv(
            data_file='sample_data.csv',
            short_window=5,
            long_window=20
        )
        return env
    return _init

# 2. Vectorize and monitor the env for logging
vec_env = DummyVecEnv([ make_env() for _ in range(4) ])       # 4 parallel envs
vec_env = VecMonitor(vec_env)                                # record episode rewards

# 3. Instantiate the PPO agent
model = PPO(
    policy='MlpPolicy',
    env=vec_env,
    verbose=1,
    tensorboard_log='./ppo_trading_tensorboard/',             # for TB logging
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

# 4. Set up periodic evaluation on a held-out env
eval_env = DummyVecEnv([ make_env() ])
eval_env = VecMonitor(eval_env)                       # single env for eval
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./logs/best_model/',
    log_path='./logs/eval/',
    eval_freq=5000,                                           # every 5000 steps
    n_eval_episodes=5,
    deterministic=True
)

# 5. Train the agent
model.learn(
    total_timesteps=500_000,                                   # adjust as needed
    callback=eval_callback,
    tb_log_name='ppo_baseline'
)

# 6. Save the final model
model.save('ppo_trading_baseline')

# 7. Evaluate and render a single episode
obs = eval_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
# Render the final episode
eval_env.envs[0].render()
