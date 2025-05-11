* The best Model folder is Checkpoints of your best-performing models (saved by EvalCallback)
  * model = PPO.load("best_model.zip")
* The eval folder is evaluation results. Per-episode rewards & lenght on the eval env also binary version.
  * data = np.load("eval.npz") & print(data.files)
* PPO_TensorBoard is the folder where the tensorboard logs are saved.
  * tensorboard --logdir ./ppo_trading_tensorboard/

Key lines
rollout/ep_len_mean : Average episode length (in time-steps). If you always see the maximum (e.g. 199), the agent isn’t dying early.
rollout/ep_rew_mean : Mean cumulative reward per episode. Watching this over time tells you if the agent is actually learning.
time/fps            : Training speed: frames (steps) per second.
time/iterations & total_timesteps : How many training cycles and total environment steps you’ve run.
train/approx_kl     : The approximate KL divergence between old and new policies. Values ≪ 0.01 are usually safe; if KL spikes, you may need to reduce your clip range or learning rate.
train/clip_fraction : Percentage of updates where the policy update was clipped. If this is high (> 0.3), your clip_range may be too small or your learning rate too large.
train/entropy_loss  : A proxy for exploration. Very low (large negative) means the policy is collapsing to deterministic actions too quickly.
train/explained_variance : How well the value function predicts returns (1.0 is perfect, 0.0 means no predictive power). Very low or negative means your value head isn’t learning—consider changing network size or learning rate.
train/policy_gradient_loss & value_loss : The raw losses. Watching these decrease (and stabilize) indicates learning is progressing.

* Environments uses -> Agents uses -> Models (Policies) 