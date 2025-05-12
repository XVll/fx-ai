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



II. SOTA Model Architecture: Multi-Branch Transformer with Attention Fusion

Rationale: Explicitly handles multi-timescale features with specialized processing and allows learned interaction between processed representations.

Inputs: (Normalized)
HF Sequence: (BatchSize, T_hf, Num_HF_Features) (e.g., T_hf=60)
MF Sequence (Rolling 1m Context): (BatchSize, T_mf, Num_MF_Features) (e.g., T_mf=30)
LF Sequence (Rolling 5m Context): (BatchSize, T_lf, Num_LF_Features) (e.g., T_lf=30)
Static Features (S/R Distances, Agent State, Time Encoding): (BatchSize, Num_Static_Features)
Branch Processing:
HF Branch:
Optional: 1D Convolutional Layer(s) with activation (e.g., ReLU/GeLU) for local pattern extraction.
Linear Projection to d_model.
Add Positional Encoding.
Transformer Encoder Stack (L_hf layers, d_model, num_heads_hf).
Aggregate Output (e.g., embedding of the last time step) -> (BatchSize, d_model).
MF Branch:
Linear Projection to d_model.
Add Positional Encoding.
Transformer Encoder Stack (L_mf layers, d_model, num_heads_mf).
Aggregate Output -> (BatchSize, d_model).
LF Branch:
Linear Projection to d_model.
Add Positional Encoding.
Transformer Encoder Stack (L_lf layers, d_model, num_heads_lf).
Aggregate Output -> (BatchSize, d_model).
Static Branch:
MLP (e.g., 2 layers with ReLU/GeLU) projecting Num_Static_Features -> d_model. Output -> (BatchSize, d_model).
Fusion Mechanism:
Concatenate outputs from all branches: (BatchSize, 4 * d_model).
Pass concatenated vector through a Self-Attention Layer (or a 1-2 layer Transformer Encoder without positional encoding) to allow learned interaction between the fused branch representations.
Output of fusion layer: (BatchSize, d_fused) (where d_fused might be 4 * d_model or projected down).
Output Heads (MLPs):
Feed the final d_fused representation into separate MLPs for:
Actor: Outputs Mean and Log Standard Deviation for a Squashed Gaussian policy (use Tanh).
Critics (Twin Q-Networks for SAC): Takes fused state representation and the action as input (concatenate action before first MLP layer) -> Outputs Q-value estimates (Q1 and Q2).
