"""
Training script for Inferno RL agent using MaskablePPO.

Usage:
    python -m tools.inferno_rl.training.train

This trains the model with:
- MaskablePPO for dynamic action masking
- Curriculum learning (waves 35-49 first, then expand)
- Checkpointing and logging
"""

import os
import argparse
import pickle
import zipfile
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    raise ImportError(
        "sb3_contrib is required for MaskablePPO. "
        "Install with: pip install sb3-contrib"
    )

from .env import InfernoEnv, make_inferno_env


class WaveProgressCallback(BaseCallback):
    """Tracks mean wave reached and clear rate per start-wave category."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.final_waves: Dict[int, List[int]] = {}
        self.clears: Dict[int, int] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not info.get("episode_done"):
                continue
            start_wave = info.get("start_wave")
            if start_wave is None:
                continue
            self.final_waves.setdefault(start_wave, []).append(info.get("wave", 0))
            if info.get("inferno_complete"):
                self.clears[start_wave] = self.clears.get(start_wave, 0) + 1

        if self.logger is not None:
            for start_wave, waves in sorted(self.final_waves.items()):
                mean_wave = sum(waves) / len(waves)
                clear_rate = self.clears.get(start_wave, 0) / len(waves)
                self.logger.record(f"rollout/mean_wave_from_{start_wave}", mean_wave)
                self.logger.record(f"rollout/clear_rate_from_{start_wave}", clear_rate)
        return True


class OutcomeStatsCallback(BaseCallback):
    """
    Logs outcome stats per rollout so A/B testing is easier.

    Note: these are not episode-level rates (SB3 handles episode bookkeeping),
    but they are still very useful for comparing two runs under identical settings.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._deaths = 0
        self._timeouts = 0
        self._waves_completed = 0
        # Per-start-wave tracking
        self._episodes_per_start: Dict[int, int] = {}
        self._deaths_per_start: Dict[int, int] = {}
        self._completions_per_start: Dict[int, int] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            start_wave = info.get("start_wave")
            
            if info.get("player_died"):
                self._deaths += 1
                if start_wave is not None:
                    self._deaths_per_start[start_wave] = self._deaths_per_start.get(start_wave, 0) + 1
                    self._episodes_per_start[start_wave] = self._episodes_per_start.get(start_wave, 0) + 1
            if info.get("wave_timeout"):
                self._timeouts += 1
                if start_wave is not None:
                    self._episodes_per_start[start_wave] = self._episodes_per_start.get(start_wave, 0) + 1
            if info.get("wave_completed"):
                self._waves_completed += 1
            if info.get("inferno_complete") and start_wave is not None:
                self._completions_per_start[start_wave] = self._completions_per_start.get(start_wave, 0) + 1
                self._episodes_per_start[start_wave] = self._episodes_per_start.get(start_wave, 0) + 1
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is not None:
            self.logger.record("rollout/deaths", self._deaths)
            self.logger.record("rollout/wave_timeouts", self._timeouts)
            self.logger.record("rollout/waves_completed", self._waves_completed)
            
            # Log per-start-wave episode counts
            for start_wave in sorted(self._episodes_per_start.keys()):
                episodes = self._episodes_per_start.get(start_wave, 0)
                deaths = self._deaths_per_start.get(start_wave, 0)
                completions = self._completions_per_start.get(start_wave, 0)
                self.logger.record(f"rollout/episodes_from_{start_wave}", episodes)
                if episodes > 0:
                    self.logger.record(f"rollout/death_rate_from_{start_wave}", deaths / episodes)
        
        self._deaths = 0
        self._timeouts = 0
        self._waves_completed = 0
        self._episodes_per_start.clear()
        self._deaths_per_start.clear()
        self._completions_per_start.clear()


class RewardTermsCallback(BaseCallback):
    """Logs per-episode raw reward term contributions (averaged over episodes in a rollout)."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episodes: int = 0
        self._sum_by_term: Dict[str, float] = {}
        self._mean_per_tick_by_term: Dict[str, float] = {}

    @staticmethod
    def _sanitize_term(term: str) -> str:
        # TensorBoard tags handle most chars, but this keeps grouping clean.
        term = term.replace(" ", "_")
        term = term.replace("(", "").replace(")", "")
        term = term.replace("!", "")
        term = term.replace(":", "")
        term = term.replace("+", "plus")
        return term

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            terms = info.get("episode_reward_terms")
            if not isinstance(terms, dict) or not terms:
                continue

            ep_len = None
            ep = info.get("episode")
            if isinstance(ep, dict):
                ep_len = ep.get("l")
            if not ep_len:
                ep_len = info.get("episode_reward_term_steps")
            ep_len = max(1, int(ep_len or 1))

            self._episodes += 1
            for term, value in terms.items():
                if not isinstance(value, (int, float)):
                    continue
                self._sum_by_term[term] = self._sum_by_term.get(term, 0.0) + float(value)
                self._mean_per_tick_by_term[term] = self._mean_per_tick_by_term.get(term, 0.0) + float(value) / ep_len
        return True

    def _on_rollout_end(self) -> None:
        if self.logger is None or self._episodes == 0:
            self._episodes = 0
            self._sum_by_term.clear()
            self._mean_per_tick_by_term.clear()
            return

        self.logger.record("raw_reward_terms/episodes", self._episodes)
        for term, total in sorted(self._sum_by_term.items()):
            key = self._sanitize_term(term)
            self.logger.record(f"raw_reward_terms/ep_sum_mean/{key}", total / self._episodes)
        for term, total_mean_per_tick in sorted(self._mean_per_tick_by_term.items()):
            key = self._sanitize_term(term)
            self.logger.record(f"raw_reward_terms/ep_mean_per_tick_mean/{key}", total_mean_per_tick / self._episodes)

        self._episodes = 0
        self._sum_by_term.clear()
        self._mean_per_tick_by_term.clear()


def get_mask_fn(env):
    """Get action mask from environment."""
    def mask_fn(_) -> np.ndarray:
        return env.action_masks()
    return mask_fn


def make_masked_env(start_wave: int, max_wave: int, seed: int = 0):
    """Create environment with action masking wrapper."""
    def _init():
        env = InfernoEnv(start_wave=start_wave, max_wave=max_wave)
        env = Monitor(env)
        env = ActionMasker(env, get_mask_fn(env.unwrapped))
        return env
    return _init


def parse_wave_weights(weights_str: Optional[str]) -> Optional[Dict[int, float]]:
    """
    Parse wave weights string format: "wave:weight,wave:weight"
    Example: "50:0.6,35:0.3,1:0.1" -> {50: 0.6, 35: 0.3, 1: 0.1}
    """
    if not weights_str:
        return None
    
    try:
        weights = {}
        parts = weights_str.split(",")
        for part in parts:
            wave, weight = part.split(":")
            weights[int(wave)] = float(weight)
        return weights
    except ValueError:
        raise ValueError(f"Invalid wave weights format: {weights_str}. Expected format: 'wave:weight,wave:weight'")


def train(
    total_timesteps: int = 1_000_000,
    start_wave: int = 35,
    max_wave: int = 49,
    start_wave_weights: Optional[Dict[int, float]] = None,
    n_envs: int = 4,
    save_dir: str = "models/inferno",
    log_dir: str = "logs/inferno",
    seed: int = 42,
    learning_rate: float = 1e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = 0.015,
    normalize_reward: bool = True,
    clip_reward: float = 10.0,
    checkpoint_freq: int = 1_000_000,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    verbose: int = 1,
    load_path: Optional[str] = None,
    log_reward_terms: bool = False,
):
    """
    Train the Inferno RL agent.
    
    Args:
        total_timesteps: Total training timesteps
        start_wave: Wave to start training from (default if no weights)
        max_wave: Maximum wave for this curriculum stage
        start_wave_weights: Optional dict of {wave: probability} for mixed starts
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        seed: Random seed
        learning_rate: Learning rate
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        target_kl: KL divergence threshold for early stopping (prevents policy collapse)
        normalize_reward: Whether to normalize rewards using VecNormalize
        clip_reward: Reward clipping threshold when normalizing
        checkpoint_freq: Save checkpoint every N total steps (default 1M)
        eval_freq: Evaluate every N steps
        eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
        load_path: Path to load existing model
        log_reward_terms: Log per-episode raw reward term contributions to TensorBoard
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inferno_w{start_wave}-{max_wave}_{timestamp}"
    
    # Create vectorized training environment
    if start_wave_weights:
        print(f"Creating {n_envs} training environments (mixed starts: {start_wave_weights})...")
    else:
        print(f"Creating {n_envs} training environments (waves {start_wave}-{max_wave})...")

    def make_masked_env_factory(start_wave_: int, max_wave_: int, seed_: int = 0, weights: Optional[Dict[int, float]] = None):
        def _init():
            env = InfernoEnv(
                start_wave=start_wave_,
                max_wave=max_wave_,
                start_wave_weights=weights,
                record_reward_terms=log_reward_terms,
            )
            env = Monitor(env)
            env = ActionMasker(env, get_mask_fn(env.unwrapped))
            return env
        return _init

    env_fns = [make_masked_env_factory(start_wave, max_wave, seed + i, start_wave_weights) for i in range(n_envs)]
    
    if n_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)
    
    # Create evaluation environment (always deterministic start_wave for consistency)
    eval_env = DummyVecEnv([make_masked_env_factory(start_wave, max_wave, seed + 100, None)])
    
    # Wrap with VecNormalize for reward normalization (improves value function learning)
    if normalize_reward:
        print(f"Enabling reward normalization (clip_reward={clip_reward})...")
        train_env = VecNormalize(
            train_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=clip_reward,
            gamma=gamma,
        )
        # Eval env: don't normalize rewards (for true metrics), don't update stats
        eval_env = VecNormalize(
            eval_env,
            norm_obs=False,
            norm_reward=False,
            training=False,
        )
    
    # Load VecNormalize stats if loading existing model.
    if load_path and normalize_reward:
        vecnorm_path = f"{load_path}_vecnormalize.pkl"
        if os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize stats from {vecnorm_path}")
            train_env = VecNormalize.load(vecnorm_path, train_env)
            train_env.training = True
            train_env.norm_reward = True
            # Keep eval env wrapped with the same normalization stats for sync
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
    
    # Create or load model (SB3 saves as .zip; accept path with or without extension)
    load_path_resolved = None
    if load_path:
        if os.path.exists(load_path):
            load_path_resolved = load_path
        elif os.path.exists(f"{load_path}.zip"):
            load_path_resolved = f"{load_path}.zip"
        else:
            print(f"Warning: --load {load_path} not found (tried {load_path} and {load_path}.zip); creating new model.")
    if load_path_resolved:
        print(f"Loading model from {load_path_resolved}")
        model = MaskablePPO.load(
            load_path_resolved,
            env=train_env,
            tensorboard_log=log_dir,
        )
        # Restore num_timesteps from zip (SB3 load can leave it 0 when env is replaced)
        try:
            with zipfile.ZipFile(load_path_resolved, "r") as zf:
                data = pickle.load(zf.open("data"))
                saved_steps = data.get("num_timesteps", 0)
                model.num_timesteps = saved_steps
            print(f"Resuming from step {model.num_timesteps:,}")
        except Exception:
            pass
    else:
        print("Creating new MaskablePPO model...")
        print(f"  target_kl: {target_kl}")
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            verbose=verbose,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs={
                "net_arch": [256, 256, 128],
            },
        )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=save_dir,
        name_prefix=run_name,
        verbose=1,
    )
    
    # Save best model per run to avoid overwriting previous "best" artifacts
    best_dir = os.path.join(save_dir, "best", run_name)
    os.makedirs(best_dir, exist_ok=True)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=f"{log_dir}/eval",
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        verbose=1,
    )
    
    max_wave_callback = WaveProgressCallback(verbose=1)
    outcome_callback = OutcomeStatsCallback(verbose=1)
    callbacks_list = [checkpoint_callback, eval_callback, max_wave_callback, outcome_callback]
    if log_reward_terms:
        callbacks_list.append(RewardTermsCallback(verbose=0))
    callbacks = CallbackList(callbacks_list)
    
    # When resuming, --timesteps is ADDITIONAL steps (target total = loaded + timesteps)
    if load_path_resolved:
        target_timesteps = model.num_timesteps + total_timesteps
        print(f"Starting training for {total_timesteps:,} more steps (target total: {target_timesteps:,})...")
    else:
        target_timesteps = total_timesteps
        print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"  Model: MaskablePPO")
    print(f"  Waves: {start_wave}-{max_wave}")
    print(f"  Envs: {n_envs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target KL: {target_kl}")
    print(f"  Reward normalization: {normalize_reward}")
    
    model.learn(
        total_timesteps=target_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=run_name,
    )
    
    # Save final model
    final_path = f"{save_dir}/{run_name}_final"
    model.save(final_path)
    print(f"Final model saved to {final_path}")
    
    # Save VecNormalize stats for consistent inference
    if normalize_reward:
        vecnorm_path = f"{final_path}_vecnormalize.pkl"
        train_env.save(vecnorm_path)
        print(f"VecNormalize stats saved to {vecnorm_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Inferno RL Agent")
    
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--start-wave", type=int, default=35,
                       help="Starting wave")
    parser.add_argument("--max-wave", type=int, default=49,
                       help="Maximum wave")
    parser.add_argument("--mixed-waves", type=str, default=None,
                       help="Mixed start waves (e.g. '50:0.6,35:0.3,1:0.1')")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--save-dir", type=str, default="models/inferno",
                       help="Model save directory")
    parser.add_argument("--log-dir", type=str, default="logs/inferno",
                       help="Tensorboard log directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--load", type=str, default=None,
                       help="Path to load existing model")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--target-kl", type=float, default=0.015,
                       help="KL divergence threshold for early stopping")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                       help="Entropy coefficient (higher = more exploration)")
    parser.add_argument("--no-reward-norm", action="store_true",
                       help="Disable reward normalization")
    parser.add_argument("--clip-reward", type=float, default=10.0,
                       help="Reward clipping threshold")
    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000,
                       help="Save checkpoint every N steps (default 1M)")
    parser.add_argument("--log-reward-terms", action="store_true",
                        help="Log per-episode raw reward terms (TensorBoard: raw_reward_terms/*)")
    
    args = parser.parse_args()
    
    start_wave_weights = parse_wave_weights(args.mixed_waves)

    train(
        total_timesteps=args.timesteps,
        start_wave=args.start_wave,
        max_wave=args.max_wave,
        start_wave_weights=start_wave_weights,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        load_path=args.load,
        learning_rate=args.lr,
        target_kl=args.target_kl,
        ent_coef=args.ent_coef,
        normalize_reward=not args.no_reward_norm,
        clip_reward=args.clip_reward,
        checkpoint_freq=args.checkpoint_freq,
        log_reward_terms=args.log_reward_terms,
    )


if __name__ == "__main__":
    main()
