"""
Collect behaviour-cloning data by rolling out a trained RL model.

Uses an existing trained checkpoint as the expert.  Saves (obs, action, action_mask,
logits) tuples compatible with pretrain_bc.py — with logits enabling KL
distillation into a different architecture.

Usage:
    python -m tools.inferno_rl.pretrain.collect_from_model \
        --model models/V21_climb/inferno_gpu_w35-66_20260225_090159_2700.pt \
        --episodes 2000 \
        --start-wave 55 --max-wave 66 \
        --output models/bc_data/v21_rollouts.npz
"""

import argparse
import os
import random
import time

import numpy as np
import torch as th

from ..ppo.ppo import PPO
from ..simulator.simulator import InfernoSimulator
from ..training.observation import build_observation
from ..training.actions import decode_policy_action, get_mask_for_action_space


def collect(
    model_path: str,
    n_episodes: int,
    start_wave: int,
    max_wave: int,
    output_path: str,
    deterministic: bool = False,
    device: str = "cpu",
    seed: int = 0,
) -> None:
    print(f"Loading model from {model_path}")
    ppo = PPO.load(model_path, trainable=False, device=device)
    print(f"  trained_steps={ppo.meta.trained_steps:,}")
    print(f"  obs_normalized={ppo.meta.normalized_observations}")
    action_head_sizes = ppo.policy_params.action_head_sizes

    all_obs: list[np.ndarray] = []
    all_actions: list[int | np.ndarray] = []
    all_legacy_actions: list[int] = []
    all_masks: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    total_steps = 0
    total_deaths = 0
    total_completions = 0
    t_start = time.time()
    print_every = max(1, n_episodes // 20)

    print(f"\nCollecting: {n_episodes} episodes, waves {start_wave}-{max_wave}")
    print(f"  deterministic={deterministic}  device={device}")

    for ep in range(n_episodes):
        ep_seed = seed + ep
        random.seed(ep_seed)
        sim = InfernoSimulator(start_wave, max_wave)
        sim.reset()

        ep_done = False
        ep_steps = 0

        while not ep_done:
            obs = build_observation(sim.state, sim.get_ticks_in_wave())
            mask = get_mask_for_action_space(sim.state, action_head_sizes)

            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)
            mask_t = th.as_tensor(mask, dtype=th.bool).unsqueeze(0)

            actions, _, _, _, probs, _ = ppo.predict(
                obs_t, mask_t,
                deterministic=deterministic,
                return_log_probs=False,
                return_entropy=False,
                return_values=False,
                return_probs=True,
            )

            raw_action = actions.squeeze(0).detach().cpu().numpy()
            action = int(raw_action.item()) if np.asarray(raw_action).ndim == 0 else raw_action.astype(np.int32)
            legacy_action = decode_policy_action(raw_action)
            # Convert probs to logits (log-space) for distillation
            logits = th.log(probs.clamp(min=1e-8)).squeeze(0).cpu().numpy()

            all_obs.append(obs)
            all_actions.append(action)
            all_legacy_actions.append(legacy_action)
            all_masks.append(mask)
            all_logits.append(logits)

            ep_steps += 1
            total_steps += 1

            result = sim.step(legacy_action)
            if result.is_terminal():
                if result.player_died:
                    total_deaths += 1
                elif result.inferno_complete:
                    total_completions += 1
                ep_done = True

        if (ep + 1) % print_every == 0 or ep == n_episodes - 1:
            elapsed = time.time() - t_start
            eps_sec = (ep + 1) / elapsed if elapsed > 0 else 0
            eta = (n_episodes - ep - 1) / eps_sec if eps_sec > 0 else 0
            print(
                f"  [{ep+1:5d}/{n_episodes}]  steps={total_steps:,}  "
                f"deaths={total_deaths}  completions={total_completions}  "
                f"eps/s={eps_sec:.1f}  eta={eta:.0f}s"
            )

    # Build arrays
    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int32)
    action_masks = np.array(all_masks, dtype=bool)
    logits = np.array(all_logits, dtype=np.float32)

    elapsed = time.time() - t_start
    print(f"\nCollection complete in {elapsed:.1f}s")
    print(f"  Total steps:    {total_steps:,}")
    print(f"  Deaths:         {total_deaths} ({100*total_deaths/n_episodes:.1f}%)")
    print(f"  Completions:    {total_completions} ({100*total_completions/n_episodes:.1f}%)")
    print(f"  observations:   {observations.shape}")
    print(f"  actions:        {actions.shape}")
    print(f"  action_masks:   {action_masks.shape}")
    print(f"  logits:         {logits.shape}")

    # Action distribution
    legacy_actions = np.array(all_legacy_actions, dtype=np.int32)
    unique, counts = np.unique(legacy_actions, return_counts=True)
    print(f"\nAction distribution (top 10):")
    top_idx = np.argsort(-counts)[:10]
    for i in top_idx:
        print(f"  action {unique[i]:>2}: {counts[i]:>7} ({100*counts[i]/total_steps:.1f}%)")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        action_masks=action_masks,
        logits=logits,
    )
    saved_path = output_path if output_path.endswith(".npz") else output_path + ".npz"
    file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
    print(f"\nSaved to {saved_path} ({file_size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect BC data from a trained RL model"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--start-wave", type=int, default=55)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--output", type=str, default="models/bc_data/v21_rollouts.npz")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy actions (default: stochastic sampling)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed (episode i uses seed+i)")
    args = parser.parse_args()

    collect(
        model_path=args.model,
        n_episodes=args.episodes,
        start_wave=args.start_wave,
        max_wave=args.max_wave,
        output_path=args.output,
        deterministic=args.deterministic,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
