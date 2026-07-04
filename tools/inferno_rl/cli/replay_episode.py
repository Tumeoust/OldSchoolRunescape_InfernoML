"""
Tick-by-tick episode replay with value function tracking.

Replays a single episode with a fixed seed and outputs per-tick state,
action, and critic value. The value delta between ticks pinpoints exactly
when the critic detects danger.

Usage:
    python -m tools.inferno_rl.cli.replay_episode \
      --model models/V21_climb/...pt --seed 42 --start-wave 49 \
      --include-value --output-format json
"""

import argparse
import json
import random
import sys

import numpy as np
import torch as th

from ..eval import _action_name
from ..inference_state import StatefulPolicyRunner
from ..ppo.ppo import PPO
from ..simulator.simulator import InfernoSimulator
from ..training.observation import (
    TemporalState, build_observation,
    GRID_WIDTH, GRID_HEIGHT,
    MAX_HEALTH, MAX_ATTACK_COOLDOWN,
    update_temporal_state,
)
from ..training.actions import (
    decode_policy_action,
    get_mask_for_action_space,
    policy_action_mask_to_legacy_mask,
    policy_action_probabilities_to_legacy,
    uses_factored_policy_actions,
)

WEAPON_NAMES = ["BoFa", "Blowpipe", "IceBarrage", "BloodBarrage"]


def _decode_player(obs: np.ndarray) -> dict:
    return {
        "x": round(float(obs[0] * GRID_WIDTH), 1),
        "y": round(float(obs[1] * GRID_HEIGHT), 1),
        "hp": round(float(obs[2] * MAX_HEALTH)),
        "cooldown": round(float(obs[3] * MAX_ATTACK_COOLDOWN), 1),
        "weapon": WEAPON_NAMES[int(np.argmax(obs[4:8]))],
    }


def replay_episode(ppo: PPO, seed: int, start_wave: int, max_wave: int,
                   include_value: bool) -> dict:
    """Replay one episode and return structured tick data."""
    random.seed(seed)
    sim = InfernoSimulator(start_wave, max_wave)
    sim.reset()
    runner = StatefulPolicyRunner(ppo)
    runner.reset()
    temporal = TemporalState()
    uses_factored_actions = uses_factored_policy_actions(ppo.policy_params.action_head_sizes)

    ticks = []
    prev_value = None

    while True:
        obs = build_observation(
            sim.state,
            sim.get_ticks_in_wave(),
            temporal=temporal,
            dead_mobs=sim.dead_mobs,
        )
        mask = get_mask_for_action_space(sim.state, ppo.policy_params.action_head_sizes)

        tick_data = {
            "tick": sim.state.current_tick,
            "wave": sim.state.current_wave,
            "tick_in_wave": sim.get_ticks_in_wave(),
            "player": _decode_player(obs),
        }

        if include_value:
            prediction = runner.predict(
                obs,
                mask,
                deterministic=True,
                return_actions=True,
                return_values=True,
                return_probs=True,
            )
            assert prediction.action is not None
            assert prediction.value is not None
            action = decode_policy_action(prediction.action)
            value = prediction.value
            top_actions = []
            if prediction.probs is not None:
                legacy_probs = (
                    policy_action_probabilities_to_legacy(prediction.probs)
                    if uses_factored_actions
                    else prediction.probs
                )
                legacy_mask = (
                    policy_action_mask_to_legacy_mask(mask)
                    if uses_factored_actions
                    else mask
                )
                sorted_indices = np.argsort(legacy_probs)[::-1]
                for idx in sorted_indices[:5]:
                    if legacy_mask[idx]:
                        top_actions.append((int(idx), _action_name(int(idx)), float(legacy_probs[idx])))
            tick_data["value"] = round(value, 3)
            if prev_value is not None:
                tick_data["value_delta"] = round(value - prev_value, 3)
            tick_data["top_actions"] = [
                {"index": idx, "name": name, "prob": round(prob, 4)}
                for idx, name, prob in top_actions
            ]
            prev_value = value
        else:
            prediction = runner.predict(obs, mask, deterministic=True)
            assert prediction.action is not None
            action = decode_policy_action(prediction.action)

        tick_data["action"] = {"index": action, "name": _action_name(action)}

        result = sim.step(action)
        update_temporal_state(temporal, result.executed_action, result)

        if result.wave_completed:
            tick_data["event"] = "wave_completed"
        if result.player_died:
            tick_data["event"] = "death"
        if result.wave_timeout:
            tick_data["event"] = "timeout"

        ticks.append(tick_data)

        if result.is_terminal():
            break

    # Summary
    outcome = "cleared" if not (result.player_died or result.wave_timeout) else (
        "died" if result.player_died else "timeout"
    )
    return {
        "seed": seed,
        "start_wave": start_wave,
        "max_wave": max_wave,
        "outcome": outcome,
        "final_wave": result.wave_number,
        "total_ticks": len(ticks),
        "ticks": ticks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tick-by-tick episode replay")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-wave", type=int, default=49)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--include-value", action="store_true",
                        help="Include critic value and action probabilities per tick")
    parser.add_argument("--output-format", choices=["json", "text"], default="text")
    parser.add_argument("--last-n", type=int, default=0,
                        help="Only output last N ticks (0 = all)")
    args = parser.parse_args()

    ppo = PPO.load(args.model, trainable=False)
    print(f"Loaded {args.model} (trained_steps={ppo.meta.trained_steps:,})",
          file=sys.stderr)

    episode = replay_episode(ppo, args.seed, args.start_wave, args.max_wave,
                             args.include_value)

    if args.last_n > 0:
        episode["ticks"] = episode["ticks"][-args.last_n:]

    if args.output_format == "json":
        json.dump(episode, sys.stdout, indent=2)
        print()
    else:
        print(f"Seed={args.seed}  Outcome={episode['outcome']}  "
              f"Final wave={episode['final_wave']}  Ticks={episode['total_ticks']}")
        for t in episode["ticks"]:
            p = t["player"]
            line = (f"  T{t['tick']:4d} W{t['wave']:2d}  "
                    f"({p['x']:4.0f},{p['y']:4.0f}) hp={p['hp']:2d} "
                    f"cd={p['cooldown']:.0f} {p['weapon']:>12s}  "
                    f"→ {t['action']['name']}")
            if "value" in t:
                delta_str = f" Δ={t['value_delta']:+.2f}" if "value_delta" in t else ""
                line += f"  V={t['value']:.2f}{delta_str}"
            if "event" in t:
                line += f"  [{t['event'].upper()}]"
            print(line)


if __name__ == "__main__":
    main()
