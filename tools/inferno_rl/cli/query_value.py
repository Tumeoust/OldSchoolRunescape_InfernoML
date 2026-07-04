"""
Standalone value/action query — evaluate critic on arbitrary observations.

Usage:
    python -m tools.inferno_rl.cli.query_value \
      --model models/V21_climb/...pt \
      --obs 0.47,0.72,1.0,0.0,1.0,0.0,0.0,0.0,...

    python -m tools.inferno_rl.cli.query_value \
      --model models/V21_climb/...pt \
      --state-json state.json
"""

import argparse
import json
import sys

import numpy as np

from ..eval import _action_name
from ..ppo.ppo import PPO
from ..critic import get_action_and_value
from ..training.actions import get_expected_action_mask_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Query critic value and action probs")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--obs", help="Comma-separated observation floats")
    group.add_argument("--state-json", help="Path to JSON file with obs and mask arrays")
    args = parser.parse_args()

    ppo = PPO.load(args.model, trainable=False)
    expected_obs_size = ppo.policy_params.critic_input_size
    expected_mask_size = get_expected_action_mask_size(ppo.policy_params.action_head_sizes)
    print(f"Loaded {args.model} (trained_steps={ppo.meta.trained_steps:,})",
          file=sys.stderr)

    if args.obs:
        obs = np.array([float(x) for x in args.obs.split(",")], dtype=np.float32)
        if len(obs) != expected_obs_size:
            print(f"Error: expected {expected_obs_size} obs values, got {len(obs)}",
                  file=sys.stderr)
            sys.exit(1)
        mask = np.ones(expected_mask_size, dtype=bool)
    else:
        with open(args.state_json) as f:
            data = json.load(f)
        obs = np.array(data["obs"], dtype=np.float32)
        mask = np.array(data.get("mask", [True] * expected_mask_size), dtype=bool)
        if len(obs) != expected_obs_size:
            print(
                f"Error: expected {expected_obs_size} obs values, got {len(obs)}",
                file=sys.stderr,
            )
            sys.exit(1)

    action, value, top_actions = get_action_and_value(ppo, obs, mask)

    result = {
        "value": round(value, 4),
        "action": {"index": action, "name": _action_name(action)},
        "top_actions": [
            {"index": idx, "name": name, "prob": round(prob, 4)}
            for idx, name, prob in top_actions
        ],
    }

    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
