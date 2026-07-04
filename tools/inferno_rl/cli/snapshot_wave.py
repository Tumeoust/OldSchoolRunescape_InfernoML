"""
Rich state snapshot CLI — dumps complete game state per tick from raw SimulatorState.

Three modes (combinable):
  --snapshot           Initial wave state only (spawns, positions, pillar HP)
  --play-rl N          RL plays N ticks, dump state after each
  --actions "A,B,C"    Replay specific actions by name

Combinable: --play-rl 5 --actions "..." = RL for 5 ticks, then custom actions.

Usage:
    python -m tools.inferno_rl.cli.snapshot_wave \
      --seed 42 --wave 62 --snapshot

    python -m tools.inferno_rl.cli.snapshot_wave \
      --seed 42 --wave 62 --model models/V21_climb/...pt --play-rl 20

    python -m tools.inferno_rl.cli.snapshot_wave \
      --seed 42 --wave 62 --actions "ATTACK_TARGET_1,STAY,SWITCH_BOFA"

    python -m tools.inferno_rl.cli.snapshot_wave \
      --seed 42 --wave 62 --model models/V21_climb/...pt \
      --play-rl 5 --actions "ATTACK_TARGET_1,STAY" \
      --output-format json
"""

import argparse
import json
import random
import sys

import numpy as np

from ..eval import _action_name
from ..simulator.simulator import InfernoSimulator
from ..training.observation import build_observation
from ..testing.actions import get_action_mask, InfernoAction
from .state_decoder import decode_full_tick, format_tick_text

# Map action name strings to action indices
ACTION_NAME_MAP = {
    "STAY": 0,
    "NO_ACTION": InfernoAction.NO_ACTION_IDX,
    "SWITCH_BOFA": InfernoAction.SWITCH_BOFA,
    "SW_BOFA": InfernoAction.SWITCH_BOFA,
    "SWITCH_BLOWPIPE": InfernoAction.SWITCH_BLOWPIPE,
    "SW_BLOWPIPE": InfernoAction.SWITCH_BLOWPIPE,
    "SWITCH_ICE_BARRAGE": InfernoAction.SWITCH_ICE_BARRAGE,
    "SW_ICE": InfernoAction.SWITCH_ICE_BARRAGE,
    "SWITCH_BLOOD_BARRAGE": InfernoAction.SWITCH_BLOOD_BARRAGE,
    "SW_BLOOD": InfernoAction.SWITCH_BLOOD_BARRAGE,
}
for target_index in range(14):
    action = 33 + target_index
    ACTION_NAME_MAP[f"ATTACK_TARGET_{target_index + 1}"] = action
    ACTION_NAME_MAP[f"ATK_T{target_index + 1}"] = action


def parse_action_name(name: str) -> int:
    """Parse an action name to its index. Supports MOVE_N_2 style too."""
    name = name.strip().upper()
    if name in ACTION_NAME_MAP:
        return ACTION_NAME_MAP[name]
    # Try MOVE_DIR_DIST pattern
    if name.startswith("MOVE_"):
        from ..testing.actions import get_movement_params
        for i in range(1, 33):
            if _action_name(i) == name:
                return i
    # Try raw int
    try:
        idx = int(name)
        if 0 <= idx < 52:
            return idx
    except ValueError:
        pass
    raise ValueError(f"Unknown action: {name!r}. "
                     f"Valid: {', '.join(sorted(ACTION_NAME_MAP.keys()))}")


def _get_rl_info(ppo, obs, mask) -> tuple[float, list[dict]]:
    """Get RL value and top action probabilities."""
    from ..critic import get_action_and_value
    action, value, top_actions = get_action_and_value(ppo, obs, mask)
    top_list = [
        {"index": idx, "name": name, "prob": round(prob, 4)}
        for idx, name, prob in top_actions
    ]
    return value, top_list


def snapshot_wave(seed: int, wave: int, start_wave: int, max_wave: int,
                  snapshot_only: bool, play_rl: int, action_list: list[int],
                  model_path: str | None, output_format: str) -> None:
    """Run the snapshot and print output."""
    # Set up simulator
    random.seed(seed)
    sim = InfernoSimulator(start_wave, max_wave)
    sim.reset()

    # Advance to target wave if needed
    if wave > start_wave:
        # Fast-forward by running waves until we reach target
        # Reset with the target wave directly
        random.seed(seed)
        sim = InfernoSimulator(wave, max_wave)
        sim.reset()

    # Load model if needed
    ppo = None
    if model_path and play_rl > 0:
        from ..ppo.ppo import PPO
        ppo = PPO.load(model_path, trainable=False)
        print(f"Loaded model: {model_path} "
              f"(trained_steps={ppo.meta.trained_steps:,})", file=sys.stderr)

    ticks_data = []
    tick_num = 0

    def capture_tick(action: int | None = None):
        nonlocal tick_num
        obs = build_observation(sim.state, sim.get_ticks_in_wave())
        mask = get_action_mask(sim.state)

        rl_value = None
        rl_top = None
        if ppo is not None:
            rl_value, rl_top = _get_rl_info(ppo, obs, mask)

        action_str = _action_name(action) if action is not None else None
        data = decode_full_tick(
            sim.state, sim.state.current_tick, sim.state.current_wave,
            sim.get_ticks_in_wave(), action, action_str,
            rl_value, rl_top,
        )
        ticks_data.append(data)
        tick_num += 1

    # Phase 1: Initial snapshot
    if snapshot_only and play_rl == 0 and not action_list:
        capture_tick()
    else:
        # Always capture initial state
        capture_tick()

    # Phase 2: RL plays N ticks
    if play_rl > 0 and ppo is not None:
        for _ in range(play_rl):
            obs = build_observation(sim.state, sim.get_ticks_in_wave())
            mask = get_action_mask(sim.state)
            from ..critic import get_action_and_value
            action, _, _ = get_action_and_value(ppo, obs, mask)
            result = sim.step(action)
            capture_tick(action)
            if result.is_terminal():
                break

    # Phase 3: Custom actions
    for action in action_list:
        result = sim.step(action)
        capture_tick(action)
        if result.is_terminal():
            break

    # Output
    output = {
        "seed": seed,
        "wave": wave,
        "ticks": ticks_data,
    }

    if output_format == "json":
        json.dump(output, sys.stdout, indent=2)
        print()
    else:
        print(f"Seed={seed}  Wave={wave}  Ticks={len(ticks_data)}")
        print()
        for data in ticks_data:
            print(format_tick_text(data))
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rich state snapshot — dump complete game state per tick",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--wave", type=int, required=True, help="Wave number to snapshot")
    parser.add_argument("--start-wave", type=int, default=None,
                        help="Start wave for simulator (defaults to --wave)")
    parser.add_argument("--max-wave", type=int, default=66, help="Max wave (default: 66)")
    parser.add_argument("--snapshot", action="store_true",
                        help="Show initial wave state only")
    parser.add_argument("--play-rl", type=int, default=0,
                        help="Let RL play N ticks")
    parser.add_argument("--actions", type=str, default="",
                        help="Comma-separated action names to replay")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to .pt checkpoint (required for --play-rl)")
    parser.add_argument("--output-format", choices=["text", "json"], default="text",
                        help="Output format (default: text)")

    args = parser.parse_args()

    start_wave = args.start_wave if args.start_wave is not None else args.wave

    # Parse action list
    action_list = []
    if args.actions:
        for name in args.actions.split(","):
            action_list.append(parse_action_name(name))

    # Validation
    if args.play_rl > 0 and not args.model:
        parser.error("--play-rl requires --model")

    if not args.snapshot and args.play_rl == 0 and not action_list:
        # Default to snapshot mode
        args.snapshot = True

    snapshot_wave(
        seed=args.seed,
        wave=args.wave,
        start_wave=start_wave,
        max_wave=args.max_wave,
        snapshot_only=args.snapshot,
        play_rl=args.play_rl,
        action_list=action_list,
        model_path=args.model,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
