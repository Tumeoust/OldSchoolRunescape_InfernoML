"""
Headless model run with per-wave JSON logging (same format as play_human --log-dir).

Usage:
    python -m tools.inferno_rl.model_log_run \
        --model models/V25_drill/inferno_gpu_w49-66_20260227_133128_1600.pt \
        --start-wave 49 --max-wave 66 --seed 42 \
        --log-dir logs/model_v25_seed42
"""

import argparse
import json
import os
import random
from typing import Dict, Optional

from .eval import load_model, _action_name
from .simulator.simulator import InfernoSimulator
from .training.observation import TemporalState, build_observation, update_temporal_state
from .training.actions import get_mask_for_action_space
from .training.rewards import InfernoReward, normalize_reward_term_name


def run_logged_episode(
    model_path: str,
    start_wave: int,
    max_wave: int,
    seed: int,
    log_dir: str,
):
    os.makedirs(log_dir, exist_ok=True)

    model = load_model(model_path)
    sim = InfernoSimulator(start_wave, max_wave)
    sim.initial_barrage_enabled = False
    reward_calc = InfernoReward()

    random.seed(seed)
    sim.reset()
    model.reset()

    total_reward = 0.0
    tick_count = 0
    cumulative_categories: Dict[str, float] = {}
    wave_tick_logs = []
    current_log_wave = start_wave
    done = False
    temporal = TemporalState()

    while not done:
        obs = build_observation(
            sim.state,
            sim.get_ticks_in_wave(),
            temporal=temporal,
            dead_mobs=sim.dead_mobs,
        )
        mask = get_mask_for_action_space(sim.state, model.ppo.policy_params.action_head_sizes)
        action = model.predict(obs, mask)

        result = sim.step(action)
        update_temporal_state(temporal, result.executed_action, result)
        breakdown = reward_calc.calculate_with_breakdown(result)
        total_reward += breakdown.total
        tick_count += 1

        # Per-tick entry
        entry = {
            "tick": tick_count,
            "wave": sim.state.current_wave,
            "action": _action_name(action),
            "player_pos": [sim.state.player_x, sim.state.player_y],
            "player_hp": result.health_at_step_start,
            "weapon": sim.state.current_preset.value,
            "enemies_remaining": result.enemies_remaining,
            "npcs_with_los": result.npcs_with_los_now,
            "tick_reward": round(breakdown.total, 4),
            "cumulative_reward": round(total_reward, 4),
            "components": {
                name: round(value, 4)
                for name, value in breakdown.get_nonzero_components()
            },
        }
        wave_tick_logs.append(entry)

        # Accumulate categories
        for name, value in breakdown.get_nonzero_components():
            norm = normalize_reward_term_name(name)
            cumulative_categories[norm] = cumulative_categories.get(norm, 0.0) + value

        # Wave transition
        if result.wave_completed:
            _flush_wave_log(log_dir, current_log_wave, wave_tick_logs, None)
            wave_tick_logs = []
            current_log_wave = sim.state.current_wave

        # Terminal
        term_type = None
        if result.is_terminal():
            done = True
            if result.player_died:
                term_type = "DEATH"
            elif result.inferno_complete:
                term_type = "COMPLETE"
            else:
                term_type = "TIMEOUT"
            # Flush remaining ticks for the final wave
            if wave_tick_logs:
                _flush_wave_log(log_dir, current_log_wave, wave_tick_logs, term_type)
                wave_tick_logs = []

    # Episode summary
    summary = {
        "seed": seed,
        "start_wave": start_wave,
        "max_wave_reached": sim.state.current_wave,
        "total_ticks": tick_count,
        "total_reward": round(total_reward, 4),
        "terminal": term_type,
        "category_totals": {k: round(v, 4) for k, v in cumulative_categories.items()},
    }
    filepath = os.path.join(log_dir, "episode_summary.json")
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEpisode summary -> {filepath}")
    print(f"  Terminal: {term_type}, Max wave: {sim.state.current_wave}, "
          f"Ticks: {tick_count}, Reward: {total_reward:+.1f}")


def _flush_wave_log(
    log_dir: str,
    wave: int,
    tick_logs: list,
    terminal_type: Optional[str],
):
    if not tick_logs:
        return
    filepath = os.path.join(log_dir, f"wave_{wave:02d}.json")
    wave_reward = sum(t["tick_reward"] for t in tick_logs)
    category_totals: Dict[str, float] = {}
    for tick_entry in tick_logs:
        for name, value in tick_entry["components"].items():
            norm = normalize_reward_term_name(name)
            category_totals[norm] = category_totals.get(norm, 0.0) + value

    wave_data = {
        "wave": wave,
        "ticks": len(tick_logs),
        "total_reward": round(wave_reward, 4),
        "terminal": terminal_type,
        "category_totals": {k: round(v, 4) for k, v in category_totals.items()},
        "ticks_log": tick_logs,
    }
    with open(filepath, "w") as f:
        json.dump(wave_data, f, indent=2)
    print(f"  Logged wave {wave}: {len(tick_logs)} ticks, reward {wave_reward:+.1f} -> {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Headless model run with per-wave JSON logging")
    parser.add_argument("--model", "-m", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--start-wave", type=int, default=49)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", required=True, help="Output directory for wave logs")
    args = parser.parse_args()

    run_logged_episode(args.model, args.start_wave, args.max_wave, args.seed, args.log_dir)


if __name__ == "__main__":
    main()
