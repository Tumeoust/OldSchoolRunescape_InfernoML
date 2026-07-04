"""
Death review mode for the hybrid solver workflow.

Two-phase design:
1. Headless scan — run all seeds without rendering (fast)
2. Visual review — open pygame only for death episodes, starting at death wave

Usage:
    python -m tools.inferno_rl.visualizer.review_deaths \
        --model models/V21_climb/inferno_gpu_w55-66_20260224_195520_6103.pt \
        --seeds 17,20,26,28,29 --start-wave 49 --max-wave 66
"""

import argparse
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..simulator.simulator import InfernoSimulator
from ..training.observation import TemporalState, build_observation, update_temporal_state
from ..training.actions import get_mask_for_action_space
from ..training.rewards import InfernoReward
from .run_visual import load_model
from .visualizer import InfernoVisualizer, HistoryFrame


@dataclass
class DeathRecord:
    """Result of a headless episode scan."""
    seed: int
    died: bool
    death_wave: int
    total_ticks: int
    frames: List[HistoryFrame] = field(default_factory=list)
    wave_start_index: int = 0  # Frame index where death wave began


def _scan_seed(
    seed: int,
    model,
    start_wave: int,
    max_wave: int,
    history_ticks: int,
) -> DeathRecord:
    """Run one episode headless, keeping a rolling history buffer."""
    simulator = InfernoSimulator(start_wave, max_wave)
    reward_calculator = InfernoReward()
    visualizer = InfernoVisualizer()  # Only used for _snapshot_state, never initialized

    random.seed(seed)
    simulator.reset()
    if hasattr(model, "reset"):
        model.reset()
    temporal = TemporalState()

    history = deque(maxlen=history_ticks)
    wave_start_offset = 0  # Offset into deque where current wave started
    current_wave = simulator.state.current_wave

    # Record initial frame
    snapshot = visualizer._snapshot_state(simulator.state)
    history.append(HistoryFrame(state_snapshot=snapshot, action=0, reward=0.0, tick=0))

    while True:
        obs = build_observation(
            simulator.state,
            simulator.get_ticks_in_wave(),
            temporal=temporal,
            dead_mobs=simulator.dead_mobs,
        )
        mask = get_mask_for_action_space(
            simulator.state,
            getattr(model, "action_head_sizes", [43]),
        )
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)

        result = simulator.step(action)
        update_temporal_state(temporal, result.executed_action, result)
        breakdown = reward_calculator.calculate_with_breakdown(result)

        snapshot = visualizer._snapshot_state(simulator.state)
        frame = HistoryFrame(
            state_snapshot=snapshot,
            action=action,
            reward=breakdown.total,
            tick=simulator.state.current_tick,
            reward_breakdown=breakdown.get_nonzero_components(),
        )

        # Track wave transitions — note deque may have dropped old frames
        if simulator.state.current_wave != current_wave:
            current_wave = simulator.state.current_wave
            wave_start_offset = len(history)
        history.append(frame)

        if result.is_terminal():
            frames_list = list(history)
            # Clamp wave_start_offset to valid range (deque may have evicted early frames)
            wave_start_index = min(wave_start_offset, len(frames_list) - 1)
            return DeathRecord(
                seed=seed,
                died=result.player_died,
                death_wave=simulator.state.current_wave,
                total_ticks=simulator.state.current_tick,
                frames=frames_list,
                wave_start_index=wave_start_index,
            )


def _review_loop(visualizer: InfernoVisualizer, deaths: List[DeathRecord], fps: int):
    """Visual review loop over death episodes."""
    simulator = InfernoSimulator(1, 66)  # Dummy — state restored from snapshots

    for i, death in enumerate(deaths):
        # Load frames into visualizer
        visualizer.history = death.frames
        visualizer.history_index = death.wave_start_index
        visualizer.manual_mode = True
        visualizer.review_info = {
            "seed": death.seed,
            "death_wave": death.death_wave,
            "current": i + 1,
            "total": len(deaths),
        }

        # Restore state from wave start frame
        frame = death.frames[death.wave_start_index]
        visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
        visualizer.last_action = frame.action
        visualizer.last_reward = frame.reward
        visualizer.last_reward_breakdown = frame.reward_breakdown

        caption = f"Death Review: Seed {death.seed} died W{death.death_wave} [{i+1}/{len(deaths)}]"
        import pygame
        pygame.display.set_caption(caption)

        # Initial render
        visualizer.render(simulator.state, frame.action, frame.reward)

        advance_to_next = False
        auto_play = False
        while not advance_to_next:
            visualizer._handle_events()

            if visualizer.has_next_seed_request():
                advance_to_next = True
                continue

            if visualizer.has_jump_start_request():
                visualizer.history_index = death.wave_start_index
                frame = death.frames[visualizer.history_index]
                visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                visualizer.last_action = frame.action
                visualizer.last_reward = frame.reward
                visualizer.last_reward_breakdown = frame.reward_breakdown
                visualizer.render(simulator.state, frame.action, frame.reward)
                continue

            if visualizer.has_jump_end_request():
                visualizer.history_index = len(death.frames) - 1
                frame = death.frames[visualizer.history_index]
                visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                visualizer.last_action = frame.action
                visualizer.last_reward = frame.reward
                visualizer.last_reward_breakdown = frame.reward_breakdown
                visualizer.render(simulator.state, frame.action, frame.reward)
                continue

            if visualizer.has_step_backward_request():
                frame = visualizer.step_backward_in_history()
                if frame:
                    visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                    visualizer.last_action = frame.action
                    visualizer.last_reward = frame.reward
                    visualizer.last_reward_breakdown = frame.reward_breakdown
                    visualizer.render(simulator.state, frame.action, frame.reward)
                continue

            if visualizer.has_step_forward_request():
                frame = visualizer.step_forward_in_history()
                if frame:
                    visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                    visualizer.last_action = frame.action
                    visualizer.last_reward = frame.reward
                    visualizer.last_reward_breakdown = frame.reward_breakdown
                    visualizer.render(simulator.state, frame.action, frame.reward)
                continue

            # J toggles auto-playback within the recorded history
            # (handled by visualizer.toggle_manual_mode toggling auto_play here)
            if not visualizer.manual_mode:
                # User pressed J -> auto-play forward
                auto_play = True
                visualizer.manual_mode = True  # Stay in manual for key handling

            if auto_play and visualizer.can_step_forward_in_history():
                frame = visualizer.step_forward_in_history()
                if frame:
                    visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                    visualizer.last_action = frame.action
                    visualizer.last_reward = frame.reward
                    visualizer.last_reward_breakdown = frame.reward_breakdown
                    visualizer.render(simulator.state, frame.action, frame.reward)
                    visualizer.clock.tick(fps)
                    continue
                else:
                    auto_play = False
            elif auto_play:
                auto_play = False

            # Idle re-render
            visualizer.render(simulator.state, visualizer.last_action, visualizer.last_reward)
            visualizer.clock.tick(30)

    # Cleanup review_info after all deaths
    visualizer.review_info = None


def main():
    parser = argparse.ArgumentParser(description="Death Review Mode for Hybrid Solver")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--seeds", type=str, required=True,
                        help="Comma-separated seed list (e.g. 17,20,26,28,29)")
    parser.add_argument("--start-wave", type=int, default=49)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--history-ticks", type=int, default=300,
                        help="Ticks of history to keep per episode (default 300)")
    parser.add_argument("--fps", type=int, default=4, help="Playback FPS for auto-play")
    parser.add_argument("--tile-size", type=int, default=20)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    # Phase 1: headless scan
    print(f"\nScanning {len(seeds)} seeds (W{args.start_wave}-{args.max_wave})...")
    deaths: List[DeathRecord] = []
    for seed in seeds:
        record = _scan_seed(seed, model, args.start_wave, args.max_wave, args.history_ticks)
        outcome = f"died W{record.death_wave}" if record.died else "cleared"
        print(f"  Seed {seed}: {outcome} ({record.total_ticks} ticks)")
        if record.died:
            deaths.append(record)

    if not deaths:
        print("\nNo deaths found.")
        return

    print(f"\n{len(deaths)} deaths found. Opening review...")
    print("Controls: LEFT/RIGHT=Step, HOME=Wave start, END=Death tick, N=Next seed, J=Auto-play, ESC=Quit")

    # Phase 2: visual review
    visualizer = InfernoVisualizer(tile_size=args.tile_size, fps=args.fps)
    visualizer.initialize()
    try:
        _review_loop(visualizer, deaths, args.fps)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
