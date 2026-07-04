"""
Per-wave death analysis for Inferno RL models.

Runs batch evaluation and produces a histogram of which waves kill the model.

Usage:
    python -m tools.inferno_rl.death_analysis \
      --model models/V25_drill/inferno_gpu_w49-66_..._1100.pt \
      --episodes 100 --start-wave 49 --max-wave 66 --seed 0

    # Parallel (10 workers):
    python -m tools.inferno_rl.death_analysis \
      --model models/V44/...1900.pt \
      --episodes 100 --start-wave 49 --max-wave 66 --seed 0 --workers 10
"""

import argparse
import multiprocessing as mp
import random
import sys
import time
from collections import Counter

import numpy as np

from .eval import _PPOWrapper, load_model
from .simulator.equipment import LoadoutId, Loadout, LOADOUTS, DEFAULT_LOADOUT
from .simulator.eval_loadouts import configure_sim_for_eval
from .simulator.simulator import InfernoSimulator
from .training.observation import TemporalState, build_observation, update_temporal_state
from .training.actions import get_mask_for_action_space


def _run_chunk(args: tuple) -> Counter:
    """Worker function for multiprocessing. Loads its own model copy."""
    model_path, start_wave, max_wave, seed_start, n_episodes, worker_id, loadout_name, real_stats = args
    model = load_model(model_path)
    loadout = LOADOUTS[LoadoutId[loadout_name]] if loadout_name else None
    death_waves = Counter()

    for i in range(n_episodes):
        seed = seed_start + i
        random.seed(seed)
        sim = InfernoSimulator(start_wave, max_wave)
        sim.initial_barrage_enabled = True
        if loadout is not None:
            sim.set_loadout(loadout)
        if real_stats:
            lid = loadout.id if loadout is not None else DEFAULT_LOADOUT.id
            configure_sim_for_eval(sim, lid)
        sim.reset()
        model.reset()
        temporal = TemporalState()

        done = False
        outcome = 0

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

            if result.is_terminal():
                if result.player_died:
                    outcome = result.wave_number
                elif result.wave_timeout:
                    outcome = -result.wave_number
                else:
                    outcome = 0
                done = True

        death_waves[outcome] += 1
        if outcome > 0:
            status = f"died W{outcome}"
        elif outcome < 0:
            status = f"timeout W{-outcome}"
        else:
            status = "cleared"
        print(f"  [w{worker_id}] seed={seed:<5d}  {status}", flush=True)

    return death_waves


def run_death_analysis(model: _PPOWrapper, start_wave: int, max_wave: int,
                       episodes: int, seed_offset: int,
                       loadout: Loadout | None = None,
                       real_stats: bool = False) -> Counter:
    """Run episodes sequentially. Use run_death_analysis_parallel for multiprocessing."""
    death_waves = Counter()
    t_start = time.time()
    print_every = max(1, episodes // 10)

    for ep in range(episodes):
        seed = seed_offset + ep
        random.seed(seed)
        sim = InfernoSimulator(start_wave, max_wave)
        sim.initial_barrage_enabled = True
        if loadout is not None:
            sim.set_loadout(loadout)
        if real_stats:
            lid = loadout.id if loadout is not None else DEFAULT_LOADOUT.id
            configure_sim_for_eval(sim, lid)
        sim.reset()
        model.reset()
        temporal = TemporalState()

        done = False
        outcome = 0  # 0 = cleared

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

            if result.is_terminal():
                if result.player_died:
                    outcome = result.wave_number  # positive = death
                elif result.wave_timeout:
                    outcome = -result.wave_number  # negative = timeout
                else:
                    outcome = 0  # cleared
                done = True

        death_waves[outcome] += 1
        if outcome > 0:
            status = f"died W{outcome}"
        elif outcome < 0:
            status = f"timeout W{-outcome}"
        else:
            status = "cleared"
        print(f"  ep {ep:3d}  seed={seed:<5d}  {status}", flush=True)

        finished = ep + 1
        if finished % print_every == 0 or finished == episodes:
            elapsed = time.time() - t_start
            eps_per_sec = finished / elapsed if elapsed > 0 else 0
            eta = (episodes - finished) / eps_per_sec if eps_per_sec > 0 else 0
            cleared = death_waves[0]
            died = sum(v for k, v in death_waves.items() if k > 0)
            timed_out = sum(v for k, v in death_waves.items() if k < 0)
            print(
                f"  [{finished:4d}/{episodes}]  "
                f"died={died}/{finished} ({died/finished*100:.1f}%)  "
                f"timeout={timed_out}/{finished} ({timed_out/finished*100:.1f}%)  "
                f"eta={eta:.0f}s",
                flush=True,
            )

    return death_waves


def run_death_analysis_parallel(model_path: str, start_wave: int, max_wave: int,
                                episodes: int, seed_offset: int,
                                workers: int,
                                loadout: Loadout | None = None,
                                real_stats: bool = False) -> Counter:
    """Run episodes across multiple processes. Each worker loads its own model."""
    chunk_size = episodes // workers
    remainder = episodes % workers
    loadout_name = loadout.id.name if loadout is not None else None

    chunks = []
    seed_cursor = seed_offset
    for w in range(workers):
        n = chunk_size + (1 if w < remainder else 0)
        chunks.append((model_path, start_wave, max_wave, seed_cursor, n, w, loadout_name, real_stats))
        seed_cursor += n

    t_start = time.time()
    print(f"  Spawning {workers} workers ({episodes} episodes, {chunk_size}-{chunk_size + 1} each)...", flush=True)

    with mp.Pool(workers) as pool:
        results = pool.map(_run_chunk, chunks)

    merged = Counter()
    for r in results:
        merged += r

    elapsed = time.time() - t_start
    cleared = merged[0]
    died = sum(v for k, v in merged.items() if k > 0)
    print(f"  Done in {elapsed:.0f}s — {cleared}/{episodes} cleared ({cleared/episodes*100:.1f}%), "
          f"{died}/{episodes} died ({died/episodes*100:.1f}%)", flush=True)
    return merged


def print_histogram(model_path: str, start_wave: int, max_wave: int,
                    episodes: int, seed_offset: int, death_waves: Counter) -> None:
    """Print the per-wave death/timeout histogram table."""
    cleared = death_waves[0]
    total_died = sum(v for k, v in death_waves.items() if k > 0)
    total_timeout = sum(v for k, v in death_waves.items() if k < 0)

    print(f"\nDeath Analysis: {model_path}")
    print(f"{episodes} episodes, seeds {seed_offset}-{seed_offset + episodes - 1}, waves {start_wave}-{max_wave}\n")
    print(f"{'Wave':>4s} | {'Deaths':>6s} | {'Timeouts':>8s} | {'Survival':>8s} | {'Cum.Survival':>12s}")
    print(f"-----+--------+----------+----------+-------------")

    survivors = episodes
    for wave in range(start_wave, max_wave + 1):
        deaths_this_wave = death_waves.get(wave, 0)
        timeouts_this_wave = death_waves.get(-wave, 0)
        failures = deaths_this_wave + timeouts_this_wave
        survival_pct = (1 - failures / survivors) * 100 if survivors > 0 else 0
        survivors -= failures
        cum_survival_pct = survivors / episodes * 100
        print(f"  {wave:>2d} | {deaths_this_wave:>6d} | {timeouts_this_wave:>8d} | {survival_pct:>7.1f}% | {cum_survival_pct:>11.1f}%")

    print(f"-----+--------+----------+----------+-------------")
    print(f"     |   CLEARED:  {cleared}/{episodes} ({cleared/episodes*100:.1f}%)")
    print(f"     |   DIED:     {total_died}/{episodes} ({total_died/episodes*100:.1f}%)")
    print(f"     |   TIMEOUT:  {total_timeout}/{episodes} ({total_timeout/episodes*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-wave death analysis for Inferno RL")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-wave", type=int, default=49)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--seed", type=int, default=0, help="Base seed offset")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (each loads its own model copy)")
    parser.add_argument("--loadout", type=str, default=None,
                        help="Force a single loadout (e.g. CRYSTAL_BP, MAX_TBOW)")
    parser.add_argument("--real-stats", action="store_true",
                        help="Use real defence levels and equipment defence (not 1-def training stats)")
    args = parser.parse_args()

    loadout: Loadout | None = None
    if args.loadout:
        try:
            loadout = LOADOUTS[LoadoutId[args.loadout]]
        except KeyError:
            valid = ", ".join(lid.name for lid in LoadoutId)
            print(f"Error: unknown loadout '{args.loadout}'. Valid: {valid}", file=sys.stderr)
            sys.exit(1)

    if args.real_stats:
        print("  [real-stats] Using real defence levels and equipment defence bonuses")

    if args.workers > 1:
        death_waves = run_death_analysis_parallel(
            args.model, args.start_wave, args.max_wave,
            args.episodes, args.seed, args.workers,
            loadout=loadout, real_stats=args.real_stats,
        )
    else:
        model = load_model(args.model)
        death_waves = run_death_analysis(model, args.start_wave, args.max_wave, args.episodes, args.seed,
                                         loadout=loadout, real_stats=args.real_stats)
    print_histogram(args.model, args.start_wave, args.max_wave, args.episodes, args.seed, death_waves)


if __name__ == "__main__":
    main()
