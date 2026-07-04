"""
Eval runner for hybrid solver with A/B comparison against pure RL.

Usage:
    python -m tools.inferno_rl.cli.run_hybrid \
      --model models/V21_climb/...pt \
      --overrides tools/inferno_rl/hybrid/overrides.json \
      --episodes 200 --compare-pure-rl --output-format json
"""

import argparse
import json
import random
import sys
import time
from collections import Counter

import numpy as np

from ..eval import load_model
from ..ppo.ppo import PPO
from ..simulator.simulator import InfernoSimulator
from ..training.observation import build_observation
from ..training.actions import get_action_mask
from ..hybrid.overrides import load_rules
from ..hybrid.solver import HybridSolver


def run_hybrid_episodes(solver: HybridSolver, start_wave: int, max_wave: int,
                        episodes: int, seed_offset: int,
                        label: str = "hybrid") -> dict:
    """Run episodes with the hybrid solver, return stats."""
    death_waves = Counter()
    t_start = time.time()
    print_every = max(1, episodes // 10)
    total_override_counts: dict[str, int] = {}

    for ep in range(episodes):
        seed = seed_offset + ep
        random.seed(seed)
        sim = InfernoSimulator(start_wave, max_wave)
        sim.reset()
        solver.reset()

        outcome = 0  # 0 = cleared

        while True:
            tick_result = solver.step(sim)
            result = sim.step(tick_result.action)

            if result.is_terminal():
                if result.player_died:
                    outcome = result.wave_number
                elif result.wave_timeout:
                    outcome = -result.wave_number
                break

        death_waves[outcome] += 1

        # Accumulate override stats
        for rule_name, count in solver.get_override_stats().items():
            total_override_counts[rule_name] = (
                total_override_counts.get(rule_name, 0) + count
            )

        status = ("cleared" if outcome == 0 else
                  f"died W{outcome}" if outcome > 0 else
                  f"timeout W{-outcome}")
        print(f"  [{label}] ep {ep:3d}  seed={seed:<5d}  {status}", flush=True)

        finished = ep + 1
        if finished % print_every == 0 or finished == episodes:
            elapsed = time.time() - t_start
            eta = (episodes - finished) / (finished / elapsed) if elapsed > 0 else 0
            cleared = death_waves[0]
            died = sum(v for k, v in death_waves.items() if k > 0)
            print(f"  [{label}] [{finished:4d}/{episodes}]  "
                  f"cleared={cleared}/{finished} ({cleared/finished*100:.1f}%)  "
                  f"died={died}/{finished} ({died/finished*100:.1f}%)  "
                  f"eta={eta:.0f}s", flush=True)

    cleared = death_waves[0]
    total_died = sum(v for k, v in death_waves.items() if k > 0)
    total_timeout = sum(v for k, v in death_waves.items() if k < 0)

    per_wave = {}
    for wave in range(start_wave, max_wave + 1):
        deaths = death_waves.get(wave, 0)
        timeouts = death_waves.get(-wave, 0)
        if deaths > 0 or timeouts > 0:
            per_wave[str(wave)] = {"deaths": deaths, "timeouts": timeouts}

    return {
        "episodes": episodes,
        "clear_rate": round(cleared / episodes, 4),
        "death_rate": round(total_died / episodes, 4),
        "timeout_rate": round(total_timeout / episodes, 4),
        "cleared": cleared,
        "died": total_died,
        "timed_out": total_timeout,
        "per_wave": per_wave,
        "override_counts": total_override_counts,
    }


def run_pure_rl_episodes(ppo_wrapper, start_wave: int, max_wave: int,
                         episodes: int, seed_offset: int) -> dict:
    """Run episodes with pure RL policy, return stats."""
    from ..death_analysis import run_death_analysis

    death_waves = run_death_analysis(ppo_wrapper, start_wave, max_wave,
                                     episodes, seed_offset)

    cleared = death_waves[0]
    total_died = sum(v for k, v in death_waves.items() if k > 0)
    total_timeout = sum(v for k, v in death_waves.items() if k < 0)

    per_wave = {}
    for wave in range(start_wave, max_wave + 1):
        deaths = death_waves.get(wave, 0)
        timeouts = death_waves.get(-wave, 0)
        if deaths > 0 or timeouts > 0:
            per_wave[str(wave)] = {"deaths": deaths, "timeouts": timeouts}

    return {
        "episodes": episodes,
        "clear_rate": round(cleared / episodes, 4),
        "death_rate": round(total_died / episodes, 4),
        "timeout_rate": round(total_timeout / episodes, 4),
        "cleared": cleared,
        "died": total_died,
        "timed_out": total_timeout,
        "per_wave": per_wave,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid solver eval with A/B comparison")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--overrides", required=True, help="Path to overrides JSON")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--start-wave", type=int, default=49)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-pure-rl", action="store_true",
                        help="Also run pure RL baseline for comparison")
    parser.add_argument("--output-format", choices=["json", "text"], default="text")
    args = parser.parse_args()

    # Load model and rules
    ppo = PPO.load(args.model, trainable=False)
    print(f"Loaded {args.model} (trained_steps={ppo.meta.trained_steps:,})",
          file=sys.stderr)
    rules = load_rules(args.overrides)
    enabled_rules = [r for r in rules if r.enabled]
    print(f"Loaded {len(rules)} rules ({len(enabled_rules)} enabled)",
          file=sys.stderr)

    # Run hybrid
    solver = HybridSolver(ppo, rules)
    hybrid_stats = run_hybrid_episodes(
        solver, args.start_wave, args.max_wave, args.episodes, args.seed,
    )

    # Optionally run pure RL baseline
    baseline_stats = None
    if args.compare_pure_rl:
        model_wrapper = load_model(args.model)
        baseline_stats = run_pure_rl_episodes(
            model_wrapper, args.start_wave, args.max_wave,
            args.episodes, args.seed,
        )

    if args.output_format == "json":
        result = {
            "model": args.model,
            "overrides": args.overrides,
            "start_wave": args.start_wave,
            "max_wave": args.max_wave,
            "hybrid": hybrid_stats,
        }
        if baseline_stats:
            result["baseline"] = baseline_stats
            delta = hybrid_stats["clear_rate"] - baseline_stats["clear_rate"]
            result["delta"] = f"{delta:+.1%}"
        json.dump(result, sys.stdout, indent=2)
        print()
    else:
        print(f"\n=== Hybrid Solver Results ===")
        print(f"  Clear rate: {hybrid_stats['clear_rate']:.1%}  "
              f"({hybrid_stats['cleared']}/{hybrid_stats['episodes']})")
        print(f"  Death rate: {hybrid_stats['death_rate']:.1%}")
        print(f"  Timeout rate: {hybrid_stats['timeout_rate']:.1%}")
        if hybrid_stats.get("override_counts"):
            print(f"\n  Override rule fire counts:")
            for name, count in sorted(hybrid_stats["override_counts"].items(),
                                       key=lambda x: -x[1]):
                print(f"    {name}: {count}")

        if baseline_stats:
            print(f"\n=== Pure RL Baseline ===")
            print(f"  Clear rate: {baseline_stats['clear_rate']:.1%}  "
                  f"({baseline_stats['cleared']}/{baseline_stats['episodes']})")
            print(f"  Death rate: {baseline_stats['death_rate']:.1%}")
            delta = hybrid_stats["clear_rate"] - baseline_stats["clear_rate"]
            print(f"\n=== Delta: {delta:+.1%} ===")


if __name__ == "__main__":
    main()
