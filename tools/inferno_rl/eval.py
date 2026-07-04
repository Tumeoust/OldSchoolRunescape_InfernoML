"""
Headless evaluation script for Inferno RL models.

Usage:
    python -m tools.inferno_rl.eval --model models/V10/...122.pt --episodes 200
    python -m tools.inferno_rl.eval --model models/V10/...122.pt --compare models/V9/...4882.pt
"""

import argparse
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np

from .adaptive_curriculum import EvalSummary
from .inference_state import StatefulPolicyRunner
from .ppo.ppo import PPO
from .simulator.equipment import LoadoutId, Loadout, LOADOUTS, DEFAULT_LOADOUT
from .simulator.eval_loadouts import configure_sim_for_eval
from .simulator.simulator import InfernoSimulator
from .simulator.state import PILLAR_MAX_HP, NUM_PILLARS
from .training.observation import (
    TemporalState, build_observation, update_temporal_state,
)
from .training.actions import (
    InfernoAction,
    decode_policy_action,
    get_mask_for_action_space,
    get_movement_params,
)

EVAL_START_WAVES = [35, 46, 55, 63]
DEFAULT_MAX_WAVE = 66
DEFAULT_EPISODES = 200
DEATH_LOG_TICKS = 5


def _action_name(action: int) -> str:
    if action == 0:
        return "STAY"
    if 1 <= action <= 32:
        dx, dy, dist = get_movement_params(action)
        dirs = {(0,1):"N",(0,-1):"S",(1,0):"E",(-1,0):"W",
                (1,1):"NE",(-1,1):"NW",(1,-1):"SE",(-1,-1):"SW"}
        return f"MOVE_{dirs.get((dx,dy),'?')}_{dist}"
    if InfernoAction.is_attack(action):
        return f"ATK_T{InfernoAction.get_target_index(action) + 1}"
    names = {
        InfernoAction.NO_ACTION_IDX: "NO_OP",
        InfernoAction.SWITCH_BOFA: "SW_BOFA",
        InfernoAction.SWITCH_BLOWPIPE: "SW_BLOWPIPE",
        InfernoAction.SWITCH_ICE_BARRAGE: "SW_ICE",
        InfernoAction.SWITCH_BLOOD_BARRAGE: "SW_BLOOD",
    }
    return names.get(action, f"UNK_{action}")


class _PPOWrapper:
    def __init__(self, ppo: PPO):
        self._ppo = ppo
        self._runner = StatefulPolicyRunner(ppo)
        self.observation_version = ppo.policy_params.observation_version

    def predict(self, obs: np.ndarray, mask: np.ndarray) -> int:
        prediction = self._runner.predict(obs, mask, deterministic=True)
        assert prediction.action is not None
        return decode_policy_action(prediction.action)

    def reset(self) -> None:
        self._runner.reset()

    @property
    def ppo(self) -> PPO:
        return self._ppo


def load_model(path: str) -> _PPOWrapper:
    ppo = PPO.load(path, trainable=False)
    print(f"  Loaded {path}  (trained_steps={ppo.meta.trained_steps:,})")
    return _PPOWrapper(ppo)


def _coerce_model(model: PPO | _PPOWrapper) -> _PPOWrapper:
    if isinstance(model, _PPOWrapper):
        return model
    return _PPOWrapper(model)


def run_episodes(model: PPO | _PPOWrapper, start_wave: int, max_wave: int, n: int,
                  seed_offset: int = 0, log_deaths: bool = False,
                  loadout: Loadout | None = None,
                  real_stats: bool = False) -> dict:
    """Run n episodes from start_wave; return raw stats."""
    wrapped = _coerce_model(model)
    deaths = 0
    timeouts = 0
    completions = 0  # reached max_wave
    wave_cleared_start = 0  # cleared at least the start wave
    max_waves = []
    death_counts_by_wave: dict[int, int] = defaultdict(int)
    all_pillar_final_hp: list[list[float]] = []
    all_pillar_death_waves: list[list[int | None]] = []
    t_start = time.time()
    print_every = max(1, n // 10)

    for ep in range(n):
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
        wrapped.reset()
        temporal = TemporalState()

        ep_done = False
        max_wave_reached = start_wave
        recent_ticks = []  # ring buffer of last DEATH_LOG_TICKS
        pillar_death_waves: list[int | None] = [None] * NUM_PILLARS
        prev_pillar_alive = list(sim.state.pillar_alive)

        while not ep_done:
            obs = build_observation(
                sim.state,
                sim.get_ticks_in_wave(),
                temporal=temporal,
                dead_mobs=sim.dead_mobs,
            )
            mask = get_mask_for_action_space(
                sim.state,
                wrapped.ppo.policy_params.action_head_sizes,
            )
            action = wrapped.predict(obs, mask)

            recent_ticks.append((sim.state, obs.copy(), action, mask.copy(), sim.state.current_wave))
            if len(recent_ticks) > DEATH_LOG_TICKS:
                recent_ticks.pop(0)

            result = sim.step(action)
            update_temporal_state(temporal, result.executed_action, result)

            for pi in range(NUM_PILLARS):
                if prev_pillar_alive[pi] and not sim.state.pillar_alive[pi]:
                    pillar_death_waves[pi] = sim.state.current_wave
            prev_pillar_alive = list(sim.state.pillar_alive)

            if result.wave_number > max_wave_reached:
                max_wave_reached = result.wave_number

            if result.is_terminal():
                if result.player_died:
                    deaths += 1
                    death_counts_by_wave[max_wave_reached] += 1
                    outcome = f"died W{max_wave_reached}"
                    if log_deaths:
                        print(f"    Death at W{max_wave_reached} seed={seed}")
                elif result.inferno_complete:
                    completions += 1
                    outcome = f"cleared W{max_wave_reached}"
                else:
                    timeouts += 1
                    outcome = f"completed W{max_wave_reached}"
                if max_wave_reached > start_wave or result.inferno_complete:
                    wave_cleared_start += 1
                ep_done = True

        pillar_hp_pct = [sim.state.pillar_hp[i] / PILLAR_MAX_HP * 100 for i in range(NUM_PILLARS)]
        all_pillar_final_hp.append(pillar_hp_pct)
        all_pillar_death_waves.append(pillar_death_waves)

        max_waves.append(max_wave_reached)
        pil_str = "  ".join(
            f"{'NW NE  S'.split()[i]}={'--' if pillar_death_waves[i] is not None else f'{pillar_hp_pct[i]:.0f}%'}"
            f"{'(W' + str(pillar_death_waves[i]) + ')' if pillar_death_waves[i] is not None else ''}"
            for i in range(NUM_PILLARS)
        )
        print(f"    ep {ep:3d}  seed={seed:<5d}  {outcome:<16s}  pillars: {pil_str}", flush=True)

        done = ep + 1
        if done % print_every == 0 or done == n:
            elapsed = time.time() - t_start
            eps_per_sec = done / elapsed if elapsed > 0 else 0
            remaining = (n - done) / eps_per_sec if eps_per_sec > 0 else 0
            print(
                f"    [{done:4d}/{n}]  death={deaths/done*100:4.1f}%  "
                f"complete={completions/done*100:4.1f}%  "
                f"mean_wave={np.mean(max_waves):.1f}  "
                f"eta={remaining:.0f}s",
                flush=True,
            )

    return {
        "n": n,
        "deaths": deaths,
        "timeouts": timeouts,
        "completions": completions,
        "wave_cleared_start": wave_cleared_start,
        "max_waves": max_waves,
        "death_counts_by_wave": dict(death_counts_by_wave),
        "pillar_final_hp": all_pillar_final_hp,
        "pillar_death_waves": all_pillar_death_waves,
    }


def _run_eval_chunk(args: tuple) -> dict:
    """Worker function for multiprocessing. Loads its own model copy."""
    model_path, start_wave, max_wave, seed_start, n_episodes, worker_id, loadout_name, real_stats = args
    model = load_model(model_path)
    loadout = LOADOUTS[LoadoutId[loadout_name]] if loadout_name else None
    stats = run_episodes(model, start_wave, max_wave, n_episodes, seed_offset=seed_start,
                         loadout=loadout, real_stats=real_stats)
    return stats


def _merge_stats(results: list[dict]) -> dict:
    """Merge stats dicts from multiple workers."""
    merged_death_counts: dict[int, int] = defaultdict(int)
    for r in results:
        for wave, count in r["death_counts_by_wave"].items():
            merged_death_counts[wave] += count
    return {
        "n": sum(r["n"] for r in results),
        "deaths": sum(r["deaths"] for r in results),
        "timeouts": sum(r["timeouts"] for r in results),
        "completions": sum(r["completions"] for r in results),
        "wave_cleared_start": sum(r["wave_cleared_start"] for r in results),
        "max_waves": [w for r in results for w in r["max_waves"]],
        "death_counts_by_wave": dict(merged_death_counts),
        "pillar_final_hp": [hp for r in results for hp in r["pillar_final_hp"]],
        "pillar_death_waves": [dw for r in results for dw in r["pillar_death_waves"]],
    }


def run_episodes_parallel(model_path: str, start_wave: int, max_wave: int,
                          n: int, workers: int, loadout: Loadout | None = None,
                          real_stats: bool = False) -> dict:
    """Run n episodes across multiple processes. Each worker loads its own model."""
    chunk_size = n // workers
    remainder = n % workers
    loadout_name = loadout.id.name if loadout is not None else None

    chunks = []
    seed_cursor = 0
    for w in range(workers):
        ep_count = chunk_size + (1 if w < remainder else 0)
        chunks.append((model_path, start_wave, max_wave, seed_cursor, ep_count, w, loadout_name, real_stats))
        seed_cursor += ep_count

    t_start = time.time()
    print(f"  Spawning {workers} workers ({n} episodes, {chunk_size}-{chunk_size + 1} each)...", flush=True)

    with mp.Pool(workers) as pool:
        results = pool.map(_run_eval_chunk, chunks)

    merged = _merge_stats(results)
    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.0f}s — {merged['completions']}/{n} cleared "
          f"({merged['completions']/n*100:.1f}%), "
          f"{merged['deaths']}/{n} died ({merged['deaths']/n*100:.1f}%)", flush=True)
    return merged


def summarize_eval(stats: dict) -> EvalSummary:
    n = max(1, int(stats["n"]))
    mean_max_wave = float(np.mean(stats["max_waves"])) if stats["max_waves"] else 0.0
    return EvalSummary(
        full_clear_rate=stats["completions"] / n * 100.0,
        death_rate=stats["deaths"] / n * 100.0,
        timeout_rate=stats.get("timeouts", 0) / n * 100.0,
        mean_max_wave=mean_max_wave,
        death_counts_by_wave=dict(stats.get("death_counts_by_wave", {})),
        episodes=n,
    )


def evaluate_policy(
    model: PPO | _PPOWrapper,
    start_wave: int,
    max_wave: int,
    episodes: int,
    seed_offset: int = 0,
    log_deaths: bool = False,
    loadout: Loadout | None = None,
    real_stats: bool = False,
) -> EvalSummary:
    return summarize_eval(
        run_episodes(
            model,
            start_wave=start_wave,
            max_wave=max_wave,
            n=episodes,
            seed_offset=seed_offset,
            log_deaths=log_deaths,
            loadout=loadout,
            real_stats=real_stats,
        )
    )


def print_stats(label: str, start_wave: int, stats: dict) -> None:
    n = stats["n"]
    death_rate = stats["deaths"] / n * 100
    completion_rate = stats["completions"] / n * 100
    clear_rate = stats["wave_cleared_start"] / n * 100
    mean_wave = np.mean(stats["max_waves"])
    median_wave = np.median(stats["max_waves"])
    print(
        f"  {label:30s}  start={start_wave}  "
        f"death={death_rate:5.1f}%  "
        f"cleared_start={clear_rate:5.1f}%  "
        f"complete={completion_rate:5.1f}%  "
        f"mean_wave={mean_wave:.1f}  median={median_wave:.0f}"
    )
    pillar_names = ["NW", "NE", " S"]
    pillar_hp_data = stats.get("pillar_final_hp", [])
    pillar_dw_data = stats.get("pillar_death_waves", [])
    if pillar_hp_data:
        for pi in range(NUM_PILLARS):
            hp_vals = [ep[pi] for ep in pillar_hp_data]
            mean_hp = np.mean(hp_vals)
            death_waves = [ep[pi] for ep in pillar_dw_data if ep[pi] is not None]
            died_pct = len(death_waves) / n * 100
            dw_str = f", mean death W{np.mean(death_waves):.0f}" if death_waves else ""
            print(f"    Pillar {pillar_names[pi]}: {mean_hp:5.1f}% avg HP  "
                  f"(died {died_pct:4.1f}%{dw_str})")


def eval_model(label: str, model_path: str, start_waves: list[int], max_wave: int, episodes: int,
               seed_offset: int = 0, log_deaths: bool = False,
               loadout: Loadout | None = None,
               real_stats: bool = False, workers: int = 1) -> dict[int, dict]:
    print(f"\n=== {label} ===")
    results = {}
    for sw in start_waves:
        t0 = time.time()
        if workers > 1:
            stats = run_episodes_parallel(model_path, sw, max_wave, episodes,
                                          workers, loadout=loadout, real_stats=real_stats)
        else:
            model = load_model(model_path)
            stats = run_episodes(model, sw, max_wave, episodes, seed_offset, log_deaths, loadout,
                                 real_stats=real_stats)
        elapsed = time.time() - t0
        print_stats(f"wave {sw}", sw, stats)
        print(f"    ({elapsed:.1f}s for {episodes} eps)")
        results[sw] = stats
    return results


def compare(label_a: str, stats_a: dict, label_b: str, stats_b: dict, start_waves: list[int]) -> None:
    print(f"\n=== Delta: {label_a} vs {label_b} ===")
    for sw in start_waves:
        a = stats_a[sw]
        b = stats_b[sw]
        d_death = (b["deaths"] - a["deaths"]) / a["n"] * 100
        d_clear = (b["wave_cleared_start"] - a["wave_cleared_start"]) / a["n"] * 100
        d_complete = (b["completions"] - a["completions"]) / a["n"] * 100
        d_mean = np.mean(b["max_waves"]) - np.mean(a["max_waves"])
        sign = lambda v: ("+" if v >= 0 else "") + f"{v:.1f}"
        print(
            f"  wave {sw:2d}:  death {sign(d_death)}%  "
            f"cleared_start {sign(d_clear)}%  "
            f"complete {sign(d_complete)}%  "
            f"mean_wave {sign(d_mean)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless eval for Inferno RL model")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--compare", default=None, help="Second model to compare against")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--start-waves", default=None,
                        help="Comma-separated start waves (default: 35,46,55,63)")
    parser.add_argument("--max-wave", type=int, default=DEFAULT_MAX_WAVE)
    parser.add_argument("--seed", type=int, default=0, help="Base seed offset (episode i uses seed offset+i)")
    parser.add_argument("--log-deaths", action="store_true", help="Print decoded obs/action for last ticks before each death")
    parser.add_argument("--loadout", type=str, default=None,
                        help="Force a single loadout (e.g. CRYSTAL_BP, MAX_TBOW)")
    parser.add_argument("--all-loadouts", action="store_true",
                        help="Run eval separately for each loadout")
    parser.add_argument("--real-stats", action="store_true",
                        help="Use real defence levels and equipment defence (not 1-def training stats)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel workers (each loads its own model copy)")
    args = parser.parse_args()

    if args.loadout and args.all_loadouts:
        print("Error: --loadout and --all-loadouts are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    start_waves = (
        [int(w) for w in args.start_waves.split(",")]
        if args.start_waves
        else EVAL_START_WAVES
    )

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

    if args.all_loadouts:
        all_results: dict[str, dict[int, dict]] = {}
        for lid in LoadoutId:
            lo = LOADOUTS[lid]
            results = eval_model(
                lid.name, args.model, start_waves, args.max_wave,
                args.episodes, args.seed, args.log_deaths, loadout=lo,
                real_stats=args.real_stats, workers=args.workers,
            )
            all_results[lid.name] = results

        # Per-loadout summary table
        sw_label = ",".join(str(w) for w in start_waves)
        print(f"\n=== Per-Loadout Summary (W{sw_label}, {args.episodes} eps each) ===")
        print(f"{'Loadout':<20s} {'Clear%':>7s} {'Death%':>7s} {'Mean Wave':>10s}")
        for name, results in all_results.items():
            total_n = sum(s["n"] for s in results.values())
            total_clears = sum(s["completions"] for s in results.values())
            total_deaths = sum(s["deaths"] for s in results.values())
            all_waves = [w for s in results.values() for w in s["max_waves"]]
            mean_wave = np.mean(all_waves) if all_waves else 0.0
            print(
                f"{name:<20s} {total_clears / total_n * 100:6.1f}% "
                f"{total_deaths / total_n * 100:6.1f}% "
                f"{mean_wave:10.1f}"
            )
    else:
        results_a = eval_model(
            "Model A", args.model, start_waves, args.max_wave,
            args.episodes, args.seed, args.log_deaths, loadout=loadout,
            real_stats=args.real_stats, workers=args.workers,
        )

        if args.compare:
            results_b = eval_model(
                "Model B (compare)", args.compare, start_waves, args.max_wave,
                args.episodes, args.seed, args.log_deaths, loadout=loadout,
                real_stats=args.real_stats, workers=args.workers,
            )
            compare("A", results_a, "B", results_b, start_waves)


if __name__ == "__main__":
    main()
