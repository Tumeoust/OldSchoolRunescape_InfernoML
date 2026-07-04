"""
Per-wave death analysis for V21 Inferno RL models.

V21 uses 186-dim MLP-only observations (no LSTM, no pillar-relative features,
no nibbler entity slots). This script reimplements the V21 observation builder
to evaluate V21 checkpoints against the current simulator.

Usage:
    python -m tools.inferno_rl.death_analysis_v21 \
      --model models/V21_climb/inferno_gpu_w55-66_..._2200.pt \
      --episodes 100 --start-wave 49 --max-wave 66 --seed 0
"""

import argparse
import random
import time
from collections import Counter

import numpy as np

from .eval import _PPOWrapper, load_model
from .simulator.simulator import InfernoSimulator
from .simulator.entity import EntityTypes
from .simulator.equipment import GearPreset
from .simulator.geometry import GRID_WIDTH, GRID_HEIGHT, InfernoLineOfSight
from .training.actions import get_action_mask

# V21 observation constants
V21_PLAYER_STATE_SIZE = 8
V21_PILLAR_STATE_SIZE = 12
V21_ENTITY_SLOT_SIZE = 10
V21_MAX_ENTITY_SLOTS = 16
V21_WAVE_CONTEXT_SIZE = 6
V21_OBSERVATION_SIZE = 186  # 8 + 12 + 160 + 6

V21_MAX_HEALTH = 99.0
V21_MAX_WAVE = 66.0
V21_MAX_PILLAR_HP = 255.0
V21_MAX_ATTACK_COOLDOWN = 5.0
V21_MAX_ENTITY_COOLDOWN = 8.0
V21_MAX_TICK_IN_WAVE = 500.0
V21_MAX_DISTANCE = 30.0
V21_NUM_ENTITY_TYPES = 8
V21_NUM_WEAPON_TYPES = 4

V21_PILLAR_CENTERS = [
    (1.0, 21.0),   # NW
    (18.0, 23.0),  # NE
    (11.0, 7.0),   # S
]

# V21 weapon indices: 0=ranged(tbow/rcb/acb/bofa), 1=blowpipe, 2=ice barrage, 3=blood barrage
# Blood barrage gets its own slot (index 3) so the model can distinguish ice vs blood.

V21_ENTITY_TYPE_MAP = {
    EntityTypes.MAGER: 0,
    EntityTypes.RANGER: 1,
    EntityTypes.MELEE: 2,
    EntityTypes.BLOB: 3,
    EntityTypes.BAT: 4,
    EntityTypes.NIBBLER: 5,
    EntityTypes.JAD: 6,
    EntityTypes.BLOB_MAGE: 7,
    EntityTypes.BLOB_RANGE: 7,
    EntityTypes.BLOB_MELEE: 7,
}


def v21_get_sorted_entities(state):
    """V21 entity sorting: by threat priority, excludes nibblers."""
    entities = [
        e for e in state.entities
        if not e.is_dead() and e.entity_type != EntityTypes.NIBBLER
    ]
    entities.sort(key=lambda e: e.entity_type.base_priority)
    return entities


def v21_build_observation(state, tick_in_wave):
    """Build 186-dim V21 observation from current simulator state."""
    obs = np.zeros(V21_OBSERVATION_SIZE, dtype=np.float32)
    idx = 0

    # Player state (8)
    obs[idx] = state.player_x / GRID_WIDTH
    obs[idx + 1] = state.player_y / GRID_HEIGHT
    obs[idx + 2] = state.player_health / V21_MAX_HEALTH
    obs[idx + 3] = min(state.get_player_attack_cooldown() / V21_MAX_ATTACK_COOLDOWN, 1.0)
    idx += 4
    if state.current_preset == GearPreset.BOFA:
        weapon_idx = 0
    elif state.current_preset == GearPreset.BLOWPIPE:
        weapon_idx = 1
    else:
        weapon_idx = 3 if state.use_blood_barrage else 2
    for i in range(V21_NUM_WEAPON_TYPES):
        obs[idx + i] = 1.0 if i == weapon_idx else 0.0
    idx += V21_NUM_WEAPON_TYPES

    # Pillar state (12)
    for i in range(3):
        obs[idx] = 1.0 if state.pillar_alive[i] else 0.0
        obs[idx + 1] = state.pillar_hp[i] / V21_MAX_PILLAR_HP
        obs[idx + 2] = V21_PILLAR_CENTERS[i][0] / GRID_WIDTH
        obs[idx + 3] = V21_PILLAR_CENTERS[i][1] / GRID_HEIGHT
        idx += 4

    # Entity slots (160)
    sorted_entities = v21_get_sorted_entities(state)
    for slot in range(V21_MAX_ENTITY_SLOTS):
        if slot < len(sorted_entities):
            e = sorted_entities[slot]
            obs[idx] = 1.0
            obs[idx + 1] = V21_ENTITY_TYPE_MAP.get(e.entity_type, 7) / V21_NUM_ENTITY_TYPES
            obs[idx + 2] = e.x / GRID_WIDTH
            obs[idx + 3] = e.y / GRID_HEIGHT
            obs[idx + 4] = e.current_health / e.entity_type.max_health
            obs[idx + 5] = min(max(0, e.attack_delay) / V21_MAX_ENTITY_COOLDOWN, 1.0)
            obs[idx + 6] = 1.0 if InfernoLineOfSight.can_entity_attack_player(
                e, state.player_x, state.player_y, state.pillar_alive) else 0.0
            obs[idx + 7] = 1.0 if InfernoLineOfSight.can_player_attack_entity(
                state.player_x, state.player_y, state.player_attack_range, e, state.pillar_alive) else 0.0
            dist = InfernoLineOfSight.get_distance_from_npc(
                e.x, e.y, e.entity_type.size_in_tiles, state.player_x, state.player_y)
            obs[idx + 8] = min(dist / V21_MAX_DISTANCE, 1.0)
            obs[idx + 9] = 1.0 if (e.frozen > 0 or e.stunned > 0) else 0.0
        idx += V21_ENTITY_SLOT_SIZE

    # Wave context (6)
    obs[idx] = state.current_wave / V21_MAX_WAVE
    obs[idx + 1] = min(tick_in_wave / V21_MAX_TICK_IN_WAVE, 1.0)
    total = len(state.entities)
    alive = len([e for e in state.entities if not e.is_dead()])
    obs[idx + 2] = alive / total if total > 0 else 0.0
    obs[idx + 3] = sum(1 for a in state.pillar_alive if a) / 3.0
    nibbler_count = sum(1 for e in state.entities if not e.is_dead() and e.entity_type == EntityTypes.NIBBLER)
    obs[idx + 4] = min(nibbler_count / 10.0, 1.0)
    threat_count = sum(1 for e in state.entities if not e.is_dead() and e.entity_type != EntityTypes.NIBBLER)
    obs[idx + 5] = min(threat_count / 10.0, 1.0)

    return obs


def run_death_analysis(model, start_wave, max_wave, episodes, seed_offset):
    """Run episodes and return Counter mapping outcome -> count.

    Keys: positive int = died on that wave, 0 = cleared, negative int = timed out on that wave.
    """
    death_waves = Counter()
    t_start = time.time()
    print_every = max(1, episodes // 10)

    for ep in range(episodes):
        seed = seed_offset + ep
        random.seed(seed)
        sim = InfernoSimulator(start_wave, max_wave)
        sim.reset()
        model.reset()

        done = False
        outcome = 0

        while not done:
            obs = v21_build_observation(sim.state, sim.get_ticks_in_wave())
            mask = get_action_mask(sim.state)
            action = model.predict(obs, mask)
            result = sim.step(action)

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
        print(f"  ep {ep:3d}  seed={seed:<5d}  {status}", flush=True)

        finished = ep + 1
        if finished % print_every == 0 or finished == episodes:
            elapsed = time.time() - t_start
            eps_per_sec = finished / elapsed if elapsed > 0 else 0
            eta = (episodes - finished) / eps_per_sec if eps_per_sec > 0 else 0
            cleared = death_waves[0]
            died = sum(v for k, v in death_waves.items() if k > 0)
            timed_out = sum(v for k, v in death_waves.items() if k < 0)
            print(f"  [{finished:4d}/{episodes}]  died={died}/{finished} ({died/finished*100:.1f}%)  timeout={timed_out}/{finished} ({timed_out/finished*100:.1f}%)  eta={eta:.0f}s", flush=True)

    return death_waves


def print_histogram(model_path, start_wave, max_wave, episodes, seed_offset, death_waves):
    """Print per-wave death/timeout histogram."""
    cleared = death_waves[0]
    total_died = sum(v for k, v in death_waves.items() if k > 0)
    total_timeout = sum(v for k, v in death_waves.items() if k < 0)

    print(f"\nDeath Analysis (V21): {model_path}")
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


def main():
    parser = argparse.ArgumentParser(description="Per-wave death analysis for V21 Inferno RL")
    parser.add_argument("--model", required=True, help="Path to V21 .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-wave", type=int, default=49)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.model)
    death_waves = run_death_analysis(model, args.start_wave, args.max_wave, args.episodes, args.seed)
    print_histogram(args.model, args.start_wave, args.max_wave, args.episodes, args.seed, death_waves)


if __name__ == "__main__":
    main()
