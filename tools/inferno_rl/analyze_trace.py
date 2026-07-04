"""
Benchmark real Inferno trace data against the Python simulator.

Compares: LOS, NPC pathfinding, attack timing, blob splits.
Loads a JSONL trace from InfernoLogging and replays each tick
through the simulator's geometry/pathfinding to find disagreements.
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Direct module imports to avoid triggering simulator/__init__.py
# which pulls in the full sim with training dependencies
import importlib.util

_sim_dir = os.path.join(os.path.dirname(__file__), "simulator")

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_entity_mod = _load_module("simulator.entity", os.path.join(_sim_dir, "entity.py"))
_geo_mod = _load_module("simulator.geometry", os.path.join(_sim_dir, "geometry.py"))
_path_mod = _load_module("simulator.pathfinding", os.path.join(_sim_dir, "pathfinding.py"))

SimulatorGeometry = _geo_mod.SimulatorGeometry
InfernoLineOfSight = _geo_mod.InfernoLineOfSight
PILLARS = _geo_mod.PILLARS
GRID_WIDTH = _geo_mod.GRID_WIDTH
GRID_HEIGHT = _geo_mod.GRID_HEIGHT

OSRSPathfinding = _path_mod.OSRSPathfinding
NpcCollisionResolver = _path_mod.NpcCollisionResolver

EntityTypes = _entity_mod.EntityTypes
InfernoEntityType = _entity_mod.InfernoEntityType


# ---- Type mapping from trace strings to sim EntityTypes ----

TYPE_MAP = {
    "NIBBLER": EntityTypes.NIBBLER,
    "BAT": EntityTypes.BAT,
    "BLOB": EntityTypes.BLOB,
    "BLOB_MAGE": EntityTypes.BLOB_MAGE,
    "BLOB_RANGE": EntityTypes.BLOB_RANGE,
    "BLOB_MELEE": EntityTypes.BLOB_MELEE,
    "MELEE": EntityTypes.MELEE,
    "RANGER": EntityTypes.RANGER,
    "MAGER": EntityTypes.MAGER,
    "JAD": EntityTypes.JAD,
    "HEALER": EntityTypes.HEALER,
}


@dataclass
class Disagreement:
    tick: int
    npc_id: int
    npc_type: str
    category: str  # "los", "can_attack", "movement", "attack_timing"
    expected: str
    actual: str
    context: str = ""


def load_trace(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_pillar_alive(tick_data: dict) -> List[bool]:
    return [p["alive"] for p in tick_data["pillars"]]


def get_entity_type(type_str: str) -> Optional[InfernoEntityType]:
    return TYPE_MAP.get(type_str)


def build_walkability_checker(pillar_alive: List[bool], entities: List[dict],
                               current_npc_id: int, current_npc_type: str):
    """Build walkability checker matching simulator logic."""

    def checker(x, y, size):
        # Bounds + pillar check
        if not SimulatorGeometry.is_valid_tile_for_size(x, y, size, pillar_alive):
            return False

        # NPC collision (skip nibblers, skip self)
        for other in entities:
            if other["npc_id"] == current_npc_id:
                continue
            other_type = other["type"]
            # Nibblers don't collide
            if other_type == "NIBBLER" or current_npc_type == "NIBBLER":
                continue
            other_size = other["size"]
            if SimulatorGeometry.do_footprints_overlap(x, y, size,
                                                        other["xs"], other["ys"], other_size):
                return False
        return True

    return checker


# ---- LOS helpers matching simulator semantics ----

def sim_pure_los_any_tile(nx, ny, size, px, py, pillar_alive):
    """
    Match the simulator's NPC-sees-target check: Bresenham LOS from ANY NPC tile
    to player, no range limit. Returns True if any tile has unblocked Bresenham ray.
    """
    for dx in range(size):
        for dy in range(size):
            tx, ty = nx + dx, ny + dy
            # Skip if tile is on a pillar
            if SimulatorGeometry.is_on_pillar(tx, ty, pillar_alive):
                continue
            if SimulatorGeometry.is_on_pillar(px, py, pillar_alive):
                return False
            if tx == px and ty == py:
                return True
            # Use sim's Bresenham with high range to simulate "no range limit"
            has = InfernoLineOfSight.has_line_of_sight(
                tx, ty, px, py, 1, 50, False, pillar_alive
            )
            if has:
                return True
    return False


def sim_can_attack_any_tile(nx, ny, size, attack_range, px, py, pillar_alive):
    """
    Match the simulator's NPC-can-attack check: iterate ALL NPC tiles, check range + LOS.
    Melee (range=1) uses orthogonal adjacency.
    """
    if attack_range == 1:
        return SimulatorGeometry.is_orthogonally_adjacent(px, py, nx, ny, size)

    for dx in range(size):
        for dy in range(size):
            tx, ty = nx + dx, ny + dy
            dist = max(abs(tx - px), abs(ty - py))
            if dist <= attack_range:
                if SimulatorGeometry.is_on_pillar(tx, ty, pillar_alive):
                    continue
                if SimulatorGeometry.is_on_pillar(px, py, pillar_alive):
                    return False
                if tx == px and ty == py:
                    continue  # Collision (on same tile)
                has = InfernoLineOfSight.has_line_of_sight(
                    tx, ty, px, py, 1, attack_range, False, pillar_alive
                )
                if has:
                    return True
    return False


# ---- LOS Benchmark ----

def benchmark_los(trace: List[dict]) -> List[Disagreement]:
    """
    Compare simulator LOS with recorded npc_los and npc_can_attack.

    Simulator semantics:
    - npc_los: Bresenham from ANY NPC tile, no range check
    - npc_can_attack: in-range tiles + LOS, melee=adjacency
    """
    disagreements = []

    for tick_data in trace:
        tick = tick_data["tick"]
        px = tick_data["player"]["xs"]
        py = tick_data["player"]["ys"]
        pillar_alive = get_pillar_alive(tick_data)

        for npc in tick_data["npcs"]:
            etype = get_entity_type(npc["type"])
            if etype is None:
                continue

            nx, ny = npc["xs"], npc["ys"]
            size = npc["size"]
            attack_range = etype.attack_range

            # --- npc_los: pure LOS from any tile ---
            sim_los = sim_pure_los_any_tile(nx, ny, size, px, py, pillar_alive)
            recorded_los = npc["npc_los"]

            if sim_los != recorded_los:
                disagreements.append(Disagreement(
                    tick=tick, npc_id=npc["npc_id"], npc_type=npc["type"],
                    category="los",
                    expected=str(recorded_los), actual=str(sim_los),
                    context=f"npc=({nx},{ny}) size={size} player=({px},{py})"
                ))

            # --- npc_can_attack: range + LOS from any in-range tile ---
            sim_can_attack = sim_can_attack_any_tile(
                nx, ny, size, attack_range, px, py, pillar_alive
            )
            recorded_can_attack = npc["npc_can_attack"]

            if sim_can_attack != recorded_can_attack:
                # Also compare against the sim's native method (closest-tile)
                sim_native = InfernoLineOfSight.npc_has_los_to_player(
                    nx, ny, size, px, py, attack_range, pillar_alive
                )
                disagreements.append(Disagreement(
                    tick=tick, npc_id=npc["npc_id"], npc_type=npc["type"],
                    category="can_attack",
                    expected=str(recorded_can_attack), actual=str(sim_can_attack),
                    context=(f"npc=({nx},{ny}) size={size} player=({px},{py}) range={attack_range} "
                             f"native_sim={sim_native}")
                ))

    return disagreements


# ---- Movement Benchmark ----

def benchmark_movement(trace: List[dict]) -> List[Disagreement]:
    """Compare simulator pathfinding with recorded NPC movement."""
    disagreements = []

    # Build tick-indexed NPC maps
    for i in range(len(trace) - 1):
        curr = trace[i]
        next_tick = trace[i + 1]

        # Skip wave transitions (NPCs despawn/spawn)
        if curr["wave"] != next_tick["wave"]:
            continue

        tick = curr["tick"]
        px = curr["player"]["xs"]
        py = curr["player"]["ys"]
        pillar_alive = get_pillar_alive(curr)

        # Map NPCs by ID for lookup
        next_npcs = {n["npc_id"]: n for n in next_tick["npcs"]}

        for npc in curr["npcs"]:
            npc_id = npc["npc_id"]
            if npc_id not in next_npcs:
                continue  # NPC died or disappeared

            etype = get_entity_type(npc["type"])
            if etype is None:
                continue

            # Skip nibblers (target pillar, not player)
            if npc["type"] == "NIBBLER":
                continue

            # Skip meleers (special dig mechanics add too much complexity)
            if npc["type"] == "MELEE":
                continue

            nx, ny = npc["xs"], npc["ys"]
            size = npc["size"]
            actual_nx = next_npcs[npc_id]["xs"]
            actual_ny = next_npcs[npc_id]["ys"]

            # If NPC didn't move, check if simulator agrees
            actual_dx = actual_nx - nx
            actual_dy = actual_ny - ny

            # Check if NPC has LOS (should stay put if yes)
            # The sim uses can_entity_attack_player (closest-tile approach)
            has_los = InfernoLineOfSight.npc_has_los_to_player(
                nx, ny, size, px, py, etype.attack_range, pillar_alive
            )

            # Simulate movement
            checker = build_walkability_checker(pillar_alive, curr["npcs"], npc_id, npc["type"])

            # Check for player-under-NPC collision resolution
            is_under = NpcCollisionResolver.is_player_under_npc(nx, ny, size, px, py)

            if is_under:
                # Collision resolution is random - can't predict exact direction
                # Just check that real game also moved (or stayed)
                if actual_dx == 0 and actual_dy == 0:
                    # NPC stayed - valid (blocked direction was chosen)
                    pass
                else:
                    # NPC moved - should be a cardinal direction
                    is_cardinal = (abs(actual_dx) + abs(actual_dy) == 1)
                    if not is_cardinal:
                        disagreements.append(Disagreement(
                            tick=tick, npc_id=npc_id, npc_type=npc["type"],
                            category="movement",
                            expected="cardinal collision resolution",
                            actual=f"moved ({actual_dx},{actual_dy})",
                            context=f"player under NPC at ({nx},{ny})"
                        ))
                continue

            if has_los:
                # NPC has LOS - should not move
                if actual_dx != 0 or actual_dy != 0:
                    disagreements.append(Disagreement(
                        tick=tick, npc_id=npc_id, npc_type=npc["type"],
                        category="movement",
                        expected=f"stay at ({nx},{ny}) (has LOS)",
                        actual=f"moved to ({actual_nx},{actual_ny})",
                        context=f"player=({px},{py})"
                    ))
                continue

            # No LOS - simulate movement
            sim_x, sim_y = OSRSPathfinding.simulate_npc_movement(
                nx, ny, px, py, size, etype.move_speed, checker
            )

            sim_dx = sim_x - nx
            sim_dy = sim_y - ny

            if (sim_dx, sim_dy) != (actual_dx, actual_dy):
                disagreements.append(Disagreement(
                    tick=tick, npc_id=npc_id, npc_type=npc["type"],
                    category="movement",
                    expected=f"move ({sim_dx},{sim_dy}) to ({sim_x},{sim_y})",
                    actual=f"move ({actual_dx},{actual_dy}) to ({actual_nx},{actual_ny})",
                    context=f"from ({nx},{ny}) target=({px},{py}) size={size} speed={etype.move_speed}"
                ))

    return disagreements


# ---- Attack Timing Benchmark ----

def benchmark_attack_timing(trace: List[dict]) -> List[Disagreement]:
    """Check if attack intervals match expected NPC attack speeds."""
    disagreements = []

    # Track per-NPC attack ticks
    npc_attacks: Dict[int, List[Tuple[int, str, int]]] = defaultdict(list)

    for tick_data in trace:
        tick = tick_data["tick"]
        for npc in tick_data["npcs"]:
            if npc["attacking"] and npc["attack_tick"] == tick:
                # This is the actual attack initiation tick
                npc_attacks[npc["npc_id"]].append(
                    (tick, npc["type"], npc["attack_count"])
                )

    for npc_id, attacks in npc_attacks.items():
        if len(attacks) < 2:
            continue

        npc_type = attacks[0][1]
        etype = get_entity_type(npc_type)
        if etype is None:
            continue

        expected_speed = etype.attack_speed

        for j in range(1, len(attacks)):
            interval = attacks[j][0] - attacks[j - 1][0]
            # Allow some slack for LOS loss between attacks
            if interval != expected_speed and interval < expected_speed:
                disagreements.append(Disagreement(
                    tick=attacks[j][0], npc_id=npc_id, npc_type=npc_type,
                    category="attack_timing",
                    expected=f"interval >= {expected_speed} (attack speed)",
                    actual=f"interval = {interval}",
                    context=f"attacks at ticks {attacks[j-1][0]} and {attacks[j][0]}"
                ))

    return disagreements


# ---- Blob Split Analysis ----

def analyze_blob_splits(trace: List[dict]) -> List[str]:
    """Analyze blob death -> split spawn patterns."""
    findings = []

    # Track blob disappearances and sub-blob appearances
    prev_npcs = {}
    for tick_data in trace:
        tick = tick_data["tick"]
        curr_npcs = {n["npc_id"]: n for n in tick_data["npcs"]}

        # Check for new NPCs (potential blob spawns)
        new_ids = set(curr_npcs.keys()) - set(prev_npcs.keys())
        gone_ids = set(prev_npcs.keys()) - set(curr_npcs.keys())

        new_blob_subs = [curr_npcs[nid] for nid in new_ids
                         if curr_npcs[nid]["type"] in ("BLOB_MAGE", "BLOB_RANGE", "BLOB_MELEE")]
        gone_blobs = [prev_npcs[gid] for gid in gone_ids
                      if prev_npcs[gid]["type"] == "BLOB"]

        if new_blob_subs and gone_blobs:
            blob = gone_blobs[0]
            findings.append(
                f"Tick {tick}: BLOB at ({blob['xs']},{blob['ys']}) died -> "
                f"{len(new_blob_subs)} sub-blobs spawned at "
                f"{[(s['type'], s['xs'], s['ys']) for s in new_blob_subs]}"
            )

        prev_npcs = curr_npcs

    return findings


# ---- Attack Interval Summary ----

def summarize_attack_intervals(trace: List[dict]) -> Dict[str, List[int]]:
    """Extract per-type attack interval distributions."""
    npc_attacks: Dict[int, List[Tuple[int, str]]] = defaultdict(list)

    for tick_data in trace:
        tick = tick_data["tick"]
        for npc in tick_data["npcs"]:
            if npc["attacking"] and npc["attack_tick"] == tick:
                npc_attacks[npc["npc_id"]].append((tick, npc["type"]))

    intervals_by_type: Dict[str, List[int]] = defaultdict(list)
    for npc_id, attacks in npc_attacks.items():
        if len(attacks) < 2:
            continue
        npc_type = attacks[0][1]
        for j in range(1, len(attacks)):
            interval = attacks[j][0] - attacks[j - 1][0]
            intervals_by_type[npc_type].append(interval)

    return intervals_by_type


# ---- Main ----

def main():
    trace_dir = os.path.expanduser("~/inferno-traces")
    files = sorted(f for f in os.listdir(trace_dir) if f.endswith(".jsonl"))
    if not files:
        print("No trace files found")
        return

    trace_path = os.path.join(trace_dir, files[-1])  # Latest trace
    print(f"Loading: {trace_path}")
    trace = load_trace(trace_path)
    print(f"Loaded {len(trace)} ticks, waves {set(t['wave'] for t in trace)}")
    print()

    # ---- LOS ----
    print("=" * 60)
    print("LOS BENCHMARK")
    print("=" * 60)
    los_issues = benchmark_los(trace)
    los_only = [d for d in los_issues if d.category == "los"]
    can_atk_only = [d for d in los_issues if d.category == "can_attack"]

    total_los_checks = sum(len(t["npcs"]) for t in trace)
    print(f"Total NPC-tick LOS checks: {total_los_checks}")
    print(f"LOS disagreements: {len(los_only)}")
    print(f"can_attack disagreements: {len(can_atk_only)}")

    # Break down by direction: sim says True but game False vs sim False but game True
    los_sim_true = [d for d in los_only if d.actual == "True"]
    los_sim_false = [d for d in los_only if d.actual == "False"]
    print(f"  sim=True game=False (sim too permissive): {len(los_sim_true)}")
    print(f"  sim=False game=True (sim too restrictive): {len(los_sim_false)}")

    # Break down by NPC type
    los_by_type = defaultdict(int)
    for d in los_only:
        los_by_type[d.npc_type] += 1
    if los_by_type:
        print(f"  By type: {dict(sorted(los_by_type.items()))}")

    if los_only:
        print("\nFirst 10 LOS disagreements:")
        for d in los_only[:10]:
            print(f"  tick {d.tick}: {d.npc_type} sim={d.actual} game={d.expected} | {d.context}")

    can_atk_sim_true = [d for d in can_atk_only if d.actual == "True"]
    can_atk_sim_false = [d for d in can_atk_only if d.actual == "False"]
    print(f"\ncan_attack breakdown:")
    print(f"  sim=True game=False (sim too permissive): {len(can_atk_sim_true)}")
    print(f"  sim=False game=True (sim too restrictive): {len(can_atk_sim_false)}")

    if can_atk_only:
        print("\nAll can_attack disagreements:")
        for d in can_atk_only:
            print(f"  tick {d.tick}: {d.npc_type} sim={d.actual} game={d.expected} | {d.context}")

    # ---- Movement ----
    print()
    print("=" * 60)
    print("MOVEMENT BENCHMARK")
    print("=" * 60)
    move_issues = benchmark_movement(trace)
    total_npc_ticks = sum(len(t["npcs"]) for t in trace[:-1] if t["wave"] == trace[trace.index(t) + 1]["wave"]) if len(trace) > 1 else 0
    print(f"Movement disagreements: {len(move_issues)}")

    if move_issues:
        # Group by type
        by_type = defaultdict(list)
        for d in move_issues:
            by_type[d.npc_type].append(d)

        for ntype, issues in sorted(by_type.items()):
            print(f"\n  {ntype}: {len(issues)} disagreements")
            for d in issues[:5]:
                print(f"    tick {d.tick}: {d.expected} vs {d.actual} | {d.context}")

    # ---- Attack Timing ----
    print()
    print("=" * 60)
    print("ATTACK TIMING")
    print("=" * 60)
    intervals = summarize_attack_intervals(trace)
    for npc_type, ivs in sorted(intervals.items()):
        etype = get_entity_type(npc_type)
        expected = etype.attack_speed if etype else "?"
        print(f"  {npc_type}: intervals={ivs} (expected speed={expected})")

    timing_issues = benchmark_attack_timing(trace)
    if timing_issues:
        print(f"\n  Attack timing violations (interval < speed): {len(timing_issues)}")
        for d in timing_issues[:10]:
            print(f"    tick {d.tick}: {d.npc_type} {d.actual} | {d.context}")
    else:
        print("\n  No attack timing violations (all intervals >= attack speed)")

    # ---- Blob Splits ----
    print()
    print("=" * 60)
    print("BLOB SPLITS")
    print("=" * 60)
    splits = analyze_blob_splits(trace)
    if splits:
        for s in splits:
            print(f"  {s}")
    else:
        print("  No blob splits observed")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_issues = len(los_only) + len(can_atk_only) + len(move_issues) + len(timing_issues)
    print(f"Total disagreements: {total_issues}")
    print(f"  LOS: {len(los_only)}")
    print(f"  can_attack: {len(can_atk_only)}")
    print(f"  Movement: {len(move_issues)}")
    print(f"  Attack timing: {len(timing_issues)}")

    accuracy = 1 - (total_issues / max(total_los_checks * 2 + len(trace), 1))
    print(f"  Overall agreement: {accuracy:.1%}")


if __name__ == "__main__":
    main()
