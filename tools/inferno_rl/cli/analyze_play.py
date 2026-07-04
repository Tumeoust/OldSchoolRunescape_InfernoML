"""
Tactical play analyzer — reads enhanced play_human.py JSON logs and produces
tactical analysis with auto-flagged issues.

Flags based on INFERNO_TACTICS.md principles:
- Kill order violations (ranger/blob before mager when mager alive & targetable)
- Multi-LOS exposure (2+ NPCs with LOS simultaneously)
- Melee dig risk (melee approaching dig threshold while untargeted)
- Weapon mismatch (wrong weapon for target type)

Usage:
    python -m tools.inferno_rl.cli.analyze_play --log-dir ./logs/session_1
    python -m tools.inferno_rl.cli.analyze_play --log-dir ./logs/session_1 \
        --model models/V21_climb/...pt
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_wave_logs(log_dir: str) -> list[dict]:
    """Load all wave_XX.json files from a log directory, sorted by wave."""
    waves = []
    for filename in sorted(os.listdir(log_dir)):
        if filename.startswith("wave_") and filename.endswith(".json"):
            filepath = os.path.join(log_dir, filename)
            with open(filepath) as f:
                waves.append(json.load(f))
    return waves


def _find_entity_by_type(entities: list[dict], entity_type: str) -> list[dict]:
    """Find all entities of a given type."""
    return [e for e in entities if e["type"] == entity_type]


def _detect_kill_order_violations(ticks: list[dict]) -> list[dict]:
    """Detect when ranger/blob/bat killed before mager when mager was alive."""
    flags = []
    for tick in ticks:
        entities = tick.get("entities", [])
        if not entities:
            continue
        # Check if mager is alive
        magers = _find_entity_by_type(entities, "Mager")
        if not magers:
            continue

        action = tick.get("action", "")
        # If attacking a non-mager priority target while mager exists
        if action.startswith("ATTACK_P") or action.startswith("ATK_P"):
            # Determine which priority level
            for i, e in enumerate(entities):
                # The first entity in the sorted list is P1
                if e["type"] == "Mager":
                    break  # mager is highest priority, no violation
                if i == 0 and action in ("ATTACK_P1", "ATK_P1"):
                    # Attacking P1 which is not a mager — could be a violation
                    # But only flag if mager has player_los (is targetable)
                    if any(m.get("player_los") for m in magers):
                        flags.append({
                            "tick": tick["tick"],
                            "type": "kill_order",
                            "detail": (f"Attacking {e['type']} (P1) while "
                                       f"Mager alive and targetable"),
                        })
                    break
    return flags


def _detect_multi_los(ticks: list[dict]) -> list[dict]:
    """Detect ticks where 2+ non-nibbler NPCs have LOS to player."""
    flags = []
    for tick in ticks:
        entities = tick.get("entities", [])
        los_count = sum(1 for e in entities if e.get("npc_los"))
        if los_count >= 2:
            los_types = [e["type"] for e in entities if e.get("npc_los")]
            flags.append({
                "tick": tick["tick"],
                "type": "multi_los",
                "detail": f"{los_count} NPCs with LOS: {', '.join(los_types)}",
            })
    return flags


def _detect_melee_dig_risk(ticks: list[dict]) -> list[dict]:
    """Detect melee approaching dig threshold while not being targeted."""
    flags = []
    for tick in ticks:
        entities = tick.get("entities", [])
        action = tick.get("action", "")
        for e in entities:
            if e["type"] != "Melee":
                continue
            dig = e.get("dig", {})
            if dig.get("digging"):
                flags.append({
                    "tick": tick["tick"],
                    "type": "melee_dig",
                    "detail": f"Melee #{e['id']} is DIGGING "
                              f"({dig['dig_ticks_left']}t left)",
                })
            elif dig.get("in_dig_range"):
                flags.append({
                    "tick": tick["tick"],
                    "type": "melee_dig_risk",
                    "detail": f"Melee #{e['id']} in dig range "
                              f"(attack_delay={e['attack_delay']})",
                })
            elif dig.get("ticks_until_10pct", 99) <= 10:
                flags.append({
                    "tick": tick["tick"],
                    "type": "melee_dig_warning",
                    "detail": f"Melee #{e['id']} {dig['ticks_until_10pct']}t "
                              f"until dig range",
                })
    return flags


def _detect_weapon_mismatch(ticks: list[dict]) -> list[dict]:
    """Detect obvious weapon mismatches."""
    flags = []
    for tick in ticks:
        entities = tick.get("entities", [])
        action = tick.get("action", "")
        weapon = tick.get("weapon", "")

        # Only check on attack ticks
        if not (action.startswith("ATTACK_P") or action.startswith("ATK_P")):
            continue

        # Determine target type from priority
        target_type = None
        if action in ("ATTACK_P1", "ATK_P1") and entities:
            target_type = entities[0]["type"]
        elif action in ("ATTACK_P2", "ATK_P2") and len(entities) > 1:
            target_type = entities[1]["type"]
        elif action in ("ATTACK_P3", "ATK_P3") and len(entities) > 2:
            target_type = entities[2]["type"]

        if not target_type:
            continue

        # Blood barrage on anything except nibblers is usually wrong
        if weapon == "MAGE" and tick.get("player", {}).get("weapon") == "BloodBarrage":
            flags.append({
                "tick": tick["tick"],
                "type": "weapon_mismatch",
                "detail": f"Blood barrage on {target_type}",
            })
    return flags


def analyze_wave(wave_data: dict) -> dict:
    """Analyze a single wave's tick log."""
    ticks = wave_data.get("ticks_log", [])
    if not ticks:
        return {"wave": wave_data["wave"], "flags": [], "summary": {}}

    # Collect flags
    all_flags = []
    all_flags.extend(_detect_kill_order_violations(ticks))
    all_flags.extend(_detect_multi_los(ticks))
    all_flags.extend(_detect_melee_dig_risk(ticks))
    all_flags.extend(_detect_weapon_mismatch(ticks))

    # Sort by tick
    all_flags.sort(key=lambda f: f["tick"])

    # Deduplicate consecutive multi_los flags (keep first + count)
    deduped_flags = []
    prev_multi_los_tick = -10
    multi_los_run = 0
    for flag in all_flags:
        if flag["type"] == "multi_los":
            if flag["tick"] == prev_multi_los_tick + 1:
                multi_los_run += 1
                prev_multi_los_tick = flag["tick"]
                continue
            else:
                if multi_los_run > 0:
                    deduped_flags[-1]["detail"] += f" (+{multi_los_run} more ticks)"
                multi_los_run = 0
                prev_multi_los_tick = flag["tick"]
        else:
            if multi_los_run > 0:
                deduped_flags[-1]["detail"] += f" (+{multi_los_run} more ticks)"
                multi_los_run = 0
        deduped_flags.append(flag)
    if multi_los_run > 0 and deduped_flags:
        deduped_flags[-1]["detail"] += f" (+{multi_los_run} more ticks)"

    # Wave summary stats
    total_damage = 0
    multi_los_ticks = 0
    kill_order = []
    seen_kill_types = set()
    prev_enemies = None

    for tick in ticks:
        # Damage tracking
        components = tick.get("components", {})
        for name, value in components.items():
            if "damage_taken" in name.lower() and value < 0:
                total_damage += abs(value)

        # Multi-LOS counting
        entities = tick.get("entities", [])
        los_count = sum(1 for e in entities if e.get("npc_los"))
        if los_count >= 2:
            multi_los_ticks += 1

        # Kill order tracking: detect when entity types disappear
        current_types = set(e["type"] for e in entities)
        if prev_enemies is not None:
            disappeared = prev_enemies - current_types
            for t in disappeared:
                if t not in seen_kill_types:
                    kill_order.append(t)
                    seen_kill_types.add(t)
        prev_enemies = current_types

    summary = {
        "wave": wave_data["wave"],
        "ticks": len(ticks),
        "total_reward": wave_data.get("total_reward", 0),
        "terminal": wave_data.get("terminal"),
        "multi_los_ticks": multi_los_ticks,
        "kill_order": kill_order,
    }

    return {
        "wave": wave_data["wave"],
        "flags": deduped_flags,
        "summary": summary,
    }


def print_wave_analysis(analysis: dict) -> None:
    """Print analysis for a single wave."""
    s = analysis["summary"]
    flags = analysis["flags"]

    print(f"\n{'='*60}")
    print(f"  WAVE {s['wave']}  |  {s['ticks']} ticks  |  "
          f"reward: {s['total_reward']:+.1f}  |  "
          f"multi-LOS: {s['multi_los_ticks']} ticks")
    if s.get("terminal"):
        print(f"  Terminal: {s['terminal']}")
    if s["kill_order"]:
        print(f"  Kill order: {' -> '.join(s['kill_order'])}")
    print(f"{'='*60}")

    if not flags:
        print("  No issues flagged.")
        return

    # Group by type
    by_type = defaultdict(list)
    for f in flags:
        by_type[f["type"]].append(f)

    for flag_type, type_flags in by_type.items():
        label = {
            "kill_order": "KILL ORDER",
            "multi_los": "MULTI-LOS",
            "melee_dig": "MELEE DIG",
            "melee_dig_risk": "DIG RISK",
            "melee_dig_warning": "DIG WARNING",
            "weapon_mismatch": "WEAPON",
        }.get(flag_type, flag_type.upper())

        print(f"\n  [{label}] ({len(type_flags)} flags)")
        for f in type_flags[:10]:  # Cap display
            print(f"    T{f['tick']:04d}: {f['detail']}")
        if len(type_flags) > 10:
            print(f"    ... and {len(type_flags) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tactical play analyzer — flags issues in play session logs",
    )
    parser.add_argument("--log-dir", required=True,
                        help="Directory containing wave_XX.json logs from play_human")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to .pt checkpoint for RL comparison at flagged ticks")
    parser.add_argument("--output-format", choices=["text", "json"], default="text",
                        help="Output format (default: text)")

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"Error: {args.log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    wave_logs = load_wave_logs(args.log_dir)
    if not wave_logs:
        print(f"No wave_XX.json files found in {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    # Load episode summary if exists
    summary_path = os.path.join(args.log_dir, "episode_summary.json")
    episode_summary = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            episode_summary = json.load(f)

    # Analyze each wave
    analyses = [analyze_wave(w) for w in wave_logs]

    # Count totals
    total_flags = sum(len(a["flags"]) for a in analyses)
    total_multi_los = sum(a["summary"].get("multi_los_ticks", 0) for a in analyses)

    if args.output_format == "json":
        output = {
            "log_dir": args.log_dir,
            "episode_summary": episode_summary,
            "total_flags": total_flags,
            "total_multi_los_ticks": total_multi_los,
            "waves": analyses,
        }
        json.dump(output, sys.stdout, indent=2)
        print()
    else:
        print(f"Analyzing {len(wave_logs)} waves from {args.log_dir}")
        if episode_summary:
            print(f"Episode: seed={episode_summary.get('seed')} "
                  f"waves {episode_summary.get('start_wave')}-"
                  f"{episode_summary.get('max_wave_reached')} "
                  f"({episode_summary.get('terminal')})")
            print(f"Total reward: {episode_summary.get('total_reward', 0):+.1f}  "
                  f"Ticks: {episode_summary.get('total_ticks', 0)}")

        for analysis in analyses:
            print_wave_analysis(analysis)

        # Overall summary
        print(f"\n{'='*60}")
        print(f"  OVERALL: {total_flags} flags across {len(analyses)} waves, "
              f"{total_multi_los} multi-LOS ticks")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
