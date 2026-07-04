"""
Shared state decoding helpers for CLI diagnostic tools.

Decodes raw SimulatorState into rich dicts suitable for JSON serialization
or text formatting. Used by snapshot_wave.py, analyze_play.py, and
play_human.py logging.
"""

from typing import Optional

from ..simulator.entity import PlacedEntity, EntityTypes
from ..simulator.exact_targeting import get_exact_target_by_slot, get_exact_target_entities
from ..simulator.geometry import InfernoLineOfSight, PILLARS
from ..simulator.state import SimulatorState, PILLAR_MAX_HP
from ..testing.actions import InfernoAction

PILLAR_NAMES = ["NW", "NE", "S"]
PILLAR_NIBBLER_TARGETS = {0: "NW", 1: "NE", 2: "S"}

ENTITY_TYPE_DISPLAY = {
    EntityTypes.NIBBLER: "Nibbler",
    EntityTypes.BAT: "Bat",
    EntityTypes.BLOB: "Blob",
    EntityTypes.BLOB_MAGE: "BlobMage",
    EntityTypes.BLOB_RANGE: "BlobRange",
    EntityTypes.BLOB_MELEE: "BlobMelee",
    EntityTypes.RANGER: "Ranger",
    EntityTypes.MAGER: "Mager",
    EntityTypes.MELEE: "Melee",
    EntityTypes.JAD: "Jad",
    EntityTypes.HEALER: "Healer",
    EntityTypes.ZUK_HEALER: "ZukHealer",
    EntityTypes.ZUK: "Zuk",
}

def _entity_type_name(entity: PlacedEntity) -> str:
    return ENTITY_TYPE_DISPLAY.get(entity.entity_type, entity.entity_type.name)


def _weapon_name(state: SimulatorState) -> str:
    from ..simulator.equipment import GearPreset
    if state.current_preset == GearPreset.MAGE:
        return "BloodBarrage" if state.use_blood_barrage else "IceBarrage"
    if state.current_preset == GearPreset.BOFA:
        return "BoFa"
    if state.current_preset == GearPreset.BLOWPIPE:
        return "Blowpipe"
    return state.current_preset.value


def _melee_dig_info(entity: PlacedEntity) -> dict:
    """Compute melee dig timer info from attack_delay."""
    if entity.entity_type != EntityTypes.MELEE:
        return {}
    if entity.dig_sequence_time > 0:
        return {"digging": True, "dig_ticks_left": entity.dig_sequence_time}
    # Dig triggers when attack_delay goes very negative while no LOS
    # -38: 10% chance per tick, -50: guaranteed
    ticks_to_10pct = max(0, -entity.attack_delay - 38) if entity.attack_delay < -38 else 0
    ticks_to_guaranteed = max(0, -entity.attack_delay - 50) if entity.attack_delay < -50 else 0
    ticks_until_10pct = max(0, 38 + entity.attack_delay) if entity.attack_delay > -38 else 0
    ticks_until_guaranteed = max(0, 50 + entity.attack_delay) if entity.attack_delay > -50 else 0
    return {
        "digging": False,
        "in_dig_range": entity.attack_delay <= -38,
        "ticks_until_10pct": ticks_until_10pct,
        "ticks_until_guaranteed": ticks_until_guaranteed,
    }


def _blob_scan_info(entity: PlacedEntity) -> dict:
    """Decode blob scan state."""
    if entity.entity_type != EntityTypes.BLOB:
        return {}
    return {
        "scanned_prayer": entity.scanned_prayer,
        "had_los": entity.had_los,
    }


def decode_entity(entity: PlacedEntity, state: SimulatorState) -> dict:
    """Decode a single entity into a rich dict from raw state."""
    npc_los = InfernoLineOfSight.can_entity_attack_player(
        entity, state.player_x, state.player_y, state.pillar_alive
    )
    player_los = InfernoLineOfSight.can_player_attack_entity(
        state.player_x, state.player_y, state.player_attack_range,
        entity, state.pillar_alive
    )
    dist = InfernoLineOfSight.get_distance_from_npc(
        entity.x, entity.y, entity.entity_type.size_in_tiles,
        state.player_x, state.player_y
    )

    result = {
        "id": entity.id,
        "type": _entity_type_name(entity),
        "x": entity.x,
        "y": entity.y,
        "hp": entity.current_health,
        "hp_max": entity.entity_type.max_health,
        "hp_pct": round(entity.current_health / entity.entity_type.max_health * 100),
        "attack_delay": entity.attack_delay,
        "stunned": entity.stunned,
        "frozen": entity.frozen,
        "npc_los": npc_los,
        "player_los": player_los,
        "distance": dist,
    }

    # Melee-specific
    dig_info = _melee_dig_info(entity)
    if dig_info:
        result["dig"] = dig_info

    # Blob-specific
    blob_info = _blob_scan_info(entity)
    if blob_info:
        result["blob"] = blob_info

    return result


def decode_entities_from_state(state: SimulatorState) -> list[dict]:
    """Decode all alive entities in the exact-target ordering."""
    sorted_ents = get_exact_target_entities(state)
    return [decode_entity(e, state) for e in sorted_ents]


def decode_pillars(state: SimulatorState) -> dict:
    """Decode pillar HP/alive status."""
    pillars = {}
    for i, name in enumerate(PILLAR_NAMES):
        pillars[name] = {
            "alive": state.pillar_alive[i],
            "hp": state.pillar_hp[i],
            "hp_max": PILLAR_MAX_HP,
        }
    return pillars


def decode_nibblers(state: SimulatorState) -> dict:
    """Decode nibbler count and target pillar."""
    nibblers = [e for e in state.entities if not e.is_dead() and e.entity_type == EntityTypes.NIBBLER]
    target_pillar = None
    if nibblers:
        idx = nibblers[0].target_pillar_index
        target_pillar = PILLAR_NIBBLER_TARGETS.get(idx, f"unknown({idx})")
    return {
        "alive": len(nibblers),
        "target_pillar": target_pillar,
    }


def decode_player(state: SimulatorState) -> dict:
    """Decode player state."""
    return {
        "x": state.player_x,
        "y": state.player_y,
        "hp": state.player_health,
        "weapon": _weapon_name(state),
        "attack_cooldown": state.get_player_attack_cooldown(),
        "prayer": state.active_prayer,
    }


def resolve_attack_target(action: int, state: SimulatorState) -> Optional[str]:
    """Resolve what entity an ATTACK_TARGET_N action targets."""
    target_index = InfernoAction.get_target_index(action)
    if target_index < 0:
        return None
    entity = get_exact_target_by_slot(state, target_index)
    if entity is not None:
        return f"{_entity_type_name(entity)} #{entity.id}"
    return None


def decode_full_tick(state: SimulatorState, tick: int, wave: int,
                     tick_in_wave: int, action: Optional[int] = None,
                     action_name: Optional[str] = None,
                     rl_value: Optional[float] = None,
                     rl_top_actions: Optional[list] = None) -> dict:
    """Build a complete tick snapshot dict."""
    data = {
        "tick": tick,
        "wave": wave,
        "tick_in_wave": tick_in_wave,
        "player": decode_player(state),
        "pillars": decode_pillars(state),
        "nibblers": decode_nibblers(state),
        "entities": decode_entities_from_state(state),
    }
    if action is not None:
        data["action"] = action_name or str(action)
        target = resolve_attack_target(action, state)
        if target:
            data["action_target"] = target
    if rl_value is not None:
        data["rl_value"] = round(rl_value, 3)
    if rl_top_actions is not None:
        data["rl_top_actions"] = rl_top_actions
    return data


def format_tick_text(tick_data: dict) -> str:
    """Format a tick snapshot dict into human-readable text."""
    lines = []
    p = tick_data["player"]
    pillars = tick_data["pillars"]

    # Header line
    pillar_str = " ".join(
        f"{name}:{info['hp']}" if info['alive'] else f"{name}:DEAD"
        for name, info in pillars.items()
    )
    header = (f"T{tick_data['tick']:03d} W{tick_data['wave']:02d} | "
              f"Player ({p['x']},{p['y']}) HP:{p['hp']} "
              f"{p['weapon']} CD:{p['attack_cooldown']} | "
              f"Pillars: {pillar_str}")
    lines.append(header)

    # Entities
    entities = tick_data["entities"]
    nibblers = tick_data["nibblers"]
    alive_count = len(entities) + nibblers["alive"]
    los_count = sum(1 for e in entities if e["npc_los"])
    lines.append(f"  Entities ({alive_count} alive, {los_count} with LOS):")

    for i, e in enumerate(entities):
        los_str = "YES" if e["npc_los"] else "--"
        plos_str = "yes" if e["player_los"] else "no"
        frozen_str = f" frz:{e['frozen']}" if e["frozen"] > 0 else ""
        stunned_str = f" stn:{e['stunned']}" if e["stunned"] > 0 else ""

        extra = ""
        if "dig" in e:
            dig = e["dig"]
            if dig.get("digging"):
                extra += f"  DIGGING({dig['dig_ticks_left']}t)"
            elif dig.get("in_dig_range"):
                extra += f"  dig:ACTIVE"
            else:
                extra += f"  dig:{dig['ticks_until_10pct']}t"
        if "blob" in e:
            blob = e["blob"]
            scan = blob["scanned_prayer"] or "none"
            extra += f"  scan:{scan}"

        line = (f"    #{i} {e['type']:>8s} ({e['x']:2d},{e['y']:2d}) "
                f"HP:{e['hp_pct']:3d}% CD:{e['attack_delay']:2d} "
                f"los:{los_str:>3s} plos:{plos_str:>3s} dist:{e['distance']:2d}"
                f"{frozen_str}{stunned_str}{extra}")
        lines.append(line)

    if nibblers["alive"] > 0:
        lines.append(f"  Nibblers: {nibblers['alive']} alive, "
                     f"targeting {nibblers['target_pillar']} pillar")

    # Action
    if "action" in tick_data:
        action_line = f"  Action: {tick_data['action']}"
        if "action_target" in tick_data:
            action_line += f" (-> {tick_data['action_target']})"
        lines.append(action_line)

    # RL info
    if "rl_value" in tick_data:
        rl_line = f"  [RL] Value: {tick_data['rl_value']:.1f}"
        if "rl_top_actions" in tick_data:
            top = tick_data["rl_top_actions"][:3]
            top_str = " ".join(f"{a['name']}({a['prob']:.2f})" for a in top)
            rl_line += f"  Top: {top_str}"
        lines.append(rl_line)

    return "\n".join(lines)
