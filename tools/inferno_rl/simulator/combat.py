"""
OSRS combat formulas as pure functions + pre-computed lookup tables.

All (preset x NPC) combinations computed once at import time.
Runtime cost per attack: 1 dict lookup + 2 random calls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import floor
from typing import Dict, Tuple

from .entity import InfernoEntityType, EntityTypes
from .equipment import (
    GearPreset, AggregateStats, PRESET_STATS,
    Loadout, LoadoutId, LOADOUTS,
)
from .npc_stats import NpcCombatStats, NPC_STATS


# ============================================================
# Constants
# ============================================================

PLAYER_RANGED_LEVEL = 99
PLAYER_MAGIC_LEVEL = 99
PLAYER_DEFENCE_LEVEL = 99

RIGOUR_ATTACK_MULT = 1.20       # +20% ranged attack
RIGOUR_STRENGTH_MULT = 1.23     # +23% ranged strength
RIGOUR_DEFENCE_MULT = 1.25      # +25% defence

AUGURY_MAGIC_ATTACK_MULT = 1.25   # +25% magic attack
AUGURY_MAGIC_DAMAGE = 0.04        # +4% magic damage (additive)
AUGURY_DEFENCE_MULT = 1.25        # +25% defence (both regular and magic)

CRYSTAL_SET_DAMAGE_MULT = 1.15    # +15% damage
CRYSTAL_SET_ACCURACY_MULT = 1.30  # +30% accuracy


# ============================================================
# Player -> NPC formulas
# ============================================================

def ranged_max_hit(
    equip_ranged_str: int,
    has_crystal_set: bool,
    *,
    ranged_level: int = 99,
    prayer_str_mult: float = RIGOUR_STRENGTH_MULT,
) -> int:
    # OSRS: effective = floor(level * prayer) + stance + invisible_boost
    effective_str = floor(ranged_level * prayer_str_mult) + 8
    base = floor(0.5 + effective_str * (equip_ranged_str + 64) / 640)
    if has_crystal_set:
        base = floor(base * CRYSTAL_SET_DAMAGE_MULT)
    return base


def magic_max_hit(
    base_spell_hit: int,
    equip_magic_dmg_pct: float,
    *,
    prayer_dmg_bonus: float = AUGURY_MAGIC_DAMAGE,
) -> int:
    return floor(base_spell_hit * (1.0 + equip_magic_dmg_pct + prayer_dmg_bonus))


def player_ranged_attack_roll(
    equip_ranged_atk: int,
    has_crystal_set: bool,
    *,
    ranged_level: int = 99,
    prayer_atk_mult: float = RIGOUR_ATTACK_MULT,
) -> int:
    # OSRS: effective = floor(level * prayer) + stance + invisible_boost
    effective = floor(ranged_level * prayer_atk_mult) + 8
    roll = effective * (equip_ranged_atk + 64)
    if has_crystal_set:
        roll = floor(roll * CRYSTAL_SET_ACCURACY_MULT)
    return roll


def player_magic_attack_roll(
    equip_magic_atk: int,
    *,
    magic_level: int = 99,
    prayer_atk_mult: float = AUGURY_MAGIC_ATTACK_MULT,
) -> int:
    # OSRS: effective = floor(level * prayer) + stance + invisible_boost
    effective = floor(magic_level * prayer_atk_mult) + 8
    return effective * (equip_magic_atk + 64)


# ============================================================
# NPC defence rolls (player attacking NPC)
# ============================================================

def npc_def_roll_vs_ranged(npc: NpcCombatStats) -> int:
    return (npc.defence_level + 9) * (npc.ranged_defence + 64)


def npc_def_roll_vs_magic(npc: NpcCombatStats) -> int:
    # NPCs use magic_level directly for magic defence, not the player-style combined formula
    return (npc.magic_level + 9) * (npc.magic_defence + 64)


# ============================================================
# NPC -> Player attack rolls
# ============================================================

def npc_melee_attack_roll(npc: NpcCombatStats) -> int:
    return (npc.attack_level + 9) * (npc.melee_attack + 64)


def npc_ranged_attack_roll(npc: NpcCombatStats) -> int:
    return (npc.ranged_level + 9) * (npc.ranged_attack + 64)


def npc_magic_attack_roll(npc: NpcCombatStats) -> int:
    return (npc.magic_level + 9) * (npc.magic_attack + 64)


# ============================================================
# Player defence rolls (NPC attacking player)
# Prayer inferred from preset: BOFA/BLOWPIPE -> Rigour, MAGE -> Augury
# ============================================================

def player_def_roll_vs_melee(
    equip_slash_def: int,
    is_augury: bool,
    *,
    defence_level: int = 99,
) -> int:
    # Both Rigour and Augury give +25% defence
    effective = floor(defence_level * 1.25) + 9
    return effective * (equip_slash_def + 64)


def player_def_roll_vs_ranged(
    equip_ranged_def: int,
    is_augury: bool,
    *,
    defence_level: int = 99,
) -> int:
    effective = floor(defence_level * 1.25) + 9
    return effective * (equip_ranged_def + 64)


def player_def_roll_vs_magic(
    equip_magic_def: int,
    is_augury: bool,
    *,
    magic_level: int = 99,
    defence_level: int = 99,
) -> int:
    if is_augury:
        # Augury boosts both magic level and defence level in the magic def formula
        eff = floor(0.7 * floor(magic_level * AUGURY_DEFENCE_MULT)
                     + 0.3 * floor(defence_level * AUGURY_DEFENCE_MULT)) + 9
    else:
        # Rigour: +25% defence level but NOT magic level
        eff = floor(0.7 * magic_level
                     + 0.3 * floor(defence_level * RIGOUR_DEFENCE_MULT)) + 9
    return eff * (equip_magic_def + 64)


# ============================================================
# Hit chance (standard OSRS formula)
# ============================================================

def hit_chance(atk_roll: int, def_roll: int) -> float:
    if atk_roll >= def_roll:
        return 1.0 - (def_roll + 2) / (2 * (atk_roll + 1))
    return atk_roll / (2 * (def_roll + 1))


# ============================================================
# Twisted bow passive
# ============================================================

def tbow_accuracy_modifier(target_magic_level: int) -> float:
    """Twisted bow accuracy scaling. Returns multiplier (e.g. 1.40 = +40%)."""
    m = min(max(target_magic_level, 1), 250)
    modifier = 140 + (3 * m - 10) // 100 - ((3 * m // 10) - 100) ** 2 // 100
    return max(0, min(140, modifier)) / 100.0


def tbow_damage_modifier(target_magic_level: int) -> float:
    """Twisted bow damage scaling. Returns multiplier (capped at 2.50)."""
    m = min(max(target_magic_level, 1), 250)
    modifier = 250 + (3 * m - 14) // 100 - ((3 * m // 10) - 140) ** 2 // 100
    return max(0, min(250, modifier)) / 100.0


# ============================================================
# Pre-computed lookup tables
# ============================================================

@dataclass(frozen=True)
class PrecomputedPlayerAttack:
    """Pre-computed player attack stats against a specific NPC type."""
    accuracy: float
    max_hit: int


@dataclass(frozen=True)
class PrecomputedNpcAttack:
    """Pre-computed NPC attack stats against a player in a specific gear preset."""
    melee_accuracy: float
    melee_max_hit: int
    ranged_accuracy: float
    ranged_max_hit: int
    magic_accuracy: float
    magic_max_hit: int


@dataclass(frozen=True)
class CombatTables:
    """Bundled player + NPC attack lookup tables for a specific loadout."""
    player_attack: dict[tuple[GearPreset, InfernoEntityType], PrecomputedPlayerAttack]
    npc_attack: dict[tuple[InfernoEntityType, GearPreset], PrecomputedNpcAttack]


def _build_player_attack_table() -> dict[tuple[GearPreset, InfernoEntityType], PrecomputedPlayerAttack]:
    """Build player -> NPC attack lookup table."""
    table: dict[tuple[GearPreset, InfernoEntityType], PrecomputedPlayerAttack] = {}

    for preset in GearPreset:
        stats = PRESET_STATS[preset]

        for npc_type, npc_stats in NPC_STATS.items():
            if preset in (GearPreset.BOFA, GearPreset.BLOWPIPE):
                # Ranged attack
                atk = player_ranged_attack_roll(stats.ranged_attack, stats.has_crystal_set)
                dfn = npc_def_roll_vs_ranged(npc_stats)
                acc = hit_chance(atk, dfn)
                mh = ranged_max_hit(stats.ranged_strength, stats.has_crystal_set)
            else:
                # Magic attack (MAGE preset)
                atk = player_magic_attack_roll(stats.magic_attack)
                dfn = npc_def_roll_vs_magic(npc_stats)
                acc = hit_chance(atk, dfn)
                mh = magic_max_hit(stats.base_spell_max_hit, stats.magic_damage_percent)

            table[(preset, npc_type)] = PrecomputedPlayerAttack(accuracy=acc, max_hit=mh)

    return table


def _build_npc_attack_table() -> dict[tuple[InfernoEntityType, GearPreset], PrecomputedNpcAttack]:
    """Build NPC -> player attack lookup table."""
    table: dict[tuple[InfernoEntityType, GearPreset], PrecomputedNpcAttack] = {}

    for preset in GearPreset:
        stats = PRESET_STATS[preset]
        is_augury = (preset == GearPreset.MAGE)

        p_def_melee = player_def_roll_vs_melee(stats.slash_defence, is_augury)
        p_def_ranged = player_def_roll_vs_ranged(stats.ranged_defence, is_augury)
        p_def_magic = player_def_roll_vs_magic(stats.magic_defence, is_augury)

        for npc_type, npc_stats in NPC_STATS.items():
            n_atk_melee = npc_melee_attack_roll(npc_stats)
            n_atk_ranged = npc_ranged_attack_roll(npc_stats)
            n_atk_magic = npc_magic_attack_roll(npc_stats)

            table[(npc_type, preset)] = PrecomputedNpcAttack(
                melee_accuracy=hit_chance(n_atk_melee, p_def_melee),
                melee_max_hit=npc_stats.max_hit_melee,
                ranged_accuracy=hit_chance(n_atk_ranged, p_def_ranged),
                ranged_max_hit=npc_stats.max_hit_ranged,
                magic_accuracy=hit_chance(n_atk_magic, p_def_magic),
                magic_max_hit=npc_stats.max_hit_magic,
            )

    return table


# Pre-computed at import time (backward compat — CRYSTAL_BP / level 99 defaults)
PLAYER_ATTACK_TABLE = _build_player_attack_table()
NPC_ATTACK_TABLE = _build_npc_attack_table()


# ============================================================
# Per-loadout combat tables
# ============================================================

def build_combat_tables(loadout: Loadout) -> CombatTables:
    """Build both player and NPC attack tables for a specific loadout."""
    rl = loadout.levels.ranged
    ml = loadout.levels.magic
    dl = loadout.levels.defence
    p = loadout.prayers

    player_table: dict[tuple[GearPreset, InfernoEntityType], PrecomputedPlayerAttack] = {}
    npc_table: dict[tuple[InfernoEntityType, GearPreset], PrecomputedNpcAttack] = {}

    for preset in GearPreset:
        stats = loadout.preset_stats[preset]
        is_augury = (preset == GearPreset.MAGE)

        # --- Player -> NPC ---
        for npc_type, npc_stats in NPC_STATS.items():
            if preset in (GearPreset.BOFA, GearPreset.BLOWPIPE):
                atk = player_ranged_attack_roll(
                    stats.ranged_attack, stats.has_crystal_set,
                    ranged_level=rl, prayer_atk_mult=p.ranged_atk,
                )
                dfn = npc_def_roll_vs_ranged(npc_stats)
                mh = ranged_max_hit(
                    stats.ranged_strength, stats.has_crystal_set,
                    ranged_level=rl, prayer_str_mult=p.ranged_str,
                )

                # Tbow passive: applies when loadout has tbow and preset is BOFA
                # (BOFA slot represents the "main ranged weapon")
                if loadout.has_tbow and preset == GearPreset.BOFA:
                    atk = floor(atk * tbow_accuracy_modifier(npc_stats.magic_level))
                    mh = floor(mh * tbow_damage_modifier(npc_stats.magic_level))

                acc = hit_chance(atk, dfn)
            else:
                atk = player_magic_attack_roll(
                    stats.magic_attack, magic_level=ml, prayer_atk_mult=p.magic_atk,
                )
                dfn = npc_def_roll_vs_magic(npc_stats)
                acc = hit_chance(atk, dfn)
                mh = magic_max_hit(
                    stats.base_spell_max_hit, stats.magic_damage_percent,
                    prayer_dmg_bonus=p.magic_dmg,
                )

            player_table[(preset, npc_type)] = PrecomputedPlayerAttack(accuracy=acc, max_hit=mh)

        # --- NPC -> Player ---
        p_def_melee = player_def_roll_vs_melee(stats.slash_defence, is_augury, defence_level=dl)
        p_def_ranged = player_def_roll_vs_ranged(stats.ranged_defence, is_augury, defence_level=dl)
        p_def_magic = player_def_roll_vs_magic(
            stats.magic_defence, is_augury, magic_level=ml, defence_level=dl,
        )

        for npc_type, npc_stats in NPC_STATS.items():
            n_atk_melee = npc_melee_attack_roll(npc_stats)
            n_atk_ranged = npc_ranged_attack_roll(npc_stats)
            n_atk_magic = npc_magic_attack_roll(npc_stats)

            npc_table[(npc_type, preset)] = PrecomputedNpcAttack(
                melee_accuracy=hit_chance(n_atk_melee, p_def_melee),
                melee_max_hit=npc_stats.max_hit_melee,
                ranged_accuracy=hit_chance(n_atk_ranged, p_def_ranged),
                ranged_max_hit=npc_stats.max_hit_ranged,
                magic_accuracy=hit_chance(n_atk_magic, p_def_magic),
                magic_max_hit=npc_stats.max_hit_magic,
            )

    return CombatTables(player_attack=player_table, npc_attack=npc_table)


ALL_COMBAT_TABLES: dict[LoadoutId, CombatTables] = {
    lid: build_combat_tables(loadout) for lid, loadout in LOADOUTS.items()
}


# ============================================================
# Expected barrage damage
# ============================================================

def compute_expected_barrage_damage(
    tables: CombatTables | None = None,
) -> dict[InfernoEntityType, dict]:
    """Pre-compute expected barrage damage per NPC type.

    Uses existing pre-computed accuracy/max_hit tables.
    Returns {npc_type: {"accuracy": float, "expected_damage": int}}.
    """
    lookup = tables.player_attack if tables is not None else PLAYER_ATTACK_TABLE
    results = {}
    for npc_type in NPC_STATS:
        entry = lookup.get((GearPreset.MAGE, npc_type))
        if entry is not None:
            expected = int(entry.accuracy * entry.max_hit / 2.0)
            results[npc_type] = {"accuracy": entry.accuracy, "expected_damage": expected}
    return results


# ============================================================
# Convenience roll functions (called per-attack at runtime)
# ============================================================

def roll_player_damage(
    preset: GearPreset,
    npc_type: InfernoEntityType,
    tables: CombatTables | None = None,
) -> int:
    """Roll player damage against an NPC. Returns 0 on miss (splash)."""
    lookup = tables.player_attack if tables is not None else PLAYER_ATTACK_TABLE
    entry = lookup.get((preset, npc_type))
    if entry is None:
        return 0
    if random.random() >= entry.accuracy:
        return 0  # Splash
    return random.randint(0, entry.max_hit)


def roll_npc_damage(
    npc_type: InfernoEntityType,
    preset: GearPreset,
    style: str,
    tables: CombatTables | None = None,
) -> int:
    """
    Roll NPC damage against the player in a given gear preset.

    Args:
        npc_type: The attacking NPC type
        preset: Player's current gear preset (determines defence)
        style: Attack style - "melee", "ranged", or "magic"
        tables: Optional CombatTables; falls back to global tables if None.

    Returns:
        Damage dealt (0 = miss)
    """
    lookup = tables.npc_attack if tables is not None else NPC_ATTACK_TABLE
    entry = lookup.get((npc_type, preset))
    if entry is None:
        return 0

    if style == "melee":
        acc, max_h = entry.melee_accuracy, entry.melee_max_hit
    elif style == "ranged":
        acc, max_h = entry.ranged_accuracy, entry.ranged_max_hit
    else:  # magic
        acc, max_h = entry.magic_accuracy, entry.magic_max_hit

    if max_h <= 0:
        return 0
    if random.random() >= acc:
        return 0  # Miss
    return random.randint(0, max_h)
