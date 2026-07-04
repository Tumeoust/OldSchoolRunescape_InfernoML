"""
Eval-only loadout definitions with real defence levels and equipment defence bonuses.

Training loadouts use defence=1 and uniform equipment defence of 30.
These eval loadouts use actual game stats for more realistic performance evaluation.
The trained policy is unaffected — defence is not in the observation space.
"""

from __future__ import annotations

from .combat import CombatTables, build_combat_tables
from .equipment import (
    AggregateStats,
    GearPreset,
    Loadout,
    LoadoutId,
    LOADOUTS,
    PlayerLevels,
    EAGLE_EYE_MYSTIC_MIGHT,
    compute_aggregate_stats,
)


# Real crystal armour preset stats (without uniform defence override).
# Crystal body + Crystal legs + static slots (helm, ava's, gloves, boots, ring).
_REAL_CRYSTAL_PRESET_STATS: dict[GearPreset, AggregateStats] = {
    preset: compute_aggregate_stats(preset, uniform_defence=None) for preset in GearPreset
}


def _with_defence(base: AggregateStats, stab: int, slash: int, crush: int,
                  ranged: int, magic: int) -> AggregateStats:
    """Keep offensive stats from base, replace defence with provided values."""
    return AggregateStats(
        ranged_attack=base.ranged_attack,
        ranged_strength=base.ranged_strength,
        magic_attack=base.magic_attack,
        magic_damage_percent=base.magic_damage_percent,
        stab_defence=stab,
        slash_defence=slash,
        crush_defence=crush,
        ranged_defence=ranged,
        magic_defence=magic,
        attack_speed=base.attack_speed,
        attack_range=base.attack_range,
        has_crystal_set=base.has_crystal_set,
        base_spell_max_hit=base.base_spell_max_hit,
    )


# ============================================================
# Shared defence stats by armour set
# ============================================================

# God d'hide ranged setup (Budget/Mid loadouts).
_GOD_DHIDE_DEF = dict(stab=127, slash=116, crush=140, magic=120, ranged=120)

# Budget/Mid mage setup: Ahrim's + Crystal shield (same as crystal loadout mage).
_BUDGET_MAGE_STATS = _REAL_CRYSTAL_PRESET_STATS[GearPreset.MAGE]

# Max mage setup: used by Crystal and Max Tbow loadouts.
_MAX_MAGE_STATS = AggregateStats(
    magic_attack=68, magic_damage_percent=0.23,
    attack_speed=5, attack_range=10, base_spell_max_hit=31,
    stab_defence=180, slash_defence=170, crush_defence=208,
    ranged_defence=136, magic_defence=140,
)


# ============================================================
# Eval loadout definitions
# Defence level = ranged level for each loadout
# ============================================================

def _budget_mid_preset_stats(lid: LoadoutId) -> dict[GearPreset, AggregateStats]:
    """Build eval preset stats for Budget/Mid (god d'hides ranged, ahrim's mage)."""
    training = LOADOUTS[lid].preset_stats
    return {
        GearPreset.BOFA: _with_defence(training[GearPreset.BOFA], **_GOD_DHIDE_DEF),
        GearPreset.BLOWPIPE: _with_defence(training[GearPreset.BLOWPIPE], **_GOD_DHIDE_DEF),
        GearPreset.MAGE: _BUDGET_MAGE_STATS,
    }


def _crystal_preset_stats() -> dict[GearPreset, AggregateStats]:
    """Crystal armour ranged presets + max mage preset."""
    return {
        GearPreset.BOFA: _REAL_CRYSTAL_PRESET_STATS[GearPreset.BOFA],
        GearPreset.BLOWPIPE: _REAL_CRYSTAL_PRESET_STATS[GearPreset.BLOWPIPE],
        GearPreset.MAGE: _MAX_MAGE_STATS,
    }


# --- BUDGET_RCB (def=80) ---
_EVAL_BUDGET_RCB = Loadout(
    id=LoadoutId.BUDGET_RCB,
    levels=PlayerLevels(hitpoints=80, ranged=80, magic=94, defence=80),
    preset_stats=_budget_mid_preset_stats(LoadoutId.BUDGET_RCB),
    has_blowpipe=True,
    prayers=EAGLE_EYE_MYSTIC_MIGHT,
)

# --- MID_ACB (def=85) ---
_EVAL_MID_ACB = Loadout(
    id=LoadoutId.MID_ACB,
    levels=PlayerLevels(hitpoints=85, ranged=85, magic=90, defence=85),
    preset_stats=_budget_mid_preset_stats(LoadoutId.MID_ACB),
    has_blowpipe=True,
    prayers=EAGLE_EYE_MYSTIC_MIGHT,
)

# --- CRYSTAL_BP (def=90) ---
_EVAL_CRYSTAL_BP = Loadout(
    id=LoadoutId.CRYSTAL_BP,
    levels=PlayerLevels(hitpoints=90, ranged=90, magic=94, defence=90),
    preset_stats=_crystal_preset_stats(),
    has_blowpipe=True,
)

# --- CRYSTAL_NO_BP (def=90) ---
_EVAL_CRYSTAL_NO_BP = Loadout(
    id=LoadoutId.CRYSTAL_NO_BP,
    levels=PlayerLevels(hitpoints=90, ranged=90, magic=94, defence=90),
    preset_stats=_crystal_preset_stats(),
    has_blowpipe=False,
)

# --- MAX_TBOW (def=99) ---
_tbow_bp = LOADOUTS[LoadoutId.MAX_TBOW].preset_stats[GearPreset.BLOWPIPE]

_EVAL_MAX_TBOW = Loadout(
    id=LoadoutId.MAX_TBOW,
    levels=PlayerLevels(hitpoints=99, ranged=99, magic=99, defence=99),
    preset_stats={
        GearPreset.BOFA: AggregateStats(
            ranged_attack=222, ranged_strength=97, magic_attack=-17,
            attack_speed=5, attack_range=10,
            stab_defence=118, slash_defence=108, crush_defence=131,
            ranged_defence=130, magic_defence=142,
        ),
        GearPreset.BLOWPIPE: AggregateStats(
            ranged_attack=_tbow_bp.ranged_attack,
            ranged_strength=_tbow_bp.ranged_strength,
            magic_attack=_tbow_bp.magic_attack,
            attack_speed=2, attack_range=5,
            stab_defence=118, slash_defence=108, crush_defence=131,
            ranged_defence=130, magic_defence=142,
        ),
        GearPreset.MAGE: _MAX_MAGE_STATS,
    },
    has_blowpipe=True,
    has_tbow=True,
)

# ============================================================
# Public API
# ============================================================

EVAL_LOADOUTS: dict[LoadoutId, Loadout] = {
    LoadoutId.BUDGET_RCB: _EVAL_BUDGET_RCB,
    LoadoutId.MID_ACB: _EVAL_MID_ACB,
    LoadoutId.CRYSTAL_BP: _EVAL_CRYSTAL_BP,
    LoadoutId.CRYSTAL_NO_BP: _EVAL_CRYSTAL_NO_BP,
    LoadoutId.MAX_TBOW: _EVAL_MAX_TBOW,
}

EVAL_COMBAT_TABLES: dict[LoadoutId, CombatTables] = {
    lid: build_combat_tables(loadout) for lid, loadout in EVAL_LOADOUTS.items()
}


def configure_sim_for_eval(sim: 'InfernoSimulator', loadout_id: LoadoutId) -> None:
    """Configure simulator with real stats for evaluation."""
    eval_loadout = EVAL_LOADOUTS[loadout_id]
    sim.state.max_health = eval_loadout.levels.hitpoints
    sim.state.has_blowpipe = eval_loadout.has_blowpipe
    sim.state.loadout_preset_stats = eval_loadout.preset_stats
    sim.combat_tables = EVAL_COMBAT_TABLES[loadout_id]
