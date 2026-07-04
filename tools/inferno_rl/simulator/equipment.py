"""
Equipment definitions, gear presets, and aggregated stat computation for Inferno RL.

Near-max gear: BoFa + Crystal armor, Blowpipe (dragon darts), Ancient Sceptre + Ahrim's.
Stats sourced from OSRS Wiki.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass(frozen=True)
class EquipmentStats:
    """Combat bonuses for a single equipment item."""
    ranged_attack: int = 0
    ranged_strength: int = 0
    magic_attack: int = 0
    magic_damage_percent: float = 0.0  # e.g. 0.10 = +10%
    stab_defence: int = 0
    slash_defence: int = 0
    crush_defence: int = 0
    ranged_defence: int = 0
    magic_defence: int = 0


@dataclass(frozen=True)
class Equipment:
    """A single equipment item with its slot, stats, and weapon properties."""
    name: str
    slot: str
    stats: EquipmentStats
    attack_speed: int = 0      # Only for weapons (ticks between attacks)
    attack_range: int = 0      # Only for weapons
    is_two_handed: bool = False
    base_spell_max_hit: int = 0  # Only for magic weapons (ice/blood barrage base)


class GearPreset(Enum):
    """Gear presets the agent can switch between."""
    # "Main ranged weapon" — maps to RCB/BoFa/Tbow depending on loadout.
    BOFA = "Bow of Faerdhinen"
    BLOWPIPE = "Toxic Blowpipe"
    MAGE = "Ancient Sceptre"


@dataclass(frozen=True)
class AggregateStats:
    """Totalled bonuses for a complete gear preset, ready for combat formulas."""
    ranged_attack: int = 0
    ranged_strength: int = 0
    magic_attack: int = 0
    magic_damage_percent: float = 0.0
    stab_defence: int = 0
    slash_defence: int = 0
    crush_defence: int = 0
    ranged_defence: int = 0
    magic_defence: int = 0
    attack_speed: int = 4
    attack_range: int = 10
    has_crystal_set: bool = False
    base_spell_max_hit: int = 0


# ============================================================
# Item definitions (stats from OSRS Wiki)
# ============================================================

BOW_OF_FAERDHINEN = Equipment(
    name="Bow of Faerdhinen (c)",
    slot="weapon",
    stats=EquipmentStats(ranged_attack=128, ranged_strength=106),
    attack_speed=4,
    attack_range=10,
    is_two_handed=True,
)

TOXIC_BLOWPIPE = Equipment(
    name="Toxic Blowpipe (dragon darts)",
    slot="weapon",
    stats=EquipmentStats(ranged_attack=30, ranged_strength=35),  # 20 bp + 15 darts
    attack_speed=2,
    attack_range=5,
    is_two_handed=True,
)

ANCIENT_SCEPTRE = Equipment(
    name="Ancient Sceptre",
    slot="weapon",
    stats=EquipmentStats(magic_attack=20, magic_damage_percent=0.10),
    attack_speed=5,
    attack_range=10,
    is_two_handed=False,
    base_spell_max_hit=31,  # Ice/blood barrage base max hit
)

CRYSTAL_SHIELD = Equipment(
    name="Crystal Shield",
    slot="shield",
    stats=EquipmentStats(
        stab_defence=51, slash_defence=54, crush_defence=53,
        ranged_defence=80, magic_defence=-10,
    ),
)

CRYSTAL_BODY = Equipment(
    name="Crystal Body",
    slot="body",
    stats=EquipmentStats(ranged_attack=30, ranged_defence=57, magic_defence=-15),
)

CRYSTAL_LEGS = Equipment(
    name="Crystal Legs",
    slot="legs",
    stats=EquipmentStats(ranged_attack=18, ranged_defence=39, magic_defence=-12),
)

AHRIMS_ROBE_TOP = Equipment(
    name="Ahrim's Robe Top",
    slot="body",
    stats=EquipmentStats(magic_attack=30, magic_defence=20, ranged_defence=-5),
)

AHRIMS_ROBE_BOTTOM = Equipment(
    name="Ahrim's Robe Bottom",
    slot="legs",
    stats=EquipmentStats(magic_attack=22, magic_defence=15, ranged_defence=-4),
)

NECKLACE_OF_ANGUISH = Equipment(
    name="Necklace of Anguish",
    slot="amulet",
    stats=EquipmentStats(ranged_attack=15, ranged_strength=5),
)

OCCULT_NECKLACE = Equipment(
    name="Occult Necklace",
    slot="amulet",
    stats=EquipmentStats(magic_attack=12, magic_damage_percent=0.10),
)

# ============================================================
# Static slot bonuses (Crystal helm, Ava's assembler,
# Barrows gloves, Ancient d'hide boots, Ring of suffering (i))
# These items never change between presets.
# ============================================================

STATIC_BONUSES = EquipmentStats(
    ranged_attack=36,
    ranged_strength=2,
    magic_attack=-14,
    stab_defence=49,
    slash_defence=45,
    crush_defence=51,
    ranged_defence=56,
    magic_defence=48,
)

# ============================================================
# Gear presets — items that swap per preset
# ============================================================

PRESET_ITEMS: dict[GearPreset, list] = {
    GearPreset.BOFA: [
        BOW_OF_FAERDHINEN,  # weapon (2H, no shield)
        CRYSTAL_BODY,       # body
        CRYSTAL_LEGS,       # legs
        NECKLACE_OF_ANGUISH,  # amulet
    ],
    GearPreset.BLOWPIPE: [
        TOXIC_BLOWPIPE,     # weapon (2H, no shield)
        CRYSTAL_BODY,       # body
        CRYSTAL_LEGS,       # legs
        NECKLACE_OF_ANGUISH,  # amulet
    ],
    GearPreset.MAGE: [
        ANCIENT_SCEPTRE,    # weapon (1H)
        CRYSTAL_SHIELD,     # shield
        AHRIMS_ROBE_TOP,    # body
        AHRIMS_ROBE_BOTTOM, # legs
        OCCULT_NECKLACE,    # amulet
    ],
}


def _with_uniform_defence(stats: AggregateStats, defence: int = 30) -> AggregateStats:
    """Replace all defensive bonuses with a uniform value (V46: 1-def training)."""
    return AggregateStats(
        ranged_attack=stats.ranged_attack,
        ranged_strength=stats.ranged_strength,
        magic_attack=stats.magic_attack,
        magic_damage_percent=stats.magic_damage_percent,
        stab_defence=defence,
        slash_defence=defence,
        crush_defence=defence,
        ranged_defence=defence,
        magic_defence=defence,
        attack_speed=stats.attack_speed,
        attack_range=stats.attack_range,
        has_crystal_set=stats.has_crystal_set,
        base_spell_max_hit=stats.base_spell_max_hit,
    )


def compute_aggregate_stats(
    preset: GearPreset,
    *,
    uniform_defence: int | None = 30,
) -> AggregateStats:
    """Sum all item stats + static bonuses for a preset. Detect crystal set effect.

    Args:
        uniform_defence: Flatten all defence bonuses to this value (training default: 30).
                         Pass None to keep real equipment defence values (for eval).
    """
    items = PRESET_ITEMS[preset]

    # Start with static bonuses
    ranged_attack = STATIC_BONUSES.ranged_attack
    ranged_strength = STATIC_BONUSES.ranged_strength
    magic_attack = STATIC_BONUSES.magic_attack
    magic_damage_percent = STATIC_BONUSES.magic_damage_percent
    stab_defence = STATIC_BONUSES.stab_defence
    slash_defence = STATIC_BONUSES.slash_defence
    crush_defence = STATIC_BONUSES.crush_defence
    ranged_defence = STATIC_BONUSES.ranged_defence
    magic_defence = STATIC_BONUSES.magic_defence

    attack_speed = 4
    attack_range = 10
    base_spell_max_hit = 0

    # Track crystal set pieces for set effect detection
    has_crystal_body = False
    has_crystal_legs = False
    has_bofa = False

    for item in items:
        s = item.stats
        ranged_attack += s.ranged_attack
        ranged_strength += s.ranged_strength
        magic_attack += s.magic_attack
        magic_damage_percent += s.magic_damage_percent
        stab_defence += s.stab_defence
        slash_defence += s.slash_defence
        crush_defence += s.crush_defence
        ranged_defence += s.ranged_defence
        magic_defence += s.magic_defence

        if item.attack_speed > 0:
            attack_speed = item.attack_speed
            attack_range = item.attack_range
            base_spell_max_hit = item.base_spell_max_hit

        if item.name == "Crystal Body":
            has_crystal_body = True
        if item.name == "Crystal Legs":
            has_crystal_legs = True
        if item.name == "Bow of Faerdhinen (c)":
            has_bofa = True

    # Crystal set effect: BoFa + Crystal Body + Crystal Legs + Crystal Helm (in static)
    # → +15% damage, +30% accuracy (applied in combat.py formulas)
    has_crystal_set = has_bofa and has_crystal_body and has_crystal_legs

    raw = AggregateStats(
        ranged_attack=ranged_attack,
        ranged_strength=ranged_strength,
        magic_attack=magic_attack,
        magic_damage_percent=magic_damage_percent,
        stab_defence=stab_defence,
        slash_defence=slash_defence,
        crush_defence=crush_defence,
        ranged_defence=ranged_defence,
        magic_defence=magic_defence,
        attack_speed=attack_speed,
        attack_range=attack_range,
        has_crystal_set=has_crystal_set,
        base_spell_max_hit=base_spell_max_hit,
    )
    if uniform_defence is not None:
        return _with_uniform_defence(raw, uniform_defence)
    return raw


# Pre-computed at import time. Zero per-tick cost.
PRESET_STATS: dict[GearPreset, AggregateStats] = {
    preset: compute_aggregate_stats(preset) for preset in GearPreset
}


# ============================================================
# Multi-loadout support
# ============================================================

class LoadoutId(Enum):
    BUDGET_RCB = auto()
    MID_ACB = auto()
    CRYSTAL_BP = auto()
    CRYSTAL_NO_BP = auto()
    MAX_TBOW = auto()


@dataclass(frozen=True)
class PlayerLevels:
    """Combat levels that affect max hits and accuracy."""
    hitpoints: int
    ranged: int
    magic: int
    defence: int


@dataclass(frozen=True)
class PrayerMultipliers:
    """Offensive prayer multipliers for combat formulas."""
    ranged_atk: float = 1.20      # Rigour
    ranged_str: float = 1.23      # Rigour
    magic_atk: float = 1.25       # Augury
    magic_dmg: float = 0.04       # Augury (+4% additive)


RIGOUR_AUGURY = PrayerMultipliers()
EAGLE_EYE_MYSTIC_MIGHT = PrayerMultipliers(
    ranged_atk=1.15, ranged_str=1.15,
    magic_atk=1.15, magic_dmg=0.0,
)


@dataclass(frozen=True)
class Loadout:
    """Complete gear + level definition for a training loadout."""
    id: LoadoutId
    levels: PlayerLevels
    preset_stats: dict[GearPreset, AggregateStats]
    has_blowpipe: bool
    has_tbow: bool = False
    prayers: PrayerMultipliers = field(default_factory=PrayerMultipliers)


# Crystal loadout preset stats (reuse existing computation)
_CRYSTAL_PRESET_STATS: dict[GearPreset, AggregateStats] = {
    preset: compute_aggregate_stats(preset) for preset in GearPreset
}

_BUDGET_RCB = Loadout(
    id=LoadoutId.BUDGET_RCB,
    levels=PlayerLevels(hitpoints=80, ranged=80, magic=94, defence=1),
    preset_stats={
        GearPreset.BOFA: _with_uniform_defence(AggregateStats(
            ranged_attack=173, ranged_strength=112, magic_attack=-26,
            attack_speed=5, attack_range=7,
        )),
        GearPreset.BLOWPIPE: _with_uniform_defence(AggregateStats(
            ranged_attack=121, ranged_strength=42, magic_attack=-14,
            attack_speed=2, attack_range=5,
        )),
        GearPreset.MAGE: _with_uniform_defence(AggregateStats(
            magic_attack=43,
            attack_speed=5, attack_range=10, base_spell_max_hit=30,
        )),
    },
    has_blowpipe=True,
    prayers=EAGLE_EYE_MYSTIC_MIGHT,
)

_MID_ACB = Loadout(
    id=LoadoutId.MID_ACB,
    levels=PlayerLevels(hitpoints=85, ranged=85, magic=90, defence=1),
    preset_stats={
        # Armadyl crossbow + book of law + diamond bolts (e), 1-def pure gear
        GearPreset.BOFA: _with_uniform_defence(AggregateStats(
            ranged_attack=170, ranged_strength=112, magic_attack=-18,
            attack_speed=5, attack_range=8,
        )),
        GearPreset.BLOWPIPE: _with_uniform_defence(AggregateStats(
            ranged_attack=100, ranged_strength=42, magic_attack=-8,
            attack_speed=2, attack_range=5,
        )),
        GearPreset.MAGE: _with_uniform_defence(AggregateStats(
            magic_attack=35,
            attack_speed=5, attack_range=10, base_spell_max_hit=30,
        )),
    },
    has_blowpipe=True,
    prayers=EAGLE_EYE_MYSTIC_MIGHT,
)

_CRYSTAL_BP = Loadout(
    id=LoadoutId.CRYSTAL_BP,
    levels=PlayerLevels(hitpoints=90, ranged=90, magic=94, defence=1),
    preset_stats=_CRYSTAL_PRESET_STATS,
    has_blowpipe=True,
)

_CRYSTAL_NO_BP = Loadout(
    id=LoadoutId.CRYSTAL_NO_BP,
    levels=PlayerLevels(hitpoints=90, ranged=90, magic=94, defence=1),
    preset_stats=_CRYSTAL_PRESET_STATS,
    has_blowpipe=False,
)

_MAX_TBOW = Loadout(
    id=LoadoutId.MAX_TBOW,
    levels=PlayerLevels(hitpoints=99, ranged=99, magic=99, defence=1),
    preset_stats={
        GearPreset.BOFA: _with_uniform_defence(AggregateStats(
            ranged_attack=199, ranged_strength=95, magic_attack=-17,
            attack_speed=5, attack_range=10,
        )),
        GearPreset.BLOWPIPE: _with_uniform_defence(AggregateStats(
            ranged_attack=159, ranged_strength=110, magic_attack=-17,
            attack_speed=2, attack_range=5,
        )),
        GearPreset.MAGE: compute_aggregate_stats(GearPreset.MAGE),
    },
    has_blowpipe=True,
    has_tbow=True,
)

LOADOUTS: dict[LoadoutId, Loadout] = {
    LoadoutId.BUDGET_RCB: _BUDGET_RCB,
    LoadoutId.MID_ACB: _MID_ACB,
    LoadoutId.CRYSTAL_BP: _CRYSTAL_BP,
    LoadoutId.CRYSTAL_NO_BP: _CRYSTAL_NO_BP,
    LoadoutId.MAX_TBOW: _MAX_TBOW,
}

DEFAULT_LOADOUT = LOADOUTS[LoadoutId.CRYSTAL_BP]
