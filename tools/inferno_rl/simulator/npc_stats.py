"""
NPC combat stats for Inferno enemies.

All values from OSRS Wiki. Used by combat.py for accuracy/damage calculations.
"""

from dataclasses import dataclass

from .entity import EntityTypes, InfernoEntityType


@dataclass(frozen=True)
class NpcCombatStats:
    """Full combat stats for an NPC type."""
    # Defensive (for player → NPC accuracy)
    defence_level: int
    magic_level: int
    magic_defence: int = 0
    ranged_defence: int = 0
    # Offensive (for NPC → player accuracy)
    attack_level: int = 1      # melee attack level
    ranged_level: int = 1
    melee_attack: int = 0      # melee attack bonus
    ranged_attack: int = 0     # ranged attack bonus
    magic_attack: int = 0      # magic attack bonus
    # Max hits
    max_hit_melee: int = 0
    max_hit_ranged: int = 0
    max_hit_magic: int = 0


# NPC stat table — keyed by InfernoEntityType
NPC_STATS: dict[InfernoEntityType, NpcCombatStats] = {
    EntityTypes.NIBBLER: NpcCombatStats(
        defence_level=15, magic_level=15,
        magic_defence=-20, ranged_defence=-20,
        max_hit_melee=5,  # attacks pillars only, not the player
    ),
    EntityTypes.BAT: NpcCombatStats(
        defence_level=55, magic_level=120,
        magic_defence=-20, ranged_defence=45,
        ranged_level=120, ranged_attack=30,
        max_hit_ranged=15,
    ),
    EntityTypes.BLOB: NpcCombatStats(
        defence_level=95, magic_level=160,
        magic_defence=25, ranged_defence=25,
        attack_level=160, ranged_level=160,
        melee_attack=45, ranged_attack=45, magic_attack=45,
        max_hit_melee=30, max_hit_ranged=30, max_hit_magic=30,
    ),
    EntityTypes.BLOB_RANGE: NpcCombatStats(
        defence_level=95, magic_level=1,
        ranged_defence=25,
        ranged_level=120, ranged_attack=25,
        max_hit_ranged=5,
    ),
    EntityTypes.BLOB_MAGE: NpcCombatStats(
        defence_level=95, magic_level=120,
        magic_defence=25,
        magic_attack=25,
        max_hit_magic=5,
    ),
    EntityTypes.BLOB_MELEE: NpcCombatStats(
        defence_level=95, magic_level=1,
        attack_level=120, melee_attack=25,
        max_hit_melee=5,
    ),
    EntityTypes.MELEE: NpcCombatStats(
        defence_level=120, magic_level=120,
        magic_defence=30, ranged_defence=50,
        attack_level=210,
        max_hit_melee=46,
    ),
    EntityTypes.RANGER: NpcCombatStats(
        defence_level=60, magic_level=90,
        attack_level=140, ranged_level=250,
        ranged_attack=40,
        max_hit_melee=19, max_hit_ranged=46,
    ),
    EntityTypes.MAGER: NpcCombatStats(
        defence_level=260, magic_level=300,
        attack_level=370,
        magic_attack=80,
        max_hit_melee=50, max_hit_magic=70,
    ),
}
