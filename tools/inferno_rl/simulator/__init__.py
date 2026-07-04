"""
Inferno Simulator for RL Training.

A headless, tick-accurate reimplementation of the fight, designed for
fast vectorized RL training.
"""

from .entity import InfernoEntityType, AttackStyle, PlacedEntity, EntityTypes
from .geometry import SimulatorGeometry, InfernoLineOfSight
from .state import SimulatorState
from .equipment import GearPreset, PRESET_STATS, LoadoutId, Loadout, LOADOUTS, DEFAULT_LOADOUT
from .combat import roll_player_damage, roll_npc_damage, CombatTables, ALL_COMBAT_TABLES
from .pathfinding import OSRSPathfinding
from .simulator import InfernoSimulator, StepResult, PlayerDamageEvent

__all__ = [
    'InfernoEntityType',
    'AttackStyle',
    'PlacedEntity',
    'EntityTypes',
    'SimulatorGeometry',
    'InfernoLineOfSight',
    'SimulatorState',
    'GearPreset',
    'PRESET_STATS',
    'LoadoutId',
    'Loadout',
    'LOADOUTS',
    'DEFAULT_LOADOUT',
    'roll_player_damage',
    'roll_npc_damage',
    'CombatTables',
    'ALL_COMBAT_TABLES',
    'OSRSPathfinding',
    'InfernoSimulator',
    'StepResult',
    'PlayerDamageEvent',
]
