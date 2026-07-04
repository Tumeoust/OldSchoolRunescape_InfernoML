"""
Shared observation constants, types, and small helpers for observation v4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..simulator.entity import EntityTypes, InfernoEntityType, PlacedEntity
from ..simulator.equipment import GearPreset
from ..simulator.exact_targeting import (
    EXACT_TARGET_ENTITY_TYPES,
    EXACT_TARGET_TYPE_ONE_HOT_SIZE,
    MAX_TARGET_SLOTS,
)
from ..simulator.geometry import GRID_HEIGHT, GRID_WIDTH
from ..simulator.priority import combat_entity_sort_key
from ..simulator.state import SimulatorState
from .actions import InfernoAction

MAX_HEALTH = 99.0
MAX_WAVE = 66.0
MAX_ATTACK_COOLDOWN = 5.0
MAX_ENTITY_COOLDOWN = 8.0
MAX_ENTITY_STUN_TICKS = 4.0
MAX_ENTITY_FROZEN_TICKS = 32.0
MAX_TICK_IN_WAVE = 500.0
MAX_DISTANCE = 30.0
NUM_WEAPON_TYPES = 4
NUM_PRAYERS = 4
MAX_NIBBLER_COUNT = 10.0
MAX_PLAYER_ATTACK_RANGE = 10.0

LOADOUT_BLOCK_SIZE = 7

MAX_WEAPON_SPEED = 5.0
MAX_RANGED_ATTACK_BONUS = 200.0
MAX_RANGED_STRENGTH_BONUS = 130.0
MAX_MAGIC_ATTACK_BONUS = 80.0

PILLAR_CENTERS = [
    (1.0, 21.0),
    (18.0, 23.0),
    (11.0, 7.0),
]
NE_PILLAR_CENTER = PILLAR_CENTERS[1]

GLOBAL_V4_SIZE = 51
TEMPORAL_V3_SIZE = 7

MAX_SAME_NPC_COUNT = 2.0

NEIGHBORHOOD_TILES = 9
NEIGHBORHOOD_FEATURES = 12
NEIGHBORHOOD_FORECAST_SIZE = NEIGHBORHOOD_TILES * NEIGHBORHOOD_FEATURES
MAX_NEIGHBORHOOD_LOS_COUNT = 9.0
MAX_NEIGHBORHOOD_IMMINENT_COUNT = 6.0

THREAT_HORIZON_SIZE = 9

SLOT_CORE_SIZE = 21
EXACT_TARGET_SLOT_COUNT = MAX_TARGET_SLOTS
EXACT_TARGET_SLOT_SIZE = SLOT_CORE_SIZE + EXACT_TARGET_TYPE_ONE_HOT_SIZE
EXACT_TARGET_SLOTS_TOTAL = EXACT_TARGET_SLOT_COUNT * EXACT_TARGET_SLOT_SIZE

OBSERVATION_PUBLIC_SIZE = (
    GLOBAL_V4_SIZE
    + NEIGHBORHOOD_FORECAST_SIZE
    + THREAT_HORIZON_SIZE
    + TEMPORAL_V3_SIZE
    + EXACT_TARGET_SLOTS_TOTAL
    + LOADOUT_BLOCK_SIZE
)
OBSERVATION_PRIVILEGED_SIZE = 0
OBSERVATION_TOTAL_SIZE = OBSERVATION_PUBLIC_SIZE

MAX_BLOB_SCAN_COUNT = 2.0
MAX_DEAD_POOL_COUNT = 5.0
MAX_RESURRECTION_HAZARD = 10.0

TEMPORAL_BUFFER_SIZE = 5


@dataclass
class TemporalState:
    """Rolling temporal features for v3+ observations."""

    damage_taken_buffer: list[int] = field(
        default_factory=lambda: [0] * TEMPORAL_BUFFER_SIZE
    )
    damage_dealt_buffer: list[int] = field(
        default_factory=lambda: [0] * TEMPORAL_BUFFER_SIZE
    )
    ticks_since_last_attack: int = 0
    ticks_since_engagement: int = 0
    prev_action_movement: bool = False
    prev_action_attack: bool = False
    prev_action_weapon_switch: bool = False

    def reset(self) -> None:
        self.damage_taken_buffer = [0] * TEMPORAL_BUFFER_SIZE
        self.damage_dealt_buffer = [0] * TEMPORAL_BUFFER_SIZE
        self.ticks_since_last_attack = 0
        self.ticks_since_engagement = 0
        self.prev_action_movement = False
        self.prev_action_attack = False
        self.prev_action_weapon_switch = False


def update_temporal_state(temporal: TemporalState | None, action: int, result) -> None:
    if temporal is None:
        return

    temporal.damage_taken_buffer.pop(0)
    temporal.damage_taken_buffer.append(result.damage_taken)
    temporal.damage_dealt_buffer.pop(0)
    temporal.damage_dealt_buffer.append(result.damage_dealt)
    if InfernoAction.is_attack(action):
        temporal.ticks_since_last_attack = 0
    else:
        temporal.ticks_since_last_attack += 1
    if result.player_attacked_on_cooldown:
        temporal.ticks_since_engagement = 0
    else:
        temporal.ticks_since_engagement += 1
    temporal.prev_action_movement = InfernoAction.is_movement(action)
    temporal.prev_action_attack = InfernoAction.is_attack(action)
    temporal.prev_action_weapon_switch = InfernoAction.is_weapon_switch(action)


_PRAYER_INDEX = {
    None: 0,
    "PROTECT_FROM_MAGIC": 1,
    "PROTECT_FROM_MISSILES": 2,
    "PROTECT_FROM_MELEE": 3,
}
_GLOBAL_COUNT_TYPES = [
    EntityTypes.BAT,
    EntityTypes.BLOB,
    EntityTypes.MELEE,
    EntityTypes.RANGER,
    EntityTypes.MAGER,
]
_DEAD_POOL_TYPES = [
    EntityTypes.BAT,
    EntityTypes.BLOB,
    EntityTypes.MELEE,
    EntityTypes.RANGER,
    EntityTypes.MAGER,
]


def get_observation_size() -> int:
    return OBSERVATION_TOTAL_SIZE


def get_public_observation_size() -> int:
    return OBSERVATION_PUBLIC_SIZE


def get_observation_low() -> float:
    return -1.0


def get_weapon_index(preset: GearPreset, use_blood_barrage: bool) -> int:
    if preset == GearPreset.BOFA:
        return 0
    if preset == GearPreset.BLOWPIPE:
        return 1
    return 3 if use_blood_barrage else 2


def get_sorted_combat_entities(state: SimulatorState) -> list[PlacedEntity]:
    entities = [
        e for e in state.entities
        if not e.is_dead() and e.entity_type != EntityTypes.NIBBLER
    ]
    entities.sort(
        key=lambda e: combat_entity_sort_key(
            e, state.player_x, state.player_y, state.pillar_alive,
        )
    )
    return entities


def _encode_prayer_one_hot(out: np.ndarray, offset: int, prayer: str | None) -> None:
    prayer_idx = _PRAYER_INDEX.get(prayer, 0)
    out[offset + prayer_idx] = 1.0


def _normalize_signed(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(np.clip(value / scale, -1.0, 1.0))


def _get_alive_entities(state: SimulatorState) -> list[PlacedEntity]:
    return [e for e in state.entities if not e.is_dead()]
