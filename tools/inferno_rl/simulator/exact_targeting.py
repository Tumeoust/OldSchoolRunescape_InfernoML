from __future__ import annotations

from typing import Iterable, Optional

from .entity import EntityTypes, InfernoEntityType, PlacedEntity
from .geometry import InfernoLineOfSight, SimulatorGeometry
from .priority import combat_entity_sort_key
from .state import PILLAR_MAX_HP, SimulatorState


MAX_TARGET_SLOTS = 14

EXACT_TARGET_ENTITY_TYPES: list[InfernoEntityType] = [
    EntityTypes.MAGER,
    EntityTypes.RANGER,
    EntityTypes.MELEE,
    EntityTypes.BLOB,
    EntityTypes.BLOB_MAGE,
    EntityTypes.BLOB_RANGE,
    EntityTypes.BLOB_MELEE,
    EntityTypes.BAT,
    EntityTypes.NIBBLER,
]
EXACT_TARGET_TYPE_ONE_HOT_SIZE = len(EXACT_TARGET_ENTITY_TYPES)

_PILLAR_IMPORTANCE = {
    1: 0,  # NE
    0: 1,  # NW
    2: 2,  # S
}


def _nibbler_pillar_urgency(
    state: SimulatorState,
    nibbler: PlacedEntity,
) -> tuple[int, int, int]:
    pillar_index = nibbler.target_pillar_index
    if pillar_index < 0 or pillar_index >= len(state.pillar_alive):
        return (2, PILLAR_MAX_HP + 1, 3)
    if not state.pillar_alive[pillar_index]:
        return (1, PILLAR_MAX_HP + 1, _PILLAR_IMPORTANCE.get(pillar_index, 3))
    return (
        0,
        state.pillar_hp[pillar_index],
        _PILLAR_IMPORTANCE.get(pillar_index, 3),
    )


def nibbler_sort_key(
    state: SimulatorState,
    nibbler: PlacedEntity,
) -> tuple[int, int, int, float, int]:
    pillar_urgency = _nibbler_pillar_urgency(state, nibbler)
    distance = InfernoLineOfSight.get_distance_from_npc(
        nibbler.x,
        nibbler.y,
        nibbler.entity_type.size_in_tiles,
        state.player_x,
        state.player_y,
    )
    return (*pillar_urgency, distance, nibbler.id)


def get_exact_target_entities(state: SimulatorState) -> list[PlacedEntity]:
    combat_entities = [
        entity for entity in state.entities
        if not entity.is_dead() and entity.entity_type != EntityTypes.NIBBLER
    ]
    combat_entities.sort(
        key=lambda entity: combat_entity_sort_key(
            entity,
            state.player_x,
            state.player_y,
            state.pillar_alive,
        )
    )

    nibblers = [
        entity for entity in state.entities
        if not entity.is_dead() and entity.entity_type == EntityTypes.NIBBLER
    ]
    nibblers.sort(key=lambda entity: nibbler_sort_key(state, entity))

    return combat_entities + nibblers


def get_exact_target_slots(state: SimulatorState) -> list[PlacedEntity]:
    return get_exact_target_entities(state)[:MAX_TARGET_SLOTS]


def get_exact_target_by_slot(
    state: SimulatorState,
    slot_index: int,
) -> Optional[PlacedEntity]:
    if slot_index < 0:
        return None
    targets = get_exact_target_slots(state)
    if slot_index >= len(targets):
        return None
    return targets[slot_index]


def get_exact_target_slot_index(
    state: SimulatorState,
    target: PlacedEntity,
) -> Optional[int]:
    for index, entity in enumerate(get_exact_target_slots(state)):
        if entity is target or entity.id == target.id:
            return index
    return None


def get_exact_target_type_index(entity_type: InfernoEntityType) -> int:
    return EXACT_TARGET_ENTITY_TYPES.index(entity_type)


def count_adjacent_nibblers(
    candidate: PlacedEntity,
    nibblers: Iterable[PlacedEntity],
) -> int:
    count = 0
    for other in nibblers:
        if other is candidate:
            continue
        distance = SimulatorGeometry.chebyshev_distance(
            candidate.x,
            candidate.y,
            other.x,
            other.y,
        )
        if distance <= 1:
            count += 1
    return count


def select_center_nibbler(entities: Iterable[PlacedEntity]) -> Optional[PlacedEntity]:
    nibblers = sorted(
        (
            entity for entity in entities
            if not entity.is_dead() and entity.entity_type == EntityTypes.NIBBLER
        ),
        key=lambda entity: entity.id,
    )
    if not nibblers:
        return None

    best_nibbler = None
    best_adjacent = -1
    for candidate in nibblers:
        adjacent = count_adjacent_nibblers(candidate, nibblers)
        if adjacent > best_adjacent:
            best_nibbler = candidate
            best_adjacent = adjacent
    return best_nibbler
