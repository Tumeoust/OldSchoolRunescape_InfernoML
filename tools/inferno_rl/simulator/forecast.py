from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np

from .entity import AttackStyle, EntityTypes, InfernoEntityType, PlacedEntity
from .geometry import InfernoLineOfSight, SimulatorGeometry
from .movement_actions import (
    DIRECTIONAL_MOVE_ACTIONS,
    PLAYER_MOVE_DIRECTIONS,
    get_movement_params,
    iter_legacy_movement_actions,
)
from .pathfinding import OSRSPathfinding

try:
    from . import forecast_fast as _forecast_fast_backend
except (ModuleNotFoundError, ImportError):
    _forecast_fast_backend = None


DIG_SEQUENCE_DURATION = 6
POST_DIG_ATTACK_DELAY = 6
POST_DIG_FROZEN_TICKS = 2
DIG_TRIGGER_RANDOM_THRESHOLD = -38
DIG_TRIGGER_GUARANTEED = -50


@dataclass(frozen=True)
class ThreatStyleCounts:
    magic: int = 0
    ranged: int = 0
    melee: int = 0


@dataclass(frozen=True)
class DirectionalForecast:
    settled_step_distance: int
    los_count: int
    imminent_attacks: int


@dataclass(frozen=True)
class NeighborhoodForecast:
    settled_step_distance: float
    los_count: float
    los_delta: float
    min_attack_delay: float
    imminent_magic: float
    imminent_ranged: float
    imminent_melee: float
    unprotected_after_auto_prayer: float
    blob_scan_triggers: float
    priority_target_attackable: float = 0.0
    best_los_in_2_steps: float = 0.0
    steps_to_single_los: float = 1.0


# NPC types that count for multi-step BFS forecast (same set as LOS separation reward).
_BFS_DANGEROUS_TYPES = frozenset({
    EntityTypes.MAGER, EntityTypes.RANGER, EntityTypes.MELEE, EntityTypes.BLOB,
})


@dataclass(frozen=True)
class MovementResolutionTable:
    player_x: int
    player_y: int
    destinations: np.ndarray

    def destination_x_for_action(self, legacy_movement_action: int) -> int:
        if 0 <= legacy_movement_action < self.destinations.shape[0]:
            return int(self.destinations[legacy_movement_action, 0])
        return self.player_x

    def destination_y_for_action(self, legacy_movement_action: int) -> int:
        if 0 <= legacy_movement_action < self.destinations.shape[0]:
            return int(self.destinations[legacy_movement_action, 1])
        return self.player_y

    def destination_for_action(self, legacy_movement_action: int) -> tuple[int, int]:
        return (
            self.destination_x_for_action(legacy_movement_action),
            self.destination_y_for_action(legacy_movement_action),
        )

    def directional_destination_x(self, direction_index: int) -> int:
        return self.destination_x_for_action(DIRECTIONAL_MOVE_ACTIONS[direction_index])

    def directional_destination_y(self, direction_index: int) -> int:
        return self.destination_y_for_action(DIRECTIONAL_MOVE_ACTIONS[direction_index])

    def directional_destination(self, direction_index: int) -> tuple[int, int]:
        return (
            self.directional_destination_x(direction_index),
            self.directional_destination_y(direction_index),
        )


@dataclass(frozen=True)
class RawTileThreatSummary:
    settled_x: int
    settled_y: int
    settled_step_distance: int
    predicted_prayer: str | None
    los_count: int
    min_attack_delay: float
    imminent_magic: int
    imminent_ranged: int
    imminent_melee: int
    total_imminent: int
    unprotected_after_auto_prayer: int
    blob_scan_triggers: int
    blocked_move: bool = False
    priority_target_attackable: bool = False


@dataclass(frozen=True)
class TickThreatCache:
    movement_table: MovementResolutionTable
    neighborhood_summaries: list[RawTileThreatSummary]
    npcs_with_los_now: int
    current_imminent_attacks: int
    type_alive_counts: dict[InfernoEntityType, int]


@dataclass
class ForecastEntityState:
    entity_type: InfernoEntityType
    x: int
    y: int
    attack_delay: int
    stunned: int
    frozen: int
    scanned_prayer: str | None
    had_los: bool


_MAX_NEIGHBORHOOD_LOS_COUNT = 9.0
_MAX_NEIGHBORHOOD_STYLE_COUNT = 5.0
_MAX_NEIGHBORHOOD_UNPROTECTED_COUNT = 6.0
_MAX_NEIGHBORHOOD_BLOB_SCAN_TRIGGERS = 2.0
_MAX_NEIGHBORHOOD_ATTACK_DELAY = 8.0
_ARENA_WIDTH = 29
_ARENA_HEIGHT = 30
_NUM_PILLAR_MASKS = 8
_NUM_MOVEMENT_ACTIONS = 33
_PRECOMPUTED_MOVEMENT_DESTINATIONS: np.ndarray | None = None
_ALL_FORECAST_ENTITY_TYPES = tuple(EntityTypes.all_types())
_FORECAST_ENTITY_TYPE_TO_ID = {
    entity_type: index for index, entity_type in enumerate(_ALL_FORECAST_ENTITY_TYPES)
}
_SCANNED_PRAYER_TO_ID = {None: 0, "MAGIC": 1, "RANGED": 2}
_SCANNED_PRAYER_FROM_ID = {value: key for key, value in _SCANNED_PRAYER_TO_ID.items()}
_PROTECT_PRAYER_TO_ID = {
    None: 0,
    "PROTECT_FROM_MAGIC": 1,
    "PROTECT_FROM_MISSILES": 2,
    "PROTECT_FROM_MELEE": 3,
}
_PROTECT_PRAYER_FROM_ID = {
    value: key for key, value in _PROTECT_PRAYER_TO_ID.items()
}


def _pillar_alive_to_mask(pillar_alive: list[bool]) -> int:
    return (
        int(bool(pillar_alive[0]))
        | (int(bool(pillar_alive[1])) << 1)
        | (int(bool(pillar_alive[2])) << 2)
    )


def _pillar_mask_to_alive(mask: int) -> list[bool]:
    return [
        bool(mask & 0b001),
        bool(mask & 0b010),
        bool(mask & 0b100),
    ]


def _build_precomputed_movement_destinations() -> np.ndarray:
    destinations = np.zeros(
        (
            _NUM_PILLAR_MASKS,
            _ARENA_WIDTH,
            _ARENA_HEIGHT,
            _NUM_MOVEMENT_ACTIONS,
            2,
        ),
        dtype=np.int16,
    )
    for pillar_mask in range(_NUM_PILLAR_MASKS):
        pillar_alive = _pillar_mask_to_alive(pillar_mask)

        def checker(x: int, y: int, size: int) -> bool:
            return size == 1 and SimulatorGeometry.is_valid_tile(x, y, pillar_alive)

        for player_x in range(_ARENA_WIDTH):
            for player_y in range(_ARENA_HEIGHT):
                destinations[pillar_mask, player_x, player_y, 0, 0] = player_x
                destinations[pillar_mask, player_x, player_y, 0, 1] = player_y
                for legacy_action in iter_legacy_movement_actions():
                    dx, dy, distance = get_movement_params(legacy_action)
                    target_x = max(0, min(_ARENA_WIDTH - 1, player_x + dx * distance))
                    target_y = max(0, min(_ARENA_HEIGHT - 1, player_y + dy * distance))
                    settled_x, settled_y = OSRSPathfinding.simulate_player_movement(
                        player_x,
                        player_y,
                        target_x,
                        target_y,
                        2,
                        checker,
                    )
                    destinations[pillar_mask, player_x, player_y, legacy_action, 0] = settled_x
                    destinations[pillar_mask, player_x, player_y, legacy_action, 1] = settled_y
    return destinations


def _get_precomputed_movement_destinations() -> np.ndarray:
    global _PRECOMPUTED_MOVEMENT_DESTINATIONS
    if _PRECOMPUTED_MOVEMENT_DESTINATIONS is None:
        _PRECOMPUTED_MOVEMENT_DESTINATIONS = _build_precomputed_movement_destinations()
    return _PRECOMPUTED_MOVEMENT_DESTINATIONS


def build_movement_resolution_table(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    actions: tuple[int, ...] | None = None,
) -> MovementResolutionTable:
    """Resolve all legacy movement actions from a precomputed arena lookup."""
    del actions
    destinations = _get_precomputed_movement_destinations()[
        _pillar_alive_to_mask(pillar_alive),
        player_x,
        player_y,
    ]
    return MovementResolutionTable(
        player_x=player_x,
        player_y=player_y,
        destinations=destinations,
    )


def is_player_melee_adjacent_to_npc(
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
) -> bool:
    """Return True when the player is cardinally adjacent to any occupied NPC tile."""
    return _is_player_melee_adjacent_to_npc_state(
        entity.x,
        entity.y,
        entity.entity_type.size_in_tiles,
        player_x,
        player_y,
    )


def _is_player_melee_adjacent_to_npc_state(
    npc_x: int,
    npc_y: int,
    npc_size: int,
    player_x: int,
    player_y: int,
) -> bool:
    for dx in range(npc_size):
        for dy in range(npc_size):
            npc_tile_x = npc_x + dx
            npc_tile_y = npc_y + dy
            if player_x == npc_tile_x and player_y == npc_tile_y + 1:
                return True
            if player_x == npc_tile_x and player_y == npc_tile_y - 1:
                return True
            if player_x == npc_tile_x + 1 and player_y == npc_tile_y:
                return True
            if player_x == npc_tile_x - 1 and player_y == npc_tile_y:
                return True
    return False


def _entity_can_attack_player_from(
    entity_type: InfernoEntityType,
    npc_x: int,
    npc_y: int,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
) -> bool:
    attack_range = entity_type.attack_range
    npc_size = entity_type.size_in_tiles
    distance = InfernoLineOfSight.get_distance_from_npc(
        npc_x,
        npc_y,
        npc_size,
        player_x,
        player_y,
    )
    if distance > attack_range:
        return False
    return InfernoLineOfSight.npc_has_los_to_player(
        npc_x,
        npc_y,
        npc_size,
        player_x,
        player_y,
        attack_range,
        pillar_alive,
    )


def predict_npc_position(
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
) -> tuple[int, int]:
    """Predict where an NPC will stand after one tick of movement resolution."""
    future_frozen = max(0, entity.frozen - 1)
    return _predict_npc_position_after_decrement_values(
        entity,
        player_x,
        player_y,
        pillar_alive,
        future_frozen,
    )


def _predict_npc_position_after_decrement(
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
) -> tuple[int, int]:
    """Predict NPC movement when timer decrement has already been applied."""
    return _predict_npc_position_after_decrement_values(
        entity,
        player_x,
        player_y,
        pillar_alive,
        entity.frozen,
    )


def _predict_npc_position_after_decrement_values(
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    frozen_after_decrement: int,
) -> tuple[int, int]:
    if frozen_after_decrement > 0 or entity.is_dead():
        return (entity.x, entity.y)

    has_los = _entity_can_attack_player_from(
        entity.entity_type,
        entity.x,
        entity.y,
        player_x,
        player_y,
        pillar_alive,
    )
    if has_los:
        return (entity.x, entity.y)

    def checker(x: int, y: int, size: int) -> bool:
        return SimulatorGeometry.is_valid_tile_for_size(x, y, size, pillar_alive)

    return OSRSPathfinding.simulate_npc_movement(
        entity.x,
        entity.y,
        player_x,
        player_y,
        entity.entity_type.size_in_tiles,
        entity.entity_type.move_speed,
        checker,
    )


def compute_dig_pressure(
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
) -> float:
    """Expose melee dig readiness without leaking the RNG roll itself."""
    if entity.entity_type != EntityTypes.MELEE:
        return 0.0
    if entity.dig_sequence_time > 0:
        return 1.0
    has_los = InfernoLineOfSight.can_entity_attack_player(
        entity, player_x, player_y, pillar_alive,
    )
    if has_los or entity.attack_delay > DIG_TRIGGER_RANDOM_THRESHOLD:
        return 0.0
    window = DIG_TRIGGER_RANDOM_THRESHOLD - DIG_TRIGGER_GUARANTEED
    progress = DIG_TRIGGER_RANDOM_THRESHOLD - entity.attack_delay
    return float(max(0.0, min(progress / window, 1.0)))


def _deterministic_blob_scan(
    scanned_prayer: str | None,
    active_prayer: str | None,
) -> str | None:
    if scanned_prayer is not None:
        return scanned_prayer
    if active_prayer == "PROTECT_FROM_MAGIC":
        return "RANGED"
    if active_prayer == "PROTECT_FROM_MISSILES":
        return "MAGIC"
    return None


def _resolve_attack_style(
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    active_prayer: str | None,
) -> str | None:
    return _resolve_attack_style_for_state(
        entity.entity_type,
        entity.x,
        entity.y,
        entity.scanned_prayer,
        player_x,
        player_y,
        active_prayer,
    )


def _resolve_attack_style_for_state(
    entity_type: InfernoEntityType,
    npc_x: int,
    npc_y: int,
    scanned_prayer: str | None,
    player_x: int,
    player_y: int,
    active_prayer: str | None,
) -> str | None:
    # Adjacency melee is NOT predicted. In OSRS, adjacent non-melee NPCs have
    # a ~40% chance to melee, but this is unpredictable RNG. The optimal play is
    # to always pray against the primary style. Melee hits are unavoidable damage.
    attack_style = entity_type.attack_style
    if attack_style == AttackStyle.MAGIC:
        return "magic"
    if attack_style == AttackStyle.RANGED:
        return "ranged"
    if attack_style == AttackStyle.MELEE:
        return "melee"
    if attack_style == AttackStyle.MAGIC_RANGED:
        scan = _deterministic_blob_scan(scanned_prayer, active_prayer)
        if scan == "MAGIC":
            return "magic"
        if scan == "RANGED":
            return "ranged"
        return None
    return None


def _is_scanned_blob_imminent(entity: PlacedEntity) -> bool:
    return (
        entity.entity_type == EntityTypes.BLOB
        and entity.scanned_prayer is not None
        and entity.attack_delay <= 0
    )


def _is_imminent_with_los(entity: PlacedEntity) -> bool:
    if entity.entity_type == EntityTypes.BLOB:
        return _is_scanned_blob_imminent(entity)
    return entity.attack_delay <= 0 and entity.stunned <= 0


def summarize_current_threat_context(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
) -> tuple[int, int, dict[InfernoEntityType, int]]:
    npcs_with_los_now = 0
    current_imminent_attacks = 0
    type_alive_counts: dict[InfernoEntityType, int] = {}

    for entity in combat_entities:
        if entity.is_dead() or entity.entity_type == EntityTypes.NIBBLER:
            continue
        entity_type = entity.entity_type
        type_alive_counts[entity_type] = type_alive_counts.get(entity_type, 0) + 1
        if InfernoLineOfSight.can_entity_attack_player(
            entity,
            player_x,
            player_y,
            pillar_alive,
        ):
            npcs_with_los_now += 1
            if _is_imminent_with_los(entity):
                current_imminent_attacks += 1

    return npcs_with_los_now, current_imminent_attacks, type_alive_counts


def _marshal_forecast_entities(
    combat_entities: list[PlacedEntity],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    entity_count = len(combat_entities)
    entity_type_ids = np.zeros(entity_count, dtype=np.int16)
    xs = np.zeros(entity_count, dtype=np.int16)
    ys = np.zeros(entity_count, dtype=np.int16)
    attack_delays = np.zeros(entity_count, dtype=np.int16)
    stunned = np.zeros(entity_count, dtype=np.int16)
    frozen = np.zeros(entity_count, dtype=np.int16)
    scanned_prayers = np.zeros(entity_count, dtype=np.int8)
    had_los = np.zeros(entity_count, dtype=np.bool_)

    for index, entity in enumerate(combat_entities):
        entity_type_ids[index] = _FORECAST_ENTITY_TYPE_TO_ID[entity.entity_type]
        xs[index] = entity.x
        ys[index] = entity.y
        attack_delays[index] = entity.attack_delay
        stunned[index] = entity.stunned
        frozen[index] = entity.frozen
        scanned_prayers[index] = _SCANNED_PRAYER_TO_ID[entity.scanned_prayer]
        had_los[index] = entity.had_los

    return (
        entity_type_ids,
        xs,
        ys,
        attack_delays,
        stunned,
        frozen,
        scanned_prayers,
        had_los,
    )


def _build_neighborhood_settled_tiles(
    player_x: int,
    player_y: int,
    movement_table: MovementResolutionTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    settled_xs = np.zeros(len(PLAYER_MOVE_DIRECTIONS) + 1, dtype=np.int16)
    settled_ys = np.zeros(len(PLAYER_MOVE_DIRECTIONS) + 1, dtype=np.int16)
    settled_distances = np.zeros(len(PLAYER_MOVE_DIRECTIONS) + 1, dtype=np.int16)
    blocked_moves = np.zeros(len(PLAYER_MOVE_DIRECTIONS) + 1, dtype=np.bool_)
    settled_xs[0] = player_x
    settled_ys[0] = player_y

    for direction_index, _ in enumerate(PLAYER_MOVE_DIRECTIONS, start=1):
        settled_x = movement_table.directional_destination_x(direction_index - 1)
        settled_y = movement_table.directional_destination_y(direction_index - 1)
        settled_distance = max(abs(settled_x - player_x), abs(settled_y - player_y))
        settled_xs[direction_index] = settled_x
        settled_ys[direction_index] = settled_y
        settled_distances[direction_index] = settled_distance
        blocked_moves[direction_index] = settled_distance <= 0

    return settled_xs, settled_ys, settled_distances, blocked_moves


def _raw_summary_from_backend_row(row: tuple) -> RawTileThreatSummary:
    predicted_prayer_id = int(row[3])
    return RawTileThreatSummary(
        settled_x=int(row[0]),
        settled_y=int(row[1]),
        settled_step_distance=int(row[2]),
        predicted_prayer=_PROTECT_PRAYER_FROM_ID[predicted_prayer_id],
        los_count=int(row[4]),
        min_attack_delay=float(row[5]),
        imminent_magic=int(row[6]),
        imminent_ranged=int(row[7]),
        imminent_melee=int(row[8]),
        total_imminent=int(row[9]),
        unprotected_after_auto_prayer=int(row[10]),
        blob_scan_triggers=int(row[11]),
        blocked_move=bool(row[12]),
    )


def _build_forecast_entity_state(entity: PlacedEntity) -> ForecastEntityState:
    return ForecastEntityState(
        entity_type=entity.entity_type,
        x=entity.x,
        y=entity.y,
        attack_delay=entity.attack_delay,
        stunned=entity.stunned,
        frozen=entity.frozen,
        scanned_prayer=entity.scanned_prayer,
        had_los=entity.had_los,
    )


def _advance_forecast_state(
    entity: ForecastEntityState,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    active_prayer: str | None,
) -> str | None:
    entity.attack_delay -= 1
    if entity.stunned > 0:
        entity.stunned -= 1
    if entity.frozen > 0:
        entity.frozen -= 1

    predicted_x, predicted_y = _predict_forecast_state_position(
        entity,
        player_x,
        player_y,
        pillar_alive,
    )
    entity.x = predicted_x
    entity.y = predicted_y

    if entity.entity_type == EntityTypes.BLOB:
        has_los = _entity_can_attack_player_from(
            entity.entity_type,
            predicted_x,
            predicted_y,
            player_x,
            player_y,
            pillar_alive,
        )
        just_gained_los = has_los and not entity.had_los
        scan_ready = has_los and entity.scanned_prayer is None and entity.attack_delay <= 0
        if just_gained_los or scan_ready:
            entity.scanned_prayer = _deterministic_blob_scan(None, active_prayer)
            entity.attack_delay = entity.entity_type.attack_speed
            entity.had_los = has_los
            return None
        entity.had_los = has_los
        if entity.scanned_prayer is None or entity.attack_delay > 0:
            return None
        style = _resolve_attack_style_for_state(
            entity.entity_type,
            predicted_x,
            predicted_y,
            entity.scanned_prayer,
            player_x,
            player_y,
            active_prayer,
        )
        if style is None:
            return None
        entity.attack_delay = entity.entity_type.attack_speed
        entity.scanned_prayer = None
        return style

    can_attack = entity.attack_delay <= 0 and entity.stunned <= 0
    if not can_attack:
        return None

    has_los = _entity_can_attack_player_from(
        entity.entity_type,
        predicted_x,
        predicted_y,
        player_x,
        player_y,
        pillar_alive,
    )
    if not has_los and not _is_player_melee_adjacent_to_npc_state(
        predicted_x,
        predicted_y,
        entity.entity_type.size_in_tiles,
        player_x,
        player_y,
    ):
        return None

    style = _resolve_attack_style_for_state(
        entity.entity_type,
        predicted_x,
        predicted_y,
        entity.scanned_prayer,
        player_x,
        player_y,
        active_prayer,
    )
    if style is None:
        return None
    entity.attack_delay = entity.entity_type.attack_speed
    return style


def _predict_forecast_state_position(
    entity: ForecastEntityState,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
) -> tuple[int, int]:
    if entity.frozen > 0:
        return (entity.x, entity.y)

    has_los = _entity_can_attack_player_from(
        entity.entity_type,
        entity.x,
        entity.y,
        player_x,
        player_y,
        pillar_alive,
    )
    if has_los:
        return (entity.x, entity.y)

    def checker(x: int, y: int, size: int) -> bool:
        return SimulatorGeometry.is_valid_tile_for_size(x, y, size, pillar_alive)

    return OSRSPathfinding.simulate_npc_movement(
        entity.x,
        entity.y,
        player_x,
        player_y,
        entity.entity_type.size_in_tiles,
        entity.entity_type.move_speed,
        checker,
    )


def _advance_forecast_entity(
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    active_prayer: str | None,
) -> str | None:
    state = _build_forecast_entity_state(entity)
    return _advance_forecast_state(state, player_x, player_y, pillar_alive, active_prayer)


def _style_to_prayer(style: str | None) -> str | None:
    if style == "magic":
        return "PROTECT_FROM_MAGIC"
    if style == "ranged":
        return "PROTECT_FROM_MISSILES"
    if style == "melee":
        return "PROTECT_FROM_MELEE"
    return None


def _blocked_raw_tile_summary(settled_x: int, settled_y: int) -> RawTileThreatSummary:
    return RawTileThreatSummary(
        settled_x=settled_x,
        settled_y=settled_y,
        settled_step_distance=0,
        predicted_prayer=None,
        los_count=0,
        min_attack_delay=0.0,
        imminent_magic=0,
        imminent_ranged=0,
        imminent_melee=0,
        total_imminent=0,
        unprotected_after_auto_prayer=0,
        blob_scan_triggers=0,
        blocked_move=True,
    )


def _summarize_raw_tile_threats(
    settled_x: int,
    settled_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    active_prayer: str | None,
    settled_step_distance: int = 0,
    blocked_move: bool = False,
) -> RawTileThreatSummary:
    if blocked_move:
        return _blocked_raw_tile_summary(settled_x, settled_y)

    los_count = 0
    min_attack_delay = float("inf")
    imminent_magic = 0
    imminent_ranged = 0
    imminent_melee = 0
    blob_scan_triggers = 0
    total_imminent = 0
    dangerous_style: str | None = None
    highest_priority = float("inf")

    for entity in combat_entities:
        if entity.is_dead() or entity.entity_type == EntityTypes.NIBBLER:
            continue

        future_attack_delay = entity.attack_delay - 1
        future_stunned = entity.stunned - 1 if entity.stunned > 0 else entity.stunned
        future_frozen = entity.frozen - 1 if entity.frozen > 0 else entity.frozen

        predicted_x, predicted_y = _predict_npc_position_after_decrement_values(
            entity,
            settled_x,
            settled_y,
            pillar_alive,
            future_frozen,
        )
        has_los = _entity_can_attack_player_from(
            entity.entity_type,
            predicted_x,
            predicted_y,
            settled_x,
            settled_y,
            pillar_alive,
        )
        if has_los:
            los_count += 1
            if future_attack_delay < min_attack_delay:
                min_attack_delay = future_attack_delay

        if entity.entity_type == EntityTypes.BLOB:
            just_gained_los = has_los and not entity.had_los
            scan_ready = has_los and entity.scanned_prayer is None and future_attack_delay <= 0
            if just_gained_los or scan_ready:
                blob_scan_triggers += 1

        style: str | None = None
        if entity.entity_type == EntityTypes.BLOB:
            if entity.scanned_prayer is not None and future_attack_delay <= 0:
                style = _resolve_attack_style_for_state(
                    entity.entity_type,
                    predicted_x,
                    predicted_y,
                    entity.scanned_prayer,
                    settled_x,
                    settled_y,
                    active_prayer,
                )
        elif future_attack_delay <= 0 and future_stunned <= 0:
            if has_los or _is_player_melee_adjacent_to_npc_state(
                predicted_x,
                predicted_y,
                entity.entity_type.size_in_tiles,
                settled_x,
                settled_y,
            ):
                style = _resolve_attack_style_for_state(
                    entity.entity_type,
                    predicted_x,
                    predicted_y,
                    entity.scanned_prayer,
                    settled_x,
                    settled_y,
                    active_prayer,
                )

        if style is None:
            continue

        total_imminent += 1
        if style == "magic":
            imminent_magic += 1
        elif style == "ranged":
            imminent_ranged += 1
        elif style == "melee":
            imminent_melee += 1

        priority = entity.entity_type.base_priority
        if priority < highest_priority:
            highest_priority = priority
            dangerous_style = style

    predicted_prayer = _style_to_prayer(dangerous_style)
    protected_count = 0
    if predicted_prayer == "PROTECT_FROM_MAGIC":
        protected_count = imminent_magic
    elif predicted_prayer == "PROTECT_FROM_MISSILES":
        protected_count = imminent_ranged
    elif predicted_prayer == "PROTECT_FROM_MELEE":
        protected_count = imminent_melee

    return RawTileThreatSummary(
        settled_x=settled_x,
        settled_y=settled_y,
        settled_step_distance=settled_step_distance,
        predicted_prayer=predicted_prayer,
        los_count=los_count,
        min_attack_delay=min_attack_delay,
        imminent_magic=imminent_magic,
        imminent_ranged=imminent_ranged,
        imminent_melee=imminent_melee,
        total_imminent=total_imminent,
        unprotected_after_auto_prayer=max(0, total_imminent - protected_count),
        blob_scan_triggers=blob_scan_triggers,
        blocked_move=False,
    )


def predict_auto_prayer_for_position(
    entities: list[PlacedEntity],
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    active_prayer: str | None,
) -> str | None:
    """Mirror the simulator's one-tick prayer predictor for an arbitrary tile."""
    return _summarize_raw_tile_threats(
        player_x,
        player_y,
        pillar_alive,
        entities,
        active_prayer,
    ).predicted_prayer


def _summarize_next_tick_tile_threats(
    settled_x: int,
    settled_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    active_prayer: str | None,
) -> tuple[int, float, int, int, int, int, int]:
    """Return raw next-tick tile threat counts for observation features."""
    summary = _summarize_raw_tile_threats(
        settled_x,
        settled_y,
        pillar_alive,
        combat_entities,
        active_prayer,
    )
    return (
        summary.los_count,
        summary.min_attack_delay,
        summary.imminent_magic,
        summary.imminent_ranged,
        summary.imminent_melee,
        summary.unprotected_after_auto_prayer,
        summary.blob_scan_triggers,
    )


def _build_neighborhood_tile_threat_summaries_python(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    active_prayer: str | None,
    movement_table: MovementResolutionTable | None = None,
) -> list[RawTileThreatSummary]:
    if movement_table is None:
        movement_table = build_movement_resolution_table(
            player_x,
            player_y,
            pillar_alive,
            actions=DIRECTIONAL_MOVE_ACTIONS,
        )

    summaries = [
        _summarize_raw_tile_threats(
            player_x,
            player_y,
            pillar_alive,
            combat_entities,
            active_prayer,
            settled_step_distance=0,
        )
    ]

    for direction_index, _ in enumerate(PLAYER_MOVE_DIRECTIONS):
        settled_x = movement_table.directional_destination_x(direction_index)
        settled_y = movement_table.directional_destination_y(direction_index)
        settled_distance = max(abs(settled_x - player_x), abs(settled_y - player_y))
        if settled_distance <= 0:
            summaries.append(_blocked_raw_tile_summary(settled_x, settled_y))
            continue
        summaries.append(
            _summarize_raw_tile_threats(
                settled_x,
                settled_y,
                pillar_alive,
                combat_entities,
                active_prayer,
                settled_step_distance=settled_distance,
            )
        )

    return summaries


def build_neighborhood_tile_threat_summaries(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    active_prayer: str | None,
    movement_table: MovementResolutionTable | None = None,
) -> list[RawTileThreatSummary]:
    if movement_table is None:
        movement_table = build_movement_resolution_table(
            player_x,
            player_y,
            pillar_alive,
            actions=DIRECTIONAL_MOVE_ACTIONS,
        )

    if _forecast_fast_backend is None:
        return _build_neighborhood_tile_threat_summaries_python(
            player_x,
            player_y,
            pillar_alive,
            combat_entities,
            active_prayer,
            movement_table=movement_table,
        )

    settled_xs, settled_ys, settled_distances, blocked_moves = _build_neighborhood_settled_tiles(
        player_x,
        player_y,
        movement_table,
    )
    (
        entity_type_ids,
        xs,
        ys,
        attack_delays,
        stunned,
        frozen,
        scanned_prayers,
        had_los,
    ) = _marshal_forecast_entities(combat_entities)
    raw_rows = _forecast_fast_backend.build_neighborhood_tile_threat_summaries(
        settled_xs,
        settled_ys,
        settled_distances,
        blocked_moves,
        _pillar_alive_to_mask(pillar_alive),
        entity_type_ids,
        xs,
        ys,
        attack_delays,
        stunned,
        frozen,
        scanned_prayers,
        had_los,
        _PROTECT_PRAYER_TO_ID[active_prayer],
    )
    return [_raw_summary_from_backend_row(tuple(row)) for row in raw_rows]


def build_directional_tile_threat_summaries(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    active_prayer: str | None,
    movement_table: MovementResolutionTable | None = None,
) -> list[RawTileThreatSummary]:
    return build_neighborhood_tile_threat_summaries(
        player_x,
        player_y,
        pillar_alive,
        combat_entities,
        active_prayer,
        movement_table=movement_table,
    )[1:]


def _forecast_threat_styles_python(
    entities: list[PlacedEntity],
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    active_prayer: str | None,
    horizons: int = 3,
) -> list[ThreatStyleCounts]:
    """Forecast deterministic incoming attack styles for a stationary player."""
    forecast_entities = [
        _build_forecast_entity_state(entity)
        for entity in entities
        if not entity.is_dead()
    ]
    counts: list[ThreatStyleCounts] = []
    for _ in range(horizons):
        magic = 0
        ranged = 0
        melee = 0
        for entity in forecast_entities:
            if entity.entity_type == EntityTypes.NIBBLER:
                continue
            style = _advance_forecast_state(
                entity,
                player_x,
                player_y,
                pillar_alive,
                active_prayer,
            )
            if style == "magic":
                magic += 1
            elif style == "ranged":
                ranged += 1
            elif style == "melee":
                melee += 1
        counts.append(ThreatStyleCounts(magic=magic, ranged=ranged, melee=melee))
    return counts


def forecast_threat_styles(
    entities: list[PlacedEntity],
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    active_prayer: str | None,
    horizons: int = 3,
) -> list[ThreatStyleCounts]:
    if _forecast_fast_backend is None:
        return _forecast_threat_styles_python(
            entities,
            player_x,
            player_y,
            pillar_alive,
            active_prayer,
            horizons=horizons,
        )

    combat_entities = [entity for entity in entities if not entity.is_dead()]
    (
        entity_type_ids,
        xs,
        ys,
        attack_delays,
        stunned,
        frozen,
        scanned_prayers,
        had_los,
    ) = _marshal_forecast_entities(combat_entities)
    raw_counts = _forecast_fast_backend.forecast_threat_styles(
        entity_type_ids,
        xs,
        ys,
        attack_delays,
        stunned,
        frozen,
        scanned_prayers,
        had_los,
        player_x,
        player_y,
        _pillar_alive_to_mask(pillar_alive),
        _PROTECT_PRAYER_TO_ID[active_prayer],
        horizons,
    )
    return [
        ThreatStyleCounts(
            magic=int(magic),
            ranged=int(ranged),
            melee=int(melee),
        )
        for magic, ranged, melee in raw_counts
    ]


def forecast_directional_movement(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    movement_table: MovementResolutionTable | None = None,
) -> list[DirectionalForecast]:
    """Forecast one-tick settlement and danger for canonical 2-tile moves."""
    summaries = build_directional_tile_threat_summaries(
        player_x,
        player_y,
        pillar_alive,
        combat_entities,
        active_prayer=None,
        movement_table=movement_table,
    )
    return [
        DirectionalForecast(
            settled_step_distance=summary.settled_step_distance,
            los_count=summary.los_count,
            imminent_attacks=summary.total_imminent,
        )
        for summary in summaries
    ]


def forecast_blob_scan_triggers(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    movement_table: MovementResolutionTable | None = None,
) -> list[int]:
    """Count blob scan triggers for the 8 canonical 2-tile movement directions."""
    summaries = build_directional_tile_threat_summaries(
        player_x,
        player_y,
        pillar_alive,
        combat_entities,
        active_prayer=None,
        movement_table=movement_table,
    )
    return [summary.blob_scan_triggers for summary in summaries]


def build_tick_threat_cache(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    active_prayer: str | None,
    npcs_with_los_now: int | None = None,
    current_imminent_attacks: int | None = None,
    type_alive_counts: dict[InfernoEntityType, int] | None = None,
) -> TickThreatCache:
    if (
        npcs_with_los_now is None
        or current_imminent_attacks is None
        or type_alive_counts is None
    ):
        (
            computed_los_now,
            computed_imminent_attacks,
            computed_type_alive_counts,
        ) = summarize_current_threat_context(
            player_x,
            player_y,
            pillar_alive,
            combat_entities,
        )
        if npcs_with_los_now is None:
            npcs_with_los_now = computed_los_now
        if current_imminent_attacks is None:
            current_imminent_attacks = computed_imminent_attacks
        if type_alive_counts is None:
            type_alive_counts = computed_type_alive_counts

    movement_table = build_movement_resolution_table(
        player_x,
        player_y,
        pillar_alive,
    )
    neighborhood_summaries = build_neighborhood_tile_threat_summaries(
        player_x,
        player_y,
        pillar_alive,
        combat_entities,
        active_prayer,
        movement_table=movement_table,
    )
    return TickThreatCache(
        movement_table=movement_table,
        neighborhood_summaries=neighborhood_summaries,
        npcs_with_los_now=npcs_with_los_now,
        current_imminent_attacks=current_imminent_attacks,
        type_alive_counts=dict(type_alive_counts),
    )


def _bfs_multistep_forecast(
    origin_x: int,
    origin_y: int,
    pillar_alive: list[bool],
    dangerous_entities: list[PlacedEntity],
    priority_target: PlacedEntity | None,
    attack_range: int,
    los_cache: dict[tuple[int, int, int], bool],
    max_depth: int = 2,
) -> tuple[float, float]:
    """BFS from origin to find best LOS position within max_depth steps.

    Returns (best_los_normalized, steps_to_single_los_normalized).
    """
    n_dangerous = len(dangerous_entities)
    if n_dangerous == 0:
        return 0.0, 0.0

    from .pathfinding import OSRS_BFS_DIRECTIONS

    best_los = n_dangerous
    steps_to_single = max_depth + 1  # sentinel: not found

    visited: dict[tuple[int, int], int] = {(origin_x, origin_y): 0}
    queue: deque[tuple[int, int, int]] = deque()
    queue.append((origin_x, origin_y, 0))

    while queue:
        x, y, depth = queue.popleft()

        # Count dangerous NPCs with LOS to this tile
        los_count = 0
        for i, entity in enumerate(dangerous_entities):
            key = (x, y, i)
            if key not in los_cache:
                los_cache[key] = InfernoLineOfSight.npc_has_los_to_player(
                    entity.x, entity.y, entity.entity_type.size_in_tiles,
                    x, y, entity.entity_type.attack_range, pillar_alive,
                )
            if los_cache[key]:
                los_count += 1

        best_los = min(best_los, los_count)

        if los_count == 1 and depth < steps_to_single:
            if priority_target is None or priority_target.is_dead():
                steps_to_single = depth
            elif InfernoLineOfSight.can_player_attack_entity(
                x, y, attack_range, priority_target, pillar_alive,
            ):
                steps_to_single = depth

        if best_los == 0 and steps_to_single == 0:
            break  # can't improve

        if depth >= max_depth:
            continue

        for dx, dy in OSRS_BFS_DIRECTIONS:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if dx != 0 and dy != 0:
                # Diagonal: check destination + both cardinal intermediates
                if not SimulatorGeometry.is_valid_tile(nx, ny, pillar_alive):
                    continue
                if not SimulatorGeometry.is_valid_tile(x + dx, y, pillar_alive):
                    continue
                if not SimulatorGeometry.is_valid_tile(x, y + dy, pillar_alive):
                    continue
            else:
                if not SimulatorGeometry.is_valid_tile(nx, ny, pillar_alive):
                    continue
            visited[(nx, ny)] = depth + 1
            queue.append((nx, ny, depth + 1))

    best_los_norm = best_los / n_dangerous
    steps_norm = min(steps_to_single, 3) / 3.0
    return best_los_norm, steps_norm


def forecast_neighborhood_safety(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
    current_los_count: int,
    active_prayer: str | None,
    movement_table: MovementResolutionTable | None = None,
    raw_summaries: list[RawTileThreatSummary] | None = None,
    priority_target: PlacedEntity | None = None,
    attack_range: int = 0,
) -> list[NeighborhoodForecast]:
    """Forecast safety features for current tile + 8 directional 2-tile moves."""
    if raw_summaries is None:
        raw_summaries = build_neighborhood_tile_threat_summaries(
            player_x,
            player_y,
            pillar_alive,
            combat_entities,
            active_prayer,
            movement_table=movement_table,
        )

    # Filter dangerous entities and create shared LOS cache for BFS
    dangerous_entities = [
        e for e in combat_entities
        if not e.is_dead() and e.entity_type in _BFS_DANGEROUS_TYPES
    ]
    los_cache: dict[tuple[int, int, int], bool] = {}

    forecasts: list[NeighborhoodForecast] = []
    for summary in raw_summaries:
        if summary.blocked_move:
            forecasts.append(NeighborhoodForecast(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ))
            continue

        target_attackable = 0.0
        if priority_target is not None and not priority_target.is_dead():
            if InfernoLineOfSight.can_player_attack_entity(
                summary.settled_x, summary.settled_y, attack_range,
                priority_target, pillar_alive,
            ):
                target_attackable = 1.0

        bfs_best, bfs_steps = _bfs_multistep_forecast(
            summary.settled_x, summary.settled_y,
            pillar_alive, dangerous_entities,
            priority_target, attack_range, los_cache,
        )

        los_delta = summary.los_count - current_los_count
        forecasts.append(NeighborhoodForecast(
            settled_step_distance=min(summary.settled_step_distance / 2.0, 1.0),
            los_count=min(summary.los_count / _MAX_NEIGHBORHOOD_LOS_COUNT, 1.0),
            los_delta=max(-1.0, min(los_delta / _MAX_NEIGHBORHOOD_LOS_COUNT, 1.0)),
            min_attack_delay=(
                1.0 if summary.min_attack_delay == float("inf")
                else max(-1.0, min(summary.min_attack_delay / _MAX_NEIGHBORHOOD_ATTACK_DELAY, 1.0))
            ),
            imminent_magic=min(summary.imminent_magic / _MAX_NEIGHBORHOOD_STYLE_COUNT, 1.0),
            imminent_ranged=min(summary.imminent_ranged / _MAX_NEIGHBORHOOD_STYLE_COUNT, 1.0),
            imminent_melee=min(summary.imminent_melee / _MAX_NEIGHBORHOOD_STYLE_COUNT, 1.0),
            unprotected_after_auto_prayer=min(
                summary.unprotected_after_auto_prayer / _MAX_NEIGHBORHOOD_UNPROTECTED_COUNT,
                1.0,
            ),
            blob_scan_triggers=min(
                summary.blob_scan_triggers / _MAX_NEIGHBORHOOD_BLOB_SCAN_TRIGGERS,
                1.0,
            ),
            priority_target_attackable=target_attackable,
            best_los_in_2_steps=bfs_best,
            steps_to_single_los=bfs_steps,
        ))

    return forecasts
