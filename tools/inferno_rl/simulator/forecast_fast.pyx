# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
"""
Compiled forecast helper for Inferno RL.

This module keeps the public API in forecast.py unchanged. It operates on
primitive arrays so forecast.py can marshal PlacedEntity objects once per call
and fall back to the Python implementation when this extension is unavailable.
"""

cimport numpy as np
import numpy as np

from tools.inferno_rl.simulator.entity import AttackStyle, EntityTypes
from tools.inferno_rl.simulator.geometry import InfernoLineOfSight, SimulatorGeometry
from tools.inferno_rl.simulator.pathfinding import OSRSPathfinding


cdef int STYLE_NONE = 0
cdef int STYLE_MAGIC = 1
cdef int STYLE_RANGED = 2
cdef int STYLE_MELEE = 3

cdef int SCANNED_NONE = 0
cdef int SCANNED_MAGIC = 1
cdef int SCANNED_RANGED = 2

cdef int PRAYER_NONE = 0
cdef int PRAYER_MAGIC = 1
cdef int PRAYER_RANGED = 2
cdef int PRAYER_MELEE = 3

cdef int ATTACK_STYLE_NONE = 0
cdef int ATTACK_STYLE_MAGIC = 1
cdef int ATTACK_STYLE_RANGED = 2
cdef int ATTACK_STYLE_MELEE = 3
cdef int ATTACK_STYLE_MAGIC_RANGED = 4
cdef int ATTACK_STYLE_MAGIC_RANGED_MELEE = 5

_ENTITY_TYPES = tuple(EntityTypes.all_types())
_ENTITY_ATTACK_RANGE = tuple(int(entity_type.attack_range) for entity_type in _ENTITY_TYPES)
_ENTITY_SIZE = tuple(int(entity_type.size_in_tiles) for entity_type in _ENTITY_TYPES)
_ENTITY_ATTACK_SPEED = tuple(int(entity_type.attack_speed) for entity_type in _ENTITY_TYPES)
_ENTITY_MOVE_SPEED = tuple(int(entity_type.move_speed) for entity_type in _ENTITY_TYPES)
_ENTITY_BASE_PRIORITY = tuple(int(entity_type.base_priority) for entity_type in _ENTITY_TYPES)
_ENTITY_ATTACK_STYLE = tuple(
    ATTACK_STYLE_MAGIC if entity_type.attack_style == AttackStyle.MAGIC else
    ATTACK_STYLE_RANGED if entity_type.attack_style == AttackStyle.RANGED else
    ATTACK_STYLE_MELEE if entity_type.attack_style == AttackStyle.MELEE else
    ATTACK_STYLE_MAGIC_RANGED if entity_type.attack_style == AttackStyle.MAGIC_RANGED else
    ATTACK_STYLE_MAGIC_RANGED_MELEE
    for entity_type in _ENTITY_TYPES
)
_NIBBLER_TYPE_ID = _ENTITY_TYPES.index(EntityTypes.NIBBLER)
_BLOB_TYPE_ID = _ENTITY_TYPES.index(EntityTypes.BLOB)


def _mask_to_pillar_alive(int pillar_mask):
    return [
        bool(pillar_mask & 0b001),
        bool(pillar_mask & 0b010),
        bool(pillar_mask & 0b100),
    ]


cdef inline bint _is_player_melee_adjacent_to_npc_state(
    int npc_x,
    int npc_y,
    int npc_size,
    int player_x,
    int player_y,
):
    cdef int dx
    cdef int dy
    cdef int npc_tile_x
    cdef int npc_tile_y

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


cdef inline int _deterministic_blob_scan(int scanned_prayer_id, int active_prayer_id):
    if scanned_prayer_id != SCANNED_NONE:
        return scanned_prayer_id
    if active_prayer_id == PRAYER_MAGIC:
        return SCANNED_RANGED
    if active_prayer_id == PRAYER_RANGED:
        return SCANNED_MAGIC
    return SCANNED_NONE


cdef inline int _style_to_prayer_id(int style_id):
    if style_id == STYLE_MAGIC:
        return PRAYER_MAGIC
    if style_id == STYLE_RANGED:
        return PRAYER_RANGED
    if style_id == STYLE_MELEE:
        return PRAYER_MELEE
    return PRAYER_NONE


def _entity_can_attack_player_from(
    int entity_type_id,
    int npc_x,
    int npc_y,
    int player_x,
    int player_y,
    pillar_alive,
):
    attack_range = _ENTITY_ATTACK_RANGE[entity_type_id]
    npc_size = _ENTITY_SIZE[entity_type_id]
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


def _predict_npc_position_after_decrement_values(
    int entity_type_id,
    int npc_x,
    int npc_y,
    int player_x,
    int player_y,
    pillar_alive,
    int frozen_after_decrement,
):
    if frozen_after_decrement > 0:
        return npc_x, npc_y

    has_los = _entity_can_attack_player_from(
        entity_type_id,
        npc_x,
        npc_y,
        player_x,
        player_y,
        pillar_alive,
    )
    if has_los:
        return npc_x, npc_y

    npc_size = _ENTITY_SIZE[entity_type_id]
    move_speed = _ENTITY_MOVE_SPEED[entity_type_id]

    def checker(x: int, y: int, size: int) -> bool:
        return SimulatorGeometry.is_valid_tile_for_size(x, y, size, pillar_alive)

    return OSRSPathfinding.simulate_npc_movement(
        npc_x,
        npc_y,
        player_x,
        player_y,
        npc_size,
        move_speed,
        checker,
    )


def _predict_forecast_state_position(
    int entity_type_id,
    int npc_x,
    int npc_y,
    int player_x,
    int player_y,
    pillar_alive,
    int frozen,
):
    if frozen > 0:
        return npc_x, npc_y

    has_los = _entity_can_attack_player_from(
        entity_type_id,
        npc_x,
        npc_y,
        player_x,
        player_y,
        pillar_alive,
    )
    if has_los:
        return npc_x, npc_y

    npc_size = _ENTITY_SIZE[entity_type_id]
    move_speed = _ENTITY_MOVE_SPEED[entity_type_id]

    def checker(x: int, y: int, size: int) -> bool:
        return SimulatorGeometry.is_valid_tile_for_size(x, y, size, pillar_alive)

    return OSRSPathfinding.simulate_npc_movement(
        npc_x,
        npc_y,
        player_x,
        player_y,
        npc_size,
        move_speed,
        checker,
    )


cdef inline int _resolve_attack_style_for_state(
    int entity_type_id,
    int npc_x,
    int npc_y,
    int scanned_prayer_id,
    int player_x,
    int player_y,
    int active_prayer_id,
):
    # Adjacency melee is NOT predicted -- unpredictable RNG in OSRS.
    # Always predict primary style; melee hits are unavoidable damage.
    cdef int attack_style_id = _ENTITY_ATTACK_STYLE[entity_type_id]
    if attack_style_id == ATTACK_STYLE_MAGIC:
        return STYLE_MAGIC
    if attack_style_id == ATTACK_STYLE_RANGED:
        return STYLE_RANGED
    if attack_style_id == ATTACK_STYLE_MELEE:
        return STYLE_MELEE
    if attack_style_id == ATTACK_STYLE_MAGIC_RANGED:
        scanned_prayer_id = _deterministic_blob_scan(
            scanned_prayer_id,
            active_prayer_id,
        )
        if scanned_prayer_id == SCANNED_MAGIC:
            return STYLE_MAGIC
        if scanned_prayer_id == SCANNED_RANGED:
            return STYLE_RANGED
    return STYLE_NONE


def build_neighborhood_tile_threat_summaries(
    np.ndarray settled_xs,
    np.ndarray settled_ys,
    np.ndarray settled_distances,
    np.ndarray blocked_moves,
    int pillar_mask,
    np.ndarray entity_type_ids,
    np.ndarray xs,
    np.ndarray ys,
    np.ndarray attack_delays,
    np.ndarray stunned,
    np.ndarray frozen,
    np.ndarray scanned_prayers,
    np.ndarray had_los,
    int active_prayer_id,
):
    cdef int tile_count = settled_xs.shape[0]
    cdef int entity_count = entity_type_ids.shape[0]
    cdef int tile_index
    cdef int entity_index
    cdef int settled_x
    cdef int settled_y
    cdef int settled_distance
    cdef int entity_type_id
    cdef int future_attack_delay
    cdef int future_stunned
    cdef int future_frozen
    cdef int predicted_x
    cdef int predicted_y
    cdef int los_count
    cdef double min_attack_delay
    cdef int imminent_magic
    cdef int imminent_ranged
    cdef int imminent_melee
    cdef int blob_scan_triggers
    cdef int total_imminent
    cdef int dangerous_style
    cdef int highest_priority
    cdef int priority
    cdef int style_id
    cdef int predicted_prayer_id
    cdef int protected_count
    cdef bint has_los
    cdef bint blocked_move
    cdef bint just_gained_los
    cdef bint scan_ready

    pillar_alive = _mask_to_pillar_alive(pillar_mask)
    summaries = []

    for tile_index in range(tile_count):
        settled_x = int(settled_xs[tile_index])
        settled_y = int(settled_ys[tile_index])
        settled_distance = int(settled_distances[tile_index])
        blocked_move = bool(blocked_moves[tile_index])
        if blocked_move:
            summaries.append(
                (
                    settled_x,
                    settled_y,
                    0,
                    PRAYER_NONE,
                    0,
                    0.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    True,
                )
            )
            continue

        los_count = 0
        min_attack_delay = float("inf")
        imminent_magic = 0
        imminent_ranged = 0
        imminent_melee = 0
        blob_scan_triggers = 0
        total_imminent = 0
        dangerous_style = STYLE_NONE
        highest_priority = 10 ** 9

        for entity_index in range(entity_count):
            entity_type_id = int(entity_type_ids[entity_index])
            future_attack_delay = int(attack_delays[entity_index]) - 1
            future_stunned = (
                int(stunned[entity_index]) - 1
                if int(stunned[entity_index]) > 0
                else int(stunned[entity_index])
            )
            future_frozen = (
                int(frozen[entity_index]) - 1
                if int(frozen[entity_index]) > 0
                else int(frozen[entity_index])
            )

            predicted_x, predicted_y = _predict_npc_position_after_decrement_values(
                entity_type_id,
                int(xs[entity_index]),
                int(ys[entity_index]),
                settled_x,
                settled_y,
                pillar_alive,
                future_frozen,
            )
            has_los = _entity_can_attack_player_from(
                entity_type_id,
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

            if entity_type_id == _BLOB_TYPE_ID:
                just_gained_los = has_los and not bool(had_los[entity_index])
                scan_ready = (
                    has_los
                    and int(scanned_prayers[entity_index]) == SCANNED_NONE
                    and future_attack_delay <= 0
                )
                if just_gained_los or scan_ready:
                    blob_scan_triggers += 1

            style_id = STYLE_NONE
            if entity_type_id == _BLOB_TYPE_ID:
                if (
                    int(scanned_prayers[entity_index]) != SCANNED_NONE
                    and future_attack_delay <= 0
                ):
                    style_id = _resolve_attack_style_for_state(
                        entity_type_id,
                        predicted_x,
                        predicted_y,
                        int(scanned_prayers[entity_index]),
                        settled_x,
                        settled_y,
                        active_prayer_id,
                    )
            elif future_attack_delay <= 0 and future_stunned <= 0:
                if has_los or _is_player_melee_adjacent_to_npc_state(
                    predicted_x,
                    predicted_y,
                    _ENTITY_SIZE[entity_type_id],
                    settled_x,
                    settled_y,
                ):
                    style_id = _resolve_attack_style_for_state(
                        entity_type_id,
                        predicted_x,
                        predicted_y,
                        int(scanned_prayers[entity_index]),
                        settled_x,
                        settled_y,
                        active_prayer_id,
                    )

            if style_id == STYLE_NONE:
                continue

            total_imminent += 1
            if style_id == STYLE_MAGIC:
                imminent_magic += 1
            elif style_id == STYLE_RANGED:
                imminent_ranged += 1
            elif style_id == STYLE_MELEE:
                imminent_melee += 1

            priority = _ENTITY_BASE_PRIORITY[entity_type_id]
            if priority < highest_priority:
                highest_priority = priority
                dangerous_style = style_id

        predicted_prayer_id = _style_to_prayer_id(dangerous_style)
        protected_count = 0
        if predicted_prayer_id == PRAYER_MAGIC:
            protected_count = imminent_magic
        elif predicted_prayer_id == PRAYER_RANGED:
            protected_count = imminent_ranged
        elif predicted_prayer_id == PRAYER_MELEE:
            protected_count = imminent_melee

        summaries.append(
            (
                settled_x,
                settled_y,
                settled_distance,
                predicted_prayer_id,
                los_count,
                min_attack_delay,
                imminent_magic,
                imminent_ranged,
                imminent_melee,
                total_imminent,
                max(0, total_imminent - protected_count),
                blob_scan_triggers,
                False,
            )
        )

    return summaries


def forecast_threat_styles(
    np.ndarray entity_type_ids,
    np.ndarray xs,
    np.ndarray ys,
    np.ndarray attack_delays,
    np.ndarray stunned,
    np.ndarray frozen,
    np.ndarray scanned_prayers,
    np.ndarray had_los,
    int player_x,
    int player_y,
    int pillar_mask,
    int active_prayer_id,
    int horizons,
):
    cdef int entity_count = entity_type_ids.shape[0]
    cdef int horizon_index
    cdef int entity_index
    cdef int entity_type_id
    cdef int predicted_x
    cdef int predicted_y
    cdef int style_id
    cdef int magic
    cdef int ranged
    cdef int melee
    cdef bint has_los
    cdef bint just_gained_los
    cdef bint scan_ready

    pillar_alive = _mask_to_pillar_alive(pillar_mask)
    xs_local = np.array(xs, copy=True, dtype=np.int16)
    ys_local = np.array(ys, copy=True, dtype=np.int16)
    attack_delays_local = np.array(attack_delays, copy=True, dtype=np.int16)
    stunned_local = np.array(stunned, copy=True, dtype=np.int16)
    frozen_local = np.array(frozen, copy=True, dtype=np.int16)
    scanned_local = np.array(scanned_prayers, copy=True, dtype=np.int8)
    had_los_local = np.array(had_los, copy=True, dtype=np.bool_)

    counts = []
    for horizon_index in range(horizons):
        magic = 0
        ranged = 0
        melee = 0
        for entity_index in range(entity_count):
            entity_type_id = int(entity_type_ids[entity_index])
            if entity_type_id == _NIBBLER_TYPE_ID:
                continue

            attack_delays_local[entity_index] -= 1
            if stunned_local[entity_index] > 0:
                stunned_local[entity_index] -= 1
            if frozen_local[entity_index] > 0:
                frozen_local[entity_index] -= 1

            predicted_x, predicted_y = _predict_forecast_state_position(
                entity_type_id,
                int(xs_local[entity_index]),
                int(ys_local[entity_index]),
                player_x,
                player_y,
                pillar_alive,
                int(frozen_local[entity_index]),
            )
            xs_local[entity_index] = predicted_x
            ys_local[entity_index] = predicted_y

            if entity_type_id == _BLOB_TYPE_ID:
                has_los = _entity_can_attack_player_from(
                    entity_type_id,
                    predicted_x,
                    predicted_y,
                    player_x,
                    player_y,
                    pillar_alive,
                )
                just_gained_los = has_los and not bool(had_los_local[entity_index])
                scan_ready = (
                    has_los
                    and int(scanned_local[entity_index]) == SCANNED_NONE
                    and int(attack_delays_local[entity_index]) <= 0
                )
                if just_gained_los or scan_ready:
                    scanned_local[entity_index] = _deterministic_blob_scan(
                        SCANNED_NONE,
                        active_prayer_id,
                    )
                    attack_delays_local[entity_index] = _ENTITY_ATTACK_SPEED[entity_type_id]
                    had_los_local[entity_index] = has_los
                    continue
                had_los_local[entity_index] = has_los
                if (
                    int(scanned_local[entity_index]) == SCANNED_NONE
                    or int(attack_delays_local[entity_index]) > 0
                ):
                    continue
                style_id = _resolve_attack_style_for_state(
                    entity_type_id,
                    predicted_x,
                    predicted_y,
                    int(scanned_local[entity_index]),
                    player_x,
                    player_y,
                    active_prayer_id,
                )
                if style_id == STYLE_NONE:
                    continue
                attack_delays_local[entity_index] = _ENTITY_ATTACK_SPEED[entity_type_id]
                scanned_local[entity_index] = SCANNED_NONE
            else:
                if (
                    int(attack_delays_local[entity_index]) > 0
                    or int(stunned_local[entity_index]) > 0
                ):
                    continue
                has_los = _entity_can_attack_player_from(
                    entity_type_id,
                    predicted_x,
                    predicted_y,
                    player_x,
                    player_y,
                    pillar_alive,
                )
                if not has_los and not _is_player_melee_adjacent_to_npc_state(
                    predicted_x,
                    predicted_y,
                    _ENTITY_SIZE[entity_type_id],
                    player_x,
                    player_y,
                ):
                    continue
                style_id = _resolve_attack_style_for_state(
                    entity_type_id,
                    predicted_x,
                    predicted_y,
                    int(scanned_local[entity_index]),
                    player_x,
                    player_y,
                    active_prayer_id,
                )
                if style_id == STYLE_NONE:
                    continue
                attack_delays_local[entity_index] = _ENTITY_ATTACK_SPEED[entity_type_id]

            if style_id == STYLE_MAGIC:
                magic += 1
            elif style_id == STYLE_RANGED:
                ranged += 1
            elif style_id == STYLE_MELEE:
                melee += 1

        counts.append((magic, ranged, melee))

    return counts
