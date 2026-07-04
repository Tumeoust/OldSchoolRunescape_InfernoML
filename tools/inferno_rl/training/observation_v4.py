"""
Observation v4 builder.

Observation v4 keeps the prior global / temporal / forecast concepts, but the
entity block is now a flat exact-target section:
- 14 exact target slots shared with the simulator's attack resolution
- no support buckets
- no overflow block
"""

from __future__ import annotations

import math

import numpy as np

from ..simulator.entity import EntityTypes, InfernoEntityType, PlacedEntity
from ..simulator.equipment import GearPreset
from ..simulator.exact_targeting import (
    get_exact_target_slots,
    get_exact_target_type_index,
)
from ..simulator.forecast import (
    RawTileThreatSummary,
    TickThreatCache,
    compute_dig_pressure,
    forecast_neighborhood_safety,
    forecast_threat_styles,
)
from ..simulator.geometry import InfernoLineOfSight
from ..simulator.state import SimulatorState, WAVE_SPAWN_DELAY
from .observation_common import (
    EXACT_TARGET_SLOT_COUNT,
    EXACT_TARGET_SLOT_SIZE,
    EXACT_TARGET_SLOTS_TOTAL,
    GLOBAL_V4_SIZE,
    GRID_HEIGHT,
    GRID_WIDTH,
    LOADOUT_BLOCK_SIZE,
    MAX_ATTACK_COOLDOWN,
    MAX_BLOB_SCAN_COUNT,
    MAX_DEAD_POOL_COUNT,
    MAX_DISTANCE,
    MAX_ENTITY_COOLDOWN,
    MAX_ENTITY_FROZEN_TICKS,
    MAX_ENTITY_STUN_TICKS,
    MAX_HEALTH,
    MAX_MAGIC_ATTACK_BONUS,
    MAX_NIBBLER_COUNT,
    MAX_PLAYER_ATTACK_RANGE,
    MAX_RANGED_ATTACK_BONUS,
    MAX_RANGED_STRENGTH_BONUS,
    MAX_RESURRECTION_HAZARD,
    MAX_SAME_NPC_COUNT,
    MAX_TICK_IN_WAVE,
    MAX_WAVE,
    MAX_WEAPON_SPEED,
    NE_PILLAR_CENTER,
    NEIGHBORHOOD_FEATURES,
    NEIGHBORHOOD_FORECAST_SIZE,
    NUM_PRAYERS,
    NUM_WEAPON_TYPES,
    OBSERVATION_PUBLIC_SIZE,
    OBSERVATION_TOTAL_SIZE,
    SLOT_CORE_SIZE,
    TEMPORAL_V3_SIZE,
    THREAT_HORIZON_SIZE,
    _DEAD_POOL_TYPES,
    _GLOBAL_COUNT_TYPES,
    _encode_prayer_one_hot,
    _get_alive_entities,
    _normalize_signed,
    get_weapon_index,
)


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


def _encode_slot_core(
    obs: np.ndarray,
    offset: int,
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    player_attack_range: int,
    pillar_alive: list[bool],
    player_attack_cooldown: int = 0,
    attack_target: PlacedEntity | None = None,
) -> None:
    obs[offset] = 1.0

    closest_x, closest_y = InfernoLineOfSight.get_closest_point_on_npc(
        player_x, player_y, entity.x, entity.y, entity.entity_type.size_in_tiles,
    )
    closest_dist = InfernoLineOfSight.get_distance_from_npc(
        entity.x, entity.y, entity.entity_type.size_in_tiles, player_x, player_y,
    )

    obs[offset + 1] = _normalize_signed(closest_x - player_x, GRID_WIDTH)
    obs[offset + 2] = _normalize_signed(closest_y - player_y, GRID_HEIGHT)
    obs[offset + 3] = min(closest_dist / MAX_DISTANCE, 1.0)
    obs[offset + 4] = entity.current_health / entity.entity_type.max_health
    obs[offset + 5] = _normalize_signed(entity.attack_delay, MAX_ENTITY_COOLDOWN)
    obs[offset + 6] = min(entity.stunned / MAX_ENTITY_STUN_TICKS, 1.0)
    obs[offset + 7] = min(entity.frozen / MAX_ENTITY_FROZEN_TICKS, 1.0)
    obs[offset + 8] = 1.0 if InfernoLineOfSight.can_entity_attack_player(
        entity, player_x, player_y, pillar_alive,
    ) else 0.0
    obs[offset + 9] = 1.0 if InfernoLineOfSight.can_player_attack_entity(
        player_x, player_y, player_attack_range, entity, pillar_alive,
    ) else 0.0
    obs[offset + 10] = compute_dig_pressure(entity, player_x, player_y, pillar_alive)
    obs[offset + 11] = 1.0 if closest_dist <= 1.0 else 0.0

    ne_cx, ne_cy = int(NE_PILLAR_CENTER[0]), int(NE_PILLAR_CENTER[1])
    obs[offset + 12] = _normalize_signed(closest_x - ne_cx, GRID_WIDTH)
    obs[offset + 13] = _normalize_signed(closest_y - ne_cy, GRID_HEIGHT)
    obs[offset + 14] = 1.0 if entity.entity_type == EntityTypes.BLOB and entity.scanned_prayer == "MAGIC" else 0.0
    obs[offset + 15] = 1.0 if entity.entity_type == EntityTypes.BLOB and entity.scanned_prayer == "RANGED" else 0.0

    # can_attack_now: cooldown==0 AND in range AND player has LOS
    player_los = InfernoLineOfSight.can_player_attack_entity(
        player_x, player_y, player_attack_range, entity, pillar_alive,
    )
    obs[offset + 16] = 1.0 if (player_attack_cooldown == 0 and player_los) else 0.0

    # is_current_attack_target: this entity is the queued attack target
    obs[offset + 17] = 1.0 if (attack_target is not None and entity is attack_target) else 0.0

    # Pillar-relative angular features around NE pillar center
    ne_cx, ne_cy = NE_PILLAR_CENTER
    npc_angle = math.atan2(closest_y - ne_cy, closest_x - ne_cx)
    player_angle = math.atan2(player_y - ne_cy, player_x - ne_cx)
    angular_diff = abs(npc_angle - player_angle)
    if angular_diff > math.pi:
        angular_diff = 2.0 * math.pi - angular_diff
    obs[offset + 18] = angular_diff / math.pi  # 0 = same face, 1 = opposite
    obs[offset + 19] = math.sin(npc_angle)
    obs[offset + 20] = math.cos(npc_angle)


def _encode_exact_target_slot(
    obs: np.ndarray,
    offset: int,
    entity: PlacedEntity,
    player_x: int,
    player_y: int,
    player_attack_range: int,
    pillar_alive: list[bool],
    player_attack_cooldown: int = 0,
    attack_target: PlacedEntity | None = None,
) -> None:
    _encode_slot_core(
        obs,
        offset,
        entity,
        player_x,
        player_y,
        player_attack_range,
        pillar_alive,
        player_attack_cooldown=player_attack_cooldown,
        attack_target=attack_target,
    )
    type_offset = offset + SLOT_CORE_SIZE
    type_index = get_exact_target_type_index(entity.entity_type)
    obs[type_offset + type_index] = 1.0


def _compute_global_context(
    state: SimulatorState,
    combat_entities: list[PlacedEntity],
) -> tuple[int, int, dict[InfernoEntityType, int]]:
    current_los_count = 0
    current_imminent = 0
    type_alive_counts: dict[InfernoEntityType, int] = {}

    for entity in combat_entities:
        type_alive_counts[entity.entity_type] = type_alive_counts.get(entity.entity_type, 0) + 1
        if InfernoLineOfSight.can_entity_attack_player(
            entity, state.player_x, state.player_y, state.pillar_alive,
        ):
            current_los_count += 1
            if _is_imminent_with_los(entity):
                current_imminent += 1

    return current_los_count, current_imminent, type_alive_counts


def _fill_global_v4(
    obs: np.ndarray,
    idx: int,
    state: SimulatorState,
    tick_in_wave: int,
    alive_entities: list[PlacedEntity],
    combat_entities: list[PlacedEntity],
    nibbler_count: int,
    current_imminent: int,
    type_alive_counts: dict[InfernoEntityType, int],
    dead_mobs: list[PlacedEntity] | None,
) -> int:
    obs[idx] = state.player_x / GRID_WIDTH
    obs[idx + 1] = state.player_y / GRID_HEIGHT
    obs[idx + 2] = state.player_health / state.max_health
    obs[idx + 3] = min(state.get_player_attack_cooldown() / MAX_ATTACK_COOLDOWN, 1.0)
    idx += 4

    weapon_idx = get_weapon_index(state.current_preset, state.use_blood_barrage)
    obs[idx + weapon_idx] = 1.0
    idx += NUM_WEAPON_TYPES

    _encode_prayer_one_hot(obs, idx, state.active_prayer)
    idx += NUM_PRAYERS

    for i in range(3):
        obs[idx] = 1.0 if state.pillar_alive[i] else 0.0
        obs[idx + 1] = state.pillar_hp[i] / 255.0
        idx += 2

    obs[idx] = state.current_wave / MAX_WAVE
    obs[idx + 1] = min(tick_in_wave / MAX_TICK_IN_WAVE, 1.0)
    idx += 2

    obs[idx] = min(nibbler_count / MAX_NIBBLER_COUNT, 1.0)
    idx += 1

    miniblob_count = sum(
        type_alive_counts.get(entity_type, 0)
        for entity_type in (EntityTypes.BLOB_MAGE, EntityTypes.BLOB_RANGE, EntityTypes.BLOB_MELEE)
    )
    obs[idx] = min(miniblob_count / 6.0, 1.0)
    idx += 1

    ne_cx, ne_cy = int(NE_PILLAR_CENTER[0]), int(NE_PILLAR_CENTER[1])
    obs[idx] = _normalize_signed(state.player_x - ne_cx, GRID_WIDTH)
    obs[idx + 1] = _normalize_signed(state.player_y - ne_cy, GRID_HEIGHT)
    idx += 2

    obs[idx] = min(current_imminent / 5.0, 1.0)
    obs[idx + 1] = min(state.player_attack_range / MAX_PLAYER_ATTACK_RANGE, 1.0)
    idx += 2

    between_waves = state.wave_complete_timer >= 0 and len(combat_entities) == 0
    obs[idx] = 1.0 if between_waves else 0.0
    idx += 1

    for entity_type in _GLOBAL_COUNT_TYPES:
        obs[idx] = min(state.wave_kills.get(entity_type, 0) / MAX_SAME_NPC_COUNT, 1.0)
        idx += 1

    blob_scanned_magic = 0
    blob_scanned_ranged = 0
    blob_scanned_imminent = 0
    for entity in alive_entities:
        if entity.entity_type != EntityTypes.BLOB or entity.scanned_prayer is None:
            continue
        if entity.scanned_prayer == "MAGIC":
            blob_scanned_magic += 1
        elif entity.scanned_prayer == "RANGED":
            blob_scanned_ranged += 1
        if entity.attack_delay <= 0:
            blob_scanned_imminent += 1
    obs[idx] = min(blob_scanned_magic / MAX_BLOB_SCAN_COUNT, 1.0)
    obs[idx + 1] = min(blob_scanned_ranged / MAX_BLOB_SCAN_COUNT, 1.0)
    obs[idx + 2] = min(blob_scanned_imminent / MAX_BLOB_SCAN_COUNT, 1.0)
    idx += 3

    obs[idx] = (
        state.wave_complete_timer / WAVE_SPAWN_DELAY
        if state.wave_complete_timer >= 0 else 0.0
    )
    idx += 1

    _encode_prayer_one_hot(obs, idx, state.queued_prayer)
    idx += NUM_PRAYERS

    nibblers_nw = 0
    nibblers_ne = 0
    nibblers_s = 0
    for entity in alive_entities:
        if entity.entity_type != EntityTypes.NIBBLER:
            continue
        if entity.target_pillar_index == 0:
            nibblers_nw += 1
        elif entity.target_pillar_index == 1:
            nibblers_ne += 1
        elif entity.target_pillar_index == 2:
            nibblers_s += 1
    obs[idx] = min(nibblers_nw / MAX_NIBBLER_COUNT, 1.0)
    obs[idx + 1] = min(nibblers_ne / MAX_NIBBLER_COUNT, 1.0)
    obs[idx + 2] = min(nibblers_s / MAX_NIBBLER_COUNT, 1.0)
    idx += 3

    dead_mobs_list = dead_mobs or []
    dead_counts: dict[InfernoEntityType, int] = {}
    for entity in dead_mobs_list:
        dead_counts[entity.entity_type] = dead_counts.get(entity.entity_type, 0) + 1
    for dead_type in _DEAD_POOL_TYPES:
        obs[idx] = min(dead_counts.get(dead_type, 0) / MAX_DEAD_POOL_COUNT, 1.0)
        idx += 1

    alive_magers = type_alive_counts.get(EntityTypes.MAGER, 0)
    resurrection_hazard = alive_magers * len(dead_mobs_list)
    obs[idx] = min(resurrection_hazard / MAX_RESURRECTION_HAZARD, 1.0)
    idx += 1

    obs[idx] = 1.0 if alive_magers > 0 else 0.0
    idx += 1

    obs[idx] = 1.0 if (
        state.attack_target is not None and not state.attack_target.is_dead()
    ) else 0.0
    idx += 1

    return idx


def _fill_temporal(obs: np.ndarray, idx: int, temporal) -> int:
    if temporal is not None:
        obs[idx] = min(sum(temporal.damage_taken_buffer) / MAX_HEALTH, 1.0)
        obs[idx + 1] = min(sum(temporal.damage_dealt_buffer) / MAX_HEALTH, 1.0)
        obs[idx + 2] = min(temporal.ticks_since_last_attack / 10.0, 1.0)
        obs[idx + 3] = min(temporal.ticks_since_engagement / 20.0, 1.0)
        obs[idx + 4] = 1.0 if temporal.prev_action_movement else 0.0
        obs[idx + 5] = 1.0 if temporal.prev_action_attack else 0.0
        obs[idx + 6] = 1.0 if temporal.prev_action_weapon_switch else 0.0
    return idx + TEMPORAL_V3_SIZE


def _fill_exact_target_slots(
    obs: np.ndarray,
    slot_base: int,
    exact_targets: list[PlacedEntity],
    player_x: int,
    player_y: int,
    player_attack_range: int,
    pillar_alive: list[bool],
    player_attack_cooldown: int = 0,
    attack_target: PlacedEntity | None = None,
) -> None:
    for slot_index, entity in enumerate(exact_targets[:EXACT_TARGET_SLOT_COUNT]):
        offset = slot_base + slot_index * EXACT_TARGET_SLOT_SIZE
        _encode_exact_target_slot(
            obs,
            offset,
            entity,
            player_x,
            player_y,
            player_attack_range,
            pillar_alive,
            player_attack_cooldown=player_attack_cooldown,
            attack_target=attack_target,
        )


def _fill_loadout_block(obs: np.ndarray, idx: int, state: SimulatorState) -> int:
    obs[idx] = 1.0 if state.has_blowpipe else 0.0
    ranged = state.loadout_preset_stats[GearPreset.BOFA]
    obs[idx + 1] = ranged.attack_speed / MAX_WEAPON_SPEED
    obs[idx + 2] = state.player_attack_range / MAX_PLAYER_ATTACK_RANGE
    obs[idx + 3] = ranged.ranged_attack / MAX_RANGED_ATTACK_BONUS
    obs[idx + 4] = ranged.ranged_strength / MAX_RANGED_STRENGTH_BONUS
    mage = state.loadout_preset_stats[GearPreset.MAGE]
    obs[idx + 5] = mage.magic_attack / MAX_MAGIC_ATTACK_BONUS
    obs[idx + 6] = state.max_health / 99.0
    return idx + LOADOUT_BLOCK_SIZE


def build_observation_v4(
    state: SimulatorState,
    tick_in_wave: int,
    temporal,
    dead_mobs: list[PlacedEntity] | None,
    neighborhood_summaries: list[RawTileThreatSummary] | None = None,
    tick_threat_cache: TickThreatCache | None = None,
) -> np.ndarray:
    obs = np.zeros(OBSERVATION_TOTAL_SIZE, dtype=np.float32)

    alive_entities = _get_alive_entities(state)
    combat_entities = [entity for entity in alive_entities if entity.entity_type != EntityTypes.NIBBLER]
    nibbler_count = sum(1 for entity in alive_entities if entity.entity_type == EntityTypes.NIBBLER)
    exact_targets = get_exact_target_slots(state)
    if tick_threat_cache is not None:
        current_los_count = tick_threat_cache.npcs_with_los_now
        current_imminent = tick_threat_cache.current_imminent_attacks
        type_alive_counts = tick_threat_cache.type_alive_counts
        neighborhood_summaries = tick_threat_cache.neighborhood_summaries
    else:
        current_los_count, current_imminent, type_alive_counts = _compute_global_context(
            state,
            combat_entities,
        )

    idx = _fill_global_v4(
        obs,
        0,
        state,
        tick_in_wave,
        alive_entities,
        combat_entities,
        nibbler_count,
        current_imminent,
        type_alive_counts,
        dead_mobs,
    )
    assert idx == GLOBAL_V4_SIZE, f"Global block ended at {idx}, expected {GLOBAL_V4_SIZE}"

    # Priority target for neighborhood attackability: attack_target if alive, else first exact target
    priority_target = state.attack_target
    if priority_target is None or priority_target.is_dead():
        priority_target = exact_targets[0] if exact_targets else None

    neighborhood = forecast_neighborhood_safety(
        state.player_x,
        state.player_y,
        state.pillar_alive,
        combat_entities,
        current_los_count,
        state.active_prayer,
        raw_summaries=neighborhood_summaries,
        priority_target=priority_target,
        attack_range=state.player_attack_range,
    )
    for tile in neighborhood:
        obs[idx] = tile.settled_step_distance
        obs[idx + 1] = tile.los_count
        obs[idx + 2] = tile.los_delta
        obs[idx + 3] = tile.min_attack_delay
        obs[idx + 4] = tile.imminent_magic
        obs[idx + 5] = tile.imminent_ranged
        obs[idx + 6] = tile.imminent_melee
        obs[idx + 7] = tile.unprotected_after_auto_prayer
        obs[idx + 8] = tile.blob_scan_triggers
        obs[idx + 9] = tile.priority_target_attackable
        obs[idx + 10] = tile.best_los_in_2_steps
        obs[idx + 11] = tile.steps_to_single_los
        idx += NEIGHBORHOOD_FEATURES
    assert idx == GLOBAL_V4_SIZE + NEIGHBORHOOD_FORECAST_SIZE

    threat_horizon = forecast_threat_styles(
        combat_entities,
        state.player_x,
        state.player_y,
        state.pillar_alive,
        state.active_prayer,
        horizons=3,
    )
    for horizon in threat_horizon:
        obs[idx] = min(horizon.magic / 5.0, 1.0)
        obs[idx + 1] = min(horizon.ranged / 5.0, 1.0)
        obs[idx + 2] = min(horizon.melee / 5.0, 1.0)
        idx += 3
    assert idx == GLOBAL_V4_SIZE + NEIGHBORHOOD_FORECAST_SIZE + THREAT_HORIZON_SIZE

    idx = _fill_temporal(obs, idx, temporal)

    _fill_exact_target_slots(
        obs,
        idx,
        exact_targets,
        state.player_x,
        state.player_y,
        state.player_attack_range,
        state.pillar_alive,
        player_attack_cooldown=state.get_player_attack_cooldown(),
        attack_target=state.attack_target,
    )
    idx += EXACT_TARGET_SLOTS_TOTAL

    idx = _fill_loadout_block(obs, idx, state)
    assert idx == OBSERVATION_PUBLIC_SIZE
    assert idx == OBSERVATION_TOTAL_SIZE
    return obs
