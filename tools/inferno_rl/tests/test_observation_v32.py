import math

import numpy as np

from tools.inferno_rl.simulator.entity import EntityTypes, PlacedEntity
from tools.inferno_rl.simulator.exact_targeting import (
    EXACT_TARGET_ENTITY_TYPES,
    get_exact_target_slots,
)
from tools.inferno_rl.simulator.forecast import (
    build_tick_threat_cache,
    forecast_blob_scan_triggers,
    predict_auto_prayer_for_position,
)
from tools.inferno_rl.simulator.state import SimulatorState
from tools.inferno_rl.simulator.simulator import InfernoSimulator
from tools.inferno_rl.training.actions import get_policy_action_mask
from tools.inferno_rl.training.observation import (
    TemporalState,
    build_observation,
    get_observation_low,
    get_observation_size,
    get_public_observation_size,
)
from tools.inferno_rl.training.observation_common import (
    EXACT_TARGET_SLOT_COUNT,
    EXACT_TARGET_SLOT_SIZE,
    EXACT_TARGET_SLOTS_TOTAL,
    GLOBAL_V4_SIZE,
    LOADOUT_BLOCK_SIZE,
    MAX_ENTITY_COOLDOWN,
    MAX_ENTITY_FROZEN_TICKS,
    MAX_ENTITY_STUN_TICKS,
    NEIGHBORHOOD_FEATURES,
    NEIGHBORHOOD_FORECAST_SIZE,
    SLOT_CORE_SIZE,
    TEMPORAL_V3_SIZE,
    THREAT_HORIZON_SIZE,
)


def _base_state() -> SimulatorState:
    state = SimulatorState()
    state.current_wave = 63
    state.current_tick = 50
    state.wave_complete_timer = -1
    state.pillar_hp = [255, 255, 255]
    state.pillar_alive = [True, True, True]
    state.player_x = 16
    state.player_y = 23
    state.entities = []
    return state


def _exact_slot_base() -> int:
    return (
        GLOBAL_V4_SIZE
        + NEIGHBORHOOD_FORECAST_SIZE
        + THREAT_HORIZON_SIZE
        + TEMPORAL_V3_SIZE
    )


def _loadout_base() -> int:
    return _exact_slot_base() + EXACT_TARGET_SLOTS_TOTAL


def _exact_slot_offset(slot: int) -> int:
    return _exact_slot_base() + slot * EXACT_TARGET_SLOT_SIZE


def _tile_offset(tile_index: int) -> int:
    return GLOBAL_V4_SIZE + tile_index * NEIGHBORHOOD_FEATURES


def _assert_slot_matches_entity(obs, slot_index: int, entity: PlacedEntity) -> None:
    offset = _exact_slot_offset(slot_index)
    assert obs[offset] == 1.0
    type_offset = offset + SLOT_CORE_SIZE
    type_idx = EXACT_TARGET_ENTITY_TYPES.index(entity.entity_type)
    assert obs[type_offset + type_idx] == 1.0


def test_observation_v4_size_is_public_only() -> None:
    assert get_public_observation_size() == 602
    assert get_observation_size() == 602
    assert get_observation_low() == -1.0


def test_exact_target_slots_follow_shared_ordering() -> None:
    state = _base_state()
    mager = PlacedEntity(EntityTypes.MAGER, 16, 27, 0)
    ranger = PlacedEntity(EntityTypes.RANGER, 6, 23, 0)
    melee = PlacedEntity(EntityTypes.MELEE, 20, 23, 0)
    nibbler = PlacedEntity(EntityTypes.NIBBLER, 9, 16, 0)
    nibbler.target_pillar_index = 1
    state.entities = [ranger, melee, nibbler, mager]

    exact_targets = get_exact_target_slots(state)
    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])

    assert [entity.entity_type for entity in exact_targets] == [
        EntityTypes.MAGER,
        EntityTypes.RANGER,
        EntityTypes.MELEE,
        EntityTypes.NIBBLER,
    ]
    for slot_index, entity in enumerate(exact_targets):
        _assert_slot_matches_entity(obs, slot_index, entity)


def test_wave_63_all_alive_exposes_both_blobs_as_distinct_slots() -> None:
    state = _base_state()
    mager = PlacedEntity(EntityTypes.MAGER, 16, 27, 0)
    ranger = PlacedEntity(EntityTypes.RANGER, 9, 23, 0)
    melee = PlacedEntity(EntityTypes.MELEE, 24, 23, 0)
    blob_a = PlacedEntity(EntityTypes.BLOB, 6, 16, 0)
    blob_b = PlacedEntity(EntityTypes.BLOB, 22, 16, 0)
    nibblers = []
    for idx, x in enumerate((8, 9, 10)):
        nibbler = PlacedEntity(EntityTypes.NIBBLER, x, 16, idx)
        nibbler.target_pillar_index = 1
        nibblers.append(nibbler)
    state.entities = [blob_b, melee, nibblers[0], mager, blob_a, nibblers[1], ranger, nibblers[2]]

    exact_targets = get_exact_target_slots(state)
    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])

    blob_slots = [idx for idx, entity in enumerate(exact_targets) if entity.entity_type == EntityTypes.BLOB]
    assert len(blob_slots) == 2
    assert [entity.entity_type for entity in exact_targets[:2]] == [
        EntityTypes.MAGER,
        EntityTypes.RANGER,
    ]
    assert blob_slots[-1] >= 3
    for slot_index in blob_slots:
        _assert_slot_matches_entity(obs, slot_index, exact_targets[slot_index])


def test_wave_63_split_state_exposes_all_blob_family_members_and_nibblers() -> None:
    sim = InfernoSimulator(start_wave=63, max_wave=63)
    state = sim.state
    state.current_wave = 63
    state.current_tick = 50
    state.wave_complete_timer = -1
    state.pillar_hp = [255, 255, 255]
    state.pillar_alive = [True, True, True]
    state.player_x = 16
    state.player_y = 23
    state.entities = []

    mager = PlacedEntity(EntityTypes.MAGER, 16, 27, 0)
    ranger = PlacedEntity(EntityTypes.RANGER, 9, 23, 0)
    melee = PlacedEntity(EntityTypes.MELEE, 24, 23, 0)
    live_blob = PlacedEntity(EntityTypes.BLOB, 0, 0, 0)
    state.entities.extend([mager, ranger, melee, live_blob])

    dead_blob = PlacedEntity(EntityTypes.BLOB, 14, 21, 0)
    dead_blob.current_health = 0
    sim._spawn_blob_splits(dead_blob)

    for idx, x in enumerate((8, 9, 10)):
        nibbler = PlacedEntity(EntityTypes.NIBBLER, x, 16, idx)
        nibbler.target_pillar_index = 1
        state.entities.append(nibbler)

    exact_targets = get_exact_target_slots(state)
    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])

    assert len(exact_targets) == 10
    blob_family = [
        entity.entity_type for entity in exact_targets
        if entity.entity_type in {
            EntityTypes.BLOB,
            EntityTypes.BLOB_MAGE,
            EntityTypes.BLOB_RANGE,
            EntityTypes.BLOB_MELEE,
        }
    ]
    assert set(blob_family) == {
        EntityTypes.BLOB,
        EntityTypes.BLOB_MAGE,
        EntityTypes.BLOB_RANGE,
        EntityTypes.BLOB_MELEE,
    }
    assert sum(1 for entity in exact_targets if entity.entity_type == EntityTypes.NIBBLER) == 3
    for slot_index, entity in enumerate(exact_targets):
        _assert_slot_matches_entity(obs, slot_index, entity)
    for slot_index in range(len(exact_targets), EXACT_TARGET_SLOT_COUNT):
        assert obs[_exact_slot_offset(slot_index)] == 0.0


def test_synthetic_max_expanded_state_fills_all_14_exact_target_slots() -> None:
    state = _base_state()
    entities = [
        PlacedEntity(EntityTypes.MAGER, 16, 27, 0),
        PlacedEntity(EntityTypes.RANGER, 9, 23, 0),
        PlacedEntity(EntityTypes.MELEE, 24, 23, 0),
        PlacedEntity(EntityTypes.BAT, 2, 2, 0),
        PlacedEntity(EntityTypes.BLOB, 6, 16, 0),
        PlacedEntity(EntityTypes.BLOB_MAGE, 18, 18, 0),
        PlacedEntity(EntityTypes.BLOB_RANGE, 20, 18, 0),
        PlacedEntity(EntityTypes.BLOB_MELEE, 22, 18, 0),
    ]
    for idx, x in enumerate((4, 5, 6, 7, 8, 9)):
        nibbler = PlacedEntity(EntityTypes.NIBBLER, x, 16, idx)
        nibbler.target_pillar_index = 1 if idx < 3 else 0
        entities.append(nibbler)
    state.entities = entities

    exact_targets = get_exact_target_slots(state)
    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])

    assert len(exact_targets) == EXACT_TARGET_SLOT_COUNT == 14
    assert math.isclose(sum(obs[_exact_slot_offset(slot)] for slot in range(EXACT_TARGET_SLOT_COUNT)), 14.0)
    for slot_index, entity in enumerate(exact_targets):
        _assert_slot_matches_entity(obs, slot_index, entity)
    assert _loadout_base() + LOADOUT_BLOCK_SIZE == get_public_observation_size()


def test_public_slot_timers_are_signed_and_continuous() -> None:
    state = _base_state()
    mager = PlacedEntity(EntityTypes.MAGER, 16, 27, 0)
    mager.attack_delay = -3
    mager.stunned = 2
    mager.frozen = 16
    state.entities = [mager]

    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])
    offset = _exact_slot_offset(0)

    assert math.isclose(obs[offset + 5], -3 / MAX_ENTITY_COOLDOWN, rel_tol=1e-6)
    assert math.isclose(obs[offset + 6], 2 / MAX_ENTITY_STUN_TICKS, rel_tol=1e-6)
    assert math.isclose(obs[offset + 7], 16 / MAX_ENTITY_FROZEN_TICKS, rel_tol=1e-6)


def test_neighborhood_unprotected_after_auto_prayer_matches_predictor_logic() -> None:
    state = _base_state()
    mager = PlacedEntity(EntityTypes.MAGER, 16, 27, 0)
    ranger = PlacedEntity(EntityTypes.RANGER, 10, 23, 0)
    state.entities = [mager, ranger]

    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])
    stay_offset = _tile_offset(0)
    predicted_prayer = predict_auto_prayer_for_position(
        state.entities,
        state.player_x,
        state.player_y,
        state.pillar_alive,
        state.active_prayer,
    )

    assert predicted_prayer == "PROTECT_FROM_MAGIC"
    assert math.isclose(obs[stay_offset + 7], 1 / 6.0, rel_tol=1e-6)


def test_neighborhood_blob_scan_triggers_match_directional_forecast_logic() -> None:
    state = _base_state()
    blob = PlacedEntity(EntityTypes.BLOB, 0, 6, 0)
    blob.attack_delay = 1
    blob.scanned_prayer = None
    blob.had_los = False
    state.entities = [blob]

    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])
    direction_triggers = forecast_blob_scan_triggers(
        state.player_x,
        state.player_y,
        state.pillar_alive,
        [blob],
    )

    south_tile_offset = _tile_offset(2)
    assert math.isclose(obs[south_tile_offset + 8], direction_triggers[1] / 2.0, rel_tol=1e-6)


def test_tick_threat_cache_preserves_observation_and_action_mask() -> None:
    sim = InfernoSimulator(start_wave=63, max_wave=63)
    sim.initial_barrage_enabled = False
    sim.auto_prayer_enabled = False
    sim.reset_to_wave(63)
    sim.step(0)

    combat_entities = [
        entity for entity in sim.state.entities
        if not entity.is_dead() and entity.entity_type != EntityTypes.NIBBLER
    ]
    tick_threat_cache = build_tick_threat_cache(
        sim.state.player_x,
        sim.state.player_y,
        sim.state.pillar_alive,
        combat_entities,
        sim.state.active_prayer,
    )

    uncached_obs = build_observation(
        sim.state,
        tick_in_wave=sim.get_ticks_in_wave(),
        temporal=TemporalState(),
        dead_mobs=sim.dead_mobs,
    )
    cached_obs = build_observation(
        sim.state,
        tick_in_wave=sim.get_ticks_in_wave(),
        temporal=TemporalState(),
        dead_mobs=sim.dead_mobs,
        tick_threat_cache=tick_threat_cache,
    )
    uncached_mask = get_policy_action_mask(sim.state)
    cached_mask = get_policy_action_mask(sim.state, tick_threat_cache=tick_threat_cache)

    assert np.array_equal(cached_obs, uncached_obs)
    assert np.array_equal(cached_mask, uncached_mask)


def test_pillar_angular_separation_opposite_sides() -> None:
    """NPC on opposite side of NE pillar from player should have high separation."""
    state = _base_state()
    # Player NW of NE pillar, mager SE of NE pillar
    state.player_x = 15
    state.player_y = 26
    mager = PlacedEntity(EntityTypes.MAGER, 21, 19, 0)
    state.entities = [mager]

    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])
    offset = _exact_slot_offset(0)

    assert obs[offset + 18] > 0.5  # angular separation > 0.5 = opposite-ish
    assert -1.0 <= obs[offset + 19] <= 1.0  # sin
    assert -1.0 <= obs[offset + 20] <= 1.0  # cos


def test_pillar_angular_separation_same_side() -> None:
    """NPC on same side as player should have low separation."""
    state = _base_state()
    state.player_x = 15
    state.player_y = 25
    ranger = PlacedEntity(EntityTypes.RANGER, 14, 25, 0)
    state.entities = [ranger]

    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])
    offset = _exact_slot_offset(0)

    assert obs[offset + 18] < 0.3  # same side = low separation


def test_neighborhood_bfs_features_valid_range() -> None:
    """BFS features should be in [0, 1] range."""
    state = _base_state()
    state.player_x = 16
    state.player_y = 23
    mager = PlacedEntity(EntityTypes.MAGER, 16, 27, 0)
    ranger = PlacedEntity(EntityTypes.RANGER, 20, 20, 0)
    state.entities = [mager, ranger]

    obs = build_observation(state, tick_in_wave=10, temporal=TemporalState(), dead_mobs=[])

    for tile_idx in range(9):
        tile_offset = _tile_offset(tile_idx)
        best_los = obs[tile_offset + 10]
        steps_single = obs[tile_offset + 11]
        assert 0.0 <= best_los <= 1.0, f"tile {tile_idx} best_los={best_los}"
        assert 0.0 <= steps_single <= 1.0, f"tile {tile_idx} steps_single={steps_single}"
