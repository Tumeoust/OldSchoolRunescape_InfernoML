import math
import random

import pytest

from tools.inferno_rl.simulator.entity import EntityTypes, PlacedEntity
from tools.inferno_rl.simulator.forecast import (
    PLAYER_MOVE_DIRECTIONS,
    _build_neighborhood_tile_threat_summaries_python,
    _forecast_threat_styles_python,
    build_movement_resolution_table,
    build_neighborhood_tile_threat_summaries,
    forecast_neighborhood_safety,
    forecast_threat_styles,
    predict_auto_prayer_for_position,
    _is_scanned_blob_imminent,
    _predict_npc_position_after_decrement,
)
from tools.inferno_rl.simulator.geometry import InfernoLineOfSight, SimulatorGeometry
from tools.inferno_rl.simulator.pathfinding import OSRSPathfinding
from tools.inferno_rl.simulator.simulator import InfernoSimulator
from tools.inferno_rl.simulator.state import SimulatorState
from tools.inferno_rl.training.actions import get_movement_params


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


def _direct_settled_destination(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    legacy_action: int,
) -> tuple[int, int]:
    dx, dy, distance = get_movement_params(legacy_action)
    target_x = max(0, min(28, player_x + dx * distance))
    target_y = max(0, min(29, player_y + dy * distance))

    def checker(x: int, y: int, size: int) -> bool:
        return size == 1 and SimulatorGeometry.is_valid_tile(x, y, pillar_alive)

    return OSRSPathfinding.simulate_player_movement(
        player_x,
        player_y,
        target_x,
        target_y,
        2,
        checker,
    )


def _legacy_directional_forecasts(
    player_x: int,
    player_y: int,
    pillar_alive: list[bool],
    combat_entities: list[PlacedEntity],
) -> list[tuple[int, int, int]]:
    forecasts: list[tuple[int, int, int]] = []

    def checker(x: int, y: int, size: int) -> bool:
        return size == 1 and SimulatorGeometry.is_valid_tile(x, y, pillar_alive)

    for dx, dy in PLAYER_MOVE_DIRECTIONS:
        target_x = max(0, min(28, player_x + (dx * 2)))
        target_y = max(0, min(29, player_y + (dy * 2)))
        settled_x, settled_y = OSRSPathfinding.simulate_player_movement(
            player_x,
            player_y,
            target_x,
            target_y,
            2,
            checker,
        )
        settled_distance = max(abs(settled_x - player_x), abs(settled_y - player_y))
        los_count = 0
        imminent_attacks = 0
        for entity in combat_entities:
            forecast_entity = entity.copy()
            forecast_entity.attack_delay -= 1
            if forecast_entity.stunned > 0:
                forecast_entity.stunned -= 1
            if forecast_entity.frozen > 0:
                forecast_entity.frozen -= 1
            predicted_x, predicted_y = _predict_npc_position_after_decrement(
                forecast_entity,
                settled_x,
                settled_y,
                pillar_alive,
            )
            predicted_entity = forecast_entity.copy()
            predicted_entity.x = predicted_x
            predicted_entity.y = predicted_y
            has_los = InfernoLineOfSight.can_entity_attack_player(
                predicted_entity,
                settled_x,
                settled_y,
                pillar_alive,
            )
            if has_los:
                los_count += 1
                if (
                    forecast_entity.entity_type != EntityTypes.BLOB
                    and forecast_entity.attack_delay <= 0
                    and forecast_entity.stunned <= 0
                ):
                    imminent_attacks += 1
            if _is_scanned_blob_imminent(forecast_entity):
                imminent_attacks += 1
        forecasts.append((settled_distance, los_count, imminent_attacks))
    return forecasts


def _legacy_avoidable_extras(state: SimulatorState) -> tuple[int, int]:
    px, py = state.player_x, state.player_y
    pillar_alive = state.pillar_alive
    combat_entities: list[PlacedEntity] = []
    npcs_with_los_now = 0
    current_imminent_attacks = 0

    for entity in state.entities:
        if entity.is_dead() or entity.entity_type == EntityTypes.NIBBLER:
            continue
        combat_entities.append(entity)
        if InfernoLineOfSight.can_entity_attack_player(entity, px, py, pillar_alive):
            npcs_with_los_now += 1
            if entity.entity_type == EntityTypes.BLOB:
                if entity.scanned_prayer is not None and entity.attack_delay <= 0:
                    current_imminent_attacks += 1
            elif entity.attack_delay <= 0 and entity.stunned <= 0:
                current_imminent_attacks += 1

    best_reachable_los = npcs_with_los_now
    best_reachable_imminent = current_imminent_attacks
    for settled_distance, los_count, imminent_attacks in _legacy_directional_forecasts(
        px,
        py,
        pillar_alive,
        combat_entities,
    ):
        if settled_distance <= 0:
            continue
        best_reachable_los = min(best_reachable_los, los_count)
        best_reachable_imminent = min(best_reachable_imminent, imminent_attacks)

    return (
        max(0, npcs_with_los_now - best_reachable_los),
        max(0, current_imminent_attacks - best_reachable_imminent),
    )


def test_movement_resolution_table_matches_direct_bfs() -> None:
    for pillar_mask in range(8):
        pillar_alive = [
            bool(pillar_mask & 0b001),
            bool(pillar_mask & 0b010),
            bool(pillar_mask & 0b100),
        ]
        for player_x in range(29):
            for player_y in range(30):
                table = build_movement_resolution_table(player_x, player_y, pillar_alive)
                for legacy_action in range(1, 33):
                    assert table.destination_for_action(legacy_action) == _direct_settled_destination(
                        player_x,
                        player_y,
                        pillar_alive,
                        legacy_action,
                    )


def test_step_result_avoidable_extras_match_legacy_directional_baseline() -> None:
    random.seed(0)
    sim = InfernoSimulator(start_wave=63, max_wave=63)
    sim.initial_barrage_enabled = False
    sim.auto_prayer_enabled = False
    sim.reset_to_wave(63)

    result = sim.step(0)
    expected_los, expected_imminent = _legacy_avoidable_extras(sim.state)

    assert result.avoidable_extra_los == expected_los
    assert result.avoidable_extra_imminent == expected_imminent


def test_mixed_threat_neighborhood_forecast_and_auto_prayer_regression() -> None:
    state = _base_state()
    mager = PlacedEntity(EntityTypes.MAGER, 16, 27, 0)
    mager.attack_delay = 1
    mager.stunned = 0

    ranger = PlacedEntity(EntityTypes.RANGER, 10, 23, 0)
    ranger.attack_delay = 1
    ranger.stunned = 0

    melee = PlacedEntity(EntityTypes.MELEE, 12, 23, 0)
    melee.attack_delay = 1
    melee.stunned = 0

    blob = PlacedEntity(EntityTypes.BLOB, 6, 16, 0)
    blob.attack_delay = 1
    blob.stunned = 0
    blob.scanned_prayer = "MAGIC"
    blob.had_los = False

    state.entities = [mager, ranger, melee, blob]
    combat_entities = [entity for entity in state.entities if entity.entity_type != EntityTypes.NIBBLER]
    current_los_count = sum(
        1
        for entity in combat_entities
        if InfernoLineOfSight.can_entity_attack_player(
            entity,
            state.player_x,
            state.player_y,
            state.pillar_alive,
        )
    )

    predicted_prayer = predict_auto_prayer_for_position(
        state.entities,
        state.player_x,
        state.player_y,
        state.pillar_alive,
        state.active_prayer,
    )
    neighborhood = forecast_neighborhood_safety(
        state.player_x,
        state.player_y,
        state.pillar_alive,
        combat_entities,
        current_los_count,
        state.active_prayer,
    )
    stay = neighborhood[0]

    assert predicted_prayer == "PROTECT_FROM_MAGIC"
    assert math.isclose(stay.los_count, 4.0 / 9.0, rel_tol=1e-6)
    assert math.isclose(stay.los_delta, 0.0, rel_tol=1e-6)
    assert math.isclose(stay.min_attack_delay, 0.0, rel_tol=1e-6)
    assert math.isclose(stay.imminent_magic, 2.0 / 5.0, rel_tol=1e-6)
    assert math.isclose(stay.imminent_ranged, 1.0 / 5.0, rel_tol=1e-6)
    assert math.isclose(stay.imminent_melee, 1.0 / 5.0, rel_tol=1e-6)
    assert math.isclose(stay.unprotected_after_auto_prayer, 2.0 / 6.0, rel_tol=1e-6)
    assert math.isclose(stay.blob_scan_triggers, 1.0 / 2.0, rel_tol=1e-6)


def _snapshot_states(count: int) -> list[tuple[int, int, list[bool], str | None, list[PlacedEntity]]]:
    rng = random.Random(123)
    sim = InfernoSimulator(start_wave=63, max_wave=63)
    sim.initial_barrage_enabled = False
    sim.auto_prayer_enabled = False
    sim.reset_to_wave(63)
    snapshots: list[tuple[int, int, list[bool], str | None, list[PlacedEntity]]] = []

    while len(snapshots) < count:
        combat_entities = [
            entity.copy()
            for entity in sim.state.entities
            if not entity.is_dead() and entity.entity_type != EntityTypes.NIBBLER
        ]
        snapshots.append(
            (
                sim.state.player_x,
                sim.state.player_y,
                list(sim.state.pillar_alive),
                sim.state.active_prayer,
                combat_entities,
            )
        )
        result = sim.step(rng.randint(0, 51))
        if result.is_terminal():
            sim.reset_to_wave(63)

    return snapshots


def test_compiled_forecast_matches_python_baseline_on_snapshots() -> None:
    pytest.importorskip(
        "tools.inferno_rl.simulator.forecast_fast",
        reason="compiled forecast backend not built",
    )
    for player_x, player_y, pillar_alive, active_prayer, combat_entities in _snapshot_states(8):
        python_summaries = _build_neighborhood_tile_threat_summaries_python(
            player_x,
            player_y,
            pillar_alive,
            combat_entities,
            active_prayer,
        )
        compiled_summaries = build_neighborhood_tile_threat_summaries(
            player_x,
            player_y,
            pillar_alive,
            combat_entities,
            active_prayer,
        )
        assert compiled_summaries == python_summaries

        compiled_threats = forecast_threat_styles(
            combat_entities,
            player_x,
            player_y,
            pillar_alive,
            active_prayer,
            horizons=4,
        )
        python_threats = _forecast_threat_styles_python(
            combat_entities,
            player_x,
            player_y,
            pillar_alive,
            active_prayer,
            horizons=4,
        )
        assert compiled_threats == python_threats


def test_public_forecast_matches_python_baseline_on_snapshot() -> None:
    player_x, player_y, pillar_alive, active_prayer, combat_entities = _snapshot_states(1)[0]

    public_summaries = build_neighborhood_tile_threat_summaries(
        player_x,
        player_y,
        pillar_alive,
        combat_entities,
        active_prayer,
    )
    python_summaries = _build_neighborhood_tile_threat_summaries_python(
        player_x,
        player_y,
        pillar_alive,
        combat_entities,
        active_prayer,
    )
    public_threats = forecast_threat_styles(
        combat_entities,
        player_x,
        player_y,
        pillar_alive,
        active_prayer,
        horizons=4,
    )
    python_threats = _forecast_threat_styles_python(
        combat_entities,
        player_x,
        player_y,
        pillar_alive,
        active_prayer,
        horizons=4,
    )

    assert public_summaries == python_summaries
    assert public_threats == python_threats
