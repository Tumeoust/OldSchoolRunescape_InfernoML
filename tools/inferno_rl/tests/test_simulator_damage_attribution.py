from unittest.mock import patch

from tools.inferno_rl.simulator.entity import EntityTypes, PlacedEntity
from tools.inferno_rl.simulator.simulator import InfernoSimulator


def _make_clean_sim() -> InfernoSimulator:
    sim = InfernoSimulator(start_wave=35, max_wave=35)
    sim.auto_prayer_enabled = False
    sim.reset()
    sim.state.clear_entities()
    sim.state.wave_complete_timer = -1
    sim.state.active_prayer = None
    sim.state.player_health = 99
    return sim


def _add_ready_attacker(sim: InfernoSimulator, entity_type, x: int, y: int) -> PlacedEntity:
    entity = PlacedEntity(
        entity_type=entity_type,
        x=x,
        y=y,
        placed_tick=sim.state.current_tick,
    )
    entity.stunned = 0
    entity.attack_delay = 0
    sim.state.add_entity(entity)
    return entity


def test_records_positive_damage_with_attacker_metadata() -> None:
    sim = _make_clean_sim()
    sim.state.player_x = 10
    sim.state.player_y = 10
    attacker = _add_ready_attacker(sim, EntityTypes.RANGER, x=10, y=14)

    with patch("tools.inferno_rl.simulator.npc_combat.roll_npc_damage", return_value=7):
        result = sim.step(0)

    assert result.damage_taken == 7
    assert len(result.player_damage_events) == 1

    event = result.player_damage_events[0]
    assert event.tick == sim.state.current_tick
    assert event.attacker_id == attacker.id
    assert event.attacker_type == attacker.entity_type.name
    assert (event.attacker_x, event.attacker_y) == (attacker.x, attacker.y)
    assert event.attack_style == "RANGED"
    assert event.damage == 7


def test_melee_fallback_attack_records_melee_style() -> None:
    sim = _make_clean_sim()
    sim.state.player_x = 13
    sim.state.player_y = 11
    _add_ready_attacker(sim, EntityTypes.RANGER, x=10, y=10)

    with patch("tools.inferno_rl.simulator.npc_combat.roll_npc_damage", return_value=4):
        result = sim.step(0)

    assert result.damage_taken == 4
    assert len(result.player_damage_events) == 1
    assert result.player_damage_events[0].attack_style == "MELEE"


def test_blocked_hits_are_not_recorded() -> None:
    sim = _make_clean_sim()
    sim.state.player_x = 10
    sim.state.player_y = 10
    sim.state.active_prayer = "PROTECT_FROM_MISSILES"
    _add_ready_attacker(sim, EntityTypes.RANGER, x=10, y=14)

    with patch("tools.inferno_rl.simulator.npc_combat.roll_npc_damage", return_value=9):
        result = sim.step(0)

    assert result.damage_taken == 0
    assert result.player_damage_events == []


def test_zero_damage_hits_are_not_recorded() -> None:
    sim = _make_clean_sim()
    sim.state.player_x = 10
    sim.state.player_y = 10
    _add_ready_attacker(sim, EntityTypes.RANGER, x=10, y=14)

    with patch("tools.inferno_rl.simulator.npc_combat.roll_npc_damage", return_value=0):
        result = sim.step(0)

    assert result.damage_taken == 0
    assert result.player_damage_events == []


def test_multiple_attackers_create_multiple_damage_events() -> None:
    sim = _make_clean_sim()
    sim.state.player_x = 10
    sim.state.player_y = 10
    ranger = _add_ready_attacker(sim, EntityTypes.RANGER, x=10, y=14)
    mager = _add_ready_attacker(sim, EntityTypes.MAGER, x=14, y=10)

    with patch("tools.inferno_rl.simulator.npc_combat.roll_npc_damage", side_effect=[5, 8]):
        result = sim.step(0)

    assert result.damage_taken == 13
    assert len(result.player_damage_events) == 2
    assert {event.attacker_id for event in result.player_damage_events} == {ranger.id, mager.id}
    assert {event.attack_style for event in result.player_damage_events} == {"RANGED", "MAGIC"}
    assert all(event.tick == sim.state.current_tick for event in result.player_damage_events)


def test_damage_event_uses_actual_hp_lost_when_overkilling() -> None:
    sim = _make_clean_sim()
    sim.state.player_x = 10
    sim.state.player_y = 10
    sim.state.player_health = 3
    _add_ready_attacker(sim, EntityTypes.RANGER, x=10, y=14)

    with patch("tools.inferno_rl.simulator.npc_combat.roll_npc_damage", return_value=10):
        result = sim.step(0)

    assert sim.state.player_health == 0
    assert result.damage_taken == 3
    assert len(result.player_damage_events) == 1
    assert result.player_damage_events[0].damage == 3
