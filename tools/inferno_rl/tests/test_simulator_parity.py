"""
Tests to verify core Inferno simulator mechanics behave correctly.

These tests validate that critical mechanics work identically:
- NPC movement (dumb pathfinding)
- Line of sight calculations
- Attack timing and cooldowns
- Prayer protection
- Wave progression
- Exact target-slot resolution
"""

import unittest
import numpy as np

from ..simulator.entity import EntityTypes, PlacedEntity
from ..simulator.exact_targeting import get_exact_target_slots
from ..simulator.state import SimulatorState, spawn_wave_entities, WAVE_SPAWN_DELAY
from ..simulator.geometry import SimulatorGeometry, InfernoLineOfSight
from ..simulator.pathfinding import OSRSPathfinding
from ..simulator.simulator import InfernoSimulator
from ..training.actions import InfernoAction


class TestGeometry(unittest.TestCase):
    """Test geometry calculations."""
    
    def test_chebyshev_distance(self):
        """Test Chebyshev distance calculation."""
        self.assertEqual(SimulatorGeometry.chebyshev_distance(0, 0, 3, 4), 4)
        self.assertEqual(SimulatorGeometry.chebyshev_distance(0, 0, 5, 5), 5)
        self.assertEqual(SimulatorGeometry.chebyshev_distance(0, 0, 0, 0), 0)
    
    def test_is_on_pillar(self):
        """Test pillar collision detection."""
        # NW pillar is at (0, 20) 3x3
        self.assertTrue(SimulatorGeometry.is_on_pillar(0, 20))
        self.assertTrue(SimulatorGeometry.is_on_pillar(1, 21))
        self.assertTrue(SimulatorGeometry.is_on_pillar(2, 22))
        self.assertFalse(SimulatorGeometry.is_on_pillar(3, 20))
        
        # Test with pillar alive array
        pillar_alive = [False, True, True]  # NW dead
        self.assertFalse(SimulatorGeometry.is_on_pillar(0, 20, pillar_alive))
        self.assertTrue(SimulatorGeometry.is_on_pillar(17, 22, pillar_alive))  # NE alive
    
    def test_is_valid_tile(self):
        """Test tile validity."""
        self.assertTrue(SimulatorGeometry.is_valid_tile(10, 10))
        self.assertFalse(SimulatorGeometry.is_valid_tile(-1, 10))
        self.assertFalse(SimulatorGeometry.is_valid_tile(30, 10))
        self.assertFalse(SimulatorGeometry.is_valid_tile(0, 20))  # On pillar


class TestLineOfSight(unittest.TestCase):
    """Test line of sight calculations."""
    
    def test_basic_los(self):
        """Test basic LOS without obstacles."""
        # Clear LOS
        pillar_alive = [True, True, True]
        self.assertTrue(InfernoLineOfSight.has_line_of_sight(
            10, 10, 15, 10, 1, 10, False, pillar_alive
        ))
    
    def test_los_blocked_by_pillar(self):
        """Test LOS blocked by pillar."""
        pillar_alive = [True, True, True]
        # NW pillar is at (0, 20) - shoot through it
        self.assertFalse(InfernoLineOfSight.has_line_of_sight(
            0, 15, 0, 25, 1, 15, False, pillar_alive
        ))
    
    def test_melee_range_los(self):
        """Test melee range (range=1) LOS."""
        pillar_alive = [True, True, True]
        # Adjacent (orthogonally)
        self.assertTrue(InfernoLineOfSight.has_line_of_sight(
            10, 10, 10, 11, 1, 1, False, pillar_alive
        ))
        # Diagonal is NOT melee range
        self.assertFalse(InfernoLineOfSight.has_line_of_sight(
            10, 10, 11, 11, 1, 1, False, pillar_alive
        ))

    def test_corner_graze_eastward(self):
        """Ray (15,26)->(21,24) grazes NE pillar corner at (19,24).
        RuneLite allows this -- the extra corner check was an earlier-simulator bug."""
        pillar_alive = [True, True, True]
        self.assertTrue(InfernoLineOfSight.has_line_of_sight(
            15, 26, 21, 24, 1, 104, False, pillar_alive
        ))

    def test_corner_graze_westward(self):
        """Ray (21,26)->(15,24) grazes NE pillar corner at (17,24).
        RuneLite allows this -- the extra corner check was an earlier-simulator bug."""
        pillar_alive = [True, True, True]
        self.assertTrue(InfernoLineOfSight.has_line_of_sight(
            21, 26, 15, 24, 1, 104, False, pillar_alive
        ))

    def test_ray_through_pillar_body_blocked(self):
        """Ray (15,26)->(21,22) passes through NE pillar body -- must be blocked."""
        pillar_alive = [True, True, True]
        self.assertFalse(InfernoLineOfSight.has_line_of_sight(
            15, 26, 21, 22, 1, 104, False, pillar_alive
        ))


class TestPathfinding(unittest.TestCase):
    """Test OSRS pathfinding mechanics."""
    
    def test_npc_movement_direct(self):
        """Test NPC moves directly toward target."""
        def checker(x, y, size):
            return SimulatorGeometry.is_valid_tile(x, y)
        
        # Move toward target
        dx, dy = OSRSPathfinding.calculate_next_move(10, 10, 15, 15, 1, checker)
        self.assertEqual((dx, dy), (1, 1))  # Diagonal toward target
    
    def test_npc_blocked_diagonal(self):
        """Test NPC falls back to X cardinal when diagonal blocked."""
        def checker(x, y, size):
            if x == 11 and y == 11:  # Block diagonal
                return False
            return SimulatorGeometry.is_valid_tile(x, y)

        dx, dy = OSRSPathfinding.calculate_next_move(10, 10, 15, 15, 1, checker)
        # OSRS always tries X first when diagonal blocked
        self.assertEqual((dx, dy), (1, 0))

    def test_cardinal_always_x_first(self):
        """X cardinal is always prioritized over Y, even when Y gap is larger."""
        def checker(x, y, size):
            # Block diagonal (11,11) but allow both X (11,10) and Y (10,11)
            if x == 11 and y == 11:
                return False
            return SimulatorGeometry.is_valid_tile(x, y)

        # Target at (12, 20): y_dist=10 >> x_dist=2, but X must still win
        dx, dy = OSRSPathfinding.calculate_next_move(10, 10, 12, 20, 1, checker)
        self.assertEqual((dx, dy), (1, 0))

    def test_chebyshev_guard_blocks_y_at_distance_1(self):
        """At Chebyshev distance 1, Y-only movement is blocked (RuneLite guard)."""
        def checker(x, y, size):
            # Block diagonal (11,11) AND X cardinal (11,10)
            if (x == 11 and y == 11) or (x == 11 and y == 10):
                return False
            return SimulatorGeometry.is_valid_tile(x, y)

        # NPC at (10,10), target at (11,11): Chebyshev = 1
        # Diagonal blocked, X blocked → NPC should stay put (no Y move)
        dx, dy = OSRSPathfinding.calculate_next_move(10, 10, 11, 11, 1, checker)
        self.assertEqual((dx, dy), (0, 0))

    def test_y_allowed_at_distance_greater_than_1(self):
        """At Chebyshev distance > 1, Y-only movement is allowed as fallback."""
        def checker(x, y, size):
            # Block diagonal AND X cardinal
            if (x == 11 and y == 13) or (x == 11 and y == 12):
                return False
            return SimulatorGeometry.is_valid_tile(x, y)

        # NPC at (10,12), target at (11,14): Chebyshev = max(1,2) = 2 > 1
        dx, dy = OSRSPathfinding.calculate_next_move(10, 12, 11, 14, 1, checker)
        self.assertEqual((dx, dy), (0, 1))

    def test_dig_location_priority_order(self):
        """Dig location follows 4-position priority from osrs-sdk."""
        from ..simulator.npc_movement import NpcMovementMixin
        from ..simulator.state import SimulatorState

        sim = type('Sim', (NpcMovementMixin,), {})()
        sim.state = SimulatorState()
        sim.state.pillar_alive = [True, True, True]

        melee = PlacedEntity(EntityTypes.MELEE, 5, 5, 0)
        size = EntityTypes.MELEE.size_in_tiles  # 4

        # Player in open area — priority 1 (SW) should win
        loc = sim._calculate_dig_location(melee, 15, 15, size)
        self.assertEqual(loc, (15 - 3, 15 - 3))  # SW: (12, 12)

        # Player near SW arena corner where SW position is out of bounds
        # Priority 1 (px-3, py-3) = (-1, -1) → invalid
        # Priority 2 (px, py) = (2, 2) → valid
        loc = sim._calculate_dig_location(melee, 2, 2, size)
        self.assertEqual(loc, (2, 2))  # On player

    def test_player_bfs_pathfinding(self):
        """Test player uses BFS pathfinding."""
        def checker(x, y, size):
            return SimulatorGeometry.is_valid_tile(x, y)
        
        path = OSRSPathfinding.find_player_path(10, 10, 15, 15, checker)
        
        # Path should exist and be efficient
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[-1], (15, 15))


class TestSimulator(unittest.TestCase):
    """Test main simulator mechanics."""
    
    def test_reset(self):
        """Test simulator reset."""
        sim = InfernoSimulator(start_wave=35, max_wave=49)
        sim.reset()
        
        self.assertEqual(sim.state.current_wave, 35)
        self.assertEqual(sim.state.player_health, 99)
        self.assertTrue(len(sim.state.entities) > 0)
    
    def test_wave_spawn(self):
        """Test wave spawning."""
        state = SimulatorState()
        state.current_tick = 1
        
        # Spawn wave 35 (first mager wave)
        success = spawn_wave_entities(state, 35)
        
        self.assertTrue(success)
        self.assertEqual(state.current_wave, 35)
        
        # Should have nibblers and 1 mager
        nibblers = [e for e in state.entities if e.entity_type == EntityTypes.NIBBLER]
        magers = [e for e in state.entities if e.entity_type == EntityTypes.MAGER]
        
        self.assertEqual(len(nibblers), 3)
        self.assertEqual(len(magers), 1)
    
    def test_exact_target_slot_1_resolves_to_the_first_observed_entity(self):
        """Test ATTACK_TARGET_1 resolves to the same entity as exact slot 1."""
        sim = InfernoSimulator()
        sim.reset_to_wave(35)
        
        # Skip initial barrage heuristic
        sim.step(0)  # STAY
        sim.step(0)  # STAY
        sim.step(0)  # STAY
        
        target_slots = get_exact_target_slots(sim.state)
        self.assertTrue(target_slots)
        self.assertEqual(target_slots[0].entity_type, EntityTypes.MAGER)

        sim.step(InfernoAction.ATTACK_TARGET_1)

        target = sim.state.attack_target
        self.assertIsNotNone(target)
        self.assertEqual(target.id, target_slots[0].id)
    
    def test_exact_target_nibbler_slot_can_be_selected(self):
        """Test a nibbler exact target slot resolves correctly."""
        sim = InfernoSimulator()
        sim.reset_to_wave(35)
        
        # Skip initial barrage
        for _ in range(3):
            sim.step(0)
        
        target_slots = get_exact_target_slots(sim.state)
        nibbler_slot_index = next(
            index for index, entity in enumerate(target_slots)
            if entity.entity_type == EntityTypes.NIBBLER
        )
        action = InfernoAction.action_for_target_index(nibbler_slot_index)

        sim._capture_pre_step_state()
        resolved_target = sim._resolve_attack_target(action)
        self.assertIsNotNone(resolved_target)
        self.assertEqual(resolved_target.id, target_slots[nibbler_slot_index].id)
        self.assertEqual(resolved_target.entity_type, EntityTypes.NIBBLER)

        result = sim.step(action)
        self.assertTrue(result.action_was_valid)

    def test_wave_63_blob_slot_can_be_selected_outside_old_top_three(self):
        """Test a wave-63 blob can be targeted even when it is outside the old priority-3 window."""
        sim = InfernoSimulator(start_wave=63, max_wave=63)
        state = sim.state
        state.current_wave = 63
        state.current_tick = 50
        state.wave_complete_timer = -1
        state.pillar_hp = [255, 255, 255]
        state.pillar_alive = [True, True, True]
        state.player_x = 16
        state.player_y = 23

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

        target_slots = get_exact_target_slots(state)
        blob_slot_index = max(
            index for index, entity in enumerate(target_slots)
            if entity.entity_type == EntityTypes.BLOB
        )
        self.assertGreaterEqual(blob_slot_index, 3)

        sim.step(InfernoAction.action_for_target_index(blob_slot_index))

        target = sim.state.attack_target
        self.assertIsNotNone(target)
        self.assertEqual(target.id, target_slots[blob_slot_index].id)
    
    def test_action_validity(self):
        """Test action validation."""
        sim = InfernoSimulator()
        sim.reset_to_wave(35)
        
        # Skip initial barrage
        for _ in range(3):
            sim.step(0)
        
        # Movement should be valid
        result = sim.step(1)  # MOVE_N_1
        self.assertTrue(result.action_was_valid)
        
        # Attack with no valid target should be invalid
        # (depends on state, but test the mechanism)
    
    def test_wave_progression(self):
        """Test wave progression after clearing uses the 9-tick grace period."""
        sim = InfernoSimulator(start_wave=35, max_wave=49)
        sim.reset()

        initial_wave = sim.state.current_wave
        initial_pos = (sim.state.player_x, sim.state.player_y)

        # Kill all entities so the next step enters grace period.
        for entity in sim.state.entities:
            entity.take_damage(entity.current_health + 100)

        first_clear = sim.step(0)
        self.assertTrue(first_clear.wave_completed)
        self.assertEqual(sim.state.current_wave, initial_wave)
        self.assertEqual(sim.state.wave_complete_timer, WAVE_SPAWN_DELAY)
        self.assertEqual((sim.state.player_x, sim.state.player_y), initial_pos)

        # Wave should not spawn until the full grace countdown expires.
        for expected in range(WAVE_SPAWN_DELAY - 1, 0, -1):
            result = sim.step(0)
            self.assertFalse(result.wave_completed)
            self.assertEqual(sim.state.current_wave, initial_wave)
            self.assertEqual(sim.state.wave_complete_timer, expected)
            self.assertEqual((sim.state.player_x, sim.state.player_y), initial_pos)

        sim.step(0)
        self.assertEqual(sim.state.current_wave, initial_wave + 1)
        self.assertEqual(sim.state.wave_complete_timer, -1)
        self.assertEqual((sim.state.player_x, sim.state.player_y), initial_pos)



class TestPriorityOrdering(unittest.TestCase):
    """Test threat priority ordering."""
    
    def test_priority_order(self):
        """Test threat priority (base_priority): Mager > Ranger > Blob > Melee > Bat."""
        sim = InfernoSimulator()

        # Manually verify priority values (base_priority: lower = higher threat)
        priorities = {
            EntityTypes.MAGER: sim._get_threat_priority(EntityTypes.MAGER),
            EntityTypes.RANGER: sim._get_threat_priority(EntityTypes.RANGER),
            EntityTypes.MELEE: sim._get_threat_priority(EntityTypes.MELEE),
            EntityTypes.BLOB: sim._get_threat_priority(EntityTypes.BLOB),
            EntityTypes.BAT: sim._get_threat_priority(EntityTypes.BAT),
        }

        # Verify ordering: MAGER(3) < RANGER(4) < BLOB(5) < MELEE(7) < BAT(8)
        self.assertLess(priorities[EntityTypes.MAGER], priorities[EntityTypes.RANGER])
        self.assertLess(priorities[EntityTypes.RANGER], priorities[EntityTypes.BLOB])
        self.assertLess(priorities[EntityTypes.BLOB], priorities[EntityTypes.MELEE])
        self.assertLess(priorities[EntityTypes.MELEE], priorities[EntityTypes.BAT])


if __name__ == "__main__":
    unittest.main()
