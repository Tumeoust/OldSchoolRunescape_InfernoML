"""
Player action execution, combat, and shared combat helpers.

PlayerActionsMixin handles action dispatch, movement, gear switching,
player attacks (single-target + barrage AoE), attack drag, and LOS queries.
"""

from typing import Optional, Tuple, List

from .entity import PlacedEntity, EntityTypes, AttackStyle
from .equipment import GearPreset
from .exact_targeting import get_exact_target_by_slot
from .geometry import SimulatorGeometry, InfernoLineOfSight
from .pathfinding import OSRSPathfinding
from .combat import roll_player_damage
from .forecast import is_player_melee_adjacent_to_npc

# Movement action params: action index -> (dx, dy, distance)
_MOVEMENT_PARAMS = {
    # North
    1: (0, 1, 1), 2: (0, 1, 2), 3: (0, 1, 3), 4: (0, 1, 4),
    # South
    5: (0, -1, 1), 6: (0, -1, 2), 7: (0, -1, 3), 8: (0, -1, 4),
    # East
    9: (1, 0, 1), 10: (1, 0, 2), 11: (1, 0, 3), 12: (1, 0, 4),
    # West
    13: (-1, 0, 1), 14: (-1, 0, 2), 15: (-1, 0, 3), 16: (-1, 0, 4),
    # Northeast
    17: (1, 1, 1), 18: (1, 1, 2), 19: (1, 1, 3), 20: (1, 1, 4),
    # Northwest
    21: (-1, 1, 1), 22: (-1, 1, 2), 23: (-1, 1, 3), 24: (-1, 1, 4),
    # Southeast
    25: (1, -1, 1), 26: (1, -1, 2), 27: (1, -1, 3), 28: (1, -1, 4),
    # Southwest
    29: (-1, -1, 1), 30: (-1, -1, 2), 31: (-1, -1, 3), 32: (-1, -1, 4),
}


class PlayerActionsMixin:
    """Mixin providing player action execution and combat helpers."""

    def _execute_action(self, action: int) -> bool:
        """Execute the given action. Returns True if valid."""
        if action == 0 or action == 47:  # STAY or NO_ACTION
            return True

        # Movement actions (1-32)
        if 1 <= action <= 32:
            return self._execute_movement(action)

        # Attack actions (33-46)
        if 33 <= action <= 46:
            return self._execute_attack(action)

        # Gear switch actions (48-51)
        if 48 <= action <= 51:
            return self._execute_gear_switch(action)

        return True

    def _execute_movement(self, action: int) -> bool:
        """Execute a movement action."""
        if action == 0:  # STAY
            return True

        # Calculate direction and distance from action index
        dx, dy, distance = self._get_movement_params(action)

        target_x = self.state.player_x + dx * distance
        target_y = self.state.player_y + dy * distance

        # Clamp to valid range
        target_x = max(0, min(28, target_x))
        target_y = max(0, min(29, target_y))

        # Simulate movement (2 tiles per tick when running)
        def checker(x, y, size):
            return SimulatorGeometry.is_valid_tile(x, y)

        new_x, new_y = OSRSPathfinding.simulate_player_movement(
            self.state.player_x, self.state.player_y,
            target_x, target_y,
            2,  # Running = 2 tiles per tick
            checker
        )

        self.state.player_x = new_x
        self.state.player_y = new_y

        # Movement clears attack target
        self.state.attack_target = None

        return True

    def _get_movement_params(self, action: int) -> Tuple[int, int, int]:
        """Get dx, dy, distance from movement action index."""
        return _MOVEMENT_PARAMS.get(action, (0, 0, 0))

    def _execute_attack(self, action: int) -> bool:
        """Execute an attack action."""
        target = self._resolve_attack_target(action)

        if target is None or target.is_dead():
            return False

        self.state.attack_target = target
        return True

    def _resolve_attack_target(self, action: int) -> Optional[PlacedEntity]:
        """Resolve the attack target based on action."""
        if 33 <= action <= 46:
            return get_exact_target_by_slot(self.state, action - 33)
        return None

    def _get_threat_priority(self, entity_type) -> int:
        """Get threat priority for an entity type (lower = higher priority)."""
        return entity_type.base_priority

    def _execute_gear_switch(self, action: int) -> bool:
        """Execute a gear switch action."""
        # Guard: blowpipe unavailable
        if action == 49 and not self.state.has_blowpipe:
            return False

        gear_map = {
            48: (GearPreset.BOFA, False),
            49: (GearPreset.BLOWPIPE, False),
            50: (GearPreset.MAGE, False),   # Ice barrage
            51: (GearPreset.MAGE, True),    # Blood barrage
        }

        entry = gear_map.get(action)
        if entry is None:
            return False

        preset, use_blood = entry

        # Allow switching between ice/blood even when already in mage preset
        if preset == GearPreset.MAGE:
            if self.state.current_preset == GearPreset.MAGE and self.state.use_blood_barrage == use_blood:
                return False
        else:
            if self.state.current_preset == preset:
                return False

        self.state.apply_gear_preset(preset)
        self.state.use_blood_barrage = use_blood
        return True

    def _process_player_attack(self):
        """Process player's attack."""
        target = self.state.attack_target
        if target is None or target.is_dead():
            return

        # Check attack cooldown
        if not self.state.can_player_attack():
            return

        # Check if can attack
        if not self._can_player_attack_entity(target):
            return

        # Record attack tick and weapon speed at attack time
        # This ensures weapon switching doesn't affect cooldown
        self.state.player_last_attack_tick = self.state.current_tick
        self.state.player_attack_speed_at_last_attack = self.state.player_attack_speed

        # Roll damage using OSRS accuracy + max hit formulas
        damage = roll_player_damage(self.state.current_preset, target.entity_type, self.combat_tables)

        # Apply damage
        target.take_damage(damage)
        self.damage_dealt_this_tick += damage
        if target.entity_type == EntityTypes.MAGER:
            self.damage_dealt_to_mager_this_tick += damage

        # Blood Barrage: heal 27.5% of damage dealt (Ancient Sceptre), cap at max HP
        if self.state.current_preset == GearPreset.MAGE and self.state.use_blood_barrage and damage > 0:
            heal = int(damage * 0.275)
            if heal > 0:
                self.state.player_health = min(
                    self.state.max_health,
                    self.state.player_health + heal
                )

        # Handle barrage AoE + freeze
        if self.state.current_preset == GearPreset.MAGE:
            # Freeze primary target (ice barrage only, and only on hit)
            if not self.state.use_blood_barrage and damage > 0 and not target.is_dead():
                target.frozen = 32
            self._process_barrage_aoe(target, damage)

    def _process_barrage_aoe(self, primary_target: PlacedEntity, primary_damage: int):
        """Process barrage AoE damage to nearby entities.

        Each AoE target independently rolls accuracy via roll_player_damage().
        Freeze only on hit (damage > 0) and only for ice barrage.
        """
        for entity in self.state.entities:
            if entity == primary_target or entity.is_dead():
                continue

            dist = SimulatorGeometry.chebyshev_distance(
                primary_target.x, primary_target.y,
                entity.x, entity.y
            )

            if dist <= 1:  # 3x3 AoE
                aoe_damage = roll_player_damage(self.state.current_preset, entity.entity_type, self.combat_tables)
                entity.take_damage(aoe_damage)
                self.damage_dealt_this_tick += aoe_damage
                if entity.entity_type == EntityTypes.MAGER:
                    self.damage_dealt_to_mager_this_tick += aoe_damage

                # Blood barrage AoE healing: 27.5% of AoE damage dealt
                if self.state.use_blood_barrage and aoe_damage > 0:
                    aoe_heal = int(aoe_damage * 0.275)
                    if aoe_heal > 0:
                        self.state.player_health = min(
                            self.state.max_health,
                            self.state.player_health + aoe_heal
                        )

                # Freeze effect for ice barrage (only on hit)
                if not self.state.use_blood_barrage and aoe_damage > 0:
                    entity.frozen = 32  # 32 ticks freeze

        if primary_target.is_dead():
            self.state.attack_target = None

    def _handle_attack_drag_if_needed(self):
        """Handle attack drag if player has target but can't attack."""
        if self.suppress_attack_drag_this_tick:
            return
        target = self.state.attack_target
        if target is None or target.is_dead():
            return

        if not self._can_player_attack_entity(target):
            self._handle_attack_drag(target)

    def _handle_attack_drag(self, target: PlacedEntity):
        """
        Move player towards attack target using OSRS attack-click behavior.

        OSRS behavior (from osrs-sdk Pathing.constructPaths):
        1. Generate ALL tiles on the NPC as valid endpoints
        2. Order endpoints with SW tile first (for tie-breaking)
        3. BFS finds shortest path to ANY endpoint
        4. Move up to 'speed' tiles along the path
        5. Stop early if can attack the target
        6. If path endpoint collides with NPC, skip that tile (path.shift())
        """
        if self._can_player_attack_entity(target):
            return

        new_x, new_y = self._predict_attack_drag_position(
            self.state.player_x, self.state.player_y, target
        )

        self.state.player_x = new_x
        self.state.player_y = new_y

    def _can_player_attack_entity(self, entity: PlacedEntity) -> bool:
        """Check if player can attack an entity."""
        return InfernoLineOfSight.can_player_attack_entity(
            self.state.player_x, self.state.player_y,
            self.state.player_attack_range,
            entity, self.state.pillar_alive
        )

    def _can_player_attack_entity_from(self, x: int, y: int, entity: PlacedEntity) -> bool:
        """Check if player can attack an entity from a specific position."""
        return InfernoLineOfSight.can_player_attack_entity(
            x, y, self.state.player_attack_range,
            entity, self.state.pillar_alive
        )

    def _count_npcs_with_los_to_player(self, player_x: int, player_y: int) -> int:
        """Count NPCs (excluding nibblers) that have LOS to the player at (player_x, player_y)."""
        count = 0
        for entity in self.state.entities:
            if entity.is_dead() or entity.entity_type == EntityTypes.NIBBLER:
                continue
            if InfernoLineOfSight.can_entity_attack_player(
                entity, player_x, player_y, self.state.pillar_alive
            ):
                count += 1
        return count

    def _is_player_melee_adjacent_to_npc(self, entity: PlacedEntity,
                                          player_x: int, player_y: int) -> bool:
        return is_player_melee_adjacent_to_npc(entity, player_x, player_y)
