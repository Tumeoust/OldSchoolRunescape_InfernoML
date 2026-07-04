"""
Auto prayer and position prediction.

PrayerPredictionMixin predicts the player's future position (including
attack-drag) and queues the correct protection prayer one tick ahead.
"""

from typing import Optional, Tuple, List

from .entity import PlacedEntity, EntityTypes, AttackStyle
from .geometry import SimulatorGeometry, InfernoLineOfSight
from .pathfinding import OSRSPathfinding
from .forecast import predict_npc_position
from .forecast import predict_auto_prayer_for_position


class PrayerPredictionMixin:
    """Mixin providing auto-prayer and position prediction."""

    def _process_auto_prayer(self, action: int):
        """
        Predict and queue prayer based on the model's action.

        In OSRS, prayer activated on tick N becomes active on tick N+1.
        So we predict what will happen next tick and queue prayer accordingly.
        """
        if not self.auto_prayer_enabled:
            return

        # Predict player position after action
        predicted_x, predicted_y = self._predict_player_position(action)

        pillar_alive = self.state.pillar_alive

        # Find highest priority threat that will attack next tick
        target_prayer = self._determine_prayer_for_position(
            predicted_x, predicted_y, pillar_alive
        )

        if target_prayer is not None:
            self.state.queue_prayer(target_prayer, self.state.current_tick + 1)

    def _predict_player_position(self, action: int) -> Tuple[int, int]:
        """Predict where player will be after executing action (including attack-drag)."""
        predicted_x, predicted_y = self._predict_position_after_action(action)
        predicted_target = self._predict_attack_target_after_action(action)

        if predicted_target is None or predicted_target.is_dead():
            return (predicted_x, predicted_y)

        if self._can_player_attack_entity_from(predicted_x, predicted_y, predicted_target):
            return (predicted_x, predicted_y)

        return self._predict_attack_drag_position(predicted_x, predicted_y, predicted_target)

    def _predict_position_after_action(self, action: int) -> Tuple[int, int]:
        """Predict player position after the action's direct effect (no attack-drag)."""
        if action == 0 or action > 32:  # Non-movement
            return (self.state.player_x, self.state.player_y)

        dx, dy, distance = self._get_movement_params(action)

        target_x = self.state.player_x + dx * distance
        target_y = self.state.player_y + dy * distance

        # Clamp and simulate
        target_x = max(0, min(28, target_x))
        target_y = max(0, min(29, target_y))

        def checker(x, y, size):
            return SimulatorGeometry.is_valid_tile(x, y)

        return OSRSPathfinding.simulate_player_movement(
            self.state.player_x, self.state.player_y,
            target_x, target_y, 2, checker
        )

    def _predict_attack_target_after_action(self, action: int) -> Optional[PlacedEntity]:
        """Predict attack target after action (mirrors action execution rules)."""
        if 1 <= action <= 32:
            return None

        if 33 <= action <= 46:
            resolved = self._resolve_attack_target(action)
            if resolved is not None and not resolved.is_dead():
                return resolved
            return self.state.attack_target

        return self.state.attack_target

    def _predict_attack_drag_position(self, player_x: int, player_y: int,
                                      target: PlacedEntity) -> Tuple[int, int]:
        """Predict player movement from attack-drag toward target."""
        npc_size = target.entity_type.size_in_tiles

        # Push-out: if player is standing inside the target NPC, OSRS pushes
        # the player to the nearest walkable tile outside the NPC footprint
        # (deterministic: South > West > East > North tiebreak).
        if SimulatorGeometry.is_under_npc(
            player_x, player_y, target.x, target.y, npc_size
        ):
            push_x, push_y = SimulatorGeometry.compute_push_out_tile(
                player_x, player_y,
                target.x, target.y, npc_size,
                self.state.pillar_alive,
            )
            if push_x != player_x or push_y != player_y:
                return (push_x, push_y)

        # Generate ALL tiles on the NPC as valid endpoints
        endpoints = []
        for dy in range(npc_size):
            for dx in range(npc_size):
                endpoints.append((target.x + dx, target.y + dy))

        def checker(x, y, size):
            return SimulatorGeometry.is_valid_tile(x, y, self.state.pillar_alive)

        path = OSRSPathfinding.find_player_path_to_endpoints(
            player_x, player_y, endpoints, checker
        )

        if not path:
            return (player_x, player_y)

        if path and SimulatorGeometry.do_footprints_overlap(
            path[-1][0], path[-1][1], 1,
            target.x, target.y, npc_size
        ):
            path.pop()

        if not path:
            return (player_x, player_y)

        tiles_moved = 0
        for step_x, step_y in path:
            if tiles_moved >= 2:  # Running = 2 tiles per tick
                break

            if self._can_player_attack_entity_from(player_x, player_y, target):
                break

            player_x, player_y = step_x, step_y
            tiles_moved += 1

        return (player_x, player_y)

    def _determine_prayer_for_position(self, player_x: int, player_y: int,
                                        pillar_alive: List[bool]) -> Optional[str]:
        return predict_auto_prayer_for_position(
            self.state.entities,
            player_x,
            player_y,
            pillar_alive,
            self.state.active_prayer,
        )

    def _predict_npc_position(self, entity: PlacedEntity,
                               player_x: int, player_y: int,
                               pillar_alive: List[bool]) -> Tuple[int, int]:
        """
        Predict where an NPC will be after movement.

        Movement happens AFTER timer decrement, so we predict future frozen state.
        """
        return predict_npc_position(entity, player_x, player_y, pillar_alive)
