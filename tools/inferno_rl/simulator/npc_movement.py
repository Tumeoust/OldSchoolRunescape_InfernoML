"""
NPC pathfinding and movement.

NpcMovementMixin handles NPC movement dispatch, nibbler pillar-seeking,
meleer dig mechanics, and entity collision checking.
"""

import random
from typing import Tuple, List

from .entity import PlacedEntity, EntityTypes
from .forecast import (
    DIG_SEQUENCE_DURATION,
    DIG_TRIGGER_GUARANTEED,
    DIG_TRIGGER_RANDOM_THRESHOLD,
    POST_DIG_ATTACK_DELAY,
    POST_DIG_FROZEN_TICKS,
)
from .geometry import SimulatorGeometry, InfernoLineOfSight
from .pathfinding import OSRSPathfinding, NpcCollisionResolver


class NpcMovementMixin:
    """Mixin providing NPC movement and pathfinding."""

    def _process_npc_movement(self):
        """Process movement for all NPCs."""
        player_x = self.state.player_x
        player_y = self.state.player_y
        pillar_alive = self.state.pillar_alive

        for entity in self.state.entities:
            if entity.is_dead():
                continue

            # Nibblers move towards pillars
            if entity.entity_type == EntityTypes.NIBBLER:
                if entity.can_move():
                    self._process_nibbler_movement(entity)
                continue

            # Meleers have special dig mechanics
            if entity.entity_type == EntityTypes.MELEE:
                self._process_meleer_movement(entity, player_x, player_y, pillar_alive)
                continue

            # Skip frozen entities
            if not entity.can_move():
                continue

            # Create walkability checker for this entity
            def checker(x, y, size, ent=entity):
                return self._can_entity_move_to(ent, x, y, size)

            npc_size = entity.entity_type.size_in_tiles

            # Check if player is under this NPC - requires collision resolution
            if NpcCollisionResolver.is_player_under_npc(
                entity.x, entity.y, npc_size, player_x, player_y
            ):
                # Use collision resolution: random cardinal direction step-away
                dx, dy = NpcCollisionResolver.resolve_collision(
                    entity.x, entity.y, npc_size, checker
                )
                entity.x += dx
                entity.y += dy
                continue

            # Check if needs to move (has LOS = can attack = don't need to move)
            has_los = InfernoLineOfSight.can_entity_attack_player(
                entity, player_x, player_y, pillar_alive
            )
            if has_los:
                continue

            # Normal movement towards player
            new_x, new_y = OSRSPathfinding.simulate_npc_movement(
                entity.x, entity.y,
                player_x, player_y,
                npc_size,
                entity.entity_type.move_speed,
                checker
            )

            entity.x = new_x
            entity.y = new_y

    def _process_nibbler_movement(self, nibbler: PlacedEntity):
        """Process nibbler movement towards target pillar."""
        target_pillar = nibbler.target_pillar_index

        # If no target or target pillar is dead, nibbler does NOT move
        # (Nibbler will die when target pillar dies - see _handle_pillar_death)
        if target_pillar < 0 or not self.state.is_pillar_alive(target_pillar):
            return

        pillar_center = PlacedEntity.get_pillar_center(target_pillar)
        if pillar_center is None:
            return

        def checker(x, y, size):
            return SimulatorGeometry.is_valid_tile_for_size(x, y, size, self.state.pillar_alive)

        new_x, new_y = OSRSPathfinding.simulate_npc_movement(
            nibbler.x, nibbler.y,
            pillar_center[0], pillar_center[1],
            nibbler.entity_type.size_in_tiles,
            nibbler.entity_type.move_speed,
            checker
        )

        nibbler.x = new_x
        nibbler.y = new_y

    def _process_meleer_movement(self, meleer: PlacedEntity,
                                  player_x: int, player_y: int,
                                  pillar_alive: List[bool]):
        """
        Process meleer movement including dig mechanics.

        From InfernoTrainer JalImKot.ts:
        - Meleer digs when stuck without LOS for extended time
        - attackDelay goes negative when can't attack
        - Dig triggers at -38 (10% chance) or -50 (guaranteed)
        - Dig takes 6 ticks, then teleport near player
        - Post-dig: attackDelay=6, frozen=2
        """
        npc_size = meleer.entity_type.size_in_tiles

        def checker(x, y, size):
            return self._can_entity_move_to(meleer, x, y, size)

        # Check LOS for movement and dig decisions
        has_los = InfernoLineOfSight.can_entity_attack_player(
            meleer, player_x, player_y, pillar_alive
        )

        # Normal movement if not digging
        if meleer.dig_sequence_time == 0 and meleer.dig_location is None and meleer.can_move():
            # Check if player is under meleer - requires collision resolution
            if NpcCollisionResolver.is_player_under_npc(
                meleer.x, meleer.y, npc_size, player_x, player_y
            ):
                dx, dy = NpcCollisionResolver.resolve_collision(
                    meleer.x, meleer.y, npc_size, checker
                )
                meleer.x += dx
                meleer.y += dy
            elif not has_los:
                # Move towards player if no LOS
                new_x, new_y = OSRSPathfinding.simulate_npc_movement(
                    meleer.x, meleer.y,
                    player_x, player_y,
                    npc_size,
                    meleer.entity_type.move_speed,
                    checker
                )
                meleer.x = new_x
                meleer.y = new_y

        # Check if should start digging
        # From InfernoTrainer: dig when no LOS and attackDelay very negative
        # attackDelay <= -38 with 10% chance, or <= -50 guaranteed
        if meleer.dig_sequence_time == 0 and meleer.dig_location is None and not has_los:
            should_dig = False
            if meleer.attack_delay <= DIG_TRIGGER_GUARANTEED:
                should_dig = True
            elif meleer.attack_delay <= DIG_TRIGGER_RANDOM_THRESHOLD:
                should_dig = random.random() < 0.1

            if should_dig:
                # Start dig sequence
                meleer.frozen = DIG_SEQUENCE_DURATION
                meleer.dig_sequence_time = DIG_SEQUENCE_DURATION
                meleer.dig_location = self._calculate_dig_location(
                    meleer, player_x, player_y, npc_size
                )

        # Process dig tick
        if meleer.dig_sequence_time > 0:
            meleer.dig_sequence_time -= 1

            if meleer.dig_sequence_time == 0 and meleer.dig_location:
                # Dig complete - teleport to dig location
                meleer.x = meleer.dig_location[0]
                meleer.y = meleer.dig_location[1]
                meleer.dig_location = None
                meleer.attack_delay = POST_DIG_ATTACK_DELAY
                meleer.frozen = POST_DIG_FROZEN_TICKS

    def _calculate_dig_location(self, meleer: PlacedEntity,
                                 player_x: int, player_y: int,
                                 size: int) -> Tuple[int, int]:
        """
        Calculate dig destination using the 4-position priority from osrs-sdk.

        NPC collision does not prevent dig placement. Only bounds and pillars
        block it. Positions translated from osrs-sdk (+Y south) to our +Y north:
        1. SW of player: player at NE corner of NPC footprint
        2. On player: player at SW corner
        3. West of player: player at east edge
        4. South of player: player at north edge
        Fallback: slight SW offset, then spiral outward.
        """
        s = size - 1  # 3 for 4x4 Jal-ImKot
        candidates = [
            (player_x - s, player_y - s),  # Priority 1: SW
            (player_x, player_y),           # Priority 2: On player
            (player_x - s, player_y),       # Priority 3: West
            (player_x, player_y - s),       # Priority 4: South
        ]

        for x, y in candidates:
            if self._is_valid_dig_location(x, y, size):
                return (x, y)

        # Fallback: slight SW offset (matches osrs-sdk)
        fallback_x, fallback_y = player_x - 1, player_y - 1
        if self._is_valid_dig_location(fallback_x, fallback_y, size):
            return (fallback_x, fallback_y)

        # Spiral outward from player position checking all SW corners
        for radius in range(1, 13):
            best = None
            best_dist = float('inf')
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # Only check the ring perimeter
                    x = player_x + dx
                    y = player_y + dy
                    if self._is_valid_dig_location(x, y, size):
                        dist = abs(dx) + abs(dy)
                        if dist < best_dist:
                            best = (x, y)
                            best_dist = dist
            if best is not None:
                return best

        # Absolute last resort (should never happen in 29x30 arena)
        return (meleer.x, meleer.y)

    def _is_valid_dig_location(self, x: int, y: int, size: int) -> bool:
        """Check if a dig location is valid (bounds + pillar checks only).

        In live OSRS, NPC collision does not prevent dig placement.
        Melees always dig on top of the player.
        """
        if x < 0 or y < 0 or x + size > 29 or y + size > 30:
            return False

        return SimulatorGeometry.is_valid_tile_for_size(x, y, size, self.state.pillar_alive)

    def _can_entity_move_to(self, entity: PlacedEntity, x: int, y: int, size: int) -> bool:
        """Check if an entity can move to a position.

        Note: Does NOT check player collision. In OSRS, NPCs can occupy the same
        tiles as the player. Collision resolution handles the case where the player
        is under the NPC by having the NPC step away in a random cardinal direction.

        Stacked NPCs: If another entity already overlaps our current position
        (e.g. two melees dug to the same spot), skip collision against it so
        they can walk apart. Movement is processed sequentially, so the first
        NPC moves freely and the second sees it at its new position.
        """
        if not SimulatorGeometry.is_valid_tile_for_size(x, y, size, self.state.pillar_alive):
            return False

        for other in self.state.entities:
            if other == entity or other.is_dead():
                continue
            # Nibblers don't collide
            if (other.entity_type == EntityTypes.NIBBLER or
                entity.entity_type == EntityTypes.NIBBLER):
                continue

            other_size = other.entity_type.size_in_tiles

            # Skip collision against entities already overlapping our current
            # position — allows stacked NPCs to separate.
            if SimulatorGeometry.do_footprints_overlap(
                entity.x, entity.y, size,
                other.x, other.y, other_size
            ):
                continue

            if SimulatorGeometry.do_footprints_overlap(
                x, y, size,
                other.x, other.y, other_size
            ):
                return False
        return True
