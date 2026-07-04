"""
Geometry utilities and Line of Sight calculations for Inferno simulation.

Provides grid/world coordinate conversion, Chebyshev distance, pillar collision,
and the fixed-point Bresenham line-of-sight algorithm.
"""

import math
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Arena dimensions
GRID_WIDTH = 29
GRID_HEIGHT = 30
TILE_SIZE = 30  # Pixel size for rendering

# Arena offset (base coordinates in game world)
ARENA_BASE_X = 2257
ARENA_BASE_Y = 5329

# Pillar positions [x, y, width, height] of bottom-left corner
# NW pillar: center (1, 21), NE pillar: center (18, 23), S pillar: center (11, 7)
PILLARS = [
    (0, 20, 3, 3),   # NW pillar
    (17, 22, 3, 3),  # NE pillar
    (10, 6, 3, 3)    # S pillar
]

# NE pillar center - most safespots are in this area
NE_PILLAR_X = 18
NE_PILLAR_Y = 23

# NE pillar zone: tiles we reward during combat (safespot area). Pillar is 3x3 (not standable).
# - 7x7 ring around pillar (2-tile radius) minus the 3x3 pillar = 40 standable tiles.
# - North strip: columns 17-19, rows 27-29 (ring already covers y≤26).
# - West strip: columns 11-14, rows 22-24 (4 cols × 3 pillar rows, mirrors north extent).
_NE_PILLAR_7X7 = frozenset(
    (x, y) for x in range(15, 22) for y in range(20, 27)
)
_NE_PILLAR_3X3 = frozenset(
    (x, y) for x in range(17, 20) for y in range(22, 25)
)
_NE_PILLAR_ZONE_RING = _NE_PILLAR_7X7 - _NE_PILLAR_3X3  # 40 tiles where player can stand around pillar
_NE_PILLAR_ZONE_NORTH = frozenset(
    (x, y) for x in range(17, 20) for y in range(27, 30)
)
_NE_PILLAR_ZONE_WEST = frozenset(
    (x, y) for x in range(11, 15) for y in range(22, 25)
)
NE_PILLAR_ZONE = _NE_PILLAR_ZONE_RING | _NE_PILLAR_ZONE_NORTH | _NE_PILLAR_ZONE_WEST


def is_in_ne_pillar_zone(px: int, py: int) -> bool:
    """True if (px, py) is in the NE pillar safespot zone (ring around 3x3 pillar + north strip)."""
    return (px, py) in NE_PILLAR_ZONE


# A tile: two tiles north of the NW corner of the NE pillar.
# NE pillar NW corner = (17, 22 + 2) = (17, 24). Two north = (17, 26).
# Optimal position during the 9-tick grace period between waves.
A_TILE_X = 17
A_TILE_Y = 26

# LOS blocking masks (matching osrs-sdk LineOfSightMask)
LOS_NONE = 0
LOS_FULL_MASK = 131072
LOS_EAST_MASK = 4096
LOS_WEST_MASK = 65536
LOS_NORTH_MASK = 1024
LOS_SOUTH_MASK = 16384


class SimulatorGeometry:
    """Static utility methods for geometry calculations in the Inferno simulator."""

    @staticmethod
    def grid_to_world(x: int, y: int) -> Tuple[int, int]:
        """Convert grid coordinates to world coordinates."""
        return (ARENA_BASE_X + x, ARENA_BASE_Y + y)

    @staticmethod
    def world_to_grid(world_x: int, world_y: int) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        return (world_x - ARENA_BASE_X, world_y - ARENA_BASE_Y)

    @staticmethod
    def chebyshev_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        """Calculate Chebyshev distance (max of x/y distance)."""
        return max(abs(x1 - x2), abs(y1 - y2))

    @staticmethod
    def is_orthogonally_adjacent(target_x: int, target_y: int,
                                  npc_x: int, npc_y: int, npc_size: int) -> bool:
        """
        Check if a position is orthogonally adjacent to an NPC's hitbox.
        OSRS melee attacks only work from cardinal directions (N/S/E/W).
        """
        npc_max_x = npc_x + npc_size - 1
        npc_max_y = npc_y + npc_size - 1

        # Check West
        if target_x == npc_x - 1 and npc_y <= target_y <= npc_max_y:
            return True
        # Check East
        if target_x == npc_max_x + 1 and npc_y <= target_y <= npc_max_y:
            return True
        # Check South
        if target_y == npc_y - 1 and npc_x <= target_x <= npc_max_x:
            return True
        # Check North
        if target_y == npc_max_y + 1 and npc_x <= target_x <= npc_max_x:
            return True

        return False

    @staticmethod
    def is_on_pillar(x: int, y: int, pillar_alive: Optional[List[bool]] = None) -> bool:
        """
        Check if a tile is on a pillar.
        
        Args:
            x: Grid X coordinate
            y: Grid Y coordinate
            pillar_alive: Array of pillar alive states [NW, NE, S], or None to check all
        """
        for i, pillar in enumerate(PILLARS):
            # Skip dead pillars if alive state is provided
            if pillar_alive is not None and not pillar_alive[i]:
                continue

            px, py, pw, ph = pillar
            if px <= x < px + pw and py <= y < py + ph:
                return True
        return False

    @staticmethod
    def get_pillar_index_at(x: int, y: int) -> int:
        """Get the pillar index at a given position, or -1 if not on any pillar."""
        for i, pillar in enumerate(PILLARS):
            px, py, pw, ph = pillar
            if px <= x < px + pw and py <= y < py + ph:
                return i
        return -1

    @staticmethod
    def would_overlap_pillar(x: int, y: int, size: int,
                             pillar_alive: Optional[List[bool]] = None) -> bool:
        """Check if an entity of given size would overlap any pillar."""
        for dx in range(size):
            for dy in range(size):
                if SimulatorGeometry.is_on_pillar(x + dx, y + dy, pillar_alive):
                    return True
        return False

    @staticmethod
    def is_in_bounds(x: int, y: int) -> bool:
        """Check if coordinates are within arena bounds."""
        return 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT

    @staticmethod
    def is_valid_tile(x: int, y: int, pillar_alive: Optional[List[bool]] = None) -> bool:
        """Check if a tile is valid (in bounds and not on pillar)."""
        return (SimulatorGeometry.is_in_bounds(x, y) and
                not SimulatorGeometry.is_on_pillar(x, y, pillar_alive))

    @staticmethod
    def is_valid_tile_for_size(x: int, y: int, size: int,
                                pillar_alive: Optional[List[bool]] = None) -> bool:
        """Check if all tiles in an entity's footprint are valid."""
        for dx in range(size):
            for dy in range(size):
                if not SimulatorGeometry.is_valid_tile(x + dx, y + dy, pillar_alive):
                    return False
        return True

    @staticmethod
    def is_under_npc(player_x: int, player_y: int,
                     npc_x: int, npc_y: int, npc_size: int) -> bool:
        """Check if player is under (overlapping with) an NPC."""
        return (npc_x <= player_x < npc_x + npc_size and
                npc_y <= player_y < npc_y + npc_size)

    @staticmethod
    def compute_push_out_tile(
        player_x: int, player_y: int,
        npc_x: int, npc_y: int, npc_size: int,
        pillar_alive: Optional[List[bool]] = None,
    ) -> Tuple[int, int]:
        """Compute where a player gets pushed when standing on an NPC and attacking it.

        OSRS pushes the player to the nearest walkable tile outside the NPC footprint.
        Ties broken by loop order: South > West > East > North (matching osrs-sdk).
        """
        max_dist = math.ceil(npc_size / 2)
        best_x, best_y = player_x, player_y
        best_dist_sq = float("inf")
        for yy in range(-max_dist, max_dist):
            for xx in range(-max_dist, max_dist):
                if xx == 0 and yy == 0:
                    continue
                tx, ty = player_x + xx, player_y + yy
                if not SimulatorGeometry.is_valid_tile(tx, ty, pillar_alive):
                    continue
                if SimulatorGeometry.is_under_npc(tx, ty, npc_x, npc_y, npc_size):
                    continue
                dist_sq = xx * xx + yy * yy
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_x, best_y = tx, ty
        return (best_x, best_y)

    @staticmethod
    def do_footprints_overlap(x1: int, y1: int, size1: int,
                               x2: int, y2: int, size2: int) -> bool:
        """Check if two rectangular footprints overlap."""
        return (x1 < x2 + size2 and x1 + size1 > x2 and
                y1 < y2 + size2 and y1 + size1 > y2)

    @staticmethod
    def would_npc_overlap_player_at(npc_x: int, npc_y: int, npc_size: int,
                                     player_x: int, player_y: int) -> bool:
        """
        Check if an NPC at (npc_x, npc_y) would overlap a player at (player_x, player_y).
        Used for corner safespotting logic.

        Args:
            npc_x, npc_y: NPC's SW corner position to test
            npc_size: NPC size in tiles
            player_x, player_y: Player position (always size 1)

        Returns:
            True if NPC footprint would overlap player tile
        """
        return SimulatorGeometry.do_footprints_overlap(npc_x, npc_y, npc_size, player_x, player_y, 1)


class InfernoLineOfSight:
    """
    Line of Sight calculator matching OSRS mechanics.
    
    Uses fixed-point Bresenham algorithm for LOS calculations.
    Ray direction is NPC closest tile -> player, matching RuneLite's WorldArea.hasLineOfSightTo.
    A single ray direction is used for both NPC-attacks-player and player-attacks-NPC checks.
    """

    @staticmethod
    def get_los_mask(x: int, y: int, pillar_alive: Optional[List[bool]]) -> int:
        """
        Get the LOS blocking mask for a tile.
        Pillars are full blockers (FULL_MASK).
        """
        if SimulatorGeometry.is_on_pillar(x, y, pillar_alive):
            return LOS_FULL_MASK
        return LOS_NONE

    @staticmethod
    def collision_math(x1: int, y1: int, size1: int,
                       x2: int, y2: int, size2: int) -> bool:
        """Check if two rectangles overlap."""
        return (x1 < x2 + size2 and x1 + size1 > x2 and
                y1 < y2 + size2 and y1 + size1 > y2)

    @staticmethod
    def get_closest_point_on_npc(from_x: int, from_y: int,
                                  npc_x: int, npc_y: int, npc_size: int) -> Tuple[int, int]:
        """Find the closest point on an NPC to a given position."""
        closest_x = max(npc_x, min(npc_x + npc_size - 1, from_x))
        closest_y = max(npc_y, min(npc_y + npc_size - 1, from_y))
        return (closest_x, closest_y)

    @staticmethod
    def get_distance_from_npc(npc_x: int, npc_y: int, npc_size: int,
                              target_x: int, target_y: int) -> int:
        """Get Chebyshev distance from NPC's closest tile to target."""
        closest_x, closest_y = InfernoLineOfSight.get_closest_point_on_npc(
            target_x, target_y, npc_x, npc_y, npc_size
        )
        return max(abs(closest_x - target_x), abs(closest_y - target_y))

    @staticmethod
    def _trace_ray_fixed_point(x1: int, y1: int, x2: int, y2: int,
                                dx: int, dy: int, dx_abs: int, dy_abs: int,
                                pillar_alive: Optional[List[bool]]) -> bool:
        """
        Fixed-point Bresenham ray tracing algorithm matching osrs-sdk.
        Uses 16-bit fixed-point math for sub-tile precision.
        """
        if dx_abs > dy_abs:
            # X is the major axis
            x_tile = x1
            # Fixed-point Y: start at center of tile (0x8000 = 0.5)
            y = (y1 << 16) + 0x8000
            slope = (dy << 16) // dx_abs if dx_abs != 0 else 0

            x_inc = 1 if dx > 0 else -1
            x_mask = LOS_WEST_MASK | LOS_FULL_MASK if dx > 0 else LOS_EAST_MASK | LOS_FULL_MASK

            if dy < 0:
                y -= 1
                y_mask = LOS_NORTH_MASK | LOS_FULL_MASK
            else:
                y_mask = LOS_SOUTH_MASK | LOS_FULL_MASK

            while x_tile != x2:
                x_tile += x_inc
                y_tile = y >> 16

                mask = InfernoLineOfSight.get_los_mask(x_tile, y_tile, pillar_alive)
                if (mask & x_mask) != 0:
                    return False

                y += slope
                new_y_tile = y >> 16

                if new_y_tile != y_tile:
                    mask = InfernoLineOfSight.get_los_mask(x_tile, new_y_tile, pillar_alive)
                    if (mask & y_mask) != 0:
                        return False
        else:
            # Y is the major axis
            y_tile = y1
            x = (x1 << 16) + 0x8000
            slope = (dx << 16) // dy_abs if dy_abs != 0 else 0

            y_inc = 1 if dy > 0 else -1
            y_mask = LOS_SOUTH_MASK | LOS_FULL_MASK if dy > 0 else LOS_NORTH_MASK | LOS_FULL_MASK

            if dx < 0:
                x -= 1
                x_mask = LOS_EAST_MASK | LOS_FULL_MASK
            else:
                x_mask = LOS_WEST_MASK | LOS_FULL_MASK

            while y_tile != y2:
                y_tile += y_inc
                x_tile = x >> 16

                mask = InfernoLineOfSight.get_los_mask(x_tile, y_tile, pillar_alive)
                if (mask & y_mask) != 0:
                    return False

                x += slope
                new_x_tile = x >> 16

                if new_x_tile != x_tile:
                    mask = InfernoLineOfSight.get_los_mask(new_x_tile, y_tile, pillar_alive)
                    if (mask & x_mask) != 0:
                        return False

        return True

    @staticmethod
    def has_line_of_sight(x1: int, y1: int, x2: int, y2: int,
                          size: int, attack_range: int, is_npc: bool,
                          pillar_alive: Optional[List[bool]]) -> bool:
        """
        Core LOS check matching InfernoTrainer's hasLineOfSight function.
        
        Args:
            x1, y1: Source position
            x2, y2: Target position
            size: Source entity size (1 for player, NPC size for NPCs)
            attack_range: Attack range for distance check
            is_npc: True if source is an NPC
            pillar_alive: Pillar alive states
            
        Returns:
            True if LOS exists
        """
        dx = x2 - x1
        dy = y2 - y1

        # Check if source or target is blocked
        if (InfernoLineOfSight.get_los_mask(x1, y1, pillar_alive) != LOS_NONE or
            InfernoLineOfSight.get_los_mask(x2, y2, pillar_alive) != LOS_NONE or
            InfernoLineOfSight.collision_math(x1, y1, size, x2, y2, 1)):
            return False

        # Range 1 is melee - special adjacency check
        if attack_range == 1:
            return ((0 <= dx < size and (dy == size or dy == -1)) or
                    (0 <= dy < size and (dx == -1 or dx == size)))

        # For NPCs, find closest tile and trace ray from NPC tile -> player
        if is_npc:
            tx = max(x1, min(x1 + size - 1, x2))
            ty = max(y1, min(y1 + size - 1, y2))
            return InfernoLineOfSight.has_line_of_sight(
                tx, ty, x2, y2, 1, attack_range, False, pillar_alive
            )

        # Distance check
        dx_abs = abs(dx)
        dy_abs = abs(dy)
        if dx_abs > attack_range or dy_abs > attack_range:
            return False

        # Fixed-point Bresenham ray trace
        return InfernoLineOfSight._trace_ray_fixed_point(
            x1, y1, x2, y2, dx, dy, dx_abs, dy_abs, pillar_alive
        )

    @staticmethod
    def npc_has_los_to_player(npc_x: int, npc_y: int, npc_size: int,
                               player_x: int, player_y: int, attack_range: int,
                               pillar_alive: Optional[List[bool]]) -> bool:
        """Check if an NPC has line of sight to the player."""
        return InfernoLineOfSight.has_line_of_sight(
            npc_x, npc_y, player_x, player_y, npc_size, attack_range, True, pillar_alive
        )

    @staticmethod
    def player_has_los_to_npc(player_x: int, player_y: int,
                               npc_x: int, npc_y: int, npc_size: int, attack_range: int,
                               pillar_alive: Optional[List[bool]]) -> bool:
        """Check if the player has line of sight to an NPC.
        Uses the same NPC tile -> player ray as npc_has_los_to_player."""
        return InfernoLineOfSight.has_line_of_sight(
            npc_x, npc_y, player_x, player_y, npc_size, attack_range, True, pillar_alive
        )

    @staticmethod
    def can_entity_attack_player(entity, player_x: int, player_y: int,
                                  pillar_alive: Optional[List[bool]]) -> bool:
        """Check if an entity can attack the player (has LOS and is in range)."""
        size = entity.entity_type.size_in_tiles
        attack_range = entity.entity_type.attack_range

        # Check distance first
        distance = InfernoLineOfSight.get_distance_from_npc(
            entity.x, entity.y, size, player_x, player_y
        )
        if distance > attack_range:
            return False

        # Check LOS
        return InfernoLineOfSight.npc_has_los_to_player(
            entity.x, entity.y, size, player_x, player_y, attack_range, pillar_alive
        )

    @staticmethod
    def can_player_attack_entity(player_x: int, player_y: int, attack_range: int,
                                  entity, pillar_alive: Optional[List[bool]]) -> bool:
        """Check if the player can attack an entity."""
        size = entity.entity_type.size_in_tiles

        # Check distance first (optimization: skip expensive ray trace)
        distance = InfernoLineOfSight.get_distance_from_npc(
            entity.x, entity.y, size, player_x, player_y
        )
        if distance > attack_range:
            return False

        # Same NPC tile -> player ray direction as NPC attack checks
        return InfernoLineOfSight.has_line_of_sight(
            entity.x, entity.y, player_x, player_y,
            size, attack_range, True, pillar_alive
        )
