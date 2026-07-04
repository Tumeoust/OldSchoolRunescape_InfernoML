# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
"""
Geometry utilities and Line of Sight calculations for Inferno simulation.
Cython-optimized version of geometry.py — identical public API.

When compiled to .pyd, Python's import resolution prefers the compiled
extension over the .py file. When not compiled, the pure Python fallback
(geometry.py) is used automatically.
"""

from typing import Optional, List, Tuple

# ========================================================================
# PYTHON-LEVEL CONSTANTS (must match geometry.py exactly)
# ========================================================================

GRID_WIDTH = 29
GRID_HEIGHT = 30
TILE_SIZE = 30

ARENA_BASE_X = 2257
ARENA_BASE_Y = 5329

PILLARS = [
    (0, 20, 3, 3),   # NW pillar
    (17, 22, 3, 3),  # NE pillar
    (10, 6, 3, 3)    # S pillar
]

NE_PILLAR_X = 18
NE_PILLAR_Y = 23

A_TILE_X = 17
A_TILE_Y = 26

LOS_NONE = 0
LOS_FULL_MASK = 131072
LOS_EAST_MASK = 4096
LOS_WEST_MASK = 65536
LOS_NORTH_MASK = 1024
LOS_SOUTH_MASK = 16384

# NE pillar zone (frozensets for O(1) lookup — computed once at import)
_NE_PILLAR_7X7 = frozenset(
    (x, y) for x in range(15, 22) for y in range(20, 27)
)
_NE_PILLAR_3X3 = frozenset(
    (x, y) for x in range(17, 20) for y in range(22, 25)
)
_NE_PILLAR_ZONE_RING = _NE_PILLAR_7X7 - _NE_PILLAR_3X3
_NE_PILLAR_ZONE_NORTH = frozenset(
    (x, y) for x in range(17, 20) for y in range(27, 30)
)
_NE_PILLAR_ZONE_WEST = frozenset(
    (x, y) for x in range(11, 15) for y in range(22, 25)
)
NE_PILLAR_ZONE = _NE_PILLAR_ZONE_RING | _NE_PILLAR_ZONE_NORTH | _NE_PILLAR_ZONE_WEST


def is_in_ne_pillar_zone(int px, int py):
    """True if (px, py) is in the NE pillar safespot zone."""
    return (px, py) in NE_PILLAR_ZONE


# ========================================================================
# C-LEVEL CONSTANTS (avoid Python dict/list lookups in hot path)
# ========================================================================

cdef int C_GRID_WIDTH = 29
cdef int C_GRID_HEIGHT = 30

cdef int C_LOS_NONE = 0
cdef int C_LOS_FULL_MASK = 131072
cdef int C_LOS_EAST_MASK = 4096
cdef int C_LOS_WEST_MASK = 65536
cdef int C_LOS_NORTH_MASK = 1024
cdef int C_LOS_SOUTH_MASK = 16384

# Pillar data as C arrays (avoid Python tuple unpacking in inner loops)
cdef int C_PILLAR_X[3]
cdef int C_PILLAR_Y[3]
cdef int C_PILLAR_W[3]
cdef int C_PILLAR_H[3]

C_PILLAR_X[0] = 0;  C_PILLAR_Y[0] = 20; C_PILLAR_W[0] = 3; C_PILLAR_H[0] = 3
C_PILLAR_X[1] = 17; C_PILLAR_Y[1] = 22; C_PILLAR_W[1] = 3; C_PILLAR_H[1] = 3
C_PILLAR_X[2] = 10; C_PILLAR_Y[2] = 6;  C_PILLAR_W[2] = 3; C_PILLAR_H[2] = 3


# ========================================================================
# C-LEVEL FUNCTIONS (cdef = no Python call overhead, pure C dispatch)
# ========================================================================

cdef bint c_is_on_pillar(int x, int y, list pillar_alive):
    """Check if tile is on any alive pillar. pillar_alive may be None."""
    cdef int i, px, py, pw, ph
    for i in range(3):
        if pillar_alive is not None and not pillar_alive[i]:
            continue
        px = C_PILLAR_X[i]
        py = C_PILLAR_Y[i]
        pw = C_PILLAR_W[i]
        ph = C_PILLAR_H[i]
        if px <= x < px + pw and py <= y < py + ph:
            return True
    return False


cdef int c_get_los_mask(int x, int y, list pillar_alive):
    """Get LOS blocking mask for a tile."""
    if c_is_on_pillar(x, y, pillar_alive):
        return C_LOS_FULL_MASK
    return C_LOS_NONE


cdef bint c_is_in_bounds(int x, int y):
    return 0 <= x < C_GRID_WIDTH and 0 <= y < C_GRID_HEIGHT


cdef bint c_is_valid_tile(int x, int y, list pillar_alive):
    return c_is_in_bounds(x, y) and not c_is_on_pillar(x, y, pillar_alive)


cdef bint c_is_valid_tile_for_size(int x, int y, int size, list pillar_alive):
    cdef int dx, dy
    for dx in range(size):
        for dy in range(size):
            if not c_is_valid_tile(x + dx, y + dy, pillar_alive):
                return False
    return True


cdef bint c_do_footprints_overlap(int x1, int y1, int s1, int x2, int y2, int s2):
    return (x1 < x2 + s2 and x1 + s1 > x2 and
            y1 < y2 + s2 and y1 + s1 > y2)


cdef bint c_would_overlap_pillar(int x, int y, int size, list pillar_alive):
    cdef int dx, dy
    for dx in range(size):
        for dy in range(size):
            if c_is_on_pillar(x + dx, y + dy, pillar_alive):
                return True
    return False


cdef bint c_would_npc_overlap_player_at(int nx, int ny, int ns, int px, int py):
    """Check if NPC footprint at (nx,ny) with size ns overlaps player at (px,py)."""
    return c_do_footprints_overlap(nx, ny, ns, px, py, 1)


cdef (int, int) c_compute_push_out_tile(int px, int py, int nx, int ny, int ns, list pillar_alive):
    """Compute push-out tile when player stands on NPC and attacks.

    Returns nearest walkable tile outside NPC footprint.
    Tiebreak: South > West > East > North (loop order).
    """
    cdef int max_dist = (ns + 1) // 2  # ceil(ns / 2)
    cdef int best_x = px, best_y = py
    cdef int best_dist_sq = 2147483647  # INT_MAX
    cdef int xx, yy, tx, ty, dist_sq
    for yy in range(-max_dist, max_dist):
        for xx in range(-max_dist, max_dist):
            if xx == 0 and yy == 0:
                continue
            tx = px + xx
            ty = py + yy
            if not c_is_valid_tile(tx, ty, pillar_alive):
                continue
            # Must be outside the NPC footprint
            if nx <= tx < nx + ns and ny <= ty < ny + ns:
                continue
            dist_sq = xx * xx + yy * yy
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_x = tx
                best_y = ty
    return (best_x, best_y)


cdef int c_chebyshev_distance(int x1, int y1, int x2, int y2):
    cdef int dx = x1 - x2
    cdef int dy = y1 - y2
    if dx < 0:
        dx = -dx
    if dy < 0:
        dy = -dy
    return dx if dx > dy else dy


cdef bint c_trace_ray_fixed_point(int x1, int y1, int x2, int y2,
                                   int dx, int dy, int dx_abs, int dy_abs,
                                   list pillar_alive):
    """
    Fixed-point Bresenham ray tracing matching osrs-sdk.
    Uses 16-bit fixed-point for sub-tile precision.
    long long for accumulator variables to prevent overflow during slope accumulation.
    """
    cdef int x_tile, y_tile, new_y_tile, new_x_tile
    cdef int x_inc, y_inc
    cdef int x_mask, y_mask
    cdef int mask
    # long long accumulators — safe for fixed-point math over 29-tile grid
    cdef long long y_fp, x_fp, slope

    if dx_abs > dy_abs:
        # X is the major axis
        x_tile = x1
        y_fp = (<long long>y1 << 16) + 0x8000
        # Floor division (matching Python //) — dx_abs is always positive
        slope = (<long long>dy << 16) // <long long>dx_abs if dx_abs != 0 else 0

        x_inc = 1 if dx > 0 else -1
        x_mask = (C_LOS_WEST_MASK | C_LOS_FULL_MASK) if dx > 0 else (C_LOS_EAST_MASK | C_LOS_FULL_MASK)

        if dy < 0:
            y_fp -= 1
            y_mask = C_LOS_NORTH_MASK | C_LOS_FULL_MASK
        else:
            y_mask = C_LOS_SOUTH_MASK | C_LOS_FULL_MASK

        while x_tile != x2:
            x_tile += x_inc
            y_tile = <int>(y_fp >> 16)

            mask = c_get_los_mask(x_tile, y_tile, pillar_alive)
            if (mask & x_mask) != 0:
                return False

            y_fp += slope
            new_y_tile = <int>(y_fp >> 16)

            if new_y_tile != y_tile:
                mask = c_get_los_mask(x_tile, new_y_tile, pillar_alive)
                if (mask & y_mask) != 0:
                    return False
    else:
        # Y is the major axis
        y_tile = y1
        x_fp = (<long long>x1 << 16) + 0x8000
        slope = (<long long>dx << 16) // <long long>dy_abs if dy_abs != 0 else 0

        y_inc = 1 if dy > 0 else -1
        y_mask = (C_LOS_SOUTH_MASK | C_LOS_FULL_MASK) if dy > 0 else (C_LOS_NORTH_MASK | C_LOS_FULL_MASK)

        if dx < 0:
            x_fp -= 1
            x_mask = C_LOS_EAST_MASK | C_LOS_FULL_MASK
        else:
            x_mask = C_LOS_WEST_MASK | C_LOS_FULL_MASK

        while y_tile != y2:
            y_tile += y_inc
            x_tile = <int>(x_fp >> 16)

            mask = c_get_los_mask(x_tile, y_tile, pillar_alive)
            if (mask & y_mask) != 0:
                return False

            x_fp += slope
            new_x_tile = <int>(x_fp >> 16)

            if new_x_tile != x_tile:
                mask = c_get_los_mask(new_x_tile, y_tile, pillar_alive)
                if (mask & x_mask) != 0:
                    return False

    return True


cdef bint c_has_line_of_sight(int x1, int y1, int x2, int y2,
                               int size, int attack_range, bint is_npc,
                               list pillar_alive):
    """Core LOS check matching InfernoTrainer's hasLineOfSight."""
    cdef int dx = x2 - x1
    cdef int dy = y2 - y1
    cdef int dx_abs, dy_abs
    cdef int tx, ty

    # Check if source or target is blocked, or source overlaps target
    if c_get_los_mask(x1, y1, pillar_alive) != C_LOS_NONE:
        return False
    if c_get_los_mask(x2, y2, pillar_alive) != C_LOS_NONE:
        return False
    if c_do_footprints_overlap(x1, y1, size, x2, y2, 1):
        return False

    # Melee range — special adjacency check
    if attack_range == 1:
        return ((0 <= dx < size and (dy == size or dy == -1)) or
                (0 <= dy < size and (dx == -1 or dx == size)))

    # NPC: find closest tile on NPC footprint to target, trace NPC tile -> player
    if is_npc:
        tx = x1 if x2 < x1 else (x1 + size - 1 if x2 > x1 + size - 1 else x2)
        ty = y1 if y2 < y1 else (y1 + size - 1 if y2 > y1 + size - 1 else y2)
        return c_has_line_of_sight(tx, ty, x2, y2, 1, attack_range, False, pillar_alive)

    # Distance check
    dx_abs = dx if dx >= 0 else -dx
    dy_abs = dy if dy >= 0 else -dy
    if dx_abs > attack_range or dy_abs > attack_range:
        return False

    # Fixed-point Bresenham ray trace
    return c_trace_ray_fixed_point(x1, y1, x2, y2, dx, dy, dx_abs, dy_abs, pillar_alive)


# ========================================================================
# PUBLIC PYTHON API — identical method signatures to geometry.py
# Each method extracts Python args once, then delegates to cdef C functions.
# ========================================================================

class SimulatorGeometry:
    """Static utility methods for geometry calculations in the Inferno simulator."""

    @staticmethod
    def grid_to_world(int x, int y):
        """Convert grid coordinates to world coordinates."""
        return (ARENA_BASE_X + x, ARENA_BASE_Y + y)

    @staticmethod
    def world_to_grid(int world_x, int world_y):
        """Convert world coordinates to grid coordinates."""
        return (world_x - ARENA_BASE_X, world_y - ARENA_BASE_Y)

    @staticmethod
    def chebyshev_distance(int x1, int y1, int x2, int y2):
        """Calculate Chebyshev distance (max of x/y distance)."""
        return c_chebyshev_distance(x1, y1, x2, y2)

    @staticmethod
    def is_orthogonally_adjacent(int target_x, int target_y,
                                  int npc_x, int npc_y, int npc_size):
        """Check if a position is orthogonally adjacent to an NPC's hitbox."""
        cdef int npc_max_x = npc_x + npc_size - 1
        cdef int npc_max_y = npc_y + npc_size - 1
        if target_x == npc_x - 1 and npc_y <= target_y <= npc_max_y:
            return True
        if target_x == npc_max_x + 1 and npc_y <= target_y <= npc_max_y:
            return True
        if target_y == npc_y - 1 and npc_x <= target_x <= npc_max_x:
            return True
        if target_y == npc_max_y + 1 and npc_x <= target_x <= npc_max_x:
            return True
        return False

    @staticmethod
    def is_on_pillar(int x, int y, pillar_alive=None):
        """Check if a tile is on a pillar."""
        return c_is_on_pillar(x, y, pillar_alive)

    @staticmethod
    def get_pillar_index_at(int x, int y):
        """Get the pillar index at a given position, or -1 if not on any pillar."""
        cdef int i
        for i in range(3):
            if C_PILLAR_X[i] <= x < C_PILLAR_X[i] + C_PILLAR_W[i] and C_PILLAR_Y[i] <= y < C_PILLAR_Y[i] + C_PILLAR_H[i]:
                return i
        return -1

    @staticmethod
    def would_overlap_pillar(int x, int y, int size, pillar_alive=None):
        """Check if an entity of given size would overlap any pillar."""
        return c_would_overlap_pillar(x, y, size, pillar_alive)

    @staticmethod
    def is_in_bounds(int x, int y):
        """Check if coordinates are within arena bounds."""
        return c_is_in_bounds(x, y)

    @staticmethod
    def is_valid_tile(int x, int y, pillar_alive=None):
        """Check if a tile is valid (in bounds and not on pillar)."""
        return c_is_valid_tile(x, y, pillar_alive)

    @staticmethod
    def is_valid_tile_for_size(int x, int y, int size, pillar_alive=None):
        """Check if all tiles in an entity's footprint are valid."""
        return c_is_valid_tile_for_size(x, y, size, pillar_alive)

    @staticmethod
    def is_under_npc(int player_x, int player_y,
                     int npc_x, int npc_y, int npc_size):
        """Check if player is under (overlapping with) an NPC."""
        return (npc_x <= player_x < npc_x + npc_size and
                npc_y <= player_y < npc_y + npc_size)

    @staticmethod
    def compute_push_out_tile(int player_x, int player_y,
                              int npc_x, int npc_y, int npc_size,
                              pillar_alive=None):
        """Compute where a player gets pushed when standing on an NPC and attacking it."""
        return c_compute_push_out_tile(player_x, player_y, npc_x, npc_y, npc_size, pillar_alive)

    @staticmethod
    def do_footprints_overlap(int x1, int y1, int size1,
                               int x2, int y2, int size2):
        """Check if two rectangular footprints overlap."""
        return c_do_footprints_overlap(x1, y1, size1, x2, y2, size2)

    @staticmethod
    def would_npc_overlap_player_at(int npc_x, int npc_y, int npc_size,
                                     int player_x, int player_y):
        """Check if an NPC at (npc_x, npc_y) would overlap a player."""
        return c_would_npc_overlap_player_at(npc_x, npc_y, npc_size, player_x, player_y)


class InfernoLineOfSight:
    """Line of Sight calculator matching OSRS mechanics."""

    @staticmethod
    def get_los_mask(int x, int y, pillar_alive):
        """Get the LOS blocking mask for a tile."""
        return c_get_los_mask(x, y, pillar_alive)

    @staticmethod
    def collision_math(int x1, int y1, int size1,
                       int x2, int y2, int size2):
        """Check if two rectangles overlap."""
        return c_do_footprints_overlap(x1, y1, size1, x2, y2, size2)

    @staticmethod
    def get_closest_point_on_npc(int from_x, int from_y,
                                  int npc_x, int npc_y, int npc_size):
        """Find the closest point on an NPC to a given position."""
        cdef int closest_x = max(npc_x, min(npc_x + npc_size - 1, from_x))
        cdef int closest_y = max(npc_y, min(npc_y + npc_size - 1, from_y))
        return (closest_x, closest_y)

    @staticmethod
    def get_distance_from_npc(int npc_x, int npc_y, int npc_size,
                              int target_x, int target_y):
        """Get Chebyshev distance from NPC's closest tile to target."""
        cdef int closest_x = max(npc_x, min(npc_x + npc_size - 1, target_x))
        cdef int closest_y = max(npc_y, min(npc_y + npc_size - 1, target_y))
        return c_chebyshev_distance(closest_x, closest_y, target_x, target_y)

    @staticmethod
    def _trace_ray_fixed_point(int x1, int y1, int x2, int y2,
                                int dx, int dy, int dx_abs, int dy_abs,
                                pillar_alive):
        """Fixed-point Bresenham ray tracing (exposed for testing)."""
        return c_trace_ray_fixed_point(x1, y1, x2, y2, dx, dy, dx_abs, dy_abs, pillar_alive)

    @staticmethod
    def has_line_of_sight(int x1, int y1, int x2, int y2,
                          int size, int attack_range, bint is_npc,
                          pillar_alive):
        """Core LOS check."""
        return c_has_line_of_sight(x1, y1, x2, y2, size, attack_range, is_npc, pillar_alive)

    @staticmethod
    def npc_has_los_to_player(int npc_x, int npc_y, int npc_size,
                               int player_x, int player_y, int attack_range,
                               pillar_alive):
        """Check if an NPC has line of sight to the player."""
        return c_has_line_of_sight(
            npc_x, npc_y, player_x, player_y,
            npc_size, attack_range, True, pillar_alive
        )

    @staticmethod
    def player_has_los_to_npc(int player_x, int player_y,
                               int npc_x, int npc_y, int npc_size,
                               int attack_range, pillar_alive):
        """Check if the player has line of sight to an NPC.
        Uses the same NPC tile -> player ray as npc_has_los_to_player."""
        return c_has_line_of_sight(
            npc_x, npc_y, player_x, player_y,
            npc_size, attack_range, True, pillar_alive
        )

    @staticmethod
    def can_entity_attack_player(entity, int player_x, int player_y,
                                  pillar_alive):
        """Check if an entity can attack the player (has LOS and is in range)."""
        cdef int size = entity.entity_type.size_in_tiles
        cdef int attack_range = entity.entity_type.attack_range
        cdef int ex = entity.x
        cdef int ey = entity.y
        # Inline distance check (avoid extra Python method call)
        cdef int closest_x = max(ex, min(ex + size - 1, player_x))
        cdef int closest_y = max(ey, min(ey + size - 1, player_y))
        if c_chebyshev_distance(closest_x, closest_y, player_x, player_y) > attack_range:
            return False
        return c_has_line_of_sight(
            ex, ey, player_x, player_y,
            size, attack_range, True, pillar_alive
        )

    @staticmethod
    def can_player_attack_entity(int player_x, int player_y, int attack_range,
                                  entity, pillar_alive):
        """Check if the player can attack an entity."""
        cdef int size = entity.entity_type.size_in_tiles
        cdef int ex = entity.x
        cdef int ey = entity.y
        cdef int closest_x = max(ex, min(ex + size - 1, player_x))
        cdef int closest_y = max(ey, min(ey + size - 1, player_y))
        if c_chebyshev_distance(closest_x, closest_y, player_x, player_y) > attack_range:
            return False
        # Same NPC tile -> player ray direction as NPC attack checks
        return c_has_line_of_sight(
            ex, ey, player_x, player_y,
            size, attack_range, True, pillar_alive
        )
