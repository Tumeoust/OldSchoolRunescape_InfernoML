# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
"""
OSRS-accurate pathfinding mechanics.
Cython-optimized version of pathfinding.py — identical public API.

Implements NPC pathfinding and NPC collision resolution.
"""

from typing import Callable, List, Optional, Tuple
from collections import deque
import random


# Inlined from geometry — avoids cimport which bakes in absolute module path
# and breaks when the package is imported from different root directories.
cdef bint c_would_npc_overlap_player_at(int nx, int ny, int ns, int px, int py):
    """Check if NPC footprint (nx, ny, size=ns) overlaps player at (px, py, size=1)."""
    return (nx < px + 1 and nx + ns > px and
            ny < py + 1 and ny + ns > py)

# Direction priority for OSRS BFS (W, E, S, N, SW, SE, NW, NE)
OSRS_BFS_DIRECTIONS = [
    (-1, 0),   # West
    (1, 0),    # East
    (0, -1),   # South
    (0, 1),    # North
    (-1, -1),  # South-west
    (1, -1),   # South-east
    (-1, 1),   # North-west
    (1, 1)     # North-east
]

# C arrays for BFS directions (avoid Python list indexing in loops)
cdef int C_BFS_DX[8]
cdef int C_BFS_DY[8]
C_BFS_DX[0] = -1; C_BFS_DY[0] =  0  # West
C_BFS_DX[1] =  1; C_BFS_DY[1] =  0  # East
C_BFS_DX[2] =  0; C_BFS_DY[2] = -1  # South
C_BFS_DX[3] =  0; C_BFS_DY[3] =  1  # North
C_BFS_DX[4] = -1; C_BFS_DY[4] = -1  # South-west
C_BFS_DX[5] =  1; C_BFS_DY[5] = -1  # South-east
C_BFS_DX[6] = -1; C_BFS_DY[6] =  1  # North-west
C_BFS_DX[7] =  1; C_BFS_DY[7] =  1  # North-east

# Type alias for walkability checker
WalkabilityChecker = Callable[[int, int, int], bool]


class OSRSPathfinding:
    """OSRS pathfinding utilities."""

    @staticmethod
    def calculate_next_move(int current_x, int current_y,
                             int target_x, int target_y,
                             int size,
                             checker) -> Tuple[int, int]:
        """
        Calculate the next move for an NPC using OSRS "dumb" pathfinding.
        Uses cimported c_would_npc_overlap_player_at for corner safespotting check.
        """
        cdef int dx, dy
        cdef int intended_x, intended_y

        dx = -1 if target_x < current_x else (1 if target_x > current_x else 0)
        dy = -1 if target_y < current_y else (1 if target_y > current_y else 0)

        if dx == 0 and dy == 0:
            return (0, 0)

        intended_x = current_x + dx
        intended_y = current_y + dy

        # Corner safespotting: if diagonal destination overlaps player, cancel Y
        if dx != 0 and dy != 0:
            if c_would_npc_overlap_player_at(intended_x, intended_y, size, target_x, target_y):
                dy = 0

        # 1. Try diagonal first
        if dx != 0 and dy != 0:
            if OSRSPathfinding.can_move_diagonally(current_x, current_y, dx, dy, size, checker):
                return (dx, dy)

        # 2. Always try X first (OSRS priority: diagonal > X > Y)
        if dx != 0 and checker(current_x + dx, current_y, size):
            return (dx, 0)
        # 3. Try Y only if Chebyshev distance > 1 (RuneLite WorldArea guard)
        cdef int cheb = abs(target_x - current_x)
        cdef int cheb_y = abs(target_y - current_y)
        if cheb_y > cheb:
            cheb = cheb_y
        if dy != 0 and cheb > 1:
            if checker(current_x, current_y + dy, size):
                return (0, dy)

        return (0, 0)

    @staticmethod
    def can_move_diagonally(int x, int y, int dx, int dy,
                            int size, checker) -> bint:
        """Check if a diagonal move is valid (destination + both intermediates)."""
        if not checker(x + dx, y + dy, size):
            return False
        if not checker(x + dx, y, size):
            return False
        if not checker(x, y + dy, size):
            return False
        return True

    @staticmethod
    def simulate_npc_movement(int current_x, int current_y,
                               int target_x, int target_y,
                               int size, int max_tiles,
                               checker) -> Tuple[int, int]:
        """Simulate NPC movement over multiple ticks."""
        cdef int x = current_x
        cdef int y = current_y
        cdef int tiles_moved = 0
        cdef int dx, dy

        while tiles_moved < max_tiles:
            move = OSRSPathfinding.calculate_next_move(x, y, target_x, target_y, size, checker)
            dx = move[0]
            dy = move[1]

            if dx == 0 and dy == 0:
                break

            x += dx
            y += dy
            tiles_moved += 1

            if target_x >= x and target_x < x + size and target_y >= y and target_y < y + size:
                break

        return (x, y)

    @staticmethod
    def find_player_path(int start_x, int start_y,
                          int target_x, int target_y,
                          checker,
                          int max_dist=50) -> list:
        """Find a path from start to target using BFS."""
        return OSRSPathfinding.find_player_path_to_endpoints(
            start_x, start_y, [(target_x, target_y)], checker, max_dist
        )

    @staticmethod
    def find_player_path_to_endpoints(int start_x, int start_y,
                                       list endpoints,
                                       checker,
                                       int max_dist=50) -> list:
        """Find shortest path to ANY endpoint using BFS."""
        if not endpoints:
            return []

        cdef int x, y, steps
        cdef int nx, ny
        cdef int dx, dy
        cdef int i

        endpoint_set = set(endpoints)

        if (start_x, start_y) in endpoint_set:
            return []

        queue = deque()
        visited = {}

        start_node = (start_x, start_y, None, 0)
        queue.append(start_node)
        visited[(start_x, start_y)] = start_node

        while queue:
            current = queue.popleft()
            x = current[0]
            y = current[1]
            steps = current[3]

            if (x, y) in endpoint_set:
                return OSRSPathfinding._reconstruct_path(current, visited)

            if steps >= max_dist:
                continue

            for i in range(8):
                dx = C_BFS_DX[i]
                dy = C_BFS_DY[i]
                nx = x + dx
                ny = y + dy
                key = (nx, ny)

                if key in visited:
                    continue

                if dx != 0 and dy != 0:
                    if not OSRSPathfinding.can_move_diagonally(x, y, dx, dy, 1, checker):
                        continue
                else:
                    if not checker(nx, ny, 1):
                        continue

                new_node = (nx, ny, (x, y), steps + 1)
                visited[key] = new_node
                queue.append(new_node)

        return []

    @staticmethod
    def _reconstruct_path(tuple end_node, dict visited) -> list:
        """Reconstruct path from BFS result."""
        path = []
        x = end_node[0]
        y = end_node[1]
        parent = end_node[2]

        while parent is not None:
            path.append((x, y))
            node = visited.get(parent)
            if node is None:
                break
            x = node[0]
            y = node[1]
            parent = node[2]

        path.reverse()
        return path

    @staticmethod
    def simulate_player_movement(int current_x, int current_y,
                                  int target_x, int target_y,
                                  int max_tiles,
                                  checker) -> Tuple[int, int]:
        """Simulate player movement using BFS pathfinding."""
        if current_x == target_x and current_y == target_y:
            return (current_x, current_y)

        path = OSRSPathfinding.find_player_path(
            current_x, current_y, target_x, target_y, checker
        )

        if not path:
            return (current_x, current_y)

        cdef int x = current_x
        cdef int y = current_y
        cdef int tiles_moved = 0

        for step in path:
            if tiles_moved >= max_tiles:
                break
            x = step[0]
            y = step[1]
            tiles_moved += 1

        return (x, y)


class NpcCollisionResolver:
    """Handles NPC collision resolution when player steps under an NPC."""

    CARDINAL_DIRECTIONS = [
        (0, 1),   # North
        (0, -1),  # South
        (1, 0),   # East
        (-1, 0)   # West
    ]

    @staticmethod
    def is_player_under_npc(int npc_x, int npc_y, int npc_size,
                            int player_x, int player_y) -> bint:
        """Check if player is under an NPC."""
        return (npc_x <= player_x < npc_x + npc_size and
                npc_y <= player_y < npc_y + npc_size)

    @staticmethod
    def resolve_collision(int npc_x, int npc_y, int npc_size,
                          checker) -> Tuple[int, int]:
        """Resolve collision with random cardinal direction."""
        direction = random.choice(NpcCollisionResolver.CARDINAL_DIRECTIONS)
        cdef int dx = direction[0]
        cdef int dy = direction[1]

        if checker(npc_x + dx, npc_y + dy, npc_size):
            return (dx, dy)
        return (0, 0)

    @staticmethod
    def resolve_collision_try_all(int npc_x, int npc_y, int npc_size,
                                  checker) -> Tuple[int, int]:
        """Try all cardinal directions in random order."""
        shuffled = list(NpcCollisionResolver.CARDINAL_DIRECTIONS)
        random.shuffle(shuffled)

        cdef int dx, dy
        for direction in shuffled:
            dx = direction[0]
            dy = direction[1]
            if checker(npc_x + dx, npc_y + dy, npc_size):
                return (dx, dy)
        return (0, 0)
